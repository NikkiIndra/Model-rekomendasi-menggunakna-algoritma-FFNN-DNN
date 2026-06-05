"""
train.py — MindCare Activity Recommendation Model
Fitur utama:
  - TF Functional API (bukan Sequential)
  - Custom Layer   : ResidualBlock
  - Custom Loss    : WeightedCategoricalCrossentropy
  - Custom Callback: EarlyStopping + OverfittingLogger
  - Save format    : model.keras (bukan .h5 legacy)
"""

import os
import random
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ──────────────────────────────────────────────
# SEED untuk reprodusibilitas
# ──────────────────────────────────────────────
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ──────────────────────────────────────────────
# CUSTOM LAYER: Residual Block
# ──────────────────────────────────────────────
class ResidualBlock(layers.Layer):
    """
    Custom Layer: Residual (Skip) Connection Block.
    Menambahkan koneksi pintas dari input ke output untuk
    mempermudah propagasi gradien pada jaringan yang lebih dalam.
    Arsitektur: Dense -> BatchNorm -> ReLU -> Dense -> BatchNorm -> Add(input) -> ReLU
    """
    def __init__(self, units, dropout_rate=0.2, l2_val=0.001, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.units        = units
        self.dropout_rate = dropout_rate
        self.l2_val       = l2_val

        self.dense1   = layers.Dense(units, kernel_regularizer=keras.regularizers.l2(l2_val))
        self.bn1      = layers.BatchNormalization()
        self.act1     = layers.Activation('relu')
        self.dropout  = layers.Dropout(dropout_rate)
        self.dense2   = layers.Dense(units, kernel_regularizer=keras.regularizers.l2(l2_val))
        self.bn2      = layers.BatchNormalization()

        # Projection jika dimensi input != units
        self.proj     = None
        self.act_out  = layers.Activation('relu')

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.proj = layers.Dense(self.units, kernel_regularizer=keras.regularizers.l2(self.l2_val))
        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)

        # Shortcut
        shortcut = self.proj(inputs) if self.proj is not None else inputs
        x = x + shortcut
        return self.act_out(x)

    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'l2_val': self.l2_val
        })
        return config


# ──────────────────────────────────────────────
# CUSTOM LOSS: Weighted Categorical Crossentropy
# ──────────────────────────────────────────────
class WeightedCategoricalCrossentropy(keras.losses.Loss):
    """
    Custom Loss: Categorical Crossentropy dengan bobot per kelas + Label Smoothing.
    Berguna saat distribusi label tidak seimbang — kelas minoritas
    mendapat penalti lebih besar agar tidak diabaikan model.

    class_weights   : dict {index_kelas: bobot}, misal {0: 1.0, 1: 2.0, 2: 1.5}
    label_smoothing : float [0, 1). Nilai 0.1 artinya label 1.0 digeser ke 0.9 dan
                      distribusi sisa 0.1 dibagi rata ke semua kelas. Ini mencegah
                      model terlalu "confident" → mengurangi overfitting.
    """
    def __init__(self, class_weights=None, label_smoothing=0.0, **kwargs):
        super(WeightedCategoricalCrossentropy, self).__init__(**kwargs)
        self.class_weights   = class_weights   # dict atau None
        self.label_smoothing = label_smoothing # baru: 0.0 = off, 0.1 = recommended

    def call(self, y_true, y_pred):
        # y_true dalam format sparse (integer), konversi ke one-hot
        y_true_int    = tf.cast(y_true, tf.int32)
        num_classes   = tf.shape(y_pred)[-1]
        num_classes_f = tf.cast(num_classes, tf.float32)
        y_true_onehot = tf.one_hot(tf.squeeze(y_true_int, axis=-1), depth=num_classes)

        # Label Smoothing: geser distribusi agar tidak terlalu keras
        if self.label_smoothing > 0.0:
            smooth = self.label_smoothing
            y_true_onehot = y_true_onehot * (1.0 - smooth) + (smooth / num_classes_f)

        # Cross-entropy per sampel
        y_pred      = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce_per_item = -tf.reduce_sum(y_true_onehot * tf.math.log(y_pred), axis=-1)

        # Terapkan bobot kelas (dari one-hot asli, bukan yang di-smooth)
        if self.class_weights is not None:
            orig_onehot = tf.one_hot(tf.squeeze(tf.cast(y_true, tf.int32), axis=-1), depth=num_classes)
            weights_tensor = tf.constant(
                [self.class_weights.get(i, 1.0) for i in range(len(self.class_weights))],
                dtype=tf.float32
            )
            sample_weights = tf.reduce_sum(orig_onehot * weights_tensor, axis=-1)
            ce_per_item = ce_per_item * sample_weights

        return tf.reduce_mean(ce_per_item)

    def get_config(self):
        config = super(WeightedCategoricalCrossentropy, self).get_config()
        config.update({
            'class_weights'  : self.class_weights,
            'label_smoothing': self.label_smoothing,
        })
        return config


# ──────────────────────────────────────────────
# CUSTOM CALLBACK: OverfittingLogger
# ──────────────────────────────────────────────
class OverfittingLogger(keras.callbacks.Callback):
    """
    Custom Callback: Memonitor dan melaporkan gap antara
    train accuracy dan val accuracy setiap epoch.
    Memberi peringatan jika gap melebihi threshold (indikasi overfitting).
    """
    def __init__(self, gap_threshold=0.10):
        super(OverfittingLogger, self).__init__()
        self.gap_threshold  = gap_threshold
        self.overfit_epochs = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_acc = logs.get('predictions_accuracy', logs.get('accuracy', 0))
        val_acc   = logs.get('val_predictions_accuracy', logs.get('val_accuracy', 0))
        gap       = train_acc - val_acc

        if gap > self.gap_threshold:
            status = f"WARNING: OVERFIT (gap={gap:.4f})"
            self.overfit_epochs.append(epoch + 1)
        else:
            status = f"OK (gap={gap:.4f})"

        print(f"   [OverfittingLogger] Epoch {epoch+1:03d} | "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | {status}")

    def on_train_end(self, logs=None):
        if self.overfit_epochs:
            print(f"\n[OverfittingLogger] WARNING: Overfitting terdeteksi di epoch: {self.overfit_epochs}")
        else:
            print("\n[OverfittingLogger] OK: Tidak ada indikasi overfitting signifikan.")


# ──────────────────────────────────────────────
# BUILD MODEL: Functional API
# ──────────────────────────────────────────────
def build_model(input_dim: int, num_classes: int, class_weights: dict) -> Model:
    """
    Membangun model Neural Network menggunakan Functional API.
    Arsitektur: Input → Dense → ResidualBlock × 3 → Multi-Outputs:
                1. predictions (activity classification, softmax)
                2. stress_prediction (stress regression, sigmoid)

    Perubahan v2 (anti-overfit):
      - L2 diturunkan: 0.005 -> 0.001  (regularisasi sebelumnya terlalu agresif)
      - Dropout diturunkan: 0.3 -> 0.2  (agar model bisa belajar lebih banyak)
      - Label Smoothing 0.1 ditambahkan ke loss (mencegah model terlalu confident)
      - Learning rate diturunkan: 0.001 -> 0.0005 (kurva val_acc lebih stabil)
    """
    inputs = keras.Input(shape=(input_dim,), name='stress_features')

    # Proyeksi awal - L2 dikurangi agar model tidak under-fit
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001), name='input_projection')(inputs)
    x = layers.BatchNormalization(name='input_bn')(x)

    # Tiga Residual Block - Dropout 0.2 (turun dari 0.3), L2 = 0.001
    x = ResidualBlock(256, dropout_rate=0.2, l2_val=0.001, name='residual_block_1')(x)
    x = ResidualBlock(128, dropout_rate=0.2, l2_val=0.001, name='residual_block_2')(x)
    x = ResidualBlock(64,  dropout_rate=0.2, l2_val=0.001, name='residual_block_3')(x)

    # Output 1: Aktivitas (Classification)
    outputs_activity = layers.Dense(num_classes, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.001), name='predictions')(x)

    # Output 2: Stress Percentage (Regression)
    outputs_stress = layers.Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0.001), name='stress_prediction')(x)

    model = Model(inputs=inputs, outputs=[outputs_activity, outputs_stress], name='MindCare_DNN_v2')

    # Label Smoothing 0.1: mencegah model overly-confident, menurunkan gap train-val
    custom_loss = WeightedCategoricalCrossentropy(
        class_weights=class_weights,
        label_smoothing=0.1,
        name='weighted_ce_smooth'
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # lebih stabil dari 0.001
        loss={
            'predictions': custom_loss,
            'stress_prediction': 'mse'
        },
        loss_weights={
            'predictions': 1.0,
            'stress_prediction': 1.5   # Memberikan bobot lebih pada estimasi stres agar konvergen dengan baik
        },
        metrics={
            'predictions': ['accuracy'],
            'stress_prediction': ['mae']
        }
    )
    return model


# ──────────────────────────────────────────────
# MAIN TRAINING
# ──────────────────────────────────────────────
def train_model():
    # ── 1. PATH DATASET (relatif, portable) ─────────────────
    dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    train_path  = os.path.join(dataset_dir, 'data_train.csv')
    test_path   = os.path.join(dataset_dir, 'data_test.csv')
    val_path    = os.path.join(dataset_dir, 'data_validation.csv')

    print(f"[INFO] Dataset dir : {dataset_dir}")
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    df_val   = pd.read_csv(val_path)
    print(f"[INFO] Train: {len(df_train)} | Test: {len(df_test)} | Val: {len(df_val)}")

    # ── 2. PREPROCESSING ─────────────────────────────────────
    base_feature_columns = [
        'umur', 'pekerjaan_enc', 'stress_level_1_5', 'penyebab_stres_enc',
        'durasi_stres_enc', 'kualitas_tidur_1_5', 'waktu_luang_mnt',
        'aktivitas_fisik_mnt', 'preferensi_olahraga', 'preferensi_baca',
        'preferensi_jurnal'
    ]
    target_col = 'label_aktivitas'

    # Fungsi untuk rekayasa fitur interaksi tingkat lanjut
    def add_engineered_features(df):
        df_out = df.copy()
        # 1. Stress sleep ratio
        df_out['stress_sleep_ratio'] = df_out['stress_level_1_5'] / (df_out['kualitas_tidur_1_5'] + 1e-5)
        # 2. Waktu luang vs aktivitas fisik
        df_out['luang_fisik_ratio'] = df_out['waktu_luang_mnt'] / (df_out['aktivitas_fisik_mnt'] + 1e-5)
        # 3. Total active preference score
        df_out['pref_score'] = df_out['preferensi_olahraga'] + df_out['preferensi_baca'] + df_out['preferensi_jurnal']
        # 4. Interaction of stress level and causes
        df_out['stress_cause_interact'] = df_out['stress_level_1_5'] * df_out['penyebab_stres_enc']
        # 5. Age stress interact
        df_out['age_stress_interact'] = df_out['umur'] * df_out['stress_level_1_5']
        # 6. Sleep quality interact
        df_out['sleep_dur_interact'] = df_out['kualitas_tidur_1_5'] * df_out['durasi_stres_enc']
        # 7. Physical active pref
        df_out['physical_active_pref'] = df_out['aktivitas_fisik_mnt'] * df_out['preferensi_olahraga']
        # 8. Reading pref luang
        df_out['reading_pref_luang'] = df_out['waktu_luang_mnt'] * df_out['preferensi_baca']
        # 9. Jurnal pref stress
        df_out['jurnal_pref_stress'] = df_out['preferensi_jurnal'] * df_out['stress_level_1_5']
        return df_out

    df_train = add_engineered_features(df_train)
    df_test  = add_engineered_features(df_test)
    df_val   = add_engineered_features(df_val)

    feature_columns = base_feature_columns + [
        'stress_sleep_ratio', 'luang_fisik_ratio', 'pref_score',
        'stress_cause_interact', 'age_stress_interact', 'sleep_dur_interact',
        'physical_active_pref', 'reading_pref_luang', 'jurnal_pref_stress'
    ]

    df_train = df_train.dropna(subset=feature_columns + [target_col, 'psikologis_score']).copy()
    df_test  = df_test.dropna(subset=feature_columns + [target_col, 'psikologis_score']).copy()
    df_val   = df_val.dropna(subset=feature_columns + [target_col, 'psikologis_score']).copy()

    X_train = df_train[feature_columns].values
    X_test  = df_test[feature_columns].values
    X_val   = df_val[feature_columns].values

    y_train_raw = df_train[target_col].values
    y_test_raw  = df_test[target_col].values
    y_val_raw   = df_val[target_col].values

    # Encode label
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw).reshape(-1, 1)
    y_test  = label_encoder.transform(y_test_raw).reshape(-1, 1)
    y_val   = label_encoder.transform(y_val_raw).reshape(-1, 1)

    # Stress target (regression) - normalisasi 0-100 ke 0-1
    y_train_stress = df_train['psikologis_score'].values.reshape(-1, 1) / 100.0
    y_test_stress  = df_test['psikologis_score'].values.reshape(-1, 1) / 100.0
    y_val_stress   = df_val['psikologis_score'].values.reshape(-1, 1) / 100.0

    num_classes  = len(label_encoder.classes_)
    print(f"[INFO] Kelas label  : {list(label_encoder.classes_)}")
    print(f"[INFO] Jumlah kelas : {num_classes}")

    # Hitung bobot kelas otomatis (inversely proportional to frequency)
    unique, counts = np.unique(y_train.flatten(), return_counts=True)
    total          = len(y_train)
    class_weights  = {int(cls): round(total / (num_classes * cnt), 4)
                      for cls, cnt in zip(unique, counts)}
    print(f"[INFO] Class weights: {class_weights}")

    # Normalisasi
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    X_val_scaled   = scaler.transform(X_val)

    # ── 3. BUILD & TRAIN (evaluasi) ──────────────────────────
    print("\n[TRAIN] Membangun model (Functional API + Custom Components)...")
    model = build_model(len(feature_columns), num_classes, class_weights)
    model.summary()

    overfit_cb = OverfittingLogger(gap_threshold=0.07)  # lebih ketat: 0.10 -> 0.07
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_predictions_accuracy',  # pantau val_acc dari output predictions
        patience=15,             # beri ruang agar val_acc naik
        restore_best_weights=True, mode='max', verbose=1
    )
    reduce_lr  = keras.callbacks.ReduceLROnPlateau(
        monitor='val_predictions_accuracy', factor=0.5, mode='max',
        patience=7, min_lr=1e-6,  # min_lr lebih kecil untuk fine-tuning halus
        verbose=1
    )

    print("\n[TRAIN] Melatih pada data_train | Validasi pada data_test...")
    history = model.fit(
        X_train_scaled,
        {
            'predictions': y_train,
            'stress_prediction': y_train_stress
        },
        epochs=150,       # lebih banyak epoch, EarlyStopping yang akan stop
        batch_size=64,    # batch lebih besar -> gradien lebih stabil, val_acc lebih smooth
        validation_data=(
            X_test_scaled,
            {
                'predictions': y_test,
                'stress_prediction': y_test_stress
            }
        ),
        callbacks=[overfit_cb, early_stop, reduce_lr],
        verbose=0   # supaya tidak tertimpa log OverfittingLogger
    )

    # ── 4. EVALUASI ──────────────────────────────────────────
    y_pred_prob_act, y_pred_stress = model.predict(X_test_scaled, verbose=0)
    y_pred      = np.argmax(y_pred_prob_act, axis=1)
    y_true      = y_test.flatten()

    acc = accuracy_score(y_true, y_pred)
    print("\n" + "=" * 50)
    print("HASIL EVALUASI MODEL PADA DATA TEST:")
    print("=" * 50)
    print(f"Akurasi : {acc * 100:.2f}%")
    print("\nLaporan Klasifikasi:")
    print(classification_report(
        y_true, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae_stress = mean_absolute_error(y_test_stress * 100.0, y_pred_stress * 100.0)
    rmse_stress = np.sqrt(mean_squared_error(y_test_stress * 100.0, y_pred_stress * 100.0))
    print(f"Stress MAE  : {mae_stress:.2f}%")
    print(f"Stress RMSE : {rmse_stress:.2f}%")

    # Evaluasi pada validation set
    y_val_pred_act, y_val_pred_stress = model.predict(X_val_scaled, verbose=0)
    y_val_pred = np.argmax(y_val_pred_act, axis=1)
    acc_val    = accuracy_score(y_val.flatten(), y_val_pred)
    mae_val_stress = mean_absolute_error(y_val_stress * 100.0, y_val_pred_stress * 100.0)
    print(f"Akurasi pada Validation Set: {acc_val * 100:.2f}%")
    print(f"Stress MAE pada Validation Set: {mae_val_stress:.2f}%")
    print("=" * 50)

    # ── 4.1 VISUALISASI HASIL TRAINING (CONFUSION MATRIX & OVERFITTING ANALYSIS) ──
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        print("\n[VISUALISASI] Membuat visualisasi hasil training...")
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title('Confusion Matrix - MindCare Recommendation Model')
        plt.ylabel('Actual Activity')
        plt.xlabel('Predicted Activity')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        plt.close()
        print("   [SUKSES] Confusion Matrix disimpan ke 'confusion_matrix.png' [OK]")

        # 2. Overfitting Analysis / Training History Curves
        plt.figure(figsize=(12, 5))

        # Akurasi
        plt.subplot(1, 2, 1)
        train_acc_key = 'predictions_accuracy' if 'predictions_accuracy' in history.history else 'accuracy'
        val_acc_key = 'val_predictions_accuracy' if 'val_predictions_accuracy' in history.history else 'val_accuracy'
        plt.plot(history.history[train_acc_key], label='Train Acc', color='#1f77b4', linewidth=2)
        plt.plot(history.history[val_acc_key], label='Val Acc', color='#ff7f0e', linewidth=2)
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.5)

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
        plt.plot(history.history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig('overfitting_analysis.png', dpi=300)
        plt.close()
        print("   [SUKSES] Grafik Overfitting Analysis disimpan ke 'overfitting_analysis.png' [OK]")
        
    except Exception as ev:
        print(f"   [Peringatan] Gagal membuat visualisasi: {ev}")

    # ── 5. FINAL MODEL (train on all data) ───────────────────
    print("\n[FINAL] Menggabungkan semua data untuk Final Model...")
    df_all  = pd.concat([df_train, df_test, df_val], ignore_index=True)
    X_all   = df_all[feature_columns].values
    y_all   = label_encoder.transform(df_all[target_col].values).reshape(-1, 1)
    y_all_stress = df_all['psikologis_score'].values.reshape(-1, 1) / 100.0

    X_all_scaled = scaler.fit_transform(X_all)

    # Hitung ulang bobot kelas dari full dataset
    unique_all, counts_all = np.unique(y_all.flatten(), return_counts=True)
    total_all              = len(y_all)
    class_weights_all      = {int(c): round(total_all / (num_classes * n), 4)
                               for c, n in zip(unique_all, counts_all)}

    final_model = build_model(len(feature_columns), num_classes, class_weights_all)
    print(f"[FINAL] Melatih Final Model dengan {len(X_all)} total data...")
    final_model.fit(
        X_all_scaled,
        {
            'predictions': y_all,
            'stress_prediction': y_all_stress
        },
        epochs=100,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='loss', patience=10,
                restore_best_weights=True, mode='auto', verbose=1
            )
        ],
        verbose=1
    )

    # ── 6. SAVE (.keras — format TF modern) ─────────────────
    model_path  = 'model.keras'         # ← format baru, bukan .h5
    scaler_path = 'scaler.pkl'
    le_path     = 'label_encoder.pkl'

    final_model.save(model_path)        # SavedModel / .keras format
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, le_path)

    print(f"\n[SUKSES] Model   -> '{model_path}'")
    print(f"[SUKSES] Scaler  -> '{scaler_path}'")
    print(f"[SUKSES] Encoder -> '{le_path}'")
    print("Sekarang jalankan 'python predict.py' untuk output JSON.")


if __name__ == '__main__':
    train_model()