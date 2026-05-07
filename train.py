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
    def __init__(self, units, dropout_rate=0.2, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.units        = units
        self.dropout_rate = dropout_rate

        self.dense1   = layers.Dense(units)
        self.bn1      = layers.BatchNormalization()
        self.act1     = layers.Activation('relu')
        self.dropout  = layers.Dropout(dropout_rate)
        self.dense2   = layers.Dense(units)
        self.bn2      = layers.BatchNormalization()

        # Projection jika dimensi input != units
        self.proj     = None
        self.act_out  = layers.Activation('relu')

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.proj = layers.Dense(self.units)
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
        config.update({'units': self.units, 'dropout_rate': self.dropout_rate})
        return config


# ──────────────────────────────────────────────
# CUSTOM LOSS: Weighted Categorical Crossentropy
# ──────────────────────────────────────────────
class WeightedCategoricalCrossentropy(keras.losses.Loss):
    """
    Custom Loss: Categorical Crossentropy dengan bobot per kelas.
    Berguna saat distribusi label tidak seimbang — kelas minoritas
    mendapat penalti lebih besar agar tidak diabaikan model.
    class_weights: dict {index_kelas: bobot}, misal {0: 1.0, 1: 2.0, 2: 1.5}
    """
    def __init__(self, class_weights=None, **kwargs):
        super(WeightedCategoricalCrossentropy, self).__init__(**kwargs)
        self.class_weights = class_weights  # dict atau None

    def call(self, y_true, y_pred):
        # y_true dalam format sparse (integer), konversi ke one-hot
        y_true_int    = tf.cast(y_true, tf.int32)
        num_classes   = tf.shape(y_pred)[-1]
        y_true_onehot = tf.one_hot(tf.squeeze(y_true_int, axis=-1), depth=num_classes)

        # Cross-entropy per sampel
        y_pred      = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce_per_item = -tf.reduce_sum(y_true_onehot * tf.math.log(y_pred), axis=-1)

        # Terapkan bobot kelas
        if self.class_weights is not None:
            weights_tensor = tf.constant(
                [self.class_weights.get(i, 1.0) for i in range(len(self.class_weights))],
                dtype=tf.float32
            )
            sample_weights = tf.reduce_sum(
                y_true_onehot * weights_tensor, axis=-1
            )
            ce_per_item = ce_per_item * sample_weights

        return tf.reduce_mean(ce_per_item)

    def get_config(self):
        config = super(WeightedCategoricalCrossentropy, self).get_config()
        config.update({'class_weights': self.class_weights})
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
        train_acc = logs.get('accuracy', 0)
        val_acc   = logs.get('val_accuracy', 0)
        gap       = train_acc - val_acc

        if gap > self.gap_threshold:
            status = f"⚠️  OVERFIT (gap={gap:.4f})"
            self.overfit_epochs.append(epoch + 1)
        else:
            status = f"✅ OK       (gap={gap:.4f})"

        print(f"   [OverfittingLogger] Epoch {epoch+1:03d} | "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | {status}")

    def on_train_end(self, logs=None):
        if self.overfit_epochs:
            print(f"\n[OverfittingLogger] ⚠️  Overfitting terdeteksi di epoch: {self.overfit_epochs}")
        else:
            print("\n[OverfittingLogger] ✅ Tidak ada indikasi overfitting signifikan.")


# ──────────────────────────────────────────────
# BUILD MODEL: Functional API
# ──────────────────────────────────────────────
def build_model(input_dim: int, num_classes: int, class_weights: dict) -> Model:
    """
    Membangun model Neural Network menggunakan Functional API.
    Arsitektur: Input → Dense → ResidualBlock × 2 → Dense(softmax)
    """
    inputs = keras.Input(shape=(input_dim,), name='stress_features')

    # Proyeksi awal
    x = layers.Dense(64, activation='relu', name='input_projection')(inputs)
    x = layers.BatchNormalization(name='input_bn')(x)

    # Dua Residual Block (Custom Layer)
    x = ResidualBlock(64, dropout_rate=0.2, name='residual_block_1')(x)
    x = ResidualBlock(32, dropout_rate=0.2, name='residual_block_2')(x)

    # Output
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=inputs, outputs=outputs, name='MindCare_DNN')

    custom_loss = WeightedCategoricalCrossentropy(
        class_weights=class_weights,
        name='weighted_ce'
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=custom_loss,
        metrics=['accuracy']
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
    feature_columns = [
        'umur', 'pekerjaan_enc', 'stress_level_1_5', 'penyebab_stres_enc',
        'durasi_stres_enc', 'kualitas_tidur_1_5', 'waktu_luang_mnt',
        'aktivitas_fisik_mnt', 'preferensi_olahraga', 'preferensi_baca',
        'preferensi_jurnal'
    ]
    target_col = 'label_aktivitas'

    df_train = df_train.dropna(subset=feature_columns + [target_col]).copy()
    df_test  = df_test.dropna(subset=feature_columns + [target_col]).copy()
    df_val   = df_val.dropna(subset=feature_columns + [target_col]).copy()

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

    overfit_cb = OverfittingLogger(gap_threshold=0.10)
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10,
        restore_best_weights=True, verbose=1
    )
    reduce_lr  = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=5, min_lr=1e-5, verbose=1
    )

    print("\n[TRAIN] Melatih pada data_train | Validasi pada data_test...")
    model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test_scaled, y_test),
        callbacks=[overfit_cb, early_stop, reduce_lr],
        verbose=0   # supaya tidak tertimpa log OverfittingLogger
    )

    # ── 4. EVALUASI ──────────────────────────────────────────
    y_pred_prob = model.predict(X_test_scaled, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)
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

    # Evaluasi pada validation set
    y_val_pred = np.argmax(model.predict(X_val_scaled, verbose=0), axis=1)
    acc_val    = accuracy_score(y_val.flatten(), y_val_pred)
    print(f"Akurasi pada Validation Set: {acc_val * 100:.2f}%")
    print("=" * 50)

    # ── 5. FINAL MODEL (train on all data) ───────────────────
    print("\n[FINAL] Menggabungkan semua data untuk Final Model...")
    df_all  = pd.concat([df_train, df_test, df_val], ignore_index=True)
    X_all   = df_all[feature_columns].values
    y_all   = label_encoder.transform(df_all[target_col].values).reshape(-1, 1)

    X_all_scaled = scaler.fit_transform(X_all)

    # Hitung ulang bobot kelas dari full dataset
    unique_all, counts_all = np.unique(y_all.flatten(), return_counts=True)
    total_all              = len(y_all)
    class_weights_all      = {int(c): round(total_all / (num_classes * n), 4)
                               for c, n in zip(unique_all, counts_all)}

    final_model = build_model(len(feature_columns), num_classes, class_weights_all)
    print(f"[FINAL] Melatih Final Model dengan {len(X_all)} total data...")
    final_model.fit(
        X_all_scaled, y_all,
        epochs=100,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='loss', patience=10,
                restore_best_weights=True, verbose=1
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

    print(f"\n[SUKSES] Model   → '{model_path}'")
    print(f"[SUKSES] Scaler  → '{scaler_path}'")
    print(f"[SUKSES] Encoder → '{le_path}'")
    print("Sekarang jalankan 'python predict.py' untuk output JSON.")


if __name__ == '__main__':
    train_model()
