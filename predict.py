"""
predict.py — MindCare Activity Recommendation Inference
Menghasilkan JSON output rekomendasi aktivitas berdasarkan profil stres user.
Pastikan sudah menjalankan 'python train.py' terlebih dahulu.
"""

import joblib
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Custom objects yang dibutuhkan saat load model.keras
# (harus didefinisikan ulang agar model bisa di-deserialize)
class ResidualBlock(keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.2, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.units        = units
        self.dropout_rate = dropout_rate

        self.dense1   = keras.layers.Dense(units)
        self.bn1      = keras.layers.BatchNormalization()
        self.act1     = keras.layers.Activation('relu')
        self.dropout  = keras.layers.Dropout(dropout_rate)
        self.dense2   = keras.layers.Dense(units)
        self.bn2      = keras.layers.BatchNormalization()
        self.proj     = None
        self.act_out  = keras.layers.Activation('relu')

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.proj = keras.layers.Dense(self.units)
        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        shortcut = self.proj(inputs) if self.proj is not None else inputs
        return self.act_out(x + shortcut)

    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({'units': self.units, 'dropout_rate': self.dropout_rate})
        return config


class WeightedCategoricalCrossentropy(keras.losses.Loss):
    def __init__(self, class_weights=None, **kwargs):
        super(WeightedCategoricalCrossentropy, self).__init__(**kwargs)
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        y_true_int    = tf.cast(y_true, tf.int32)
        num_classes   = tf.shape(y_pred)[-1]
        y_true_onehot = tf.one_hot(tf.squeeze(y_true_int, axis=-1), depth=num_classes)
        y_pred        = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce_per_item   = -tf.reduce_sum(y_true_onehot * tf.math.log(y_pred), axis=-1)
        if self.class_weights is not None:
            weights_tensor = tf.constant(
                [self.class_weights.get(i, 1.0) for i in range(len(self.class_weights))],
                dtype=tf.float32
            )
            sample_weights = tf.reduce_sum(y_true_onehot * weights_tensor, axis=-1)
            ce_per_item    = ce_per_item * sample_weights
        return tf.reduce_mean(ce_per_item)

    def get_config(self):
        config = super(WeightedCategoricalCrossentropy, self).get_config()
        config.update({'class_weights': self.class_weights})
        return config


# ──────────────────────────────────────────────
# INFERENCE CLASS
# ──────────────────────────────────────────────
class DeepLearningInference:
    def __init__(self,
                 model_path='model.keras',
                 scaler_path='scaler.pkl',
                 le_path='label_encoder.pkl'):
        try:
            # Load model dengan custom objects
            self.model = keras.models.load_model(
                model_path,
                custom_objects={
                    'ResidualBlock': ResidualBlock,
                    'WeightedCategoricalCrossentropy': WeightedCategoricalCrossentropy
                },
                compile=False
            )
            self.scaler = joblib.load(scaler_path)
            self.le     = joblib.load(le_path)

            # Load dataset buku (opsional — tidak error jika tidak ada)
            books_path = 'dataset/mindcare_books_dataset.csv'
            if os.path.exists(books_path):
                self.books_df = pd.read_csv(books_path)
            else:
                self.books_df = None

        except FileNotFoundError as e:
            raise Exception(
                f"File tidak ditemukan: {e}\n"
                "Pastikan Anda sudah menjalankan 'python train.py' terlebih dahulu."
            )

    # ── PREPROCESSING INPUT ───────────────────────────────────
    def transform_user_input(self, user_json: dict) -> np.ndarray:
        """Memetakan JSON mentah menjadi array numerik terstandarisasi."""
        pekerjaan_map = {'mahasiswa': 15, 'pelajar': 15, 'karyawan': 12, 'wirausaha': 3}
        penyebab_map  = {'akademik': 2, 'pekerjaan': 4, 'hubungan': 6, 'finansial': 7}

        pekerjaan_enc = pekerjaan_map.get(
            str(user_json.get('Pekerjaan', '')).lower(), 15
        )
        penyebab_enc = penyebab_map.get(
            str(user_json.get('Penyebab stres', '')).lower(), 2
        )

        # Durasi stres: pastikan numerik
        durasi = user_json.get('Durasi stres', 2)
        try:
            durasi = float(durasi)
        except (ValueError, TypeError):
            durasi = 2.0

        # Aktivitas fisik
        af_raw       = str(user_json.get('Aktivitas fisik', '')).lower()
        aktivitas_map = {'sering': 120, 'jarang': 20}
        aktivitas_fisik = aktivitas_map.get(af_raw, 30)

        # Preferensi
        pref         = user_json.get('Preferensi', {})
        pref_enc     = lambda key: 4 if str(pref.get(key, '')).lower() == 'ya' else 1
        pref_olahraga = pref_enc('olahraga')
        pref_membaca  = pref_enc('membaca')
        pref_jurnal   = pref_enc('journaling')

        features = [
            float(user_json.get('Umur', 20)),
            float(pekerjaan_enc),
            float(user_json.get('Tingkat stres', 3)),
            float(penyebab_enc),
            float(durasi),
            float(user_json.get('Kualitas tidur', 3)),
            float(user_json.get('Waktu luang per hari', 60)),
            float(aktivitas_fisik),
            float(pref_olahraga),
            float(pref_membaca),
            float(pref_jurnal),
        ]

        return self.scaler.transform([features])

    # ── REKOMENDASI BUKU ─────────────────────────────────────
    def get_book_recommendations(self, tingkat_stres, penyebab_input: str,
                                  n_books: int = 3) -> list:
        """
        Menyaring 3 buku acak dari dataset berdasarkan level stres
        dan kategori penyebab stres. Fallback ke kategori 'Umum' jika
        tidak ada hasil cukup.
        """
        if self.books_df is None:
            return []

        # Mapping tingkat stres → label level
        try:
            ts = int(tingkat_stres)
        except (ValueError, TypeError):
            ts = 3
        level = 'Ringan' if ts <= 2 else ('Sedang' if ts == 3 else 'Berat')

        # Mapping penyebab → kategori buku
        penyebab_lower = str(penyebab_input).lower()
        cat_map = {
            'akademik': 'Akademik',
            'pekerjaan': 'Pekerjaan',
            'hubungan': 'Sosial',
            'finansial': 'Keuangan',
        }
        cat = next((v for k, v in cat_map.items() if k in penyebab_lower), 'Umum')

        # Filter utama: level + kategori spesifik
        filtered = self.books_df[
            (self.books_df['stress_level_target'] == level) &
            (self.books_df['stress_category'].str.contains(cat, case=False, na=False))
        ]

        # Fallback: level + Umum jika hasil < n_books
        if len(filtered) < n_books:
            fallback = self.books_df[
                (self.books_df['stress_level_target'] == level) &
                (self.books_df['stress_category'].str.contains('Umum', case=False, na=False))
            ]
            filtered = pd.concat([filtered, fallback]).drop_duplicates(subset=['book_id'])

        # Acak dan ambil n_books
        sample    = filtered.sample(min(n_books, len(filtered)), random_state=None)
        books_out = []
        for _, row in sample.iterrows():
            book = {
                'judul'    : row.get('title', '-'),
                'penulis'  : row.get('authors', '-'),
                'kategori' : row.get('categories', '-'),
            }
            if 'thumbnail' in row and pd.notna(row['thumbnail']):
                book['thumbnail'] = row['thumbnail']
            if 'description' in row and pd.notna(row['description']):
                book['deskripsi'] = str(row['description'])[:200] + '...'
            books_out.append(book)

        return books_out

    # ── MAIN RECOMMEND ────────────────────────────────────────
    def recommend(self, user_json: dict) -> str:
        """
        Menjalankan inferensi Neural Network dan mengembalikan
        output dalam format JSON string.
        """
        user_features = self.transform_user_input(user_json)

        # Prediksi probabilitas Softmax
        predictions = self.model.predict(user_features, verbose=0)[0]
        class_names = self.le.classes_

        # Distribusi probabilitas semua kelas
        prob_dist    = {class_names[i]: float(predictions[i])
                        for i in range(len(class_names))}
        sorted_preds = sorted(prob_dist.items(), key=lambda x: x[1], reverse=True)

        # Detail aktivitas
        details_map = {
            'olahraga'  : ('jogging ringan atau senam ringan', 30),
            'journaling': ('menulis perasaan dan refleksi harian', 20),
            'membaca'   : ('buku self-improvement atau relaksasi', 25),
        }

        # Rekomendasi Utama
        utama_label, utama_prob = sorted_preds[0]
        utama_detail, utama_dur = details_map.get(utama_label, ('Aktivitas relaksasi', 30))

        rekomendasi_utama = {
            'aktivitas'  : utama_label,
            'confidence' : round(utama_prob, 2),
            'durasi'     : utama_dur,
            'detail'     : utama_detail,
        }
        # Sisipkan rekomendasi buku jika aktivitas utama adalah membaca
        if utama_label == 'membaca':
            rekomendasi_utama['rekomendasi_buku'] = self.get_book_recommendations(
                user_json.get('Tingkat stres', 3),
                user_json.get('Penyebab stres', ''),
                n_books=3
            )

        # Rekomendasi Alternatif (confidence > 5%)
        alternatif_list = []
        for label, prob in sorted_preds[1:]:
            if prob <= 0.05:
                continue
            det, dur = details_map.get(label, ('Alternatif kegiatan', 20))
            alt = {
                'aktivitas'  : label,
                'confidence' : round(prob, 2),
                'durasi'     : dur,
                'detail'     : det,
            }
            if label == 'membaca':
                alt['rekomendasi_buku'] = self.get_book_recommendations(
                    user_json.get('Tingkat stres', 3),
                    user_json.get('Penyebab stres', ''),
                    n_books=3
                )
            alternatif_list.append(alt)

        # Insight
        top_label  = utama_label
        top_pct    = round(utama_prob * 100, 1)
        result = {
            'rekomendasi_utama': rekomendasi_utama,
            'alternatif'       : alternatif_list,
            'insight'          : {
                'model_type'               : 'Deep Learning — Functional API + Residual Block',
                'distribusi_probabilitas'  : {k: round(v, 3) for k, v in prob_dist.items()},
                'alasan'                   : (
                    f"Jaringan saraf tiruan (Neural Network) dengan arsitektur Residual Block "
                    f"memiliki tingkat keyakinan {top_pct}% bahwa '{top_label}' "
                    f"adalah aktivitas yang paling sesuai dengan kondisi Anda saat ini."
                ),
            }
        }

        # JSON serializer yang aman untuk tipe NumPy
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray):  return obj.tolist()
                return super().default(obj)

        return json.dumps(result, indent=2, ensure_ascii=False, cls=NpEncoder)


# ──────────────────────────────────────────────
# CONTOH PENGGUNAAN
# ──────────────────────────────────────────────
import os

if __name__ == '__main__':
    try:
        inference = DeepLearningInference(
            model_path='model.keras',       # ← format baru
            scaler_path='scaler.pkl',
            le_path='label_encoder.pkl'
        )

        input_request = {
            'Umur'              : 22,
            'Pekerjaan'         : 'mahasiswa',
            'Tingkat stres'     : 4,
            'Penyebab stres'    : 'akademik',
            'Durasi stres'      : 3,
            'Kualitas tidur'    : 2,
            'Waktu luang per hari': 90,
            'Aktivitas fisik'   : 'jarang',
            'Preferensi'        : {
                'olahraga'  : 'tidak',
                'membaca'   : 'ya',
                'journaling': 'ya'
            },
            'Mood': 2
        }

        print("Menerima input dari pengguna...")
        json_response = inference.recommend(input_request)

        print("\n=== RESPONSE API JSON ===")
        print(json_response)

    except Exception as e:
        print(f"[ERROR] {e}")
