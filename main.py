import os, json, joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Custom Layer & Loss (harus ada agar model.keras bisa di-load) ──
class ResidualBlock(layers.Layer):
    def __init__(self, units, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.dense1  = layers.Dense(units)
        self.bn1     = layers.BatchNormalization()
        self.act1    = layers.Activation('relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.dense2  = layers.Dense(units)
        self.bn2     = layers.BatchNormalization()
        self.proj    = None
        self.act_out = layers.Activation('relu')

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.proj = layers.Dense(self.units)
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        shortcut = self.proj(inputs) if self.proj else inputs
        return self.act_out(x + shortcut)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'units': self.units, 'dropout_rate': self.dropout_rate})
        return cfg

class WeightedCategoricalCrossentropy(keras.losses.Loss):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        y_true_int    = tf.cast(y_true, tf.int32)
        num_classes   = tf.shape(y_pred)[-1]
        y_true_onehot = tf.one_hot(tf.squeeze(y_true_int, -1), depth=num_classes)
        y_pred        = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce            = -tf.reduce_sum(y_true_onehot * tf.math.log(y_pred), axis=-1)
        if self.class_weights:
            w = tf.constant([self.class_weights.get(i, 1.0) for i in range(len(self.class_weights))], dtype=tf.float32)
            ce = ce * tf.reduce_sum(y_true_onehot * w, axis=-1)
        return tf.reduce_mean(ce)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'class_weights': self.class_weights})
        return cfg

# ── Load model & artifacts ──
MODEL   = keras.models.load_model(
    "model.keras",
    custom_objects={
        'ResidualBlock': ResidualBlock,
        'WeightedCategoricalCrossentropy': WeightedCategoricalCrossentropy
    },
    compile=False
)
SCALER  = joblib.load("scaler.pkl")
LE      = joblib.load("label_encoder.pkl")

# ── Input schema ──
class UserInput(BaseModel):
    Umur: int
    Pekerjaan: str
    Tingkat_stres: int        # 1–5
    Penyebab_stres: str
    Durasi_stres: float
    Kualitas_tidur: int       # 1–5
    Waktu_luang: int          # menit/hari
    Aktivitas_fisik: str      # "sering" / "jarang"
    pref_olahraga: str        # "ya" / "tidak"
    pref_membaca: str
    pref_journaling: str

@app.post("/predict")
def predict(data: UserInput):
    pekerjaan_map = {'mahasiswa': 15, 'pelajar': 15, 'karyawan': 12, 'wirausaha': 3}
    penyebab_map  = {'akademik': 2, 'pekerjaan': 4, 'hubungan': 6, 'finansial': 7}
    aktivitas_map = {'sering': 120, 'jarang': 20}

    pref = lambda v: 4 if v.lower() == 'ya' else 1

    features = [
        float(data.Umur),
        float(pekerjaan_map.get(data.Pekerjaan.lower(), 15)),
        float(data.Tingkat_stres),
        float(penyebab_map.get(data.Penyebab_stres.lower(), 2)),
        float(data.Durasi_stres),
        float(data.Kualitas_tidur),
        float(data.Waktu_luang),
        float(aktivitas_map.get(data.Aktivitas_fisik.lower(), 30)),
        float(pref(data.pref_olahraga)),
        float(pref(data.pref_membaca)),
        float(pref(data.pref_journaling)),
    ]

    X = SCALER.transform([features])
    preds       = MODEL.predict(X, verbose=0)[0]
    class_names = LE.classes_
    idx_top     = int(np.argmax(preds))

    return {
        "rekomendasi": class_names[idx_top],
        "confidence":  round(float(preds[idx_top]), 3),
        "distribusi":  {class_names[i]: round(float(preds[i]), 3) for i in range(len(class_names))}
    }

@app.get("/")
def root():
    return {"status": "MindCare API ready"}