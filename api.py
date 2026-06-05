"""
api.py — MindCare REST API
Membungkus DeepLearningInference menjadi endpoint HTTP yang bisa dipanggil React.

Jalankan dengan:
    uvicorn api:app --reload --port 8000

Endpoint:
    GET  /          → health check
    POST /predict   → kirim profil user, terima rekomendasi JSON
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from predict import DeepLearningInference   # import class dari predict.py


# ── Inisialisasi FastAPI ───────────────────────────────────────────────────────
app = FastAPI(
    title="MindCare Recommendation API",
    description="REST API untuk sistem rekomendasi aktivitas berbasis Deep Learning (FFNN + Residual Block)",
    version="1.0.0",
)

# ── CORS — izinkan semua origin untuk production (Railway deployment) ─────────
# allow_origins=["*"] agar dapat diakses oleh full-stack frontend yang di-deploy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Ganti dengan domain frontend spesifik setelah deploy jika perlu
    allow_credentials=False,   # Harus False jika allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model sekali saat server start ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

inference_engine = DeepLearningInference(
    model_path=os.path.join(BASE_DIR, "model.keras"),
    scaler_path=os.path.join(BASE_DIR, "scaler.pkl"),
    le_path=os.path.join(BASE_DIR, "label_encoder.pkl"),
)


# ── Schema Request Body (Pydantic) ────────────────────────────────────────────
class PreferensiSchema(BaseModel):
    olahraga  : Optional[str] = Field(default="tidak", example="ya")
    membaca   : Optional[str] = Field(default="tidak", example="ya")
    journaling: Optional[str] = Field(default="tidak", example="ya")


class UserProfile(BaseModel):
    Umur                : int   = Field(..., ge=10, le=100, example=22)
    Pekerjaan           : str   = Field(..., example="mahasiswa")
    Tingkat_stres       : int   = Field(..., ge=1, le=5, example=4, alias="Tingkat stres")
    Penyebab_stres      : str   = Field(..., example="akademik", alias="Penyebab stres")
    Durasi_stres        : float = Field(default=2.0, example=3.0, alias="Durasi stres")
    Kualitas_tidur      : int   = Field(default=3, ge=1, le=5, example=2, alias="Kualitas tidur")
    Waktu_luang_per_hari: int   = Field(default=60, example=90, alias="Waktu luang per hari")
    Aktivitas_fisik     : str   = Field(default="jarang", example="jarang", alias="Aktivitas fisik")
    Preferensi          : Optional[PreferensiSchema] = PreferensiSchema()
    Mood                : Optional[int] = Field(default=3, ge=1, le=5, example=2)

    class Config:
        populate_by_name = True   # boleh pakai alias atau field name asli


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check — pastikan API berjalan."""
    return {"status": "ok", "message": "MindCare API is running 🚀"}


@app.post("/predict", tags=["Recommendation"])
def predict(profile: UserProfile):
    """
    Menerima profil psikologis user dan mengembalikan tingkat stres beserta rekomendasi aktivitas
    dan rekomendasi buku (jika relevan).

    **Body contoh:**
    ```json
    {
      "Umur": 22,
      "Pekerjaan": "mahasiswa",
      "Tingkat stres": 4,
      "Penyebab stres": "akademik",
      "Durasi stres": 3,
      "Kualitas tidur": 2,
      "Waktu luang per hari": 90,
      "Aktivitas fisik": "jarang",
      "Preferensi": {"olahraga": "tidak", "membaca": "ya", "journaling": "ya"},
      "Mood": 2
    }
    ```

    **Response contoh:**
    ```json
    {
      "stress_assessment": {
        "stress_percentage": 72.4,
        "stress_level": "Tinggi",
        "keterangan": "Pengguna menunjukkan indikasi tingkat stres yang cukup tinggi berdasarkan pola jawaban pada 11 fitur yang diberikan."
      },
      "rekomendasi_utama": {
        "aktivitas": "membaca",
        "confidence": 0.61,
        "durasi": 25,
        "detail": "buku self-improvement atau relaksasi",
        "rekomendasi_buku": [...]
      },
      "alternatif": [...],
      "insight": {...}
    }
    ```
    """
    try:
        # Konversi Pydantic model → dict dengan key seperti yang diharapkan predict.py
        user_dict = {
            "Umur"                : profile.Umur,
            "Pekerjaan"           : profile.Pekerjaan,
            "Tingkat stres"       : profile.Tingkat_stres,
            "Penyebab stres"      : profile.Penyebab_stres,
            "Durasi stres"        : profile.Durasi_stres,
            "Kualitas tidur"      : profile.Kualitas_tidur,
            "Waktu luang per hari": profile.Waktu_luang_per_hari,
            "Aktivitas fisik"     : profile.Aktivitas_fisik,
            "Preferensi"          : profile.Preferensi.model_dump() if profile.Preferensi else {},
            "Mood"                : profile.Mood,
        }

        # Jalankan inferensi — hasilnya sudah berupa JSON string
        import json
        result_str = inference_engine.recommend(user_dict)
        result_dict = json.loads(result_str)   # parse balik ke dict agar dikirim sebagai JSON native

        return result_dict

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
