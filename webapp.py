from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

from src.diabetes_app.config import (
    BEST_MODEL_KEY,
    DEFAULT_FEATURE_ORDER,
    METRICS_PATH,
    MODEL_FILES,
    MODEL_PATH,
)
from src.diabetes_app.features import age_to_cdc_bucket
from src.diabetes_app.model import load_model, predict_risk_probability

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / MODEL_PATH
METRICS_FILE = BASE_DIR / METRICS_PATH
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__, template_folder="templates")


def _to_int(value, default: int, lo: int, hi: int) -> int:
    try:
        num = int(float(value))
    except (TypeError, ValueError):
        num = default
    return max(lo, min(hi, num))


def _to_float(value, default: float, lo: float, hi: float) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = default
    return max(lo, min(hi, num))


def _risk_summary(risk_pct: float) -> tuple[str, str]:
    if risk_pct < 20:
        return "Risiko Rendah (Aman)", (
            "Pertahankan gaya hidup sehat, cek gula darah berkala, dan lanjutkan aktivitas fisik rutin."
        )
    if risk_pct < 50:
        return "Risiko Sedang (Waspada)", (
            "Mulai perbaikan pola makan dan olahraga terstruktur 150 menit per minggu, serta cek lab saat memungkinkan."
        )
    return "Risiko Tinggi (Bahaya)", (
        "Segera lakukan konsultasi medis dan pemeriksaan gula darah lanjutan (GDP/HbA1c) di fasilitas kesehatan."
    )


def _build_chat_prompt(user_message: str, screening_result: dict | None) -> str:
    if not screening_result:
        return (
            "Pengguna belum punya hasil skrining terbaru. "
            "Jawab pertanyaan secara umum dalam konteks pencegahan diabetes, "
            "tetap ringkas, jelas, dan gunakan bahasa Indonesia. "
            f"\n\nPertanyaan pengguna: {user_message}"
        )

    risk_score = screening_result.get("riskScore")
    status = screening_result.get("status")
    factors = screening_result.get("factors", [])
    bmi = screening_result.get("bmi")
    age = screening_result.get("age")
    sex = screening_result.get("sex")

    return (
        "Berperan sebagai asisten kesehatan edukatif untuk skrining diabetes. "
        "Berikan jawaban dalam bahasa Indonesia yang empatik, praktis, dan tidak menakut-nakuti. "
        "Sertakan langkah yang bisa dilakukan harian, lalu akhiri dengan disclaimer singkat bahwa ini bukan diagnosis medis."
        "\n\nData skrining pengguna:"
        f"\n- Skor Risiko: {risk_score}%"
        f"\n- Status: {status}"
        f"\n- Usia: {age}"
        f"\n- Jenis Kelamin: {sex}"
        f"\n- BMI: {bmi}"
        f"\n- Faktor Risiko: {', '.join(factors) if factors else 'Tidak ada faktor dominan'}"
        f"\n\nPertanyaan pengguna: {user_message}"
    )


@app.get("/")
@app.get("/diabetes")
@app.get("/diabetes/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
@app.post("/diabetes/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "Payload JSON tidak valid."}), 400

    age_real = _to_int(payload.get("age"), 35, 1, 120)
    age_cdc = age_to_cdc_bucket(age_real)
    sex = _to_int(payload.get("Sex"), 0, 0, 1)

    values = {
        "HighBP": _to_int(payload.get("HighBP"), 0, 0, 1),
        "HighChol": _to_int(payload.get("HighChol"), 0, 0, 1),
        "CholCheck": _to_int(payload.get("CholCheck"), 1, 0, 1),
        "BMI": _to_float(payload.get("BMI"), 23.0, 10.0, 80.0),
        "Smoker": _to_int(payload.get("Smoker"), 0, 0, 1),
        "Stroke": _to_int(payload.get("Stroke"), 0, 0, 1),
        "HeartDiseaseorAttack": _to_int(payload.get("HeartDiseaseorAttack"), 0, 0, 1),
        "PhysActivity": _to_int(payload.get("PhysActivity"), 1, 0, 1),
        "Fruits": _to_int(payload.get("Fruits"), 1, 0, 1),
        "Veggies": _to_int(payload.get("Veggies"), 1, 0, 1),
        "HvyAlcoholConsump": _to_int(payload.get("HvyAlcoholConsump"), 0, 0, 1),
        "AnyHealthcare": _to_int(payload.get("AnyHealthcare"), 1, 0, 1),
        "NoDocbcCost": _to_int(payload.get("NoDocbcCost"), 0, 0, 1),
        "GenHlth": _to_int(payload.get("GenHlth"), 3, 1, 5),
        "MentHlth": _to_int(payload.get("MentHlth"), 0, 0, 30),
        "PhysHlth": _to_int(payload.get("PhysHlth"), 0, 0, 30),
        "DiffWalk": _to_int(payload.get("DiffWalk"), 0, 0, 1),
        "Sex": sex,
        "Age": age_cdc,
        "Education": _to_int(payload.get("Education"), 4, 1, 6),
        "Income": _to_int(payload.get("Income"), 6, 1, 8),
    }

    # Muat ketiga model nyata; model utama (BEST) wajib ada untuk skor risiko.
    models = {}
    for key, rel_path in MODEL_FILES.items():
        try:
            models[key] = load_model(str(BASE_DIR / rel_path))
        except FileNotFoundError:
            if key == BEST_MODEL_KEY:
                # Fallback ke nama lama bila file per-model belum digenerate.
                try:
                    models[key] = load_model(str(MODEL_FILE))
                except FileNotFoundError:
                    return (
                        jsonify({"ok": False, "error": f"File model tidak ditemukan: {rel_path}"}),
                        503,
                    )
                except Exception:
                    app.logger.exception("Failed to load fallback diabetes model")
                    return jsonify({"ok": False, "error": "Model skrining gagal dimuat."}), 500
        except Exception:
            app.logger.exception("Failed to load model %s", key)
            if key == BEST_MODEL_KEY:
                return jsonify({"ok": False, "error": "Model skrining gagal dimuat."}), 500

    best_model = models[BEST_MODEL_KEY]
    feature_order = list(getattr(best_model, "feature_names_in_", DEFAULT_FEATURE_ORDER))
    row = [values.get(feature, 0) for feature in feature_order]
    input_df = pd.DataFrame([row], columns=feature_order)

    # Probabilitas risiko (%) untuk tiap model yang berhasil dimuat.
    scores = {}
    for key, model in models.items():
        prob = predict_risk_probability(model, input_df)
        if prob is None:
            prob = 0.75 if int(model.predict(input_df)[0]) == 1 else 0.15
        scores[key] = round(float(prob) * 100.0, 2)

    pred = int(best_model.predict(input_df)[0])
    risk_pct = scores[BEST_MODEL_KEY]
    status, recommendation = _risk_summary(risk_pct)

    factors = []
    if values["HighBP"]:
        factors.append("Hipertensi")
    if values["HighChol"]:
        factors.append("Kolesterol Tinggi")
    if values["BMI"] >= 25:
        factors.append("Kelebihan Berat Badan")
    if values["HeartDiseaseorAttack"]:
        factors.append("Riwayat Penyakit Jantung")
    if values["Smoker"]:
        factors.append("Kebiasaan Merokok")
    if not values["PhysActivity"]:
        factors.append("Kurang Aktivitas Fisik")

    return jsonify(
        {
            "ok": True,
            "risk_score": round(risk_pct, 2),
            "scores": scores,
            "best_model": BEST_MODEL_KEY,
            "status": status,
            "recommendation": recommendation,
            "factors": factors,
            "prediction": pred,
        }
    )


@app.get("/api/metrics")
@app.get("/diabetes/api/metrics")
def metrics():
    try:
        data = json.loads(METRICS_FILE.read_text())
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "Metrik model belum tersedia."}), 404
    except (ValueError, OSError):
        return jsonify({"ok": False, "error": "Metrik model gagal dibaca."}), 500
    return jsonify({"ok": True, **data})


@app.post("/api/chat")
@app.post("/diabetes/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    user_message = str(payload.get("message", "")).strip()
    if len(user_message) > 2000:
        user_message = user_message[:2000]
    screening_result = payload.get("screening_result")
    if not isinstance(screening_result, dict):
        screening_result = None

    if not user_message:
        return jsonify({"ok": False, "error": "Pesan tidak boleh kosong."}), 400

    api_key = os.getenv("SUMOPOD_API_KEY", "").strip()
    model = os.getenv("SUMOPOD_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
    if not api_key:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "SUMOPOD_API_KEY belum diset di file .env",
                }
            ),
            503,
        )

    prompt = _build_chat_prompt(user_message, screening_result)
    system_prompt = (
        "Kamu adalah HealthAssistant AI untuk edukasi dini risiko diabetes. "
        "Hanya berikan edukasi dan rekomendasi gaya hidup, bukan diagnosis final."
    )

    try:
        resp = requests.post(
            "https://ai.sumopod.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 450,
                "temperature": 0.7,
            },
            timeout=45,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not text:
            return jsonify({"ok": False, "error": "Respons AI kosong."}), 502
        return jsonify({"ok": True, "reply": text})
    except requests.HTTPError as exc:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": f"Sumopod API error: {exc.response.status_code}",
                }
            ),
            502,
        )
    except requests.RequestException:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Gagal menghubungi Sumopod API.",
                }
            ),
            502,
        )


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1")
