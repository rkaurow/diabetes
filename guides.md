# Guides - Diabetes Detection App

## 1. Ringkasan
Project ini sekarang punya 2 mode aplikasi:

1. Website mandiri (Flask + HTML Tailwind) - direkomendasikan
2. Streamlit app (versi lama, tetap tersedia)

Model prediksi yang dipakai tetap model Python yang sama: `model_diabetes_terbaik.pkl`.

## 2. Struktur Project

- `webapp.py`: backend Flask + endpoint `/api/predict` dan `/api/chat`
- `templates/index.html`: frontend website (UI utama + chatbot)
- `src/diabetes_app/`: modul Python (config, fitur, model, streamlit)
- `app.py`: entrypoint Streamlit
- `scripts/fix_model.py`: script helper kompatibilitas model
- `requirements.txt`: dependency Python
- `.env.example`: contoh konfigurasi API key Sumopod

## 3. Persiapan Environment

1. Masuk ke folder project

```bash
cd /Users/sdamedia/Developer/python-proj/diabetes
```

2. Install dependency

```bash
pip3 install -r requirements.txt
```

3. Setup API key Sumopod untuk chatbot AI

```bash
cp .env.example .env
```

Isi file `.env`:

```env
SUMOPOD_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## 4. Menjalankan Website (Flask) - Direkomendasikan

1. Jalankan server Flask

```bash
python3 webapp.py
```

2. Buka browser

```text
http://127.0.0.1:5000
```

3. Alur penggunaan
- Isi form skrining di tab Skrining Mandiri
- Klik Evaluasi Hasil Skrining
- Frontend kirim data ke backend `/api/predict`
- Setelah hasil muncul, klik tombol `Konsultasikan Lanjut dengan AI`
- Chatbot akan memanggil backend `/api/chat` dan meneruskan ke Sumopod API

## 5. Menjalankan Streamlit (Opsional)

Kalau masih ingin pakai Streamlit:

```bash
streamlit run app.py
```

## 6. API Endpoint

### `POST /api/predict`

Content-Type: `application/json`

Contoh payload minimal:

```json
{
  "age": 35,
  "Sex": 0,
  "BMI": 24.7,
  "HighBP": 0,
  "HighChol": 0,
  "CholCheck": 1,
  "Smoker": 0,
  "Stroke": 0,
  "HeartDiseaseorAttack": 0,
  "PhysActivity": 1,
  "Fruits": 1,
  "Veggies": 1,
  "HvyAlcoholConsump": 0,
  "AnyHealthcare": 1,
  "NoDocbcCost": 0,
  "GenHlth": 3,
  "MentHlth": 0,
  "PhysHlth": 0,
  "DiffWalk": 0,
  "Education": 4,
  "Income": 6
}
```

Contoh response:

```json
{
  "ok": true,
  "risk_score": 12.34,
  "status": "Risiko Rendah (Aman)",
  "recommendation": "Pertahankan gaya hidup sehat...",
  "factors": ["Hipertensi"],
  "prediction": 0
}
```

### `POST /api/chat`

Content-Type: `application/json`

Contoh payload:

```json
{
  "message": "Apa langkah pertama yang harus saya lakukan minggu ini?",
  "screening_result": {
    "riskScore": 43.2,
    "status": "Risiko Sedang (Waspada)",
    "age": 35,
    "sex": "Perempuan",
    "bmi": 28.1,
    "factors": ["Kelebihan Berat Badan", "Kurang Aktivitas Fisik"]
  }
}
```

Contoh response:

```json
{
  "ok": true,
  "reply": "Berikut langkah praktis 7 hari ke depan..."
}
```

## 7. Troubleshooting

1. Error model tidak ditemukan
- Pastikan file `model_diabetes_terbaik.pkl` ada di root project.

2. Import/module error
- Jalankan ulang `pip3 install -r requirements.txt`.

3. Chatbot error `SUMOPOD_API_KEY belum diset di file .env`
- Pastikan file `.env` ada di root project.
- Pastikan variabel `SUMOPOD_API_KEY` terisi benar.

4. Port 5000 sudah dipakai
- Ubah port di `webapp.py`, contoh:

```python
if __name__ == "__main__":
    app.run(debug=True, port=5001)
```

5. Tampilan tidak update
- Hard refresh browser (`Cmd + Shift + R`).

## 8. Catatan Penting

- Hasil prediksi adalah skrining awal, bukan diagnosis medis final.
- Untuk validasi klinis, tetap lakukan pemeriksaan laboratorium dan konsultasi dokter.
