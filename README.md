# Skrining Risiko Diabetes

Aplikasi web edukatif untuk memprediksi risiko diabetes menggunakan model machine learning (scikit-learn) yang dilatih pada dataset BRFSS. Tersedia dalam dua antarmuka: **Flask** (web + chat AI) dan **Streamlit**.

> ⚠️ Hasil prediksi bersifat edukatif, bukan diagnosis medis.

## Persiapan

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Untuk fitur chat AI (versi Flask), salin `.env.example` ke `.env` lalu isi `SUMOPOD_API_KEY`.

## Menjalankan

**Versi Flask** (UI web di `templates/index.html`):

```bash
python webapp.py
# buka http://127.0.0.1:5000
```

**Versi Streamlit:**

```bash
streamlit run app.py
```

## Struktur Proyek

| Path | Keterangan |
| --- | --- |
| `webapp.py` | Backend Flask: endpoint `/api/predict` dan `/api/chat` |
| `app.py` | Entry point Streamlit |
| `src/diabetes_app/` | Logika inti: load model, fitur, konfigurasi |
| `models/model_diabetes_terbaik.pkl` | Model terlatih |
| `diabetes.ipynb` | Notebook eksplorasi data & pelatihan model |
| `scripts/fix_model.py` | Utilitas kompatibilitas pickle NumPy |
| `docs/panduan_belajar_diabetes.md`, `docs/PANDUAN_BELAJAR_SMA.md` | Materi belajar |
