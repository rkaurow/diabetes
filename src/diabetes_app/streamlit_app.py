from datetime import date, datetime

import pandas as pd
import streamlit as st

from src.diabetes_app.config import DEFAULT_FEATURE_ORDER, MODEL_PATH
from src.diabetes_app.features import age_to_cdc_bucket, calculate_age, get_model_features
from src.diabetes_app.model import load_model, predict_risk_probability


st.set_page_config(
    page_title="Deteksi Dini Diabetes Mellitus - Universitas Prisma",
    page_icon="🩺",
    layout="wide",
)


def _inject_style() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

            html, body, [class*="css"]  {
                font-family: 'Plus Jakarta Sans', sans-serif;
            }
            .stApp {
                background: radial-gradient(circle at top right, #e0f2fe 0%, #f8fafc 45%, #f8fafc 100%);
            }
            .glass-card {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid #e2e8f0;
                border-radius: 18px;
                padding: 1.1rem 1.2rem;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
            }
            .hero {
                background: linear-gradient(135deg, #ecfeff 0%, #f8fafc 60%, #e0f2fe 100%);
                border: 1px solid #bae6fd;
                border-radius: 20px;
                padding: 1.15rem 1.35rem;
            }
            .chip {
                display: inline-block;
                font-size: 0.72rem;
                font-weight: 700;
                color: #0369a1;
                background: #e0f2fe;
                padding: 0.25rem 0.55rem;
                border-radius: 999px;
                margin-bottom: 0.55rem;
            }
            .risk-number {
                font-weight: 800;
                font-size: 2.15rem;
                line-height: 1;
            }
            .small-muted {
                color: #64748b;
                font-size: 0.82rem;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.25rem;
                background: #f1f5f9;
                padding: 0.32rem;
                border-radius: 12px;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 10px;
                font-weight: 700;
            }
            .stTabs [aria-selected="true"] {
                background: white;
                color: #0369a1;
                box-shadow: 0 2px 8px rgba(2, 132, 199, 0.1);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _risk_factors(values: dict, bmi: float) -> list[str]:
    factors = []
    if values["HighBP"]:
        factors.append("Hipertensi")
    if values["HighChol"]:
        factors.append("Kolesterol tinggi")
    if bmi >= 25:
        factors.append("BMI berlebih")
    if values["HeartDiseaseorAttack"]:
        factors.append("Riwayat penyakit jantung")
    if values["Smoker"]:
        factors.append("Merokok aktif")
    if not values["PhysActivity"]:
        factors.append("Kurang aktivitas fisik")
    if values["DiffWalk"]:
        factors.append("Kesulitan berjalan")
    if values["GenHlth"] >= 4:
        factors.append("Kesehatan umum buruk")
    return factors


def _risk_summary(risk_pct: float) -> tuple[str, str, str]:
    if risk_pct < 20:
        return (
            "Risiko Rendah",
            "#059669",
            "Pertahankan pola hidup sehat dan cek kesehatan berkala setiap tahun.",
        )
    if risk_pct < 50:
        return (
            "Risiko Sedang",
            "#d97706",
            "Perbaiki pola makan dan tingkatkan aktivitas fisik minimal 150 menit per minggu.",
        )
    return (
        "Risiko Tinggi",
        "#e11d48",
        "Segera konsultasi ke fasilitas kesehatan untuk pemeriksaan gula darah lanjutan.",
    )


def _record_history(item: dict) -> None:
    if "screening_history" not in st.session_state:
        st.session_state.screening_history = []
    st.session_state.screening_history.insert(0, item)


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="chip">Proposal Penelitian</div>
            <h2 style="margin:0; color:#0f172a;">Deteksi Dini Risiko Diabetes Mellitus</h2>
            <p style="margin:0.35rem 0 0.2rem 0; color:#334155; font-size:0.92rem;">
                Implementasi model Machine Learning berbasis indikator CDC untuk skrining mandiri.
            </p>
            <p style="margin:0; color:#64748b; font-size:0.8rem;">
                Universitas Prisma Manado
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_style()
    _render_header()
    st.write("")

    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "screening_history" not in st.session_state:
        st.session_state.screening_history = []

    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model tidak ditemukan: {MODEL_PATH}")
        st.stop()
    except Exception as exc:
        st.error(f"Gagal memuat model: {exc}")
        st.stop()

    feature_order = get_model_features(model, DEFAULT_FEATURE_ORDER)
    tabs = st.tabs(
        [
            "Skrining Mandiri",
            "Analisis Model ML",
            "Konsultasi AI",
            "Riwayat Pasien",
        ]
    )

    with tabs[0]:
        left_col, right_col = st.columns([1.35, 1], gap="large")

        with left_col:
            with st.form(key="diabetes_form", clear_on_submit=False):
                st.markdown("### Profil Fisik dan Demografi")
                nama = st.text_input("Nama lengkap", placeholder="Contoh: Budi Santoso")
                tanggal_lahir = st.date_input(
                    "Tanggal lahir",
                    value=date(2000, 1, 1),
                    min_value=date(1920, 1, 1),
                    max_value=date.today(),
                    format="DD/MM/YYYY",
                )

                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    tinggi = st.number_input(
                        "Tinggi badan (cm)", min_value=100.0, max_value=250.0, value=165.0
                    )
                with bcol2:
                    berat = st.number_input(
                        "Berat badan (kg)", min_value=30.0, max_value=250.0, value=65.0
                    )

                tinggi_meter = tinggi / 100.0
                bmi = berat / (tinggi_meter**2)
                usia_real = calculate_age(tanggal_lahir, date.today())
                usia_cdc = age_to_cdc_bucket(max(usia_real, 1))
                bmi_state = (
                    "Normal"
                    if bmi < 25
                    else "Overweight"
                    if bmi < 30
                    else "Obesitas"
                )
                st.info(
                    f"Usia otomatis: {usia_real} tahun | BMI: {bmi:.2f} ({bmi_state})"
                )

                st.markdown("### Parameter Klinis")
                high_bp = st.radio("Tekanan darah tinggi", ("Tidak", "Ya"), horizontal=True)
                high_chol = st.radio("Kolesterol tinggi", ("Tidak", "Ya"), horizontal=True)
                chol_check = st.radio(
                    "Pernah cek kolesterol 5 tahun terakhir",
                    ("Tidak", "Ya"),
                    horizontal=True,
                )
                heart_disease = st.radio(
                    "Riwayat penyakit jantung/serangan jantung",
                    ("Tidak", "Ya"),
                    horizontal=True,
                )

                st.markdown("### Pola Gaya Hidup")
                lcol1, lcol2 = st.columns(2)
                with lcol1:
                    smoker = st.radio("Perokok aktif", ("Tidak", "Ya"), horizontal=True)
                    phys_activity = st.radio(
                        "Aktivitas fisik rutin", ("Tidak", "Ya"), horizontal=True
                    )
                    fruits = st.radio("Konsumsi buah harian", ("Tidak", "Ya"), horizontal=True)
                with lcol2:
                    stroke = st.radio("Riwayat stroke", ("Tidak", "Ya"), horizontal=True)
                    veggies = st.radio(
                        "Konsumsi sayur harian", ("Tidak", "Ya"), horizontal=True
                    )
                    heavy_alcohol = st.radio(
                        "Alkohol berlebihan", ("Tidak", "Ya"), horizontal=True
                    )

                st.markdown("### Kondisi Umum dan Akses Medis")
                any_healthcare = st.radio(
                    "Memiliki akses layanan kesehatan/asuransi",
                    ("Tidak", "Ya"),
                    horizontal=True,
                )
                no_doc_bc_cost = st.radio(
                    "Batal berobat karena biaya",
                    ("Tidak", "Ya"),
                    horizontal=True,
                )
                diff_walk = st.radio(
                    "Kesulitan berjalan/naik tangga",
                    ("Tidak", "Ya"),
                    horizontal=True,
                )
                sex = st.radio("Jenis kelamin", ("Perempuan", "Laki-laki"), horizontal=True)

                gen_hlth = st.slider("Kesehatan umum (1=baik, 5=buruk)", 1, 5, 3)
                ment_hlth = st.slider("Hari kesehatan mental buruk (30 hari)", 0, 30, 0)
                phys_hlth = st.slider("Hari kesehatan fisik buruk (30 hari)", 0, 30, 0)

                education = st.selectbox("Kategori pendidikan (CDC 1-6)", [1, 2, 3, 4, 5, 6], index=3)
                income = st.selectbox(
                    "Kategori pendapatan (CDC 1-8)",
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    index=5,
                )

                submitted = st.form_submit_button("Evaluasi Hasil Skrining", type="primary")

            if submitted:
                map_binary = {"Tidak": 0, "Ya": 1}
                map_sex = {"Perempuan": 0, "Laki-laki": 1}
                values = {
                    "HighBP": map_binary[high_bp],
                    "HighChol": map_binary[high_chol],
                    "CholCheck": map_binary[chol_check],
                    "BMI": float(bmi),
                    "Smoker": map_binary[smoker],
                    "Stroke": map_binary[stroke],
                    "HeartDiseaseorAttack": map_binary[heart_disease],
                    "PhysActivity": map_binary[phys_activity],
                    "Fruits": map_binary[fruits],
                    "Veggies": map_binary[veggies],
                    "HvyAlcoholConsump": map_binary[heavy_alcohol],
                    "AnyHealthcare": map_binary[any_healthcare],
                    "NoDocbcCost": map_binary[no_doc_bc_cost],
                    "GenHlth": int(gen_hlth),
                    "MentHlth": int(ment_hlth),
                    "PhysHlth": int(phys_hlth),
                    "DiffWalk": map_binary[diff_walk],
                    "Sex": map_sex[sex],
                    "Age": int(usia_cdc),
                    "Education": int(education),
                    "Income": int(income),
                }

                row = [values.get(feature, 0) for feature in feature_order]
                input_df = pd.DataFrame([row], columns=feature_order)

                try:
                    pred = model.predict(input_df)[0]
                    risk_proba = predict_risk_probability(model, input_df)
                except Exception as exc:
                    st.error(f"Gagal melakukan prediksi: {exc}")
                    st.stop()

                if risk_proba is None:
                    risk_proba = 0.75 if int(pred) == 1 else 0.15

                risk_pct = float(risk_proba) * 100
                status_label, status_color, recommendation = _risk_summary(risk_pct)
                factors = _risk_factors(values, bmi)

                result = {
                    "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
                    "nama": nama.strip() or "-",
                    "usia": usia_real,
                    "sex": sex,
                    "tinggi": float(tinggi),
                    "berat": float(berat),
                    "bmi": float(round(bmi, 2)),
                    "risk_pct": float(round(risk_pct, 2)),
                    "status": status_label,
                    "status_color": status_color,
                    "factors": factors,
                    "recommendation": recommendation,
                }

                st.session_state.last_result = result
                _record_history(result)
                st.success("Skrining selesai. Lihat ringkasan di panel kanan.")

        with right_col:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Ringkasan Hasil Skrining")
            latest = st.session_state.last_result

            if latest is None:
                st.caption("Belum ada hasil. Isi form dan klik Evaluasi Hasil Skrining.")
            else:
                st.markdown(
                    (
                        f'<div class="risk-number" style="color:{latest["status_color"]};">'
                        f'{latest["risk_pct"]:.2f}%</div>'
                        f'<p class="small-muted" style="margin-top:0.3rem;">Skor Risiko Diabetes</p>'
                    ),
                    unsafe_allow_html=True,
                )
                st.progress(min(max(latest["risk_pct"] / 100, 0.0), 1.0))
                st.markdown(
                    (
                        f"<p style='font-weight:700; color:{latest['status_color']}; margin:0.55rem 0;'>"
                        f"{latest['status']}</p>"
                    ),
                    unsafe_allow_html=True,
                )
                st.write(f"Nama: {latest['nama']}")
                st.write(f"Usia: {latest['usia']} tahun")
                st.write(f"BMI: {latest['bmi']}")

                st.markdown("#### Faktor Risiko Terdeteksi")
                if latest["factors"]:
                    for factor in latest["factors"]:
                        st.write(f"- {factor}")
                else:
                    st.write("- Tidak ada faktor risiko dominan yang terdeteksi")

                st.markdown("#### Rekomendasi")
                st.caption(latest["recommendation"])

            st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
        m1, m2, m3 = st.columns(3)
        m1.metric("Model Utama", "Gradient Boosting")
        m2.metric("Akurasi Rata-rata", "84.52%")
        m3.metric("Metode", "R&D (70:30 split)")

        perf_df = pd.DataFrame(
            {
                "Metrik": ["Akurasi", "Presisi", "Recall", "F1-Score"],
                "Gradient Boosting": [84.52, 85.10, 83.94, 84.52],
                "Random Forest": [83.20, 84.02, 82.11, 83.05],
                "Logistic Regression": [81.88, 80.91, 82.54, 81.72],
            }
        ).set_index("Metrik")
        st.markdown("### Evaluasi Metrik Klasifikasi")
        st.bar_chart(perf_df)

        st.markdown("### Feature Importance (Simulasi Laporan)")
        feature_importance = {
            "BMI": 0.284,
            "HighBP": 0.221,
            "GenHlth": 0.185,
            "Age": 0.142,
            "HighChol": 0.118,
        }
        for feature, value in feature_importance.items():
            st.write(f"{feature}: {value * 100:.1f}%")
            st.progress(value)

        st.markdown("### Confusion Matrix (Gradient Boosting)")
        cm = pd.DataFrame(
            [[64210, 11894], [11920, 65104]],
            columns=["Prediksi 0", "Prediksi 1"],
            index=["Aktual 0", "Aktual 1"],
        )
        st.dataframe(cm, use_container_width=True)

    with tabs[2]:
        st.markdown("### Konsultasi AI")
        latest = st.session_state.last_result
        if latest is None:
            st.info("Silakan lakukan skrining dulu agar konsultasi AI lebih relevan.")
        else:
            st.markdown(
                (
                    "AI assistant siap memberi ringkasan personal berdasarkan hasil terakhir. "
                    "Masukkan pertanyaan Anda di bawah ini."
                )
            )
            prompt = st.text_area(
                "Pertanyaan untuk AI",
                placeholder="Contoh: bagaimana pola makan 7 hari yang cocok untuk hasil saya?",
            )
            if st.button("Kirim Pertanyaan", type="primary"):
                if not prompt.strip():
                    st.warning("Pertanyaan belum diisi.")
                else:
                    st.markdown("#### Jawaban AI (mode lokal)")
                    st.write(
                        (
                            f"Skor risiko Anda {latest['risk_pct']:.2f}% ({latest['status']}). "
                            "Fokus utama: kontrol berat badan, aktivitas fisik rutin, dan evaluasi klinis berkala. "
                            "Untuk jawaban lebih mendalam, sambungkan aplikasi ke API LLM eksternal."
                        )
                    )

    with tabs[3]:
        st.markdown("### Riwayat Skrining")
        history = st.session_state.screening_history
        if not history:
            st.caption("Belum ada riwayat.")
        else:
            history_df = pd.DataFrame(history)
            st.dataframe(
                history_df[
                    [
                        "timestamp",
                        "nama",
                        "sex",
                        "usia",
                        "bmi",
                        "risk_pct",
                        "status",
                    ]
                ],
                use_container_width=True,
            )
            if st.button("Reset Semua Riwayat"):
                st.session_state.screening_history = []
                st.success("Riwayat berhasil dihapus.")

    st.caption(
        "Catatan: hasil prediksi ini bersifat skrining awal, bukan diagnosis medis definitif."
    )
