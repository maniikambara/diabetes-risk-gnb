"""Aplikasi Streamlit: Deteksi Risiko Diabetes (Gaussian Naive Bayes).

Menggunakan model yang dilatih dan dievaluasi pada notebooks/03_GaussianNB.ipynb
dan notebooks/04_Evaluation.ipynb (model models/gnb_clean.joblib, dilatih pada
data yang sudah melalui tahap pembersihan pada notebooks/02_Preprocessing.ipynb).

Jalankan dengan:
    streamlit run app_streamlit.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
DATA_PATH = APP_DIR / "data" / "diabetes.csv"

# Kolom yang secara medis tidak mungkin bernilai 0 pada pasien hidup.
# Identik dengan src/preprocessing.py::ZERO_AS_MISSING_COLUMNS.
ZERO_AS_MISSING_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

FIELD_CONFIG = {
    "Pregnancies": dict(label="Jumlah Kehamilan", min_value=0, max_value=20, value=1, step=1,
                         help="Jumlah kehamilan yang pernah dialami. 0 adalah nilai valid (belum pernah hamil)."),
    "Glucose": dict(label="Kadar Glukosa Plasma (mg/dL)", min_value=0, max_value=250, value=0, step=1,
                    help="Hasil tes toleransi glukosa oral 2 jam. Isi 0 jika tidak diketahui -- akan diestimasi otomatis."),
    "BloodPressure": dict(label="Tekanan Darah Diastolik (mmHg)", min_value=0, max_value=140, value=0, step=1,
                           help="Isi 0 jika tidak diketahui -- akan diestimasi otomatis."),
    "SkinThickness": dict(label="Ketebalan Lipatan Kulit Trisep (mm)", min_value=0, max_value=100, value=0, step=1,
                           help="Isi 0 jika tidak diketahui -- akan diestimasi otomatis."),
    "Insulin": dict(label="Kadar Insulin Serum 2 Jam (mu U/mL)", min_value=0, max_value=900, value=0, step=1,
                     help="Isi 0 jika tidak diketahui -- akan diestimasi otomatis."),
    "BMI": dict(label="Indeks Massa Tubuh / BMI (kg/m\u00b2)", min_value=0.0, max_value=70.0, value=0.0, step=0.1,
                help="Isi 0 jika tidak diketahui -- akan diestimasi otomatis."),
    "DiabetesPedigreeFunction": dict(label="Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01,
                                      help="Skor riwayat diabetes pada keluarga (semakin tinggi, semakin kuat riwayat keluarga)."),
    "Age": dict(label="Usia (tahun)", min_value=1, max_value=120, value=30, step=1, help="Usia pasien dalam tahun."),
}


@st.cache_resource
def muat_artefak():
    model = joblib.load(MODELS_DIR / "gnb_clean.joblib")
    with open(MODELS_DIR / "feature_columns.json", encoding="utf-8") as f:
        kolom_fitur = json.load(f)
    with open(MODELS_DIR / "imputation_medians.json", encoding="utf-8") as f:
        median_imputasi = json.load(f)
    return model, kolom_fitur, median_imputasi


@st.cache_data
def muat_contoh_pasien():
    """Ambil satu contoh baris nyata per kelas dari dataset sebagai isian cepat."""
    if not DATA_PATH.exists():
        return None, None
    df = pd.read_csv(DATA_PATH)
    contoh_rendah = df[df["Outcome"] == 0].iloc[0].drop("Outcome").to_dict()
    contoh_tinggi = df[df["Outcome"] == 1].iloc[0].drop("Outcome").to_dict()
    return contoh_rendah, contoh_tinggi


def siapkan_input(nilai_mentah: dict, kolom_fitur: list, median_imputasi: dict):
    """Susun satu baris input dengan urutan kolom identik dengan saat
    pelatihan model, dan gantikan 0 dengan median data latih pada kolom
    fisiologis yang secara medis tidak mungkin bernilai 0.

    Median dipakai di sini dihitung HANYA dari data latih (lihat
    notebooks/02_Preprocessing.ipynb, bagian 5) -- bukan median per
    kelas seperti pada pipeline evaluasi offline, karena pada saat
    aplikasi ini dipakai, label kelas pasien justru belum diketahui.
    """
    baris = {}
    kolom_diimputasi = []
    for kolom in kolom_fitur:
        nilai = nilai_mentah[kolom]
        if kolom in ZERO_AS_MISSING_COLUMNS and nilai == 0:
            nilai = median_imputasi[kolom]
            kolom_diimputasi.append(kolom)
        baris[kolom] = nilai
    return pd.DataFrame([baris], columns=kolom_fitur), kolom_diimputasi


def main():
    st.set_page_config(page_title="Deteksi Risiko Diabetes", page_icon="\U0001FA7A", layout="centered")

    model, kolom_fitur, median_imputasi = muat_artefak()

    st.title("\U0001FA7A Deteksi Risiko Diabetes")
    st.caption("Klasifikasi Risiko Diabetes Menggunakan Gaussian Naive Bayes Berdasarkan Data Diagnostik Pasien")

    st.info(
        "Aplikasi ini adalah proyek akademik yang dilatih pada **Pima Indians "
        "Diabetes Database** (768 pasien perempuan keturunan Pima Indian, "
        "usia \u2265 21 tahun). Hasil prediksi **bukan diagnosis medis** dan "
        "tidak menggantikan pemeriksaan oleh tenaga medis profesional.",
        icon="\u2139\ufe0f",
    )

    contoh_rendah, contoh_tinggi = muat_contoh_pasien()
    if contoh_rendah is not None:
        st.write("Isi cepat dengan contoh data pasien nyata dari dataset:")
        kol1, kol2, kol3 = st.columns(3)
        if kol1.button("Contoh: risiko rendah", use_container_width=True):
            st.session_state.update(contoh_rendah)
        if kol2.button("Contoh: risiko tinggi", use_container_width=True):
            st.session_state.update(contoh_tinggi)
        if kol3.button("Kosongkan", use_container_width=True):
            for kolom in kolom_fitur:
                st.session_state.pop(kolom, None)

    st.divider()

    with st.form("form_deteksi"):
        st.subheader("Indikator Medis Pasien")
        kol_kiri, kol_kanan = st.columns(2)
        nilai_input = {}
        for i, kolom in enumerate(kolom_fitur):
            target_kolom = kol_kiri if i % 2 == 0 else kol_kanan
            cfg = FIELD_CONFIG[kolom]
            default = st.session_state.get(kolom, cfg["value"])
            nilai_input[kolom] = target_kolom.number_input(
                cfg["label"],
                min_value=cfg["min_value"],
                max_value=cfg["max_value"],
                value=type(cfg["value"])(default),
                step=cfg["step"],
                help=cfg["help"],
                key=f"input_{kolom}",
            )
        submitted = st.form_submit_button("Deteksi Risiko Diabetes", use_container_width=True, type="primary")

    if not submitted:
        return

    X_input, kolom_diimputasi = siapkan_input(nilai_input, kolom_fitur, median_imputasi)

    if kolom_diimputasi:
        st.caption(
            "Nilai berikut diisi 0 (tidak diketahui) dan diestimasi otomatis "
            "memakai median data latih: " + ", ".join(kolom_diimputasi)
        )

    prediksi = model.predict(X_input)[0]
    probabilitas = model.predict_proba(X_input)[0]
    prob_tidak_diabetes, prob_diabetes = probabilitas[0], probabilitas[1]

    st.divider()
    st.subheader("Hasil Deteksi")

    if prediksi == 1:
        st.error(f"**Berisiko diabetes** -- probabilitas posterior: {prob_diabetes:.1%}", icon="\u26a0\ufe0f")
    else:
        st.success(f"**Tidak berisiko diabetes** -- probabilitas posterior: {prob_tidak_diabetes:.1%}", icon="\u2705")

    kol1, kol2 = st.columns(2)
    kol1.metric("P(Tidak Diabetes)", f"{prob_tidak_diabetes:.1%}")
    kol2.metric("P(Diabetes)", f"{prob_diabetes:.1%}")

    grafik = pd.DataFrame(
        {"Probabilitas": [prob_tidak_diabetes, prob_diabetes]},
        index=["Tidak Diabetes", "Diabetes"],
    )
    st.bar_chart(grafik, horizontal=True)

    with st.expander("Bagaimana probabilitas ini dihitung?"):
        st.markdown(
            "Model **Gaussian Naive Bayes** menghitung probabilitas posterior "
            "tiap kelas dari probabilitas prior dikalikan kepadatan probabilitas "
            "normal (Gaussian) tiap atribut pada kelas tersebut, mengikuti "
            "Teorema Bayes dengan asumsi independensi antar atribut. Langkah "
            "perhitungannya, termasuk pembuktian bahwa hasilnya identik dengan "
            "`predict_proba()`, didemonstrasikan secara manual pada "
            "`notebooks/03_GaussianNB.ipynb`."
        )
        st.dataframe(X_input.T.rename(columns={0: "Nilai yang dipakai model"}))

    st.caption(
        "Model: Gaussian Naive Bayes (scikit-learn), dilatih pada data yang "
        "sudah melalui pembersihan nilai hilang -- lihat notebooks/02_Preprocessing.ipynb, "
        "notebooks/03_GaussianNB.ipynb, dan notebooks/04_Evaluation.ipynb untuk detail lengkap."
    )


if __name__ == "__main__":
    main()
