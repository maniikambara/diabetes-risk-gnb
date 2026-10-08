"""Aplikasi Streamlit: Deteksi Kategori Diagnosis Diabetes (Gaussian Naive Bayes, 3 kelas).

Target klasifikasi: DiagnosisType (Normal / Tipe 2 / Gestasional), dibangun
dari diagnosis klinis ASLI dua studi nyata -- Pima Indians Diabetes Database
(Tipe 2) dan Gestational Diabetes Mellitus (GDM) Data Set (Gestasional) --
digabung lewat lima atribut yang terukur di kedua dataset: Age, Pregnancies,
BMI, BloodPressure, Glucose. Model: notebooks/03_GaussianNB.ipynb.

PENTING: pengujian pada notebooks/04_Evaluation.ipynb membuktikan model
"menebak sumber data" (Pima vs GDM) LEBIH MUDAH (93,8% akurasi) daripada
menebak DiagnosisType itu sendiri (85,9%) -- kelima fitur ini sebagian
membawa sinyal populasi/status kehamilan, bukan murni tipe diabetes. Baca
ini sebelum menafsirkan hasil prediksi aplikasi ini.

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

DIAGNOSIS_LABELS = ["Normal", "Tipe 2", "Gestasional"]
KELAS_WARNA = {"Normal": "success", "Tipe 2": "error", "Gestasional": "warning"}
KELAS_IKON = {"Normal": "\u2705", "Tipe 2": "\U0001F6A8", "Gestasional": "\u26a0\ufe0f"}

# Field khusus untuk model gabungan: hanya lima fitur bersama Pima + GDM.
FIELD_CONFIG = {
    "Age": dict(label="Usia (tahun)", min_value=1, max_value=120, value=30, step=1,
                help="Usia pasien dalam tahun."),
    "Pregnancies": dict(label="Jumlah Kehamilan", min_value=0, max_value=20, value=1, step=1,
                         help="Pima: total kehamilan seumur hidup. GDM: kehamilan ke berapa saat ini. Lihat catatan keterbatasan di bawah."),
    "BMI": dict(label="Indeks Massa Tubuh / BMI (kg/m\u00b2)", min_value=10.0, max_value=70.0, value=25.0, step=0.1,
                help="Indeks massa tubuh."),
    "BloodPressure": dict(label="Tekanan Darah Diastolik (mmHg)", min_value=40, max_value=140, value=70, step=1,
                           help="Tekanan darah diastolik."),
    "Glucose": dict(label="Kadar Glukosa 2 Jam Pasca-OGTT (mg/dL)", min_value=40, max_value=420, value=100, step=1,
                     help="Hasil tes toleransi glukosa oral 2 jam."),
}


@st.cache_resource
def muat_artefak():
    model = joblib.load(MODELS_DIR / "gnb_diagnosis.joblib")
    with open(MODELS_DIR / "feature_columns.json", encoding="utf-8") as f:
        kolom_fitur = json.load(f)
    return model, kolom_fitur


@st.cache_data
def muat_contoh_pasien():
    """Ambil satu contoh baris nyata per kelas DiagnosisType dari dataset
    gabungan sebagai isian cepat."""
    path = APP_DIR / "data" / "processed" / "merged_diagnosis.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    contoh = {}
    for kelas in DIAGNOSIS_LABELS:
        baris = df[df["DiagnosisType"] == kelas]
        if len(baris) > 0:
            contoh[kelas] = baris.iloc[0].drop(labels=["DiagnosisType", "SumberData"], errors="ignore").to_dict()
    return contoh


def main():
    st.set_page_config(page_title="Deteksi Kategori Diagnosis Diabetes", page_icon="\U0001FA7A", layout="centered")

    model, kolom_fitur = muat_artefak()
    kelas_dipelajari = list(model.classes_)

    st.title("\U0001FA7A Deteksi Kategori Diagnosis Diabetes")
    st.caption("Klasifikasi 3 Kelas (Normal / Tipe 2 / Gestasional) dari Diagnosis Asli Pima + GDM, Gaussian Naive Bayes")

    st.info(
        "Model dilatih dari gabungan **Pima Indians Diabetes Database** "
        "(768 pasien, label Tipe 2 dari diagnosis T2D asli) dan "
        "**Gestational Diabetes Mellitus (GDM) Data Set** (3.525 pasien hamil, "
        "label Gestasional dari diagnosis GDM asli). Hasil prediksi **bukan "
        "diagnosis medis** dan tidak menggantikan pemeriksaan tenaga medis "
        "profesional.",
        icon="\u2139\ufe0f",
    )

    st.warning(
        "**Keterbatasan penting, mohon dibaca:** pengujian langsung pada "
        "`notebooks/04_Evaluation.ipynb` membuktikan model lebih mudah "
        "menebak **sumber data** (Pima vs GDM, akurasi 93,8%) daripada "
        "menebak **DiagnosisType** itu sendiri (85,9%). Artinya kelima "
        "fitur di bawah sebagian membawa sinyal populasi/status kehamilan "
        "(pasien GDM selalu sedang hamil, pasien Pima tidak selalu), bukan "
        "murni pola fisiologis tipe diabetes. Kategori **Tipe 2** juga "
        "hanya didukung sedikit data latih (6,2% dari total) sehingga "
        "prediksinya kurang dapat diandalkan dibanding dua kategori lain.",
        icon="\u26a0\ufe0f",
    )

    contoh_per_kelas = muat_contoh_pasien()
    if contoh_per_kelas:
        st.write("Isi cepat dengan contoh data pasien nyata dari dataset:")
        kolom_tombol = st.columns(len(contoh_per_kelas) + 1)
        for kol, (kelas, nilai_contoh) in zip(kolom_tombol, contoh_per_kelas.items()):
            if kol.button(f"Contoh: {kelas}", use_container_width=True):
                st.session_state.update(nilai_contoh)
        if kolom_tombol[-1].button("Kosongkan", use_container_width=True):
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
        submitted = st.form_submit_button("Deteksi Kategori Diagnosis", use_container_width=True, type="primary")

    if not submitted:
        return

    X_input = pd.DataFrame([nilai_input], columns=kolom_fitur)

    prediksi = model.predict(X_input)[0]
    proba_model = dict(zip(model.classes_, model.predict_proba(X_input)[0]))
    proba_lengkap = {kelas: proba_model.get(kelas, 0.0) for kelas in DIAGNOSIS_LABELS}

    st.divider()
    st.subheader("Hasil Deteksi")

    tampilan = getattr(st, KELAS_WARNA.get(prediksi, "info"))
    tampilan(f"**Kategori: {prediksi}** -- probabilitas posterior: {proba_lengkap[prediksi]:.1%}",
             icon=KELAS_IKON.get(prediksi, "\u2139\ufe0f"))

    kolom_metrik = st.columns(3)
    for kol, kelas in zip(kolom_metrik, DIAGNOSIS_LABELS):
        kol.metric(kelas, f"{proba_lengkap[kelas]:.1%}")

    grafik = pd.DataFrame({"Probabilitas": [proba_lengkap[k] for k in DIAGNOSIS_LABELS]}, index=DIAGNOSIS_LABELS)
    st.bar_chart(grafik, horizontal=True)

    with st.expander("Bagaimana probabilitas ini dihitung?"):
        st.markdown(
            "Model **Gaussian Naive Bayes** menghitung probabilitas posterior "
            "tiap kelas dari probabilitas prior dikalikan kepadatan probabilitas "
            "normal tiap atribut pada kelas tersebut, mengikuti Teorema Bayes "
            "dengan asumsi independensi antar atribut. Label `DiagnosisType` "
            "berasal dari diagnosis klinis asli dua studi (Pima + GDM), "
            "digabung lewat lima atribut yang terukur di keduanya -- lihat "
            "`notebooks/02_Preprocessing.ipynb` untuk metodologi penggabungan "
            "dan `notebooks/03_GaussianNB.ipynb` untuk pembuktian manual "
            "bahwa posterior ini identik dengan `predict_proba()`."
        )
        st.dataframe(X_input.T.rename(columns={0: "Nilai yang dipakai model"}))

    st.caption(
        "Model: Gaussian Naive Bayes (scikit-learn), target DiagnosisType 3 kelas "
        "dari diagnosis asli Pima + GDM -- lihat notebooks/01_EDA.ipynb s.d. "
        "notebooks/04_Evaluation.ipynb untuk metodologi dan keterbatasan lengkap."
    )


if __name__ == "__main__":
    main()
