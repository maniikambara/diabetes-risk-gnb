"""Titik masuk utama proyek klasifikasi risiko diabetes.

Target klasifikasi: DiagnosisType (Normal / Tipe 2 / Gestasional),
dibangun dari DIAGNOSIS KLINIS ASLI dua studi nyata:
  - Pima Indians Diabetes Database (Outcome 0/1 -> Normal/Tipe 2)
  - Gestational Diabetes Mellitus (GDM) Data Set (Class 0/1 -> Normal/Gestasional)

digabung lewat lima atribut yang punya padanan konsep langsung di
kedua dataset (Age, Pregnancies, BMI, BloodPressure, Glucose). Lihat
src/merge.py dan notebooks/01_EDA.ipynb untuk metodologi dan
keterbatasan lengkap -- termasuk pembuktian bahwa SumberData (Pima vs
GDM) lebih mudah ditebak dari kelima fitur ini daripada DiagnosisType
itu sendiri.

Jalankan dengan:
    python main.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from src.data_loader import load_data
from src.preprocessing import mark_missing, impute_missing_by_class, ZERO_AS_MISSING_COLUMNS
from src.gdm_loader import load_gdm_data, impute_gdm_missing_by_class, harmonize_gdm
from src.merge import build_pima_portion, build_gdm_portion, merge_datasets, DIAGNOSIS_LABELS
from src.pipeline import train_and_predict, split_features_target, RANDOM_STATE, TEST_SIZE
from src.evaluate import evaluate_predictions

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
GDM_PATH = BASE_DIR / "data" / "gdm_dataset.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"


def simpan_plot_confusion_matrix(cm, labels, path_output):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    n = len(labels)
    ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)
    nilai_maks = max(max(row) for row in cm) or 1
    for i in range(n):
        for j in range(n):
            warna = "white" if cm[i][j] > nilai_maks / 2 else "black"
            ax.text(j, i, cm[i][j], ha="center", va="center", color=warna)
    ax.set_title("Confusion Matrix - DiagnosisType")
    fig.tight_layout()
    fig.savefig(path_output, dpi=150)
    plt.close(fig)


def jalankan():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df_pima = load_data(DATA_PATH)
    df_pima_bersih = impute_missing_by_class(mark_missing(df_pima))

    df_gdm = load_gdm_data(GDM_PATH)
    df_gdm_bersih = impute_gdm_missing_by_class(df_gdm)
    df_gdm_harmon = harmonize_gdm(df_gdm_bersih)

    pima_portion = build_pima_portion(df_pima_bersih)
    gdm_portion = build_gdm_portion(df_gdm_harmon)
    gabungan = merge_datasets(pima_portion, gdm_portion)

    df_model = gabungan.drop(columns=["SumberData"])
    hasil = train_and_predict(df_model, target_col="DiagnosisType")
    model = hasil["model"]
    metrik = evaluate_predictions(hasil["y_test"], hasil["y_pred"], labels=DIAGNOSIS_LABELS)

    # Uji perancu: seberapa mudah SumberData ditebak dari fitur yang sama.
    X, y = split_features_target(df_model, target_col="DiagnosisType")
    X_train, X_test, _, _, idx_train, idx_test = train_test_split(
        X, y, df_model.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    sumber_train = gabungan.loc[idx_train, "SumberData"]
    sumber_test = gabungan.loc[idx_test, "SumberData"]
    model_sumber = GaussianNB().fit(X_train, sumber_train)
    akurasi_sumber = model_sumber.score(X_test, sumber_test)

    distribusi_kelas = gabungan["DiagnosisType"].value_counts().reindex(DIAGNOSIS_LABELS).to_dict()

    ringkasan = {
        "bentuk_pima": list(df_pima.shape),
        "bentuk_gdm": list(df_gdm.shape),
        "bentuk_gabungan": list(gabungan.shape),
        "distribusi_diagnosistype": distribusi_kelas,
        "kelas_dipelajari_model": list(model.classes_),
        "metrik": metrik,
        "akurasi_tebak_sumber_data": round(float(akurasi_sumber), 4),
    }

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(ringkasan, f, indent=2, ensure_ascii=False)

    simpan_plot_confusion_matrix(metrik["confusion_matrix"], DIAGNOSIS_LABELS, OUTPUT_DIR / "confusion_matrices.png")

    cetak_ringkasan(ringkasan)


def cetak_ringkasan(ringkasan):
    print("Bentuk Pima:", ringkasan["bentuk_pima"], "| Bentuk GDM:", ringkasan["bentuk_gdm"])
    print("Bentuk gabungan:", ringkasan["bentuk_gabungan"])
    print("Distribusi DiagnosisType:", ringkasan["distribusi_diagnosistype"])
    print("Kelas dipelajari model:", ringkasan["kelas_dipelajari_model"])
    print()
    m = ringkasan["metrik"]
    print(f"Akurasi DiagnosisType: {m['akurasi']}  |  Akurasi tebak SumberData: {ringkasan['akurasi_tebak_sumber_data']}")
    print("Precision/Recall/F1 makro:", m["precision_macro"], m["recall_macro"], m["f1_macro"])
    print("Per kelas:", m["per_kelas"])


if __name__ == "__main__":
    jalankan()
