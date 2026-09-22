"""Pemuatan data mentah Pima Indians Diabetes Dataset.

Modul ini hanya bertanggung jawab membaca berkas CSV dan memverifikasi
bahwa bentuk serta kolomnya sesuai dengan yang diasumsikan oleh seluruh
pipeline. Verifikasi statistik deskriptif dipakai sebagai pemeriksaan
konsistensi tiga arah antara dokumen sumber data, data yang benar-benar
dimuat, dan kode yang mengasumsikan bentuknya.
"""

from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

EXPECTED_SHAPE = (768, 9)

EXPECTED_CLASS_COUNTS = {0: 500, 1: 268}

# Rata-rata dan standar deviasi asli sebagaimana dilaporkan pada dokumen
# sumber data (UCI Machine Learning Repository / Kaggle uciml).
REFERENCE_STATISTICS = {
    "Pregnancies": (3.8, 3.4),
    "Glucose": (120.9, 32.0),
    "BloodPressure": (69.1, 19.4),
    "SkinThickness": (20.5, 16.0),
    "Insulin": (79.8, 115.2),
    "BMI": (32.0, 7.9),
    "DiabetesPedigreeFunction": (0.5, 0.3),
    "Age": (33.2, 11.8),
}


def load_data(csv_path):
    """Muat dataset mentah dan verifikasi bentuk serta nama kolomnya.

    Melempar ValueError atau FileNotFoundError secara eksplisit apabila
    asumsi tidak terpenuhi, alih-alih membiarkan pipeline berjalan dengan
    data yang diam-diam tidak sesuai.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Berkas dataset tidak ditemukan: {csv_path}")

    df = pd.read_csv(csv_path)

    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Kolom berikut tidak ditemukan pada berkas dataset: {missing_columns}"
        )
    df = df[EXPECTED_COLUMNS]

    if df.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Bentuk data tidak sesuai asumsi pipeline. "
            f"Diharapkan {EXPECTED_SHAPE}, didapat {df.shape}."
        )

    return df


def verify_reference_statistics(df, tolerance=0.15):
    """Bandingkan rata-rata dan standar deviasi tiap kolom dengan nilai
    acuan pada dokumen sumber data.

    Mengembalikan daftar selisih yang melebihi toleransi. Daftar kosong
    berarti data yang dimuat konsisten dengan dokumen sumber.
    """
    selisih = []
    for col, (ref_mean, ref_std) in REFERENCE_STATISTICS.items():
        actual_mean = round(df[col].mean(), 1)
        actual_std = round(df[col].std(), 1)
        if abs(actual_mean - ref_mean) > tolerance or abs(actual_std - ref_std) > tolerance:
            selisih.append(
                {
                    "kolom": col,
                    "rata_rata_dokumen": ref_mean,
                    "rata_rata_aktual": actual_mean,
                    "std_dokumen": ref_std,
                    "std_aktual": actual_std,
                }
            )
    return selisih


def verify_class_distribution(df, target_col="Outcome"):
    """Bandingkan distribusi kelas aktual dengan yang dilaporkan pada
    dokumen sumber data (500 negatif, 268 positif)."""
    actual_counts = df[target_col].value_counts().to_dict()
    cocok = actual_counts == EXPECTED_CLASS_COUNTS
    return {
        "cocok_dengan_dokumen": cocok,
        "distribusi_dokumen": EXPECTED_CLASS_COUNTS,
        "distribusi_aktual": actual_counts,
    }
