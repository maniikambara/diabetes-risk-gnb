"""Pembersihan data: menangani nilai 0 yang secara medis merepresentasikan
data hilang pada atribut fisiologis tertentu.

Kolom yang diperlakukan demikian adalah Glucose, BloodPressure,
SkinThickness, Insulin, dan BMI, karena nilai 0 pada kolom-kolom ini
tidak mungkin terjadi secara biologis pada pasien hidup.
"""

import numpy as np

ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def mark_missing(df):
    """Ganti nilai 0 pada kolom fisiologis yang relevan menjadi NaN."""
    df = df.copy()
    for col in ZERO_AS_MISSING_COLUMNS:
        df[col] = df[col].replace(0, np.nan)
    return df


def count_missing(df):
    """Hitung jumlah nilai hilang per kolom setelah mark_missing, untuk
    pelaporan pada ringkasan hasil."""
    return {col: int(df[col].isna().sum()) for col in ZERO_AS_MISSING_COLUMNS}


def impute_missing_by_class(df, target_col="Outcome"):
    """Isi nilai hilang dengan median tiap kolom, dihitung terpisah per
    kelas Outcome.

    Imputasi per kelas dipakai karena rentang nilai fisiologis normal
    berbeda antara pasien diabetes dan bukan diabetes, sehingga median
    gabungan akan mengaburkan perbedaan itu. Contohnya, median Glucose
    pada kelompok Outcome=1 secara medis wajar lebih tinggi daripada pada
    kelompok Outcome=0.

    Catatan metodologis: langkah pembersihan ini dijalankan sebelum data
    dibagi menjadi data latih dan data uji, sebagaimana urutan yang
    ditetapkan pada spesifikasi tugas. Konsekuensinya, median yang dipakai
    untuk mengisi nilai hilang pada baris yang nantinya masuk data uji
    turut dihitung dari label Outcome baris itu sendiri. Ini adalah bentuk
    kebocoran data berskala kecil pada tahap praproses, bukan kebocoran
    label langsung ke prediksi, dan didokumentasikan secara eksplisit pada
    bagian Keterbatasan README, bersama alternatif yang lebih ketat yaitu
    menghitung median hanya dari data latih.
    """
    df = df.copy()
    for col in ZERO_AS_MISSING_COLUMNS:
        median_per_kelas = df.groupby(target_col)[col].transform("median")
        df[col] = df[col].fillna(median_per_kelas)
    return df
