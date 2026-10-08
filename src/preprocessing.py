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


# Ambang batas kadar Glucose (mg/dL) untuk label risiko 3 kelas, sesuai
# arahan dosen pembimbing: Normal (<140), Pre-Diabetic (140-200 inklusif),
# Diabetic (>200).
GLUCOSE_NORMAL_MAX = 140
GLUCOSE_DIABETIC_MIN = 200
GLUCOSE_RISK_LABELS = ["Normal", "Pre-Diabetic", "Diabetic"]


def add_glucose_risk_label(df, glucose_col="Glucose", label_col="GlucoseRisk"):
    """Turunkan label risiko 3 kelas dari kadar Glucose.

    - Normal       : Glucose < 140
    - Pre-Diabetic : 140 <= Glucose <= 200
    - Diabetic     : Glucose > 200

    PENTING: fungsi ini harus dipanggil SETELAH nilai 0 pada kolom Glucose
    ditangani (mark_missing + impute_missing_by_class), tidak boleh pada
    data mentah. Nilai 0 pada Glucose secara medis adalah data hilang,
    bukan kadar glukosa sehat -- jika label dihitung langsung dari Glucose
    mentah, baris dengan Glucose=0 akan salah terklasifikasi sebagai
    "Normal" alih-alih ditandai sebagai data yang tidak diketahui.
    Konsekuensinya, imputasi Glucose sendiri tidak boleh dikelompokkan
    berdasarkan GlucoseRisk (karena GlucoseRisk baru bisa dihitung setelah
    Glucose bersih) -- gunakan kelas Outcome yang sudah ada sebagai
    variabel pengelompokan imputasi, sebagaimana dilakukan
    `impute_missing_by_class` di atas.
    """
    df = df.copy()
    glucose = df[glucose_col]
    kondisi = [
        glucose < GLUCOSE_NORMAL_MAX,
        (glucose >= GLUCOSE_NORMAL_MAX) & (glucose <= GLUCOSE_DIABETIC_MIN),
        glucose > GLUCOSE_DIABETIC_MIN,
    ]
    df[label_col] = np.select(kondisi, GLUCOSE_RISK_LABELS, default="Unknown")
    return df


# --- HOMA-IR proxy dan label InsulinResistance (target klasifikasi utama) ---
#
# HOMA-IR baku (Matthews et al., 1985) = (insulin puasa [uU/mL] x glukosa
# puasa [mg/dL]) / 405, dan ambang batas klinisnya (<1.9 normal, 1.9-2.9
# resistensi dini, >2.9 resistensi signifikan) dikalibrasi untuk nilai
# PUASA. Kolom Glucose dan Insulin pada dataset Pima ini BUKAN nilai
# puasa -- keduanya adalah pengukuran 2 JAM SETELAH tes toleransi glukosa
# oral (OGTT), sebagaimana didokumentasikan pada sumber dataset. Rumus
# HOMA-IR baku yang diterapkan langsung pada nilai post-load ini terbukti
# tidak valid: 767 dari 768 pasien pada dataset ini jatuh ke kategori
# "resistensi signifikan" memakai ambang batas klinis baku, karena nilai
# insulin dan glukosa post-load jauh lebih tinggi daripada nilai puasa
# yang menjadi dasar kalibrasi ambang batas tersebut.
#
# Sebagai gantinya, rumus yang sama dipakai sebagai PROXY resistensi
# insulin (bukan diagnosis HOMA-IR klinis), dan kelasnya ditentukan dari
# TERTIL distribusi proxy ini pada dataset ini sendiri -- bukan dari
# ambang batas klinis yang tidak berlaku. Ini menghasilkan 3 kelas yang
# seimbang dan tetap berkorelasi kuat dengan label Outcome lama.
INSULIN_RESISTANCE_LABELS = ["Rendah", "Sedang", "Tinggi"]


def compute_homa_ir_proxy(df, glucose_col="Glucose", insulin_col="Insulin", out_col="HOMA_IR_proxy"):
    """Hitung proxy resistensi insulin = (Insulin x Glucose) / 405.

    PENTING: harus dipanggil SETELAH Glucose dan Insulin dibersihkan
    (mark_missing + impute_missing_by_class) -- nilai 0 mentah pada kedua
    kolom ini akan membuat proxy bernilai 0, yang salah mengklasifikasikan
    baris tersebut sebagai resistensi insulin paling rendah alih-alih data
    yang tidak diketahui.
    """
    df = df.copy()
    df[out_col] = (df[insulin_col] * df[glucose_col]) / 405
    return df


def compute_tertile_thresholds(series):
    """Hitung ambang batas tertil (33%, 67%) dari sebuah series -- dipakai
    untuk menurunkan ambang batas InsulinResistance secara data-driven
    dari proxy HOMA-IR, bukan dari cutoff klinis yang tidak berlaku pada
    data post-load ini."""
    t1, t2 = series.quantile([1 / 3, 2 / 3])
    return float(t1), float(t2)


def add_insulin_resistance_label(df, t1, t2, homa_col="HOMA_IR_proxy", label_col="InsulinResistance"):
    """Turunkan label InsulinResistance 3 kelas dari proxy HOMA-IR memakai
    ambang batas tertil (t1, t2) yang sudah dihitung sebelumnya dengan
    compute_tertile_thresholds -- diteruskan secara eksplisit (bukan
    dihitung ulang di sini) agar kedua skenario perbandingan (fitur
    mentah vs fitur bersih) selalu memakai label yang identik, sama
    seperti pola pada add_glucose_risk_label."""
    df = df.copy()
    homa = df[homa_col]
    kondisi = [homa < t1, (homa >= t1) & (homa <= t2), homa > t2]
    df[label_col] = np.select(kondisi, INSULIN_RESISTANCE_LABELS, default="Unknown")
    return df
