"""Penggabungan Pima Indians Diabetes Database dengan dataset Gestational
Diabetes Mellitus (GDM) menjadi satu target klasifikasi 3 kelas berbasis
DIAGNOSIS ASLI dari kedua studi -- bukan ambang batas yang dihitung sendiri
seperti pada revisi-revisi sebelumnya (GlucoseRisk, InsulinResistance).

Metodologi mengikuti pendekatan yang dipakai Rasheed dkk. (Scientific
Reports, 2026, "Pediatric diabetes prediction using machine learning"):
menggabungkan beberapa dataset diabetes dengan skema atribut berbeda
lewat UNION kolom yang benar-benar terukur bersama, dengan label
diagnosis dipetakan ke kategori yang konsisten lintas sumber.

Pada proyek ini, union kolom dibatasi pada LIMA atribut yang punya
padanan konsep langsung di kedua dataset (Age, Pregnancies, BMI,
BloodPressure, Glucose) -- bukan union SELURUH kolom kedua dataset
dengan imputasi berat ala MICE untuk kolom yang strukturnya hilang
total di satu sumber. Pilihan ini mengorbankan atribut unik Pima
(SkinThickness, Insulin, DiabetesPedigreeFunction) dan atribut unik GDM
(HDL, Hemoglobin, Family History, dll), tetapi menghasilkan model yang
lebih sederhana untuk dijelaskan dan lebih mudah diaudit asumsinya --
lebih sesuai untuk proyek tugas kuliah dibanding union penuh dengan
MICE. Lihat notebooks/02_Preprocessing.ipynb untuk pembahasan trade-off
ini secara eksplisit.

KETERBATASAN PALING PENTING (dibahas rinci di notebooks/01_EDA.ipynb
dan notebooks/04_Evaluation.ipynb): pasien Pima TIDAK SELALU sedang
hamil saat diperiksa, sedangkan SELURUH pasien GDM SELALU sedang hamil
(by definition). Kehamilan secara fisiologis mengubah resistensi
insulin, BMI efektif, dan tekanan darah -- sehingga sebagian pola yang
dipelajari model berpotensi mencerminkan PERBEDAAN SUMBER DATA /
STATUS KEHAMILAN, bukan murni perbedaan tipe diabetes. Ini dibuktikan
secara empiris (rata-rata BMI kelas "Normal" berbeda jauh antar sumber)
pada notebook EDA.
"""

import numpy as np

SHARED_FEATURES = ["Age", "Pregnancies", "BMI", "BloodPressure", "Glucose"]
DIAGNOSIS_LABELS = ["Normal", "Tipe 2", "Gestasional"]


def build_pima_portion(df_pima_clean, outcome_col="Outcome"):
    """Ambil lima fitur bersama dari Pima yang sudah dibersihkan, dan
    petakan Outcome ke DiagnosisType: 0 -> Normal, 1 -> Tipe 2."""
    out = df_pima_clean[SHARED_FEATURES].copy()
    out["DiagnosisType"] = np.where(df_pima_clean[outcome_col] == 1, "Tipe 2", "Normal")
    out["SumberData"] = "Pima"
    return out


def build_gdm_portion(df_gdm_harmonized, class_col="Class Label(GDM /Non GDM)"):
    """Ambil lima fitur bersama dari GDM yang sudah diharmonisasi nama
    kolomnya dan dibersihkan, lalu petakan Class Label ke DiagnosisType:
    0 -> Normal, 1 -> Gestasional."""
    out = df_gdm_harmonized[SHARED_FEATURES].copy()
    out["DiagnosisType"] = np.where(df_gdm_harmonized[class_col] == 1, "Gestasional", "Normal")
    out["SumberData"] = "GDM"
    return out


def merge_datasets(df_pima_portion, df_gdm_portion):
    """Gabungkan (concat baris) kedua porsi yang sudah diharmonisasi.
    Kolom sudah identik di kedua sisi, sehingga ini murni penggabungan
    baris, bukan join berbasis kunci (tidak ada pasien yang sama di
    kedua studi)."""
    import pandas as pd

    gabungan = pd.concat([df_pima_portion, df_gdm_portion], ignore_index=True)
    return gabungan
