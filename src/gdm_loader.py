"""Pemuatan dan verifikasi dataset Gestational Diabetes Mellitus (GDM).

Sumber: "Gestational Diabetic Dat Set" (Sumathi Santhosh, Kaggle/Figshare),
3.525 pasien hamil, 17 atribut, label diagnosis klinis asli
`Class Label(GDM /Non GDM)` (0 = Non-GDM, 1 = GDM). Dipakai bersama
Pima Indians Diabetes Database pada src/merge.py untuk membangun target
klasifikasi 3 kelas berbasis diagnosis asli: Normal / Tipe 2 / Gestasional.
"""

import pandas as pd

SHEET_NAME = "GDM-Final2022"
EXPECTED_ROWS = 3525
EXPECTED_COLUMNS = 17

CLASS_COL = "Class Label(GDM /Non GDM)"

# Kolom GDM yang punya padanan konsep langsung di Pima, dan nama kolom
# Pima yang dipakai setelah harmonisasi. Hanya lima inilah yang dipakai
# bersama pada model gabungan -- lihat src/merge.py untuk alasannya.
GDM_TO_PIMA_COLUMNS = {
    "Age": "Age",
    "No of Pregnancy": "Pregnancies",
    "BMI": "BMI",
    "Dia BP": "BloodPressure",
    "OGTT": "Glucose",
}


def load_gdm_data(path):
    """Muat dataset GDM dari file Excel aslinya, verifikasi bentuknya."""
    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    if df.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValueError(
            f"Bentuk data GDM tidak sesuai dugaan: {df.shape}, "
            f"diharapkan ({EXPECTED_ROWS}, {EXPECTED_COLUMNS})"
        )
    return df


def count_missing_gdm(df):
    """Hitung nilai NaN asli per kolom pada dataset GDM.

    Berbeda dari Pima, dataset GDM sudah memakai NaN eksplisit untuk
    data hilang (bukan nilai 0 sebagai sentinel) -- diverifikasi tidak
    ada nilai 0 pada BMI maupun OGTT, sehingga tidak perlu langkah
    'tandai nilai 0' seperti pada Pima."""
    kolom_relevan = list(GDM_TO_PIMA_COLUMNS.keys())
    return df[kolom_relevan].isna().sum().to_dict()


def impute_gdm_missing_by_class(df, class_col=CLASS_COL):
    """Imputasi median per kelas Class Label untuk kolom yang dipakai
    bersama dengan Pima (Age, No of Pregnancy, BMI, Dia BP, OGTT).
    Hanya BMI dan OGTT yang punya nilai hilang pada dataset ini."""
    df = df.copy()
    for col in GDM_TO_PIMA_COLUMNS:
        if df[col].isna().any():
            median_per_kelas = df.groupby(class_col)[col].transform("median")
            df[col] = df[col].fillna(median_per_kelas)
    return df


def harmonize_gdm(df):
    """Ganti nama kolom GDM menjadi nama kolom Pima yang berpadanan,
    dan kembalikan hanya kolom yang dipakai bersama plus Class Label."""
    kolom_dipakai = list(GDM_TO_PIMA_COLUMNS.keys()) + [CLASS_COL]
    out = df[kolom_dipakai].rename(columns=GDM_TO_PIMA_COLUMNS)
    return out
