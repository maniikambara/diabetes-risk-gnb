"""Titik masuk utama proyek klasifikasi risiko diabetes.

Menjalankan pipeline Gaussian Naive Bayes dua kali pada data yang sama,
sekali tanpa tahap pembersihan data dan sekali dengan tahap pembersihan
data, lalu membandingkan hasil keduanya. Perbandingan ini menunjukkan
pengaruh kualitas data terhadap performa model.

Jalankan dengan:
    python main.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import (
    load_data,
    verify_reference_statistics,
    verify_class_distribution,
)
from src.preprocessing import mark_missing, impute_missing_by_class, count_missing
from src.pipeline import train_and_predict
from src.evaluate import evaluate_predictions

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

KELAS_LABEL = ["Tidak Diabetes", "Diabetes"]


def hitung_korelasi_atribut(df):
    """Hitung korelasi Pearson antar atribut prediktor.

    Dipakai untuk mendukung pembahasan tentang asumsi independensi antar
    atribut pada Naive Bayes, yang secara medis tidak sepenuhnya
    realistis untuk pasangan seperti Glucose dan BMI.
    """
    korelasi = df.drop(columns=["Outcome"]).corr(method="pearson")
    pasangan_tertinggi = (
        korelasi.where(~korelasi.abs().eq(1.0))
        .unstack()
        .dropna()
        .abs()
        .sort_values(ascending=False)
    )
    lima_tertinggi = []
    seen = set()
    for (a, b), nilai in pasangan_tertinggi.items():
        pasangan = tuple(sorted([a, b]))
        if pasangan in seen:
            continue
        seen.add(pasangan)
        lima_tertinggi.append(
            {"atribut_1": pasangan[0], "atribut_2": pasangan[1], "korelasi": round(korelasi.loc[a, b], 3)}
        )
        if len(lima_tertinggi) == 5:
            break
    return lima_tertinggi


def simpan_plot_confusion_matrix(cm_tanpa, cm_dengan, path_output):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, cm, judul in zip(axes, [cm_tanpa, cm_dengan], ["Tanpa Cleaning", "Dengan Cleaning"]):
        ax.imshow(cm, cmap="Blues")
        ax.set_title(judul)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(KELAS_LABEL, rotation=15)
        ax.set_yticklabels(KELAS_LABEL)
        nilai_maks = max(max(row) for row in cm)
        for i in range(2):
            for j in range(2):
                warna = "white" if cm[i][j] > nilai_maks / 2 else "black"
                ax.text(j, i, cm[i][j], ha="center", va="center", color=warna)
    fig.tight_layout()
    fig.savefig(path_output, dpi=150)
    plt.close(fig)


def jalankan():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df_mentah = load_data(DATA_PATH)
    selisih_statistik = verify_reference_statistics(df_mentah)
    verifikasi_kelas = verify_class_distribution(df_mentah)

    # Skenario 1: tanpa pembersihan data, nilai 0 dibiarkan apa adanya.
    hasil_tanpa = train_and_predict(df_mentah)
    metrik_tanpa = evaluate_predictions(hasil_tanpa["y_test"], hasil_tanpa["y_pred"])

    # Skenario 2: dengan pembersihan data (0 -> NaN -> imputasi median per kelas).
    df_hilang_ditandai = mark_missing(df_mentah)
    jumlah_nilai_hilang = count_missing(df_hilang_ditandai)
    df_bersih = impute_missing_by_class(df_hilang_ditandai)

    hasil_dengan = train_and_predict(df_bersih)
    metrik_dengan = evaluate_predictions(hasil_dengan["y_test"], hasil_dengan["y_pred"])

    korelasi_atribut = hitung_korelasi_atribut(df_bersih)

    ringkasan = {
        "verifikasi_dataset": {
            "bentuk_data": list(df_mentah.shape),
            "selisih_statistik_vs_dokumen": selisih_statistik,
            "distribusi_kelas": verifikasi_kelas,
        },
        "jumlah_nilai_hilang_per_kolom": jumlah_nilai_hilang,
        "tanpa_cleaning": metrik_tanpa,
        "dengan_cleaning": metrik_dengan,
        "korelasi_atribut_tertinggi": korelasi_atribut,
    }

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(ringkasan, f, indent=2, ensure_ascii=False)

    simpan_plot_confusion_matrix(
        metrik_tanpa["confusion_matrix"],
        metrik_dengan["confusion_matrix"],
        OUTPUT_DIR / "confusion_matrices.png",
    )

    cetak_ringkasan(ringkasan)


def cetak_ringkasan(ringkasan):
    print("Bentuk data:", ringkasan["verifikasi_dataset"]["bentuk_data"])
    print(
        "Selisih statistik vs dokumen:",
        ringkasan["verifikasi_dataset"]["selisih_statistik_vs_dokumen"] or "tidak ada",
    )
    print("Distribusi kelas:", ringkasan["verifikasi_dataset"]["distribusi_kelas"])
    print("Jumlah nilai hilang per kolom:", ringkasan["jumlah_nilai_hilang_per_kolom"])
    print()
    print("Tanpa cleaning :", ringkasan["tanpa_cleaning"])
    print("Dengan cleaning:", ringkasan["dengan_cleaning"])
    print()
    print("Korelasi atribut tertinggi:")
    for pasangan in ringkasan["korelasi_atribut_tertinggi"]:
        print(f"  {pasangan['atribut_1']} - {pasangan['atribut_2']}: {pasangan['korelasi']}")


if __name__ == "__main__":
    jalankan()
