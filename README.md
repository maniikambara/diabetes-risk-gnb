# Klasifikasi Risiko Diabetes dengan Gaussian Naive Bayes

## 1. Ide (Topik)

Klasifikasi Risiko Diabetes Menggunakan Naive Bayes Berdasarkan Data
Diagnostik Pasien.

## 2. Deskripsi Ide

Proyek ini membangun model klasifikasi **Gaussian Naive Bayes** untuk
memprediksi kategori diagnosis diabetes seorang pasien.

> **Riwayat revisi setelah presentasi dosen pembimbing:**
> 1. Target awal `Outcome` biner -> 3 kelas `GlucoseRisk` dari ambang
>    batas klinis `Glucose` saja.
> 2. Dicari faktor di luar glukosa -> target diganti `InsulinResistance`,
>    proxy resistensi insulin dari `Glucose` x `Insulin` dengan ambang
>    batas tertil data-driven.
> 3. **Revisi ini:** dicari dataset nyata (bukan sintetis, bukan ambang
>    batas buatan sendiri) yang pasiennya benar-benar didiagnosis
>    dengan tipe diabetes berbeda. Pima digabung dengan **Gestational
>    Diabetes Mellitus (GDM) Data Set** (3.525 pasien hamil nyata, data
>    rumah sakit di India) menjadi target **`DiagnosisType`** 3 kelas
>    berbasis **diagnosis klinis asli kedua studi**:
>
> | Kelas | Sumber | Dasar label |
> | --- | --- | --- |
> | Normal | Pima + GDM | `Outcome=0` (Pima) atau `Non-GDM` (GDM) |
> | Tipe 2 | Pima | `Outcome=1` -- diagnosis T2D asli studi Pima |
> | Gestasional | GDM | `Class=GDM` -- diagnosis GDM asli studi tsb. |
>
> Metodologi penggabungan mengikuti pendekatan Rasheed dkk. (Scientific
> Reports, 2026, "Pediatric diabetes prediction using machine
> learning"): union kolom yang benar-benar terukur bersama di kedua
> dataset.

## 3. Analisis Permasalahan

### 3.1 Mengapa Union Kolom Dibatasi pada Lima Atribut

Pima (8 atribut) dan GDM (17 atribut) hanya punya **lima** atribut
dengan padanan konsep langsung: `Age`, jumlah kehamilan
(`Pregnancies`), `BMI`, tekanan darah diastolik (`BloodPressure`), dan
kadar glukosa OGTT (`Glucose`). Union penuh seluruh kolom kedua
dataset (seperti pada paper acuan, memakai MICE untuk kolom yang
strukturnya hilang total di satu sumber) dipertimbangkan tapi **tidak
dipakai** -- terlalu berat untuk diaudit pada skala tugas kuliah.
Atribut unik Pima (`SkinThickness`, `Insulin`,
`DiabetesPedigreeFunction`) dan unik GDM (`HDL`, `Hemoglobin`,
`Family History`, dll.) karena itu tidak dipakai pada model gabungan.

### 3.2 Keterbatasan Paling Penting: Status Kehamilan sebagai Perancu

Pasien Pima **tidak selalu sedang hamil**, sedangkan **seluruh pasien
GDM, menurut definisi, sedang hamil**. Dibuktikan secara empiris
(lihat bagian Hasil): BMI rata-rata kelas "Normal" berbeda 7 poin
antar sumber (Pima 30,9 vs GDM 23,7), dan model lebih mudah menebak
**sumber data** (93,8% akurasi) daripada **DiagnosisType** itu sendiri
(85,9%) -- bukti konkret bahwa sebagian performa model mencerminkan
perbedaan studi/status kehamilan, bukan murni tipe diabetes.

## 4. Sumber Data

**Dataset 1 - Pima Indians Diabetes Database.** 768 pasien perempuan
Pima Indian usia >= 21 tahun, 8 atribut, label `Outcome`.
<https://www.kaggle.com/datasets/jamaltariqcheema/pima-indians-diabetes-dataset>

**Dataset 2 - Gestational Diabetes Mellitus (GDM) Data Set.** 3.525
pasien hamil (data rumah sakit di India), 17 atribut, label
`Class Label(GDM /Non GDM)`. Sumber: Sumathi Santhosh, Kaggle --
<https://www.kaggle.com/datasets/sumathisanthosh/gestational-diabetes-mellitus-gdm-data-set>.
Berkas disertakan pada `data/gdm_dataset.xlsx`.

Dataset tidak memiliki nilai kosong secara eksplisit pada Pima (nilai
0 dipakai sebagai sentinel pada lima kolom fisiologis), sementara GDM
sudah memakai `NaN` eksplisit:

| Kolom | Pima (nilai 0) | GDM (NaN) |
| --- | --- | --- |
| Glucose / OGTT | 5 (0,7%) | 513 (14,6%) |
| BloodPressure / Dia BP | 35 (4,6%) | 0 |
| BMI | 11 (1,4%) | 1.081 (30,7%) |
| Age | - | 0 |
| Pregnancies | - | 0 |

## 5. Metode Komputasi Soft Computing

Naive Bayes (Gaussian Naive Bayes). Tahapan: probabilitas prior tiap
kelas dari data latih, estimasi distribusi normal (mean, variansi)
tiap atribut per kelas, penerapan Teorema Bayes untuk probabilitas
posterior, evaluasi dengan akurasi/precision/recall/F1/confusion
matrix multi-kelas. Dibuktikan manual pada
`notebooks/03_GaussianNB.ipynb` bahwa posterior hasil hitung tangan
identik dengan `predict_proba()` scikit-learn.

## Struktur Proyek

```
diabetes-risk-gnb/
├── README.md
├── requirements.txt
├── main.py                       jalur CLI: seluruh pipeline dalam satu perintah
├── app_streamlit.py               aplikasi deteksi kategori diagnosis langsung
├── data/
│   ├── diabetes.csv               dataset Pima mentah
│   ├── gdm_dataset.xlsx           dataset GDM mentah
│   └── processed/
│       └── merged_diagnosis.csv   Pima + GDM tergabung, 5 fitur + DiagnosisType
├── notebooks/
│   ├── 01_EDA.ipynb                eksplorasi kedua dataset, pembuktian perancu sumber data
│   ├── 02_Preprocessing.ipynb      pembersihan kedua dataset, penggabungan
│   ├── 03_GaussianNB.ipynb         pelatihan model, pembuktian manual Teorema Bayes
│   └── 04_Evaluation.ipynb         evaluasi, uji langsung perancu sumber data
├── src/
│   ├── __init__.py
│   ├── data_loader.py             pemuatan dan verifikasi data Pima
│   ├── preprocessing.py           penanganan nilai 0 Pima (+ sisa fungsi GlucoseRisk/InsulinResistance dari revisi sebelumnya, tidak dipakai lagi)
│   ├── gdm_loader.py               pemuatan, verifikasi, imputasi, harmonisasi data GDM
│   ├── merge.py                    penggabungan Pima + GDM jadi target DiagnosisType
│   ├── pipeline.py                split data dan pelatihan GaussianNB (generik jumlah kelas)
│   └── evaluate.py                confusion matrix dan metrik klasifikasi multi-kelas (generik)
├── models/
│   ├── gnb_diagnosis.joblib        model utama (dipakai app_streamlit.py)
│   ├── feature_columns.json        urutan kolom fitur saat pelatihan
│   └── imputation_medians.json     median Pima mentah untuk input tidak diketahui
└── outputs/
    ├── metrics.json
    ├── confusion_matrices.png
    ├── eda_source_label_distributions.png
    ├── eda_correlation_heatmap.png
    └── preprocessing_merged_feature_distributions.png
```

## Instalasi

```
pip install -r requirements.txt
```

## Penggunaan

**Jalur notebook (direkomendasikan):** `jupyter notebook notebooks/`,
jalankan 01 -> 04 berurutan.

**Jalur CLI:** `python main.py`

**Aplikasi deteksi langsung:** `streamlit run app_streamlit.py` --
form isian lima indikator, prediksi `DiagnosisType` beserta
probabilitas posterior tiap kelas, memakai `models/gnb_diagnosis.joblib`.
Menampilkan peringatan eksplisit soal keterbatasan perancu sumber
data. Proyek sudah menyertakan artefak terlatih sehingga aplikasi
dapat langsung dijalankan.

## Konfigurasi

| Parameter | Nilai | Lokasi |
| --- | --- | --- |
| `TEST_SIZE` / `RANDOM_STATE` | 0.2 / 42 | `src/pipeline.py` |
| Lima fitur bersama (union) | Age, Pregnancies, BMI, BloodPressure, Glucose | `src/merge.py::SHARED_FEATURES` |
| Pemetaan label diagnosis | Pima Outcome + GDM Class -> DiagnosisType | `src/merge.py` |
| Pemetaan kolom GDM -> Pima | Age, No of Pregnancy, BMI, Dia BP, OGTT | `src/gdm_loader.py::GDM_TO_PIMA_COLUMNS` |

## Hasil

Hasil berikut diperoleh dari `python main.py` / `notebooks/04_Evaluation.ipynb`,
`random_state=42`.

**Distribusi `DiagnosisType`** (4.293 baris gabungan): Normal 2.653
(61,8%), Gestasional 1.372 (32,0%), Tipe 2 268 (6,2%).

| Metrik | Nilai |
| --- | --- |
| Akurasi | 0,8591 |
| Precision makro | 0,7616 |
| Recall makro | 0,7936 |
| F1 makro | 0,7708 |

Rincian per kelas:

| Kelas | Precision | Recall | F1 | Support (data uji) |
| --- | --- | --- | --- | --- |
| Normal | 0,8824 | 0,9040 | 0,8930 | 531 |
| Tipe 2 | 0,4737 | 0,6667 | 0,5538 | 54 |
| Gestasional | 0,9289 | 0,8102 | 0,8655 | 274 |

**Pembuktian perancu sumber data (bagian terpenting hasil ini):**
melatih classifier terpisah untuk menebak `SumberData` (Pima vs GDM,
bukan `DiagnosisType`) dari kelima fitur yang sama menghasilkan
**akurasi 0,9383** -- lebih tinggi daripada akurasi model pada tugas
sebenarnya (0,8591). Lima fitur ini lebih mudah dipakai mengenali studi
asal pasien daripada tipe diabetesnya.

## Keterbatasan

**Perancu sumber data / status kehamilan** -- dibuktikan kuantitatif
di atas, keterbatasan paling signifikan pada revisi ini.

**Union kolom dibatasi lima fitur** -- atribut unik tiap dataset tidak
dipakai (lihat bagian Analisis Permasalahan 3.1).

**Ketidakseimbangan kelas berat** -- `Tipe 2` hanya 6,2% data,
precision/recall-nya jauh lebih rendah dibanding dua kelas lain.

**Kebocoran data berskala kecil pada tahap imputasi** -- median per
kelas asli (bukan `DiagnosisType`) dihitung sebelum data dibagi
latih/uji, untuk kedua dataset.

**Perbedaan definisi `Pregnancies`** -- Pima: total kehamilan seumur
hidup (termasuk 0). GDM: kehamilan ke berapa yang sedang dijalani saat
ini (minimal 1, karena pasien sedang hamil). Digabung sebagai satu
kolom meski operasionalisasinya berbeda.

**Asumsi independensi antar atribut tidak sepenuhnya realistis**
(`Age`-`Pregnancies` 0,54 pada Pima).

**Belum ada validasi silang** -- hasil dari satu kali split dengan
`random_state=42`.

## Riwayat Skema Sebelumnya

Kode untuk skema `GlucoseRisk` dan `InsulinResistance` (revisi 1 dan 2)
masih ada pada `src/preprocessing.py` untuk referensi/transparansi
riwayat proyek, tetapi tidak lagi dipakai oleh `main.py`,
`app_streamlit.py`, atau notebook manapun pada revisi ini.
