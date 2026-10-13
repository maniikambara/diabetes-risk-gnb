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
>
> 4. **Perbaikan imputasi:** nilai hilang semula diisi median per kelas
>    diagnosis, sebelum data dibagi latih/uji. Seluruh nilai hilang GDM
>    ternyata ada pada kelas Non-GDM, sehingga nilai pengisi itu
>    membocorkan label. Imputasi kini dilakukan setelah split, memakai
>    median data latih tanpa label; akurasi turun dari 0,8591 ke 0,8510
>    (lihat bagian Hasil).

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
**sumber data** (93,6% akurasi) daripada **DiagnosisType** itu sendiri
(85,1%) -- bukti konkret bahwa sebagian performa model mencerminkan
perbedaan studi/status kehamilan, bukan murni tipe diabetes. Akibatnya,
untuk pasien asal Pima model nyaris tidak lebih baik dari selalu
menebak `Normal` (68,8% vs 65,6%).

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

**Seluruh nilai hilang GDM ada pada kelas Non-GDM** (1.081 `BMI` dan
513 `OGTT` dari 2.153 pasien Non-GDM; nol dari 1.372 pasien GDM).
Karena ada-tidaknya nilai hilang berkaitan dengan label, nilai hilang
tidak diisi dengan median per kelas diagnosis. Data dibagi latih/uji
lebih dulu, lalu nilai hilang diisi median tiap fitur dari **data
latih**, tanpa melihat label (`BMI` 28,4; `BloodPressure` 80;
`Glucose` 154).

## 5. Metode Komputasi Soft Computing

Naive Bayes (Gaussian Naive Bayes). Tahapan: pembagian data latih/uji
dan imputasi median data latih, probabilitas prior tiap
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
├── app_streamlit.py               aplikasi deteksi kategori diagnosis langsung (satu-satunya berkas .py)
├── data/
│   ├── diabetes.csv               dataset Pima mentah
│   ├── gdm_dataset.xlsx           dataset GDM mentah
│   └── processed/
│       └── merged_diagnosis.csv   Pima + GDM tergabung, 5 fitur + DiagnosisType + SumberData (nilai hilang belum diisi)
├── notebooks/
│   ├── 01_EDA.ipynb                eksplorasi kedua dataset, pembuktian perancu sumber data
│   ├── 02_Preprocessing.ipynb      penandaan nilai hilang kedua dataset, penggabungan
│   ├── 03_GaussianNB.ipynb         split, imputasi median data latih, pelatihan model, pembuktian manual Teorema Bayes
│   └── 04_Evaluation.ipynb         evaluasi, uji perancu sumber data, akurasi per sumber, dampak cara imputasi
├── models/
│   ├── gnb_diagnosis.joblib        model utama (dipakai app_streamlit.py)
│   ├── feature_columns.json        urutan kolom fitur saat pelatihan
│   └── imputation_medians.json     median data latih kelima fitur, dipakai mengisi nilai hilang
└── outputs/
    ├── metrics.json
    ├── confusion_matrices.png
    ├── eda_source_label_distributions.png
    ├── eda_correlation_heatmap.png
    └── preprocessing_merged_feature_distributions.png
```

Seluruh logika (pemuatan data, penandaan nilai hilang, penggabungan,
imputasi, pelatihan, evaluasi) ada di dalam keempat notebook; tidak ada
modul Python terpisah. Tiap notebook berdiri sendiri dan terhubung
hanya lewat berkas yang ditulisnya: `02` menulis `merged_diagnosis.csv`,
`03` menulis isi `models/`, `04` menulis `metrics.json` dan
`confusion_matrices.png`.

## Instalasi

```
pip install -r requirements.txt
```

## Penggunaan

**Notebook:** `jupyter notebook notebooks/`, jalankan 01 -> 04
berurutan.

**Aplikasi deteksi langsung:** `streamlit run app_streamlit.py` --
form isian lima indikator, prediksi `DiagnosisType` beserta
probabilitas posterior tiap kelas, memakai `models/gnb_diagnosis.joblib`.
Menampilkan peringatan eksplisit soal keterbatasan perancu sumber
data. Proyek sudah menyertakan artefak terlatih sehingga aplikasi
dapat langsung dijalankan.

## Konfigurasi

| Parameter | Nilai | Lokasi |
| --- | --- | --- |
| `TEST_SIZE` / `RANDOM_STATE` | 0.2 / 42 | `03_GaussianNB` dan `04_Evaluation` (harus sama; `04` memeriksanya) |
| Imputasi nilai hilang | median data latih, tanpa label, setelah split | `03_GaussianNB` |
| Lima fitur bersama (union) | Age, Pregnancies, BMI, BloodPressure, Glucose | `02_Preprocessing`, `SHARED_FEATURES` |
| Pemetaan label diagnosis | Pima Outcome + GDM Class -> DiagnosisType | `02_Preprocessing` bagian 4 |
| Pemetaan kolom GDM -> Pima | Age, No of Pregnancy, BMI, Dia BP, OGTT | `02_Preprocessing`, `GDM_TO_PIMA_COLUMNS` |

## Hasil

Hasil berikut diperoleh dari `notebooks/04_Evaluation.ipynb`,
`random_state=42`.

**Distribusi `DiagnosisType`** (4.293 baris gabungan): Normal 2.653
(61,8%), Gestasional 1.372 (32,0%), Tipe 2 268 (6,2%).

| Metrik | Nilai |
| --- | --- |
| Akurasi | 0,8510 |
| Precision makro | 0,7732 |
| Recall makro | 0,7804 |
| F1 makro | 0,7707 |

Rincian per kelas:

| Kelas | Precision | Recall | F1 | Support (data uji) |
| --- | --- | --- | --- | --- |
| Normal | 0,8576 | 0,9190 | 0,8873 | 531 |
| Tipe 2 | 0,5294 | 0,6667 | 0,5902 | 54 |
| Gestasional | 0,9324 | 0,7555 | 0,8347 | 274 |

**Pembuktian perancu sumber data (bagian terpenting hasil ini):**
melatih classifier terpisah untuk menebak `SumberData` (Pima vs GDM,
bukan `DiagnosisType`) dari kelima fitur yang sama menghasilkan
**akurasi 0,9360** -- lebih tinggi daripada akurasi model pada tugas
sebenarnya (0,8510). Lima fitur ini lebih mudah dipakai mengenali studi
asal pasien daripada tipe diabetesnya.

**Akurasi per sumber data.** Baseline = selalu menebak kelas terbanyak
pada sumber itu (`Normal` untuk keduanya):

| Sumber | Data uji | Akurasi model | Baseline |
| --- | --- | --- | --- |
| GDM | 702 | 0,8875 | 0,6097 |
| Pima | 157 | 0,6879 | 0,6561 |

Performa model hampir seluruhnya berasal dari pasien GDM. Untuk pasien
asal Pima selisihnya terhadap baseline hanya sekitar 3 poin (5 dari 157
pasien uji); 29 dari 103 pasien `Normal` asal Pima ditebak `Tipe 2`.
Akurasi keseluruhan karena itu tidak boleh dibaca sebagai kemampuan
model membedakan `Normal` dari `Tipe 2`.

**Dampak perubahan cara imputasi** (split yang sama, dari
`notebooks/04_Evaluation.ipynb`):

| Cara imputasi | Akurasi | F1 makro | Recall Gestasional | Akurasi baris lengkap | Akurasi baris terimputasi |
| --- | --- | --- | --- | --- | --- |
| Median per kelas, sebelum split (lama) | 0,8591 | 0,7708 | 0,8102 | 0,8083 | 0,9957 |
| Median data latih, tanpa label (sekarang) | 0,8510 | 0,7707 | 0,7555 | 0,7971 | 0,9957 |

Cara lama terlalu optimistis karena nilai pengisi per kelas ikut
membantu memisahkan kelas, sehingga tidak dipakai lagi. Yang tidak
hilang dengan perbaikan ini: baris uji hasil imputasi (233 baris, 228
di antaranya `Normal`) tetap ditebak benar 99,6%, jauh di atas baris
yang datanya lengkap (626 baris, 0,7971).

## Keterbatasan

**Perancu sumber data / status kehamilan** -- dibuktikan kuantitatif
di atas, keterbatasan paling signifikan pada revisi ini.

**Model nyaris tidak membantu pasien asal Pima** -- akurasi 0,6879 vs
baseline 0,6561. Konsekuensi langsung perancu di atas: kelas `Normal`
didominasi pasien GDM, sehingga pasien `Normal` asal Pima tampak mirip
`Tipe 2` di mata model.

**Union kolom dibatasi lima fitur** -- atribut unik tiap dataset tidak
dipakai (lihat bagian Analisis Permasalahan 3.1).

**Ketidakseimbangan kelas berat** -- `Tipe 2` hanya 6,2% data,
precision/recall-nya jauh lebih rendah dibanding dua kelas lain.

**Ada-tidaknya nilai hilang berkaitan dengan label** -- seluruh nilai
hilang GDM ada pada kelas Non-GDM. Imputasi kini memakai median data
latih tanpa label, sehingga nilai pengisi tidak lagi membocorkan label,
tetapi baris hasil imputasi tetap jauh lebih mudah ditebak daripada
baris lengkap. Akurasi pada baris lengkap (0,7971) adalah gambaran yang
lebih konservatif daripada akurasi keseluruhan (0,8510).

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
serta jalur CLI `main.py` dan paket `src/` sudah tidak ada pada versi
ini -- proyek kini hanya berisi notebook dan aplikasi Streamlit. Kode
lama tetap bisa dilihat pada riwayat commit git. Cara imputasi lama
(median per kelas, sebelum split) hanya muncul sebagai pembanding pada
`notebooks/04_Evaluation.ipynb`.
