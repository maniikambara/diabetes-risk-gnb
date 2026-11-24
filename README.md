# Klasifikasi Risiko Diabetes dengan Gaussian Naive Bayes

## 1. Ide (Topik)

Klasifikasi Risiko Diabetes Menggunakan Naive Bayes Berdasarkan Data
Diagnostik Pasien.

## 2. Deskripsi Ide

Di proyek ini kami membuat model **Gaussian Naive Bayes** yang menebak
kategori diagnosis diabetes seorang pasien.

> **Riwayat revisi setelah presentasi dosen pembimbing:**
> 1. Awalnya target berupa `Outcome` biner, lalu diubah jadi 3 kelas
>    `GlucoseRisk` yang hanya ditentukan dari ambang klinis `Glucose`.
> 2. Supaya ada faktor selain glukosa, target diganti `InsulinResistance`,
>    yaitu proxy resistensi insulin dari `Glucose` x `Insulin` yang dibagi
>    dengan ambang tertil dari data.
> 3. **Revisi ini:** kami mencari dataset nyata, bukan data sintetis dan
>    bukan ambang buatan sendiri, yang pasiennya memang didiagnosis dengan
>    tipe diabetes berbeda. Hasilnya, Pima digabung dengan **Gestational
>    Diabetes Mellitus (GDM) Data Set** (3.525 pasien hamil, data rumah
>    sakit rujukan di Uganda). Targetnya menjadi **`DiagnosisType`** dengan
>    3 kelas yang diambil dari **diagnosis klinis asli kedua studi**:
>
> | Kelas | Sumber | Dasar label |
> | --- | --- | --- |
> | Normal | Pima + GDM | `Outcome=0` (Pima) atau `Non-GDM` (GDM) |
> | Tipe 2 | Pima | `Outcome=1` -- diagnosis T2D asli studi Pima |
> | Gestasional | GDM | `Class=GDM` -- diagnosis GDM asli studi tsb. |
>
> Cara penggabungannya mengikuti El-Bashbishy & El-Bakry (Scientific
> Reports, 2026, "Pediatric diabetes prediction using machine
> learning", doi:10.1038/s41598-025-24964-y), yaitu union kolom yang
> memang diukur di kedua dataset.
>
> 4. **Perbaikan imputasi:** dulu nilai hilang diisi median per kelas
>    diagnosis sebelum data dibagi latih/uji. Belakangan ketahuan semua
>    nilai hilang GDM ada di kelas Non-GDM, jadi nilai pengisinya
>    membocorkan label. Sekarang imputasi dilakukan sesudah split dengan
>    median data latih tanpa melihat label. Akurasinya turun dari 0,8591
>    menjadi 0,8510 (lihat bagian Hasil).

## 3. Analisis Permasalahan

### 3.1 Mengapa Union Kolom Dibatasi pada Lima Atribut

Dari 8 atribut Pima dan 17 atribut GDM, hanya **lima** yang punya
padanan langsung: `Age`, jumlah kehamilan (`Pregnancies`), `BMI`,
tekanan darah diastolik (`BloodPressure`), dan kadar glukosa OGTT
(`Glucose`). Paper acuan memakai union semua kolom dan mengisi kolom
yang sama sekali tidak ada di salah satu sumber dengan MICE. Cara itu
sempat kami pertimbangkan, tapi **tidak dipakai** karena terlalu berat
untuk dicek ulang di tugas kuliah. Jadi atribut yang hanya ada di Pima
(`SkinThickness`, `Insulin`, `DiabetesPedigreeFunction`) atau hanya di
GDM (`HDL`, `Hemoglobin`, `Family History`, dll.) tidak masuk model.

### 3.2 Keterbatasan Paling Penting: Status Kehamilan sebagai Perancu

**Semua pasien GDM sedang hamil**, karena memang begitu definisinya,
sedangkan pasien Pima **belum tentu hamil**. Datanya menunjukkan hal ini
(lihat bagian Hasil). Rata-rata BMI kelas "Normal" di kedua sumber
selisih 7 poin (Pima 30,9, GDM 23,7). Selain itu, **sumber data** lebih
mudah ditebak (akurasi 93,6%) daripada **DiagnosisType** (85,1%).
Artinya sebagian performa model berasal dari perbedaan studi dan status
kehamilan, tidak murni dari tipe diabetes. Dampaknya terasa pada pasien
asal Pima: model hampir tidak lebih baik daripada tebakan yang selalu
memilih `Normal` (68,8% vs 65,6%).

## 4. Sumber Data

**Dataset 1 - Pima Indians Diabetes Database.** Berisi 768 pasien
perempuan Pima Indian berusia >= 21 tahun, 8 atribut, dengan label `Outcome`.
<https://www.kaggle.com/datasets/jamaltariqcheema/pima-indians-diabetes-dataset>

**Dataset 2 - Gestational Diabetes Mellitus (GDM) Data Set.** Berisi
3.525 pasien hamil, 17 kolom, dengan label `Class Label(GDM /Non GDM)`.
Datanya dari pemeriksaan kehamilan rutin di rumah sakit rujukan Uganda
(Ahishakiye dkk., 2026, *Discover Artificial Intelligence*) dan diunggah
ke Kaggle oleh Sumathi Santhosh:
<https://www.kaggle.com/datasets/sumathisanthosh/gestational-diabetes-mellitus-gdm-data-set>.
Salinannya ada di `data/gdm_dataset.xlsx`.

Pima tidak menulis nilai kosong secara eksplisit. Di lima kolom
fisiologis, nilai 0 dipakai sebagai penanda data kosong. GDM sudah
langsung memakai `NaN`:

| Kolom | Pima (nilai 0) | GDM (NaN) |
| --- | --- | --- |
| Glucose / OGTT | 5 (0,7%) | 513 (14,6%) |
| BloodPressure / Dia BP | 35 (4,6%) | 0 |
| BMI | 11 (1,4%) | 1.081 (30,7%) |
| Age | - | 0 |
| Pregnancies | - | 0 |

**Semua nilai hilang GDM ada di kelas Non-GDM**: 1.081 `BMI` dan 513
`OGTT` dari 2.153 pasien Non-GDM, sedangkan di 1.372 pasien GDM tidak
ada sama sekali. Karena ada atau tidaknya nilai hilang berhubungan
dengan label, kami tidak memakai median per kelas diagnosis. Data dibagi
latih/uji dulu, baru nilai hilang diisi median setiap fitur dari **data
latih** tanpa melihat label (`BMI` 28,4; `BloodPressure` 80;
`Glucose` 154).

## 5. Metode Komputasi Soft Computing

Metode yang dipakai adalah Naive Bayes, tepatnya Gaussian Naive Bayes.
Urutannya:

1. Bagi data latih/uji, lalu isi nilai hilang dengan median data latih.
2. Hitung probabilitas prior setiap kelas dari data latih.
3. Taksir distribusi normal (mean dan variansi) setiap atribut di tiap kelas.
4. Hitung probabilitas posterior dengan Teorema Bayes.
5. Evaluasi dengan akurasi, precision, recall, F1, dan confusion matrix multi-kelas.

Di `notebooks/03_GaussianNB.ipynb` posterior juga kami hitung manual,
dan hasilnya sama persis dengan `predict_proba()` dari scikit-learn.

## Struktur Proyek

```
diabetes-risk-gnb/
├── README.md
├── requirements.txt
├── app_streamlit.py               aplikasi cek kategori diabetes (Streamlit)
├── .streamlit/config.toml         tema aplikasi
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
│   ├── gnb_diagnosis.joblib        model terlatih utama
│   ├── feature_columns.json        urutan kolom fitur saat pelatihan
│   └── imputation_medians.json     median data latih kelima fitur, dipakai mengisi nilai hilang
└── outputs/
    ├── metrics.json
    ├── confusion_matrices.png
    ├── eda_source_label_distributions.png
    ├── eda_correlation_heatmap.png
    └── preprocessing_merged_feature_distributions.png
```

Semua logika, mulai dari memuat data, menandai nilai hilang,
menggabungkan, imputasi, pelatihan, sampai evaluasi, ditulis di keempat
notebook. Tidak ada modul Python terpisah. Setiap notebook bisa jalan
sendiri dan hanya terhubung lewat berkas yang dihasilkannya: `02`
membuat `merged_diagnosis.csv`, `03` mengisi folder `models/`, dan `04`
membuat `metrics.json` serta `confusion_matrices.png`.

Di bagian atas setiap notebook ada sel **Fungsi** berisi 2-5 fungsi kecil
untuk tahap tersebut. Sel-sel di bawahnya tinggal memanggil fungsi itu.
Kode aplikasi juga dibagi menjadi 4 bagian yang ditandai komentar.

| Bagian | Isi | Fungsi utama |
| --- | --- | --- |
| `01_EDA` | cek data, nilai kosong, perbandingan Pima vs GDM, korelasi | `muat_data`, `tandai_nol_sebagai_kosong`, `nilai_kosong_gdm_per_kelas`, `cek_statistik_pima`, `rentang` |
| `02_Preprocessing` | tandai nilai kosong, samakan kolom, beri label, gabung | `muat_data`, `tandai_nol_sebagai_kosong`, `samakan_kolom_gdm`, `beri_label` |
| `03_GaussianNB` | split + isi median, training, cross validation, posterior manual | `bagi_dan_isi`, `cek_var_smoothing`, `posterior_manual` |
| `04_Evaluation` + aplikasi | metrik, confusion matrix, uji asal data, aplikasi Streamlit | `bentuk_ulang_data_uji`, `evaluate_predictions`, `evaluate_by_group`, `gambar_confusion_matrix` |

Tabel ini sekaligus jadi pembagian tugas kelompok: setiap anggota memegang satu baris.

## Instalasi

```
pip install -r requirements.txt
```

## Penggunaan

1. Jalankan notebook `01` sampai `04` berurutan (`jupyter notebook notebooks/`).
2. Jalankan aplikasi:

```
streamlit run app_streamlit.py
```

Aplikasinya punya dua tab. Tab **Cek kategori** berisi form lima indikator.
Setelah dikirim, tab ini menampilkan kategori yang paling mungkin,
probabilitas posterior setiap kelas, dan rincian perhitungan Naive Bayes
(prior, log-likelihood, posterior). Tab **Tentang model** menampilkan metrik
data uji, akurasi per sumber data, dan confusion matrix dari `outputs/`.

## Konfigurasi

| Parameter | Nilai | Lokasi |
| --- | --- | --- |
| `TEST_SIZE` / `RANDOM_STATE` | 0.2 / 42 | `03_GaussianNB` dan `04_Evaluation` (harus sama; `04` memeriksanya) |
| Imputasi nilai hilang | median data latih, tanpa label, setelah split | `03_GaussianNB` |
| Lima fitur bersama (union) | Age, Pregnancies, BMI, BloodPressure, Glucose | `02_Preprocessing`, `SHARED_FEATURES` |
| Pemetaan label diagnosis | Pima Outcome + GDM Class -> DiagnosisType | `02_Preprocessing` bagian 4 |
| Pemetaan kolom GDM -> Pima | Age, No of Pregnancy, BMI, Dia BP, OGTT | `02_Preprocessing`, `GDM_TO_PIMA_COLUMNS` |

## Hasil

Semua angka di bawah ini berasal dari `notebooks/04_Evaluation.ipynb`
dengan `random_state=42`.

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

**Uji perancu sumber data (hasil paling penting):** kami melatih
classifier lain dengan lima fitur yang sama, tetapi targetnya
`SumberData` (Pima atau GDM), bukan `DiagnosisType`. Akurasinya
**0,9360**, lebih tinggi daripada akurasi model untuk tugas aslinya
(0,8510). Jadi kelima fitur ini lebih mudah mengenali studi asal pasien
daripada tipe diabetesnya.

**Akurasi per sumber data.** Sebagai baseline dipakai tebakan yang
selalu memilih kelas terbanyak di sumber tersebut, yaitu `Normal` untuk
keduanya:

| Sumber | Data uji | Akurasi model | Baseline |
| --- | --- | --- | --- |
| GDM | 702 | 0,8875 | 0,6097 |
| Pima | 157 | 0,6879 | 0,6561 |

Hampir semua keunggulan model datang dari pasien GDM. Pada pasien asal
Pima, model hanya unggul sekitar 3 poin dari baseline (5 dari 157 pasien
uji), dan 29 dari 103 pasien `Normal` asal Pima malah ditebak `Tipe 2`.
Karena itu akurasi keseluruhan jangan dianggap sebagai kemampuan model
membedakan `Normal` dan `Tipe 2`.

**Pengaruh cara imputasi** (split sama, angka dari
`notebooks/04_Evaluation.ipynb`):

| Cara imputasi | Akurasi | F1 makro | Recall Gestasional | Akurasi baris lengkap | Akurasi baris terimputasi |
| --- | --- | --- | --- | --- | --- |
| Median per kelas, sebelum split (lama) | 0,8591 | 0,7708 | 0,8102 | 0,8083 | 0,9957 |
| Median data latih, tanpa label (sekarang) | 0,8510 | 0,7707 | 0,7555 | 0,7971 | 0,9957 |

Cara lama kami tinggalkan karena hasilnya terlalu optimistis: nilai
pengisi per kelas ikut membantu memisahkan kelas. Namun ada satu hal
yang tidak berubah setelah perbaikan. Baris uji hasil imputasi (233
baris, 228 di antaranya `Normal`) tetap 99,6% benar, jauh lebih tinggi
daripada baris dengan data lengkap (626 baris, 0,7971).

## Keterbatasan

**Perancu sumber data / status kehamilan.** Ini keterbatasan terbesar
di revisi ini, dan angkanya sudah ditunjukkan di atas.

**Model hampir tidak membantu pasien asal Pima.** Akurasinya 0,6879,
baseline 0,6561. Penyebabnya perancu tadi: kelas `Normal` didominasi
pasien GDM, sehingga bagi model pasien `Normal` asal Pima terlihat mirip
`Tipe 2`.

**Hanya lima fitur.** Atribut yang cuma ada di salah satu dataset
tidak dipakai (lihat bagian 3.1).

**Kelas sangat tidak seimbang.** Porsi `Tipe 2` hanya 6,2% data, dan
precision/recall-nya jauh di bawah dua kelas lainnya.

**Nilai hilang berhubungan dengan label.** Semua nilai hilang GDM ada
di kelas Non-GDM. Dengan median data latih tanpa label, nilai pengisi
memang tidak lagi membocorkan label. Tetapi baris hasil imputasi masih
jauh lebih mudah ditebak daripada baris lengkap. Kalau ingin angka yang
lebih hati-hati, pakai akurasi baris lengkap (0,7971), bukan akurasi
keseluruhan (0,8510).

**Arti `Pregnancies` berbeda.** Di Pima kolom ini berarti jumlah
kehamilan seumur hidup, bisa 0. Di GDM artinya kehamilan keberapa yang
sedang dijalani, jadi minimal 1. Keduanya tetap digabung jadi satu kolom
walaupun cara mengukurnya berbeda.

**Asumsi independensi tidak sepenuhnya terpenuhi.** Di Pima, korelasi
`Age` dan `Pregnancies` 0,54.

**Cross validation hanya di data latih.** Hasil 5-fold CV di notebook
03 adalah akurasi 0,8614 ± 0,0087 dan F1 makro 0,7664 ± 0,0279, tidak
jauh dari hasil data uji. Karena `var_smoothing` default (1e-9) sudah
termasuk yang terbaik, parameternya tidak kami ubah.

## Riwayat Skema Sebelumnya

Versi ini sudah tidak menyimpan kode skema `GlucoseRisk` dan
`InsulinResistance` (revisi 1 dan 2), begitu juga `main.py` dan paket
`src/`. Kalau perlu, kode lama masih ada di riwayat commit git. Cara
imputasi lama (median per kelas sebelum split) sekarang hanya dipakai
sebagai pembanding di `notebooks/04_Evaluation.ipynb`.
