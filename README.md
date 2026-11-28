# Klasifikasi Risiko Diabetes dengan Gaussian Naive Bayes

## 1. Ide (Topik)

Klasifikasi Risiko Diabetes Menggunakan Naive Bayes Berdasarkan Data
Diagnostik Pasien.

## 2. Deskripsi Ide

Proyek ini membangun model **Gaussian Naive Bayes** yang menebak kategori
diagnosis diabetes seorang pasien perempuan: Normal / Non-GDM, Tipe 2,
atau Gestasional. Model dipakai di aplikasi Streamlit sederhana.

> **Riwayat revisi:**
> 1. Awalnya target berupa `Outcome` biner, lalu diubah menjadi 3 kelas
>    `GlucoseRisk` yang hanya ditentukan dari ambang klinis `Glucose`.
> 2. Supaya ada faktor selain glukosa, target diganti `InsulinResistance`,
>    yaitu proksi resistensi insulin dari `Glucose` x `Insulin` yang dibagi
>    dengan ambang tertil dari data.
> 3. Kami mencari dataset nyata yang pasiennya memang didiagnosis dengan
>    tipe diabetes berbeda. Pima digabung dengan **Gestational Diabetes
>    Mellitus (GDM) Data Set** (3.525 pasien hamil, data rumah sakit rujukan
>    di Uganda). Targetnya menjadi **`DiagnosisType`** dengan 3 kelas dari
>    **diagnosis klinis asli kedua studi**:
>
> | Kelas | Sumber | Dasar label |
> | --- | --- | --- |
> | Normal / Non-GDM | Pima + GDM | `Outcome=0` (Pima, tidak diabetes) atau `Non-GDM` (GDM, hamil tanpa diabetes gestasional) |
> | Tipe 2 | Pima | `Outcome=1`, diagnosis T2D asli studi Pima |
> | Gestasional | GDM | `Class=GDM`, diagnosis GDM asli studi tersebut |
>
> Cara penggabungannya mengikuti El-Bashbishy & El-Bakry (Scientific
> Reports, 2026, "Pediatric diabetes prediction using machine
> learning", doi:10.1038/s41598-025-24964-y), yaitu union kolom yang
> memang diukur di kedua dataset.
>
> 4. Imputasi median per kelas sebelum split diganti imputasi median data
>    latih, karena cara lama membocorkan label.
> 5. **Versi final:** baris kembar dan baris yang tidak lengkap dibuang
>    (tidak ada imputasi lagi), prior kelas diatur secara eksplisit,
>    evaluasi memakai repeated stratified cross validation dan satu data
>    uji akhir, metrik utama diganti balanced accuracy dan F1 makro, dan
>    aplikasi diberi input status kehamilan serta peringatan rentang input.
> 6. Kelas negatif diberi nama **Normal / Non-GDM** karena isinya pasien
>    Pima yang tidak diabetes dan pasien hamil GDM yang hasil tesnya negatif.
>    Data latih **diseimbangkan** (random oversampling) sehingga ketiga kelas
>    berjumlah sama. Data uji tetap berisi pasien asli dengan proporsi kelas
>    aslinya.

## 3. Analisis Permasalahan

### 3.1 Mengapa Hanya Lima Atribut

Dari 8 atribut Pima dan 17 atribut GDM, hanya **lima** yang punya
padanan langsung: `Age`, jumlah kehamilan (`Pregnancies`), `BMI`,
tekanan darah diastolik (`BloodPressure`), dan kadar glukosa 2 jam TTGO
(`Glucose`). Paper acuan memakai union semua kolom dan mengisi kolom
yang sama sekali tidak ada di salah satu sumber dengan MICE. Cara itu
tidak dipakai karena mengarang nilai untuk kolom yang tidak pernah
diukur. Atribut yang hanya ada di Pima (`SkinThickness`, `Insulin`,
`DiabetesPedigreeFunction`) atau hanya di GDM (`HDL`, `Hemoglobin`,
`Family History`, dan lain-lain) tidak masuk model.

### 3.2 Status Kehamilan dan Asal Data sebagai Perancu

Semua pasien GDM sedang hamil, sedangkan pasien Pima belum tentu hamil.
Kelas Tipe 2 hanya ada di Pima dan kelas Gestasional hanya ada di GDM,
sehingga asal data dan label saling terkait. Akibatnya, kelima fitur
lebih mudah menebak **asal data** (akurasi 0,9352 pada cross validation,
baseline 0,7674) daripada membedakan Normal dan Tipe 2 di dalam data Pima
(balanced accuracy pasien Pima 0,6741). Bagian Hasil menjelaskan
percobaan mitigasinya.

## 4. Sumber Data

**Dataset 1 - Pima Indians Diabetes Database.** Berisi 768 pasien
perempuan Pima Indian berusia 21 tahun ke atas, 8 atribut, dengan label `Outcome`.
<https://www.kaggle.com/datasets/jamaltariqcheema/pima-indians-diabetes-dataset>

**Dataset 2 - Gestational Diabetes Mellitus (GDM) Data Set.** Berisi
3.525 pasien hamil, 17 kolom, dengan label `Class Label(GDM /Non GDM)`.
Datanya dari pemeriksaan kehamilan rutin di rumah sakit rujukan Uganda
(Ahishakiye dkk., 2026, *Discover Artificial Intelligence*) dan diunggah
ke Kaggle oleh Sumathi Santhosh:
<https://www.kaggle.com/datasets/sumathisanthosh/gestational-diabetes-mellitus-gdm-data-set>.
Salinannya ada di `data/gdm_dataset.xlsx`.

Tidak ada data dari sumber lain dan tidak ada data hasil penyeimbangan yang
disimpan ke berkas data.
Berkas mentah `data/diabetes.csv` dan `data/gdm_dataset.xlsx` tidak diubah.

Pima menulis nilai kosong sebagai 0 pada kolom fisiologis, sedangkan GDM
langsung memakai `NaN`:

| Kolom | Pima (nilai 0) | GDM (NaN) |
| --- | --- | --- |
| Glucose / OGTT | 5 (0,7%) | 513 (14,6%) |
| BloodPressure / Dia BP | 35 (4,6%) | 0 |
| BMI | 11 (1,4%) | 1.081 (30,7%) |
| Age | - | 0 |
| Pregnancies | - | 0 |

Semua nilai kosong GDM ada di kelas Non-GDM (1.081 `BMI` dan 513 `OGTT`
dari 2.153 pasien Non-GDM, nol dari 1.372 pasien GDM). Artinya, ada atau
tidaknya nilai kosong sudah membawa informasi label.

### 4.1 Perubahan pada Dataset Gabungan

Pembersihan dilakukan di `notebooks/02_Preprocessing.ipynb` tanpa langkah
acak, jadi hasilnya selalu sama setiap kali dijalankan.

| Tahap | Jumlah baris | Keterangan |
| --- | --- | --- |
| Gabungan Pima + GDM | 4.293 | 5 fitur, `DiagnosisType`, `SumberData` |
| Buang baris kembar persis | 3.938 | 355 baris dibuang, semuanya Non-GDM dari GDM |
| Buang baris tidak lengkap | 3.160 | 778 baris dibuang: 734 Non-GDM dari GDM, 25 Normal Pima, 19 Tipe 2 Pima |

Baris kembar muncul karena setelah GDM diambil 5 kolomnya, banyak pasien
Non-GDM yang BMI dan OGTT-nya kosong menjadi identik. Sebanyak 493 baris
termasuk kelompok kembar, 477 di antaranya memuat nilai kosong.

Distribusi kelas data bersih: Normal / Non-GDM 1.539 (48,7%), Gestasional
1.372 (43,4%), Tipe 2 249 (7,9%). Menurut sumber: Pima 724 baris (475
Normal, 249 Tipe 2), GDM 2.436 baris (1.064 Non-GDM, 1.372 Gestasional).
Rasio kelas terbanyak dan tersedikit 6,18 : 1, jadi data bersih tidak
seimbang.

### 4.2 Penyeimbangan Kelas

Penyeimbangan dilakukan di `notebooks/03_GaussianNB.ipynb` dan **hanya pada
data latih**, di dalam Pipeline imblearn. Di dalam cross validation, yang
diseimbangkan hanya fold latih. Data uji akhir tidak diseimbangkan, supaya
model dinilai pada pasien asli dengan proporsi aslinya. Hasil penyeimbangan
tidak disimpan ke berkas data.

| Kelas | Data latih asli | Random undersampling | Random oversampling (dipakai) | SMOTE |
| --- | --- | --- | --- | --- |
| Normal / Non-GDM | 1.231 | 199 | 1.231 | 1.231 |
| Tipe 2 | 199 | 199 | 1.231 | 1.231 |
| Gestasional | 1.098 | 199 | 1.231 | 1.231 |
| Jumlah | 2.528 | 597 | 3.693 | 3.693 |

Random oversampling mengambil ulang baris pasien asli dari kelas yang
lebih kecil secara acak (`random_state=42`) sampai jumlahnya sama dengan
kelas terbesar. Cara ini dipilih karena F1 makro cross validation-nya
paling tinggi di antara ketiga cara penyeimbangan (lihat bagian Hasil).

## 5. Metode Komputasi Soft Computing

Metode yang dipakai adalah Naive Bayes, tepatnya Gaussian Naive Bayes.
Urutannya:

1. Sisihkan 20% data bersih sebagai data uji akhir (632 baris, stratified,
   `random_state=42`). Data ini hanya dipakai sekali di notebook 04.
2. Pada 80% sisanya (data pengembangan, 2.528 baris), bandingkan cara
   penyeimbangan kelas, transformasi, prior, dan fitur dengan repeated
   stratified 5-fold cross validation yang diulang 10 kali (50 fold). Semua
   langkah ada di dalam Pipeline sehingga hanya di-fit pada fold latih.
3. Seimbangkan data latih dengan random oversampling (2.528 baris asli
   menjadi 3.693 baris, 1.231 per kelas). Prior setiap kelas dihitung dari
   data latih yang sudah seimbang, sehingga bernilai 1/3, lalu dimasukkan
   secara eksplisit ke `GaussianNB`.
4. Taksir distribusi normal (mean dan variansi) setiap atribut di tiap kelas.
5. Hitung probabilitas posterior dengan Teorema Bayes dalam bentuk log.
6. Evaluasi dengan balanced accuracy dan F1 makro sebagai metrik utama,
   ditambah recall dan F1 per kelas, confusion matrix, akurasi, dan selang
   kepercayaan bootstrap 95%.

Di `notebooks/03_GaussianNB.ipynb` posterior juga dihitung manual, dan
hasilnya sama dengan `predict_proba()` scikit-learn untuk seluruh 2.528
data pengembangan.

## Struktur Proyek

```
diabetes-risk-gnb/
├── README.md
├── requirements.txt
├── app_streamlit.py                 aplikasi cek kategori diabetes (Streamlit)
├── .streamlit/config.toml           tema aplikasi
├── data/
│   ├── diabetes.csv                 dataset Pima mentah
│   ├── gdm_dataset.xlsx             dataset GDM mentah
│   └── processed/
│       ├── merged_diagnosis.csv          data bersih 3.160 baris (dipakai model dan aplikasi)
│       └── merged_diagnosis_mentah.csv   gabungan 4.293 baris sebelum dibersihkan (hanya untuk pembanding di notebook 04)
├── notebooks/
│   ├── 01_EDA.ipynb                 eksplorasi kedua dataset, nilai kosong, kemiringan, korelasi
│   ├── 02_Preprocessing.ipynb       penandaan nilai kosong, penggabungan, buang baris kembar dan tidak lengkap
│   ├── 03_GaussianNB.ipynb          data uji akhir, penyeimbangan kelas, cross validation, transformasi, prior, ablasi, model final, posterior manual
│   └── 04_Evaluation.ipynb          evaluasi akhir, per sumber, tebak asal data, model pembanding, kalibrasi, dampak perubahan data
├── models/
│   ├── gnb_diagnosis.joblib         model final (Pipeline berisi GaussianNB)
│   └── info_model.json              fitur, penyeimbangan, prior, jumlah data latih, dan rentang nilai data latih per kondisi kehamilan
└── outputs/
    ├── metrics.json
    ├── confusion_matrices.png
    ├── kalibrasi.png
    ├── eda_source_label_distributions.png
    ├── eda_correlation_heatmap.png
    └── preprocessing_merged_feature_distributions.png
```

Seluruh logika ada di keempat notebook; tidak ada modul Python terpisah.
Notebook terhubung lewat berkas yang dihasilkannya: `02` membuat kedua
berkas di `data/processed/`, `03` mengisi `models/`, dan `04` mengisi
`outputs/metrics.json`, `confusion_matrices.png`, dan `kalibrasi.png`.

Di bagian atas setiap notebook ada sel **Fungsi**. Sel di bawahnya tinggal
memanggil fungsi tersebut. Kode aplikasi dibagi menjadi 4 bagian yang
ditandai komentar.

| Bagian | Isi | Fungsi utama |
| --- | --- | --- |
| `01_EDA` | cek data, nilai kosong, perbandingan Pima dan GDM, kemiringan, korelasi | `muat_data`, `tandai_nol_sebagai_kosong`, `nilai_kosong_gdm_per_kelas`, `cek_statistik_pima`, `rentang` |
| `02_Preprocessing` | tandai nilai kosong, samakan kolom, beri label, gabung, bersihkan | `muat_data`, `tandai_nol_sebagai_kosong`, `samakan_kolom_gdm`, `beri_label`, `hitung_kembar`, `bersihkan` |
| `03_GaussianNB` | penyeimbangan kelas, cross validation, transformasi, prior, ablasi, model final, posterior manual | `GaussianNBPrior`, `prior_campuran`, `buat_model`, `cv_berulang`, `uji_t_terkoreksi`, `posterior_manual` |
| `04_Evaluation` + aplikasi | metrik akhir, selang kepercayaan, tebak asal data, pembanding, kalibrasi, aplikasi | `skor_prediksi`, `bootstrap_ci`, `brier_multikelas`, `gambar_confusion_matrix`, `skema_lama` |

Tabel ini juga dipakai sebagai pembagian tugas kelompok: setiap anggota
memegang satu baris.

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

Aplikasinya punya dua tab.

**Cek kategori.** Pengguna memilih status kehamilan, lalu mengisi lima
indikator. Hal-hal yang diatur di tab ini:

- Bila status **tidak hamil**, posterior Gestasional dijadikan 0 lalu dua
  kelas lain dinormalkan ulang. Label `Pregnancies` menjadi "jumlah
  kehamilan seumur hidup".
- Bila status **hamil**, label `Pregnancies` menjadi "kehamilan ke berapa
  saat ini".
- Rentang nilai diambil secara terprogram dari data latih untuk tiap
  kondisi (`models/info_model.json`). Untuk kondisi hamil, umur 20 sampai
  45 dan kehamilan 1 sampai 4.
- Input di luar rentang tetap dihitung, tetapi diberi peringatan bahwa
  hasilnya ekstrapolasi dan kurang andal.
- Kolom glukosa berlabel "Glukosa 2 jam setelah TTGO (mg/dL)", dengan teks
  bantuan bahwa gula darah puasa atau sewaktu tidak boleh dimasukkan.
  Nilai di bawah 60 atau di atas 400 mg/dL diberi peringatan.
- Ada pemberitahuan bahwa model hanya dilatih pada pasien perempuan, dan
  pernyataan bahwa aplikasi bukan alat diagnosis.
- Hasilnya berupa kategori paling mungkin, probabilitas setiap kelas, dan
  rincian perhitungan Naive Bayes.

**Tentang model.** Berisi metrik data uji akhir beserta selang
kepercayaan, metrik per kelas dan per sumber, confusion matrix, tabel
perbandingan model, dan keterangan penyeimbangan data latih.

## Konfigurasi

| Parameter | Nilai | Lokasi |
| --- | --- | --- |
| `TEST_SIZE` / `RANDOM_STATE` | 0.2 / 42 | `03_GaussianNB` dan `04_Evaluation` (harus sama; `04` memeriksanya) |
| Cross validation | repeated stratified 5-fold, 10 kali ulang | `N_SPLITS`, `N_REPEATS` di `03` dan `04` |
| Nilai kosong | baris dibuang (complete-case), tanpa imputasi | `02_Preprocessing`, `bersihkan` |
| Penyeimbangan kelas | random oversampling pada data latih (dipilih dari undersampling, oversampling, SMOTE) | `03_GaussianNB` bagian 3 dan 4, `PENYEIMBANG` |
| Prior | 1/3 per kelas, dihitung dari data latih yang sudah seimbang | `03_GaussianNB` bagian 6 |
| Transformasi fitur | tidak ada (dipilih dari tanpa, log1p, Yeo-Johnson) | `03_GaussianNB` bagian 4 |
| Lima fitur | Age, Pregnancies, BMI, BloodPressure, Glucose | `02_Preprocessing`, `SHARED_FEATURES` |
| Bootstrap selang kepercayaan | 2.000 sampel ulang, persentil 2,5 dan 97,5 | `04_Evaluation`, `N_BOOTSTRAP` |

## Hasil

Semua angka di bawah ini berasal dari eksekusi ulang notebook `03` dan
`04` (`random_state=42`) dan tersimpan di `outputs/metrics.json`.

### Model final pada data uji akhir

Data uji akhir: 632 baris pasien asli yang tidak diseimbangkan (Normal /
Non-GDM 308, Tipe 2 50, Gestasional 274).

| Metrik | Nilai | Selang kepercayaan 95% |
| --- | --- | --- |
| **Balanced accuracy** | **0,8261** | 0,7813 sampai 0,8707 |
| **F1 makro** | **0,7819** | 0,7369 sampai 0,8223 |
| Akurasi | 0,8560 | 0,8291 sampai 0,8829 |
| Recall Tipe 2 | 0,7400 | 0,6154 sampai 0,8571 |
| F1 Tipe 2 | 0,5649 | 0,4545 sampai 0,6619 |
| Brier score multikelas | 0,2469 | - |

| Kelas | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Normal / Non-GDM | 0,9161 | 0,8149 | 0,8625 | 308 |
| Tipe 2 | 0,4568 | 0,7400 | 0,5649 | 50 |
| Gestasional | 0,9134 | 0,9234 | 0,9183 | 274 |

Confusion matrix (baris = aktual Normal / Non-GDM, Tipe 2, Gestasional;
kolom = prediksi dengan urutan sama): `[[251, 37, 20], [9, 37, 4], [14, 7, 253]]`.

Cross validation 5 x 10 pada data pengembangan memberi hasil yang sejalan:
balanced accuracy 0,8250 ± 0,0249, F1 makro 0,7824 ± 0,0210, recall
Tipe 2 0,7392 ± 0,0645, F1 Tipe 2 0,5746 ± 0,0431.

### Per sumber data (data uji akhir)

| Sumber | Data uji | Akurasi | Balanced accuracy | F1 makro | Baseline kelas terbanyak |
| --- | --- | --- | --- | --- | --- |
| Pima | 136 | 0,6471 | 0,6665 | 0,6551 | 0,6324 (Normal) |
| GDM | 496 | 0,9133 | 0,9121 | 0,9221 | 0,5524 (Gestasional) |

Model jauh lebih baik pada pasien GDM. Pada pasien Pima, 34 dari 86 pasien
Normal ditebak Tipe 2, dan 5 pasien Pima ditebak Gestasional. Di aplikasi,
5 kasus terakhir ini tertangani oleh input status kehamilan.

### Tebak asal data dan mitigasinya

| Skema (CV 5 x 10) | Akurasi tebak sumber | Baseline sumber terbanyak | Balanced accuracy diagnosis | F1 makro diagnosis |
| --- | --- | --- | --- | --- |
| Tanpa mitigasi | 0,9352 | 0,7674 | 0,8250 | 0,7824 |
| Standardisasi per sumber (fit pada fold latih) | 0,7674 | 0,7674 | 0,6052 | 0,5798 |

Pada data uji akhir, akurasi tebak sumber tanpa mitigasi adalah 0,9446.
Standardisasi per sumber memang menurunkan akurasi tebak sumber sampai
sama dengan baseline, tetapi kemampuan diagnosis ikut turun jauh. Hal ini
terjadi karena Tipe 2 hanya ada di Pima dan Gestasional hanya ada di GDM,
sehingga informasi asal data tidak bisa dipisahkan dari informasi label.
Mitigasi ini tidak dipakai, dan **kebocoran asal data tetap ada**.

### Penyeimbangan, transformasi, dan prior (CV 5 x 10)

Kemiringan fitur pada data pengembangan sebelum transformasi: Age 0,855,
Pregnancies 2,530, BMI 0,611, BloodPressure 0,273, Glucose 1,133. Sesudah
Yeo-Johnson, semuanya di antara -0,021 dan 0,029. Sesudah log1p, nilainya
di antara -0,469 dan 0,221.

| Transformasi | Penyeimbangan atau prior | Balanced accuracy | F1 makro | Recall Tipe 2 |
| --- | --- | --- | --- | --- |
| tanpa | tanpa penyeimbangan, prior empiris | 0,7718 | 0,7720 | 0,5215 |
| tanpa | tanpa penyeimbangan, prior seragam | 0,8248 | 0,7828 | 0,7387 |
| tanpa | tanpa penyeimbangan, prior tuning (nested CV) | 0,8145 | 0,7793 | 0,7003 |
| tanpa | random undersampling | 0,8212 | 0,7797 | 0,7277 |
| **tanpa** | **random oversampling (dipakai)** | **0,8250** | **0,7824** | **0,7392** |
| tanpa | SMOTE | 0,8286 | 0,7811 | 0,7564 |
| log1p | random oversampling | 0,8292 | 0,7564 | 0,8397 |
| log1p | SMOTE | 0,8315 | 0,7606 | 0,8408 |
| Yeo-Johnson | random oversampling | 0,8344 | 0,7603 | 0,8488 |
| Yeo-Johnson | SMOTE | 0,8341 | 0,7628 | 0,8423 |

Tabel lengkap 18 konfigurasi ada di notebook 03 bagian 4. Model final
diwajibkan memakai data latih yang seimbang, jadi konfigurasi dipilih hanya
dari yang memakai penyeimbangan, berdasarkan F1 makro (dibulatkan 3 angka,
lalu balanced accuracy). Transformasi membuat distribusi lebih normal dan
menaikkan recall Tipe 2, tetapi menurunkan F1 makro, jadi tidak dipakai.

Dibanding tanpa penyeimbangan (prior empiris), random oversampling
menaikkan balanced accuracy (+0,0532; p < 0,0001) dan recall Tipe 2
(+0,2178; p < 0,0001). Kenaikan F1 makro (+0,0104; p = 0,32) tidak
signifikan. Hasilnya praktis sama dengan prior seragam tanpa penyeimbangan
(selisih F1 makro -0,0004; p = 0,86), karena bagi Gaussian Naive Bayes
oversampling terutama mengubah prior menjadi 1/3, sedangkan mean dan
variansi tiap kelas hanya sedikit bergeser. Nilai p berasal dari uji t
berpasangan terkoreksi Nadeau dan Bengio.

### Ablasi fitur (CV 5 x 10, random oversampling)

| Fitur dibuang | Balanced accuracy | p | F1 makro | p | Bal. acc. Pima | Bal. acc. GDM |
| --- | --- | --- | --- | --- | --- | --- |
| (tidak ada) | 0,8250 | - | 0,7824 | - | 0,6741 | 0,9114 |
| Age | 0,8253 | 0,97 | 0,7803 | 0,81 | 0,6400 | 0,9391 |
| Pregnancies | 0,8032 | 0,07 | 0,7508 | 0,0005 | 0,6463 | 0,9032 |
| BMI | 0,7641 | 0,0002 | 0,7530 | 0,024 | 0,6430 | 0,8844 |
| BloodPressure | 0,8175 | 0,30 | 0,7724 | 0,06 | 0,6731 | 0,8979 |
| Glucose | 0,7550 | < 0,0001 | 0,7062 | < 0,0001 | 0,6348 | 0,8061 |

Membuang `Pregnancies` menurunkan F1 makro secara signifikan, jadi fitur ini
**dipertahankan** walaupun artinya berbeda: di Pima berarti jumlah
kehamilan seumur hidup (0 sampai 17), di GDM berarti kehamilan ke berapa
yang sedang dijalani (1 sampai 4). Membuang `Age` tidak mengubah metrik
gabungan secara signifikan, tetapi menurunkan balanced accuracy pasien
Pima (0,6741 menjadi 0,6400), jadi `Age` juga dipertahankan.
Glukosa dan BMI paling berpengaruh.

### Perbandingan model (CV 5 x 10, fitur dan fold yang sama)

| Model | Balanced accuracy | F1 makro | Akurasi | Recall Tipe 2 |
| --- | --- | --- | --- | --- |
| Dummy (kelas terbanyak) | 0,3333 | 0,2183 | 0,4869 | 0,0000 |
| Dummy (acak bertingkat) | 0,3289 | 0,3287 | 0,4259 | 0,0748 |
| **GaussianNB final (latih seimbang)** | **0,8250** | **0,7824** | **0,8541** | **0,7392** |
| GaussianNB tanpa penyeimbangan | 0,7718 | 0,7720 | 0,8654 | 0,5215 |
| GaussianNB prior seragam | 0,8248 | 0,7828 | 0,8540 | 0,7387 |
| LDA | 0,7323 | 0,7361 | 0,8373 | 0,4587 |
| QDA | 0,7891 | 0,7768 | 0,8551 | 0,6058 |
| Logistic Regression (balanced) | 0,8357 | 0,7721 | 0,8547 | 0,7779 |
| KNN (k=15) | 0,7438 | 0,7621 | 0,8933 | 0,3492 |
| Random Forest (balanced) | 0,8297 | 0,8113 | 0,8981 | 0,6446 |
| Gradient Boosting | 0,7863 | 0,7976 | 0,9039 | 0,4747 |

Algoritma inti tetap Gaussian Naive Bayes; model lain hanya pembanding dan
tidak di-tuning. Model pembanding tidak diberi langkah penyeimbangan,
kecuali Logistic Regression dan Random Forest yang memakai
`class_weight="balanced"`. LDA, QDA, dan Logistic Regression tidak mengasumsikan
fitur saling bebas, tetapi tidak satu pun yang jelas lebih baik pada
kedua metrik utama. Random Forest unggul pada F1 makro (0,8113), tetapi
tidak bisa dihitung ulang secara manual seperti Naive Bayes.

### Kalibrasi probabilitas (out-of-fold, 5-fold, data pengembangan)

| Model | Brier score multikelas | F1 makro | Balanced accuracy |
| --- | --- | --- | --- |
| GaussianNB final (tanpa kalibrasi) | 0,2347 | 0,7830 | 0,8273 |
| Kalibrasi sigmoid | 0,2133 | 0,7769 | 0,7896 |
| Kalibrasi isotonic | 0,2055 | 0,7800 | 0,8034 |

Kalibrasi isotonic memperbaiki Brier score, tetapi menurunkan balanced
accuracy. Kalibrasi tidak dipasang di model final, supaya probabilitas di
aplikasi tetap sama dengan hitungan manual Naive Bayes. Reliability curve
ada di `outputs/kalibrasi.png`. Probabilitas di aplikasi sebaiknya dibaca
sebagai peringkat kemungkinan, bukan peluang yang sudah terkalibrasi.

### Dampak perubahan data (GaussianNB tanpa penyeimbangan, split 80:20 pada datanya masing-masing)

| Skema | Baris uji dinilai | Akurasi | Balanced accuracy | F1 makro | Akurasi baris uji lengkap | Akurasi baris uji diimputasi | Baris uji yang punya kembaran di data latih | Baris latih dengan BMI = median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a. Mentah + imputasi (skema lama) | 859 | 0,8510 | 0,7804 | 0,7707 | 0,7971 | 0,9957 | 114 | 883 |
| b. Tanpa kembar + imputasi | 788 | 0,8541 | 0,7811 | 0,7725 | 0,8208 | 0,9816 | 0 | 603 |
| c. Seperti b, dinilai pada baris lengkap | 625 | 0,8208 | 0,7776 | 0,7588 | 0,8208 | - | 0 | 603 |
| d. Tanpa kembar + complete-case, dinilai pada baris lengkap | 625 | 0,8704 | 0,8045 | 0,7924 | 0,8704 | - | 0 | 16 |

Membuang baris kembar menghapus 114 baris uji yang punya kembaran persis
di data latih. Pada data ini akurasinya justru sedikit naik (0,8510 menjadi
0,8541), karena data ujinya juga berubah. Pada baris uji yang sama (c dan
d), complete-case lebih baik daripada imputasi untuk semua metrik.
Skema imputasi menebak baris yang diisi dengan akurasi 0,98 sampai 0,99,
karena hampir semuanya Non-GDM. Pada skema complete-case tidak ada lagi
lonjakan BMI di satu nilai: hanya 16 baris latih yang BMI-nya sama dengan
median, dan semuanya nilai asli.

## Kondisi Sebelum dan Sesudah Perbaikan

Kondisi sebelum diperiksa ulang dari model dan data lama sebelum data diubah
(Tahap 0). Data uji sebelum dan sesudah berbeda (859 baris dari 4.293, dan
632 baris dari 3.160), jadi angka keduanya tidak bisa dibandingkan satu
lawan satu. Perbandingan pada split yang sama ada di tabel dampak
perubahan data di atas.

| Ukuran | Sebelum | Sesudah |
| --- | --- | --- |
| Baris data dipakai | 4.293 (dengan imputasi) | 3.160 (tanpa kembar dan tanpa nilai kosong) |
| Label kelas negatif | Normal | Normal / Non-GDM |
| Data latih | tidak seimbang (Tipe 2 7,9%) | seimbang, 1.231 per kelas (random oversampling) |
| Balanced accuracy uji | 0,7804 | 0,8261 |
| F1 makro uji | 0,7707 | 0,7819 |
| Akurasi uji | 0,8510 | 0,8560 |
| Recall Tipe 2 uji | 0,6667 | 0,7400 |
| Akurasi subset Pima (baseline) | 0,6879 (0,6561) | 0,6471 (0,6324) |
| Akurasi tebak asal data (uji) | 0,9360 | 0,9446 |
| Baris uji yang punya kembaran di data latih | 114 dari 859 | 0 dari 632 |
| BMI yang diisi median 28,4 | 1.092 (1.081 GDM + 11 Pima) | 0 |
| Akurasi baris diimputasi / baris lengkap | 0,9957 / 0,7971 | tidak ada imputasi |
| Korelasi Age dan Pregnancies (Pima) | 0,5443 | 0,5443 (data mentah, tidak berubah) |
| Evaluasi | satu split, metrik utama akurasi | CV 5 x 10 + satu data uji akhir, balanced accuracy dan F1 makro, selang kepercayaan |

## Keterbatasan dan Pekerjaan Lanjutan

**Kebocoran asal data tetap ada.** Kelima fitur bisa menebak asal data
dengan akurasi 0,9352 (baseline 0,7674). Mitigasi standardisasi per sumber
menghilangkan kemampuan itu tetapi juga merusak diagnosis, sehingga tidak
dipakai.

**Tipe 2 hanya berasal dari satu sumber.** Semua 249 pasien Tipe 2 berasal
dari Pima. Model belum pernah melihat pasien Tipe 2 yang sedang hamil atau
yang berasal dari populasi lain. Karena itu evaluasi *leave-one-source-out*
tidak bisa dipakai untuk kelas Tipe 2.

**Sampel Tipe 2 kecil.** Data uji akhir hanya memuat 50 pasien Tipe 2, sehingga
selang kepercayaan recall Tipe 2 lebar (0,6154 sampai 0,8571). Precision
Tipe 2 hanya 0,4568: lebih dari separuh tebakan Tipe 2 sebenarnya Normal /
Non-GDM, terutama pasien Normal dari Pima.

**Penyeimbangan tidak menambah informasi.** Random oversampling hanya
mengambil ulang 199 pasien Tipe 2 yang sama sampai 1.231 baris, jadi model
tidak mendapat pasien Tipe 2 baru. Efeknya bagi Gaussian Naive Bayes hampir
sama dengan prior seragam. SMOTE dan undersampling tidak memberi F1 makro
yang lebih baik.

**Model kurang membantu pasien Pima.** Akurasi pasien Pima 0,6471, hanya
sedikit di atas baseline 0,6324.

**Hanya lima fitur.** Tidak ada fitur tambahan karena hanya dua dataset ini
yang boleh dipakai. Ablasi menunjukkan Glucose dan BMI paling berpengaruh,
sedangkan Age dan BloodPressure tidak signifikan pada metrik gabungan.

**Arti `Pregnancies` berbeda** antara Pima (jumlah kehamilan seumur hidup)
dan GDM (kehamilan ke berapa saat ini). Fitur ini tetap dipakai karena
membuangnya menurunkan F1 makro secara signifikan.

**Status kehamilan pasien Pima tidak tercatat.** Aplikasi menganggap pasien
Pima tidak hamil dan pasien GDM hamil, baik untuk rentang input maupun
untuk aturan menghapus kelas Gestasional.

**Complete-case membuang 778 baris**, terutama pasien Non-GDM dari GDM.
Model final tidak lagi diuntungkan oleh pola nilai kosong, tetapi
populasi Non-GDM yang tersisa mungkin tidak mewakili seluruh pasien
Non-GDM.

**Asumsi independensi tidak sepenuhnya terpenuhi.** Korelasi `Age` dan
`Pregnancies` 0,54 pada data Pima.

**Probabilitas belum terkalibrasi.** Brier score multikelas 0,2347 (out-of-fold).

**Belum ada validasi eksternal.** Model belum diuji pada populasi lain,
misalnya pasien Indonesia. Uji seperti itu memerlukan data baru dan belum
dilakukan; hasilnya tidak disimulasikan.

**Hanya untuk perempuan dan bukan alat diagnosis.** Kedua dataset hanya
berisi pasien perempuan, dan glukosa yang dipakai adalah hasil TTGO 2 jam.

## Riwayat Skema Sebelumnya

Versi ini tidak lagi menyimpan kode skema `GlucoseRisk` dan
`InsulinResistance` (revisi 1 dan 2), `main.py`, paket `src/`, maupun
berkas `models/feature_columns.json` dan `models/imputation_medians.json`.
Kode dan berkas lama masih ada di riwayat commit git. Skema imputasi lama
hanya dipakai sebagai pembanding di `notebooks/04_Evaluation.ipynb`.
