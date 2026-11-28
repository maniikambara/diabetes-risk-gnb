# Klasifikasi Risiko Diabetes dengan Gaussian Naive Bayes

## 1. Ide (Topik)

Klasifikasi Risiko Diabetes Menggunakan Naive Bayes Berdasarkan Data
Diagnostik Pasien.

## 2. Deskripsi Ide

Proyek ini membangun model **Gaussian Naive Bayes** yang menebak kategori
diagnosis diabetes seorang pasien perempuan: Normal, Tipe 2, atau
Gestasional. Model dipakai di aplikasi Streamlit sederhana.

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
> | Normal | Pima + GDM | `Outcome=0` (Pima) atau `Non-GDM` (GDM) |
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
(balanced accuracy pasien Pima 0,6746). Bagian Hasil menjelaskan
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

Tidak ada data dari sumber lain dan tidak ada data sintetis yang disimpan.
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
| Buang baris kembar persis | 3.938 | 355 baris dibuang, semuanya Normal dari GDM |
| Buang baris tidak lengkap | 3.160 | 778 baris dibuang: 734 Normal GDM, 25 Normal Pima, 19 Tipe 2 Pima |

Baris kembar muncul karena setelah GDM diambil 5 kolomnya, banyak pasien
Non-GDM yang BMI dan OGTT-nya kosong menjadi identik. Sebanyak 493 baris
termasuk kelompok kembar, 477 di antaranya memuat nilai kosong.

Distribusi kelas data bersih: Normal 1.539 (48,7%), Gestasional 1.372
(43,4%), Tipe 2 249 (7,9%). Menurut sumber: Pima 724 baris (475 Normal,
249 Tipe 2), GDM 2.436 baris (1.064 Normal, 1.372 Gestasional).

## 5. Metode Komputasi Soft Computing

Metode yang dipakai adalah Naive Bayes, tepatnya Gaussian Naive Bayes.
Urutannya:

1. Sisihkan 20% data bersih sebagai data uji akhir (632 baris, stratified,
   `random_state=42`). Data ini hanya dipakai sekali di notebook 04.
2. Pada 80% sisanya (data pengembangan, 2.528 baris), bandingkan pilihan
   transformasi, prior, oversampling, dan fitur dengan repeated stratified
   5-fold cross validation yang diulang 10 kali (50 fold). Semua langkah ada
   di dalam Pipeline sehingga hanya di-fit pada fold latih.
3. Hitung prior setiap kelas. Model final memakai prior seragam (1/3 untuk
   setiap kelas) yang dimasukkan secara eksplisit.
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
│   ├── 03_GaussianNB.ipynb          data uji akhir, cross validation, transformasi, prior, ablasi, model final, posterior manual
│   └── 04_Evaluation.ipynb          evaluasi akhir, per sumber, tebak asal data, model pembanding, kalibrasi, dampak perubahan data
├── models/
│   ├── gnb_diagnosis.joblib         model final (Pipeline berisi GaussianNB)
│   └── info_model.json              fitur, prior, dan rentang nilai data latih per kondisi kehamilan
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
| `03_GaussianNB` | cross validation, transformasi, prior, ablasi, model final, posterior manual | `GaussianNBPrior`, `prior_campuran`, `cv_berulang`, `uji_t_terkoreksi`, `buat_model`, `posterior_manual` |
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
kepercayaan, metrik per kelas dan per sumber, confusion matrix, dan tabel
perbandingan model.

## Konfigurasi

| Parameter | Nilai | Lokasi |
| --- | --- | --- |
| `TEST_SIZE` / `RANDOM_STATE` | 0.2 / 42 | `03_GaussianNB` dan `04_Evaluation` (harus sama; `04` memeriksanya) |
| Cross validation | repeated stratified 5-fold, 10 kali ulang | `N_SPLITS`, `N_REPEATS` di `03` dan `04` |
| Nilai kosong | baris dibuang (complete-case), tanpa imputasi | `02_Preprocessing`, `bersihkan` |
| Prior | seragam, alpha = 1 (dipilih dari empiris, seragam, dan tuning) | `03_GaussianNB` bagian 3 |
| Transformasi fitur | tidak ada (dipilih dari tanpa, log1p, Yeo-Johnson) | `03_GaussianNB` bagian 3 |
| Lima fitur | Age, Pregnancies, BMI, BloodPressure, Glucose | `02_Preprocessing`, `SHARED_FEATURES` |
| Bootstrap selang kepercayaan | 2.000 sampel ulang, persentil 2,5 dan 97,5 | `04_Evaluation`, `N_BOOTSTRAP` |

## Hasil

Semua angka di bawah ini berasal dari eksekusi ulang notebook `03` dan
`04` (`random_state=42`) dan tersimpan di `outputs/metrics.json`.

### Model final pada data uji akhir

Data uji akhir: 632 baris (Normal 308, Tipe 2 50, Gestasional 274).

| Metrik | Nilai | Selang kepercayaan 95% |
| --- | --- | --- |
| **Balanced accuracy** | **0,8351** | 0,7911 sampai 0,8790 |
| **F1 makro** | **0,7893** | 0,7440 sampai 0,8286 |
| Akurasi | 0,8608 | 0,8323 sampai 0,8877 |
| Recall Tipe 2 | 0,7600 | 0,6341 sampai 0,8750 |
| F1 Tipe 2 | 0,5802 | 0,4688 sampai 0,6763 |
| Brier score multikelas | 0,2466 | - |

| Kelas | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Normal | 0,9231 | 0,8182 | 0,8675 | 308 |
| Tipe 2 | 0,4691 | 0,7600 | 0,5802 | 50 |
| Gestasional | 0,9137 | 0,9270 | 0,9203 | 274 |

Confusion matrix (baris = aktual Normal, Tipe 2, Gestasional; kolom =
prediksi dengan urutan sama): `[[252, 36, 20], [8, 38, 4], [13, 7, 254]]`.

Cross validation 5 x 10 pada data pengembangan memberi hasil yang sejalan:
balanced accuracy 0,8248 ± 0,0258, F1 makro 0,7828 ± 0,0211, recall
Tipe 2 0,7387 ± 0,0680, F1 Tipe 2 0,5767 ± 0,0445.

### Per sumber data (data uji akhir)

| Sumber | Data uji | Akurasi | Balanced accuracy | F1 makro | Baseline kelas terbanyak |
| --- | --- | --- | --- | --- | --- |
| Pima | 136 | 0,6618 | 0,6823 | 0,6702 | 0,6324 (Normal) |
| GDM | 496 | 0,9153 | 0,9140 | 0,9241 | 0,5524 (Gestasional) |

Model jauh lebih baik pada pasien GDM. Pada pasien Pima, 33 dari 86 pasien
Normal ditebak Tipe 2, dan 5 pasien Pima ditebak Gestasional. Di aplikasi,
5 kasus terakhir ini tertangani oleh input status kehamilan.

### Tebak asal data dan mitigasinya

| Skema (CV 5 x 10) | Akurasi tebak sumber | Baseline sumber terbanyak | Balanced accuracy diagnosis | F1 makro diagnosis |
| --- | --- | --- | --- | --- |
| Tanpa mitigasi | 0,9352 | 0,7674 | 0,8248 | 0,7828 |
| Standardisasi per sumber (fit pada fold latih) | 0,7674 | 0,7674 | 0,6039 | 0,5790 |

Pada data uji akhir, akurasi tebak sumber tanpa mitigasi adalah 0,9446.
Standardisasi per sumber memang menurunkan akurasi tebak sumber sampai
sama dengan baseline, tetapi kemampuan diagnosis ikut turun jauh. Hal ini
terjadi karena Tipe 2 hanya ada di Pima dan Gestasional hanya ada di GDM,
sehingga informasi asal data tidak bisa dipisahkan dari informasi label.
Mitigasi ini tidak dipakai, dan **kebocoran asal data tetap ada**.

### Transformasi dan prior (CV 5 x 10)

Kemiringan fitur pada data pengembangan sebelum transformasi: Age 0,855,
Pregnancies 2,530, BMI 0,611, BloodPressure 0,273, Glucose 1,133. Sesudah
Yeo-Johnson, semuanya di antara -0,021 dan 0,029. Sesudah log1p, nilainya
di antara -0,469 dan 0,221.

| Transformasi | Prior | Balanced accuracy | F1 makro | Recall Tipe 2 |
| --- | --- | --- | --- | --- |
| tanpa | empiris | 0,7718 | 0,7720 | 0,5215 |
| **tanpa** | **seragam (dipakai)** | **0,8248** | **0,7828** | **0,7387** |
| tanpa | tuning (nested CV) | 0,8145 | 0,7793 | 0,7003 |
| log1p | empiris | 0,7451 | 0,7487 | 0,4924 |
| log1p | seragam | 0,8273 | 0,7552 | 0,8357 |
| Yeo-Johnson | empiris | 0,7485 | 0,7534 | 0,4914 |
| Yeo-Johnson | seragam | 0,8335 | 0,7595 | 0,8488 |
| tanpa | SMOTE di dalam fold latih | 0,8286 | 0,7811 | 0,7564 |

Konfigurasi dipilih berdasarkan F1 makro. Transformasi membuat distribusi
lebih normal, tetapi tidak menaikkan F1 makro, jadi tidak dipakai.
Dibanding prior empiris, prior seragam menaikkan balanced accuracy
(+0,0530; p < 0,0001) dan recall Tipe 2 (+0,2173; p < 0,0001). Kenaikan
F1 makro (+0,0108; p = 0,31) tidak signifikan. Nilai p berasal dari uji t
berpasangan terkoreksi Nadeau dan Bengio. SMOTE memberi hasil yang mirip
dengan prior seragam, sehingga tidak dipakai.

### Ablasi fitur (CV 5 x 10, prior seragam)

| Fitur dibuang | Balanced accuracy | p | F1 makro | p | Bal. acc. Pima | Bal. acc. GDM |
| --- | --- | --- | --- | --- | --- | --- |
| (tidak ada) | 0,8248 | - | 0,7828 | - | 0,6746 | 0,9109 |
| Age | 0,8247 | 0,99 | 0,7807 | 0,81 | 0,6369 | 0,9400 |
| Pregnancies | 0,8031 | 0,06 | 0,7517 | 0,0005 | 0,6437 | 0,9051 |
| BMI | 0,7605 | 0,0001 | 0,7503 | 0,016 | 0,6362 | 0,8846 |
| BloodPressure | 0,8175 | 0,34 | 0,7728 | 0,09 | 0,6730 | 0,8981 |
| Glucose | 0,7548 | < 0,0001 | 0,7076 | < 0,0001 | 0,6350 | 0,8072 |

Membuang `Pregnancies` menurunkan F1 makro secara signifikan, jadi fitur ini
**dipertahankan** walaupun artinya berbeda: di Pima berarti jumlah
kehamilan seumur hidup (0 sampai 17), di GDM berarti kehamilan ke berapa
yang sedang dijalani (1 sampai 4). Membuang `Age` tidak mengubah metrik
gabungan secara signifikan, tetapi menurunkan balanced accuracy pasien
Pima (0,6746 menjadi 0,6369), jadi `Age` juga dipertahankan.
Glukosa dan BMI paling berpengaruh.

### Perbandingan model (CV 5 x 10, fitur dan fold yang sama)

| Model | Balanced accuracy | F1 makro | Akurasi | Recall Tipe 2 |
| --- | --- | --- | --- | --- |
| Dummy (kelas terbanyak) | 0,3333 | 0,2183 | 0,4869 | 0,0000 |
| Dummy (acak bertingkat) | 0,3289 | 0,3287 | 0,4259 | 0,0748 |
| **GaussianNB final** | **0,8248** | **0,7828** | **0,8540** | **0,7387** |
| GaussianNB prior empiris | 0,7718 | 0,7720 | 0,8654 | 0,5215 |
| LDA | 0,7323 | 0,7361 | 0,8373 | 0,4587 |
| QDA | 0,7891 | 0,7768 | 0,8551 | 0,6058 |
| Logistic Regression (balanced) | 0,8357 | 0,7721 | 0,8547 | 0,7779 |
| KNN (k=15) | 0,7438 | 0,7621 | 0,8933 | 0,3492 |
| Random Forest (balanced) | 0,8297 | 0,8113 | 0,8981 | 0,6446 |
| Gradient Boosting | 0,7863 | 0,7976 | 0,9039 | 0,4747 |

Algoritma inti tetap Gaussian Naive Bayes; model lain hanya pembanding dan
tidak di-tuning. LDA, QDA, dan Logistic Regression tidak mengasumsikan
fitur saling bebas, tetapi tidak satu pun yang jelas lebih baik pada
kedua metrik utama. Random Forest unggul pada F1 makro (0,8113), tetapi
tidak bisa dihitung ulang secara manual seperti Naive Bayes.

### Kalibrasi probabilitas (out-of-fold, 5-fold, data pengembangan)

| Model | Brier score multikelas | F1 makro | Balanced accuracy |
| --- | --- | --- | --- |
| GaussianNB final (tanpa kalibrasi) | 0,2350 | 0,7838 | 0,8262 |
| Kalibrasi sigmoid | 0,2135 | 0,7783 | 0,7904 |
| Kalibrasi isotonic | 0,2056 | 0,7832 | 0,8073 |

Kalibrasi isotonic memperbaiki Brier score, tetapi menurunkan balanced
accuracy. Kalibrasi tidak dipasang di model final, supaya probabilitas di
aplikasi tetap sama dengan hitungan manual Naive Bayes. Reliability curve
ada di `outputs/kalibrasi.png`. Probabilitas di aplikasi sebaiknya dibaca
sebagai peringkat kemungkinan, bukan peluang yang sudah terkalibrasi.

### Dampak perubahan data (GaussianNB prior empiris, split 80:20 pada datanya masing-masing)

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
karena hampir semuanya Normal. Pada skema complete-case tidak ada lagi
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
| Balanced accuracy uji | 0,7804 | 0,8351 |
| F1 makro uji | 0,7707 | 0,7893 |
| Akurasi uji | 0,8510 | 0,8608 |
| Recall Tipe 2 uji | 0,6667 | 0,7600 |
| Akurasi subset Pima (baseline) | 0,6879 (0,6561) | 0,6618 (0,6324) |
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
selang kepercayaan recall Tipe 2 lebar (0,6341 sampai 0,8750). Precision
Tipe 2 hanya 0,4691: lebih dari separuh tebakan Tipe 2 sebenarnya Normal,
terutama pasien Normal dari Pima.

**Model kurang membantu pasien Pima.** Akurasi pasien Pima 0,6618, hanya
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

**Complete-case membuang 778 baris**, terutama pasien Normal dari GDM.
Model final tidak lagi diuntungkan oleh pola nilai kosong, tetapi
populasi Normal GDM yang tersisa mungkin tidak mewakili seluruh pasien
Non-GDM.

**Asumsi independensi tidak sepenuhnya terpenuhi.** Korelasi `Age` dan
`Pregnancies` 0,54 pada data Pima.

**Probabilitas belum terkalibrasi.** Brier score multikelas 0,2350 (out-of-fold).

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
