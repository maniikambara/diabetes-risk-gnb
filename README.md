# Klasifikasi Risiko Diabetes dengan Gaussian Naive Bayes

## 1. Ide (Topik)

Klasifikasi Risiko Diabetes Menggunakan Naive Bayes Berdasarkan Data
Diagnostik Pasien.

## 2. Deskripsi Ide

Proyek ini membangun model klasifikasi menggunakan algoritma Naive
Bayes, tepatnya **Gaussian Naive Bayes** karena seluruh atribut input
berupa data numerik kontinu, untuk memprediksi apakah seorang pasien
berisiko diabetes atau tidak berdasarkan delapan indikator medis.
Model menghitung probabilitas posterior tiap kelas (diabetes/tidak
diabetes) menggunakan Teorema Bayes dengan asumsi independensi antar
atribut, kemudian memilih kelas dengan probabilitas tertinggi sebagai
hasil prediksi. Proyek juga membandingkan performa model dengan dan
tanpa tahap pembersihan data untuk menunjukkan pengaruh kualitas data
terhadap hasil klasifikasi.

## 3. Analisis Permasalahan

Diagnosis dini diabetes memerlukan pertimbangan banyak indikator
sekaligus, sementara tenaga medis sering perlu estimasi cepat berbasis
data historis sebelum pemeriksaan lanjutan dilakukan. Naive Bayes
cocok untuk kebutuhan ini karena perhitungannya ringan dan cepat
meski data berukuran besar. Namun, permasalahan yang dianalisis pada
proyek ini adalah asumsi independensi antar atribut yang menjadi dasar
Naive Bayes sebenarnya tidak sepenuhnya realistis pada data medis,
karena beberapa indikator seperti Glukosa dan BMI cenderung saling
berkorelasi. Bagian ini dijadikan poin evaluasi kritis kelompok --
menguji seberapa besar pelanggaran asumsi tersebut mempengaruhi
keandalan model dibandingkan sekadar akurasinya (lihat
`notebooks/01_EDA.ipynb` dan `notebooks/04_Evaluation.ipynb`).

## 4. Sumber Data

**Pima Indians Diabetes Database**, berisi 768 baris data pasien
perempuan keturunan Pima Indian berusia minimal 21 tahun, dengan 8
atribut prediktor numerik (`Pregnancies`, `Glucose`, `BloodPressure`,
`SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`)
dan 1 label target biner (`Outcome`: 0 = tidak diabetes, 1 = diabetes).
Data berformat CSV, ukurannya kecil dan sudah bersih secara struktural
sehingga cocok dikerjakan dalam waktu satu semester oleh kelompok
mahasiswa.

Tautan Kaggle: <https://www.kaggle.com/datasets/jamaltariqcheema/pima-indians-diabetes-dataset>

Berkas `data/diabetes.csv` pada proyek ini sudah disertakan. Isinya
diverifikasi identik dengan statistik yang dilaporkan pada dokumen
sumber data (dilakukan otomatis oleh `src/data_loader.py` dan
ditampilkan pada `notebooks/01_EDA.ipynb`): 768 baris, 9 kolom,
distribusi kelas 500 pasien tidak diabetes (65,1%) dan 268 pasien
diabetes (34,9%). Dataset tidak memiliki nilai kosong secara eksplisit,
tetapi lima kolom memiliki nilai 0 yang secara medis merepresentasikan
data hilang:

| Kolom | Jumlah nilai 0 | Persentase |
| --- | --- | --- |
| Glucose | 5 | 0,7% |
| BloodPressure | 35 | 4,6% |
| SkinThickness | 227 | 29,6% |
| Insulin | 374 | 48,7% |
| BMI | 11 | 1,4% |

## 5. Metode Komputasi Soft Computing

Naive Bayes (Gaussian Naive Bayes). Tahapan pengerjaan meliputi
penghitungan probabilitas prior tiap kelas (`P(diabetes)` dan
`P(tidak diabetes)`) dari data latih, estimasi distribusi normal
(mean dan variansi) tiap atribut untuk masing-masing kelas, penerapan
Teorema Bayes untuk menghitung probabilitas posterior
`P(kelas|atribut)` pada data uji, serta evaluasi model menggunakan
metrik akurasi, precision, recall, dan confusion matrix. Seluruh
tahapan ini didemonstrasikan secara eksplisit -- termasuk pembuktian
manual bahwa hasil perhitungan Teorema Bayes identik dengan
`predict_proba()` scikit-learn -- pada `notebooks/03_GaussianNB.ipynb`.

## Struktur Proyek

```
diabetes-risk-gnb/
├── README.md
├── requirements.txt
├── main.py                       jalur CLI: seluruh pipeline dalam satu perintah
├── app_streamlit.py               aplikasi deteksi risiko diabetes langsung
├── data/
│   ├── diabetes.csv               dataset mentah
│   └── processed/
│       └── diabetes_clean.csv     dihasilkan oleh 02_Preprocessing
├── notebooks/
│   ├── 01_EDA.ipynb                eksplorasi data, verifikasi, korelasi atribut
│   ├── 02_Preprocessing.ipynb      penanganan nilai 0 sebagai data hilang, imputasi
│   ├── 03_GaussianNB.ipynb         pelatihan model, pembuktian manual Teorema Bayes
│   └── 04_Evaluation.ipynb         evaluasi, confusion matrix, diskusi kritis
├── src/
│   ├── __init__.py
│   ├── data_loader.py             pemuatan dan verifikasi data
│   ├── preprocessing.py           penanganan nilai 0 sebagai data hilang
│   ├── pipeline.py                split data dan pelatihan GaussianNB
│   └── evaluate.py                confusion matrix dan metrik klasifikasi
├── models/                        dihasilkan oleh 03_GaussianNB
│   ├── gnb_clean.joblib            model utama (dipakai app_streamlit.py)
│   ├── gnb_raw.joblib               model pembanding (data tanpa cleaning)
│   ├── feature_columns.json        urutan kolom fitur saat pelatihan
│   └── imputation_medians.json     median data latih untuk input tidak diketahui
└── outputs/                       dihasilkan oleh notebooks/main.py
    ├── metrics.json
    ├── confusion_matrices.png
    ├── eda_class_distribution.png
    ├── eda_correlation_heatmap.png
    ├── eda_feature_distributions.png
    └── preprocessing_before_after.png
```

Setiap notebook menulis artefak nyata (dataset bersih, model, gambar,
metrik) yang dibaca oleh notebook berikutnya -- bukan sekadar
pembagian kode secara kosmetik. `src/` tetap menjadi satu-satunya
sumber logika inti (pemuatan data, preprocessing, split, evaluasi)
yang dipakai bersama oleh notebook, `main.py`, dan `app_streamlit.py`.

## Instalasi

```
pip install -r requirements.txt
```

## Penggunaan

**Jalur notebook (direkomendasikan, mengikuti alur EDA -> Preprocessing
-> GaussianNB -> Evaluation):**

```
jupyter notebook notebooks/
```

Jalankan keempat notebook secara berurutan (`01_EDA` ->
`02_Preprocessing` -> `03_GaussianNB` -> `04_Evaluation`). Setiap
notebook mengandalkan artefak yang disimpan notebook sebelumnya
(`data/processed/diabetes_clean.csv` dari 02, `models/*.joblib` dari
03), sehingga urutan menjalankannya penting.

**Jalur CLI (satu perintah, tanpa notebook):**

```
python main.py
```

Mencetak ringkasan ke terminal dan menyimpan `outputs/metrics.json`
serta `outputs/confusion_matrices.png`. Berjalan independen dari
notebook (tidak menghasilkan `models/*.joblib`), cocok untuk
reproduksi cepat hasil evaluasi pada tabel di bawah.

**Aplikasi deteksi langsung:**

```
streamlit run app_streamlit.py
```

Membuka form isian delapan indikator medis di browser dan menampilkan
prediksi beserta probabilitas posterior tiap kelas, memakai model
`models/gnb_clean.joblib` yang dihasilkan `notebooks/03_GaussianNB.ipynb`.
Jalankan notebook 01-03 (atau minimal 02 dan 03) sekali sebelum
menjalankan aplikasi ini agar `models/` dan `data/processed/` terisi --
proyek ini sudah menyertakan artefak yang sudah dihasilkan sehingga
aplikasi dapat langsung dijalankan tanpa menjalankan ulang notebook
terlebih dahulu.

## Konfigurasi

Parameter utama didefinisikan sebagai konstanta pada `src/pipeline.py`:

| Parameter | Nilai default | Keterangan |
| --- | --- | --- |
| `TEST_SIZE` | 0.2 | Proporsi data uji pada pembagian stratified |
| `RANDOM_STATE` | 42 | Seed acak agar hasil dapat direproduksi |

Kolom yang diperlakukan sebagai data hilang saat nilainya 0
didefinisikan pada `ZERO_AS_MISSING_COLUMNS` di `src/preprocessing.py`:
Glucose, BloodPressure, SkinThickness, Insulin, BMI.

## Hasil

Hasil berikut diperoleh dari `python main.py` / `notebooks/04_Evaluation.ipynb`
pada dataset yang disertakan, dengan `random_state=42`.

| Metrik | Tanpa Cleaning | Dengan Cleaning |
| --- | --- | --- |
| Akurasi | 0,7078 | 0,7273 |
| Precision (kelas Diabetes) | 0,5738 | 0,6034 |
| Recall (kelas Diabetes) | 0,6481 | 0,6481 |
| F1-score (kelas Diabetes) | 0,6087 | 0,6250 |

Confusion matrix kedua skenario disimpan pada
`outputs/confusion_matrices.png`, dan seluruh angka di atas beserta
korelasi atribut tertinggi disimpan pada `outputs/metrics.json`.

Pembersihan data meningkatkan akurasi dari 70,78% menjadi 72,73% dan
F1-score dari 0,6087 menjadi 0,6250, tanpa mengubah recall. Sebagai
pembanding, model yang selalu memprediksi kelas mayoritas (tidak
diabetes) mencapai akurasi sekitar 65% pada data uji tanpa mendeteksi
satu pun kasus diabetes -- kedua skenario Gaussian Naive Bayes
mengungguli baseline ini, tetapi recall 0,65 berarti sekitar sepertiga
pasien diabetes pada data uji masih terklasifikasi salah sebagai
tidak berisiko.

Korelasi Pearson tertinggi antar pasangan atribut pada data yang sudah
dibersihkan:

| Pasangan Atribut | Korelasi |
| --- | --- |
| BMI dan SkinThickness | 0,566 |
| Age dan Pregnancies | 0,544 |
| Glucose dan Insulin | 0,490 |
| Age dan BloodPressure | 0,325 |
| BMI dan BloodPressure | 0,286 |

## Keterbatasan

**Kebocoran data berskala kecil pada tahap imputasi.** Spesifikasi
tugas menetapkan urutan pembersihan data sebelum pembagian data latih
dan uji. Karena imputasi dilakukan per kelas Outcome sebelum data
dibagi, median yang dipakai untuk mengisi nilai hilang pada baris yang
kelak menjadi data uji ikut dihitung menggunakan label Outcome baris
itu sendiri. Ini adalah kebocoran pada tahap praproses fitur, bukan
kebocoran label langsung ke prediksi, dan pengaruhnya kecil. Untuk
`app_streamlit.py`, keterbatasan ini sengaja tidak dibawa masuk:
`models/imputation_medians.json` dihitung hanya dari data latih tanpa
bergantung pada label kelas (lihat `notebooks/02_Preprocessing.ipynb`,
bagian 5), karena label pasien baru memang belum diketahui saat
aplikasi dipakai.

**Asumsi independensi antar atribut tidak sepenuhnya realistis.**
Naive Bayes mengasumsikan atribut saling bebas secara bersyarat
terhadap kelas. BMI dan SkinThickness berkorelasi 0,566, serta Age
dan Pregnancies berkorelasi 0,544 -- secara medis maupun demografis
masuk akal namun melanggar asumsi ini. Pelanggaran asumsi seperti ini
dapat membuat model terlalu percaya diri pada probabilitas
posteriornya meskipun keputusan kelas akhirnya sering kali masih
cukup akurat. Diskusi kuantitatif lengkap ada pada
`notebooks/04_Evaluation.ipynb`.

**Ketidakseimbangan kelas.** Dari 768 data, 65,1% tidak diabetes dan
34,9% diabetes. Akurasi saja tidak cukup untuk menilai model pada data
yang tidak seimbang seperti ini, karena itu evaluasi pada proyek ini
selalu menyertakan precision, recall, dan F1-score untuk kelas
diabetes, bukan hanya akurasi.

**Imputasi median tidak menangkap ketidakpastian.** Mengisi nilai
hilang dengan satu nilai median menghilangkan variasi yang sebenarnya
ada pada nilai yang hilang tersebut, terutama pada Insulin yang hampir
separuh datanya hilang.

**Belum ada validasi silang.** Hasil pada bagian Hasil berasal dari
satu kali pembagian data latih dan uji dengan `random_state=42`.
Validasi silang k-fold akan memberi estimasi performa yang lebih
stabil dan belum diimplementasikan pada proyek ini.
