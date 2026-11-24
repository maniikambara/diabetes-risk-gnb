"""Aplikasi cek kategori diagnosis diabetes (Gaussian Naive Bayes).

Jalankan:
    streamlit run app_streamlit.py

Model diambil dari folder models/ hasil notebook 03, metrik dari outputs/ hasil notebook 04.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------- 1. Konfigurasi ----------

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
OUTPUTS = BASE / "outputs"

KELAS = ["Normal", "Tipe 2", "Gestasional"]
WARNA = {"Normal": "#1baf7a", "Tipe 2": "#eb6834", "Gestasional": "#2a78d6"}
KETERANGAN = {
    "Normal": "Profil pemeriksaan paling mirip dengan pasien yang tidak didiagnosis diabetes.",
    "Tipe 2": "Profil pemeriksaan paling mirip dengan pasien diabetes tipe 2.",
    "Gestasional": "Profil pemeriksaan paling mirip dengan pasien diabetes saat kehamilan.",
}
SARAN = {
    "Normal": "Tetap jaga pola makan dan aktivitas fisik, serta lakukan cek gula darah rutin.",
    "Tipe 2": "Sebaiknya lakukan pemeriksaan lanjutan (gula darah puasa atau HbA1c) bersama dokter.",
    "Gestasional": "Sebaiknya konsultasikan hasil tes glukosa dengan dokter kandungan atau bidan.",
}

FIELD = {
    "Age": dict(label="Usia", satuan="tahun", min=18, max=90, awal=30, step=1),
    "Pregnancies": dict(label="Jumlah kehamilan", satuan="kali", min=0, max=20, awal=1, step=1),
    "BMI": dict(label="Indeks massa tubuh (BMI)", satuan="kg/m²", min=12.0, max=70.0, awal=25.0, step=0.1),
    "BloodPressure": dict(label="Tekanan darah diastolik", satuan="mmHg", min=30, max=140, awal=75, step=1),
    "Glucose": dict(label="Glukosa 2 jam setelah TTGO", satuan="mg/dL", min=40, max=420, awal=120, step=1),
}
KELOMPOK_FIELD = {
    "Data diri": ["Age", "Pregnancies"],
    "Hasil pemeriksaan": ["BMI", "BloodPressure", "Glucose"],
}

IKON = {
    "logo": '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.7s6 6.6 6 11a6 6 0 0 1-12 0c0-4.4 6-11 6-11z"/>'
            '<path d="M9.5 14.5h5M12 12v5"/></svg>',
    "form": '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="3" width="14" height="18" rx="2"/>'
            '<path d="M9 8h6M9 12h6M9 16h4"/></svg>',
    "hasil": '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" '
             'stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l3-8 4 16 3-8h4"/></svg>',
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
  --teal: #0f766e; --teal-tua: #0b5c56; --teal-muda: #e6f4f2;
  --tinta: #10201e; --tinta-2: #4b5a58; --tinta-3: #8a9694;
  --garis: #e3ebe9; --latar: #f6faf9; --putih: #ffffff;
}
html, body, [class*="css"], .stApp, button, input { font-family: "Plus Jakarta Sans", "Segoe UI", system-ui, sans-serif; }
.stApp { background: var(--latar); }
#MainMenu, footer, header[data-testid="stHeader"] { display: none; }
.block-container { max-width: 1120px; padding: 0 1.25rem 3rem 1.25rem; }

/* navigasi */
.nav { display: flex; align-items: center; justify-content: space-between; padding: 1.1rem 0; }
.brand { display: flex; align-items: center; gap: .55rem; font-weight: 800; font-size: 1.15rem; color: var(--tinta); }
.brand .logo { width: 36px; height: 36px; border-radius: 10px; background: var(--teal); color: #fff;
  display: grid; place-items: center; }
.nav .tag { font-size: .8rem; color: var(--tinta-2); background: var(--putih); border: 1px solid var(--garis);
  padding: .35rem .75rem; border-radius: 999px; }

/* hero */
.hero { background: linear-gradient(135deg, #0f766e 0%, #13887f 60%, #1a9e8f 100%); border-radius: 22px;
  padding: 2.4rem 2.4rem 2.1rem; color: #fff; position: relative; overflow: hidden; margin-bottom: 1.6rem; }
.hero::after { content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px;
  border-radius: 50%; background: rgba(255,255,255,.08); }
.hero .kecil { font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; opacity: .85; font-weight: 600; }
.hero h1 { font-size: 2.15rem; line-height: 1.2; font-weight: 800; margin: .5rem 0 .6rem; color: #fff; max-width: 640px; padding: 0; }
.hero p { font-size: 1rem; opacity: .92; max-width: 600px; margin: 0; }
.statistik { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .8rem; margin-top: 1.6rem; max-width: 620px; }
.statistik div { background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.18); border-radius: 14px; padding: .75rem .9rem; }
.statistik b { display: block; font-size: 1.35rem; font-weight: 800; }
.statistik span { font-size: .8rem; opacity: .9; }

/* kartu */
.judul-kartu { display: flex; align-items: center; gap: .5rem; font-weight: 700; color: var(--tinta); font-size: 1.05rem; margin-bottom: .2rem; }
.judul-kartu .ik { width: 32px; height: 32px; border-radius: 9px; background: var(--teal-muda); color: var(--teal); display: grid; place-items: center; }
.sub-kartu { color: var(--tinta-2); font-size: .88rem; margin: 0 0 .6rem 0; }
.grup { font-size: .75rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; color: var(--tinta-3); margin: 1rem 0 .3rem; }
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--putih); border: 1px solid var(--garis) !important; border-radius: 18px !important;
  box-shadow: 0 1px 2px rgba(16,32,30,.04);
}

/* input dan tombol */
div[data-testid="stNumberInput"] label p { font-size: .88rem; color: var(--tinta); font-weight: 500; }
.stButton > button, .stFormSubmitButton > button { border-radius: 10px; font-weight: 600; }
.stButton > button { background: var(--putih); border: 1px solid var(--garis); color: var(--tinta-2); font-size: .85rem; }
.stButton > button:hover { border-color: var(--teal); color: var(--teal); }
.stFormSubmitButton > button { background: var(--teal); border: none; color: #fff; padding: .7rem; }
.stFormSubmitButton > button:hover { background: var(--teal-tua); color: #fff; }

/* hasil */
.kosong { text-align: center; padding: 2.5rem 1rem; color: var(--tinta-3); }
.kosong .lingkar { width: 64px; height: 64px; border-radius: 50%; background: var(--teal-muda); color: var(--teal);
  display: grid; place-items: center; margin: 0 auto .9rem; }
.kosong b { color: var(--tinta); display: block; margin-bottom: .3rem; }
.lencana { display: inline-block; font-size: .78rem; font-weight: 700; padding: .3rem .7rem; border-radius: 999px; }
.kategori { font-size: 2.1rem; font-weight: 800; margin: .5rem 0 .2rem; line-height: 1.1; }
.ket { color: var(--tinta-2); font-size: .92rem; margin: 0 0 1.1rem; }
.baris { display: grid; grid-template-columns: 92px 1fr 54px; align-items: center; gap: .6rem; margin: .5rem 0; font-size: .9rem; color: var(--tinta); }
.baris .rel { background: #eef2f1; border-radius: 999px; height: 9px; overflow: hidden; }
.baris .isi { height: 9px; border-radius: 999px; }
.baris .nilai { text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; }
.saran { background: var(--teal-muda); border-radius: 12px; padding: .8rem 1rem; font-size: .88rem; color: var(--teal-tua); margin-top: 1rem; }
.ringkas { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .6rem; margin-top: 1rem; }
.ringkas div { border: 1px solid var(--garis); border-radius: 12px; padding: .6rem .7rem; }
.ringkas small { color: var(--tinta-3); font-size: .74rem; display: block; }
.ringkas b { font-size: .95rem; color: var(--tinta); display: block; }
.ringkas em { font-style: normal; font-size: .76rem; font-weight: 600; }

/* cara kerja dan footer */
.langkah { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin-top: 1.6rem; }
.langkah div { background: var(--putih); border: 1px solid var(--garis); border-radius: 16px; padding: 1.1rem 1.2rem; }
.langkah .no { width: 30px; height: 30px; border-radius: 50%; background: var(--teal); color: #fff; font-weight: 700;
  display: grid; place-items: center; font-size: .85rem; margin-bottom: .6rem; }
.langkah b { display: block; color: var(--tinta); margin-bottom: .25rem; }
.langkah p { color: var(--tinta-2); font-size: .86rem; margin: 0; }
.kaki { margin-top: 2.2rem; padding-top: 1.2rem; border-top: 1px solid var(--garis); font-size: .8rem; color: var(--tinta-3); }

/* tab dan metrik */
.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--garis); }
.stTabs [data-baseweb="tab"] { font-weight: 600; padding: .6rem .9rem; }
.stTabs [aria-selected="true"] { color: var(--teal) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--teal) !important; }
div[data-testid="stMetric"] { background: var(--putih); border: 1px solid var(--garis); border-radius: 14px; padding: .8rem 1rem; }

/* layar kecil */
@media (max-width: 760px) {
  .block-container { padding: 0 .9rem 2rem .9rem; }
  .hero { padding: 1.6rem 1.3rem; border-radius: 18px; }
  .hero h1 { font-size: 1.55rem; }
  .langkah, .ringkas { grid-template-columns: 1fr; }
  .statistik { gap: .45rem; margin-top: 1.2rem; }
  .statistik div { padding: .55rem .6rem; }
  .statistik b { font-size: 1.1rem; }
  .statistik span { font-size: .7rem; }
  .hero p { font-size: .92rem; }
  .nav .tag { display: none; }
  .kategori { font-size: 1.7rem; }
}
</style>
"""


# ---------- 2. Muat model dan data ----------


@st.cache_resource(show_spinner=False)
def muat_model():
    model = joblib.load(MODELS / "gnb_diagnosis.joblib")
    fitur = json.loads((MODELS / "feature_columns.json").read_text(encoding="utf-8"))
    return model, fitur


@st.cache_data(show_spinner=False)
def muat_contoh():
    """Satu pasien nyata (data lengkap) per kelas dari dataset gabungan."""
    path = BASE / "data" / "processed" / "merged_diagnosis.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path).dropna()
    return {k: df[df["DiagnosisType"] == k].iloc[0][list(FIELD)].to_dict() for k in KELAS}


@st.cache_data(show_spinner=False)
def muat_metrik():
    path = OUTPUTS / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---------- 3. Perhitungan ----------


def angka(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def isi_form(nilai):
    for k, f in FIELD.items():
        v = type(f["awal"])(nilai.get(k, f["awal"]))
        st.session_state[f"in_{k}"] = min(max(v, f["min"]), f["max"])


def hitung_manual(model, x):
    """Posterior Gaussian NB dihitung langsung dari prior, mean, dan variansi model."""
    rows = []
    for i, kelas in enumerate(model.classes_):
        log_lik = -0.5 * np.log(2 * np.pi * model.var_[i]) - (x - model.theta_[i]) ** 2 / (2 * model.var_[i])
        rows.append({"Kelas": kelas, "Prior": model.class_prior_[i],
                     "log P(x|kelas)": log_lik.sum(), "skor": np.log(model.class_prior_[i]) + log_lik.sum()})
    df = pd.DataFrame(rows)
    e = np.exp(df["skor"] - df["skor"].max())
    df["Posterior"] = e / e.sum()
    return df.set_index("Kelas").loc[KELAS]


# Kategori acuan umum: BMI (WHO), tekanan diastolik (ACC/AHA), glukosa 2 jam TTGO (WHO/ADA).
def kategori_bmi(bmi):
    if bmi < 18.5:
        return "Kurang", "#2a78d6"
    if bmi < 25:
        return "Normal", "#0b8a5e"
    if bmi < 30:
        return "Berlebih", "#b46f00"
    return "Obesitas", "#d03b3b"


def kategori_tekanan(dia):
    if dia < 80:
        return "Normal", "#0b8a5e"
    if dia < 90:
        return "Tinggi tahap 1", "#b46f00"
    return "Tinggi tahap 2", "#d03b3b"


def kategori_glukosa(g):
    if g < 140:
        return "Normal", "#0b8a5e"
    if g < 200:
        return "Toleransi terganggu", "#b46f00"
    return "Tinggi", "#d03b3b"


# ---------- 4. Tampilan ----------


def judul_kartu(ikon, judul, sub):
    st.markdown(f'<div class="judul-kartu"><span class="ik">{IKON[ikon]}</span>{judul}</div>'
                f'<p class="sub-kartu">{sub}</p>', unsafe_allow_html=True)


def nav_dan_hero(metrik):
    akurasi = f"{metrik['metrik']['akurasi'] * 100:.0f}%" if metrik else "-"
    st.markdown(f"""
<div class="nav">
  <div class="brand"><span class="logo">{IKON['logo']}</span>CekDiabetes</div>
  <div class="tag">Proyek Soft Computing · Gaussian Naive Bayes</div>
</div>
<div class="hero">
  <div class="kecil">Skrining awal berbasis data</div>
  <h1>Kenali kategori diabetes dari 5 data pemeriksaan sederhana</h1>
  <p>Masukkan usia, jumlah kehamilan, BMI, tekanan darah, dan hasil tes glukosa.
  Model akan menghitung kemungkinan masuk kategori Normal, Diabetes Tipe 2, atau Diabetes Gestasional.</p>
  <div class="statistik">
    <div><b>4.293</b><span>data pasien</span></div>
    <div><b>3</b><span>kategori diagnosis</span></div>
    <div><b>{akurasi}</b><span>akurasi data uji</span></div>
  </div>
</div>
""", unsafe_allow_html=True)


def bar_probabilitas(proba):
    html = ""
    for k in KELAS:
        p = proba[k]
        html += (f'<div class="baris"><span>{k}</span>'
                 f'<div class="rel"><div class="isi" style="width:{p * 100:.1f}%;background:{WARNA[k]}"></div></div>'
                 f'<span class="nilai">{angka(p * 100, 1)}%</span></div>')
    return html


def ringkasan_pemeriksaan(nilai):
    item = [
        ("BMI", angka(nilai["BMI"], 1), *kategori_bmi(nilai["BMI"])),
        ("Diastolik", f"{nilai['BloodPressure']:.0f} mmHg", *kategori_tekanan(nilai["BloodPressure"])),
        ("Glukosa 2 jam", f"{nilai['Glucose']:.0f} mg/dL", *kategori_glukosa(nilai["Glucose"])),
    ]
    isi = "".join(f'<div><small>{n}</small><b>{v}</b><em style="color:{w}">{k}</em></div>' for n, v, k, w in item)
    return f'<div class="ringkas">{isi}</div>'


def halaman_cek(model, fitur):
    contoh = muat_contoh()
    kiri, kanan = st.columns([1, 1], gap="large")

    with kiri:
        with st.container(border=True):
            judul_kartu("form", "Data pemeriksaan", "Isi sesuai hasil pemeriksaan terakhir.")
            if contoh:
                st.caption("Coba dengan contoh pasien dari dataset:")
                cols = st.columns([1, 1, 1.3, 0.9])
                for col, k in zip(cols, KELAS):
                    col.button(k, on_click=isi_form, args=(contoh[k],), width="stretch")
                cols[3].button("Reset", on_click=isi_form, args=({},), width="stretch")

            with st.form("form_pasien", border=False):
                nilai = {}
                for grup, kolom in KELOMPOK_FIELD.items():
                    st.markdown(f'<div class="grup">{grup}</div>', unsafe_allow_html=True)
                    for k in kolom:
                        f = FIELD[k]
                        nilai[k] = st.number_input(f"{f['label']} ({f['satuan']})", min_value=f["min"],
                                                   max_value=f["max"], step=f["step"], key=f"in_{k}")
                kirim = st.form_submit_button("Cek kategori", type="primary", width="stretch")

    with kanan:
        with st.container(border=True):
            judul_kartu("hasil", "Hasil analisis", "Kemungkinan tiap kategori menurut model.")
            if not kirim:
                st.markdown(f'<div class="kosong"><div class="lingkar">{IKON["hasil"]}</div>'
                            '<b>Belum ada hasil</b>Isi data di samping lalu tekan <i>Cek kategori</i>.</div>',
                            unsafe_allow_html=True)
            else:
                x = pd.DataFrame([nilai], columns=fitur)
                proba = dict(zip(model.classes_, model.predict_proba(x)[0]))
                pred = max(proba, key=proba.get)
                w = WARNA[pred]

                st.markdown(
                    f'<span class="lencana" style="background:{w}1f;color:{w}">Kategori paling mungkin</span>'
                    f'<div class="kategori" style="color:{w}">{pred}</div>'
                    f'<p class="ket">{KETERANGAN[pred]}</p>'
                    f'{bar_probabilitas(proba)}'
                    f'{ringkasan_pemeriksaan(nilai)}'
                    f'<div class="saran">{SARAN[pred]} Hasil ini bukan diagnosis medis.</div>',
                    unsafe_allow_html=True)

                with st.expander("Lihat rincian perhitungan Naive Bayes"):
                    tabel = hitung_manual(model, x.to_numpy()[0].astype(float))
                    st.caption("skor = log(prior) + jumlah log P(xᵢ | kelas). Posterior = softmax dari skor.")
                    st.dataframe(tabel.style.format({"Prior": angka, "log P(x|kelas)": lambda v: angka(v, 2),
                                                      "skor": lambda v: angka(v, 2),
                                                      "Posterior": lambda v: angka(v * 100, 2) + "%"}),
                                 width="stretch")

    st.markdown("""
<div class="langkah">
  <div><span class="no">1</span><b>Isi data</b><p>Lima data: usia, jumlah kehamilan, BMI, tekanan darah diastolik, dan glukosa 2 jam.</p></div>
  <div><span class="no">2</span><b>Model menghitung</b><p>Gaussian Naive Bayes menghitung peluang tiap kategori dengan Teorema Bayes.</p></div>
  <div><span class="no">3</span><b>Baca hasilnya</b><p>Kategori dengan peluang terbesar ditampilkan, lengkap dengan ringkasan pemeriksaan.</p></div>
</div>
""", unsafe_allow_html=True)


def halaman_model(m):
    if m is None:
        st.info("Jalankan notebook 04 dulu untuk membuat outputs/metrics.json.")
        return
    mt = m["metrik"]
    a, b, c, d = st.columns(4)
    a.metric("Akurasi", angka(mt["akurasi"]))
    b.metric("Precision makro", angka(mt["precision_macro"]))
    c.metric("Recall makro", angka(mt["recall_macro"]))
    d.metric("F1 makro", angka(mt["f1_macro"]))

    kiri, kanan = st.columns([1, 1], gap="large")
    with kiri:
        with st.container(border=True):
            st.markdown("**Per kategori (data uji)**")
            per = pd.DataFrame(mt["per_kelas"]).T.rename(columns={"f1_score": "f1"})
            st.dataframe(per.style.format({"precision": angka, "recall": angka, "f1": angka,
                                           "support": "{:.0f}"}), width="stretch")
            st.markdown("**Akurasi per sumber data**")
            src = pd.DataFrame(m["evaluasi_per_sumber"]).T[["jumlah_data_uji", "akurasi", "baseline_kelas_terbanyak"]]
            src.columns = ["data uji", "akurasi", "baseline"]
            st.dataframe(src.style.format({"data uji": "{:.0f}", "akurasi": angka, "baseline": angka}),
                         width="stretch")
    with kanan:
        with st.container(border=True):
            st.markdown("**Confusion matrix**")
            if (OUTPUTS / "confusion_matrices.png").exists():
                st.image(str(OUTPUTS / "confusion_matrices.png"), width="stretch")

    with st.container(border=True):
        st.markdown(f"""
**Data.** Gabungan Pima Indians Diabetes (768 pasien) dan Gestational Diabetes Mellitus
(3.525 pasien hamil), memakai 5 fitur yang ada di keduanya.

**Catatan.** Model yang sama bisa menebak asal data (Pima atau GDM) dengan akurasi
{angka(m['akurasi_tebak_sumber_data'])}, lebih tinggi dari akurasi menebak diagnosis. Jadi sebagian
hasil dipengaruhi perbedaan populasi dan status kehamilan, bukan hanya tipe diabetes.
Data Tipe 2 juga sedikit (6,2%).
""")


def main():
    st.set_page_config(page_title="CekDiabetes · Skrining Kategori Diabetes", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    berkas_wajib = [MODELS / "gnb_diagnosis.joblib", MODELS / "feature_columns.json"]
    kurang = [b.relative_to(BASE).as_posix() for b in berkas_wajib if not b.exists()]
    if kurang:
        st.error("Model belum ada: " + ", ".join(kurang) + ". Jalankan notebook 01 sampai 03 dulu.")
        st.stop()

    model, fitur = muat_model()
    for k in fitur:
        st.session_state.setdefault(f"in_{k}", FIELD[k]["awal"])
    metrik = muat_metrik()

    nav_dan_hero(metrik)
    tab_cek, tab_model = st.tabs(["Cek kategori", "Tentang model"])
    with tab_cek:
        halaman_cek(model, fitur)
    with tab_model:
        halaman_model(metrik)

    st.markdown('<div class="kaki">CekDiabetes dibuat untuk tugas mata kuliah Soft Computing. '
                'Hasil aplikasi ini adalah perkiraan model statistik, bukan diagnosis dokter.</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
