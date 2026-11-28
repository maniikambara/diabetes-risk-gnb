"""Aplikasi cek kategori diagnosis diabetes (Gaussian Naive Bayes).

Jalankan:
    streamlit run app_streamlit.py

Model dan rentang data latih diambil dari folder models/ hasil notebook 03,
metrik dari outputs/ hasil notebook 04.
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

KELAS = ["Normal / Non-GDM", "Tipe 2", "Gestasional"]
WARNA = {"Normal / Non-GDM": "#1baf7a", "Tipe 2": "#eb6834", "Gestasional": "#2a78d6"}
KETERANGAN = {
    "Normal / Non-GDM": "Profil pemeriksaan paling mirip dengan pasien yang tidak didiagnosis diabetes "
                        "(Normal) atau pasien hamil yang tidak mengalami diabetes gestasional (Non-GDM).",
    "Tipe 2": "Profil pemeriksaan paling mirip dengan pasien diabetes tipe 2.",
    "Gestasional": "Profil pemeriksaan paling mirip dengan pasien diabetes saat kehamilan.",
}
SARAN = {
    "Normal / Non-GDM": "Tetap jaga pola makan dan aktivitas fisik, serta lakukan cek gula darah rutin.",
    "Tipe 2": "Sebaiknya lakukan pemeriksaan lanjutan (gula darah puasa atau HbA1c) bersama dokter.",
    "Gestasional": "Sebaiknya konsultasikan hasil tes glukosa dengan dokter kandungan atau bidan.",
}

# Batas widget hanya batas fisik yang wajar. Rentang data latih dibaca dari models/info_model.json.
FIELD = {
    "Age": dict(label="Usia (tahun)", min=10, max=100, awal=30, step=1),
    "Pregnancies": dict(label={"Tidak": "Jumlah kehamilan seumur hidup (kali)",
                               "Ya": "Kehamilan ke berapa saat ini (termasuk yang sekarang)"},
                        min=0, max=20, awal=1, step=1),
    "BMI": dict(label="Indeks massa tubuh atau BMI (kg/m²)", min=10.0, max=80.0, awal=25.0, step=0.1),
    "BloodPressure": dict(label="Tekanan darah diastolik (mmHg)", min=20, max=150, awal=75, step=1),
    "Glucose": dict(label="Glukosa 2 jam setelah TTGO (mg/dL)", min=1, max=600, awal=120, step=1,
                    bantuan="Kadar glukosa darah 2 jam setelah minum larutan glukosa 75 g (tes toleransi "
                            "glukosa oral). Jangan diisi dengan gula darah puasa atau gula darah sewaktu."),
}
NAMA_PENDEK = {"Age": "Usia", "Pregnancies": "Kehamilan", "BMI": "BMI", "BloodPressure": "Tekanan diastolik",
               "Glucose": "Glukosa 2 jam"}
NAMA_PENYEIMBANGAN = {"oversampling": "random oversampling", "undersampling": "random undersampling", "SMOTE": "SMOTE"}
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

.catatan { background: #fff7e6; border: 1px solid #f3dfb0; color: #7a5200; border-radius: 12px;
  padding: .6rem .85rem; font-size: .84rem; margin: .2rem 0 .8rem; }
.catatan-kecil { color: var(--tinta-3); font-size: .8rem; margin: .4rem 0 0; }

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
    info = json.loads((MODELS / "info_model.json").read_text(encoding="utf-8"))
    return model, info


@st.cache_data(show_spinner=False)
def muat_contoh():
    """Satu pasien nyata per kelas dari data bersih. Pasien GDM dianggap sedang hamil."""
    path = BASE / "data" / "processed" / "merged_diagnosis.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    contoh = {}
    for k in KELAS:
        baris = df[df["DiagnosisType"] == k].iloc[0]
        contoh[k] = baris[list(FIELD)].to_dict() | {"hamil": "Ya" if baris["SumberData"] == "GDM" else "Tidak"}
    return contoh


@st.cache_data(show_spinner=False)
def muat_metrik():
    path = OUTPUTS / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---------- 3. Perhitungan ----------


def angka(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def ribuan(n):
    return f"{n:,}".replace(",", ".")


def angka_bebas(x):
    """Bilangan bulat tanpa desimal, selain itu satu desimal."""
    return f"{x:.0f}" if float(x).is_integer() else angka(x, 1)


def isi_form(nilai):
    for k, f in FIELD.items():
        v = type(f["awal"])(nilai.get(k, f["awal"]))
        st.session_state[f"in_{k}"] = min(max(v, f["min"]), f["max"])
    st.session_state["hamil"] = nilai.get("hamil", "Tidak")


def sesuaikan_kehamilan(proba, hamil):
    """Pasien yang tidak hamil tidak mungkin diabetes gestasional: posterior Gestasional jadi 0,
    lalu sisa kelas dinormalkan ulang."""
    p = dict(proba)
    if hamil == "Tidak":
        p["Gestasional"] = 0.0
        total = sum(p.values())
        p = {k: v / total for k, v in p.items()}
    return p


def periksa_input(nilai, hamil, info):
    """Peringatan bila input di luar rentang data latih atau nilai glukosa tidak masuk akal."""
    kondisi = "hamil" if hamil == "Ya" else "tidak_hamil"
    teks_kondisi = "pasien hamil" if hamil == "Ya" else "pasien tidak hamil"
    luar = []
    for k, (bawah, atas) in info["rentang_input"][kondisi].items():
        if not bawah <= nilai[k] <= atas:
            luar.append(f"{NAMA_PENDEK[k]} {angka_bebas(nilai[k])} (data latih {teks_kondisi}: "
                        f"{angka_bebas(bawah)} sampai {angka_bebas(atas)})")
    lain = []
    g = nilai["Glucose"]
    if g < 30:
        lain.append(f"Glukosa {angka_bebas(g)} mg/dL terlalu rendah. Mungkin yang dimasukkan masih dalam mmol/L "
                    "(kalikan 18 untuk mengubah ke mg/dL).")
    elif g < 60:
        lain.append(f"Glukosa 2 jam {angka_bebas(g)} mg/dL jarang ditemui. Pastikan nilainya hasil TTGO 2 jam.")
    elif g > 400:
        lain.append(f"Glukosa {angka_bebas(g)} mg/dL sangat tinggi. Periksa kembali nilainya.")
    return luar, lain


def hitung_manual(model, x):
    """Posterior Gaussian NB dihitung langsung dari prior, mean, dan variansi model."""
    nb = model[-1]
    masukan = model[:-1].transform(x)[0] if len(model) > 1 else x.to_numpy(dtype=float)[0]
    rows = []
    for i, kelas in enumerate(nb.classes_):
        log_lik = -0.5 * np.log(2 * np.pi * nb.var_[i]) - (masukan - nb.theta_[i]) ** 2 / (2 * nb.var_[i])
        rows.append({"Kelas": kelas, "Prior": nb.class_prior_[i],
                     "log P(x|kelas)": log_lik.sum(), "skor": np.log(nb.class_prior_[i]) + log_lik.sum()})
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


def nav_dan_hero(metrik, info):
    jumlah = ribuan(info["jumlah_data"]["bersih"])
    bal = f"{metrik['metrik_uji_akhir']['balanced_accuracy'] * 100:.0f}%" if metrik else "-"
    st.markdown(f"""
<div class="nav">
  <div class="brand"><span class="logo">{IKON['logo']}</span>CekDiabetes</div>
  <div class="tag">Proyek Soft Computing · Gaussian Naive Bayes</div>
</div>
<div class="hero">
  <div class="kecil">Skrining awal berbasis data</div>
  <h1>Kenali kategori diabetes dari 5 data pemeriksaan sederhana</h1>
  <p>Untuk pasien perempuan. Masukkan status kehamilan, usia, jumlah kehamilan, BMI, tekanan darah,
  dan hasil tes glukosa 2 jam. Model menghitung kemungkinan masuk kategori Normal / Non-GDM, Diabetes Tipe 2,
  atau Diabetes Gestasional.</p>
  <div class="statistik">
    <div><b>{jumlah}</b><span>data pasien</span></div>
    <div><b>3</b><span>kategori diagnosis</span></div>
    <div><b>{bal}</b><span>balanced accuracy</span></div>
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


def tampil_hasil(model, info, nilai, hamil):
    x = pd.DataFrame([nilai], columns=info["fitur"])
    proba_model = dict(zip(model.classes_, model.predict_proba(x)[0]))
    proba = sesuaikan_kehamilan(proba_model, hamil)
    pred = max(proba, key=proba.get)
    w = WARNA[pred]

    luar, lain = periksa_input(nilai, hamil, info)
    if luar:
        st.warning("Hasil berikut merupakan ekstrapolasi dan kurang andal, karena ada nilai di luar rentang data "
                   "latih: " + "; ".join(luar) + ".")
    for pesan in lain:
        st.warning(pesan)

    catatan = ("Status tidak hamil, jadi kategori Gestasional tidak dipertimbangkan." if hamil == "Tidak" else
               "Data Tipe 2 pada model hanya berasal dari pasien Pima yang umumnya tidak sedang hamil.")
    st.markdown(
        f'<span class="lencana" style="background:{w}1f;color:{w}">Kategori paling mungkin</span>'
        f'<div class="kategori" style="color:{w}">{pred}</div>'
        f'<p class="ket">{KETERANGAN[pred]}</p>'
        f'{bar_probabilitas(proba)}'
        f'<p class="catatan-kecil">{catatan}</p>'
        f'{ringkasan_pemeriksaan(nilai)}'
        f'<div class="saran">{SARAN[pred]} Hasil ini bukan diagnosis medis.</div>',
        unsafe_allow_html=True)

    with st.expander("Lihat rincian perhitungan Naive Bayes"):
        tabel = hitung_manual(model, x)
        st.caption("skor = log(prior) + jumlah log P(xᵢ | kelas). Posterior = softmax dari skor.")
        format_tabel = {"Prior": angka, "log P(x|kelas)": lambda v: angka(v, 2), "skor": lambda v: angka(v, 2),
                        "Posterior": lambda v: angka(v * 100, 2) + "%"}
        if hamil == "Tidak":
            tabel["Setelah status kehamilan"] = [proba[k] for k in tabel.index]
            format_tabel["Setelah status kehamilan"] = lambda v: angka(v * 100, 2) + "%"
            st.caption("Karena tidak hamil, posterior Gestasional dijadikan 0 lalu dua kelas lain dinormalkan ulang.")
        st.dataframe(tabel.style.format(format_tabel), width="stretch")


def halaman_cek(model, info):
    contoh = muat_contoh()
    kiri, kanan = st.columns([1, 1], gap="large")

    with kiri:
        with st.container(border=True):
            judul_kartu("form", "Data pemeriksaan", "Isi sesuai hasil pemeriksaan terakhir.")
            st.markdown('<div class="catatan">Model ini hanya dilatih dengan data pasien perempuan, '
                        'sehingga hasilnya tidak berlaku untuk laki-laki.</div>', unsafe_allow_html=True)
            if contoh:
                st.caption("Coba dengan contoh pasien dari dataset:")
                cols = st.columns([1.5, 1, 1.3, 0.9])
                for col, k in zip(cols, KELAS):
                    col.button(k, on_click=isi_form, args=(contoh[k],), width="stretch")
                cols[3].button("Reset", on_click=isi_form, args=({},), width="stretch")

            hamil = st.radio("Sedang hamil?", ["Tidak", "Ya"], horizontal=True, key="hamil",
                             help="Kategori Gestasional hanya mungkin untuk pasien yang sedang hamil.")

            with st.form("form_pasien", border=False):
                nilai = {}
                for grup, kolom in KELOMPOK_FIELD.items():
                    st.markdown(f'<div class="grup">{grup}</div>', unsafe_allow_html=True)
                    for k in kolom:
                        if k not in info["fitur"]:
                            continue
                        f = FIELD[k]
                        label = f["label"][hamil] if isinstance(f["label"], dict) else f["label"]
                        nilai[k] = st.number_input(label, min_value=f["min"], max_value=f["max"], step=f["step"],
                                                   key=f"in_{k}", help=f.get("bantuan"))
                kirim = st.form_submit_button("Cek kategori", type="primary", width="stretch")

    with kanan:
        with st.container(border=True):
            judul_kartu("hasil", "Hasil analisis", "Kemungkinan tiap kategori menurut model.")
            if not kirim:
                st.markdown(f'<div class="kosong"><div class="lingkar">{IKON["hasil"]}</div>'
                            '<b>Belum ada hasil</b>Isi data di samping lalu tekan <i>Cek kategori</i>.</div>',
                            unsafe_allow_html=True)
            else:
                tampil_hasil(model, info, nilai, hamil)

    st.markdown("""
<div class="langkah">
  <div><span class="no">1</span><b>Isi data</b><p>Status kehamilan, usia, jumlah kehamilan, BMI, tekanan darah diastolik, dan glukosa 2 jam TTGO.</p></div>
  <div><span class="no">2</span><b>Model menghitung</b><p>Gaussian Naive Bayes menghitung peluang tiap kategori dengan Teorema Bayes.</p></div>
  <div><span class="no">3</span><b>Baca hasilnya</b><p>Kategori dengan peluang terbesar ditampilkan, lengkap dengan ringkasan pemeriksaan.</p></div>
</div>
""", unsafe_allow_html=True)


def halaman_model(m):
    if m is None:
        st.info("Jalankan notebook 04 dulu untuk membuat outputs/metrics.json.")
        return
    mt = m["metrik_uji_akhir"]
    ci = lambda k: f"CI 95%: {angka(mt[k][0])} sampai {angka(mt[k][1])}"
    a, b, c, d = st.columns(4)
    a.metric("Balanced accuracy", angka(mt["balanced_accuracy"]))
    a.caption(ci("balanced_accuracy_ci95"))
    b.metric("F1 makro", angka(mt["f1_makro"]))
    b.caption(ci("f1_makro_ci95"))
    c.metric("Recall Tipe 2", angka(mt["per_kelas"]["Tipe 2"]["recall"]))
    c.caption(ci("recall_tipe2_ci95"))
    d.metric("Akurasi", angka(mt["akurasi"]))
    d.caption(ci("akurasi_ci95"))

    kiri, kanan = st.columns([1, 1], gap="large")
    with kiri:
        with st.container(border=True):
            st.markdown("**Per kategori (data uji akhir)**")
            per = pd.DataFrame(mt["per_kelas"]).T.rename(columns={"f1_score": "f1"})
            st.dataframe(per.style.format({"precision": angka, "recall": angka, "f1": angka,
                                           "support": "{:.0f}"}), width="stretch")
            st.markdown("**Per sumber data**")
            src = pd.DataFrame(m["evaluasi_per_sumber"]).T[["jumlah_data_uji", "balanced_accuracy", "akurasi",
                                                            "baseline_kelas_terbanyak"]]
            src.columns = ["data uji", "balanced acc.", "akurasi", "baseline"]
            st.dataframe(src.style.format({"data uji": "{:.0f}", "balanced acc.": angka, "akurasi": angka,
                                           "baseline": angka}), width="stretch")
    with kanan:
        with st.container(border=True):
            st.markdown("**Confusion matrix**")
            if (OUTPUTS / "confusion_matrices.png").exists():
                st.image(str(OUTPUTS / "confusion_matrices.png"), width="stretch")

    with st.container(border=True):
        st.markdown("**Perbandingan model (cross validation 5 x 10 pada data pengembangan)**")
        banding = pd.DataFrame(m["perbandingan_model_cv"]).T[["balanced_accuracy", "f1_makro", "akurasi",
                                                               "recall_tipe2"]]
        banding.columns = ["balanced acc.", "F1 makro", "akurasi", "recall Tipe 2"]
        st.dataframe(banding.style.format(angka), width="stretch")
        st.caption("Aplikasi memakai GaussianNB final yang dilatih pada data latih seimbang. "
                   "Model lain hanya sebagai pembanding.")

    jd = m["jumlah_data"]
    sumber_cv = m["tebak_sumber_cv"]["tanpa mitigasi"]
    with st.container(border=True):
        st.markdown(f"""
**Data.** Gabungan Pima Indians Diabetes (768 pasien) dan Gestational Diabetes Mellitus (3.525 pasien hamil),
memakai 5 fitur yang ada di keduanya. Dari {ribuan(jd['mentah'])} baris gabungan, baris kembar dan baris yang tidak
lengkap dibuang sehingga tersisa {ribuan(jd['bersih'])} baris.

**Penyeimbangan kelas.** Data latih diseimbangkan dengan {NAMA_PENYEIMBANGAN.get(m['penyeimbangan'], m['penyeimbangan'])} sehingga tiap kategori berisi
{ribuan(m['distribusi_kelas_latih']['sesudah']['Tipe 2'])} baris ({ribuan(jd['pengembangan'])} baris asli menjadi
{ribuan(jd['latih_setelah_seimbang'])} baris). Data uji tidak diseimbangkan, jadi tetap berisi pasien asli.

**Catatan.** Fitur yang sama bisa menebak asal data (Pima atau GDM) dengan akurasi
{angka(sumber_cv['akurasi_tebak_sumber'])}, sedangkan baseline kelas terbanyak {angka(sumber_cv['baseline_sumber_terbanyak'])}.
Jadi sebagian hasil dipengaruhi perbedaan populasi dan status kehamilan, bukan hanya tipe diabetes.
Data Tipe 2 hanya {m['distribusi_kelas_bersih']['Tipe 2']} baris dan seluruhnya berasal dari Pima.
""")


def main():
    st.set_page_config(page_title="CekDiabetes · Skrining Kategori Diabetes", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    berkas_wajib = [MODELS / "gnb_diagnosis.joblib", MODELS / "info_model.json"]
    kurang = [b.relative_to(BASE).as_posix() for b in berkas_wajib if not b.exists()]
    if kurang:
        st.error("Model belum ada: " + ", ".join(kurang) + ". Jalankan notebook 01 sampai 03 dulu.")
        st.stop()

    model, info = muat_model()
    for k in info["fitur"]:
        st.session_state.setdefault(f"in_{k}", FIELD[k]["awal"])
    st.session_state.setdefault("hamil", "Tidak")
    metrik = muat_metrik()

    nav_dan_hero(metrik, info)
    tab_cek, tab_model = st.tabs(["Cek kategori", "Tentang model"])
    with tab_cek:
        halaman_cek(model, info)
    with tab_model:
        halaman_model(metrik)

    st.markdown('<div class="kaki">CekDiabetes dibuat untuk tugas mata kuliah Soft Computing. '
                'Aplikasi ini bukan alat diagnosis. Hasilnya adalah perkiraan model statistik dan tidak '
                'menggantikan pemeriksaan dokter.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
