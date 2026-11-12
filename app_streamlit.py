"""Aplikasi cek kategori diagnosis diabetes (Gaussian Naive Bayes).

Jalankan:
    streamlit run app_streamlit.py

Model dan median diambil dari folder models/ hasil notebook 03.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
OUTPUTS = BASE / "outputs"

KELAS = ["Normal", "Tipe 2", "Gestasional"]
WARNA = {"Normal": "#2a78d6", "Tipe 2": "#eb6834", "Gestasional": "#1baf7a"}
KETERANGAN = {
    "Normal": "Profil pasien paling mirip dengan pasien tanpa diabetes di data latih.",
    "Tipe 2": "Profil pasien paling mirip dengan pasien diabetes tipe 2 (data Pima).",
    "Gestasional": "Profil pasien paling mirip dengan pasien diabetes saat hamil (data GDM).",
}

FIELD = {
    "Age": dict(label="Usia", satuan="tahun", min=18, max=90, awal=30, step=1),
    "Pregnancies": dict(label="Jumlah kehamilan", satuan="kali", min=0, max=20, awal=1, step=1),
    "BMI": dict(label="BMI", satuan="kg/m²", min=12.0, max=70.0, awal=25.0, step=0.1),
    "BloodPressure": dict(label="Tekanan darah diastolik", satuan="mmHg", min=30, max=140, awal=75, step=1),
    "Glucose": dict(label="Glukosa 2 jam (OGTT)", satuan="mg/dL", min=40, max=420, awal=120, step=1),
}

CSS = """
<style>
#MainMenu, footer, header [data-testid="stToolbar"] {visibility: hidden;}
.block-container {max-width: 1080px; padding-top: 2.2rem;}
html, body, [class*="css"] {font-family: "Segoe UI", system-ui, -apple-system, sans-serif;}
h1.judul {font-size: 1.65rem; font-weight: 700; margin: 0; color: #0b0b0b; letter-spacing: -0.01em;}
p.sub {color: #52514e; margin: .25rem 0 1.4rem 0; font-size: .95rem;}
.kartu {border: 1px solid #e1e0d9; border-radius: 10px; padding: 1.1rem 1.25rem; background: #fcfcfb;}
.label-kecil {font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; color: #898781; margin-bottom: .2rem;}
.hasil {font-size: 2rem !important; font-weight: 700; line-height: 1.15; margin: .1rem 0 0 0;}
.ket {color: #52514e; font-size: .9rem; margin: .35rem 0 1rem 0;}
.baris {display: grid; grid-template-columns: 96px 1fr 52px; align-items: center; gap: .6rem; margin: .45rem 0;}
.baris .nama {font-size: .9rem; color: #0b0b0b;}
.baris .rel {background: #f0efec; border-radius: 4px; height: 10px; overflow: hidden;}
.baris .isi {height: 10px; border-radius: 4px;}
.baris .nilai {font-size: .9rem; text-align: right; font-variant-numeric: tabular-nums; color: #0b0b0b;}
.catatan {font-size: .82rem; color: #52514e; border-left: 3px solid #e1e0d9; padding: .2rem 0 .2rem .7rem; margin-top: 1rem;}
.kosong {color: #898781; font-size: .92rem; padding: 2.2rem 0; text-align: center;}
</style>
"""


@st.cache_resource
def muat_model():
    model = joblib.load(MODELS / "gnb_diagnosis.joblib")
    fitur = json.loads((MODELS / "feature_columns.json").read_text(encoding="utf-8"))
    return model, fitur


@st.cache_data
def muat_contoh():
    """Satu pasien nyata (data lengkap) per kelas dari dataset gabungan."""
    path = BASE / "data" / "processed" / "merged_diagnosis.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path).dropna()
    return {k: df[df["DiagnosisType"] == k].iloc[0][list(FIELD)].to_dict() for k in KELAS}


@st.cache_data
def muat_metrik():
    path = OUTPUTS / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


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


def bar_probabilitas(proba):
    html = ""
    for k in KELAS:
        p = proba[k]
        html += (f'<div class="baris"><div class="nama">{k}</div>'
                 f'<div class="rel"><div class="isi" style="width:{p * 100:.1f}%;background:{WARNA[k]}"></div></div>'
                 f'<div class="nilai">{angka(p * 100, 1)}%</div></div>')
    return html


def halaman_cek(model, fitur):
    contoh = muat_contoh()
    kiri, kanan = st.columns([1, 1.05], gap="large")

    with kiri:
        if contoh:
            st.markdown('<div class="label-kecil">Isi dengan contoh pasien dari dataset</div>',
                        unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns([1, 1, 1.35, 0.9])
            for col, k in zip([c1, c2, c3], KELAS):
                col.button(k, on_click=isi_form, args=(contoh[k],), width="stretch")
            c4.button("Reset", on_click=isi_form, args=({},), width="stretch")

        st.markdown('<div class="label-kecil" style="margin-top:.8rem">Data pasien</div>',
                    unsafe_allow_html=True)
        with st.form("form_pasien", border=False):
            nilai = {}
            for k in fitur:
                f = FIELD[k]
                nilai[k] = st.number_input(f"{f['label']} ({f['satuan']})", min_value=f["min"],
                                           max_value=f["max"], step=f["step"], key=f"in_{k}")
            kirim = st.form_submit_button("Cek kategori", type="primary", width="stretch")

    with kanan:
        if not kirim:
            st.markdown('<div class="kartu"><div class="kosong">Isi data pasien lalu tekan '
                        '<b>Cek kategori</b>.</div></div>', unsafe_allow_html=True)
            return

        x = pd.DataFrame([nilai], columns=fitur)
        proba = dict(zip(model.classes_, model.predict_proba(x)[0]))
        pred = max(proba, key=proba.get)

        st.markdown(
            f'<div class="kartu"><div class="label-kecil">Kategori paling mungkin</div>'
            f'<div class="hasil" style="color:{WARNA[pred]}">{pred}</div>'
            f'<p class="ket">{KETERANGAN[pred]}</p>'
            f'<div class="label-kecil">Probabilitas posterior</div>{bar_probabilitas(proba)}'
            f'<div class="catatan">Hasil ini bukan diagnosis. Untuk pasien yang tidak hamil, pembedaan '
            f'Normal dan Tipe 2 masih kurang akurat.</div></div>',
            unsafe_allow_html=True)

        with st.expander("Rincian perhitungan Naive Bayes"):
            tabel = hitung_manual(model, x.to_numpy()[0].astype(float))
            st.caption("skor = log(prior) + jumlah log P(xᵢ | kelas). Posterior = softmax dari skor.")
            st.dataframe(tabel.style.format({"Prior": "{:.3f}", "log P(x|kelas)": "{:.2f}",
                                              "skor": "{:.2f}", "Posterior": "{:.2%}"}),
                         width="stretch")


def halaman_model():
    m = muat_metrik()
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
        st.markdown("**Per kelas (data uji)**")
        per = pd.DataFrame(mt["per_kelas"]).T.rename(columns={"f1_score": "f1"})
        st.dataframe(per.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}",
                                       "support": "{:.0f}"}), width="stretch")
        st.markdown("**Akurasi per sumber data**")
        src = pd.DataFrame(m["evaluasi_per_sumber"]).T[["jumlah_data_uji", "akurasi", "baseline_kelas_terbanyak"]]
        src.columns = ["data uji", "akurasi", "baseline"]
        st.dataframe(src, width="stretch")
    with kanan:
        if (OUTPUTS / "confusion_matrices.png").exists():
            st.image(str(OUTPUTS / "confusion_matrices.png"), width="stretch")

    st.markdown(
        f"""
**Data.** Gabungan Pima Indians Diabetes (768 pasien) dan Gestational Diabetes Mellitus
(3.525 pasien hamil), memakai 5 fitur yang ada di keduanya.

**Catatan.** Model yang sama bisa menebak asal data (Pima atau GDM) dengan akurasi
{angka(m['akurasi_tebak_sumber_data'])}, lebih tinggi dari akurasi menebak diagnosis. Jadi sebagian
hasil dipengaruhi perbedaan populasi dan status kehamilan, bukan hanya tipe diabetes.
Data Tipe 2 juga sedikit (6,2%).
""")


def main():
    st.set_page_config(page_title="Cek Kategori Diabetes", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    model, fitur = muat_model()
    for k in fitur:
        st.session_state.setdefault(f"in_{k}", FIELD[k]["awal"])

    st.markdown('<h1 class="judul">Cek Kategori Diabetes</h1>'
                '<p class="sub">Gaussian Naive Bayes · Normal, Tipe 2, Gestasional · '
                'data Pima + GDM</p>', unsafe_allow_html=True)

    tab_cek, tab_model = st.tabs(["Cek pasien", "Tentang model"])
    with tab_cek:
        halaman_cek(model, fitur)
    with tab_model:
        halaman_model()


if __name__ == "__main__":
    main()
