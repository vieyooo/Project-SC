"""
app.py — Sistem Diagnostik Suara Mesin Motor Matic
Menggunakan CNN-1D + MFCC untuk deteksi kerusakan komponen
"""

import time
import pickle
import tempfile
import os

import numpy as np
import librosa
import streamlit as st
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MotoScan — Diagnostik Suara Mesin",
    page_icon="🔧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS — Palet Warna Baru (Eucalyptus, Mist, Plaster, Soot, Moss)
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

/* ── Definisi Variabel Warna ── */
:root {
  --eucalyptus: #98AA9D;
  --mist: #B3C9D6;
  --plaster: #F2EFE2;
  --soot: #2D3536;
  --moss: #697C70;
}

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background Utama ── */
.stApp {
    background-color: var(--soot);
}

/* ── Sembunyikan elemen default Streamlit ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 780px;
}

/* ── Hero / Header ── */
.hero-wrapper {
    background: linear-gradient(135deg, var(--moss) 0%, var(--soot) 100%);
    border: 1px solid var(--eucalyptus);
    border-radius: 20px;
    padding: 2.4rem 2.8rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}
.hero-badge {
    display: inline-block;
    background: var(--eucalyptus);
    color: var(--soot);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.3rem 0.85rem;
    border-radius: 99px;
    margin-bottom: 1rem;
    border: 1px solid var(--mist);
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.3rem;
    color: var(--plaster);
    line-height: 1.2;
    margin: 0 0 0.6rem;
}
.hero-title span {
    color: var(--mist);
}
.hero-desc {
    font-size: 0.97rem;
    color: var(--plaster);
    opacity: 0.85;
    line-height: 1.65;
    max-width: 800px;
    margin: 0 auto;
}

/* ── Section Label ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--mist);
    margin-bottom: 0.5rem;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--plaster);
    margin-bottom: 1rem;
}

/* ── Edu Cards ── */
.edu-card {
    background: var(--eucalyptus);
    border-radius: 16px;
    padding: 1.5rem 1.6rem;
    border: 1px solid var(--mist);
    height: 100%;
}
.edu-card-icon {
    font-size: 1.8rem;
    margin-bottom: 0.8rem;
    color: var(--soot);
}
.edu-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--soot);
    margin-bottom: 0.5rem;
}
.edu-card-body {
    font-size: 0.875rem;
    color: var(--soot);
    opacity: 0.85;
    line-height: 1.6;
}
.edu-tag {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 99px;
    margin-bottom: 0.7rem;
}
.tag-yellow { background: var(--moss); color: var(--plaster); border: 1px solid var(--mist); }
.tag-blue   { background: var(--soot); color: var(--plaster); border: 1px solid var(--mist); }

/* ── Upload Zone ── */
.upload-wrapper {
    background: var(--soot);
    border: 2px dashed var(--eucalyptus);
    border-radius: 16px;
    padding: 1.8rem 1.5rem;
    text-align: center;
    margin-bottom: 1.2rem;
    transition: border-color 0.2s;
}
.upload-wrapper:hover { border-color: var(--plaster); }

/* ── Divider ── */
.soft-divider {
    border: none;
    border-top: 1px solid var(--eucalyptus);
    margin: 1.5rem 0;
}

/* ── Result Cards ── */
.result-banner {
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
}
.result-banner-tensioner {
    background: linear-gradient(135deg, var(--moss), var(--soot));
    border: 1.5px solid var(--mist);
}
.result-banner-gardan {
    background: linear-gradient(135deg, var(--eucalyptus), var(--soot));
    border: 1.5px solid var(--mist);
}
.result-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    color: var(--mist);
}
.result-class {
    font-family: 'DM Serif Display', serif;
    font-size: 1.85rem;
    color: var(--plaster);
    margin: 0;
}
.confidence-pill {
    display: inline-block;
    font-size: 0.82rem;
    font-weight: 700;
    padding: 0.3rem 0.85rem;
    border-radius: 99px;
    margin-top: 0.6rem;
    background: var(--soot);
    border: 1px solid var(--mist);
    color: var(--plaster);
}

/* ── Insight Sections ── */
.insight-block {
    background: var(--eucalyptus);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    border: 1px solid var(--mist);
    margin-bottom: 0.9rem;
}
.insight-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.6rem;
}
.insight-icon {
    font-size: 1.1rem;
}
.insight-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--soot);
}
.insight-body {
    font-size: 0.855rem;
    color: var(--soot);
    opacity: 0.9;
    line-height: 1.65;
}
.insight-body ul {
    margin: 0.4rem 0 0 1rem;
    padding: 0;
}
.insight-body li { margin-bottom: 0.3rem; }

/* ── Footer ── */
.app-footer {
    text-align: center;
    font-size: 0.78rem;
    color: var(--mist);
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--eucalyptus);
}

/* ── Streamlit widget overrides ── */
div[data-testid="stFileUploader"] > label { display: none; }
div[data-testid="stFileUploader"] section {
    background: transparent;
    border: none;
    padding: 0;
}
.stButton > button {
    background: var(--moss);
    color: var(--plaster);
    font-size: 0.92rem;
    font-weight: 600;
    border: 1px solid var(--mist);
    border-radius: 12px;
    padding: 0.75rem 2rem;
    width: 100%;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s;
    letter-spacing: 0.01em;
}
.stButton > button:hover {
    background: var(--mist);
    color: var(--soot);
    border-color: var(--plaster);
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }

/* ── Confidence Bar ── */
.conf-bar-wrapper {
    background: var(--soot);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
    margin-top: 0.6rem;
    border: 1px solid var(--eucalyptus);
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}
.bar-orange { background: linear-gradient(90deg, var(--eucalyptus), var(--plaster)); }
.bar-blue   { background: linear-gradient(90deg, var(--moss), var(--plaster)); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# KONFIGURASI MODEL
# ─────────────────────────────────────────────────────────
SR               = 22050
N_MFCC           = 40
N_FFT            = 2048
HOP_LENGTH       = 512
TARGET_DURATION  = 3.0
MODEL_PATH       = "model_motor_matic_cnn1d.h5"
SCALER_PATH      = "scaler_motor_matic.pkl"
CONFIG_PATH      = "model_config.pkl"


# ─────────────────────────────────────────────────────────
# FUNGSI UTILITAS AUDIO
# ─────────────────────────────────────────────────────────
def load_audio_fixed(path, sr=SR, duration=TARGET_DURATION):
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        target_len = int(sr * duration)
        if len(y) >= target_len:
            start = (len(y) - target_len) // 2
            y = y[start: start + target_len]
        else:
            pad = target_len - len(y)
            y = np.pad(y, (pad // 2, pad - pad // 2), mode='constant')
        return y.astype(np.float32)
    except Exception:
        return None

def extract_mfcc_temporal_with_delta(y_signal, sr=SR, n_mfcc=N_MFCC,
                                      n_fft=N_FFT, hop_length=HOP_LENGTH):
    min_samples = n_fft
    if len(y_signal) < min_samples:
        y_signal = np.pad(y_signal, (0, min_samples - len(y_signal)), mode='constant')

    mfcc = librosa.feature.mfcc(y=y_signal, sr=sr, n_mfcc=n_mfcc,
                                 n_fft=n_fft, hop_length=hop_length)
    T = mfcc.shape[1]
    delta_width = min(9, T)
    if delta_width % 2 == 0:
        delta_width = max(3, delta_width - 1)
    if T < 3:
        mfcc = np.pad(mfcc, ((0, 0), (0, 3 - T)), mode='edge')
        delta_width = 3

    delta  = librosa.feature.delta(mfcc, width=delta_width)
    delta2 = librosa.feature.delta(mfcc, width=delta_width, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0).T
    return features.astype(np.float32)

def predict_audio(wav_path, model, scaler, max_len):
    y_sig = load_audio_fixed(wav_path)
    if y_sig is None:
        return None

    feat = extract_mfcc_temporal_with_delta(y_sig)

    T_actual, F = feat.shape
    if T_actual >= max_len:
        feat = feat[:max_len, :]
    else:
        feat = np.pad(feat, ((0, max_len - T_actual), (0, 0)), mode='constant')

    feat_norm = scaler.transform(feat.reshape(-1, F)).reshape(1, max_len, F)

    prob  = float(model.predict(feat_norm, verbose=0)[0][0])
    label = 1 if prob >= 0.5 else 0
    confidence = max(prob, 1 - prob) * 100

    return {
        "label":      label,
        "kelas":      "Tensioner Aus" if label == 0 else "Gardan — Gigi Rasio Terkikis",
        "prob_raw":   prob,
        "confidence": confidence,
    }


# ─────────────────────────────────────────────────────────
# LOAD MODEL & SCALER
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    try:
        model  = load_model(MODEL_PATH, compile=False)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "rb") as f:
                config = pickle.load(f)
            max_len = config.get("MAX_LEN", 130)
        else:
            max_len = 130
        return model, scaler, max_len, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, MAX_LEN, load_error = load_resources()


# ═════════════════════════════════════════════════════════
# HERO HEADER
# ═════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-badge">🎧 Sistem Diagnostik Audio</div>
    <h1 class="hero-title">Moto<span>Scan</span></h1>
    <p class="hero-desc">
        Unggah rekaman suara mesin motor matic Anda, dan sistem kami akan menganalisis
        apakah terdapat tanda-tanda kerusakan pada <strong>Tensioner</strong> atau
        <strong>Gardan</strong> — dalam hitungan detik, tanpa perlu ke bengkel terlebih dahulu.
    </p>
</div>
""", unsafe_allow_html=True)


# Keunggulan Singkat
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0.5rem;">
        <div style="font-size:1.5rem; margin-bottom:0.4rem;">⚡</div>
        <div style="font-size:0.82rem; font-weight:600; color:var(--plaster);">Analisis Cepat</div>
        <div style="font-size:0.78rem; color:var(--mist); margin-top:0.2rem;">Hasil dalam &lt; 5 detik</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0.5rem;">
        <div style="font-size:1.5rem; margin-bottom:0.4rem;">📊</div>
        <div style="font-size:0.82rem; font-weight:600; color:var(--plaster);">Akurasi Tinggi</div>
        <div style="font-size:0.78rem; color:var(--mist); margin-top:0.2rem;">Berbasis Deep Learning</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0.5rem;">
        <div style="font-size:1.5rem; margin-bottom:0.4rem;">📖</div>
        <div style="font-size:0.82rem; font-weight:600; color:var(--plaster);">Panduan Lengkap</div>
        <div style="font-size:0.78rem; color:var(--mist); margin-top:0.2rem;">Saran servis & perawatan</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)


# KARTU EDUKASI
st.markdown('<p class="section-label">Kenali Komponen Motor Matic Anda</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Apa yang Dideteksi oleh MotoScan?</p>', unsafe_allow_html=True)

edu_col1, edu_col2 = st.columns(2, gap="medium")

with edu_col1:
    st.markdown("""
    <div class="edu-card">
        <div class="edu-card-icon">⚙️</div>
        <span class="edu-tag tag-yellow">Label 0 — Tensioner</span>
        <div class="edu-card-title">Tensioner (Peregang Rantai CVT)</div>
        <div class="edu-card-body">
            Tensioner adalah komponen kecil yang bertugas menjaga ketegangan rantai timing
            di dalam mesin. Ketika komponen ini mulai <strong>aus atau melemah</strong>,
            rantai jadi kendor dan menimbulkan suara berisik seperti "kletek-kletek" saat
            mesin dinyalakan — terutama di pagi hari atau saat baru starter.
            <br><br>
            Jika dibiarkan, dapat menyebabkan rantai selip atau bahkan putus, yang
            berujung pada kerusakan mesin yang jauh lebih serius.
        </div>
    </div>
    """, unsafe_allow_html=True)

with edu_col2:
    st.markdown("""
    <div class="edu-card">
        <div class="edu-card-icon">🔧</div>
        <span class="edu-tag tag-blue">Label 1 — Gardan</span>
        <div class="edu-card-title">Gardan (Gigi Rasio / Final Drive)</div>
        <div class="edu-card-body">
            Gardan atau gigi rasio adalah bagian dari sistem transmisi yang meneruskan
            tenaga dari mesin ke roda belakang. Saat gigi-giginya <strong>terkikis atau
            aus</strong>, motor akan mengeluarkan suara "nguing" atau getaran halus yang
            semakin terasa saat berakselerasi atau melaju di kecepatan tertentu.
            <br><br>
            Kondisi ini biasanya disebabkan oleh kurangnya oli gardan atau pemakaian
            jangka panjang tanpa pengecekan rutin.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)


# UPLOAD AREA
st.markdown('<p class="section-label">Langkah 1 — Unggah Rekaman</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Pilih File Audio Mesin Motor Anda</p>', unsafe_allow_html=True)

st.markdown("""
<div style="background:var(--eucalyptus); border-radius:12px; padding:0.9rem 1.2rem;
            margin-bottom:1rem; font-size:0.84rem; color:var(--soot); display:flex;
            align-items:flex-start; gap:0.6rem; border: 1px solid var(--mist);">
    <span>💡</span>
    <span>Untuk hasil terbaik, rekam suara mesin dari jarak ±30 cm saat mesin berjalan
    stasioner (idle). Gunakan format <strong>.wav</strong> dengan durasi minimal 3 detik.</span>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Upload file WAV",
    type=["wav"],
    help="Format yang didukung .wav",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    st.markdown("""
    <div style="background:var(--mist); border:1px solid var(--plaster); border-radius:12px;
                padding:0.75rem 1.1rem; margin-bottom:0.8rem; font-size:0.85rem;
                color:var(--soot); display:flex; align-items:center; gap:0.5rem; font-weight:600;">
        <span>✓</span>
        <span>File berhasil diunggah. Periksa audio di bawah sebelum dianalisis.</span>
    </div>
    """, unsafe_allow_html=True)

    st.audio(uploaded_file, format="audio/wav")

    file_size_kb = round(len(uploaded_file.getvalue()) / 1024, 1)
    st.markdown(f"""
    <div style="font-size:0.78rem; color:var(--mist); margin-top:0.3rem; margin-bottom:1rem;">
        📁 {uploaded_file.name} &nbsp;·&nbsp; {file_size_kb} KB
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)


# TOMBOL ANALISIS & PREDIKSI
if uploaded_file is not None:
    st.markdown('<p class="section-label">Langkah 2 — Mulai Analisis</p>', unsafe_allow_html=True)

    if load_error:
        st.markdown(f"""
        <div style="background:var(--moss); border:1px solid var(--mist); border-radius:12px;
                    padding:1rem 1.2rem; color:var(--plaster); font-size:0.86rem;">
            ⚠️ <strong>Model belum tersedia.</strong><br>
            Pastikan file <code>model_motor_matic_cnn1d.h5</code> dan
            <code>scaler_motor_matic.pkl</code> berada di direktori yang sama dengan
            <code>app.py</code>.<br><br>
            <em>Detail: {load_error}</em>
        </div>
        """, unsafe_allow_html=True)
    else:
        run_analysis = st.button("🔍 Analisis Suara Mesin", use_container_width=True)

        if run_analysis:
            with st.spinner(""):
                loading_placeholder = st.empty()
                loading_placeholder.markdown("""
                <div style="background:var(--eucalyptus); border:1px solid var(--mist); border-radius:16px;
                            padding:2rem; text-align:center; margin:1rem 0;">
                    <div style="font-size:2.2rem; margin-bottom:0.8rem;">🎧</div>
                    <div style="font-size:1rem; font-weight:600; color:var(--soot); margin-bottom:0.4rem;">
                        Sistem sedang mendengarkan dan menganalisis suara mesin...
                    </div>
                    <div style="font-size:0.86rem; color:var(--soot);">
                        Mohon tunggu sebentar, proses ini hanya memerlukan beberapa detik.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(2)

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                result = predict_audio(tmp_path, model, scaler, MAX_LEN)
                os.unlink(tmp_path)

            loading_placeholder.empty()

            if result is None:
                st.markdown("""
                <div style="background:var(--moss); border:1px solid var(--mist); border-radius:12px;
                            padding:1rem 1.2rem; color:var(--plaster); font-size:0.86rem;">
                    ⚠️ File audio tidak dapat diproses. Pastikan file tidak rusak dan berdurasi minimal 1 detik.
                </div>
                """, unsafe_allow_html=True)
            else:
                label = result["label"]
                kelas = result["kelas"]
                confidence = result["confidence"]
                conf_pct = f"{confidence:.1f}%"
                conf_int = int(confidence)

                st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
                st.markdown('<p class="section-label">Hasil Analisis</p>', unsafe_allow_html=True)

                if label == 0:
                    banner_class = "result-banner-tensioner"
                    bar_class = "bar-orange"
                    icon = "⚙️"
                else:
                    banner_class = "result-banner-gardan"
                    bar_class = "bar-blue"
                    icon = "🔧"

                st.markdown(f"""
                <div class="result-banner {banner_class}">
                    <div class="result-label">Komponen Terdeteksi</div>
                    <p class="result-class">{icon} {kelas}</p>
                    <span class="confidence-pill">Keyakinan {conf_pct}</span>
                    <div class="conf-bar-wrapper" style="margin-top:0.8rem;">
                        <div class="conf-bar-fill {bar_class}" style="width:{conf_int}%;"></div>
                    </div>
                    <div style="font-size:0.75rem; color:var(--mist); margin-top:0.3rem;">
                        {conf_int}% keyakinan sistem terhadap hasil ini
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <p style="font-size:1rem; font-weight:700; color:var(--plaster); margin:1.2rem 0 0.8rem;">
                    📋 Apa Artinya Ini untuk Motor Anda?
                </p>
                """, unsafe_allow_html=True)

                if label == 0:
                    st.markdown("""
                    <div class="insight-block">
                        <div class="insight-header">
                            <span class="insight-icon">🔍</span>
                            <span class="insight-title">Mengapa Tensioner Bisa Aus?</span>
                        </div>
                        <div class="insight-body">
                            Tensioner bekerja terus-menerus menahan tekanan rantai mesin setiap saat motor digunakan. Penyebab utama keausan:
                            <ul>
                                <li>Pemakaian motor intensif tanpa servis rutin</li>
                                <li>Oli mesin jarang diganti atau kualitas rendah</li>
                                <li>Mesin sering digeber dalam kondisi dingin</li>
                                <li>Usia pemakaian melampaui batas servis (24.000-30.000 km)</li>
                            </ul>
                        </div>
                    </div>

                    <div class="insight-block">
                        <div class="insight-header">
                            <span class="insight-icon">🛠️</span>
                            <span class="insight-title">Tindakan yang Perlu Dilakukan</span>
                        </div>
                        <div class="insight-body">
                            Segera bawa motor ke bengkel resmi. Minta mekanik untuk:
                            <ul>
                                <li>Memeriksa kondisi tensioner dan rantai timing</li>
                                <li>Mengganti tensioner jika sudah aus</li>
                                <li>Memeriksa dan mengganti oli mesin</li>
                            </ul>
                            <strong>Biaya estimasi:</strong> Rp 80.000-250.000 tergantung tipe motor.
                        </div>
                    </div>

                    <div class="insight-block">
                        <div class="insight-header">
                            <span class="insight-icon">🛡️</span>
                            <span class="insight-title">Tips Pencegahan</span>
                        </div>
                        <div class="insight-body">
                            <ul>
                                <li>Ganti oli mesin rutin setiap 2.000-3.000 km</li>
                                <li>Panaskan mesin 1-2 menit sebelum digunakan</li>
                                <li>Servis berkala setiap 6 bulan atau 6.000 km</li>
                            </ul>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown("""
                    <div class="insight-block">
                        <div class="insight-header">
                            <span class="insight-icon">🔍</span>
                            <span class="insight-title">Mengapa Gardan Bisa Terkikis?</span>
                        </div>
                        <div class="insight-body">
                            Penyebab utama kerusakan gardan:
                            <ul>
                                <li>Oli gardan jarang atau tidak pernah diganti</li>
                                <li>Kebocoran oli karena seal rusak</li>
                                <li>Muatan berlebihan atau sering menanjak</li>
                                <li>Pemakaian motor lebih dari 4-5 tahun tanpa pengecekan</li>
                            </ul>
                        </div>
                    </div>

                    <div class="insight-block">
                        <div class="insight-header">
                            <span class="insight-icon">🛠️</span>
                            <span class="insight-title">Tindakan yang Perlu Dilakukan</span>
                        </div>
                        <div class="insight-body">
                            Segera periksakan ke bengkel. Minta mekanik untuk:
                            <ul>
                                <li>Memeriksa kondisi gardan secara visual</li>
                                <li>Mengganti oli gardan (Rp 15.000-40.000)</li>
                                <li>Jika aus parah, ganti unit gardan (Rp 150.000-500.000)</li>
                            </ul>
                        </div>
                    </div>

                    <div class="insight-block">
                        <div class="insight-header">
                            <span class="insight-icon">🛡️</span>
                            <span class="insight-title">Tips Pencegahan</span>
                        </div>
                        <div class="insight-body">
                            <ul>
                                <li>Ganti oli gardan setiap 10.000 km atau 1 tahun sekali</li>
                                <li>Waspadai kebocoran oli di sekitar roda belakang</li>
                                <li>Hindari beban berlebih dan berkendara agresif</li>
                            </ul>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div style="background:var(--eucalyptus); border-radius:12px; padding:1rem 1.2rem;
                            font-size:0.82rem; color:var(--soot); margin-top:0.5rem;
                            border: 1px solid var(--mist);">
                    <strong>⚠️ Catatan Penting</strong> Hasil analisis ini bersifat informatif.
                    Untuk diagnosis pasti, konsultasikan dengan mekanik berpengalaman.
                </div>
                """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background:var(--soot); border:2px dashed var(--eucalyptus); border-radius:16px;
                padding:2.5rem; text-align:center; color:var(--plaster);">
        <div style="font-size:2rem; margin-bottom:0.6rem;">📁</div>
        <div style="font-size:0.92rem; font-weight:500; margin-bottom:0.3rem;">
            Belum ada file yang diunggah
        </div>
        <div style="font-size:0.82rem; color:var(--mist);">
            Unggah file .wav di atas untuk memulai analisis
        </div>
    </div>
    """, unsafe_allow_html=True)


# FOOTER
st.markdown("""
<div class="app-footer">
    MotoScan · Sistem Diagnostik Audio Motor Matic<br>
    Menggunakan CNN-1D + MFCC · Dibuat untuk pemilik motor Indonesia
</div>
""", unsafe_allow_html=True)