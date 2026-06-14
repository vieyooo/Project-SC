"""
app.py — Sistem Diagnostik Suara Mesin Motor Matic
Menggunakan CNN-1D + MFCC untuk deteksi kerusakan komponen:
  Label 0 → Tensioner Aus
  Label 1 → Gardan (Gigi Rasio Terkikis)
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
# CUSTOM CSS — Dark Premium Earth Tones (Tanpa Warna Putih)
# Palette: #442d1c (Dark), #743014 (Red-Brown), #84592b (Mid-Brown), 
#          #9D9167 (Olive/Gold), #e8d1a7 (Beige/Light)
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background Utama (Coklat Gelap) ── */
.stApp {
    background-color: #442d1c;
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
    background: linear-gradient(135deg, #743014 0%, #442d1c 100%);
    border: 1px solid #84592b;
    border-radius: 20px;
    padding: 2.4rem 2.8rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}
.hero-badge {
    display: inline-block;
    background: #84592b;
    color: #e8d1a7;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.3rem 0.85rem;
    border-radius: 99px;
    margin-bottom: 1rem;
    border: 1px solid #9D9167;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.3rem;
    color: #e8d1a7;
    line-height: 1.2;
    margin: 0 0 0.6rem;
}
.hero-title span {
    color: #9D9167;
}
.hero-desc {
    font-size: 0.97rem;
    color: rgba(232, 209, 167, 0.85); /* #e8d1a7 dengan opacity */
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
    color: #9D9167;
    margin-bottom: 0.5rem;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e8d1a7;
    margin-bottom: 1rem;
}

/* ── Edu Cards ── */
.edu-card {
    background: #84592b;
    border-radius: 16px;
    padding: 1.5rem 1.6rem;
    border: 1px solid #9D9167;
    height: 100%;
}
.edu-card-icon {
    font-size: 1.8rem;
    margin-bottom: 0.8rem;
}
.edu-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #e8d1a7;
    margin-bottom: 0.5rem;
}
.edu-card-body {
    font-size: 0.875rem;
    color: rgba(232, 209, 167, 0.85);
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
.tag-yellow { background: #743014; color: #e8d1a7; border: 1px solid #9D9167; }
.tag-blue   { background: #442d1c; color: #e8d1a7; border: 1px solid #9D9167; }

/* ── Upload Zone ── */
.upload-wrapper {
    background: #442d1c;
    border: 2px dashed #9D9167;
    border-radius: 16px;
    padding: 1.8rem 1.5rem;
    text-align: center;
    margin-bottom: 1.2rem;
    transition: border-color 0.2s;
}
.upload-wrapper:hover { border-color: #e8d1a7; }

/* ── Divider ── */
.soft-divider {
    border: none;
    border-top: 1px solid #84592b;
    margin: 1.5rem 0;
}

/* ── Result Cards ── */
.result-banner {
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
}
.result-banner-tensioner {
    background: linear-gradient(135deg, #743014, #442d1c);
    border: 1.5px solid #9D9167;
}
.result-banner-gardan {
    background: linear-gradient(135deg, #84592b, #442d1c);
    border: 1.5px solid #9D9167;
}
.result-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    color: #9D9167;
}
.result-class {
    font-family: 'DM Serif Display', serif;
    font-size: 1.85rem;
    color: #e8d1a7;
    margin: 0;
}
.confidence-pill {
    display: inline-block;
    font-size: 0.82rem;
    font-weight: 700;
    padding: 0.3rem 0.85rem;
    border-radius: 99px;
    margin-top: 0.6rem;
    background: #442d1c;
    border: 1px solid #9D9167;
    color: #e8d1a7;
}

/* ── Insight Sections ── */
.insight-block {
    background: #84592b;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    border: 1px solid #9D9167;
    margin-bottom: 0.9rem;
}
.insight-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.6rem;
}
.insight-icon { font-size: 1.1rem; }
.insight-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #e8d1a7;
}
.insight-body {
    font-size: 0.855rem;
    color: rgba(232, 209, 167, 0.9);
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
    color: #9D9167;
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid #84592b;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stFileUploader"] > label { display: none; }
div[data-testid="stFileUploader"] section {
    background: transparent;
    border: none;
    padding: 0;
}
.stButton > button {
    background: #743014;
    color: #e8d1a7;
    font-size: 0.92rem;
    font-weight: 600;
    border: 1px solid #9D9167;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    width: 100%;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s;
    letter-spacing: 0.01em;
}
.stButton > button:hover {
    background: #9D9167;
    color: #442d1c;
    border-color: #e8d1a7;
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }

/* ── Confidence Bar ── */
.conf-bar-wrapper {
    background: #442d1c;
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
    margin-top: 0.6rem;
    border: 1px solid #84592b;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}
.bar-orange { background: linear-gradient(90deg, #9D9167, #e8d1a7); }
.bar-blue   { background: linear-gradient(90deg, #743014, #e8d1a7); }
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
            max_len = 130   # fallback
        return model, scaler, max_len, None
    except Exception as e:
        return None, None, None, str(e)


model, scaler, MAX_LEN, load_error = load_resources()


# ═════════════════════════════════════════════════════════
# HERO HEADER
# ═════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-badge">🔧 Sistem Diagnostik Audio</div>
    <h1 class="hero-title">Moto<span>Scan</span></h1>
    <p class="hero-desc">
        Unggah rekaman suara mesin motor matic Anda, dan sistem kami akan menganalisis
        apakah terdapat tanda-tanda kerusakan pada <strong>Tensioner</strong> atau
        <strong>Gardan</strong> — dalam hitungan detik, tanpa perlu ke bengkel terlebih dahulu.
    </p>
</div>
""", unsafe_allow_html=True)


# ─── Keunggulan Singkat ───────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0.5rem;">
        <div style="font-size:1.5rem; margin-bottom:0.4rem;">⚡</div>
        <div style="font-size:0.82rem; font-weight:600; color:#e8d1a7;">Analisis Cepat</div>
        <div style="font-size:0.78rem; color:#9D9167; margin-top:0.2rem;">Hasil dalam &lt; 5 detik</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0.5rem;">
        <div style="font-size:1.5rem; margin-bottom:0.4rem;">🎯</div>
        <div style="font-size:0.82rem; font-weight:600; color:#e8d1a7;">Akurasi Tinggi</div>
        <div style="font-size:0.78rem; color:#9D9167; margin-top:0.2rem;">Berbasis Deep Learning</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0.5rem;">
        <div style="font-size:1.5rem; margin-bottom:0.4rem;">📋</div>
        <div style="font-size:0.82rem; font-weight:600; color:#e8d1a7;">Panduan Lengkap</div>
        <div style="font-size:0.78rem; color:#9D9167; margin-top:0.2rem;">Saran servis & perawatan</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════
# KARTU EDUKASI
# ═════════════════════════════════════════════════════════
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
        <div class="edu-card-icon">🔩</div>
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


# ═════════════════════════════════════════════════════════
# UPLOAD AREA
# ═════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Langkah 1 — Unggah Rekaman</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Pilih File Audio Mesin Motor Anda</p>', unsafe_allow_html=True)

st.markdown("""
<div style="background:#84592b; border-radius:12px; padding:0.9rem 1.2rem;
            margin-bottom:1rem; font-size:0.84rem; color:#e8d1a7; display:flex;
            align-items:flex-start; gap:0.6rem; border: 1px solid #9D9167;">
    <span>💡</span>
    <span>Untuk hasil terbaik, rekam suara mesin dari jarak ±30 cm saat mesin berjalan
    stasioner (idle). Gunakan format <strong>.wav</strong> dengan durasi minimal 3 detik.</span>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Upload file WAV",
    type=["wav"],
    help="Format yang didukung: .wav",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    st.markdown("""
    <div style="background:#9D9167; border:1px solid #e8d1a7; border-radius:12px;
                padding:0.75rem 1.1rem; margin-bottom:0.8rem; font-size:0.85rem;
                color:#442d1c; display:flex; align-items:center; gap:0.5rem; font-weight:600;">
        <span>✅</span>
        <span>File berhasil diunggah. Periksa audio di bawah sebelum dianalisis.</span>
    </div>
    """, unsafe_allow_html=True)

    st.audio(uploaded_file, format="audio/wav")

    # Info file
    file_size_kb = round(len(uploaded_file.getvalue()) / 1024, 1)
    st.markdown(f"""
    <div style="font-size:0.78rem; color:#9D9167; margin-top:0.3rem; margin-bottom:1rem;">
        📁 {uploaded_file.name} &nbsp;·&nbsp; {file_size_kb} KB
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════
# TOMBOL ANALISIS & PREDIKSI
# ═════════════════════════════════════════════════════════
if uploaded_file is not None:
    st.markdown('<p class="section-label">Langkah 2 — Mulai Analisis</p>', unsafe_allow_html=True)

    # Cek apakah model tersedia
    if load_error:
        st.markdown(f"""
        <div style="background:#743014; border:1px solid #9D9167; border-radius:12px;
                    padding:1rem 1.2rem; color:#e8d1a7; font-size:0.86rem;">
            ⚠️ <strong>Model belum tersedia.</strong><br>
            Pastikan file <code>model_motor_matic_cnn1d.h5</code> dan
            <code>scaler_motor_matic.pkl</code> berada di direktori yang sama dengan
            <code>app.py</code>.<br><br>
            <em>Detail: {load_error}</em>
        </div>
        """, unsafe_allow_html=True)
    else:
        run_analysis = st.button("🔍  Analisis Suara Mesin", use_container_width=True)

        if run_analysis:
            # ── Loading Animation ──────────────────────────
            with st.spinner(""):
                loading_placeholder = st.empty()
                loading_placeholder.markdown("""
                <div style="background:#84592b; border:1px solid #9D9167; border-radius:16px;
                            padding:2rem; text-align:center; margin:1rem 0;">
                    <div style="font-size:2.2rem; margin-bottom:0.8rem;">🎧</div>
                    <div style="font-size:1rem; font-weight:600; color:#e8d1a7; margin-bottom:0.4rem;">
                        Sistem sedang mendengarkan dan menganalisis suara mesin...
                    </div>
                    <div style="font-size:0.86rem; color:#9D9167;">
                        Mohon tunggu sebentar, proses ini hanya memerlukan beberapa detik.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(2)   # Delay buatan agar efek loading terasa natural

                # ── Simpan ke tempfile & prediksi ──────────
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                result = predict_audio(tmp_path, model, scaler, MAX_LEN)
                os.unlink(tmp_path)

            loading_placeholder.empty()

            # ── Tampilkan Hasil ────────────────────────────
            if result is None:
                st.markdown("""
                <div style="background:#743014; border:1px solid #9D9167; border-radius:12px;
                            padding:1rem 1.2rem; color:#e8d1a7; font-size:0.86rem;">
                    ⚠️ File audio tidak dapat diproses. Pastikan file tidak rusak dan
                    berdurasi minimal 1 detik.
                </div>
                """, unsafe_allow_html=True)
            else:
                label      = result["label"]
                kelas      = result["kelas"]
                confidence = result["confidence"]
                conf_pct   = f"{confidence:.1f}%"
                conf_int   = int(confidence)

                st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
                st.markdown('<p class="section-label">Hasil Analisis</p>', unsafe_allow_html=True)

                # Banner Hasil
                if label == 0:
                    banner_class = "result-banner-tensioner"
                    conf_class   = "pill-orange"
                    bar_class    = "bar-orange"
                    emoji        = "⚙️"
                else:
                    banner_class = "result-banner-gardan"
                    conf_class   = "pill-blue"
                    bar_class    = "bar-blue"
                    emoji        = "🔩"

                st.markdown(f"""
                <div class="result-banner {banner_class}">
                    <div class="result-label">
                        Komponen Terdeteksi
                    </div>
                    <p class="result-class">{emoji} {kelas}</p>
                    <span class="confidence-pill {conf_class}">
                        Tingkat Keyakinan: {conf_pct}
                    </span>
                    <div class="conf-bar-wrapper" style="margin-top:0.8rem;">
                        <div class="conf-bar-fill {bar_class}" style="width:{conf_int}%;"></div>
                    </div>
                    <div style="font-size:0.75rem; color:#9D9167; margin-top:0.3rem;">
                        {conf_int}% keyakinan sistem terhadap hasil ini
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Insight Berdasarkan Label ──────────────
                st.markdown("""
                <p style="font-size:1rem; font-weight:700; color:#e8d1a7; margin:1.2rem 0 0.8rem;">
                    Apa Artinya Ini untuk Motor Anda?
                </p>
                """, unsafe_allow_html=True)

                if label == 0:
                    # ──── Tensioner Aus ────────────────────
                    st.markdown("""
                    <div class="insight-block">
                        <div class="insight-header">
                            <span class="insight-icon">🔍</span>
                            <span class="insight-title">Mengapa Tensioner Bisa Aus?</span>
                        </div>
                        <div class="insight-body">
                            Tensioner bekerja terus-menerus menahan tekanan rantai mesin setiap
                            saat motor digunakan. Seiring waktu, komponen ini mengalami keausan
                            alami. Penyebab utama yang mempercepatnya antara lain:
                            <ul>
                                <li>Pemakaian motor yang sangat intensif tanpa servis rutin</li>
                                <li>Oli mesin yang jarang diganti atau kualitasnya rendah — tensioner
                                    kehilangan pelumasan yang cukup</li>
                                <li>Mesin sering dinyalakan dalam kondisi dingin ekstrem lalu langsung
                                    digeber (RPM tinggi)</li>
                                <li>Usia pemakaian yang sudah melampaui batas servis (umumnya 24.000–30.000 km)</li>
                            </ul>
                        </div>
                    </div>

                    <div class="insight-block">
                        <div class="insight-header">
                            <span class="insight-icon">🛠️</span>
                            <span class="insight-title">Tindakan yang Perlu Dilakukan</span>
                        </div>
                        <div class="insight-body">
                            Segera bawa motor Anda ke bengkel resmi atau bengkel kepercayaan.
                            Sampaikan kepada mekanik bahwa Anda mencurigai ada masalah pada
                            <strong>tensioner rantai mesin</strong>. Minta mekanik untuk:
                            <ul>
                                <li>Memeriksa kondisi tensioner — apakah masih bisa dikencangkan
                                    (adjuster) atau perlu diganti unit penuh</li>
                                <li>Mengecek sekaligus kondisi rantai timing — jika rantai sudah
                                    mulai longgar atau aus, ganti bersama tensioner</li>
                                <li>Memeriksa kondisi oli mesin dan menggantinya jika sudah waktunya</li>
                                <li>Melakukan test drive setelah penggantian untuk memastikan suara
                                    berisik sudah hilang</li>
                            </ul>
                            <strong>Biaya estimasi:</strong> Penggantian tensioner umumnya berkisar
                            Rp 80.000–250.000 tergantung merek dan tipe motor Anda.
                        </div>
                    </div>

                    <div class="insight-block">
                        <div class="insight-header">
                            <span class="insight-icon">🛡️</span>
                            <span class="insight-title">Tips Pencegahan ke Depannya</span>
                        </div>
                        <div class="insight-body">
                            Setelah diperbaiki, lakukan kebiasaan berikut agar masalah yang sama
                            tidak terulang:
                            <ul>
                                <li><strong>Ganti oli mesin secara rutin</strong> setiap 2.000–3.000 km
                                    atau sesuai rekomendasi buku manual. Gunakan oli yang sesuai spesifikasi
                                    motor Anda.</li>
                                <li><strong>Hindari langsung tancap gas</strong> saat mesin baru dinyalakan.
                                    Panaskan mesin 1–2 menit terlebih dahulu agar oli tersirkulasi sempurna.</li>
                                <li><strong>Lakukan servis berkala</strong> di bengkel setiap 6 bulan atau
                                    6.000 km. Minta mekanik untuk selalu memeriksa kondisi tensioner dan
                                    rantai mesin.</li>
                                <li><strong>Perhatikan suara mesin</strong> sejak dini. Bunyi "kletek" di
                                    pagi hari adalah tanda awal yang tidak boleh diabaikan.</li>
                            </ul>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    # ──── Gardan / Gigi Rasio ──────────────
                    st.markdown("""
                    <div class="insight-block">
                        <div class="insight-header">
                            <span class="insight-icon">🔍</span>
                            <span class="insight-title">Mengapa Gardan Bisa Terkikis?</span>
                        </div>
                        <div class="insight-body">
                            Gardan (gigi rasio) meneruskan tenaga dari CVT ke roda belakang melalui
                            gesekan antargigi yang terjadi ribuan kali per menit. Kerusakan pada
                            bagian ini umumnya dipicu oleh:
                            <ul>
                                <li><strong>Oli gardan yang jarang atau tidak pernah diganti</strong> —
                                    ini adalah penyebab paling umum. Banyak pemilik motor tidak tahu
                                    bahwa gardan memiliki oli tersendiri yang perlu diganti</li>
                                <li>Oli gardan yang bocor karena seal rusak, sehingga gigi bergesekan
                                    tanpa pelumas yang cukup</li>
                                <li>Kebiasaan mengendarai dengan muatan berlebihan atau sering menanjak
                                    di jalan terjal</li>
                                <li>Pemakaian motor yang sudah sangat lama (lebih dari 4–5 tahun)
                                    tanpa pengecekan gardan sama sekali</li>
                            </ul>
                        </div>
                    </div>

                    <div class="insight-block">
                        <div class="insight-header">
                            <span class="insight-icon">🛠️</span>
                            <span class="insight-title">Tindakan yang Perlu Dilakukan</span>
                        </div>
                        <div class="insight-body">
                            Jangan tunda terlalu lama — gigi yang terkikis bisa memperparah kerusakan
                            lebih dalam. Segera ke bengkel dan sampaikan kepada mekanik bahwa Anda
                            mencurigai ada masalah pada <strong>gardan atau gigi rasio</strong>.
                            Minta mekanik untuk:
                            <ul>
                                <li>Membuka dan memeriksa kondisi gardan secara visual — periksa
                                    apakah ada serbuk logam atau gigi yang sudah aus/patah</li>
                                <li><strong>Ganti oli gardan</strong> terlebih dahulu jika belum pernah
                                    atau sudah sangat lama (oli gardan idealnya diganti setiap 10.000 km)</li>
                                <li>Jika gigi sudah aus parah, lakukan penggantian unit gardan lengkap
                                    atau ganti set gigi rasio sesuai rekomendasi mekanik</li>
                                <li>Periksa kondisi seal gardan — seal yang bocor harus diganti agar
                                    oli tidak cepat habis lagi</li>
                            </ul>
                            <strong>Biaya estimasi:</strong> Ganti oli gardan sangat terjangkau
                            (Rp 15.000–40.000), sedangkan penggantian gigi rasio bisa berkisar
                            Rp 150.000–500.000 tergantung tingkat kerusakan dan tipe motor.
                        </div>
                    </div>

                    <div class="insight-block">
                        <div class="insight-header">
                            <span class="insight-icon">🛡️</span>
                            <span class="insight-title">Tips Pencegahan ke Depannya</span>
                        </div>
                        <div class="insight-body">
                            Masalah gardan hampir selalu bisa dicegah dengan perawatan sederhana:
                            <ul>
                                <li><strong>Ganti oli gardan secara rutin</strong> setiap 10.000 km
                                    atau 1 tahun sekali. Minta ini secara spesifik saat servis, karena
                                    mekanik terkadang lupa jika tidak diminta.</li>
                                <li><strong>Gunakan oli gardan yang sesuai spesifikasi</strong> motor
                                    Anda. Cek buku manual atau tanyakan ke mekanik resmi.</li>
                                <li><strong>Waspadai kebocoran oli</strong> di sekitar roda belakang
                                    atau di bawah bodi motor — itu bisa jadi tanda seal gardan bocor.</li>
                                <li><strong>Hindari beban berlebih</strong> dan berkendara agresif di
                                    jalan rusak atau menanjak curam secara terus-menerus.</li>
                                <li>Jika mendengar suara "nguing" atau getaran halus saat akselerasi,
                                    langsung periksakan — lebih murah ditangani lebih awal.</li>
                            </ul>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Catatan Akhir ──────────────────────────
                st.markdown("""
                <div style="background:#84592b; border-radius:12px; padding:1rem 1.2rem;
                            font-size:0.82rem; color:#e8d1a7; margin-top:0.5rem;
                            border: 1px solid #9D9167;">
                    <strong>⚠️ Catatan Penting:</strong> Hasil analisis ini bersifat informatif dan
                    didasarkan pada pola suara yang terdeteksi. Untuk diagnosis yang pasti, tetap
                    disarankan untuk berkonsultasi langsung dengan mekanik berpengalaman di bengkel
                    resmi atau terpercaya.
                </div>
                """, unsafe_allow_html=True)

else:
    # ── State Kosong ───────────────────────────────────────
    st.markdown("""
    <div style="background:#442d1c; border:2px dashed #84592b; border-radius:16px;
                padding:2.5rem; text-align:center; color:#e8d1a7;">
        <div style="font-size:2rem; margin-bottom:0.6rem;">📁</div>
        <div style="font-size:0.92rem; font-weight:500; color:#e8d1a7; margin-bottom:0.3rem;">
            Belum ada file yang diunggah
        </div>
        <div style="font-size:0.82rem; color:#9D9167;">
            Unggah file .wav di atas untuk memulai analisis
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════
st.markdown("""
<div class="app-footer">
    MotoScan &nbsp;·&nbsp; Sistem Diagnostik Audio Motor Matic<br>
    Menggunakan CNN-1D + MFCC &nbsp;·&nbsp; Dibuat dengan ❤️ untuk pemilik motor Indonesia
</div>
""", unsafe_allow_html=True)