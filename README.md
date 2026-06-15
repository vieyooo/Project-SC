# MotoScan — Sistem Peringatan Dini Kerusakan Komponen Mesin Motor Matic Berbasis Audio 

MotoScan adalah sebuah platform berbasis Web App (Streamlit) yang dirancang sebagai *Proof of Concept* (PoC) untuk mendeteksi dan mendiagnosis kerusakan mekanis pada komponen mesin motor matic secara dini menggunakan analisis sinyal akustik (audio). 

Sistem ini memanfaatkan teknologi *Deep Learning* dengan arsitektur **Convolutional Neural Network 1-Dimensi (CNN-1D)** dan ekstraksi fitur **Mel-Frequency Cepstral Coefficients (MFCC)** untuk mengklasifikasikan jenis kerusakan blok mesin tanpa perlu melakukan pembongkaran fisik (*non-destructive testing*).

---

## Fitur Utama Aplikasi

- **Analisis Akustik Temporal:** Mengekstrak 120 fitur suara komprehensif yang menggabungkan matriks MFCC, Delta (kecepatan gelombang), dan Delta-Delta (akselerasi gelombang) untuk menangkap ritme hentakan mesin dari detik ke detik.
- **Diagnosis Instan & Confidence Score:** Menampilkan hasil klasifikasi kerusakan secara *live* dalam hitungan detik dilengkapi dengan indikator tingkat keyakinan model (*Confidence Bar*).
- **Antarmuka Premium & Kalem:** Desain UI minimalis berbasis Streamlit dengan tema khusus *Premium Dark Earth-Tone* yang nyaman di mata dan profesional.
- **Panduan Solusi & Edukasi Bengkel:** Dilengkapi kartu informasi edukasi komponen serta rekomendasi tindakan servis yang konkret (apa yang harus dikatakan kepada mekanik bengkel) beserta tips pencegahannya.
- **Natural Loading Effect:** Animasi pemrosesan suara yang interaktif untuk memastikan kenyamanan pengalaman pengguna (*User Experience*).

---

## Ruang Lingkup Klasifikasi

Model cerdas ini dilatih khusus untuk membedakan dua jenis anomali suara kerusakan komponen krusial pada motor matic:
1. **Label 0 — Tensioner Aus (Peregang Rantai CVT):** Ditandai dengan karakter suara ketukan ritmis (*transient noise*) akibat rantai keteng yang kendur.
2. **Label 1 — Gardan / Gigi Rasio Terkikis:** Ditandai dengan karakter suara dengungan konstan (*steady-state whine*) akibat gesekan ekstrem antargir transmisi belakang.

---

## Performa Model AI

Arsitektur CNN-1D dirancang secara ramping (~59 ribu parameter) dan dilengkapi dengan *Dropout* (40%), *L2 Regularization*, serta *Global Average Pooling* untuk mencegah *overfitting*.
- **Mean Cross-Validation Accuracy (5-Fold):** `97.69% (± 2.24%)`
- **Held-Out Test Set Accuracy:** `100.00%` *(Uji coba pada 46 sampel steril yang tidak pernah dilihat model saat training)*

**Batasan Masalah & Keterbatasan Proyek (Limitation of Study):**
Akurasi tinggi pada pengujian awal ini dipengaruhi oleh karakteristik dataset sekunder yang disegmentasi menjadi jendela waktu 3 detik dari sumber rekaman beberapa video yang digabungkan menjadi 1 (*Source Data Leakage*). Sistem ini berfungsi optimal sebagai *Proof of Concept* validasi arsitektur. Untuk implementasi industri skala massal, model disarankan untuk dilatih ulang menggunakan variasi data primer dari ratusan blok mesin motor yang berbeda.

---

## Struktur Direktori Proyek

```text
Project_SC/
│
├── .gitignore               # Daftar pengecualian file berat (Dataset/ & venv/)
├── README.md                # Dokumentasi proyek (File ini)
├── app.py                   # Source code utama antarmuka Web App Streamlit
├── requirements.txt         # Daftar dependencies library Python yang dibutuhkan
│
# --- File Artefak Model AI ---
├── model_motor_matic_cnn1d.h5   # Wujud fisik otak AI (Bobot Model CNN-1D)
├── scaler_motor_matic.pkl      # Standar normalisasi fitur suara (StandardScaler)
└── model_config.pkl            # Konfigurasi parameter dimensi audio (MAX_LEN, dll)
