# 📈 **Prediksi Harga Saham Tesla Menggunakan GRU**

**Prediksi Harga Saham Tesla (GRU)** adalah aplikasi berbasis **Machine Learning** yang dirancang untuk memprediksi **harga saham Tesla (TSLA)** menggunakan model **Gated Recurrent Unit (GRU)**.

Sistem ini mencakup proses **data preprocessing, exploratory data analysis (EDA), training model GRU**, hingga **visualisasi hasil prediksi secara interaktif melalui Streamlit Dashboard**.

Proyek ini dikembangkan sebagai bagian dari tugas akademik mata kuliah **Machine Learning / Deep Learning** oleh **Kelompok 5**.

---

## 👥 **Kelompok 5**

| Nama              | NPM            |
| :---------------- | :--------------|
| **Raahmatunnisa** | 2308107010016  |
| **Davina Aura**   | 2308107010052  |
| **Sifa Jema**     | 2308107010080  |
| **Thahira Riska** | 2308107010024  |


## 🧠 **Deskripsi Sistem**

Sistem ini bekerja sebagai **tool analisis dan prediksi harga saham** berbasis time series yang memanfaatkan kemampuan **GRU** dalam menangkap pola historis data harga saham.

### **Cara Kerja Sistem**

1. **Input Data**

   * Dataset historis saham **Tesla (TSLA)** dalam format CSV
   * Fitur utama: *Open, High, Low, Close, Volume*

2. **Processing**

   * Data cleaning & normalisasi (MinMaxScaler)
   * Pembentukan sequence time-series
   * Training model **GRU**
   * Evaluasi performa model

3. **Output**

   * Prediksi harga saham
   * Visualisasi:

     * Actual vs Predicted
     * Error Distribution
     * Detailed Comparison Table
   * Insight performa model

### **Tujuan Utama**

* Menerapkan **Deep Learning (GRU)** pada data time series saham
* Membandingkan harga aktual dan hasil prediksi
* Menyediakan dashboard interaktif untuk analisis hasil model

---

## ⚙️ **Cara Instalasi dan Menjalankan**

---

### 🔹 **1. Clone Repository**

```bash
git clone https://github.com/Raahmatunnisa/Prediksi-Harga-Saham-GRU.git
cd Prediksi-Harga-Saham-GRU
```

---

### 🔹 **2. Setup Virtual Environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

---

### 🔹 **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

### 🔹 **4. Jalankan Aplikasi Streamlit**

```bash
streamlit run app.py
```

Aplikasi akan berjalan di:

```
http://localhost:8501
```

Atau versi online:

🔗 **Live Demo:**
👉 [https://prediksi-harga-saham-gru-kelompok5.streamlit.app/](https://prediksi-harga-saham-gru-kelompok5.streamlit.app/)

---

## 📂 **Struktur Proyek**

```
Prediksi-Harga-Saham-GRU/
├── data/
│   └── TSLA.csv                 # Dataset saham Tesla
│
├── model/
│   └── gru_tsla.h5              # Model GRU terlatih
│
├── notebook/
│   ├── tesla_gru_prediction.ipynb
│   └── tesla_gru_prediction_fix.ipynb
│
├── plot/                        # Folder output visualisasi
│
├── modules/
│   ├── data_fetcher.py          # Load & handle dataset
│   ├── data_processor.py        # Preprocessing & scaling
│   ├── model_trainer.py         # Training model GRU
│   ├── predictor.py             # Prediksi harga saham
│   └── utils.py                 # Helper functions
│
├── scaler_min.npy               # Min scaler
├── scaler_scale.npy             # Scale scaler
├── gru_tsla.keras               # Model format keras
├── app.py                       # Streamlit app
├── requirements.txt
└── README.md
```

---

## 🎯 **Fitur-Fitur Utama**

### ✅ **1. Prediksi Harga Saham dengan GRU**

* Model **GRU (Gated Recurrent Unit)** untuk time series
* Menggunakan sequence length yang dapat diatur
* Output harga prediksi berbasis data historis

---

### ✅ **2. Interactive Streamlit Dashboard**

* Pengaturan sequence length
* Informasi status model
* Visualisasi real-time hasil prediksi

---

### ✅ **3. Actual vs Predicted Visualization**

* Scatter plot perbandingan harga aktual dan prediksi
* Garis *perfect prediction* sebagai baseline

---

### ✅ **4. Error Distribution Analysis**

* Histogram distribusi error
* Garis zero-error untuk analisis bias model

---

### ✅ **5. Detailed Comparison Table**

* Tabel harga aktual vs prediksi
* Selisih error per data point

---

## 💻 **Teknologi yang Digunakan**

| Komponen                 | Teknologi               |
| :----------------------- | :---------------------- |
| **Programming Language** | Python 3.8+             |
| **Deep Learning**        | TensorFlow, Keras (GRU) |
| **Data Processing**      | Pandas, NumPy           |
| **Visualization**        | Matplotlib, Streamlit   |
| **Scaling**              | MinMaxScaler            |
| **Deployment**           | Streamlit Cloud         |

---

## 📊 **Dataset**

* **Sumber:** Yahoo Finance
* **Kode Saham:** TSLA (Tesla Inc.)
* **Periode:** Data historis harian
* **Fitur:** Open, High, Low, Close, Volume

---

## 📸 **Screenshots**

> *(Opsional — bisa ditambahkan nanti)*

* Dashboard Utama
* Error Distribution
* Actual vs Predicted
* Detailed Comparison Table

---

## 🚀 **Deployment**

Aplikasi telah dideploy menggunakan **Streamlit Cloud**:

🔗 [https://prediksi-harga-saham-gru-kelompok5.streamlit.app/](https://prediksi-harga-saham-gru-kelompok5.streamlit.app/)

---

## 📝 **License**

Proyek ini dikembangkan untuk **tujuan akademik**.

**© 2025 Kelompok 5 — All Rights Reserved**

Dilarang memperjualbelikan atau mendistribusikan ulang tanpa izin seluruh anggota kelompok.

---

## 🙏 **Acknowledgments**

* Yahoo Finance — Data saham
* TensorFlow & Keras Community
* Streamlit Community
* Dosen & Asisten Praktikum

---

### ✨ *"Turning Time Series Data into Actionable Insights"*
