import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Tesla Stock Price Prediction", layout="wide", page_icon="📈")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Tesla Stock Price Prediction dengan GRU")
st.markdown("### Multi-Feature Model (OHLCV)")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Pengaturan")
sequence_length = st.sidebar.slider("Sequence Length", 30, 120, 60, 10)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Model Info")

# Cek file yang diperlukan
model_path = "gru_tsla.keras"
scaler_min_path = "scaler_min.npy"
scaler_scale_path = "scaler_scale.npy"

files_exist = all([
    os.path.exists(model_path),
    os.path.exists(scaler_min_path),
    os.path.exists(scaler_scale_path)
])

if files_exist:
    st.sidebar.success("✅ Model tersedia")
    st.sidebar.info(f"📊 Model: {model_path}")
else:
    st.sidebar.error("❌ File model tidak ditemukan!")
    st.error("""
    ### ⚠️ File Model Tidak Ditemukan!
    
    Pastikan file berikut ada di folder yang sama dengan `app.py`:
    - `gru_tsla.keras`
    - `scaler_min.npy`
    - `scaler_scale.npy`
    
    **Cara mendapatkan file:**
    1. Download dari hasil training sebelumnya
    2. Atau jalankan script training untuk generate file baru
    """)
    st.stop()

# Main content
if st.button("🚀 Jalankan Prediksi", type="primary"):
    
    try:
        progress = st.progress(0)
        status = st.empty()
        
        # Step 1: Download data
        status.text("📥 Mengunduh data TSLA...")
        progress.progress(20)
        data = yf.download("TSLA", period="5y", progress=False, auto_adjust=True)
        data = data[["Open", "High", "Low", "Close", "Volume"]]
        st.success(f"✅ Data downloaded: {len(data)} rows")
        
        # Step 2: Preprocessing
        status.text("⚙️ Preprocessing data...")
        progress.progress(40)
        
        scaler = MinMaxScaler()
        scaler.min_ = np.load(scaler_min_path)
        scaler.scale_ = np.load(scaler_scale_path)
        scaled_data = scaler.transform(data)
        
        X = []
        y = []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 3])  # Close price
        
        X = np.array(X)
        y = np.array(y)
        
        train_size = int(len(X) * 0.8)
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        st.info(f"📊 Training data: {train_size} samples | Test data: {len(X_test)} samples")
        
        # Step 3: Load model
        status.text("🤖 Loading model...")
        progress.progress(60)
        model = load_model(model_path, compile=False)
        st.success("✅ Model loaded successfully!")
        
        # Step 4: Predict
        status.text("🔮 Generating predictions...")
        progress.progress(80)
        pred_scaled = model.predict(X_test, verbose=0, batch_size=128)
        
        # Inverse transform
        close_scaler = MinMaxScaler()
        close_scaler.min_ = scaler.min_[3]
        close_scaler.scale_ = scaler.scale_[3]
        pred_prices = close_scaler.inverse_transform(pred_scaled)
        real_prices = close_scaler.inverse_transform(y_test.reshape(-1, 1))
        
        progress.progress(100)
        status.text("✅ Prediction completed!")
        
        # Calculate metrics
        mae = mean_absolute_error(real_prices, pred_prices)
        rmse = np.sqrt(mean_squared_error(real_prices, pred_prices))
        mape = np.mean(np.abs((real_prices - pred_prices) / real_prices)) * 100
        
        # Display metrics
        st.markdown("---")
        st.markdown("## 📊 Hasil Evaluasi Model")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📈 Total Samples", f"{len(real_prices)}")
        with col2:
            st.metric("💰 MAE", f"${mae:.2f}")
        with col3:
            st.metric("📉 RMSE", f"${rmse:.2f}")
        with col4:
            st.metric("📊 MAPE", f"{mape:.2f}%")
        
        # Price range
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Harga Minimum", f"${real_prices.min():.2f}")
        with col2:
            st.metric("Harga Maksimum", f"${real_prices.max():.2f}")
        with col3:
            st.metric("Harga Rata-rata", f"${real_prices.mean():.2f}")
        
        # Plot
        st.markdown("---")
        st.markdown("## 📈 Visualisasi Prediksi")
        
        tab1, tab2 = st.tabs(["📊 Full Data", "🔍 Last 100 Days"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(real_prices, label="Harga Asli", linewidth=2, color='#1f77b4', alpha=0.8)
            ax.plot(pred_prices, label="Prediksi GRU", linewidth=2, color='#ff7f0e', linestyle='--', alpha=0.8)
            ax.set_title("Prediksi Harga Saham Tesla - Full Test Data", fontsize=16, fontweight='bold')
            ax.set_xlabel("Days", fontsize=12)
            ax.set_ylabel("Price (USD)", fontsize=12)
            ax.legend(fontsize=11, loc='best')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with tab2:
            fig, ax = plt.subplots(figsize=(15, 6))
            last_n = 100
            ax.plot(real_prices[-last_n:], label="Harga Asli", linewidth=2, color='#1f77b4', marker='o', markersize=3)
            ax.plot(pred_prices[-last_n:], label="Prediksi GRU", linewidth=2, color='#ff7f0e', marker='s', markersize=3, alpha=0.7)
            ax.set_title(f"Prediksi Harga Saham Tesla - Last {last_n} Days", fontsize=16, fontweight='bold')
            ax.set_xlabel("Days", fontsize=12)
            ax.set_ylabel("Price (USD)", fontsize=12)
            ax.legend(fontsize=11, loc='best')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Error distribution
        st.markdown("---")
        st.markdown("## 📉 Error Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            errors = (real_prices - pred_prices).flatten()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_title("Error Distribution", fontsize=14, fontweight='bold')
            ax.set_xlabel("Error (USD)", fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(real_prices, pred_prices, alpha=0.5, s=20, color='green')
            ax.plot([real_prices.min(), real_prices.max()], 
                   [real_prices.min(), real_prices.max()], 
                   'r--', linewidth=2, label='Perfect Prediction')
            ax.set_title("Actual vs Predicted", fontsize=14, fontweight='bold')
            ax.set_xlabel("Actual Price (USD)", fontsize=11)
            ax.set_ylabel("Predicted Price (USD)", fontsize=11)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Data table
        st.markdown("---")
        st.markdown("## 📋 Detailed Comparison Table")
        
        # Show selector
        show_rows = st.selectbox("Tampilkan data:", ["Last 20", "Last 50", "Last 100", "All Data"])
        
        if show_rows == "Last 20":
            n = 20
        elif show_rows == "Last 50":
            n = 50
        elif show_rows == "Last 100":
            n = 100
        else:
            n = len(real_prices)
        
        comparison_df = pd.DataFrame({
            'Day': range(len(real_prices) - n, len(real_prices)),
            'Actual Price (USD)': real_prices[-n:].flatten(),
            'Predicted Price (USD)': pred_prices[-n:].flatten(),
            'Difference (USD)': (real_prices[-n:] - pred_prices[-n:]).flatten()
        })
        comparison_df['Error (%)'] = (abs(comparison_df['Difference (USD)']) / comparison_df['Actual Price (USD)'] * 100).round(2)
        
        st.dataframe(
            comparison_df.style.format({
                'Actual Price (USD)': '${:.2f}',
                'Predicted Price (USD)': '${:.2f}',
                'Difference (USD)': '${:.2f}',
                'Error (%)': '{:.2f}%'
            }).background_gradient(subset=['Error (%)'], cmap='RdYlGn_r'),
            use_container_width=True,
            height=400
        )
        
        # Download option
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data CSV",
            data=csv,
            file_name="tesla_prediction_results.csv",
            mime="text/csv"
        )
        
        # Summary
        st.markdown("---")
        st.markdown("## 📝 Kesimpulan & Interpretasi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            ### ✅ Model Performance
            
            **Metrics:**
            - MAE (Mean Absolute Error): **${mae:.2f}**
            - RMSE (Root Mean Squared Error): **${rmse:.2f}**
            - MAPE (Mean Absolute Percentage Error): **{mape:.2f}%**
            
            **Interpretasi:**
            - Model memiliki rata-rata error sekitar ±${mae:.2f} USD per prediksi
            - Error relatif terhadap harga adalah {mape:.2f}%
            - Performa model tergolong {'sangat baik' if mape < 5 else 'baik' if mape < 10 else 'cukup baik'}
            """)
        
        with col2:
            st.info(f"""
            ### 📊 Data Statistics
            
            **Price Range:**
            - Minimum: ${real_prices.min():.2f}
            - Maximum: ${real_prices.max():.2f}
            - Average: ${real_prices.mean():.2f}
            - Std Dev: ${real_prices.std():.2f}
            
            **Dataset:**
            - Total test samples: {len(real_prices)}
            - Sequence length: {sequence_length}
            - Features: OHLCV (5 features)
            """)
        
        # Model info
        with st.expander("🔍 Model Architecture & Details"):
            st.markdown("""
            ### Model Architecture:
            - **Layer 1:** GRU (64 units, return_sequences=True)
            - **Layer 2:** GRU (64 units)
            - **Layer 3:** Dense (1 output)
            
            ### Training Configuration:
            - **Optimizer:** Adam
            - **Loss Function:** Mean Squared Error (MSE)
            - **Input Features:** Open, High, Low, Close, Volume
            - **Sequence Length:** 60 days
            - **Train/Test Split:** 80/20
            
            ### Model Strengths:
            - ✅ Captures temporal patterns in stock prices
            - ✅ Utilizes multiple features (OHLCV) for better prediction
            - ✅ GRU architecture handles long-term dependencies
            - ✅ Suitable for short to medium-term predictions
            
            ### Limitations:
            - ⚠️ Predictions are based on historical patterns
            - ⚠️ Cannot predict sudden market events
            - ⚠️ Not suitable for long-term forecasting
            - ⚠️ Should not be used as sole investment advice
            """)
        
        st.success("✅ Analisis selesai!")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

else:
    # Instructions
    st.info("""
    ### 👋 Selamat Datang!
    
    Aplikasi ini menggunakan **pre-trained GRU model** untuk memprediksi harga saham Tesla.
    
    **Cara menggunakan:**
    1. Pastikan file model sudah ada di folder ini
    2. Klik tombol **"Jalankan Prediksi"** di atas
    3. Tunggu proses download data dan prediksi
    4. Lihat hasil visualisasi dan analisis
    
    **File yang diperlukan:**
    - `gru_tsla.keras` - Model yang sudah ditraining
    - `scaler_min.npy` - Scaler parameters
    - `scaler_scale.npy` - Scaler parameters
    """)
    
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e8/Tesla_logo.png", width=200)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p><strong>Tesla Stock Price Prediction with GRU</strong></p>
        <p>Dibuat dengan ❤️ menggunakan Streamlit & TensorFlow | Data dari Yahoo Finance</p>
    </div>
    """, unsafe_allow_html=True)