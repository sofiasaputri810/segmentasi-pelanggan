
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load semua model
xgb_model = joblib.load('xgb_delivery_model.pkl')
kmeans_model = joblib.load('kmeans_rfm_model.pkl')
scaler = joblib.load('rfm_scaler.pkl')

st.set_page_config(page_title="E-Commerce Analytics", layout="wide")
st.title("📦 E-Commerce: Delivery Risk & Customer Segmentation")

# Membuat 2 Tab
tab1, tab2 = st.tabs(["🚚 Prediksi Keterlambatan", "👥 Segmentasi Pelanggan"])

# ==========================================
# TAB 1: PREDIKSI KETERLAMBATAN
# ==========================================
with tab1:
    st.header("Prediksi Risiko Keterlambatan Pesanan")
    st.write("Masukkan detail pesanan untuk melihat probabilitas keterlambatan.")
    
    col1, col2 = st.columns(2)
    with col1:
        # State dilambangkan dengan angka (misal: 0=SP, 1=RJ, dst)
        state_encoded = st.number_input("Kode State Pelanggan (0-26)", min_value=0, max_value=26, value=10)
        price = st.number_input("Total Harga Produk (BRL)", min_value=0.0, value=100.0)
    with col2:
        freight_value = st.number_input("Biaya Pengiriman (BRL)", min_value=0.0, value=15.0)
        category_encoded = st.number_input("Kode Kategori Produk (0-70)", min_value=0, max_value=70, value=5)
    
    if st.button("Prediksi Risiko"):
        # Buat dataframe dari input
        input_df = pd.DataFrame([{
            'state_encoded': state_encoded,
            'price': price,
            'freight_value': freight_value,
            'category_encoded': category_encoded
        }])
        
        proba = xgb_model.predict_proba(input_df)[0][1]
        
        st.subheader("Hasil Prediksi")
        st.metric(label="Probabilitas Terlambat", value=f"{proba*100:.1f}%")
        
        if proba > 0.5:
            st.error("⚠️ Risiko TINGGI. Pesanan ini berpotensi terlambat. Segera koordinasikan dengan tim logistik.")
        else:
            st.success("✅ Risiko RENDAH. Pesanan diprediksi akan tiba tepat waktu.")

# ==========================================
# TAB 2: SEGMENTASI PELANGGAN
# ==========================================
with tab2:
    st.header("Segmentasi Pelanggan (RFM)")
    st.write("Masukkan nilai RFM pelanggan untuk mengetahui segmen mereka.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (Hari sejak beli terakhir)", min_value=0, value=30)
    with col2:
        frequency = st.number_input("Frequency (Jumlah transaksi)", min_value=1, value=1)
    with col3:
        monetary = st.number_input("Monetary (Total belanja)", min_value=0.0, value=100.0)
        
    if st.button("Cek Segmen"):
        # Standarisasi input
        input_rfm = np.array([[recency, frequency, monetary]])
        input_scaled = scaler.transform(input_rfm)
        
        # Prediksi klaster
        cluster = kmeans_model.predict(input_scaled)[0]
        
        st.subheader("Hasil Segmentasi")
        
        # Berdasarkan hasil training kita tadi:
        # Cluster 0: Pembeli Sekali
        # Cluster 1: Pelanggan Setia (Repeat)
        if cluster == 0:
            st.info("👤 **Pembeli Sekali (One-Time Buyer)**: Pelanggan ini cenderung membeli sekali dan tidak kembali. Disarankan untuk memberikan voucher diskon untuk pembelian berikutnya.")
        else:
            st.success("⭐ **Pelanggan Setia (Loyal Customer)**: Pelanggan bernilai tinggi yang sering bertransaksi. Pertahankan dengan program loyalitas atau akses awal ke produk baru.")
