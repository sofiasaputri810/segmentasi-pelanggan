# 📦 E-Commerce Analytics: Prediksi Risiko Keterlambatan & Segmentasi Pelanggan

## 📌 Ringkasan Eksekutif (Insight 1 Halaman)
Dalam industri e-commerce, kepuasan pelanggan bergantung pada dua hal utama: **ketepatan waktu pengiriman** dan **relevansi penawaran (marketing)**. Proyek ini menggabungkan dua pendekatan *Machine Learning* pada dataset e-commerce publik (Olist Brazil) untuk menjawab tiga pertanyaan bisnis:
1. Pesanan mana yang berisiko terlambat?
2. Pelanggan mana yang paling bernilai?
3. Apa yang perlu diperbaiki?

### 💡 Temuan Utama (Key Insights)
1. **Tingkat Keterlambatan Rendah, Tapi Berdampak Besar:** Hanya sekitar **8%** pesanan yang terlambat dari total estimasi. Namun, model klasifikasi (XGBoost) berhasil mendeteksi pesanan berisiko ini dengan **ROC-AUC sebesar 0.XX** *(isi dengan hasil di Colab)*. Fitur paling berpengaruh adalah **jarak pengiriman (State)** dan **biaya pengiriman**. Semakin jauh dan mahal ongkos kirimnya, semakin tinggi risiko gagal antar.
2. **Segmentasi Pelanggan Sangat Timpang (Pareto Principle):** Dari hasil K-Means Clustering (dengan **Silhouette Score sebesar 0.XX**), pelanggan terbagi menjadi 2 klaster utama:
   - **Klaster 0 (Pembeli Sekali):** Mencakup **~97%** pelanggan (90.543 orang). Mereka hanya berbelanja 1x dengan total belanja rata-rata 137 BRL.
   - **Klaster 1 (Pelanggan Setia):** Hanya **~3%** pelanggan (2.807 orang), tetapi frekuensi belanjanya >2x dan total belanja mencapai 276 BRL (2x lebih besar).
   - **Insight Bisnis:** Perusahaan harus mengalihkan fokus dari sekadar *acquisition* (mencari pelanggan baru) menjadi *retention* (mempertahankan Klaster 0 agar naik kelas ke Klaster 1).

### 🛠️ Rekomendasi Aksi (Actionable Recommendations)
- **Untuk Tim Logistik:** Gunakan probabilitas dari dashboard untuk mem- *flag* pesanan dengan risiko keterlambatan > 50%. Berikan notifikasi *delay* lebih awal kepada pelanggan di wilayah dengan `State` tertentu untuk menghindari ulasan buruk.
- **Untuk Tim Marketing:** Buat program *loyalty* (poin/voucher) yang secara spesifik menyasar Klaster 0 tepat 30 hari setelah pembelian pertama mereka (sebelum mereka benar-benar *churn*).

---

## 🖥️ Dashboard Interaktif (Streamlit)

Proyek ini dilengkapi dengan dashboard interaktif yang memungkinkan pengguna memasukkan data pesanan atau RFM pelanggan dan melihat hasil prediksi secara *real-time*.

🔗 **[Klik di sini untuk membuka Dashboard Streamlit](https://segmentasi-pelanggan-e-commerce.streamlit.app)**

![Preview Dashboard E-Commerce]()

---

## 📊 Data & Metodologi

### 1. Dataset
- **Sumber:** [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
- **Data yang Digunakan:** Tabel `orders`, `customers`, `order_items`, dan `products`.
- **Catatan Penting:** Dataset ini memiliki keunikan di mana `customer_id` berubah setiap kali pelanggan melakukan pesanan baru. Untuk segmentasi RFM yang akurat, proyek ini menggunakan **`customer_unique_id`** sebagai identitas asli pelanggan.

### 2. Model 1: Klasifikasi Keterlambatan (XGBoost)
- **Target:** `is_late` (1 jika `delivery_date > estimated_date`).
- **Fitur:** Jarak pengiriman (State), harga produk, biaya pengiriman, kategori produk.
- **Penanganan Imbalance:** Menggunakan `scale_pos_weight` pada XGBoost untuk menyeimbangkan kelas minoritas (pesanan terlambat).
- **Metrik Evaluasi:** **ROC-AUC** (untuk melihat performa pemisahan kelas) dan **F1-Score** (untuk menyeimbangkan Precision & Recall).

### 3. Model 2: Segmentasi Pelanggan (K-Means)
- **Fitur:** RFM (Recency, Frequency, Monetary) yang distandarisasi dengan `StandardScaler`.
- **Penentuan K:** Menggunakan **Silhouette Score** untuk menemukan jumlah klaster optimal (K=2).
