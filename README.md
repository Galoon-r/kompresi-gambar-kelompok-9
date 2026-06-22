# 🧠 ImageProcs — Kompresi PCA & Pengenalan Wajah (Eigenfaces)

Aplikasi web interaktif berbasis **Streamlit** yang mengimplementasikan **Principal Component Analysis (PCA)** untuk dua studi kasus: kompresi citra digital dan pengenalan wajah menggunakan metode **Eigenfaces**.

Proyek ini disusun sebagai bagian dari tugas akhir semester mata kuliah **Aljabar Linear**, menghubungkan konsep dekomposisi matriks dan reduksi dimensi dengan penerapan praktis di bidang pengolahan citra.

## ✨ Fitur

Aplikasi terdiri atas tiga mode yang dapat dipilih melalui sidebar:

### 🗜️ Kompresi PCA
Unggah sebuah gambar, atur jumlah komponen utama (`n_components`) melalui slider, dan lihat hasil kompresi secara langsung berdampingan dengan gambar asli. Dilengkapi statistik ukuran berkas, persentase penghematan, dan tombol unduh hasil (PNG).

### 👁️ Perbandingan Wajah
Unggah dua foto wajah untuk membandingkan tingkat kemiripannya. Sistem otomatis mendeteksi dan memotong area wajah (Haar Cascade), lalu menghitung **cosine similarity** untuk menentukan apakah kedua wajah **mirip/sama** atau **berbeda**, lengkap dengan visualisasi bar similarity dan threshold yang dapat diatur.

### 🔍 Pengenalan Wajah
Latih model PCA dari dataset wajah (`.zip` berisi folder per individu), lalu identifikasi wajah baru terhadap database tersebut. Menampilkan nama hasil identifikasi beserta **top-5 kandidat** berdasarkan skor kemiripan. Model juga bisa dilatih sebelumnya di Google Colab dan diimpor sebagai berkas `.pkl`.

## 🧮 Dasar Teori

| Konsep | Penerapan |
|---|---|
| **PCA** (`sklearn.decomposition.PCA`) | Mereduksi dimensi tiap kanal warna (R, G, B) untuk kompresi citra; mereduksi dimensi vektor wajah (10.000 piksel) ke eigenspace |
| **Cosine Similarity** | Mengukur kemiripan antar wajah dalam eigenspace atau ruang piksel mentah |
| **Haar Cascade Classifier** (OpenCV) | Deteksi dan pemotongan otomatis area wajah |

> **Catatan teknis:** PCA hanya diterapkan ketika jumlah sampel cukup untuk estimasi variansi yang bermakna. Pada mode Perbandingan Wajah (2 sampel), kemiripan dihitung langsung dari cosine similarity vektor piksel mentah tanpa PCA, karena PCA pada 2 sampel menghasilkan nilai ekstrem (±1) yang tidak merepresentasikan kemiripan aktual.

## 🛠️ Tech Stack

- **Streamlit** — antarmuka web interaktif
- **scikit-learn** — implementasi PCA & cosine similarity
- **OpenCV (opencv-python)** — pengolahan citra & deteksi wajah
- **Pillow (PIL)** — I/O dan manipulasi gambar
- **NumPy** — operasi numerik dan array

## 🚀 Instalasi & Menjalankan

```bash
# Clone repository
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install streamlit pillow numpy opencv-python scikit-learn

# Jalankan aplikasi
streamlit run imageprocs_app.py
```

Aplikasi akan terbuka otomatis di `http://localhost:8501`.

## 📁 Struktur Dataset (untuk Pengenalan Wajah)

Dataset wajah harus dikompresi dalam format `.zip` dengan struktur folder berikut, satu folder per individu:

```
dataset.zip
├── nama_orang_1/
│   ├── foto1.jpg
│   └── foto2.jpg
├── nama_orang_2/
│   └── foto1.jpg
└── ...
```

## 📦 Model Pre-trained (Opsional)

Model PCA dapat dilatih di luar aplikasi (misalnya di Google Colab dengan akses GPU/TPU gratis) lalu diserialisasi menggunakan `joblib`:

```python
import joblib

model_data = {"pca": pca, "X_pca": X_pca, "labels": labels}
joblib.dump(model_data, "eigenfaces_model.pkl")
```

Berkas `.pkl` tersebut kemudian dapat diunggah langsung pada mode **Pengenalan Wajah** tanpa perlu melatih ulang dari dataset mentah.

## ⚠️ Batasan

- Deteksi wajah menggunakan Haar Cascade (bukan deep learning), sehingga rentan terhadap *false positive* pada kondisi pencahayaan/sudut tertentu.
- Akurasi pengenalan wajah sangat bergantung pada jumlah dan variasi citra per individu dalam dataset.
- Kompresi PCA diterapkan per kanal warna, bukan pada representasi citra secara keseluruhan.

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik (Tugas Akhir Semester — Aljabar Linear).
