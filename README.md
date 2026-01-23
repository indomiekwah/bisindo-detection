# BISINDO Sign Language Recognition

Proyek deteksi Bahasa Isyarat Indonesia (BISINDO) menggunakan MediaPipe dan Deep Learning.

## Struktur Folder

```
BisindoCV/
├── venv/                       # Virtual environment
├── data/
│   ├── raw/
│   │   ├── wl_bisindo/         # Dataset WL-BISINDO (32 kata)
│   │   └── .../             # Dataset rekaman sendiri
│   ├── landmarks/              # Hasil ekstraksi landmark
│   └── processed/              # Data siap training
├── src/
│   ├── extract_landmarks.py    # Script ekstraksi landmark
│   ├── preprocess.py           # Script preprocessing
│   └── predict.py              # Script prediksi real-time
├── models/                     # Model yang sudah ditrain
├── colab_notebooks/            # Notebook untuk training di Colab
├── requirements.txt
└── README.md
```

## Quick Start (Immediate Testing)

Untuk langsung menguji deteksi BISINDO setelah menginstall repository:

```powershell
# 1. Buat virtual environment
python -m venv venv

# 2. Aktivasi virtual environment
.\venv\Scripts\Activate

# 3. Install dependencies dari requirements.txt
pip install -r requirements.txt

# 4. Jalankan prediksi real-time
py src/predict.py
```

## Setup Environment & Full Workflow

### Step 1: Buat Virtual Environment dan Install Dependencies

```powershell
# Buat virtual environment
python -m venv venv

# Aktivasi
.\venv\Scripts\Activate

# Install semua dependencies
pip install -r requirements.txt
```

**Opsi**: Jika ingin langsung test deteksi setelah ini, jalankan `py src/predict.py`

### Step 2: Download Dataset

Download dataset WL-BISINDO dari [Kaggle](https://www.kaggle.com/datasets/glennleonali/wl-bisindo).

### Step 3: Ekstraksi dan Organisir Dataset

Ekstraksi dataset yang sudah didownload dan buat struktur folder sebagai berikut:

```
├── data/
│   ├── raw/
│   │   ├── wl_bisindo/         # Dataset WL-BISINDO (32 kata)
│   │   └── .../                # Dataset yang direkam sendiri
│   ├── landmarks/              # Hasil ekstraksi landmark
│   └── processed/              # Data siap training
```

### Step 4: Ekstraksi Landmark dari Video

Dengan virtual environment tetap aktif, jalankan script ekstraksi landmark:

```powershell
py src/extract_landmarks_orientation.py
```

Script ini akan:
- Mengekstrak koordinat landmark dari setiap video menggunakan MediaPipe Holistic
- Menghasilkan file `.npy` per video (num_frames × num_features)
- Menyimpan metadata dataset di `metadata.json`
- Menghasilkan file terproses di folder `data/processed/`

### Step 5: Training Model (Google Colab)

1. Download file berikut dari folder `data/processed/`:
   - `X_train.npy`
   - `X_test.npy`
   - `y_train.npy`
   - `y_test.npy`
   - `label_encoder.pkl`
   - `metadata.json`

2. Upload file-file tersebut ke Google Drive

3. Buka notebook `notebooks/train_bisindo_optimized.ipynb` di Google Colab

4. Update path direktori sesuai dengan lokasi file di Google Drive

5. Jalankan training notebook hingga selesai

### Step 6: Download Model Terlatih

Setelah training selesai, download model yang sudah terlatih dan simpan di folder:

```
models/
```

### Step 7: Jalankan Prediksi Real-time

Dengan virtual environment tetap aktif, jalankan:

```powershell
py src/predict.py
```

Aplikasi akan membuka kamera dan melakukan deteksi BISINDO secara real-time.

## Menggabungkan Dataset

Jika Anda memiliki dataset WL-BISINDO dan dataset custom:

```powershell
# Ekstraksi masing-masing
python src/extract_landmarks_orientation.py --input data/raw/wl_bisindo --output data/landmarks/wl_bisindo
python src/extract_landmarks_orientation.py --input data/raw/custom --output data/landmarks/custom

# PENTING: Gabungkan folder landmarks secara manual atau buat script merger
# Lalu preprocess gabungan
python src/preprocess.py --landmarks-dir data/landmarks/combined --output-dir data/processed
```

## Format Video untuk Rekaman Dataset Sendiri

- **Format:** MP4 (H.264 codec)
- **Resolusi:** 640×480 atau 1280×720
- **FPS:** 30 fps
- **Durasi:** 2-5 detik per gesture
- **Background:** Polos, minim distraksi
- **Pencahayaan:** Terang, merata
- **Framing:** Setengah badan ke atas
**Perhatikan bagian kiri dan kanan, sesuaikan dengan dataset WL_Bisindo**

## Dependencies

```
mediapipe>=0.10.0
opencv-python>=4.8.0
tensorflow>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

## Troubleshooting

### MediaPipe tidak mendeteksi tangan
- Pastikan pencahayaan cukup
- Pastikan tangan terlihat jelas dalam frame
- Coba kurangi `min_detection_confidence`

### Out of Memory saat preprocessing
- Proses per batch dengan menambah parameter `--max-samples`
- Kurangi `sequence_length`

### Video tidak terbaca
- Pastikan codec video didukung (MP4/H.264 recommended)
- Install codec tambahan: `pip install opencv-python-headless`

## Credits

Dataset BISINDO diunduh dari:
- **WL-BISINDO Dataset** - [Kaggle](https://www.kaggle.com/datasets/glennleonali/wl-bisindo)

Proyek ini menggunakan MediaPipe untuk ekstraksi landmark dan TensorFlow untuk pemodelan deep learning.