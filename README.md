# Enhanced Polyester Detection Pipeline

🔍 **Sistem deteksi sampah tekstil polyester di perairan laut menggunakan teknik pengolahan citra digital berbasis konvolusi 2D**

## 📋 Deskripsi Proyek

Program ini mengimplementasikan sistem deteksi sampah tekstil polyester di lingkungan perairan laut yang tercemar menggunakan metode pengolahan citra tradisional (non-deep learning). Sistem ini mampu mengidentifikasi berbagai kondisi polyester mulai dari yang terendam air jernih hingga kondisi ekstrem seperti air keruh, pencahayaan buruk, dan bercampur dengan debris.

### 🎯 Tujuan Utama
- Mengidentifikasi sampah tekstil polyester dalam berbagai kondisi lingkungan laut
- Memisahkan polyester dari sampah organik, plastik, dan material lainnya
- Mendukung proses daur ulang dengan deteksi otomatis yang akurat

## 🚀 Fitur Utama

### ✨ **Adaptive Preprocessing (8 Metode Manual)**
- **Manual Brightness Adjustment** - untuk kondisi gelap 1
- **Manual Contrast Enhancement** - untuk kondisi underwater/murky 1
- **Manual Contrast Stretching** - untuk kontras rendah 1
- **Manual Histogram Equalization** - untuk pencahayaan tidak merata 1
- **Manual Median Filter 7x7** - untuk noise dan debris 2
- **Manual Gaussian Filter 5x5** - untuk smoothing 2
- **Morphological Opening** - untuk debris removal 2
- **Laplacian Sharpening** - untuk blur dan wet condition 3

### 📊 **Ekstraksi Fitur Komprehensif**
- **Fitur Warna**: Grayscale Histogram (8 fitur) 3
- **Fitur Tekstur**: Local Binary Pattern/LBP (14 fitur) 3
- **Fitur Bentuk**: Contour Analysis (9 fitur) 4

### 🎯 **Klasifikasi Hybrid (2 Metode)**
- **Template Matching**: Cross-correlation dengan dataset polyester 4
- **Rule-based Classification**: 12 rules berbasis karakteristik polyester 4

### 🗃️ **Database Kondisi Kompleks**
- Environmental contexts: clear_water, murky_water, various_lighting, with_debris
- Polyester colors: bright_colors, faded_colors, transparent_polyester
- Polyester conditions: dry_condition, wet_condition, floating, partially_submerged
- Polyester shapes: clothing_fragments, fabric_pieces, microfiber_clusters, thread_bundles

## 📁 Struktur Folder

```
TubesNoGUI/
├── main.py                    # Script utama
├── image_processing.py        # Preprocessing & feature extraction
├── polyester_detection.py     # Classification & template matching
├── samples/                   # Input images (min. 60 RGB images)
├── dataset/                   # Template dataset polyester
│   ├── environmental_contexts/
│   │   ├── clear_water/
│   │   ├── murky_water/
│   │   ├── various_lighting/
│   │   └── with_debris/
│   ├── polyester_colors/
│   │   ├── bright_colors/
│   │   ├── faded_colors/
│   │   └── transparent_polyester/
│   ├── polyester_conditions/
│   │   ├── dry_condition/
│   │   ├── wet_condition/
│   │   ├── floating/
│   │   └── partially_submerged/
│   └── polyester_shapes/
│       ├── clothing_fragments/
│       ├── fabric_pieces/
│       ├── microfiber_clusters/
│       └── thread_bundles/
└── output/                    # Hasil processing
    ├── 001_namaImage/
    │   ├── 01_original.jpg
    │   ├── 02_01_brightness_enhancement.jpg
    │   ├── 02_02_underwater_contrast.jpg
    │   ├── ...
    │   ├── 03_final_preprocessed.jpg
    │   ├── 04_color_features.jpg
    │   ├── 05_texture_features.jpg
    │   ├── 06_shape_features.jpg
    │   ├── 07_classification_result.jpg
    │   └── 08_classification_report.txt
    ├── 002_namaImage/
    └── ...
```

## 🛠️ Requirements

### Library yang Digunakan (Sesuai Ketentuan):
```python
import cv2              # OpenCV 
import numpy as np      # NumPy 
from skimage import feature  # scikit-image 
import os, sys, io      # Library standar Python 
from PIL import Image   # Pillow (hanya untuk konversi/penyimpanan)
import matplotlib.pyplot as plt  # Matplotlib (hanya untuk visualisasi)
```

### Spesifikasi Citra:
- **Format**: RGB (WAJIB)
- **Ekstensi**: .jpg, .jpeg, .png, .bmp, .tiff
- **Jumlah minimum**: 60 citra
- **Resolusi**: Minimal 512x512 piksel

## 🚀 Cara Penggunaan

### 1. Persiapan Dataset
```bash
# Buat folder dan isi dengan gambar
mkdir samples
# Masukkan minimal 60 gambar RGB ke folder samples/

mkdir dataset
# Opsional: Isi subfolder dataset dengan template polyester
```

### 2. Menjalankan Program
```bash
python main.py
```

### 3. Output yang Dihasilkan
Setiap gambar akan diproses dan menghasilkan folder terpisah dengan format:
- `001_namaImage/`, `002_namaImage/`, dst.
- Berisi semua tahap preprocessing, feature extraction, dan hasil klasifikasi

## 📊 Metode yang Diimplementasikan

### **Preprocessing (8 Metode Manual)**
| Metode | Referensi | Kondisi Aplikasi |
|--------|-----------|------------------|
| Manual Brightness Adjustment | A4 | Gambar gelap |
| Manual Contrast Enhancement | A5 | Underwater/murky |
| Manual Contrast Stretching | A6 | Kontras rendah |
| Manual Histogram Equalization | A11 | Pencahayaan tidak merata |
| Manual Median Filter 7x7 | D5 | Noise dan debris |
| Manual Gaussian Filter 5x5 | D3 | Edge density tinggi |
| Morphological Opening | G1 | Debris removal |
| Laplacian Sharpening | D4 | Blur dan wet condition |

### **Feature Extraction (3 Kategori)**
| Kategori | Metode | Jumlah Fitur |
|----------|--------|-------------|
| Warna | HSV Statistics | 12 fitur |
| Tekstur | Local Binary Pattern (LBP) | 14 fitur |
| Bentuk | Contour-based Analysis | 9 fitur |

### **Klasifikasi (2 Metode)**
| Metode | Deskripsi |
|--------|-----------|
| Template Matching | Cross-correlation dengan dataset polyester |
| Rule-based Classification | 12 rules berbasis karakteristik HSV, LBP, dan shape |

## 📈 Hasil Output

### **File yang Dihasilkan per Gambar:**
1. **01_original.jpg** - Gambar asli
2. **02_XX_metodename.jpg** - Setiap tahap preprocessing
3. **03_final_preprocessed.jpg** - Hasil akhir preprocessing
4. **04_color_features.jpg** - Visualisasi fitur warna
5. **05_texture_features.jpg** - Visualisasi fitur tekstur
6. **06_shape_features.jpg** - Visualisasi fitur bentuk
7. **07_classification_result.jpg** - Hasil klasifikasi
8. **08_classification_report.txt** - Laporan detail

### **Contoh Laporan Klasifikasi:**
```
=== POLYESTER DETECTION REPORT ===
Detection Status: POLYESTER DETECTED
Material Type: POLYESTER
Confidence: 87.50%
Primary Method: Template + Rule Hybrid (clothing_fragments)
Secondary Method: HSV Color + LBP Texture + Contour Shape

=== TEMPLATE MATCHING INFO ===
Templates Used: 15
Best Category: clothing_fragments
Best Subcategory: polyester_shapes
Best Score: 0.567
Detection Context: clothing_fragments_polyester_shapes

=== DETECTION SCORES (Enhanced Rules) ===
Total Score: 14/18

=== CONFIDENCE INDICATORS ===
✓ High synthetic saturation
✓ Uniform synthetic texture (LBP)
✓ Solid regular structure
✓ Clothing fragment shape
```

## 🎓 Keunggulan Implementasi

### **Sesuai Ketentuan Akademik:**
- ✅ Hanya menggunakan library yang diizinkan
- ✅ Implementasi manual semua metode preprocessing
- ✅ Format RGB enforced
- ✅ Minimal 60 citra input
- ✅ 3 kategori feature extraction
- ✅ 2 metode klasifikasi

### **Adaptive & Intelligent:**
- 🧠 Deteksi otomatis kondisi gambar (gelap, blur, noise, debris, dll.)
- ⚙️ Pemilihan preprocessing adaptif berdasarkan kondisi
- 🎯 Hybrid classification dengan fallback mechanism
- 📊 Confidence scoring berdasarkan multiple indicators

### **Comprehensive Database Support:**
- 🌊 15 subcategories kondisi polyester
- 🎨 Variasi warna: cerah, pudar, transparan
- 🌍 Kondisi lingkungan: air jernih, keruh, debris, pencahayaan
- 📐 Bentuk: fragments, pieces, microfiber, threads

## 🔧 Troubleshooting

### **Dataset Kosong:**
```
⚠️ Dataset kosong. Menggunakan rule-based detection saja.
```
- Program tetap berjalan dengan rule-based classification
- Template matching otomatis di-bypass

### **Error Library:**
- Pastikan hanya menggunakan library yang diizinkan
- Install: `pip install opencv-python scikit-image numpy`

### **Gambar Tidak Terbaca:**
- Pastikan format RGB (.jpg, .png, .bmp)
- Cek path dan permission folder samples/

## 📚 Referensi Metode

Program ini mengimplementasikan metode-metode dari referensi akademik:
- **A4, A5, A6, A11**: Manual adjustment methods
- **D1, D3, D4, D5**: Convolution and filtering methods  
- **G1**: Morphological operations
- **H3**: Contour-based shape analysis

## 👥 Tim Pengembang

Program dikembangkan untuk memenuhi tugas mata kuliah Pengolahan Citra Digital dengan fokus pada:
- Implementasi metode tradisional (non-deep learning)
- Deteksi sampah tekstil untuk konservasi lingkungan laut
- Pendekatan adaptive processing berbasis kondisi citra

---

**🌊 Mendukung upaya pelestarian ekosistem laut melalui teknologi pengolahan citra digital**