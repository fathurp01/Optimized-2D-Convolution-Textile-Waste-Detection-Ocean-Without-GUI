import cv2
import numpy as np
from skimage import feature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import os

# ============================================================================
# UTILITY FUNCTIONS - Color Space Conversions (Manual Implementation)
# ============================================================================


def rgb_to_gray(img_rgb):
    """Manual RGB to Grayscale conversion (PERSIS seperti referensi A3)"""
    H, W = img_rgb.shape[:2]
    gray = np.zeros((H, W), np.uint8)
    for i in range(H):
        for j in range(W):
            gray[i, j] = np.clip(
                0.299 * img_rgb[i, j, 0]
                + 0.587 * img_rgb[i, j, 1]
                + 0.114 * img_rgb[i, j, 2],
                0,
                255,
            )
    return gray


def rgb_to_hsv(img_rgb):
    """Convert RGB to HSV"""
    H, W = img_rgb.shape[:2]
    hsv_img = np.zeros((H, W, 3), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            # Get RGB values (normalized to 0-1)
            r = img_rgb[i, j, 0] / 255.0
            g = img_rgb[i, j, 1] / 255.0
            b = img_rgb[i, j, 2] / 255.0

            # Find max and min values
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            delta = max_val - min_val

            # Calculate VALUE (V)
            v = max_val

            # Calculate SATURATION (S)
            if max_val == 0:
                s = 0
            else:
                s = delta / max_val

            # Calculate HUE (H)
            if delta == 0:
                h = 0
            elif max_val == r:
                h = ((g - b) / delta) % 6
            elif max_val == g:
                h = (b - r) / delta + 2
            else:  # max_val == b
                h = (r - g) / delta + 4

            h = h * 60  # Convert to degrees

            # Store HSV values
            hsv_img[i, j, 0] = h  # Hue (0-360)
            hsv_img[i, j, 1] = s * 255  # Saturation (0-255)
            hsv_img[i, j, 2] = v * 255  # Value (0-255)

    return hsv_img.astype(np.uint8)


def hsv_to_rgb(img_hsv):
    """Convert HSV to RGB"""
    H, W = img_hsv.shape[:2]
    rgb_img = np.zeros((H, W, 3), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            # Get HSV values (normalized)
            h = img_hsv[i, j, 0]  # Hue (0-360)
            s = img_hsv[i, j, 1] / 255.0  # Saturation (0-1)
            v = img_hsv[i, j, 2] / 255.0  # Value (0-1)

            # Calculate RGB using HSV to RGB formula
            c = v * s  # Chroma
            x = c * (1 - abs((h / 60) % 2 - 1))
            m = v - c

            if 0 <= h < 60:
                r_prime, g_prime, b_prime = c, x, 0
            elif 60 <= h < 120:
                r_prime, g_prime, b_prime = x, c, 0
            elif 120 <= h < 180:
                r_prime, g_prime, b_prime = 0, c, x
            elif 180 <= h < 240:
                r_prime, g_prime, b_prime = 0, x, c
            elif 240 <= h < 300:
                r_prime, g_prime, b_prime = x, 0, c
            else:  # 300 <= h < 360
                r_prime, g_prime, b_prime = c, 0, x

            # Final RGB values
            r = (r_prime + m) * 255
            g = (g_prime + m) * 255
            b = (b_prime + m) * 255

            # Store RGB values
            rgb_img[i, j, 0] = np.clip(r, 0, 255)
            rgb_img[i, j, 1] = np.clip(g, 0, 255)
            rgb_img[i, j, 2] = np.clip(b, 0, 255)

    return rgb_img.astype(np.uint8)


# ============================================================================
# IMAGE QUALITY ANALYSIS - Enhanced untuk kondisi database kompleks
# ============================================================================


def analyze_image_quality(img_rgb):
    """Enhanced analysis untuk kondisi database yang kompleks"""
    # RGB to Gray conversion
    img_gray = rgb_to_gray(img_rgb)
    analysis = {}

    # hitung rata-rata kecerahan gambar
    brightness = np.mean(img_gray)
    analysis["brightness"] = brightness
    analysis["is_dark"] = brightness < 80
    analysis["is_bright"] = brightness > 180

    # Hitung kontras gambar (standar deviasi)
    contrast = img_gray.std()
    analysis["contrast"] = contrast
    analysis["is_low_contrast"] = contrast < 30

    # Deteksi blur menggunakan varian Laplacian
    blur_score = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    analysis["blur_score"] = blur_score
    analysis["is_blurry"] = blur_score < 100

    # Deteksi noise: selisih antara median blur dan gambar asli
    noise_level = np.std(cv2.medianBlur(img_gray, 5) - img_gray)
    analysis["noise_level"] = noise_level
    analysis["is_noisy"] = noise_level > 15

    # Hitung edge density menggunakan operator Canny
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    analysis["edge_density"] = edge_density

    # Deteksi kondisi khusus berdasarkan threshold pada metrik
    analysis["is_underwater"] = brightness < 100 and contrast < 25
    analysis["is_reflective"] = brightness > 150 and contrast < 30
    analysis["has_debris"] = edge_density > 0.15
    analysis["is_faded"] = brightness > 130 and contrast < 40
    analysis["is_transparent"] = brightness > 200 and contrast < 20

    return analysis


# ============================================================================
# 8 METODE PREPROCESSING
# ============================================================================


def manual_brightness_adjustment(img_gray, brightness=80):
    """METODE 1: Manual brightness adjustment (PERSIS referensi A4)"""
    H, W = img_gray.shape[:2]
    result = img_gray.copy()
    for i in range(H):
        for j in range(W):
            # Tambahkan nilai brightness ke setiap piksel dan pastikan tetap dalam rentang 0-255
            result[i, j] = np.clip(img_gray[i, j] + brightness, 0, 255)
    return result


def manual_contrast_enhancement(img_gray, contrast=1.7):
    """METODE 2: Manual contrast enhancement (PERSIS referensi A5)"""
    H, W = img_gray.shape[:2]
    result = img_gray.copy()
    for i in range(H):
        for j in range(W):
            # Tambahkan nilai contrast ke setiap piksel dan pastikan tetap dalam rentang 0-255
            result[i, j] = np.clip(contrast * img_gray[i, j], 0, 255)
    return result


def manual_contrast_stretching(img_gray):
    """METODE 3: Manual contrast stretching (PERSIS referensi A6)"""
    min_val = np.min(img_gray)
    max_val = np.max(img_gray)

    # Jika rentang intensitas 0 (semua piksel sama), kembalikan gambar asli
    if max_val - min_val == 0:
        return img_gray

    H, W = img_gray.shape[:2]
    result = np.zeros((H, W), np.uint8)

    for i in range(H):
        for j in range(W):
            # Rumus kontras stretching: skala nilai piksel ke rentang 0-255
            result[i, j] = np.clip(
                255 * (img_gray[i, j] - min_val) / (max_val - min_val), 0, 255
            )

    return result


def manual_histogram_equalization(img_gray):
    """METODE 4: Manual histogram equalization (PERSIS referensi A11)"""
    # Hitung histogram dari citra grayscale
    hist, bins = np.histogram(img_gray.flatten(), 256, [0, 256])

    # Hitung cumulative distribution function (CDF) dari histogram
    cdf = hist.cumsum()

    # Normalisasi CDF untuk visualisasi
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Masking nilai nol pada CDF dan terapkan histogram equalization
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype("uint8")

    equalized_img = cdf[img_gray]

    return equalized_img


def apply_median_filter_manual(img_gray):
    """METODE 5: Manual median filter 7x7 (PERSIS referensi D5)"""
    hasil = img_gray.copy()
    h, w = img_gray.shape

    # Iterasi setiap piksel gambar, kecuali tepi (karena window 7x7)
    for i in range(3, h - 3):
        for j in range(3, w - 3):
            # Ambil semua 49 tetangga dalam window 7x7 di sekitar piksel (i, j)
            neighbors = [
                img_gray[i + k, j + l] for k in range(-3, 4) for l in range(-3, 4)
            ]
            # Urutkan nilai tetangga untuk mencari median
            neighbors.sort()
            # Median adalah elemen ke-24 (indeks 24 dari 0-48)
            hasil[i, j] = neighbors[24]  # median dari 49 elemen

    return hasil


def manual_convolution_2d(X, F):
    """Manual 2D convolution (PERSIS referensi D1)"""
    X_height, X_width = X.shape
    F_height, F_width = F.shape

    out_height = X_height - F_height + 1
    out_width = X_width - F_width + 1

    out = np.zeros((out_height, out_width))

    for i in range(out_height):
        for j in range(out_width):
            out[i, j] = np.sum(X[i : i + F_height, j : j + F_width] * F)

    return out


def apply_gaussian_filter_manual(img_gray):
    """METODE 6: Manual Gaussian filter 5x5 (PERSIS referensi D3)"""
    # Gaussian filter dengan kernel 5x5
    kernel = (1.0 / 345) * np.array(
        [
            [1, 5, 7, 5, 1],
            [5, 20, 33, 20, 5],
            [7, 33, 55, 33, 7],
            [5, 20, 33, 20, 5],
            [1, 5, 7, 5, 1],
        ],
        dtype=np.float32,
    )

    hasil = manual_convolution_2d(img_gray.astype(np.float32), kernel)
    hasil = np.clip(hasil, 0, 255).astype(np.uint8)

    return hasil


def apply_morphological_opening(img_gray):
    """METODE 7: Morphological opening (PERSIS referensi G1)"""
    # Lakukan threshold biner menggunakan metode OTSU untuk mendapatkan citra biner
    ret, threshold = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Buat elemen struktur berbentuk elips dan menerapkan morphological opening
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hasil = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, strel)

    return hasil


def apply_sharpening_laplace(img_gray):
    """METODE 8: Laplacian sharpening (PERSIS referensi D4)"""
    # Laplacian sharpening kernel 5x5
    kernel = (1.0 / 16) * np.array(
        [
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0],
        ],
        dtype=np.float32,
    )

    hasil = manual_convolution_2d(img_gray.astype(np.float32), kernel)
    hasil = np.clip(hasil, 0, 255).astype(np.uint8)

    return hasil


# ============================================================================
# ADAPTIVE PREPROCESSING UNTUK DATABASE KOMPLEKS
# ============================================================================


def adaptive_preprocessing(img_rgb, analysis):
    """Adaptive preprocessing berdasarkan kondisi image"""

    img_gray = rgb_to_gray(img_rgb)

    preprocessing_steps = []
    current_img = img_gray.copy()

    # Step 1: Reduksi noise jika gambar terdeteksi noisy atau terdapat debris
    if analysis.get("is_noisy", False) or analysis.get("has_debris", False):
        print("     [P01] Applying median filter for noise/debris reduction...")
        current_img = apply_median_filter_manual(current_img)
        preprocessing_steps.append(("median_filter", current_img.copy()))

    # Step 2: Penyesuaian kecerahan jika gambar terlalu gelap
    if analysis.get("is_dark", False):
        print("     [P02] Brightening dark image...")
        current_img = manual_brightness_adjustment(current_img, brightness=50)
        preprocessing_steps.append(("brightness_adjustment", current_img.copy()))

    # Step 3: Peningkatan kontras jika gambar memiliki kontras rendah
    if analysis.get("is_low_contrast", False):
        print("     [P03] Enhancing contrast...")
        current_img = manual_contrast_enhancement(current_img, contrast=1.5)
        preprocessing_steps.append(("contrast_enhancement", current_img.copy()))

    # Step 4: Kontras stretching untuk memperluas rentang intensitas
    if analysis.get("is_faded", False) or analysis.get("is_underwater", False):
        print("     [P04] Applying contrast stretching...")
        current_img = manual_contrast_stretching(current_img)
        preprocessing_steps.append(("contrast_stretching", current_img.copy()))

    # Step 5: Histogram equalization untuk kondisi gambar keruh/underwater
    if analysis.get("is_underwater", False):
        print("     [P05] Applying histogram equalization for underwater conditions...")
        current_img = manual_histogram_equalization(current_img)
        preprocessing_steps.append(("histogram_equalization", current_img.copy()))

    # Step 6: Gaussian filter untuk smoothing jika gambar noisy
    if analysis.get("is_noisy", False):
        print("     [P06] Applying Gaussian smoothing...")
        current_img = apply_gaussian_filter_manual(current_img)
        preprocessing_steps.append(("gaussian_filter", current_img.copy()))

    # Step 7: Morphological opening untuk menghilangkan debris pada gambar
    if analysis.get("has_debris", False):
        print("     [P07] Applying morphological opening for debris...")
        current_img = apply_morphological_opening(current_img)
        preprocessing_steps.append(("morphological_opening", current_img.copy()))

    # Step 8: Penajaman gambar menggunakan Laplacian jika gambar blur atau underwater
    if analysis.get("is_blurry", False) or analysis.get("is_underwater", False):
        print("     [P08] Applying Laplacian sharpening...")
        current_img = apply_sharpening_laplace(current_img)
        preprocessing_steps.append(("laplacian_sharpening", current_img.copy()))

    final_rgb = cv2.cvtColor(current_img, cv2.COLOR_GRAY2RGB)

    return final_rgb, preprocessing_steps


# ============================================================================
# FEATURE EXTRACTION METHODS - 3 METODE SESUAI KETENTUAN
# ============================================================================


def extract_color_features(img_input):
    """Extract grayscale histogram features - SESUAI REFERENSI A9"""

    print(f"     [F01] Input shape: {img_input.shape}")

    # Cek apakah input berupa citra RGB
    if len(img_input.shape) == 3 and img_input.shape[2] == 3:
        print("     [F01] Converting RGB to grayscale...")
        img_gray = rgb_to_gray(img_input)
        is_converted = True
    # Jika input sudah grayscale (1 channel)
    elif len(img_input.shape) == 2:
        print("     [F01] Input is already grayscale")
        img_gray = img_input.copy()
        is_converted = False
    else:
        # Format citra tidak didukung
        raise ValueError(f"Unsupported image format: {img_input.shape}")

    # Hitung histogram grayscale (256 bin, rentang 0-255)
    hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()

    # Hitung fitur statistik sederhana dari citra grayscale
    features = {
        "gray_mean": np.mean(img_gray),  # rata-rata
        "gray_std": np.std(img_gray),  # standar deviasi
        "gray_peak": np.argmax(hist_gray),  # indeks bin dengan frekuensi tertinggi
        "brightness": np.mean(img_gray),  # rata-rata kecerahan
        "contrast": np.std(img_gray),  # kontras / standar deviasi
    }

    print(
        f"     [F01] Grayscale features: Mean({features['gray_mean']:.1f}), Peak({features['gray_peak']})"
    )

    vis_img = create_simple_grayscale_visualization(
        img_gray, features, hist_gray, is_converted
    )

    return features, vis_img


def create_simple_grayscale_visualization(
    img_gray, features, hist_gray, is_converted=False
):
    """Create simple grayscale histogram visualization"""

    # Membuat figure matplotlib dengan layout 1 baris 2 kolom
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Grayscale Analysis", fontsize=14, fontweight="bold")

    # 1. Menampilkan gambar grayscale asli/hasil konversi
    ax1.imshow(img_gray, cmap="gray")
    if is_converted:
        ax1.set_title("Converted to Grayscale", fontsize=12)
    else:
        ax1.set_title("Grayscale Image", fontsize=12)
    ax1.axis("off")

    # 2. Menampilkan histogram sederhana dari citra grayscale
    x_range = np.arange(256)
    ax2.fill_between(x_range, hist_gray, color="gray", alpha=0.7)
    ax2.set_title("Histogram", fontsize=12)
    ax2.set_xlabel("Pixel Intensity")
    ax2.set_ylabel("Frequency")
    ax2.set_xlim(0, 255)
    ax2.grid(True, alpha=0.3)

    # Menampilkan statistik sederhana (mean dan peak) pada histogram
    stats_text = f"Mean: {features['gray_mean']:.0f}\nPeak: {features['gray_peak']}"
    ax2.text(
        0.7,
        0.8,
        stats_text,
        transform=ax2.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    # Menyimpan figure ke buffer memory (BytesIO)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)

    # Membaca buffer sebagai gambar PIL, lalu konversi ke format OpenCV (BGR)
    pil_img = Image.open(buf)
    opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    plt.close()
    buf.close()

    return opencv_img


def extract_texture_features(img_gray):
    """Extract LBP texture features"""

    # Parameter untuk LBP: radius dan jumlah titik di sekeliling radius
    radius = 3
    n_points = 8 * radius

    # Hitung Local Binary Pattern (LBP) pada citra grayscale
    lbp = feature.local_binary_pattern(img_gray, n_points, radius, method="uniform")

    # Hitung fitur-fitur tekstur dari hasil LBP dan citra grayscale
    features = {
        "lbp_mean": np.mean(lbp),  # Rata-rata nilai LBP
        "lbp_std": np.std(lbp),  # Standar deviasi nilai LBP
        "lbp_uniformity": len(np.unique(lbp)) / (n_points + 2),  # Uniformitas pola LBP
        "contrast": np.std(img_gray) ** 2,  # Kontras citra (varian)
        "homogeneity": 1.0 / (1.0 + np.var(img_gray)),  # Homogenitas citra
        "energy": np.sum(img_gray**2)
        / (img_gray.shape[0] * img_gray.shape[1]),  # Energi citra
    }

    print(
        f"     [F02] LBP features: mean={features['lbp_mean']:.2f}, uniformity={features['lbp_uniformity']:.3f}"
    )

    # Buat visualisasi hasil analisis tekstur
    vis_img = create_texture_visualization(img_gray, lbp, features)

    return features, vis_img


def create_texture_visualization(img_gray, lbp_image, features):
    """Create LBP texture analysis visualization using matplotlib - MATPLOTLIB ONLY"""

    # Membuat figure matplotlib dengan layout 2x2 untuk analisis LBP
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("LBP Texture Analysis", fontsize=16, fontweight="bold")

    # 1. Menampilkan citra grayscale asli
    ax1.imshow(img_gray, cmap="gray")
    ax1.set_title("Original Grayscale", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # 2. Menampilkan citra hasil LBP (peta pola tekstur)
    ax2.imshow(lbp_image, cmap="hot")
    ax2.set_title("LBP Pattern Map", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # 3. Menampilkan histogram distribusi nilai LBP
    lbp_hist, bins = np.histogram(
        lbp_image.ravel(), bins=50, range=(0, np.max(lbp_image))
    )
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax3.bar(
        bin_centers,
        lbp_hist,
        width=bins[1] - bins[0],
        color="orange",
        alpha=0.7,
        edgecolor="black",
    )
    ax3.set_title("LBP Histogram Distribution", fontsize=12, fontweight="bold")
    ax3.set_xlabel("LBP Value")
    ax3.set_ylabel("Frequency")
    ax3.set_xlim(0, 255)
    ax3.grid(True, alpha=0.3)

    # Menampilkan statistik LBP (mean, std, uniformity) pada histogram
    lbp_stats = f"Mean: {features['lbp_mean']:.2f}\nStd: {features['lbp_std']:.2f}\nUniformity: {features['lbp_uniformity']:.3f}"
    ax3.text(
        0.02,
        0.98,
        lbp_stats,
        transform=ax3.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 4. Analisis kualitas tekstur (indikator dan klasifikasi)
    ax4.axis("off")

    # Membuat daftar analisis kualitas tekstur berdasarkan fitur LBP
    texture_analysis = [
        f"LBP Mean: {features['lbp_mean']:.2f}",
        f"LBP Std Dev: {features['lbp_std']:.2f}",
        f"LBP Uniformity: {features['lbp_uniformity']:.3f}",
        f"Contrast: {features.get('contrast', 0):.2f}",
        f"Homogeneity: {features.get('homogeneity', 0):.3f}",
        f"Energy: {features.get('energy', 0):.3f}",
        "",
        "Texture Classification:",
    ]

    # Klasifikasi tipe tekstur berdasarkan nilai uniformity dan std dev
    if features["lbp_uniformity"] > 0.1:
        texture_analysis.append("• Uniform Texture (Synthetic)")
    elif features["lbp_std"] < 10:
        texture_analysis.append("• Low Variation (Smooth)")
    else:
        texture_analysis.append("• High Variation (Complex)")

    # Klasifikasi pola intensitas LBP
    if features["lbp_mean"] < 10:
        texture_analysis.append("• Low Intensity Pattern")
    elif features["lbp_mean"] > 20:
        texture_analysis.append("• High Intensity Pattern")
    else:
        texture_analysis.append("• Moderate Intensity Pattern")

    # Menampilkan analisis tekstur pada subplot ke-4
    analysis_text = "\n".join(texture_analysis)
    ax4.text(
        0.1,
        0.9,
        analysis_text,
        transform=ax4.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )

    ax4.set_title("Texture Quality Analysis", fontsize=12, fontweight="bold")

    plt.tight_layout()

    # Menyimpan figure ke buffer memory (BytesIO)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    # Membaca buffer sebagai gambar PIL, lalu konversi ke format OpenCV (BGR)
    pil_img = Image.open(buf)
    opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    plt.close()
    buf.close()

    return opencv_img


def extract_shape_features(img_gray):
    """Extract contour-based shape features"""

    # Lakukan threshold biner pada citra grayscale
    _, threshold = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # Temukan kontur pada citra hasil threshold
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Hitung total area dari semua kontur yang signifikan (area > 100 piksel)
    total_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100)

    shapes_detected = {}  # Dictionary untuk menyimpan tipe shape yang terdeteksi
    circularities = []  # List untuk menyimpan nilai circularity setiap kontur
    solidities = []  # List untuk menyimpan nilai solidity setiap kontur

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 100:
            # Hitung keliling kontur
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                # Hitung circularity (kelingkaran) kontur
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                circularities.append(circularity)

                # Hitung solidity (kepadatan) kontur
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    solidities.append(solidity)

                # Klasifikasikan bentuk berdasarkan nilai circularity
                if circularity > 0.7:
                    shapes_detected[i] = "Circle"
                elif circularity > 0.5:
                    shapes_detected[i] = "Oval"
                else:
                    shapes_detected[i] = "Irregular"

    # Hitung fitur-fitur shape secara keseluruhan
    features = {
        "total_shapes": len([c for c in contours if cv2.contourArea(c) > 100]),
        "total_area": total_area,
        "circularity": np.mean(circularities) if circularities else 0,
        "solidity": np.mean(solidities) if solidities else 0,
        "dominant_shape": max(
            shapes_detected.values(), key=list(shapes_detected.values()).count
        )
        if shapes_detected
        else "None",
        "aspect_ratio": 1.0,
    }

    print(
        f"     [F03] Shape features: {features['total_shapes']} shapes, circularity={features['circularity']:.3f}"
    )

    vis_img = create_shape_identification_visualization(
        img_gray, threshold, contours, features, shapes_detected
    )

    return features, vis_img


def create_shape_identification_visualization(
    img_gray, threshold, contours, features, shapes_detected
):
    """Create shape analysis visualization using matplotlib - MATPLOTLIB ONLY"""

    # Membuat figure matplotlib dengan layout 2x2 untuk analisis shape
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Shape Identification Analysis", fontsize=16, fontweight="bold")

    # 1. Menampilkan citra grayscale asli
    ax1.imshow(img_gray, cmap="gray")
    ax1.set_title("Original Grayscale", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # 2. Menampilkan hasil threshold biner
    ax2.imshow(threshold, cmap="gray")
    ax2.set_title("Binary Threshold", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # 3. Menampilkan deteksi kontur dengan pewarnaan berbeda
    ax3.imshow(img_gray, cmap="gray")

    # Daftar warna untuk membedakan kontur
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "cyan", "magenta"]

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 100:
            color = colors[i % len(colors)]

            # Mengambil titik-titik kontur
            contour_points = contour.reshape(-1, 2)
            # Menggambar kontur pada subplot
            ax3.plot(
                contour_points[:, 0], contour_points[:, 1], color=color, linewidth=2
            )

            # Menambahkan label tipe shape pada tengah bounding box kontur
            x, y, w, h = cv2.boundingRect(contour)
            shape_type = shapes_detected.get(i, "Unknown")
            ax3.text(
                x + w // 2,
                y + h // 2,
                f"{shape_type}",
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    ax3.set_title("Detected Contours & Shapes", fontsize=12, fontweight="bold")
    ax3.axis("off")

    # 4. Ringkasan analisis shape
    ax4.axis("off")

    # Membuat ringkasan analisis shape
    shape_analysis = [
        f"Total Shapes: {features['total_shapes']}",
        f"Dominant Shape: {features['dominant_shape']}",
        f"Total Area: {features['total_area']:.0f} px²",
        f"Average Circularity: {features['circularity']:.3f}",
        f"Average Solidity: {features['solidity']:.3f}",
        f"Aspect Ratio: {features.get('aspect_ratio', 0):.2f}",
        "",
        "Shape Distribution:",
    ]

    # Menghitung jumlah masing-masing tipe shape
    shape_counts = {}
    for shape_type in shapes_detected.values():
        shape_counts[shape_type] = shape_counts.get(shape_type, 0) + 1

    for shape_type, count in shape_counts.items():
        shape_analysis.append(f"• {shape_type}: {count}")

    shape_analysis.extend(
        [
            "",
            "Classification Indicators:",
        ]
    )

    # Indikator klasifikasi berdasarkan fitur shape
    if features["circularity"] > 0.7:
        shape_analysis.append("• High Circularity (Round objects)")
    elif features["circularity"] < 0.3:
        shape_analysis.append("• Low Circularity (Irregular shapes)")

    if features["solidity"] > 0.8:
        shape_analysis.append("• High Solidity (Solid objects)")
    elif features["solidity"] < 0.6:
        shape_analysis.append("• Low Solidity (Fragmented)")

    if features["total_shapes"] > 5:
        shape_analysis.append("• Multiple fragments detected")

    # Menampilkan ringkasan analisis pada subplot ke-4
    analysis_text = "\n".join(shape_analysis)
    ax4.text(
        0.1,
        0.9,
        analysis_text,
        transform=ax4.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
    )

    ax4.set_title("Shape Analysis Summary", fontsize=12, fontweight="bold")

    plt.tight_layout()

    # Menyimpan figure ke buffer memory (BytesIO)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    # Membaca buffer sebagai gambar PIL, lalu konversi ke format OpenCV (BGR)
    pil_img = Image.open(buf)
    opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    plt.close()
    buf.close()

    return opencv_img


def apply_preprocessing_pipeline(img_rgb):
    """Apply complete preprocessing pipeline dengan 8 metode"""

    # Analisis kualitas gambar
    analysis = analyze_image_quality(img_rgb)

    # preprocessing adaptif berdasarkan hasil analisis
    preprocessed_img, preprocessing_steps = adaptive_preprocessing(img_rgb, analysis)

    return preprocessed_img, preprocessing_steps, analysis
