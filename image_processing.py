import cv2
import numpy as np
from skimage import feature

# ============================================================================
# UTILITY FUNCTIONS - Color Space Conversions (Manual Implementation)
# ============================================================================


def rgb_to_gray_manual(img_rgb):
    """Manual RGB to Grayscale conversion (dari referensi)"""
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
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)


def hsv_to_rgb(img_hsv):
    """Convert HSV to RGB"""
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)


# ============================================================================
# IMAGE QUALITY ANALYSIS - Enhanced untuk kondisi database kompleks
# ============================================================================


def analyze_image_quality(img_gray):
    """Enhanced analysis untuk kondisi database yang kompleks"""
    analysis = {}

    # Basic analysis
    brightness = np.mean(img_gray)
    analysis["brightness"] = brightness
    analysis["is_dark"] = brightness < 80
    analysis["is_bright"] = brightness > 180

    contrast = img_gray.std()
    analysis["contrast"] = contrast
    analysis["is_low_contrast"] = contrast < 30

    # Enhanced detection for specific conditions
    blur_score = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    analysis["blur_score"] = blur_score
    analysis["is_blurry"] = blur_score < 100

    # Noise detection
    noise_level = np.std(cv2.medianBlur(img_gray, 5) - img_gray)
    analysis["noise_level"] = noise_level
    analysis["is_noisy"] = noise_level > 15

    # Edge density for debris detection
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    analysis["edge_density"] = edge_density

    # Special condition detection berdasarkan database
    analysis["is_underwater"] = brightness < 100 and contrast < 25  # murky_water
    analysis["is_reflective"] = brightness > 150 and contrast < 30  # wet_condition
    analysis["has_debris"] = edge_density > 0.15  # with_debris
    analysis["is_faded"] = brightness > 130 and contrast < 40  # faded_colors
    analysis["is_transparent"] = (
        brightness > 200 and contrast < 20
    )  # transparent_polyester

    return analysis


# ============================================================================
# 8 METODE PREPROCESSING DARI REFERENSI
# ============================================================================


def manual_brightness_adjustment(img_gray, brightness=80):
    """METODE 1: Manual brightness adjustment (dari referensi A4)"""
    H, W = img_gray.shape[:2]
    for i in range(H):
        for j in range(W):
            a = img_gray.item(i, j)
            b = np.clip(a + brightness, 0, 255)
            img_gray[i, j] = b
    return img_gray


def manual_contrast_enhancement(img_gray, contrast=1.7):
    """METODE 2: Manual contrast enhancement (dari referensi A5)"""
    H, W = img_gray.shape[:2]
    for i in range(H):
        for j in range(W):
            a = img_gray.item(i, j)
            b = np.clip(a * contrast, 0, 255)
            img_gray[i, j] = b
    return img_gray


def manual_contrast_stretching(img_gray):
    """METODE 3: Manual contrast stretching (dari referensi A6)"""
    H, W = img_gray.shape[:2]
    minV = np.min(img_gray)
    maxV = np.max(img_gray)

    for i in range(H):
        for j in range(W):
            a = img_gray.item(i, j)
            b = float(a - minV) / (maxV - minV) * 255
            img_gray[i, j] = b
    return img_gray


def manual_histogram_equalization(img_gray):
    """METODE 4: Manual histogram equalization (dari referensi A11)"""
    hist, bins = np.histogram(img_gray.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype("uint8")
    return cdf[img_gray]


def apply_median_filter_manual(img_gray):
    """METODE 5: Manual median filter 7x7 (dari referensi D5)"""
    hasil = img_gray.copy()
    h, w = img_gray.shape

    for i in range(3, h - 3):
        for j in range(3, w - 3):
            neighbors = [
                img_gray[i + k, j + l] for k in range(-3, 4) for l in range(-3, 4)
            ]
            neighbors.sort()
            hasil[i, j] = neighbors[24]  # median dari 49 elemen
    return hasil


def apply_gaussian_filter_manual(img_gray):
    """METODE 6: Gaussian filter 5x5 (dari referensi D3)"""
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

    # Manual convolution dari referensi
    return manual_convolution_2d(img_gray, kernel)


def apply_morphological_opening(img_gray):
    """METODE 7: Morphological opening (dari referensi G1)"""
    # Convert to binary first
    ret, threshold = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hasil = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, strel)
    return hasil


def apply_sharpening_laplace(img_gray):
    """METODE 8: Laplacian sharpening (dari referensi D4)"""
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

    return manual_convolution_2d(img_gray, kernel)


def manual_convolution_2d(X, F):
    """Manual 2D convolution implementation (dari referensi D1)"""
    X_height = X.shape[0]
    X_width = X.shape[1]
    F_height = F.shape[0]
    F_width = F.shape[1]
    H = (F_height) // 2
    W = (F_width) // 2
    out = np.zeros((X_height, X_width))

    for i in np.arange(H + 1, X_height - H):
        for j in np.arange(W + 1, X_width - W):
            sum_val = 0
            for k in np.arange(-H, H + 1):
                for l in np.arange(-W, W + 1):
                    a = X[i + k, j + l]
                    w = F[H + k, W + l]
                    sum_val += w * a
            out[i, j] = sum_val
    return out.astype(np.uint8)


# ============================================================================
# ADAPTIVE PREPROCESSING UNTUK DATABASE KOMPLEKS
# ============================================================================


def adaptive_preprocessing(img_rgb, analysis):
    """Enhanced adaptive preprocessing untuk kondisi database kompleks"""
    img_gray = rgb_to_gray_manual(img_rgb)
    preprocessing_steps = []

    preprocessing_steps.append(("original", img_gray.copy()))

    # METODE 1: Brightness adjustment untuk kondisi gelap (clear_water dalam kondisi gelap)
    if analysis["is_dark"]:
        print("     🌙 Applying brightness enhancement...")
        img_gray = manual_brightness_adjustment(img_gray.copy(), brightness=60)
        preprocessing_steps.append(("brightness_enhancement", img_gray.copy()))

    # METODE 2: Contrast enhancement untuk kondisi underwater/murky
    if analysis["is_underwater"]:
        print("     🌊 Applying contrast enhancement for underwater...")
        img_gray = manual_contrast_enhancement(img_gray.copy(), contrast=2.0)
        preprocessing_steps.append(("underwater_contrast", img_gray.copy()))

    # METODE 3: Contrast stretching untuk kontras rendah
    if analysis["is_low_contrast"] or analysis["is_faded"]:
        print("     📈 Applying contrast stretching...")
        img_gray = manual_contrast_stretching(img_gray.copy())
        preprocessing_steps.append(("contrast_stretching", img_gray.copy()))

    # METODE 4: Histogram equalization untuk pencahayaan tidak merata
    if analysis["is_low_contrast"] or analysis["is_underwater"]:
        print("     📊 Applying histogram equalization...")
        img_gray = manual_histogram_equalization(img_gray)
        preprocessing_steps.append(("histogram_equalization", img_gray.copy()))

    # METODE 5: Median filter untuk noise dan debris
    if analysis["is_noisy"] or analysis["has_debris"]:
        print("     🧹 Applying median filter for noise/debris reduction...")
        img_gray = apply_median_filter_manual(img_gray.copy())
        preprocessing_steps.append(("median_filter", img_gray.copy()))

    # METODE 6: Gaussian filter untuk smoothing (various_lighting)
    if analysis["edge_density"] > 0.2:  # Too much detail/noise
        print("     🌀 Applying Gaussian smoothing...")
        img_gray = apply_gaussian_filter_manual(img_gray.copy().astype(np.float32))
        img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
        preprocessing_steps.append(("gaussian_smoothing", img_gray.copy()))

    # METODE 7: Morphological opening untuk debris removal
    if analysis["has_debris"]:
        print("     🔧 Applying morphological opening for debris...")
        opened = apply_morphological_opening(img_gray.copy())
        # Convert back to grayscale
        img_gray = cv2.bitwise_not(opened)
        preprocessing_steps.append(("morphological_opening", img_gray.copy()))

    # METODE 8: Laplacian sharpening untuk blur dan wet condition
    if analysis["is_blurry"] or analysis["is_reflective"]:
        print("     🔍 Applying Laplacian sharpening...")
        sharpened = apply_sharpening_laplace(img_gray.copy().astype(np.float32))
        img_gray = cv2.addWeighted(img_gray.astype(np.float32), 1.2, sharpened, -0.2, 0)
        img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
        preprocessing_steps.append(("laplacian_sharpening", img_gray.copy()))

    # Convert back to RGB for compatibility
    img_rgb_result = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    return img_rgb_result, preprocessing_steps


# ============================================================================
# FEATURE EXTRACTION METHODS (tetap sama - 1 metode per kategori)
# ============================================================================


def extract_color_features(img_rgb):
    """Extract color features using HSV statistics only"""
    img_hsv = rgb_to_hsv(img_rgb)
    features = {}

    # Calculate HSV statistics
    for i, channel in enumerate(["H", "S", "V"]):
        channel_data = img_hsv[:, :, i]
        features[f"{channel}_mean"] = float(np.mean(channel_data))
        features[f"{channel}_std"] = float(np.std(channel_data))
        features[f"{channel}_min"] = float(np.min(channel_data))
        features[f"{channel}_max"] = float(np.max(channel_data))

    # Create visualization
    vis_img = create_color_visualization(img_rgb, features)

    print(f"     ✅ Extracted {len(features)} HSV color features")
    return features, vis_img


def create_color_visualization(img_rgb, features):
    """Create color features visualization"""
    h, w = img_rgb.shape[:2]
    vis_img = np.zeros((h, w * 2, 3), dtype=np.uint8)

    vis_img[:, :w] = img_rgb

    analysis_img = np.zeros((h, w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        analysis_img,
        f"H_mean: {features['H_mean']:.1f}",
        (10, 30),
        font,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        analysis_img,
        f"S_mean: {features['S_mean']:.1f}",
        (10, 50),
        font,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        analysis_img,
        f"V_mean: {features['V_mean']:.1f}",
        (10, 70),
        font,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        analysis_img,
        f"S_std: {features['S_std']:.1f}",
        (10, 110),
        font,
        0.5,
        (255, 255, 255),
        1,
    )

    vis_img[:, w:] = analysis_img
    return vis_img


def extract_texture_features(img_gray):
    """Extract texture features using LBP only"""
    print("     🔄 Calculating LBP texture features...")

    try:
        radius = 3
        n_points = 24
        lbp = feature.local_binary_pattern(img_gray, n_points, radius, method="uniform")

        lbp_hist, _ = np.histogram(
            lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2)
        )
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= lbp_hist.sum() + 1e-7

        features = {}
        for i in range(min(10, len(lbp_hist))):
            features[f"lbp_bin_{i}"] = float(lbp_hist[i])

        features["lbp_mean"] = float(np.mean(lbp))
        features["lbp_std"] = float(np.std(lbp))
        features["lbp_variance"] = float(np.var(lbp))
        features["lbp_uniformity"] = float(np.sum(lbp_hist**2))

        vis_img = create_texture_visualization(img_gray, lbp, features)

        print(f"     ✅ Extracted {len(features)} LBP texture features")
        return features, vis_img

    except Exception as e:
        print(f"Warning: LBP calculation failed: {e}")
        features = {}
        for i in range(10):
            features[f"lbp_bin_{i}"] = 0.0
        features.update(
            {
                "lbp_mean": 0.0,
                "lbp_std": 0.0,
                "lbp_variance": 0.0,
                "lbp_uniformity": 0.0,
            }
        )
        vis_img = np.zeros((img_gray.shape[0], img_gray.shape[1] * 3), dtype=np.uint8)
        return features, vis_img


def create_texture_visualization(img_gray, lbp_image, features):
    """Create texture features visualization"""
    h, w = img_gray.shape
    vis_img = np.zeros((h, w * 3), dtype=np.uint8)

    vis_img[:, :w] = img_gray

    if lbp_image is not None and lbp_image.shape == img_gray.shape:
        lbp_normalized = (
            (lbp_image - lbp_image.min())
            / (lbp_image.max() - lbp_image.min() + 1e-7)
            * 255
        ).astype(np.uint8)
        vis_img[:, w : 2 * w] = lbp_normalized
    else:
        vis_img[:, w : 2 * w] = img_gray

    analysis_img = np.zeros((h, w), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        analysis_img,
        f"LBP Mean: {features.get('lbp_mean', 0):.2f}",
        (10, 30),
        font,
        0.4,
        255,
        1,
    )
    cv2.putText(
        analysis_img,
        f"LBP Std: {features.get('lbp_std', 0):.2f}",
        (10, 50),
        font,
        0.4,
        255,
        1,
    )
    cv2.putText(
        analysis_img,
        f"Uniformity: {features.get('lbp_uniformity', 0):.3f}",
        (10, 90),
        font,
        0.4,
        255,
        1,
    )

    vis_img[:, 2 * w :] = analysis_img
    return vis_img


def extract_shape_features(img_gray):
    """Extract shape features using contour analysis only (dari referensi H3)"""
    features = {}

    # Apply Otsu threshold seperti di referensi
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours seperti di referensi
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        features["area"] = float(area)
        features["perimeter"] = float(perimeter)

        if perimeter > 0:
            features["circularity"] = float(4 * np.pi * area / (perimeter * perimeter))
        else:
            features["circularity"] = 0.0

        x, y, w, h = cv2.boundingRect(largest_contour)
        features["aspect_ratio"] = float(w) / h if h > 0 else 0.0
        features["extent"] = float(area) / (w * h) if w * h > 0 else 0.0

        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        features["solidity"] = float(area) / hull_area if hull_area > 0 else 0.0

        # Hu moments (first 3 only)
        moments = cv2.moments(largest_contour)
        if moments["m00"] > 0:
            hu_moments = cv2.HuMoments(moments)
            for i in range(3):
                hu = hu_moments[i][0]
                features[f"hu_moment_{i}"] = float(
                    -np.sign(hu) * np.log10(abs(hu)) if hu != 0 else 0
                )

    vis_img = create_shape_visualization(img_gray, binary, contours, features)

    print(f"     ✅ Extracted {len(features)} contour-based shape features")
    return features, vis_img


def create_shape_visualization(img_gray, binary, contours, features):
    """Create shape features visualization"""
    h, w = img_gray.shape
    vis_img = np.zeros((h, w * 3), dtype=np.uint8)

    vis_img[:, :w] = img_gray
    vis_img[:, w : 2 * w] = binary

    contour_img = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_img, contours, -1, 255, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        contour_img, f"Area: {features.get('area', 0):.0f}", (10, 30), font, 0.4, 255, 1
    )
    cv2.putText(
        contour_img,
        f"Circ: {features.get('circularity', 0):.2f}",
        (10, 50),
        font,
        0.4,
        255,
        1,
    )
    cv2.putText(
        contour_img,
        f"Sol: {features.get('solidity', 0):.2f}",
        (10, 90),
        font,
        0.4,
        255,
        1,
    )

    vis_img[:, 2 * w :] = contour_img
    return vis_img
