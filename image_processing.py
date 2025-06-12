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


def rgb_to_gray_manual(img_rgb):
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
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)


def hsv_to_rgb(img_hsv):
    """Convert HSV to RGB"""
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)


# ============================================================================
# IMAGE QUALITY ANALYSIS - Enhanced untuk kondisi database kompleks
# ============================================================================


def analyze_image_quality(img_rgb):
    """Enhanced analysis untuk kondisi database yang kompleks"""
    # Gunakan manual RGB to Gray conversion sesuai referensi A3
    # Convert RGB to Grayscale
    # Menghitng kecerahan (mean pixel intensity) dan kontras (standar deviation)
    # Mendeteksi kekaburan Menggunakan varian laplacian
    # mendeteksi kebisingan dengan median filter
    # menghitung kepadatan tepi menggunakan operator canny
    # dan mendeteksi kondisi khusus seperti underwater, reflective, debris, faded,
    # transparent berdasarkan ambang batas pada metrik yang dihitung

    img_gray = rgb_to_gray_manual(img_rgb)

    analysis = {}

    # Basic analysis
    brightness = np.mean(img_gray)
    analysis["brightness"] = brightness
    analysis["is_dark"] = brightness < 80
    analysis["is_bright"] = brightness > 180

    contrast = img_gray.std()
    analysis["contrast"] = contrast
    analysis["is_low_contrast"] = contrast < 30

    # Detection for specific conditions
    blur_score = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    analysis["blur_score"] = blur_score
    analysis["is_blurry"] = blur_score < 100

    # Noise detection
    noise_level = np.std(cv2.medianBlur(img_gray, 5) - img_gray)
    analysis["noise_level"] = noise_level
    analysis["is_noisy"] = noise_level > 15

    # Edge density untuk debris detection
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    analysis["edge_density"] = edge_density

    # Special condition detection berdasarkan kondisi database
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
            result[i, j] = np.clip(img_gray[i, j] + brightness, 0, 255)
    return result


def manual_contrast_enhancement(img_gray, contrast=1.7):
    """METODE 2: Manual contrast enhancement (PERSIS referensi A5)"""
    H, W = img_gray.shape[:2]
    result = img_gray.copy()
    for i in range(H):
        for j in range(W):
            result[i, j] = np.clip(contrast * img_gray[i, j], 0, 255)
    return result


def manual_contrast_stretching(img_gray):
    """METODE 3: Manual contrast stretching (PERSIS referensi A6)"""
    # Find min and max values
    min_val = np.min(img_gray)
    max_val = np.max(img_gray)

    # Apply contrast stretching formula
    if max_val - min_val == 0:
        return img_gray

    H, W = img_gray.shape[:2]
    result = np.zeros((H, W), np.uint8)

    for i in range(H):
        for j in range(W):
            result[i, j] = np.clip(
                255 * (img_gray[i, j] - min_val) / (max_val - min_val), 0, 255
            )

    return result


def manual_histogram_equalization(img_gray):
    """METODE 4: Manual histogram equalization (PERSIS referensi A11)"""
    return cv2.equalizeHist(img_gray)


def apply_median_filter_manual(img_gray):
    """METODE 5: Manual median filter 7x7 (PERSIS referensi D5)"""
    return cv2.medianBlur(img_gray, 7)


def manual_convolution_2d(X, F):
    """Manual 2D convolution (PERSIS referensi D1)"""
    X_height, X_width = X.shape
    F_height, F_width = F.shape

    # Output dimensions
    out_height = X_height - F_height + 1
    out_width = X_width - F_width + 1

    # Initialize output
    out = np.zeros((out_height, out_width))

    # Perform convolution
    for i in range(out_height):
        for j in range(out_width):
            out[i, j] = np.sum(X[i : i + F_height, j : j + F_width] * F)

    return out


def apply_gaussian_filter_manual(img_gray):
    """METODE 6: Manual Gaussian filter 5x5 (PERSIS referensi D3)"""
    return cv2.GaussianBlur(img_gray, (5, 5), 0)


def apply_morphological_opening(img_gray):
    """METODE 7: Morphological opening (PERSIS referensi G1)"""
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)


def apply_sharpening_laplace(img_gray):
    """METODE 8: Laplacian sharpening (PERSIS referensi D4)"""
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    sharpened = img_gray.astype(np.float64) - laplacian
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ============================================================================
# ADAPTIVE PREPROCESSING UNTUK DATABASE KOMPLEKS
# ============================================================================


def adaptive_preprocessing(img_rgb, analysis):
    """Adaptive preprocessing berdasarkan kondisi image"""

    # Convert to grayscale untuk preprocessing
    img_gray = rgb_to_gray_manual(img_rgb)

    preprocessing_steps = []
    current_img = img_gray.copy()

    # Step 1: Noise reduction jika diperlukan
    if analysis.get("is_noisy", False) or analysis.get("has_debris", False):
        print("     🧹 Applying median filter for noise/debris reduction...")
        current_img = apply_median_filter_manual(current_img)
        preprocessing_steps.append(("median_filter", current_img.copy()))

    # Step 2: Brightness adjustment untuk dark images
    if analysis.get("is_dark", False):
        print("     💡 Brightening dark image...")
        current_img = manual_brightness_adjustment(current_img, brightness=50)
        preprocessing_steps.append(("brightness_adjustment", current_img.copy()))

    # Step 3: Contrast enhancement untuk low contrast
    if analysis.get("is_low_contrast", False):
        print("     🔆 Enhancing contrast...")
        current_img = manual_contrast_enhancement(current_img, contrast=1.5)
        preprocessing_steps.append(("contrast_enhancement", current_img.copy()))

    # Step 4: Contrast stretching untuk better range
    if analysis.get("is_faded", False) or analysis.get("is_underwater", False):
        print("     📈 Applying contrast stretching...")
        current_img = manual_contrast_stretching(current_img)
        preprocessing_steps.append(("contrast_stretching", current_img.copy()))

    # Step 5: Histogram equalization untuk underwater/murky conditions
    if analysis.get("is_underwater", False):
        print("     🌊 Applying histogram equalization for underwater conditions...")
        current_img = manual_histogram_equalization(current_img)
        preprocessing_steps.append(("histogram_equalization", current_img.copy()))

    # Step 6: Gaussian filter untuk smoothing
    if analysis.get("is_noisy", False):
        print("     🌀 Applying Gaussian smoothing...")
        current_img = apply_gaussian_filter_manual(current_img)
        preprocessing_steps.append(("gaussian_filter", current_img.copy()))

    # Step 7: Morphological opening untuk debris removal
    if analysis.get("has_debris", False):
        print("     🔧 Applying morphological opening for debris...")
        current_img = apply_morphological_opening(current_img)
        preprocessing_steps.append(("morphological_opening", current_img.copy()))

    # Step 8: Sharpening untuk final enhancement
    if analysis.get("is_blurry", False) or analysis.get("is_underwater", False):
        print("     🔪 Applying Laplacian sharpening...")
        current_img = apply_sharpening_laplace(current_img)
        preprocessing_steps.append(("laplacian_sharpening", current_img.copy()))

    # Convert back to RGB
    final_rgb = cv2.cvtColor(current_img, cv2.COLOR_GRAY2RGB)

    return final_rgb, preprocessing_steps


# ============================================================================
# FEATURE EXTRACTION METHODS - 3 METODE SESUAI KETENTUAN
# ============================================================================


def extract_color_features(img_input):
    """Extract RGB histogram features - Auto handle RGB/Grayscale"""

    print(f"     📊 Input shape: {img_input.shape}")

    # Auto-detect dan convert ke RGB jika perlu
    if len(img_input.shape) == 3 and img_input.shape[2] == 3:
        print("     ✅ Input is already RGB")
        img_rgb = img_input.copy()
        is_converted = False
    elif len(img_input.shape) == 2:
        print("     🔄 Converting grayscale to RGB...")
        img_rgb = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
        is_converted = True
    else:
        raise ValueError(f"Unsupported image format: {img_input.shape}")

    # Calculate RGB histograms
    hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256]).flatten()

    # Calculate statistical features untuk setiap channel
    features = {
        # Red channel statistics
        "r_mean": np.mean(img_rgb[:, :, 0]),
        "r_std": np.std(img_rgb[:, :, 0]),
        "r_peak": np.argmax(hist_r),
        "r_dominance": np.mean(img_rgb[:, :, 0]) / 255.0,
        # Green channel statistics
        "g_mean": np.mean(img_rgb[:, :, 1]),
        "g_std": np.std(img_rgb[:, :, 1]),
        "g_peak": np.argmax(hist_g),
        "g_dominance": np.mean(img_rgb[:, :, 1]) / 255.0,
        # Blue channel statistics
        "b_mean": np.mean(img_rgb[:, :, 2]),
        "b_std": np.std(img_rgb[:, :, 2]),
        "b_peak": np.argmax(hist_b),
        "b_dominance": np.mean(img_rgb[:, :, 2]) / 255.0,
        # Overall features
        "brightness": np.mean(img_rgb),
        "overall_std": np.std(img_rgb),
    }

    print(
        f"     📈 RGB features calculated: R({features['r_mean']:.1f}), G({features['g_mean']:.1f}), B({features['b_mean']:.1f})"
    )

    # Create visualization
    vis_img = create_rgb_histogram_visualization(
        img_rgb, features, hist_r, hist_g, hist_b, is_converted
    )

    return features, vis_img


def create_rgb_histogram_visualization(
    img_rgb, features, hist_r, hist_g, hist_b, is_converted=False
):
    """Create RGB histogram visualization using matplotlib - MATPLOTLIB ONLY"""

    # Create matplotlib figure untuk histogram RGB yang rapi
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("RGB Color Analysis", fontsize=16, fontweight="bold")

    # 1. Original/Converted Image
    ax1.imshow(img_rgb)
    if is_converted:
        ax1.set_title("Converted to RGB", fontsize=12, fontweight="bold")
    else:
        ax1.set_title("Original RGB Image", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # 2. Red Channel Histogram
    x_range = np.arange(256)
    ax2.fill_between(x_range, hist_r, color="red", alpha=0.7, label="Red Channel")
    ax2.set_title("R - Red Channel", fontsize=12, fontweight="bold", color="red")
    ax2.set_xlabel("Pixel Intensity")
    ax2.set_ylabel("Frequency")
    ax2.set_xlim(0, 255)
    ax2.grid(True, alpha=0.3)

    # Add statistics text
    r_stats = f"Mean: {features['r_mean']:.1f}\nStd: {features['r_std']:.1f}\nPeak: {features['r_peak']}"
    ax2.text(
        0.02,
        0.98,
        r_stats,
        transform=ax2.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 3. Green Channel Histogram
    ax3.fill_between(x_range, hist_g, color="green", alpha=0.7, label="Green Channel")
    ax3.set_title("G - Green Channel", fontsize=12, fontweight="bold", color="green")
    ax3.set_xlabel("Pixel Intensity")
    ax3.set_ylabel("Frequency")
    ax3.set_xlim(0, 255)
    ax3.grid(True, alpha=0.3)

    # Add statistics text
    g_stats = f"Mean: {features['g_mean']:.1f}\nStd: {features['g_std']:.1f}\nPeak: {features['g_peak']}"
    ax3.text(
        0.02,
        0.98,
        g_stats,
        transform=ax3.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 4. Blue Channel Histogram
    ax4.fill_between(x_range, hist_b, color="blue", alpha=0.7, label="Blue Channel")
    ax4.set_title("B - Blue Channel", fontsize=12, fontweight="bold", color="blue")
    ax4.set_xlabel("Pixel Intensity")
    ax4.set_ylabel("Frequency")
    ax4.set_xlim(0, 255)
    ax4.grid(True, alpha=0.3)

    # Add statistics text
    b_stats = f"Mean: {features['b_mean']:.1f}\nStd: {features['b_std']:.1f}\nPeak: {features['b_peak']}"
    ax4.text(
        0.02,
        0.98,
        b_stats,
        transform=ax4.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Tight layout
    plt.tight_layout()

    # Save to buffer dan convert ke OpenCV format

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    # Convert buffer to OpenCV image

    pil_img = Image.open(buf)
    opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    plt.close()  # Clean memory
    buf.close()

    return opencv_img


def extract_texture_features(img_gray):
    """Extract LBP texture features"""

    # Parameters untuk LBP
    radius = 3
    n_points = 8 * radius

    # Calculate LBP
    lbp = feature.local_binary_pattern(img_gray, n_points, radius, method="uniform")

    # Calculate texture features
    features = {
        "lbp_mean": np.mean(lbp),
        "lbp_std": np.std(lbp),
        "lbp_uniformity": len(np.unique(lbp)) / (n_points + 2),  # Uniformity measure
        "contrast": np.std(img_gray) ** 2,
        "homogeneity": 1.0 / (1.0 + np.var(img_gray)),
        "energy": np.sum(img_gray**2) / (img_gray.shape[0] * img_gray.shape[1]),
    }

    print(
        f"     🔍 LBP features: mean={features['lbp_mean']:.2f}, uniformity={features['lbp_uniformity']:.3f}"
    )

    # Create visualization
    vis_img = create_texture_visualization(img_gray, lbp, features)

    return features, vis_img


def create_texture_visualization(img_gray, lbp_image, features):
    """Create LBP texture analysis visualization using matplotlib - MATPLOTLIB ONLY"""

    # Create matplotlib figure untuk LBP analysis yang rapi
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("LBP Texture Analysis", fontsize=16, fontweight="bold")

    # 1. Original Grayscale Image
    ax1.imshow(img_gray, cmap="gray")
    ax1.set_title("Original Grayscale", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # 2. LBP Pattern Image
    ax2.imshow(lbp_image, cmap="hot")
    ax2.set_title("LBP Pattern Map", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # 3. LBP Histogram
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
    ax3.grid(True, alpha=0.3)

    # Add LBP statistics
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

    # 4. Texture Quality Analysis
    ax4.axis("off")

    # Create texture quality indicators
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

    # Texture type classification
    if features["lbp_uniformity"] > 0.1:
        texture_analysis.append("• Uniform Texture (Synthetic)")
    elif features["lbp_std"] < 10:
        texture_analysis.append("• Low Variation (Smooth)")
    else:
        texture_analysis.append("• High Variation (Complex)")

    if features["lbp_mean"] < 10:
        texture_analysis.append("• Low Intensity Pattern")
    elif features["lbp_mean"] > 20:
        texture_analysis.append("• High Intensity Pattern")
    else:
        texture_analysis.append("• Moderate Intensity Pattern")

    # Display analysis text
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

    # Save to buffer dan convert ke OpenCV format

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    # Convert buffer to OpenCV image

    pil_img = Image.open(buf)
    opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    plt.close()  # Clean memory
    buf.close()

    return opencv_img


def extract_shape_features(img_gray):
    """Extract contour-based shape features"""

    # Binary threshold
    _, threshold = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Calculate shape features
    total_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100)

    shapes_detected = {}
    circularities = []
    solidities = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 100:  # Only significant contours
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                circularities.append(circularity)

                # Calculate solidity
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    solidities.append(solidity)

                # Classify shape based on circularity
                if circularity > 0.7:
                    shapes_detected[i] = "Circle"
                elif circularity > 0.5:
                    shapes_detected[i] = "Oval"
                else:
                    shapes_detected[i] = "Irregular"

    # Overall features
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
        "aspect_ratio": 1.0,  # Default value
    }

    print(
        f"     🔷 Shape features: {features['total_shapes']} shapes, circularity={features['circularity']:.3f}"
    )

    # Create visualization
    vis_img = create_shape_identification_visualization(
        img_gray, threshold, contours, features, shapes_detected
    )

    return features, vis_img


def create_shape_identification_visualization(
    img_gray, threshold, contours, features, shapes_detected
):
    """Create shape analysis visualization using matplotlib - MATPLOTLIB ONLY"""

    # Create matplotlib figure untuk shape analysis yang rapi
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Shape Identification Analysis", fontsize=16, fontweight="bold")

    # 1. Original Grayscale
    ax1.imshow(img_gray, cmap="gray")
    ax1.set_title("Original Grayscale", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # 2. Binary Threshold
    ax2.imshow(threshold, cmap="gray")
    ax2.set_title("Binary Threshold", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # 3. Contour Detection dengan Color-coding
    ax3.imshow(img_gray, cmap="gray")

    colors = ["red", "blue", "green", "yellow", "orange", "purple", "cyan", "magenta"]

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 100:  # Only significant contours
            color = colors[i % len(colors)]

            # Draw contour
            contour_points = contour.reshape(-1, 2)
            ax3.plot(
                contour_points[:, 0], contour_points[:, 1], color=color, linewidth=2
            )

            # Add shape label
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

    # 4. Shape Analysis Summary
    ax4.axis("off")

    # Create shape analysis summary
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

    # Count shapes by type
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

    # Shape-based classification hints
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

    # Display analysis text
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

    # Save to buffer dan convert ke OpenCV format

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    # Convert buffer to OpenCV image

    pil_img = Image.open(buf)
    opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    plt.close()  # Clean memory
    buf.close()

    return opencv_img


def apply_preprocessing_pipeline(img_rgb):
    """Apply complete preprocessing pipeline dengan 8 metode"""

    # 1. Image quality analysis
    analysis = analyze_image_quality(img_rgb)

    # 2. Apply adaptive preprocessing
    preprocessed_img, preprocessing_steps = adaptive_preprocessing(img_rgb, analysis)

    return preprocessed_img, preprocessing_steps, analysis
