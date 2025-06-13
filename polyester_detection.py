import cv2
import numpy as np
import os
from image_processing import (
    extract_color_features,
    extract_texture_features,
    extract_shape_features,
)


def load_polyester_dataset(dataset_path="dataset"):
    """Load comprehensive polyester template dataset - sesuai struktur folder"""
    polyester_templates = []

    # Struktur dataset sesuai dengan folder yang sudah dibuat
    template_categories = [
        # Environmental contexts
        "environmental_contexts/clear_water",
        "environmental_contexts/murky_water",
        "environmental_contexts/various_lighting",
        "environmental_contexts/with_debris",
        # Polyester colors
        "polyester_colors/bright_colors",
        "polyester_colors/faded_colors",
        "polyester_colors/transparent_polyester",
        # Polyester conditions
        "polyester_conditions/dry_condition",
        "polyester_conditions/wet_condition",
        "polyester_conditions/floating",
        "polyester_conditions/partially_submerged",
        # Polyester shapes
        "polyester_shapes/clothing_fragments",
        "polyester_shapes/fabric_pieces",
        "polyester_shapes/microfiber_clusters",
        "polyester_shapes/thread_bundles",
    ]

    templates_loaded = 0
    category_count = {}

    if not os.path.exists(dataset_path):
        print(
            f"[WAR] Dataset folder '{dataset_path}' not found. Using feature-based detection only."
        )
        return polyester_templates

    # Load dari struktur folder yang sudah dibuat
    for category in template_categories:
        category_path = os.path.join(dataset_path, category)
        category_name = category.split("/")[-1]
        category_count[category_name] = 0

        if os.path.exists(category_path):
            for file in os.listdir(category_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    img_path = os.path.join(category_path, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        polyester_templates.append(
                            {
                                "image": img,
                                "category": category_name,
                                "subcategory": category.split("/")[0],
                                "filename": file,
                                "full_path": img_path,
                            }
                        )
                        templates_loaded += 1
                        category_count[category_name] += 1

    # Fallback: load dari root dataset jika struktur folder kosong
    if templates_loaded == 0:
        print("[FALLBACK] Using flat dataset structure...")
        for file in os.listdir(dataset_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                file_path = os.path.join(dataset_path, file)
                if os.path.isfile(file_path):
                    img = cv2.imread(file_path)
                    if img is not None:
                        polyester_templates.append(
                            {
                                "image": img,
                                "category": "general_polyester",
                                "subcategory": "mixed",
                                "filename": file,
                                "full_path": file_path,
                            }
                        )
                        templates_loaded += 1

    print(f"   1. Loaded {templates_loaded} POLYESTER template images")
    if category_count and any(count > 0 for count in category_count.values()):
        print("   2. Templates per category:")
        for cat, count in category_count.items():
            if count > 0:
                print(f"      • {cat}: {count} images")
    elif templates_loaded > 0:
        print(f"   2. Loaded from flat structure: {templates_loaded} images")

    return polyester_templates


# ============================================================================
# 2 METODE KLASIFIKASI UTAMA (SESUAI KETENTUAN - TANPA ML/DL)
# ============================================================================


def enhanced_rule_based_classification(
    color_features, texture_features, shape_features
):
    """Enhanced rule-based classification SESUAI REFERENSI A9 + H3 + LBP"""

    score = 0
    max_score = 12
    reasons = []

    # ============================================================================
    # GRAYSCALE COLOR RULES (4 rules) - SESUAI REFERENSI A9
    # ============================================================================

    # Rule 1: Analisis kecerahan grayscale (polyester cenderung cerah/sintetis)
    brightness = color_features.get("brightness", 0)
    if 100 <= brightness <= 180:  # Cerah sedang hingga terang (tampilan sintetis)
        score += 2
        reasons.append("[R01] Synthetic brightness range (100-180)")
    elif brightness > 180:  # Sangat terang (kemungkinan polyester)
        score += 1
        reasons.append("[R01] High synthetic brightness (>180)")

    # Rule 2: Analisis kontras grayscale (keseragaman sintetis)
    contrast = color_features.get("contrast", 0)
    if 20 <= contrast <= 60:  # Kontras sedang (sintetis)
        score += 2
        reasons.append("[R02] Moderate contrast (synthetic material)")
    elif contrast < 20:  # Sangat seragam (kemungkinan polyester)
        score += 1
        reasons.append("[R02] High uniformity (synthetic)")

    # Rule 3: Analisis puncak grayscale (posisi puncak histogram)
    gray_peak = color_features.get("gray_peak", 0)
    if 80 <= gray_peak <= 200:  # Rentang sintetis tipikal
        score += 1
        reasons.append("[R03] Synthetic grayscale peak distribution")

    # Rule 4: Analisis rata-rata grayscale (intensitas keseluruhan)
    gray_mean = color_features.get("gray_mean", 0)
    if 90 <= gray_mean <= 170:  # Rentang polyester tipikal
        score += 1
        reasons.append("[R04] Typical polyester brightness")

    # ============================================================================
    # LBP TEXTURE RULES (4 rules) - ANALISIS TEKSTUR
    # ============================================================================

    # Rule 5: Uniformitas LBP (polyester memiliki tekstur seragam)
    lbp_uniformity = texture_features.get("lbp_uniformity", 0)
    if lbp_uniformity > 0.1:  # Uniformitas tinggi
        score += 2
        reasons.append("[R05] High texture uniformity (LBP)")
    elif lbp_uniformity > 0.05:
        score += 1
        reasons.append("[R05] Moderate texture uniformity")

    # Rule 6: Analisis rata-rata LBP (pola tekstur sintetis)
    lbp_mean = texture_features.get("lbp_mean", 0)
    if 5 <= lbp_mean <= 15:  # Rentang sintetis tipikal
        score += 1
        reasons.append("[R06] Synthetic texture pattern (LBP mean)")

    # Rule 7: Standar deviasi LBP (konsistensi sintetis)
    lbp_std = texture_features.get("lbp_std", 0)
    if lbp_std < 10:  # Variasi rendah (sintetis)
        score += 1
        reasons.append("[R07] Low texture variation (synthetic)")

    # Rule 8: Kontras tekstur (sifat permukaan material)
    texture_contrast = texture_features.get("contrast", 0)
    if texture_contrast > 500:  # Kontras tekstur tinggi (material sintetis)
        score += 1
        reasons.append("[R08] High texture contrast (synthetic)")

    # ============================================================================
    # SHAPE RULES (4 rules) - SESUAI REFERENSI H3
    # ============================================================================

    # Rule 9: Analisis jumlah bentuk (fragmen polyester)
    total_shapes = shape_features.get("total_shapes", 0)
    if 1 <= total_shapes <= 10:  # Jumlah fragmen yang wajar
        score += 1
        reasons.append("[R09] Reasonable fragment count")

    # Rule 10: Analisis circularity (karakteristik fragmen plastik)
    circularity = shape_features.get("circularity", 0)
    if 0.1 <= circularity <= 0.8:  # Circularity sedang (fragmen)
        score += 1
        reasons.append("[R10] Fragment-like circularity")

    # Rule 11: Analisis solidity (karakteristik material plastik)
    solidity = shape_features.get("solidity", 0)
    if solidity > 0.7:  # Objek solid (tidak berongga)
        score += 1
        reasons.append("[R11] Solid material characteristic")

    # Rule 12: Analisis area (klasifikasi berbasis ukuran)
    total_area = shape_features.get("total_area", 0)
    if total_area > 1000:  # Area material signifikan
        score += 1
        reasons.append("[R12] Significant material area")

    # Hitung confidence
    confidence = (score / max_score) * 100

    # Keputusan klasifikasi
    if confidence >= 60:
        classification = "POLYESTER"
        material_type = "POLYESTER"
    elif confidence >= 40:
        classification = "POSSIBLE_POLYESTER"
        material_type = "POSSIBLE_POLYESTER"
    else:
        classification = "NOT_POLYESTER"
        material_type = "OTHER_MATERIAL"

    return {
        "classification": classification,
        "material_type": material_type,
        "confidence": confidence,
        "score": score,
        "max_score": max_score,
        "reasons": reasons,
        "method": "Enhanced Rule-based (Grayscale A9 + LBP + Contour H3)",
    }


def template_matching_classification(img_rgb, polyester_templates, threshold=0.4):
    """Template matching classification using normalized cross-correlation"""

    # Handle jika template kosong
    if not polyester_templates or len(polyester_templates) == 0:
        return {
            "classification": "NOT_POLYESTER",
            "material_type": "OTHER_MATERIAL",
            "confidence": 0.0,
            "best_score": 0.0,
            "best_category": "none",
            "matches_found": 0,
            "method": "Template Matching (no templates)",
        }

    best_match = None
    best_score = 0
    best_category = "unknown"

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = img_gray.shape

    matches_found = 0

    for template_data in polyester_templates:
        # Ambil data template
        if isinstance(template_data, dict) and "image" in template_data:
            template_img = template_data["image"]
            category = template_data.get("category", "unknown")
        else:
            print(f"[WARNING] Skipping invalid template: {template_data}")
            continue

        # Validasi gambar template
        if template_img is None:
            continue

        # Konversi template ke grayscale jika perlu
        if len(template_img.shape) == 3:
            template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template_img

        # Resize template agar ukurannya sama dengan input
        template_resized = cv2.resize(template_gray, (w, h))

        # Lakukan template matching dengan normalized cross-correlation
        result = cv2.matchTemplate(img_gray, template_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        # Simpan skor terbaik dan kategori jika ditemukan
        if max_val > best_score:
            best_score = max_val
            best_category = category
            best_match = template_data

        # Hitung jumlah template yang melewati threshold
        if max_val > threshold:
            matches_found += 1

    # Jika skor terbaik melewati threshold, anggap sebagai polyester
    if best_score > threshold:
        confidence = min(best_score * 100, 95)  # Confidence dibatasi maksimal 95%

        return {
            "classification": "POLYESTER" if confidence > 60 else "POSSIBLE_POLYESTER",
            "material_type": "POLYESTER",
            "confidence": confidence,
            "best_score": best_score,
            "best_category": best_category,
            "matches_found": matches_found,
            "method": f"Template Matching ({best_category})",
        }

    # Jika tidak ada match yang cukup baik, kembalikan hasil NOT_POLYESTER
    return {
        "classification": "NOT_POLYESTER",
        "material_type": "OTHER_MATERIAL",
        "confidence": best_score * 100,
        "best_score": best_score,
        "best_category": "none",
        "matches_found": 0,
        "method": "Template Matching (no match)",
    }


def classify_material(
    img_rgb,
    polyester_templates=None,
    color_features=None,
    texture_features=None,
    shape_features=None,
):
    """Main classification function - Support both signatures"""

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Ekstraksi fitur jika belum tersedia
    if color_features is None or texture_features is None or shape_features is None:
        color_features, _ = extract_color_features(img_rgb)
        texture_features, _ = extract_texture_features(img_gray)
        shape_features, _ = extract_shape_features(img_gray)

    # Lakukan template matching jika dataset template polyester tersedia
    template_result = None
    if polyester_templates and len(polyester_templates) > 0:
        template_result = template_matching_classification(img_rgb, polyester_templates)

    # Lakukan klasifikasi berbasis aturan (rule-based)
    rule_result = enhanced_rule_based_classification(
        color_features, texture_features, shape_features
    )

    # Keputusan hybrid: jika confidence template tinggi, gabungkan dengan rule-based
    if template_result and template_result.get("confidence", 0) > 60:
        # Confidence template tinggi - gabungkan confidence template dan rule-based
        final_confidence = (
            template_result["confidence"] * 0.6 + rule_result["confidence"] * 0.4
        )
        final_result = template_result.copy()
        final_result["confidence"] = final_confidence
        final_result["method"] = (
            f"Template + Rule Hybrid ({template_result['best_category']})"
        )
        final_result["secondary_method"] = "RGB Color + LBP Texture + Contour Shape"
    else:
        # Confidence template rendah atau tidak ada template - gunakan rule-based saja
        final_result = rule_result
        if template_result:
            final_result["template_info"] = (
                f"Template confidence too low: {template_result['confidence']:.1f}%"
            )

    return final_result, color_features, texture_features, shape_features


def get_preprocessing_method_name(analysis):
    """Get preprocessing method name - Updated untuk 8 metode"""
    methods = []

    if analysis.get("is_dark", False):
        methods.append("Manual Brightness Adjustment")
    if analysis.get("is_underwater", False):
        methods.append("Manual Contrast Enhancement")
    if analysis.get("is_low_contrast", False) or analysis.get("is_faded", False):
        methods.append("Manual Contrast Stretching")
    if analysis.get("is_low_contrast", False) or analysis.get("is_underwater", False):
        methods.append("Manual Histogram Equalization")
    if analysis.get("is_noisy", False) or analysis.get("has_debris", False):
        methods.append("Manual Median Filter")
    if analysis.get("edge_density", 0) > 0.2:
        methods.append("Manual Gaussian Filter")
    if analysis.get("has_debris", False):
        methods.append("Morphological Opening")
    if analysis.get("is_blurry", False) or analysis.get("is_reflective", False):
        methods.append("Laplacian Sharpening")

    if not methods:
        methods.append("No Preprocessing Applied")

    return " + ".join(methods)
