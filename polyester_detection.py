import cv2
import numpy as np
import os


def load_polyester_dataset(dataset_path="dataset"):
    """Load comprehensive polyester template dataset - sesuai struktur folder"""
    polyester_templates = []

    # Struktur dataset sesuai dengan folder yang sudah dibuat
    template_categories = [
        # Environmental contexts - semua berisi polyester dalam kondisi berbeda
        "environmental_contexts/clear_water",      # polyester di air jernih
        "environmental_contexts/murky_water",      # polyester di air keruh
        "environmental_contexts/various_lighting", # polyester dengan pencahayaan berbeda
        "environmental_contexts/with_debris",      # polyester dengan kotoran

        # Polyester colors - variasi warna polyester
        "polyester_colors/bright_colors",          # polyester warna cerah
        "polyester_colors/faded_colors",           # polyester warna pudar
        "polyester_colors/transparent_polyester",  # polyester transparan

        # Polyester conditions - kondisi fisik polyester
        "polyester_conditions/dry_condition",      # polyester kering
        "polyester_conditions/wet_condition",      # polyester basah
        "polyester_conditions/floating",           # polyester mengapung
        "polyester_conditions/partially_submerged", # polyester terendam sebagian

        # Polyester shapes - bentuk-bentuk polyester
        "polyester_shapes/clothing_fragments",     # potongan pakaian polyester
        "polyester_shapes/fabric_pieces",          # potongan kain polyester
        "polyester_shapes/microfiber_clusters",    # kumpulan microfiber polyester
        "polyester_shapes/thread_bundles"          # bundel benang polyester
    ]

    templates_loaded = 0
    category_count = {}

    if not os.path.exists(dataset_path):
        print(f"⚠️ Dataset folder '{dataset_path}' not found. Using feature-based detection only.")
        return polyester_templates

    # Load dari struktur folder yang sudah dibuat
    for category in template_categories:
        category_path = os.path.join(dataset_path, category)
        category_name = category.split('/')[-1]
        category_count[category_name] = 0

        if os.path.exists(category_path):
            for file in os.listdir(category_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    img_path = os.path.join(category_path, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        polyester_templates.append({
                            'image': img,
                            'category': category_name,
                            'subcategory': category.split('/')[0],
                            'filename': file,
                            'full_path': img_path
                        })
                        templates_loaded += 1
                        category_count[category_name] += 1

    # Fallback: load dari root dataset jika struktur folder kosong
    if templates_loaded == 0:
        print("📁 Using flat dataset structure...")
        for file in os.listdir(dataset_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                # Skip folder, hanya ambil file
                file_path = os.path.join(dataset_path, file)
                if os.path.isfile(file_path):
                    img = cv2.imread(file_path)
                    if img is not None:
                        polyester_templates.append({
                            'image': img,
                            'category': 'general_polyester',
                            'subcategory': 'mixed',
                            'filename': file,
                            'full_path': file_path
                        })
                        templates_loaded += 1

    print(f"📁 Loaded {templates_loaded} POLYESTER template images")
    if category_count and any(count > 0 for count in category_count.values()):
        print("📊 Templates per category:")
        for cat, count in category_count.items():
            if count > 0:
                print(f"   • {cat}: {count} images")
    elif templates_loaded > 0:
        print(f"📊 Loaded from flat structure: {templates_loaded} images")

    return polyester_templates


# ============================================================================
# 2 METODE KLASIFIKASI UTAMA
# ============================================================================


def template_matching_classification(img_rgb, polyester_templates, threshold=0.6):
    """Enhanced template matching - dengan category awareness"""
    if not polyester_templates:
        return False, 0.0, {}

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    match_scores = []
    category_scores = {}

    # Handle both new structure (dict) and old structure (plain image)
    templates_to_use = []

    for template_data in polyester_templates:
        if isinstance(template_data, dict):
            # New structure dengan metadata
            templates_to_use.append(template_data)
        else:
            # Old structure (backward compatibility)
            templates_to_use.append({
                'image': template_data,
                'category': 'legacy',
                'subcategory': 'unknown',
                'filename': 'unknown'
            })

    # Limit untuk performa - ambil max 2 per subcategory
    subcategories = {}
    for template_data in templates_to_use:
        subcategory = template_data['subcategory']
        if subcategory not in subcategories:
            subcategories[subcategory] = []
        if len(subcategories[subcategory]) < 2:
            subcategories[subcategory].append(template_data)

    for subcategory, templates in subcategories.items():
        category_scores[subcategory] = []

        for template_data in templates:
            template_img = template_data['image']
            template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

            # Resize template to match input image
            template_resized = cv2.resize(template_gray, (img_gray.shape[1], img_gray.shape[0]))

            # Template matching using normalized correlation
            correlation = cv2.matchTemplate(img_gray, template_resized, cv2.TM_CCOEFF_NORMED)
            score = np.max(correlation)

            match_scores.append(score)
            category_scores[subcategory].append({
                'score': score,
                'category': template_data['category'],
                'filename': template_data['filename']
            })

    if match_scores:
        best_score = max(match_scores)

        # Find best matching subcategory
        best_subcategory = ""
        best_category = ""

        for subcategory, matches in category_scores.items():
            if matches:
                best_match_in_subcat = max(matches, key=lambda x: x['score'])
                if best_match_in_subcat['score'] == best_score:
                    best_subcategory = subcategory
                    best_category = best_match_in_subcat['category']
                    break

        match_info = {
            'best_score': best_score,
            'best_category': best_category,
            'best_subcategory': best_subcategory,
            'subcategory_scores': {k: np.mean([m['score'] for m in v]) for k, v in category_scores.items() if v},
            'total_templates': len(match_scores),
            'detection_context': f"{best_subcategory}_{best_category}" if best_subcategory and best_category else "general"
        }

        # Polyester detection decision
        if best_score > threshold:
            is_polyester = True
            confidence = min(95, best_score * 100)
        else:
            is_polyester = False
            confidence = max(5, (1 - best_score) * 100)

        return is_polyester, confidence, match_info

    return False, 0.0, {}


def rule_based_classification(color_features, texture_features, shape_features):
    """METODE KLASIFIKASI 2: Enhanced rule-based classification untuk database kompleks"""
    score = 0
    confidence_indicators = []

    # HSV Color Rules (dari HSV statistics)
    saturation_mean = color_features.get("S_mean", 0)
    value_mean = color_features.get("V_mean", 0)
    saturation_std = color_features.get("S_std", 0)
    hue_std = color_features.get("H_std", 0)

    # Rule 1: Polyester memiliki saturasi tinggi (synthetic dyes) - untuk bright_colors
    if saturation_mean > 80:
        score += 3
        confidence_indicators.append("High synthetic saturation")

    # Rule 2: Faded colors detection - untuk faded_colors
    elif saturation_mean < 50 and value_mean > 100:
        score += 2
        confidence_indicators.append("Faded synthetic colors")

    # Rule 3: Polyester memiliki brightness yang konsisten
    if value_mean > 120 and saturation_std < 30:
        score += 2
        confidence_indicators.append("Consistent synthetic brightness")

    # Rule 4: Transparent polyester detection - untuk transparent_polyester
    elif value_mean > 200 and saturation_mean < 30:
        score += 2
        confidence_indicators.append("Transparent synthetic material")

    # Rule 5: Color uniformity (low hue variance) - untuk synthetic materials
    if hue_std < 20:
        score += 1
        confidence_indicators.append("Uniform synthetic color")

    # LBP Texture Rules
    lbp_std = texture_features.get("lbp_std", 0)
    lbp_uniformity = texture_features.get("lbp_uniformity", 0)
    lbp_variance = texture_features.get("lbp_variance", 0)
    lbp_mean = texture_features.get("lbp_mean", 0)

    # Rule 6: Polyester memiliki texture yang uniform (low LBP variance)
    if lbp_std < 50 and lbp_variance < 2000:
        score += 3
        confidence_indicators.append("Uniform synthetic texture (LBP)")

    # Rule 7: Polyester memiliki pola yang regular
    if lbp_uniformity > 0.1:
        score += 2
        confidence_indicators.append("Regular pattern structure (LBP)")

    # Rule 8: Microfiber detection - untuk microfiber_clusters
    elif lbp_std > 70 and lbp_mean < 50:
        score += 2
        confidence_indicators.append("Microfiber texture pattern")

    # Shape Rules (dari contour analysis)
    solidity = shape_features.get("solidity", 0)
    circularity = shape_features.get("circularity", 0)
    extent = shape_features.get("extent", 0)
    area = shape_features.get("area", 0)
    aspect_ratio = shape_features.get("aspect_ratio", 0)

    # Rule 9: Polyester fabrics memiliki bentuk yang solid dan regular
    if solidity > 0.8 and extent > 0.6:
        score += 2
        confidence_indicators.append("Solid regular structure")

    # Rule 10: Regular geometric patterns - untuk fabric_pieces
    if 0.3 < circularity < 0.7:
        score += 1
        confidence_indicators.append("Regular geometric pattern")

    # Rule 11: Clothing fragments detection - untuk clothing_fragments
    elif aspect_ratio > 2.0 and area > 1000:
        score += 2
        confidence_indicators.append("Clothing fragment shape")

    # Rule 12: Thread bundles detection - untuk thread_bundles
    elif aspect_ratio > 5.0 and solidity < 0.5:
        score += 2
        confidence_indicators.append("Thread bundle structure")

    # Calculate final result
    max_score = 18  # Updated total possible score

    if score >= 10:  # High confidence
        is_polyester = True
        confidence = min(95, 50 + (score * 3))
    elif score >= 6:  # Medium confidence
        is_polyester = True
        confidence = min(80, 30 + (score * 4))
    else:  # Low confidence - not polyester
        is_polyester = False
        confidence = min(85, 40 + ((max_score - score) * 2))

    return (
        is_polyester,
        confidence,
        {
            "total_score": score,
            "max_score": max_score,
            "confidence_indicators": confidence_indicators,
        },
    )


def create_classification_visualization(img_rgb, classification_result):
    """Create enhanced POLYESTER detection result visualization"""
    h, w = img_rgb.shape[:2]
    vis_img = np.zeros((h, w + 350, 3), dtype=np.uint8)

    # Original image
    vis_img[:h, :w] = img_rgb

    # Classification result panel
    is_polyester = classification_result["is_polyester"]

    # Color coding
    if is_polyester:
        bg_color = (0, 100, 0)  # Dark green
        text_color = (0, 255, 0)  # Bright green
        status_text = "POLYESTER DETECTED"
    else:
        bg_color = (100, 0, 0)  # Dark red
        text_color = (0, 0, 255)  # Bright red
        status_text = "NOT POLYESTER"

    # Fill background
    vis_img[:h, w:] = bg_color

    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(vis_img, status_text, (w + 10, 40), font, 0.6, text_color, 2)

    confidence = classification_result["confidence"]
    cv2.putText(vis_img, f"Confidence: {confidence:.1f}%", (w + 10, 80), font, 0.6, (255, 255, 255), 2)

    # Show method dengan context
    method_text = classification_result['primary_method']
    if len(method_text) > 20:
        method_text = method_text[:20] + "..."
    cv2.putText(vis_img, f"Method: {method_text}", (w + 10, 120), font, 0.4, (255, 255, 255), 1)

    # Template matching info (jika ada)
    if "template_info" in classification_result and classification_result["template_info"]:
        template_info = classification_result["template_info"]
        if template_info.get('detection_context'):
            context = template_info['detection_context'][:15]
            cv2.putText(vis_img, f"Context: {context}", (w + 10, 140), font, 0.4, (255, 255, 255), 1)

        templates_used = template_info.get('total_templates', 0)
        cv2.putText(vis_img, f"Templates: {templates_used}", (w + 10, 160), font, 0.4, (255, 255, 255), 1)

    # Rule scores
    if "rule_scores" in classification_result:
        scores = classification_result["rule_scores"]
        cv2.putText(vis_img, "--- POLYESTER Rules ---", (w + 10, 190), font, 0.4, (200, 200, 200), 1)
        cv2.putText(vis_img, f"Score: {scores.get('total_score', 0)}/{scores.get('max_score', 0)}", (w + 10, 210), font, 0.4, text_color, 1)

        # Key indicators
        cv2.putText(vis_img, "--- Indicators ---", (w + 10, 240), font, 0.4, (200, 200, 200), 1)
        indicators = scores.get("confidence_indicators", [])
        y_pos = 260
        for i, indicator in enumerate(indicators[:4]):  # Show top 4 indicators
            indicator_text = indicator[:20] if len(indicator) > 20 else indicator
            cv2.putText(vis_img, f"• {indicator_text}", (w + 10, y_pos), font, 0.3, (255, 255, 255), 1)
            y_pos += 18

    return vis_img


def classify_material(
    img_rgb, color_features, texture_features, shape_features, quality_analysis
):
    """Main polyester detection using 2 classification methods"""

    # Load polyester templates
    polyester_templates = load_polyester_dataset()

    # METODE KLASIFIKASI 1: Enhanced Template matching
    template_result = False
    template_confidence = 0.0
    template_info = {}

    if polyester_templates:
        template_result, template_confidence, template_info = template_matching_classification(
            img_rgb, polyester_templates
        )

    # METODE KLASIFIKASI 2: Rule-based classification (tetap sama)
    rule_result, rule_confidence, rule_scores = rule_based_classification(
        color_features, texture_features, shape_features
    )

    # Enhanced hybrid decision
    if polyester_templates and template_confidence > 60:
        if template_result and rule_result:
            final_result = True
            final_confidence = (template_confidence + rule_confidence) / 2
            primary_method = f"Template + Rule Hybrid ({template_info.get('best_category', 'unknown')})"
        elif template_result or rule_result:
            final_result = True
            final_confidence = max(template_confidence, rule_confidence)
            primary_method = f"Template + Rule Hybrid ({template_info.get('detection_context', 'partial')})"
        else:
            final_result = False
            final_confidence = (template_confidence + rule_confidence) / 2
            primary_method = "Template + Rule Hybrid (No Match)"
    else:
        final_result = rule_result
        final_confidence = rule_confidence
        primary_method = "Rule-based Polyester Classification"

    # Create enhanced result
    classification_result = {
        "is_polyester": final_result,
        "type": "polyester" if final_result else "not_polyester",
        "confidence": final_confidence,
        "detection_status": "POLYESTER DETECTED" if final_result else "NOT POLYESTER",
        "primary_method": primary_method,
        "secondary_method": "HSV Color + LBP Texture + Contour Shape",
        "preprocessing_method": get_preprocessing_method_name(quality_analysis),
        "features": {
            "color": len(color_features),
            "texture": len(texture_features),
            "shape": len(shape_features)
        },
        "rule_scores": rule_scores,
        "template_info": template_info  # Enhanced template info
    }

    result_image = create_classification_visualization(img_rgb, classification_result)
    classification_result["result_image"] = result_image

    return classification_result


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
