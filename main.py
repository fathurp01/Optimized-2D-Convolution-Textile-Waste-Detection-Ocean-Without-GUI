import os
import sys
import cv2
import numpy as np
from image_processing import (
    analyze_image_quality,
    adaptive_preprocessing,
    extract_color_features,
    extract_texture_features,
    extract_shape_features,
)
from polyester_detection import classify_material


def create_result_folder(image_path, index):
    """Create a result folder for each processed image"""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    result_folder = os.path.join("output", f"{index:03d}_{base_name}")
    os.makedirs(result_folder, exist_ok=True)
    return result_folder


def ensure_folders():
    """Ensure required folders exist"""
    folders = ["samples", "output", "dataset"]  # Tambah dataset folder
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"📁 Created folder: {folder}")


def get_image_files():
    """Get all image files from samples folder"""
    samples_folder = "samples"
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = []

    if not os.path.exists(samples_folder):
        print(f"❌ Samples folder '{samples_folder}' not found!")
        return []

    for file in os.listdir(samples_folder):
        if file.lower().endswith(image_extensions):
            image_files.append(os.path.join(samples_folder, file))

    return sorted(image_files)


def save_processing_results(result_folder, results):
    """Save all processing results with enhanced template info"""
    # Save original image
    if "original" in results:
        cv2.imwrite(os.path.join(result_folder, "01_original.jpg"), results["original"])

    # Save ALL preprocessing steps (sekarang ada 8 metode)
    if "preprocessing_steps" in results:
        for i, (step_name, step_img) in enumerate(results["preprocessing_steps"], 2):
            filename = f"02_{i - 1:02d}_{step_name}.jpg"
            cv2.imwrite(os.path.join(result_folder, filename), step_img)

    # Save final preprocessed image
    if "preprocessed" in results:
        cv2.imwrite(
            os.path.join(result_folder, "03_final_preprocessed.jpg"),
            results["preprocessed"],
        )

    # Save feature extraction results
    if "color_features" in results:
        cv2.imwrite(
            os.path.join(result_folder, "04_color_features.jpg"),
            results["color_features"],
        )

    if "texture_features" in results:
        cv2.imwrite(
            os.path.join(result_folder, "05_texture_features.jpg"),
            results["texture_features"],
        )

    if "shape_features" in results:
        cv2.imwrite(
            os.path.join(result_folder, "06_shape_features.jpg"),
            results["shape_features"],
        )

    # Save classification results
    if "classification_result" in results:
        cv2.imwrite(
            os.path.join(result_folder, "07_classification_result.jpg"),
            results["classification_result"],
        )

    # Enhanced classification report dengan template info
    if "classification" in results:
        with open(
            os.path.join(result_folder, "08_classification_report.txt"), "w", encoding='utf-8'
        ) as f:
            classification = results['classification']
            
            f.write("=== POLYESTER DETECTION REPORT ===\n")
            f.write(f"Detection Status: {classification['detection_status']}\n")
            f.write(f"Material Type: {classification['type'].upper()}\n")
            f.write(f"Confidence: {classification['confidence']:.2f}%\n")
            f.write(f"Primary Method: {classification['primary_method']}\n")
            f.write(f"Secondary Method: {classification['secondary_method']}\n")
            
            # Enhanced template matching info
            if "template_info" in classification and classification["template_info"]:
                template_info = classification["template_info"]
                f.write("\n=== TEMPLATE MATCHING INFO ===\n")
                f.write(f"Templates Used: {template_info.get('total_templates', 0)}\n")
                f.write(f"Best Category: {template_info.get('best_category', 'N/A')}\n")
                f.write(f"Best Subcategory: {template_info.get('best_subcategory', 'N/A')}\n")
                f.write(f"Best Score: {template_info.get('best_score', 0):.3f}\n")
                f.write(f"Detection Context: {template_info.get('detection_context', 'N/A')}\n")
                
                if 'subcategory_scores' in template_info:
                    f.write("\n=== SUBCATEGORY SCORES ===\n")
                    for subcategory, score in template_info['subcategory_scores'].items():
                        f.write(f"{subcategory}: {score:.3f}\n")
            
            f.write("\n=== PREPROCESSING METHODS APPLIED ===\n")
            f.write(f"Adaptive Processing: {classification['preprocessing_method']}\n")

            f.write("\n=== FEATURE EXTRACTION SUMMARY ===\n")
            if "features" in classification:
                features = classification["features"]
                f.write(f"HSV Color Features: {features.get('color', 'N/A')}\n")
                f.write(f"LBP Texture Features: {features.get('texture', 'N/A')}\n")
                f.write(f"Contour Shape Features: {features.get('shape', 'N/A')}\n")

            # Add enhanced rule scores (12 rules now)
            if "rule_scores" in classification:
                scores = classification["rule_scores"]
                f.write("\n=== DETECTION SCORES (Enhanced Rules) ===\n")
                f.write(
                    f"Total Score: {scores.get('total_score', 0)}/{scores.get('max_score', 18)}\n"
                )

                # List confidence indicators
                if "confidence_indicators" in scores:
                    f.write("\n=== CONFIDENCE INDICATORS ===\n")
                    for indicator in scores["confidence_indicators"]:
                        f.write(f"✓ {indicator}\n")

            # Add preprocessing steps info
            if "preprocessing_steps" in results:
                f.write("\n=== PREPROCESSING STEPS DETAILS ===\n")
                for step_name, _ in results["preprocessing_steps"]:
                    f.write(f"- {step_name.replace('_', ' ').title()}\n")

            # Database condition analysis
            if "quality_analysis" in results:
                analysis = results["quality_analysis"]
                f.write("\n=== IMAGE CONDITION ANALYSIS ===\n")
                f.write(f"Brightness: {analysis.get('brightness', 0):.1f}\n")
                f.write(f"Contrast: {analysis.get('contrast', 0):.1f}\n")
                f.write(f"Blur Score: {analysis.get('blur_score', 0):.1f}\n")
                f.write(f"Edge Density: {analysis.get('edge_density', 0):.3f}\n")

                # Special conditions detected
                f.write("\n=== SPECIAL CONDITIONS DETECTED ===\n")
                conditions = []
                if analysis.get("is_dark", False):
                    conditions.append("Dark Image")
                if analysis.get("is_underwater", False):
                    conditions.append("Underwater/Murky Condition")
                if analysis.get("is_reflective", False):
                    conditions.append("Reflective/Wet Surface")
                if analysis.get("has_debris", False):
                    conditions.append("Contains Debris")
                if analysis.get("is_faded", False):
                    conditions.append("Faded Colors")
                if analysis.get("is_transparent", False):
                    conditions.append("Transparent Material")

                if conditions:
                    for condition in conditions:
                        f.write(f"• {condition}\n")
                else:
                    f.write("• Normal Conditions\n")


def process_single_image(image_path, index):
    """Process a single image through the complete pipeline"""
    print(f"\n🔄 Processing image {index}: {os.path.basename(image_path)}")

    # Create result folder
    result_folder = create_result_folder(image_path, index)

    try:
        # Load image (ensure RGB format)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"❌ Failed to load image: {image_path}")
            return False

        # Convert to RGB for processing (WAJIB RGB sesuai ketentuan)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Initialize results dictionary
        results = {"original": img_bgr}

        # 1. Analyze image quality for adaptive processing (Enhanced untuk database kompleks)
        print("   📊 Analyzing image quality...")
        quality_analysis = analyze_image_quality(img_gray)
        results["quality_analysis"] = quality_analysis

        # Print detected conditions
        conditions = []
        if quality_analysis.get("is_dark", False):
            conditions.append("dark")
        if quality_analysis.get("is_underwater", False):
            conditions.append("underwater/murky")
        if quality_analysis.get("is_reflective", False):
            conditions.append("reflective/wet")
        if quality_analysis.get("has_debris", False):
            conditions.append("debris")
        if quality_analysis.get("is_faded", False):
            conditions.append("faded")

        if conditions:
            print(f"   🔍 Detected conditions: {', '.join(conditions)}")

        # 2. Adaptive preprocessing (8 metode dari referensi)
        print("   ⚙️ Applying adaptive preprocessing (8 methods available)...")
        preprocessed_img, preprocessing_steps = adaptive_preprocessing(
            img_rgb, quality_analysis
        )
        results["preprocessed"] = cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2BGR)

        # Convert preprocessing steps for saving
        converted_steps = []
        for name, img in preprocessing_steps:
            if len(img.shape) == 3:
                # RGB image
                converted_steps.append((name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
            else:
                # Grayscale image
                converted_steps.append((name, img))
        results["preprocessing_steps"] = converted_steps

        print(f"   ✅ Applied {len(preprocessing_steps) - 1} preprocessing steps")

        # 3. Extract features (1 metode per kategori)
        print("   🎨 Extracting HSV color features...")
        color_features, color_img = extract_color_features(preprocessed_img)
        results["color_features"] = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

        print("   🏗️ Extracting LBP texture features...")
        texture_features, texture_img = extract_texture_features(
            cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2GRAY)
        )
        results["texture_features"] = texture_img

        print("   📐 Extracting contour shape features...")
        shape_features, shape_img = extract_shape_features(
            cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2GRAY)
        )
        results["shape_features"] = shape_img

        # 4. Classify material untuk polyester detection (2 metode klasifikasi)
        print("   🔍 Detecting polyester material (Template + Rule-based)...")
        classification_result = classify_material(
            preprocessed_img,
            color_features,
            texture_features,
            shape_features,
            quality_analysis,
        )

        results["classification"] = classification_result
        results["classification_result"] = cv2.cvtColor(
            classification_result["result_image"], cv2.COLOR_RGB2BGR
        )

        # 5. Save all results
        print("   💾 Saving results...")
        save_processing_results(result_folder, results)

        # Enhanced output
        detection_status = classification_result["detection_status"]
        confidence = classification_result["confidence"]
        method = classification_result["primary_method"]

        print(f"   ✅ {detection_status} - {confidence:.1f}% confidence ({method})")

        return True

    except Exception as e:
        print(f"   ❌ Error processing image: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def print_methods_summary():
    """Print summary of methods used"""
    print("\n📋 METHODS SUMMARY")
    print("=" * 50)
    print("🔧 PREPROCESSING METHODS (8 from reference):")
    print("   1. Manual Brightness Adjustment (A4)")
    print("   2. Manual Contrast Enhancement (A5)")
    print("   3. Manual Contrast Stretching (A6)")
    print("   4. Manual Histogram Equalization (A11)")
    print("   5. Manual Median Filter 7x7 (D5)")
    print("   6. Manual Gaussian Filter 5x5 (D3)")
    print("   7. Morphological Opening (G1)")
    print("   8. Laplacian Sharpening (D4)")

    print("\n📊 FEATURE EXTRACTION (1 method each):")
    print("   • Color: HSV Statistics")
    print("   • Texture: LBP (Local Binary Pattern)")
    print("   • Shape: Contour-based Analysis")

    print("\n🎯 CLASSIFICATION METHODS (2 main):")
    print("   1. Template Matching (Correlation)")
    print("   2. Rule-based Classification (12 rules)")

    print("\n🗃️ DATABASE CONDITIONS SUPPORTED:")
    print("   • Environmental: clear_water, murky_water, various_lighting, with_debris")
    print("   • Colors: bright_colors, faded_colors, transparent_polyester")
    print(
        "   • Conditions: dry_condition, wet_condition, floating, partially_submerged"
    )
    print(
        "   • Shapes: clothing_fragments, fabric_pieces, microfiber_clusters, thread_bundles"
    )


def main():
    print("🚀 Starting Enhanced Polyester Detection Pipeline")
    print("=" * 60)

    # Print methods summary
    print_methods_summary()

    # Ensure required folders exist
    ensure_folders()

    # Get all image files
    image_files = get_image_files()

    if len(image_files) < 60:
        print(
            f"\n⚠️ Warning: Only {len(image_files)} images found. Minimum 60 required."
        )
        if len(image_files) == 0:
            print("❌ No images found in samples folder!")
            print("📝 Please add RGB images to the 'samples' folder")
            sys.exit(1)
    else:
        print(f"\n📁 Found {len(image_files)} RGB images for processing")

    # Process all images
    print(f"\n🔄 Starting processing of {len(image_files)} images...")
    successful = 0
    failed = 0
    polyester_detected = 0

    for index, image_path in enumerate(image_files, 1):
        success = process_single_image(image_path, index)
        if success:
            successful += 1
            # Check if polyester was detected (optional counting)
            try:
                result_folder = create_result_folder(image_path, index)
                report_path = os.path.join(
                    result_folder, "08_classification_report.txt"
                )
                if os.path.exists(report_path):
                    with open(report_path, "r") as f:
                        content = f.read()
                        if "POLYESTER DETECTED" in content:
                            polyester_detected += 1
            except:
                pass
        else:
            failed += 1

    # Final summary
    print("\n" + "=" * 60)
    print("📊 ENHANCED POLYESTER DETECTION SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully processed: {successful} images")
    print(f"❌ Failed to process: {failed} images")
    print(f"🔍 Polyester detected in: {polyester_detected} images")
    print(f"📁 Results saved in: output/ folder")
    print(f"📋 Each result contains: preprocessing steps, features, classification")
    print("🎉 Processing completed!")

    if successful > 0:
        print(
            f"\n💡 Check output/001_*/08_classification_report.txt for detailed analysis"
        )


if __name__ == "__main__":
    main()
