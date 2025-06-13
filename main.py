import os
import sys
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from image_processing import (
    analyze_image_quality,
    adaptive_preprocessing,
    apply_preprocessing_pipeline,
    extract_color_features,
    extract_texture_features,
    extract_shape_features,
)
from polyester_detection import classify_material, load_polyester_dataset


def create_result_folder(image_path, index):
    """Create a result folder for each processed image"""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    result_folder = os.path.join("output", f"{index:03d}_{base_name}")
    os.makedirs(result_folder, exist_ok=True)
    return result_folder


def ensure_folders():
    """Ensure required folders exist"""
    folders = ["samples", "output", "dataset"]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"CREATED folder: {folder}")


def get_image_files():
    """Get all image files from samples folder"""
    samples_folder = "samples"
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = []

    if not os.path.exists(samples_folder):
        print(f"ERROR: Samples folder '{samples_folder}' not found!")
        return []

    for file in os.listdir(samples_folder):
        if file.lower().endswith(image_extensions):
            image_files.append(os.path.join(samples_folder, file))

    return sorted(image_files)


def save_processing_results(result_folder, results):
    """Save all processing results"""
    # Save original image
    if "original" in results:
        cv2.imwrite(os.path.join(result_folder, "01_original.jpg"), results["original"])

    # Save ALL preprocessing steps
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

    # Classification report
    if "classification" in results:
        with open(
            os.path.join(result_folder, "08_classification_report.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            classification = results["classification"]

            f.write("=== POLYESTER DETECTION REPORT ===\n")
            f.write(
                f"Classification: {classification.get('classification', 'UNKNOWN')}\n"
            )
            f.write(
                f"Material Type: {classification.get('material_type', 'UNKNOWN').upper()}\n"
            )
            f.write(f"Confidence: {classification.get('confidence', 0):.2f}%\n")
            f.write(f"Method: {classification.get('method', 'Unknown')}\n")
            f.write(
                f"Score: {classification.get('score', 0)}/{classification.get('max_score', 18)}\n"
            )

            # Template matching info
            if "template_info" in classification:
                f.write(f"\nTemplate Info: {classification['template_info']}\n")

            if "best_category" in classification:
                f.write(
                    f"Best Template Category: {classification.get('best_category', 'N/A')}\n"
                )
                f.write(
                    f"Best Template Score: {classification.get('best_score', 0):.3f}\n"
                )

            # Reasons/confidence indicators
            if "reasons" in classification and classification["reasons"]:
                f.write("\n=== CONFIDENCE INDICATORS ===\n")
                for reason in classification["reasons"]:
                    f.write(f"{reason}\n")

            # Preprocessing method info
            if "quality_analysis" in results:
                from polyester_detection import get_preprocessing_method_name

                preprocessing_method = get_preprocessing_method_name(
                    results["quality_analysis"]
                )
                f.write(f"\n=== PREPROCESSING APPLIED ===\n")
                f.write(f"Methods: {preprocessing_method}\n")

            # Image condition analysis
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

            # Preprocessing steps details
            if "preprocessing_steps" in results:
                f.write("\n=== PREPROCESSING STEPS APPLIED ===\n")
                for step_name, _ in results["preprocessing_steps"]:
                    f.write(f"- {step_name.replace('_', ' ').title()}\n")


def create_classification_visualization_with_bbox(
    img_rgb, classification_result, shape_features
):
    """Create classification result visualization using matplotlib - PROFESSIONAL VERSION"""

    try:
        # Matplotlib figure untuk klasifikasi
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(
            "POLYESTER DETECTION CLASSIFICATION RESULT", fontsize=18, fontweight="bold"
        )

        # Classification info
        classification = classification_result.get("classification", "UNKNOWN")
        confidence = classification_result.get("confidence", 0)
        method = classification_result.get("method", "Unknown")
        score = classification_result.get("score", 0)
        max_score = classification_result.get("max_score", 18)

        # Get bounding boxes
        bounding_boxes = generate_bounding_boxes_for_classification(
            img_rgb, shape_features, classification_result
        )

        # Sisi Kiri: Original image dengan bounding boxes
        ax1.imshow(img_rgb)
        ax1.set_title(
            "Detection Results with Bounding Boxes", fontsize=14, fontweight="bold"
        )
        ax1.axis("off")

        # Draw bounding boxes di matplotlib
        if bounding_boxes and classification in ["POLYESTER", "POSSIBLE_POLYESTER"]:
            colors = ["red", "lime", "yellow", "cyan", "magenta", "orange"]

            for i, bbox in enumerate(bounding_boxes):
                x, y, bbox_w, bbox_h = bbox

                bbox_area = bbox_w * bbox_h
                if bbox_area > 1000:
                    bbox_conf = min(confidence + 15, 95)
                    box_color = "lime"
                    label = f"POLYESTER-{i + 1}: {bbox_conf:.0f}%"
                elif bbox_area > 500:
                    bbox_conf = confidence
                    box_color = "yellow"
                    label = f"POSSIBLE-{i + 1}: {bbox_conf:.0f}%"
                else:
                    bbox_conf = max(confidence - 20, 30)
                    box_color = "orange"
                    label = f"FRAGMENT-{i + 1}: {bbox_conf:.0f}%"

                # Draw bounding box
                from matplotlib.patches import Rectangle

                rect = Rectangle(
                    (x, y),
                    bbox_w,
                    bbox_h,
                    linewidth=3,
                    edgecolor=box_color,
                    facecolor="none",
                )
                ax1.add_patch(rect)

                # Add label with background
                ax1.text(
                    x,
                    y - 5,
                    label,
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=box_color, alpha=0.8),
                    color="black",
                )

        # Sisi Kanan: Comprehensive Analysis Dashboard
        ax2.axis("off")
        ax2.set_title(
            "Classification Analysis Dashboard", fontsize=14, fontweight="bold"
        )

        # Create analysis sections
        analysis_sections = []

        # Section 1: Detection Summary
        analysis_sections.append("TARGET DETECTION SUMMARY")
        analysis_sections.append("=" * 30)

        # Color-coded classification
        if classification == "POLYESTER":
            class_display = "CHECK POLYESTER DETECTED"
            class_color = "green"
        elif classification == "POSSIBLE_POLYESTER":
            class_display = "WARNING POSSIBLE POLYESTER"
            class_color = "orange"
        else:
            class_display = "CROSS NOT POLYESTER"
            class_color = "red"

        analysis_sections.append(f"Classification: {class_display}")
        analysis_sections.append(f"Confidence Level: {confidence:.1f}%")
        analysis_sections.append(f"Detection Score: {score}/{max_score}")
        analysis_sections.append(
            f"Objects Detected: {len(bounding_boxes) if bounding_boxes else 0}"
        )
        analysis_sections.append("")

        # Section 2: Method Information
        analysis_sections.append("SEARCH DETECTION METHOD")
        analysis_sections.append("=" * 30)
        analysis_sections.append(f"Primary Method: {method}")

        # Template matching info
        if "template_info" in classification_result:
            analysis_sections.append(
                f"Template Info: {classification_result['template_info']}"
            )

        if "best_category" in classification_result:
            analysis_sections.append(
                f"Best Match: {classification_result.get('best_category', 'N/A')}"
            )
            analysis_sections.append(
                f"Match Score: {classification_result.get('best_score', 0):.3f}"
            )

        analysis_sections.append("")

        # Section 3: Confidence Indicators
        if "reasons" in classification_result and classification_result["reasons"]:
            analysis_sections.append("CHART CONFIDENCE INDICATORS")
            analysis_sections.append("=" * 30)
            for reason in classification_result["reasons"]:
                analysis_sections.append(f"• {reason}")
            analysis_sections.append("")

        # Section 4: Feature Analysis Summary
        analysis_sections.append("TRENDING FEATURE ANALYSIS")
        analysis_sections.append("=" * 30)

        # Add shape feature info
        total_shapes = shape_features.get("total_shapes", 0)
        circularity = shape_features.get("circularity", 0)
        dominant_shape = shape_features.get("dominant_shape", "None")

        analysis_sections.append(f"Shapes Found: {total_shapes}")
        analysis_sections.append(f"Dominant Shape: {dominant_shape}")
        analysis_sections.append(f"Avg Circularity: {circularity:.3f}")

        # Classification hint
        if total_shapes > 5:
            analysis_sections.append("• Multiple fragments (typical for polyester)")
        if circularity < 0.5:
            analysis_sections.append("• Irregular shapes (synthetic material)")

        analysis_sections.append("")

        # Section 5: Material Properties
        analysis_sections.append("TEST MATERIAL PROPERTIES")
        analysis_sections.append("=" * 30)

        material_type = classification_result.get("material_type", "UNKNOWN").upper()
        analysis_sections.append(f"Material Type: {material_type}")

        # Add specific polyester indicators
        if classification in ["POLYESTER", "POSSIBLE_POLYESTER"]:
            analysis_sections.append("• Synthetic polymer detected")
            analysis_sections.append("• Consistent with polyester properties")
            analysis_sections.append("• High durability characteristics")
        else:
            analysis_sections.append("• Natural or unknown material")
            analysis_sections.append("• Does not match polyester profile")

        # Display all analysis text
        analysis_text = "\n".join(analysis_sections)

        # Create text box with proper styling
        text_props = dict(
            boxstyle="round,pad=0.8",
            facecolor="lightblue",
            alpha=0.9,
            edgecolor="navy",
            linewidth=2,
        )
        ax2.text(
            0.05,
            0.95,
            analysis_text,
            transform=ax2.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=9,
            fontfamily="monospace",
            bbox=text_props,
        )

        # Add status indicator in corner
        status_props = dict(boxstyle="round,pad=0.5", facecolor=class_color, alpha=0.8)
        ax2.text(
            0.95,
            0.05,
            f"{confidence:.0f}%",
            transform=ax2.transAxes,
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=20,
            fontweight="bold",
            color="white",
            bbox=status_props,
        )

        plt.tight_layout()

        # Save to buffer dan convert ke OpenCV format
        from io import BytesIO

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)

        # Convert buffer to OpenCV image
        from PIL import Image

        pil_img = Image.open(buf)
        opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        plt.close()
        buf.close()

        return opencv_img

    except Exception as e:
        print(f"     ERROR creating classification visualization: {e}")
        raise e


def generate_bounding_boxes_for_classification(
    img_rgb, shape_features, classification_result
):
    """Generate bounding boxes specifically for classification visualization"""

    # Convert to grayscale untuk contour detection
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bounding_boxes = []

    # Only generate bounding boxes if classified as polyester
    classification = classification_result.get("classification", "UNKNOWN")
    if classification in ["POLYESTER", "POSSIBLE_POLYESTER"]:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Only significant objects
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append((x, y, w, h))

    return bounding_boxes


def standardize_image_size(img_rgb, target_height=400):
    """Standardize image size untuk konsistensi output"""

    h, w = img_rgb.shape[:2]

    # Calculate new width maintaining aspect ratio
    aspect_ratio = w / h
    target_width = int(target_height * aspect_ratio)

    # Resize image
    resized_img = cv2.resize(
        img_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA
    )

    return resized_img


def process_single_image(image_path, index, polyester_templates):
    """Process single image - FLOW TIDAK BERUBAH, hanya color feature yang auto-convert"""

    filename = os.path.basename(image_path)
    name_only = os.path.splitext(filename)[0]
    result_folder = os.path.join("output", f"{index:03d}_{name_only}")

    os.makedirs(result_folder, exist_ok=True)

    try:
        # 1. Load and convert image
        print(f"\nProcessing image {index}: {filename}")
        print("   1. Loading and converting to RGB...")

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Cannot load image: {image_path}")

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        print(f"   2. Original image shape: {img_rgb.shape}")

        # Standardize image size
        img_rgb = standardize_image_size(img_rgb, target_height=400)
        print(f"   3. Standardized to size: {img_rgb.shape}")

        # Store results
        results = {"original": cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)}

        # 2. Preprocessing (8 metode) - TETAP SAMA
        print("   4. Applying preprocessing pipeline (8 methods)...")
        preprocessed_img, preprocessing_steps, quality_analysis = (
            apply_preprocessing_pipeline(img_rgb)
        )
        results["preprocessing_steps"] = preprocessing_steps
        results["preprocessed"] = cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2BGR)
        results["quality_analysis"] = quality_analysis

        # 3. Feature extraction
        img_gray = cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2GRAY)

        # RGB COLOR FEATURES - AUTO HANDLE GRAYSCALE/RGB
        print("   5. Extracting RGB histogram features...")
        # PASS APAPUN (RGB atau grayscale), function akan handle auto-conversion
        color_features, color_img = extract_color_features(
            preprocessed_img
        )  # Auto-handle di dalam function
        results["color_features"] = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

        # LBP TEXTURE FEATURES - TETAP SAMA (gunakan grayscale)
        print("   6. Extracting LBP texture features...")
        texture_features, texture_img = extract_texture_features(img_gray)
        results["texture_features"] = texture_img

        # SHAPE FEATURES - TETAP SAMA (gunakan grayscale)
        print("   7. Extracting contour shape identification...")
        shape_features, shape_img = extract_shape_features(img_gray)
        results["shape_features"] = shape_img

        # 4. Classification - TETAP SAMA
        print("   8. Detecting polyester material...")
        (
            classification_result,
            extracted_color_features,
            extracted_texture_features,
            extracted_shape_features,
        ) = classify_material(
            preprocessed_img,  # Tetap pass preprocessed_img
            polyester_templates,
        )

        # Create classification visualization
        classification_viz = create_classification_visualization_with_bbox(
            preprocessed_img, classification_result, shape_features
        )

        results["classification"] = classification_result
        results["classification_result"] = cv2.cvtColor(
            classification_viz, cv2.COLOR_RGB2BGR
        )

        # 5. Save all results - TETAP SAMA
        print("   9. Saving results...")
        save_processing_results(result_folder, results)

        # Enhanced output
        classification = classification_result.get("classification", "UNKNOWN")
        confidence = classification_result.get("confidence", 0)

        bounding_boxes = generate_bounding_boxes_for_classification(
            preprocessed_img, shape_features, classification_result
        )
        num_objects = len(bounding_boxes)

        print(f"   10. {classification} - {confidence:.1f}% confidence")
        print(f"   11. {num_objects} objects detected")

        return True

    except Exception as e:
        print(f"   ERROR Error processing {filename}: {str(e)}")
        return False


def print_methods_summary():
    """Print summary of methods used"""
    print("\n[01] METHODS SUMMARY")
    print("=" * 50)
    print("[PRE] PREPROCESSING METHODS (8 from reference):")
    print("   1. Manual Brightness Adjustment (A4)")
    print("   2. Manual Contrast Enhancement (A5)")
    print("   3. Manual Contrast Stretching (A6)")
    print("   4. Manual Histogram Equalization (A11)")
    print("   5. Manual Median Filter 7x7 (D5)")
    print("   6. Manual Gaussian Filter 5x5 (D3)")
    print("   7. Morphological Opening (G1)")
    print("   8. Laplacian Sharpening (D4)")

    print("\n[FEX] FEATURE EXTRACTION (1 method each):")
    print("   1. Color: Grayscale Histogram Statistics (A9)")
    print("   2. Texture: LBP (Local Binary Pattern)")
    print("   3. Shape: Contour Shape Identification (H3)")

    print("\n[CLS] CLASSIFICATION METHODS (2 main):")
    print("   1. Template Matching (Correlation)")
    print("   2. Rule-based Classification (12 rules)")

    print("\n[SUP] DATABASE CONDITIONS SUPPORTED:")
    print(
        "   1. Environmental: clear_water, murky_water, various_lighting, with_debris"
    )
    print("   2. Colors: bright_colors, faded_colors, transparent_polyester")


def main():
    print("Starting Enhanced Polyester Detection Pipeline")
    print("=" * 60)

    # Print methods summary
    print_methods_summary()

    # Ensure required folders exist
    ensure_folders()

    # LOAD DATASET TEMPLATES ONCE - MOVE HERE
    print("\n[FOLDER] Loading polyester template dataset...")
    polyester_templates = load_polyester_dataset("dataset")

    # Get all image files
    image_files = get_image_files()

    if len(image_files) < 60:
        print(
            f"\nWARNING Warning: Only {len(image_files)} images found. Minimum 60 required."
        )
        if len(image_files) == 0:
            print("ERROR No images found in samples folder!")
            print("Please add RGB images to the 'samples' folder")
            print("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff")
            sys.exit(1)
    else:
        print(f"\nFILES Found {len(image_files)} RGB images for processing")

    # Process all images
    print(f"\nStarting processing of {len(image_files)} images...")
    successful = 0
    failed = 0
    polyester_detected = 0
    possible_polyester = 0

    for index, image_path in enumerate(image_files, 1):
        success = process_single_image(image_path, index, polyester_templates)

        if success:
            successful += 1

            # Check classification result
            try:
                filename = os.path.basename(image_path)
                name_only = os.path.splitext(filename)[0]
                result_folder = os.path.join("output", f"{index:03d}_{name_only}")

                # Read classification report to check detection
                report_path = os.path.join(
                    result_folder, "08_classification_report.txt"
                )
                if os.path.exists(report_path):
                    with open(report_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if "Classification: POLYESTER" in content:
                            polyester_detected += 1
                        elif "Classification: POSSIBLE_POLYESTER" in content:
                            possible_polyester += 1
            except:
                pass  # Skip if cannot read report
        else:
            failed += 1

        # Progress indicator every 10 images
        if index % 10 == 0:
            print(f"   Progress: {index}/{len(image_files)} images processed")

    # Final summary
    print("\n" + "=" * 60)
    print("ENHANCED POLYESTER DETECTION SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {len(image_files)}")
    print(f"Successfully processed: {successful} images")
    print(f"Failed to process: {failed} images")
    print(f"POLYESTER detected: {polyester_detected} images")
    print(f"POSSIBLE_POLYESTER: {possible_polyester} images")
    print(
        f"NOT_POLYESTER: {successful - polyester_detected - possible_polyester} images"
    )

    # Detection rate
    if successful > 0:
        detection_rate = ((polyester_detected + possible_polyester) / successful) * 100
        print(f"Detection rate: {detection_rate:.1f}%")

    print(f"\nResults saved in: output/ folder")
    print(f"Each result folder contains:")
    print("   • 01_original.jpg - Original image")
    print("   • 02_XX_method.jpg - Preprocessing steps")
    print("   • 03_final_preprocessed.jpg - Final preprocessed")
    print(
        "   • 04_color_features.jpg - Grayscale histogram analysis"
    )  # ✅ DIUBAH dari RGB ke Grayscale
    print("   • 05_texture_features.jpg - LBP texture analysis")
    print("   • 06_shape_features.jpg - Contour shape analysis")
    print("   • 07_classification_result.jpg - Classification visualization")
    print("   • 08_classification_report.txt - Detailed analysis report")

    print("\nProcessing completed!")

    if successful > 0:
        # Show sample results
        print(f"\nSAMPLE RESULTS:")
        sample_count = min(3, successful)
        for i in range(1, sample_count + 1):
            try:
                # Find first successful result folder
                output_folders = [
                    f for f in os.listdir("output") if f.startswith(f"{i:03d}_")
                ]
                if output_folders:
                    result_folder = os.path.join("output", output_folders[0])
                    report_path = os.path.join(
                        result_folder, "08_classification_report.txt"
                    )

                    if os.path.exists(report_path):
                        with open(report_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            classification = "UNKNOWN"
                            confidence = "0%"
                            for line in lines:
                                if line.startswith("Classification:"):
                                    classification = line.strip().split(": ")[1]
                                if line.startswith("Confidence:"):
                                    confidence = line.strip().split(": ")[1]

                        image_name = output_folders[0].replace(f"{i:03d}_", "")
                        print(f"   {i}. {image_name}: {classification} ({confidence})")
            except:
                pass


if __name__ == "__main__":
    main()
