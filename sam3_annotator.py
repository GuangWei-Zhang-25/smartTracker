"""
SAM3 Auto-Annotator for YOLO Training
=====================================
Uses SAM3's text prompt feature to automatically segment objects
and export annotations in YOLO segmentation format.

Requirements:
- transformers (from main branch)
- torch with CUDA
- HuggingFace authentication for model access
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import Counter
import time
import os

# SAM3 imports via transformers
try:
    from PIL import Image
    import torch
    from transformers import Sam3Model, Sam3Processor
    SAM3_AVAILABLE = True
except ImportError as e:
    SAM3_AVAILABLE = False
    print(f"Warning: SAM3 not available. Error: {e}")


@dataclass
class AnnotatorConfig:
    """Configuration for SAM3 annotator."""
    text_prompt: str = "mouse"  # Text prompt for segmentation
    confidence_threshold: float = 0.5  # Minimum confidence score
    class_id: int = 0  # YOLO class ID for the object
    class_name: str = "mouse"  # Class name for dataset.yaml
    min_mask_area: int = 100  # Minimum mask area in pixels
    max_masks_per_image: int = 5  # Maximum masks to keep per image
    output_format: str = "yolo_seg"  # "yolo_seg" or "yolo_bbox"
    device: str = "cuda"  # "cuda" or "cpu"
    save_visualizations: bool = False  # Save visualization images
    visualization_dir: str = None  # Directory for visualization output


def draw_masks_on_image(image: np.ndarray, results: List[dict],
                        colors: List[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Draw segmentation masks on image for visualization.

    Args:
        image: BGR image (H, W, 3)
        results: List of dicts with 'mask', 'polygon', 'score'
        colors: List of BGR colors for each mask

    Returns:
        Annotated image
    """
    if colors is None:
        # Generate distinct colors
        colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
        ]

    output = image.copy()
    overlay = image.copy()

    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        mask = result.get("mask")
        polygon = result.get("polygon", [])
        score = result.get("score", 0)

        if mask is not None:
            # Draw semi-transparent mask
            mask_bool = mask > 127 if mask.max() > 1 else mask > 0.5
            overlay[mask_bool] = color

        # Draw polygon outline
        if polygon:
            h, w = image.shape[:2]
            pts = np.array([(int(x * w), int(y * h)) for x, y in polygon], np.int32)
            cv2.polylines(output, [pts], True, color, 2)

            # Draw score label
            if len(pts) > 0:
                x, y = pts[0]
                cv2.putText(output, f"{score:.2f}", (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Blend overlay with original
    cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)

    return output


def mask_to_polygon(mask: np.ndarray, epsilon_factor: float = 0.002) -> List[Tuple[float, float]]:
    """
    Convert binary mask to polygon points (normalized 0-1).

    Args:
        mask: Binary mask (H, W) with values 0 or 1/255
        epsilon_factor: Polygon simplification factor

    Returns:
        List of (x, y) normalized coordinates
    """
    # Ensure binary mask
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Simplify polygon
    epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Normalize coordinates
    h, w = mask.shape
    polygon = []
    for point in approx.squeeze():
        if len(point.shape) == 0:
            continue
        x_norm = float(point[0]) / w
        y_norm = float(point[1]) / h
        polygon.append((x_norm, y_norm))

    return polygon


def mask_to_bbox(mask: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convert binary mask to YOLO bbox format (normalized).

    Returns:
        (x_center, y_center, width, height) normalized 0-1
    """
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    # Find bounding box
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    h, w = mask.shape

    # Convert to YOLO format (center + size, normalized)
    x_center = ((x_min + x_max) / 2) / w
    y_center = ((y_min + y_max) / 2) / h
    width = (x_max - x_min) / w
    height = (y_max - y_min) / h

    return (x_center, y_center, width, height)


def polygon_to_yolo_string(class_id: int, polygon: List[Tuple[float, float]]) -> str:
    """Convert polygon to YOLO segmentation format string."""
    if len(polygon) < 3:
        return None
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
    return f"{class_id} {coords}"


def bbox_to_yolo_string(class_id: int, bbox: Tuple[float, float, float, float]) -> str:
    """Convert bbox to YOLO detection format string."""
    if bbox is None:
        return None
    x_c, y_c, w, h = bbox
    return f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"


class Sam3Annotator:
    """
    Automatic annotation using SAM3 text prompts via HuggingFace transformers.
    """

    def __init__(self, config: AnnotatorConfig = None, hf_token: str = None):
        self.config = config or AnnotatorConfig()
        self.hf_token = hf_token
        self.model = None
        self.processor = None
        self.device = None
        self.stats = {
            "images_processed": 0,
            "images_with_detections": 0,
            "total_masks": 0,
            "processing_time": 0,
        }

    def load_model(self):
        """Load SAM3 model from HuggingFace."""
        if not SAM3_AVAILABLE:
            raise RuntimeError(
                "SAM3 not installed. Please run:\n"
                "  pip install git+https://github.com/huggingface/transformers.git"
            )

        # Set device
        if self.config.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        print("Loading SAM3 model from HuggingFace...")
        print("(This may take a few minutes on first run)")
        start = time.time()

        # Load with authentication
        self.processor = Sam3Processor.from_pretrained(
            "facebook/sam3",
            token=self.hf_token
        )
        self.model = Sam3Model.from_pretrained(
            "facebook/sam3",
            token=self.hf_token
        ).to(self.device)

        print(f"Model loaded in {time.time() - start:.1f}s")

    def segment_image(self, image_path: str) -> List[dict]:
        """
        Segment objects in image using text prompt.

        Returns:
            List of dicts with 'mask', 'bbox', 'score', 'polygon'
        """
        if self.model is None:
            self.load_model()

        # Load image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (W, H)

        # Process with text prompt
        inputs = self.processor(
            images=image,
            text=self.config.text_prompt,
            return_tensors="pt"
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process to get masks
        results_list = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.config.confidence_threshold,
            mask_threshold=0.5,
            target_sizes=[original_size[::-1]]  # (H, W)
        )

        results = []

        if len(results_list) > 0:
            result_data = results_list[0]
            masks = result_data.get("masks", [])
            scores = result_data.get("scores", [])

            for i, (mask, score) in enumerate(zip(masks, scores)):
                # Convert mask to numpy
                if hasattr(mask, 'cpu'):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)

                if mask_np.ndim > 2:
                    mask_np = mask_np.squeeze()

                # Filter by area
                mask_area = np.sum(mask_np > 0.5)
                if mask_area < self.config.min_mask_area:
                    continue

                # Convert to polygon
                binary_mask = (mask_np > 0.5).astype(np.uint8) * 255
                polygon = mask_to_polygon(binary_mask)
                bbox = mask_to_bbox(binary_mask)

                if len(polygon) >= 3:
                    score_val = float(score) if hasattr(score, 'item') else score
                    results.append({
                        "mask": binary_mask,
                        "bbox": bbox,
                        "score": score_val,
                        "polygon": polygon,
                        "area": int(mask_area),
                    })

        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:self.config.max_masks_per_image]

        return results

    def annotate_directory(self, input_dir: str, output_dir: str = None,
                          progress_callback=None) -> dict:
        """
        Annotate all images in a directory.

        Args:
            input_dir: Directory with images (supports sharded structure)
            output_dir: Output directory for labels (default: input_dir/../labels)
            progress_callback: Optional callback(current, total, detections)

        Returns:
            Statistics dictionary
        """
        input_dir = Path(input_dir)

        # Set up output directory
        if output_dir is None:
            output_dir = input_dir.parent / "labels"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set up visualization directory if enabled
        vis_dir = None
        if self.config.save_visualizations:
            if self.config.visualization_dir:
                vis_dir = Path(self.config.visualization_dir)
            else:
                vis_dir = output_dir.parent / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)

        # Find all images (including in sharded subdirectories)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.rglob(f"*{ext}"))
            image_files.extend(input_dir.rglob(f"*{ext.upper()}"))

        image_files = sorted(set(image_files))
        total_images = len(image_files)

        if total_images == 0:
            print(f"No images found in {input_dir}")
            return self.stats

        print(f"Found {total_images} images to annotate")
        print(f"Text prompt: '{self.config.text_prompt}'")
        print(f"Output: {output_dir}")
        if vis_dir:
            print(f"Visualizations: {vis_dir}")
        print("-" * 50)

        # Reset stats
        self.stats = {
            "images_processed": 0,
            "images_with_detections": 0,
            "total_masks": 0,
            "processing_time": 0,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "text_prompt": self.config.text_prompt,
            "detection_counts": [],  # Track detections per image for majority voting
        }

        start_time = time.time()

        for i, image_path in enumerate(image_files):
            try:
                # Segment image
                results = self.segment_image(str(image_path))

                self.stats["images_processed"] += 1
                # Track detection count for this image
                self.stats["detection_counts"].append(len(results))

                if results:
                    self.stats["images_with_detections"] += 1
                    self.stats["total_masks"] += len(results)

                    # Determine output path (preserve shard structure)
                    relative_path = image_path.relative_to(input_dir)
                    label_path = output_dir / relative_path.with_suffix(".txt")
                    label_path.parent.mkdir(parents=True, exist_ok=True)

                    # Write YOLO annotation
                    with open(label_path, "w") as f:
                        for result in results:
                            if self.config.output_format == "yolo_seg":
                                line = polygon_to_yolo_string(
                                    self.config.class_id, result["polygon"]
                                )
                            else:
                                line = bbox_to_yolo_string(
                                    self.config.class_id, result["bbox"]
                                )
                            if line:
                                f.write(line + "\n")

                    # Save visualization if enabled
                    if vis_dir:
                        image = cv2.imread(str(image_path))
                        if image is not None:
                            vis_image = draw_masks_on_image(image, results)
                            vis_path = vis_dir / relative_path
                            vis_path.parent.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(vis_path), vis_image)

                # Progress update
                if progress_callback:
                    progress_callback(i + 1, total_images, self.stats["total_masks"])
                elif (i + 1) % 5 == 0 or i == total_images - 1:
                    pct = ((i + 1) / total_images) * 100
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (total_images - i - 1) / rate if rate > 0 else 0
                    print(f"Progress: {pct:.1f}% | {self.stats['total_masks']} detections | "
                          f"ETA: {eta:.0f}s", end="\r")

            except Exception as e:
                print(f"\nError processing {image_path}: {e}")
                continue

        self.stats["processing_time"] = time.time() - start_time

        # Calculate majority vote for object count
        if self.stats["detection_counts"]:
            count_distribution = Counter(self.stats["detection_counts"])
            most_common_count, frequency = count_distribution.most_common(1)[0]
            self.stats["majority_object_count"] = most_common_count
            self.stats["majority_confidence"] = frequency / len(self.stats["detection_counts"])
            self.stats["count_distribution"] = dict(count_distribution)
        else:
            self.stats["majority_object_count"] = 0
            self.stats["majority_confidence"] = 0
            self.stats["count_distribution"] = {}

        # Remove raw counts from saved stats (too verbose)
        stats_to_save = {k: v for k, v in self.stats.items() if k != "detection_counts"}

        # Save statistics
        stats_path = output_dir / "annotation_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats_to_save, f, indent=2)

        print(f"\n\nCompleted in {self.stats['processing_time']:.1f}s")
        print(f"Processed: {self.stats['images_processed']} images")
        print(f"With detections: {self.stats['images_with_detections']} images")
        print(f"Total masks: {self.stats['total_masks']}")
        print(f"\nObject count analysis (majority voting):")
        print(f"  Most common count: {self.stats['majority_object_count']} objects")
        print(f"  Confidence: {self.stats['majority_confidence']*100:.1f}%")
        print(f"  Distribution: {self.stats['count_distribution']}")

        return self.stats

    def create_dataset_yaml(self, dataset_dir: str):
        """
        Create YOLO dataset.yaml file.

        Args:
            dataset_dir: Root directory for the dataset
        """
        dataset_dir = Path(dataset_dir)

        # Create directory structure
        (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

        # Create dataset.yaml
        yaml_content = f"""# YOLO Dataset Configuration
# Generated by SAM3 Annotator

path: {dataset_dir.absolute()}
train: images/train
val: images/val

names:
  {self.config.class_id}: {self.config.class_name}
"""

        yaml_path = dataset_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        print(f"Created {yaml_path}")
        return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 Auto-Annotator - Generate YOLO annotations using text prompts"
    )
    parser.add_argument("input_dir", help="Directory containing images to annotate")
    parser.add_argument("-o", "--output", help="Output directory for labels")
    parser.add_argument("--prompt", default="mouse",
                       help="Text prompt for segmentation (default: mouse)")
    parser.add_argument("--class-id", type=int, default=0,
                       help="YOLO class ID (default: 0)")
    parser.add_argument("--class-name", default="mouse",
                       help="Class name for dataset.yaml (default: mouse)")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Minimum confidence threshold (default: 0.5)")
    parser.add_argument("--format", choices=["yolo_seg", "yolo_bbox"], default="yolo_seg",
                       help="Output format (default: yolo_seg)")
    parser.add_argument("--min-area", type=int, default=100,
                       help="Minimum mask area in pixels (default: 100)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                       help="Device to run on (default: cuda)")
    parser.add_argument("--token", help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--visualize", action="store_true",
                       help="Save visualization images with masks overlaid")
    parser.add_argument("--vis-dir", help="Directory for visualization output (default: <output>/../visualizations)")

    args = parser.parse_args()

    # Get token from args or environment
    hf_token = args.token or os.environ.get("HF_TOKEN")

    config = AnnotatorConfig(
        text_prompt=args.prompt,
        confidence_threshold=args.confidence,
        class_id=args.class_id,
        class_name=args.class_name,
        min_mask_area=args.min_area,
        output_format=args.format,
        device=args.device,
        save_visualizations=args.visualize,
        visualization_dir=args.vis_dir,
    )

    annotator = Sam3Annotator(config, hf_token=hf_token)
    stats = annotator.annotate_directory(args.input_dir, args.output)

    # Print recommendation based on majority voting
    if stats.get("majority_object_count", 0) > 0:
        print(f"\nRecommendation: Use --num-objects {stats['majority_object_count']} "
              f"for filter_labels.py based on majority voting "
              f"({stats['majority_confidence']*100:.0f}% confidence)")


if __name__ == "__main__":
    main()
