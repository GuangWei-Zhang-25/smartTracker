"""
YOLO Annotation Visualizer
==========================
Visualizes YOLO segmentation annotations overlaid on images
for human validation of auto-generated labels.

Usage:
    python visualize_annotations.py <images_dir> <labels_dir> [options]

Options:
    --output, -o      Output directory for visualization images
    --random, -r      Number of random samples to visualize (default: all)
    --interactive     Open interactive viewer (press any key for next)
    --grid            Create a grid montage of samples
    --grid-size       Grid dimensions, e.g., "4x4" (default: 3x3)
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import random
from typing import List, Tuple, Optional
import json


def parse_yolo_segmentation(label_path: str) -> List[dict]:
    """
    Parse YOLO segmentation format label file.

    Format: class_id x1 y1 x2 y2 ... xn yn

    Returns:
        List of dicts with 'class_id' and 'polygon' (normalized coords)
    """
    annotations = []

    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 7:  # class_id + at least 3 points (6 coords)
                    continue

                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                # Convert to list of (x, y) tuples
                polygon = []
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        polygon.append((coords[i], coords[i + 1]))

                annotations.append({
                    'class_id': class_id,
                    'polygon': polygon
                })
    except Exception as e:
        print(f"Error parsing {label_path}: {e}")

    return annotations


def draw_polygon(image: np.ndarray, polygon: List[Tuple[float, float]],
                 color: Tuple[int, int, int], alpha: float = 0.4) -> np.ndarray:
    """
    Draw a filled polygon with transparency on the image.

    Args:
        image: BGR image
        polygon: List of normalized (x, y) coordinates
        color: BGR color tuple
        alpha: Transparency (0-1)

    Returns:
        Image with polygon drawn
    """
    h, w = image.shape[:2]

    # Convert normalized coords to pixel coords
    pts = np.array([(int(x * w), int(y * h)) for x, y in polygon], dtype=np.int32)

    # Create overlay for transparency
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts], color)

    # Blend
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Draw outline
    cv2.polylines(result, [pts], True, color, 2)

    return result


def visualize_single(image_path: str, label_path: str,
                    class_names: dict = None, show_info: bool = True) -> np.ndarray:
    """
    Visualize annotations for a single image.

    Args:
        image_path: Path to the image file
        label_path: Path to the YOLO label file
        class_names: Dict mapping class_id to name
        show_info: Whether to show info text on image

    Returns:
        Annotated image (BGR)
    """
    if class_names is None:
        class_names = {0: "mouse"}

    # Color palette for different classes
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return None

    # Parse annotations
    annotations = parse_yolo_segmentation(label_path)

    # Draw each annotation
    for i, ann in enumerate(annotations):
        class_id = ann['class_id']
        polygon = ann['polygon']
        color = colors[class_id % len(colors)]

        image = draw_polygon(image, polygon, color, alpha=0.35)

        # Draw class label at polygon centroid
        if polygon:
            cx = sum(p[0] for p in polygon) / len(polygon)
            cy = sum(p[1] for p in polygon) / len(polygon)
            h, w = image.shape[:2]
            px, py = int(cx * w), int(cy * h)

            class_name = class_names.get(class_id, f"class_{class_id}")
            label_text = f"{class_name}"

            # Draw label background
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (px - 5, py - text_h - 5), (px + text_w + 5, py + 5), color, -1)
            cv2.putText(image, label_text, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Add info text
    if show_info:
        info_text = f"{Path(image_path).name} | {len(annotations)} detection(s)"
        cv2.putText(image, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    return image


def create_grid(images: List[np.ndarray], grid_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
    """
    Create a grid montage from multiple images.

    Args:
        images: List of images (will be resized to match)
        grid_size: (rows, cols)

    Returns:
        Grid image
    """
    rows, cols = grid_size
    n_cells = rows * cols

    if not images:
        return None

    # Determine cell size from first image
    h, w = images[0].shape[:2]
    cell_w = 400
    cell_h = int(cell_w * h / w)

    # Create grid
    grid = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)

    for i, img in enumerate(images[:n_cells]):
        row = i // cols
        col = i % cols

        # Resize image to cell size
        resized = cv2.resize(img, (cell_w, cell_h))

        # Place in grid
        y1 = row * cell_h
        y2 = y1 + cell_h
        x1 = col * cell_w
        x2 = x1 + cell_w
        grid[y1:y2, x1:x2] = resized

    return grid


def find_matching_pairs(images_dir: str, labels_dir: str) -> List[Tuple[str, str]]:
    """
    Find matching image-label pairs.

    Returns:
        List of (image_path, label_path) tuples
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    pairs = []

    # Find all images (including in subdirectories)
    for img_path in images_dir.rglob('*'):
        if img_path.suffix.lower() not in image_extensions:
            continue

        # Look for matching label
        relative = img_path.relative_to(images_dir)
        label_path = labels_dir / relative.with_suffix('.txt')

        # Also check flat structure
        if not label_path.exists():
            label_path = labels_dir / (img_path.stem + '.txt')

        if label_path.exists():
            pairs.append((str(img_path), str(label_path)))

    return sorted(pairs)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO segmentation annotations for validation"
    )
    parser.add_argument("images_dir", help="Directory containing images")
    parser.add_argument("labels_dir", help="Directory containing YOLO labels")
    parser.add_argument("-o", "--output", help="Output directory for visualizations")
    parser.add_argument("-r", "--random", type=int, help="Number of random samples")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive viewer (press key for next, 'q' to quit)")
    parser.add_argument("--grid", action="store_true", help="Create grid montage")
    parser.add_argument("--grid-size", default="3x3", help="Grid size (e.g., 3x3, 4x4)")
    parser.add_argument("--class-names", help="JSON file with class name mapping")

    args = parser.parse_args()

    # Parse grid size
    if args.grid:
        try:
            rows, cols = map(int, args.grid_size.lower().split('x'))
            grid_size = (rows, cols)
        except:
            grid_size = (3, 3)

    # Load class names
    class_names = {0: "mouse"}
    if args.class_names:
        with open(args.class_names) as f:
            class_names = {int(k): v for k, v in json.load(f).items()}

    # Find matching pairs
    pairs = find_matching_pairs(args.images_dir, args.labels_dir)

    if not pairs:
        print(f"No matching image-label pairs found!")
        print(f"  Images dir: {args.images_dir}")
        print(f"  Labels dir: {args.labels_dir}")
        return

    print(f"Found {len(pairs)} image-label pairs")

    # Sample if requested
    if args.random and args.random < len(pairs):
        pairs = random.sample(pairs, args.random)
        print(f"Randomly sampled {len(pairs)} pairs")

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None

    # Process images
    visualized = []

    for i, (img_path, label_path) in enumerate(pairs):
        print(f"Processing {i+1}/{len(pairs)}: {Path(img_path).name}", end='\r')

        vis_img = visualize_single(img_path, label_path, class_names)

        if vis_img is None:
            continue

        visualized.append(vis_img)

        # Save individual visualization
        if output_dir and not args.grid:
            out_path = output_dir / f"vis_{Path(img_path).stem}.jpg"
            cv2.imwrite(str(out_path), vis_img)

        # Interactive mode
        if args.interactive:
            cv2.imshow("Annotation Validation", vis_img)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    print()  # New line after progress

    # Create and save grid
    if args.grid and visualized:
        grid_img = create_grid(visualized, grid_size)

        if output_dir:
            grid_path = output_dir / "validation_grid.jpg"
            cv2.imwrite(str(grid_path), grid_img)
            print(f"Saved grid to: {grid_path}")

        if args.interactive:
            cv2.imshow("Validation Grid", grid_img)
            cv2.waitKey(0)

    if args.interactive:
        cv2.destroyAllWindows()

    if output_dir:
        print(f"Saved {len(visualized)} visualizations to: {output_dir}")

    # Print summary
    print(f"\nSummary:")
    print(f"  Total pairs processed: {len(pairs)}")
    print(f"  Successfully visualized: {len(visualized)}")


if __name__ == "__main__":
    main()
