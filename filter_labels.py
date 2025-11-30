"""
Label Filter - Automatic filtering of YOLO labels
==================================================
Filters labels based on number of detections per frame.
Use --num-objects 2 to keep only frames with exactly 2 detections.
"""

import shutil
from pathlib import Path
import argparse
import json


def count_detections(label_path: str) -> int:
    """Count number of detections in a YOLO label file."""
    count = 0
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and len(line.split()) >= 7:  # class_id + at least 3 points
                    count += 1
    except:
        pass
    return count


def filter_labels(images_dir: str, labels_dir: str, output_dir: str,
                 num_objects: int = 2, copy_images: bool = True):
    """
    Filter labels to keep only frames with specified number of objects.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO labels
        output_dir: Output directory for filtered data
        num_objects: Number of objects required (default: 2)
        copy_images: Also copy matching images

    Returns:
        Statistics dict
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    # Create output directories
    out_labels = output_dir / "labels"
    out_images = output_dir / "images"
    out_labels.mkdir(parents=True, exist_ok=True)
    if copy_images:
        out_images.mkdir(parents=True, exist_ok=True)

    # Find all label files (recursively search subdirectories)
    label_files = list(labels_dir.rglob("*.txt"))
    label_files = [f for f in label_files if not f.name.startswith("annotation")]

    stats = {
        'total_labels': len(label_files),
        'filtered_labels': 0,
        'by_count': {},
    }

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    print(f"Filtering labels for exactly {num_objects} objects...")
    print(f"Input labels: {labels_dir}")
    print(f"Output: {output_dir}")
    print("-" * 50)

    for label_path in label_files:
        n_det = count_detections(str(label_path))

        # Track distribution
        stats['by_count'][n_det] = stats['by_count'].get(n_det, 0) + 1

        if n_det == num_objects:
            # Copy label
            shutil.copy2(label_path, out_labels / label_path.name)
            stats['filtered_labels'] += 1

            # Find and copy matching image
            if copy_images:
                stem = label_path.stem
                for ext in image_extensions:
                    # Check main directory
                    img_path = images_dir / f"{stem}{ext}"
                    if img_path.exists():
                        shutil.copy2(img_path, out_images / img_path.name)
                        break
                    # Check subdirectories
                    for subdir in images_dir.iterdir():
                        if subdir.is_dir():
                            img_path = subdir / f"{stem}{ext}"
                            if img_path.exists():
                                shutil.copy2(img_path, out_images / img_path.name)
                                break

    # Save stats
    stats_path = output_dir / "filter_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nFiltering complete!")
    print(f"Total labels scanned: {stats['total_labels']}")
    print(f"Labels with {num_objects} objects: {stats['filtered_labels']}")
    print(f"\nDetection count distribution:")
    for count, num in sorted(stats['by_count'].items()):
        pct = 100 * num / stats['total_labels']
        marker = " <-- selected" if count == num_objects else ""
        print(f"  {count} objects: {num} frames ({pct:.1f}%){marker}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Filter YOLO labels by number of detections"
    )
    parser.add_argument("images_dir", help="Directory containing images")
    parser.add_argument("labels_dir", help="Directory containing YOLO labels")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-n", "--num-objects", type=int, default=2,
                       help="Number of objects required (default: 2)")
    parser.add_argument("--no-images", action="store_true",
                       help="Don't copy images, only labels")

    args = parser.parse_args()

    filter_labels(
        args.images_dir,
        args.labels_dir,
        args.output,
        num_objects=args.num_objects,
        copy_images=not args.no_images
    )


if __name__ == "__main__":
    main()
