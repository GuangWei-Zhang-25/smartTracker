"""
Batch Pipeline for Smart Tracker
=================================
Complete pipeline: Frame extraction -> SAM3 annotation -> Dataset prep -> YOLO training -> Tracking

Key Features:
- SAM3 annotation with majority voting to determine expected object count
- Automatic visualization generation for human validation
- Expected object count passed to tracker for consistent ID management
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil
import json
import random


def run_command(cmd, description):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: Command returned non-zero exit code: {result.returncode}")
    return result.returncode


def count_files(directory, pattern="*.jpg"):
    """Count files matching pattern in directory (including subdirs)."""
    return len(list(Path(directory).rglob(pattern)))


def process_video(video_path: str, output_dir: str, text_prompt: str = "mouse",
                  target_frames: int = 500, min_objects: int = None,
                  epochs: int = 50, batch_size: int = 2,
                  auto_object_count: bool = True):
    """
    Process a single video through the complete pipeline.

    Args:
        video_path: Path to input video
        output_dir: Base output directory for this video
        text_prompt: SAM3 text prompt for object detection
        target_frames: Target number of frames to extract
        min_objects: Minimum objects per frame for training (None = use majority vote)
        epochs: Training epochs
        batch_size: Training batch size
        auto_object_count: Use SAM3 majority voting to determine object count
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define paths
    frames_dir = output_dir / "frames"
    labels_dir = output_dir / "labels"
    vis_dir = output_dir / "visualizations"  # For human validation
    filtered_dir = output_dir / "filtered"
    dataset_dir = output_dir / "dataset"
    runs_dir = output_dir / "runs"
    tracking_dir = output_dir / "tracking_output"

    print(f"\n{'#'*70}")
    print(f"# Processing: {video_path.name}")
    print(f"# Output: {output_dir}")
    print(f"# Text prompt: '{text_prompt}'")
    print(f"{'#'*70}")

    scripts_dir = Path("C:/Users/guangwei/Documents/SmartTracker")

    # Step 1: Smart Frame Extraction
    print("\n" + "="*70)
    print("PHASE 1: Smart Frame Extraction")
    print("="*70)

    cmd = [
        sys.executable, str(scripts_dir / "smart_frame_sampler.py"),
        str(video_path),
        "-o", str(frames_dir),
        "--sensitivity", "medium",
        "--min-interval", "0.5",
        "--max-interval", "10.0"
    ]
    run_command(cmd, "Extracting frames with smart sampling")

    n_frames = count_files(frames_dir, "*.jpg")
    print(f"Extracted {n_frames} frames")

    if n_frames == 0:
        print("ERROR: No frames extracted!")
        return None

    # Step 2: SAM3 Annotation (with visualization for human validation)
    print("\n" + "="*70)
    print("PHASE 2: SAM3 Annotation (with visualization)")
    print("="*70)

    cmd = [
        sys.executable, str(scripts_dir / "sam3_annotator.py"),
        str(frames_dir),
        "-o", str(labels_dir),
        "--prompt", text_prompt,
        "--format", "yolo_seg",
        "--confidence", "0.3",
        "--visualize",  # Generate visualizations for human validation
        "--vis-dir", str(vis_dir)
    ]
    run_command(cmd, f"Annotating frames with SAM3 (prompt: '{text_prompt}')")

    n_labels = count_files(labels_dir, "*.txt")
    n_vis = count_files(vis_dir, "*.jpg") + count_files(vis_dir, "*.png")
    print(f"Generated {n_labels} label files")
    print(f"Generated {n_vis} visualization images for human validation")
    print(f"  -> Review visualizations at: {vis_dir}")

    if n_labels == 0:
        print("ERROR: No labels generated!")
        return None

    # Read annotation stats to get majority object count
    expected_object_count = 1  # Default
    stats_path = labels_dir / "annotation_stats.json"
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            ann_stats = json.load(f)
        expected_object_count = ann_stats.get("majority_object_count", 1)
        majority_conf = ann_stats.get("majority_confidence", 0) * 100
        count_dist = ann_stats.get("count_distribution", {})
        print(f"\nSAM3 Majority Voting Results:")
        print(f"  Expected object count: {expected_object_count}")
        print(f"  Confidence: {majority_conf:.1f}%")
        print(f"  Distribution: {count_dist}")

    # Use user-specified min_objects if provided, otherwise use majority vote
    if min_objects is not None:
        filter_count = min_objects
        print(f"  Using user-specified object count: {filter_count}")
    else:
        filter_count = expected_object_count
        print(f"  Using majority vote object count: {filter_count}")

    # Step 3: Filter labels by object count
    print("\n" + "="*70)
    print(f"PHASE 3: Filter Labels (exactly {filter_count} objects)")
    print("="*70)

    cmd = [
        sys.executable, str(scripts_dir / "filter_labels.py"),
        str(frames_dir),  # images_dir
        str(labels_dir),  # labels_dir
        "-o", str(filtered_dir),
        "-n", str(filter_count)
    ]
    run_command(cmd, f"Filtering labels with {filter_count} objects")

    n_filtered = count_files(filtered_dir / "labels", "*.txt")
    print(f"Filtered to {n_filtered} frames with exactly {filter_count} objects")

    if n_filtered < 10:
        print(f"WARNING: Only {n_filtered} frames after filtering. May need more data.")
        if n_filtered == 0:
            print("ERROR: No frames passed filter!")
            return None

    # Step 4: Prepare YOLO Dataset
    print("\n" + "="*70)
    print("PHASE 4: Prepare YOLO Dataset")
    print("="*70)

    cmd = [
        sys.executable, str(scripts_dir / "prepare_yolo_dataset.py"),
        str(filtered_dir / "images"),
        str(filtered_dir / "labels"),
        "-o", str(dataset_dir),
        "--val-split", "0.2",
        "--class-name", text_prompt
    ]
    run_command(cmd, "Creating train/val split")

    # Step 5: Train YOLO Model
    print("\n" + "="*70)
    print("PHASE 5: Train YOLO Segmentation Model")
    print("="*70)

    dataset_yaml = dataset_dir / "dataset.yaml"
    if not dataset_yaml.exists():
        print(f"ERROR: Dataset config not found: {dataset_yaml}")
        return None

    cmd = [
        sys.executable, str(scripts_dir / "train_yolo.py"),
        str(dataset_yaml),
        "--model", "n",
        "--epochs", str(epochs),
        "--imgsz", "480",
        "--batch", str(batch_size),
        "--device", "0",
        "--workers", "0",
        "--mosaic", "0.0",
        "--project", str(runs_dir),
        "--name", "model"
    ]
    run_command(cmd, f"Training YOLOv8n-seg for {epochs} epochs")

    # Find best model
    best_model = runs_dir / "model" / "weights" / "best.pt"
    if not best_model.exists():
        # Try alternate paths
        for p in runs_dir.glob("**/best.pt"):
            best_model = p
            break

    if not best_model.exists():
        print(f"WARNING: Best model not found at expected path")
        return None

    print(f"Best model saved: {best_model}")

    # Step 6: Run Tracking (with expected object count from majority voting)
    print("\n" + "="*70)
    print("PHASE 6: Run YOLO Tracker")
    print("="*70)
    print(f"Using expected object count: {filter_count} (from {'user' if min_objects is not None else 'majority vote'})")

    cmd = [
        sys.executable, str(scripts_dir / "yolo_tracker.py"),
        str(video_path),
        "--model", str(best_model),
        "-o", str(tracking_dir),
        "--proximity", "100",
        "--conf", "0.5",
        "--expected-objects", str(filter_count)  # Use majority vote object count
    ]
    run_command(cmd, f"Running tracker on original video (expected {filter_count} objects)")

    # Summary
    print("\n" + "#"*70)
    print(f"# PIPELINE COMPLETE: {video_path.name}")
    print("#"*70)
    print(f"  Frames extracted: {n_frames}")
    print(f"  Labels generated: {n_labels}")
    print(f"  Expected object count: {filter_count} (majority vote)")
    print(f"  Frames after filter: {n_filtered}")
    print(f"  Model: {best_model}")
    print(f"  Visualizations: {vis_dir}")
    print(f"  Tracking output: {tracking_dir}")
    print("#"*70)
    print(f"\n*** IMPORTANT: Review visualizations at {vis_dir} for annotation quality ***")

    return {
        'video': str(video_path),
        'frames': n_frames,
        'labels': n_labels,
        'expected_object_count': filter_count,
        'filtered': n_filtered,
        'model': str(best_model),
        'visualizations': str(vis_dir),
        'tracking_output': str(tracking_dir)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch pipeline for Smart Tracker"
    )
    parser.add_argument("video", help="Input video path")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-p", "--prompt", default="mouse",
                       help="Text prompt for SAM3 (default: mouse)")
    parser.add_argument("-n", "--frames", type=int, default=500,
                       help="Target frames to extract (default: 500)")
    parser.add_argument("--min-objects", type=int, default=None,
                       help="Objects per frame for filtering (default: auto from SAM3 majority vote)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=2,
                       help="Training batch size (default: 2)")

    args = parser.parse_args()

    result = process_video(
        args.video,
        args.output,
        text_prompt=args.prompt,
        target_frames=args.frames,
        min_objects=args.min_objects,  # None = use majority vote
        epochs=args.epochs,
        batch_size=args.batch
    )

    if result:
        print("\nPipeline completed successfully!")
        print(json.dumps(result, indent=2))
    else:
        print("\nPipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
