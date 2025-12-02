"""
Prepare YOLO Segmentation Training Data from SAM3 Tracked Clips
================================================================
Extracts frames and masks from SAM3 tracked videos and converts them
to YOLO segmentation format (polygon annotations).

YOLO Segmentation format:
- Images in images/train/ and images/val/
- Labels in labels/train/ and labels/val/
- Each label file has one line per object: class_id x1 y1 x2 y2 ... xn yn (normalized)

Usage:
    python prepare_yolo_segmentation_data.py <clips_dir> -o <output_dir> [--val-split 0.2]
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys
import os
import warnings
import json
import gc
import random
import shutil

warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def mask_to_polygon(mask, simplify_epsilon=2.0):
    """
    Convert binary mask to polygon points (normalized 0-1).
    Returns list of (x, y) normalized coordinates or None if no valid contour.
    """
    if mask is None or mask.size == 0:
        return None

    height, width = mask.shape[:2]

    # Ensure binary uint8
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Get largest contour
    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < 100:  # Skip tiny masks
        return None

    # Simplify contour
    epsilon = simplify_epsilon
    approx = cv2.approxPolyDP(largest, epsilon, True)

    # Need at least 3 points for a polygon
    if len(approx) < 3:
        return None

    # Convert to normalized coordinates
    points = []
    for pt in approx:
        x = pt[0][0] / width
        y = pt[0][1] / height
        # Clamp to [0, 1]
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        points.extend([x, y])

    return points


def draw_visualization(frame, masks, polygons, obj_ids, colors=None):
    """
    Draw visualization showing masks and polygon annotations on frame.

    Args:
        frame: BGR frame
        masks: dict of obj_id -> binary mask
        polygons: dict of obj_id -> polygon points (normalized)
        obj_ids: list of object IDs
        colors: optional list of BGR colors

    Returns:
        Annotated frame
    """
    if colors is None:
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
        ]

    height, width = frame.shape[:2]
    output = frame.copy()
    overlay = frame.copy()

    # Draw masks as semi-transparent overlays
    for obj_id in obj_ids:
        if obj_id not in masks:
            continue

        mask = masks[obj_id]
        color = colors[(obj_id - 1) % len(colors)]

        # Squeeze mask to 2D if needed
        while mask.ndim > 2:
            mask = mask.squeeze(0) if mask.shape[0] == 1 else mask[0]

        if mask.size == 0:
            continue

        mask_bool = mask > 0.5
        overlay[mask_bool] = color

        # Draw contour
        mask_uint8 = (mask_bool * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(output, contours, -1, color, 2)

    # Blend overlay
    cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)

    # Draw YOLO polygons (in different style - dashed/yellow)
    for obj_id in obj_ids:
        if obj_id not in polygons or polygons[obj_id] is None:
            continue

        poly = polygons[obj_id]
        color = colors[(obj_id - 1) % len(colors)]

        # Convert normalized to pixel coordinates
        pts = []
        for i in range(0, len(poly), 2):
            x = int(poly[i] * width)
            y = int(poly[i+1] * height)
            pts.append([x, y])

        if len(pts) >= 3:
            pts_arr = np.array(pts, dtype=np.int32)
            # Draw polygon outline (thicker, different shade)
            cv2.polylines(output, [pts_arr], True, (255, 255, 255), 1)

            # Draw vertices
            for pt in pts:
                cv2.circle(output, (pt[0], pt[1]), 3, color, -1)

        # Add label near centroid
        if len(pts) >= 3:
            cx = int(np.mean([p[0] for p in pts]))
            cy = int(np.mean([p[1] for p in pts]))
            label = f"#{obj_id}"
            cv2.putText(output, label, (cx - 15, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Add info text
    cv2.putText(output, f"Objects: {len([oid for oid in obj_ids if oid in masks])}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return output


def process_clip_for_yolo(clip_path, detections, video_info, output_images_dir, output_labels_dir,
                          class_id=0, min_mask_area=100, vis_dir=None):
    """
    Process a tracked clip to extract frames and YOLO segmentation labels.

    Args:
        clip_path: Path to video clip
        detections: List of detection dicts with centroids
        video_info: Dict with width, height, fps, total_frames
        output_images_dir: Directory for output images
        output_labels_dir: Directory for output labels
        class_id: YOLO class ID for mouse (default 0)
        min_mask_area: Minimum mask area to include
        vis_dir: Optional directory for visualization outputs

    Returns:
        Number of frames with valid annotations
    """
    import torch

    clip_path = Path(clip_path)
    output_images_dir = Path(output_images_dir)
    output_labels_dir = Path(output_labels_dir)

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Setup visualization directory if specified
    if vis_dir:
        vis_dir = Path(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Load SAM3 tracker
    from sam3.model_builder import build_sam3_video_model

    sam3_model = build_sam3_video_model()
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    predictor = predictor.cuda()
    predictor.eval()

    try:
        # Initialize video state
        inference_state = predictor.init_state(video_path=str(clip_path))
        num_frames = inference_state["num_frames"]

        # Add point prompts for each detection
        for det in detections:
            obj_id = det['id']
            centroid = det['centroid']

            points = torch.tensor([[centroid[0], centroid[1]]], dtype=torch.float32)
            labels = torch.tensor([1], dtype=torch.int32)

            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                points=points,
                labels=labels,
                clear_old_points=True,
            )

        # Propagate and collect masks
        video_segments = {}
        for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
            inference_state,
            0,
            num_frames,
            False,
            propagate_preflight=True,
        ):
            video_segments[frame_idx] = {
                obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
                for i, obj_id in enumerate(obj_ids)
            }

        # Read video and save frames with labels
        cap = cv2.VideoCapture(str(clip_path))
        clip_stem = clip_path.stem
        frames_saved = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            masks = video_segments.get(frame_idx, {})

            # Convert masks to YOLO segmentation format
            label_lines = []
            valid_masks = 0
            polygons_for_vis = {}  # For visualization
            masks_for_vis = {}     # For visualization

            for obj_id, mask in masks.items():
                # Squeeze to 2D
                while mask.ndim > 2:
                    mask = mask.squeeze(0) if mask.shape[0] == 1 else mask[0]

                if mask.size == 0:
                    continue

                # Check mask area
                mask_area = np.sum(mask > 0.5)
                if mask_area < min_mask_area:
                    continue

                masks_for_vis[obj_id] = mask

                # Convert to polygon
                mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
                polygon = mask_to_polygon(mask_uint8)

                if polygon and len(polygon) >= 6:  # At least 3 points
                    # YOLO format: class_id x1 y1 x2 y2 ... xn yn
                    coords_str = ' '.join([f'{c:.6f}' for c in polygon])
                    label_lines.append(f'{class_id} {coords_str}')
                    valid_masks += 1
                    polygons_for_vis[obj_id] = polygon

            # Only save frames with valid annotations
            if valid_masks > 0:
                # Save image
                img_name = f'{clip_stem}_frame_{frame_idx:04d}.jpg'
                img_path = output_images_dir / img_name
                cv2.imwrite(str(img_path), frame)

                # Save label
                label_name = f'{clip_stem}_frame_{frame_idx:04d}.txt'
                label_path = output_labels_dir / label_name
                with open(label_path, 'w') as f:
                    f.write('\n'.join(label_lines))

                # Save visualization if vis_dir specified
                if vis_dir:
                    vis_frame = draw_visualization(
                        frame, masks_for_vis, polygons_for_vis, list(masks_for_vis.keys())
                    )
                    vis_name = f'{clip_stem}_frame_{frame_idx:04d}_vis.jpg'
                    vis_path = vis_dir / vis_name
                    cv2.imwrite(str(vis_path), vis_frame)

                frames_saved += 1

            frame_idx += 1

        cap.release()
        return frames_saved

    finally:
        gc.collect()
        torch.cuda.empty_cache()


def run_detection_and_export(clips_dir, output_dir, prompt="mouse", expected_objects=2,
                             val_split=0.2, min_mask_area=100, class_name="mouse",
                             save_visualizations=True, sample_every=1):
    """
    Full pipeline: detect on first frame, propagate, export YOLO format.

    Args:
        clips_dir: Directory containing video clips
        output_dir: Output directory for YOLO dataset
        prompt: Text prompt for detection
        expected_objects: Expected number of objects per frame
        val_split: Validation split ratio
        min_mask_area: Minimum mask area to include
        class_name: Class name for YOLO config
        save_visualizations: Whether to save visualization images (default True)
        sample_every: Sample every Nth clip for even distribution (default 1 = all clips)
    """
    import torch
    from PIL import Image

    clips_dir = Path(clips_dir)
    output_dir = Path(output_dir)

    # Find clips
    all_clips = sorted([c for c in clips_dir.glob("*.mp4") if not c.stem.endswith('_tracked')])

    # Sample every Nth clip for even distribution
    if sample_every > 1:
        clips = all_clips[::sample_every]
        print(f"Sampling every {sample_every}th clip: {len(clips)} of {len(all_clips)} clips")
    else:
        clips = all_clips

    print(f"\n{'='*60}")
    print("Prepare YOLO Segmentation Training Data")
    print(f"{'='*60}")
    print(f"Clips directory: {clips_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(clips)} clips")
    print(f"Prompt: '{prompt}'")
    print(f"Expected objects: {expected_objects}")
    print(f"Validation split: {val_split}")
    print(f"Class name: {class_name}")
    print(f"Save visualizations: {save_visualizations}")
    print(f"{'='*60}\n")

    # Setup output directories
    train_images = output_dir / "images" / "train"
    train_labels = output_dir / "labels" / "train"
    val_images = output_dir / "images" / "val"
    val_labels = output_dir / "labels" / "val"

    # Visualization directories (separate for train/val to track)
    vis_dir_train = output_dir / "visualizations" / "train" if save_visualizations else None
    vis_dir_val = output_dir / "visualizations" / "val" if save_visualizations else None

    dirs_to_create = [train_images, train_labels, val_images, val_labels]
    if save_visualizations:
        dirs_to_create.extend([vis_dir_train, vis_dir_val])

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

    # Create dataset.yaml
    yaml_content = f"""# YOLO Segmentation Dataset
path: {output_dir}
train: images/train
val: images/val

names:
  0: {class_name}
"""
    with open(output_dir / "dataset.yaml", 'w') as f:
        f.write(yaml_content)
    print(f"Created dataset.yaml")

    # Shuffle clips for train/val split
    random.shuffle(clips)
    val_count = max(1, int(len(clips) * val_split))
    val_clips = set(clips[:val_count])
    train_clips = set(clips[val_count:])

    print(f"Train clips: {len(train_clips)}, Val clips: {len(val_clips)}")

    # Load SAM3 image model for detection
    print(f"\nLoading SAM3 Image Model...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    image_model = build_sam3_image_model(device='cuda')
    processor = Sam3Processor(
        model=image_model,
        device='cuda',
        confidence_threshold=0.5
    )
    print("Image model loaded!")

    total_train_frames = 0
    total_val_frames = 0
    results = []

    for i, clip_path in enumerate(clips):
        print(f"\n[{i+1}/{len(clips)}] Processing: {clip_path.name}")

        is_val = clip_path in val_clips
        images_dir = val_images if is_val else train_images
        labels_dir = val_labels if is_val else train_labels
        current_vis_dir = vis_dir_val if is_val else vis_dir_train

        try:
            # Read first frame
            cap = cv2.VideoCapture(str(clip_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, frame = cap.read()
            cap.release()

            if not ret:
                print(f"  Error: Could not read video")
                continue

            video_info = {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames
            }

            # Detect on first frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            state = processor.set_image(pil_image)
            state = processor.set_text_prompt(prompt, state)

            masks = state.get('masks', None)
            scores = state.get('scores', None)

            if masks is None or (hasattr(masks, 'numel') and masks.numel() == 0):
                print(f"  No detections!")
                continue

            if hasattr(masks, 'cpu'):
                masks = masks.float().cpu().numpy()
            if hasattr(scores, 'cpu'):
                scores = scores.float().cpu().numpy()

            # Extract detections
            detections = []
            for j in range(len(masks)):
                mask = masks[j]
                score = float(scores[j]) if j < len(scores) else 0.0

                while mask.ndim > 2:
                    mask = mask.squeeze(0) if mask.shape[0] == 1 else mask[0]

                if mask.size == 0:
                    continue

                binary = (mask > 0.5).astype(np.uint8)
                mask_area = np.sum(binary)

                if mask_area < min_mask_area:
                    continue

                coords = np.where(binary > 0)
                if len(coords[0]) == 0:
                    continue

                cy = float(np.mean(coords[0])) / height
                cx = float(np.mean(coords[1])) / width

                detections.append({
                    'id': len(detections) + 1,
                    'centroid': [cx, cy],
                    'score': score,
                    'area': int(mask_area)
                })

            # Keep top N
            detections = sorted(detections, key=lambda x: x['score'], reverse=True)
            if len(detections) > expected_objects:
                detections = detections[:expected_objects]

            if len(detections) == 0:
                print(f"  No valid detections after filtering!")
                continue

            print(f"  Found {len(detections)} objects, processing...")

            # Unload image model temporarily for video processing
            # (share GPU memory)

            # Process clip and export
            frames_saved = process_clip_for_yolo(
                clip_path, detections, video_info,
                images_dir, labels_dir,
                class_id=0, min_mask_area=min_mask_area,
                vis_dir=current_vis_dir
            )

            if is_val:
                total_val_frames += frames_saved
            else:
                total_train_frames += frames_saved

            print(f"  Exported {frames_saved} frames ({'val' if is_val else 'train'})")

            results.append({
                'clip': str(clip_path),
                'frames_exported': frames_saved,
                'split': 'val' if is_val else 'train',
                'detections': len(detections)
            })

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

    # Cleanup image model
    del processor, image_model
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total train frames: {total_train_frames}")
    print(f"Total val frames: {total_val_frames}")
    print(f"Dataset saved to: {output_dir}")
    print(f"{'='*60}")

    # Save results
    with open(output_dir / "export_results.json", 'w') as f:
        json.dump({
            'train_frames': total_train_frames,
            'val_frames': total_val_frames,
            'clips': results
        }, f, indent=2)

    return total_train_frames, total_val_frames


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO Segmentation Training Data")
    parser.add_argument("clips_dir", help="Directory containing video clips")
    parser.add_argument("-o", "--output", required=True, help="Output directory for YOLO dataset")
    parser.add_argument("--prompt", default="mouse", help="Text prompt for detection (default: mouse)")
    parser.add_argument("--expected-objects", type=int, default=2,
                       help="Expected number of objects (default: 2)")
    parser.add_argument("--val-split", type=float, default=0.2,
                       help="Validation split ratio (default: 0.2)")
    parser.add_argument("--min-area", type=int, default=100,
                       help="Minimum mask area (default: 100)")
    parser.add_argument("--class-name", default="mouse",
                       help="Class name for YOLO (default: mouse)")
    parser.add_argument("--no-vis", action="store_true",
                       help="Disable saving visualizations (default: save visualizations)")
    parser.add_argument("--sample-every", type=int, default=1,
                       help="Sample every Nth clip for even distribution (default: 1 = all clips)")

    args = parser.parse_args()

    run_detection_and_export(
        clips_dir=args.clips_dir,
        output_dir=args.output,
        prompt=args.prompt,
        expected_objects=args.expected_objects,
        val_split=args.val_split,
        min_mask_area=args.min_area,
        class_name=args.class_name,
        save_visualizations=not args.no_vis,
        sample_every=args.sample_every
    )


if __name__ == "__main__":
    main()
