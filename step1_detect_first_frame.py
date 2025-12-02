"""
Step 1: Detect mice in first frame using SAM3 Image Model
=========================================================
Detects mice using text prompt, extracts centroid points from the actual
segmentation masks (not bounding box centers), and saves to JSON.

This ensures points are always inside the actual mouse body, even when
mice are close together.

Usage:
    python step1_detect_first_frame.py <video_path> --prompt "mouse" -o detections.json
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

warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def get_mask_centroid_normalized(mask, height, width):
    """
    Get the centroid of a mask in normalized [0, 1] coordinates.
    The centroid is guaranteed to be inside the mask.
    """
    if hasattr(mask, 'cpu'):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = np.array(mask)

    while mask_np.ndim > 2:
        mask_np = mask_np.squeeze(0) if mask_np.shape[0] == 1 else mask_np[0]

    if mask_np.size == 0:
        return None

    binary = (mask_np > 0.5).astype(np.uint8)
    coords = np.where(binary > 0)

    if len(coords[0]) == 0:
        return None

    # Calculate centroid (mean of all mask pixels)
    cy = float(np.mean(coords[0]))
    cx = float(np.mean(coords[1]))

    # Normalize to [0, 1]
    cx_norm = cx / width
    cy_norm = cy / height

    return [cx_norm, cy_norm]


def mask_to_bbox_xyxy(mask, height, width):
    """Convert mask to bounding box in [x1, y1, x2, y2] format (absolute pixels) for display."""
    if hasattr(mask, 'cpu'):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = np.array(mask)

    while mask_np.ndim > 2:
        mask_np = mask_np.squeeze(0) if mask_np.shape[0] == 1 else mask_np[0]

    if mask_np.size == 0:
        return None

    binary = (mask_np > 0.5).astype(np.uint8)
    coords = np.where(binary > 0)

    if len(coords[0]) == 0:
        return None

    y1, y2 = coords[0].min(), coords[0].max()
    x1, x2 = coords[1].min(), coords[1].max()

    return [int(x1), int(y1), int(x2), int(y2)]


def filter_merged_masks(masks, scores, expected_objects=2, max_area_ratio=2.5):
    """Filter out masks that likely contain multiple objects merged together."""
    if len(masks) <= 1:
        return masks, scores

    areas = [np.sum(m > 0.5) for m in masks]
    sorted_areas = sorted(areas)

    if len(sorted_areas) >= expected_objects:
        reference_area = np.median(sorted_areas[:expected_objects])
    else:
        reference_area = np.median(sorted_areas)

    filtered_masks = []
    filtered_scores = []

    for mask, score, area in zip(masks, scores, areas):
        ratio = area / reference_area if reference_area > 0 else 1.0
        if ratio <= max_area_ratio:
            filtered_masks.append(mask)
            filtered_scores.append(score)
        else:
            print(f"    Rejected merged mask: area={area:,}, ratio={ratio:.1f}x")

    return filtered_masks, filtered_scores


def detect_first_frame(video_path, text_prompt, output_json, confidence=0.5,
                       min_area=100, max_area_ratio=2.5, expected_objects=2):
    """
    Detect mice in first frame using SAM3 image model with text prompt.
    Saves centroid points (from actual mask, not bbox) to JSON file.
    """
    import torch
    from PIL import Image

    video_path = Path(video_path)
    output_json = Path(output_json)

    print(f"\n{'='*60}")
    print("Step 1: Detect mice in first frame")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Prompt: '{text_prompt}'")
    print(f"Output: {output_json}")
    print(f"Expected objects: {expected_objects}")
    print(f"{'='*60}\n")

    # Read first frame
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height, width = frame.shape[:2]
    cap.release()

    if not ret:
        print("Error: Could not read video")
        return None

    print(f"Video info: {width}x{height}, {total_frames} frames at {fps:.1f} FPS")

    # Convert to PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Load SAM3 image model
    print(f"\nLoading SAM3 Image Model...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    image_model = build_sam3_image_model(device='cuda')
    processor = Sam3Processor(
        model=image_model,
        device='cuda',
        confidence_threshold=confidence
    )
    print("Model loaded!")

    # Run inference
    print(f"\nRunning inference with prompt: '{text_prompt}'...")
    state = processor.set_image(pil_image)
    state = processor.set_text_prompt(text_prompt, state)

    # Extract masks and scores
    masks = state.get('masks', None)
    scores = state.get('scores', None)

    if masks is None or (hasattr(masks, 'numel') and masks.numel() == 0):
        print("No masks detected!")
        del processor, image_model
        torch.cuda.empty_cache()
        return None

    # Convert tensors
    if hasattr(masks, 'cpu'):
        masks = masks.float().cpu().numpy()
    if hasattr(scores, 'cpu'):
        scores = scores.float().cpu().numpy()

    print(f"Raw detections: {len(masks)}")

    # Filter by area
    valid_masks = []
    valid_scores = []

    for i in range(len(masks)):
        mask = masks[i]
        score = float(scores[i]) if i < len(scores) else 0.0

        while mask.ndim > 2:
            mask = mask.squeeze(0) if mask.shape[0] == 1 else mask[0]

        if mask.size == 0 or mask.ndim != 2:
            continue

        mask_area = np.sum(mask > 0.5)
        if mask_area >= min_area:
            valid_masks.append(mask)
            valid_scores.append(score)

    print(f"After min_area filter: {len(valid_masks)}")

    # Filter merged masks
    if len(valid_masks) > 0:
        valid_masks, valid_scores = filter_merged_masks(
            valid_masks, valid_scores,
            expected_objects=expected_objects,
            max_area_ratio=max_area_ratio
        )
        print(f"After merged filter: {len(valid_masks)}")

    # Keep top N by score
    if len(valid_masks) > expected_objects:
        sorted_indices = np.argsort(valid_scores)[::-1][:expected_objects]
        valid_masks = [valid_masks[i] for i in sorted_indices]
        valid_scores = [valid_scores[i] for i in sorted_indices]
        print(f"After top-{expected_objects} filter: {len(valid_masks)}")

    # Extract centroid points from actual masks
    detections = []
    for i, (mask, score) in enumerate(zip(valid_masks, valid_scores)):
        centroid = get_mask_centroid_normalized(mask, height, width)
        bbox = mask_to_bbox_xyxy(mask, height, width)
        mask_area = int(np.sum(mask > 0.5))

        if centroid:
            detection = {
                'id': i + 1,
                'centroid': centroid,  # [x, y] normalized, inside the mask
                'bbox_xyxy': bbox,      # [x1, y1, x2, y2] absolute pixels
                'score': float(score),
                'area': mask_area
            }
            detections.append(detection)
            print(f"  Detection {i+1}: score={score:.2f}, area={mask_area:,}, "
                  f"centroid=({centroid[0]:.3f}, {centroid[1]:.3f}), bbox={bbox}")

    # Save to JSON
    result = {
        'video_path': str(video_path),
        'video_info': {
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames
        },
        'prompt': text_prompt,
        'detections': detections,
        'num_detections': len(detections)
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved {len(detections)} detections to: {output_json}")

    # Save visualization
    vis_path = output_json.with_suffix('.jpg')
    vis_frame = frame.copy()

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        cx = int(det['centroid'][0] * width)
        cy = int(det['centroid'][1] * height)

        # Draw centroid point
        cv2.circle(vis_frame, (cx, cy), 8, color, -1)
        cv2.circle(vis_frame, (cx, cy), 10, (255, 255, 255), 2)

        # Draw bbox
        x1, y1, x2, y2 = det['bbox_xyxy']
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

        # Label
        cv2.putText(vis_frame, f"#{det['id']} s={det['score']:.2f}",
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(vis_frame, f"Detections: {len(detections)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(str(vis_path), vis_frame)
    print(f"Saved visualization to: {vis_path}")

    # Cleanup
    del processor, image_model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Step 1 Complete!")
    print(f"{'='*60}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Step 1: Detect mice in first frame")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("-o", "--output", required=True, help="Output JSON path")
    parser.add_argument("--prompt", default="mouse",
                       help="Text prompt for detection (default: mouse)")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold (default: 0.5)")
    parser.add_argument("--min-area", type=int, default=100,
                       help="Minimum mask area (default: 100)")
    parser.add_argument("--max-area-ratio", type=float, default=2.5,
                       help="Max area ratio for merged filter (default: 2.5)")
    parser.add_argument("--expected-objects", type=int, default=2,
                       help="Expected number of objects (default: 2)")

    args = parser.parse_args()

    detect_first_frame(
        video_path=args.video_path,
        text_prompt=args.prompt,
        output_json=args.output,
        confidence=args.confidence,
        min_area=args.min_area,
        max_area_ratio=args.max_area_ratio,
        expected_objects=args.expected_objects
    )


if __name__ == "__main__":
    main()
