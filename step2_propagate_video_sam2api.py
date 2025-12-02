"""
Step 2: Propagate detections through video using SAM3 Tracker (SAM2-style API)
=============================================================================
Uses the SAM3TrackerPredictor with point prompts (SAM2-style API).
This API supports adding multiple point prompts for different objects
without needing a text prompt to initialize.

Based on the official SAM3 example: sam3_for_sam2_video_task_example.ipynb

Usage:
    python step2_propagate_video_sam2api.py <detections.json> -o output_video.mp4
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


def draw_masks_on_frame(frame, masks_per_obj, colors=None):
    """Draw segmentation masks on a video frame.

    Args:
        frame: BGR video frame
        masks_per_obj: dict of obj_id -> boolean mask array
        colors: optional list of BGR colors
    """
    if colors is None:
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
        ]

    output = frame.copy()
    overlay = frame.copy()
    mask_count = 0

    if masks_per_obj is None or len(masks_per_obj) == 0:
        return output, 0

    for obj_id, mask in masks_per_obj.items():
        color = colors[(obj_id - 1) % len(colors)]

        # Convert to numpy if needed
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)

        # Squeeze to 2D if needed
        while mask_np.ndim > 2:
            mask_np = mask_np.squeeze(0) if mask_np.shape[0] == 1 else mask_np[0]

        if mask_np.size == 0 or mask_np.ndim != 2:
            continue

        # Create binary mask
        mask_bool = mask_np > 0.5
        mask_area = np.sum(mask_bool)

        if mask_area < 100:  # Skip tiny masks
            continue

        mask_count += 1

        # Apply color overlay
        overlay[mask_bool] = color

        # Draw contour
        mask_uint8 = (mask_bool * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cv2.drawContours(output, contours, -1, color, 2)
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                label = f"#{obj_id}"
                cv2.putText(output, label, (cx - 15, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Blend overlay
    cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)

    # Add object count
    cv2.putText(output, f"Objects: {mask_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return output, mask_count


def propagate_video(detections_json, output_path):
    """
    Use SAM3TrackerPredictor (SAM2-style API) with point prompts.

    This API:
    1. init_state() to initialize video
    2. add_new_points_or_box() for each object's point prompt
    3. propagate_in_video() to track through frames
    """
    import torch

    detections_json = Path(detections_json)
    output_path = Path(output_path)

    print(f"\n{'='*60}")
    print("Step 2: Propagate detections using SAM3TrackerPredictor")
    print(f"{'='*60}")
    print(f"Detections: {detections_json}")
    print(f"Output: {output_path}")

    # Load detections from step1
    with open(detections_json, 'r') as f:
        data = json.load(f)

    video_path = Path(data['video_path'])
    detections = data['detections']
    video_info = data['video_info']

    print(f"Video: {video_path}")
    print(f"Size: {video_info['width']}x{video_info['height']}")
    print(f"Frames: {video_info['total_frames']} at {video_info['fps']:.1f} FPS")
    print(f"Detections: {len(detections)}")

    for det in detections:
        print(f"  #{det['id']}: centroid=({det['centroid'][0]:.3f}, {det['centroid'][1]:.3f}), score={det['score']:.2f}")

    if len(detections) == 0:
        print("Error: No detections to propagate!")
        return None

    print(f"{'='*60}\n")

    # Load SAM3TrackerPredictor using build_sam3_video_model
    # This is the correct approach from the official example notebook
    print(f"Loading SAM3TrackerPredictor...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    from sam3.model_builder import build_sam3_video_model

    # Build the video model and get tracker with proper backbone
    sam3_model = build_sam3_video_model()
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone  # Critical: share backbone!
    predictor = predictor.cuda()
    predictor.eval()
    print("Tracker loaded with backbone!")

    try:
        # Step 1: Initialize state with video
        print(f"\nInitializing video state...")
        inference_state = predictor.init_state(video_path=str(video_path))
        num_frames = inference_state["num_frames"]
        print(f"  Loaded {num_frames} frames")

        # Step 2: Add point prompts for each detection
        print(f"\nAdding {len(detections)} point prompts on frame 0...")

        for det in detections:
            obj_id = det['id']
            centroid = det['centroid']  # [x, y] normalized [0, 1]

            # SAM3 TrackerPredictor uses normalized coordinates [0, 1]
            # Create point tensor (normalized coordinates)
            points = torch.tensor([[centroid[0], centroid[1]]], dtype=torch.float32)
            labels = torch.tensor([1], dtype=torch.int32)  # 1 = foreground

            # Add point prompt
            frame_idx, obj_ids, low_res_masks, video_res_masks = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                points=points,
                labels=labels,
                clear_old_points=True,
            )

            print(f"  Added point #{obj_id} at normalized ({centroid[0]:.3f}, {centroid[1]:.3f})")

        # Step 3: Propagate through video
        print(f"\nPropagating through video...")
        video_segments = {}

        # propagate_in_video signature:
        # (inference_state, start_frame_idx, max_frame_num_to_track, reverse, ...)
        for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
            inference_state,
            0,           # start_frame_idx
            num_frames,  # max_frame_num_to_track
            False,       # reverse
            propagate_preflight=True,
        ):
            # Store masks for each frame
            video_segments[frame_idx] = {
                obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
                for i, obj_id in enumerate(obj_ids)
            }

            if frame_idx % 30 == 0:
                print(f"  Frame {frame_idx}/{num_frames}: {len(obj_ids)} objects")

        print(f"Propagation complete: {len(video_segments)} frames")

        # Step 4: Write annotated output video
        print(f"\nWriting annotated video...")
        cap = cv2.VideoCapture(str(video_path))

        # Use XVID codec for better compatibility (works on most players)
        # For .avi output use XVID, for .mp4 try avc1/H264
        output_ext = output_path.suffix.lower()
        if output_ext == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            # Try H264 first, fall back to XVID if not available
            fourcc = cv2.VideoWriter_fourcc(*'avc1')

        writer = cv2.VideoWriter(
            str(output_path), fourcc,
            video_info['fps'],
            (video_info['width'], video_info['height'])
        )

        # Check if writer opened successfully, try fallback codec
        if not writer.isOpened():
            print("  H264 codec not available, falling back to XVID...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # Change extension to .avi for XVID
            output_path = output_path.with_suffix('.avi')
            writer = cv2.VideoWriter(
                str(output_path), fourcc,
                video_info['fps'],
                (video_info['width'], video_info['height'])
            )

        total_objects = 0
        frames_with_detections = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            masks = video_segments.get(frame_idx, {})
            annotated, mask_count = draw_masks_on_frame(frame, masks)

            if mask_count > 0:
                frames_with_detections += 1
                total_objects += mask_count

            writer.write(annotated)
            frame_idx += 1

        cap.release()
        writer.release()

        avg_objects = total_objects / max(1, frames_with_detections)

        print(f"\n{'='*60}")
        print("Results")
        print(f"{'='*60}")
        print(f"Output: {output_path}")
        print(f"Target objects: {len(detections)}")
        print(f"Frames with detections: {frames_with_detections}/{video_info['total_frames']}")
        print(f"Average objects per frame: {avg_objects:.1f}")
        print(f"{'='*60}")

        return {
            'total_frames': video_info['total_frames'],
            'frames_with_detections': frames_with_detections,
            'avg_objects': avg_objects,
            'output_path': str(output_path)
        }

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Step 2: Propagate with SAM3TrackerPredictor")
    parser.add_argument("detections_json", help="Path to detections JSON from step 1")
    parser.add_argument("-o", "--output", required=True, help="Output video path")

    args = parser.parse_args()

    propagate_video(
        detections_json=args.detections_json,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
