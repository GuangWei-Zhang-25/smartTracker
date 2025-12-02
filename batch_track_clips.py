"""
Batch Track Clips - Two-Step SAM3 Mouse Tracking Pipeline
==========================================================
Processes all video clips in a folder using:
  Step 1: SAM3 Image Model - detect mice on first frame, extract mask centroids
  Step 2: SAM3 Tracker - propagate through video using point prompts

This approach:
- Uses mask centroids (not bbox centers) for point prompts
- Properly initializes tracker with backbone from video model
- Works within 16GB GPU memory constraints

Usage:
    python batch_track_clips.py <clips_dir> -o <output_dir> [--prompt "mouse"]
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
    """Draw segmentation masks on a video frame."""
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

        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)

        while mask_np.ndim > 2:
            mask_np = mask_np.squeeze(0) if mask_np.shape[0] == 1 else mask_np[0]

        if mask_np.size == 0 or mask_np.ndim != 2:
            continue

        mask_bool = mask_np > 0.5
        mask_area = np.sum(mask_bool)

        if mask_area < 100:
            continue

        mask_count += 1
        overlay[mask_bool] = color

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

    cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)
    cv2.putText(output, f"Objects: {mask_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return output, mask_count


def step1_detect_first_frame(video_path, prompt, expected_objects=2, min_area=500):
    """
    Step 1: Use SAM3 Image Model to detect objects on first frame.
    Returns list of detections with mask centroids.
    """
    import torch
    from PIL import Image

    video_path = Path(video_path)

    # Read first frame
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, None

    # Convert to PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Load SAM3 image model
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    image_model = build_sam3_image_model(device='cuda')
    processor = Sam3Processor(
        model=image_model,
        device='cuda',
        confidence_threshold=0.5
    )

    # Run inference
    state = processor.set_image(pil_image)
    state = processor.set_text_prompt(prompt, state)

    # Extract masks and scores from state dict
    masks = state.get('masks', None)
    scores = state.get('scores', None)

    if masks is None or (hasattr(masks, 'numel') and masks.numel() == 0):
        # Cleanup
        del processor, image_model
        gc.collect()
        torch.cuda.empty_cache()
        return None, None

    # Convert tensors to numpy if needed
    if hasattr(masks, 'cpu'):
        masks = masks.float().cpu().numpy()
    if hasattr(scores, 'cpu'):
        scores = scores.float().cpu().numpy()

    # Process detections
    detections = []

    for i in range(len(masks)):
        mask = masks[i]
        score = float(scores[i]) if i < len(scores) else 0.0

        # Squeeze to 2D if needed
        while mask.ndim > 2:
            mask = mask.squeeze(0) if mask.shape[0] == 1 else mask[0]

        mask_np = mask

        if mask_np.size == 0:
            continue

        binary = (mask_np > 0.5).astype(np.uint8)
        mask_area = np.sum(binary)

        if mask_area < min_area:
            continue

        # Calculate mask centroid (normalized)
        coords = np.where(binary > 0)
        if len(coords[0]) == 0:
            continue

        cy = float(np.mean(coords[0])) / height
        cx = float(np.mean(coords[1])) / width

        detections.append({
            'centroid': [cx, cy],
            'score': float(score),
            'area': int(mask_area)
        })

    # Sort by score and keep top N
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    if len(detections) > expected_objects:
        detections = detections[:expected_objects]

    # Add IDs
    for i, det in enumerate(detections):
        det['id'] = i + 1

    # Cleanup
    del processor, image_model
    gc.collect()
    torch.cuda.empty_cache()

    video_info = {
        'width': width,
        'height': height,
        'fps': fps,
        'total_frames': total_frames
    }

    return detections, video_info


def step2_propagate_video(video_path, detections, video_info, output_path):
    """
    Step 2: Use SAM3 Tracker to propagate through video using point prompts.
    """
    import torch

    video_path = Path(video_path)
    output_path = Path(output_path)

    # Load SAM3 tracker with proper backbone setup
    from sam3.model_builder import build_sam3_video_model

    sam3_model = build_sam3_video_model()
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone  # Critical!
    predictor = predictor.cuda()
    predictor.eval()

    try:
        # Initialize video state
        inference_state = predictor.init_state(video_path=str(video_path))
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

        # Propagate through video
        video_segments = {}

        for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
            inference_state,
            0,           # start_frame_idx
            num_frames,  # max_frame_num_to_track
            False,       # reverse
            propagate_preflight=True,
        ):
            video_segments[frame_idx] = {
                obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
                for i, obj_id in enumerate(obj_ids)
            }

        # Write annotated output video
        cap = cv2.VideoCapture(str(video_path))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

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

        return {
            'total_frames': video_info['total_frames'],
            'frames_with_detections': frames_with_detections,
            'avg_objects': avg_objects,
            'output_path': str(output_path)
        }

    finally:
        gc.collect()
        torch.cuda.empty_cache()


def process_clip(clip_path, output_dir, prompt="mouse", expected_objects=2):
    """Process a single clip through both steps."""
    import torch

    clip_path = Path(clip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_name = clip_path.stem
    output_path = output_dir / f"{clip_name}_tracked.avi"

    print(f"\n{'='*60}")
    print(f"Processing: {clip_path.name}")
    print(f"{'='*60}")

    # Step 1: Detect on first frame
    print(f"\nStep 1: Detecting objects with prompt '{prompt}'...")
    detections, video_info = step1_detect_first_frame(
        clip_path, prompt, expected_objects=expected_objects
    )

    if detections is None or len(detections) == 0:
        print(f"  ERROR: No objects detected!")
        return {'error': 'No detections', 'clip': str(clip_path)}

    print(f"  Found {len(detections)} objects:")
    for det in detections:
        print(f"    #{det['id']}: centroid=({det['centroid'][0]:.3f}, {det['centroid'][1]:.3f}), score={det['score']:.2f}")

    # Force GPU cleanup between steps
    gc.collect()
    torch.cuda.empty_cache()

    # Step 2: Propagate through video
    print(f"\nStep 2: Propagating through video...")
    result = step2_propagate_video(clip_path, detections, video_info, output_path)

    if 'error' in result:
        print(f"  ERROR: {result['error']}")
        return result

    print(f"\nResults:")
    print(f"  Output: {result['output_path']}")
    print(f"  Frames with detections: {result['frames_with_detections']}/{result['total_frames']}")
    print(f"  Average objects per frame: {result['avg_objects']:.1f}")

    return result


def process_all_clips(clips_dir, output_dir, prompt="mouse", expected_objects=2):
    """Process all video clips in directory."""
    import torch

    clips_dir = Path(clips_dir)
    output_dir = Path(output_dir)

    # Find all mp4 clips (exclude already processed outputs)
    clips = sorted([
        c for c in clips_dir.glob("*.mp4")
        if not c.stem.endswith('_tracked') and not c.stem.endswith('_propagated')
    ])

    print(f"\n{'='*60}")
    print("SAM3 Two-Step Mouse Tracking Pipeline")
    print(f"{'='*60}")
    print(f"Clips directory: {clips_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(clips)} clips to process")
    print(f"Prompt: '{prompt}'")
    print(f"Expected objects: {expected_objects}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}")

    results = []

    for i, clip_path in enumerate(clips):
        print(f"\n[{i+1}/{len(clips)}] {clip_path.name}")

        try:
            result = process_clip(
                clip_path, output_dir,
                prompt=prompt,
                expected_objects=expected_objects
            )
            result['clip'] = str(clip_path)
            results.append(result)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({'error': str(e), 'clip': str(clip_path)})

        # Force cleanup between clips
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]

    print(f"Processed: {len(results)} clips")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print(f"\nSuccessful clips:")
        for r in successful:
            print(f"  {Path(r['clip']).name}: {r['frames_with_detections']}/{r['total_frames']} frames, avg={r['avg_objects']:.1f} objects")

    if failed:
        print(f"\nFailed clips:")
        for r in failed:
            print(f"  {Path(r['clip']).name}: {r['error']}")

    # Save results
    results_path = output_dir / "batch_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch Track Clips with SAM3 Two-Step Pipeline")
    parser.add_argument("clips_dir", help="Directory containing video clips")
    parser.add_argument("-o", "--output", required=True, help="Output directory for tracked videos")
    parser.add_argument("--prompt", default="mouse", help="Text prompt for detection (default: mouse)")
    parser.add_argument("--expected-objects", type=int, default=2,
                       help="Expected number of objects (default: 2)")

    args = parser.parse_args()

    process_all_clips(
        clips_dir=args.clips_dir,
        output_dir=args.output,
        prompt=args.prompt,
        expected_objects=args.expected_objects
    )


if __name__ == "__main__":
    main()
