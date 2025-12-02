"""
Sample Video at 1 FPS - Create clips for SAM3 tracking
=======================================================
Samples a video at 1 frame per second and creates short clips
suitable for SAM3 tracking. Each clip is N seconds (frames at 1fps).

Usage:
    python sample_video_1fps.py <video_path> -o <output_dir> [--clip-duration 30]
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def sample_video_1fps(video_path, output_dir, clip_duration=30, max_clips=None):
    """
    Sample video at 1 FPS and create short clips.

    Args:
        video_path: Path to source video
        output_dir: Directory to save clips
        clip_duration: Seconds per clip (frames at 1fps)
        max_clips: Maximum number of clips to create (None = all)

    Returns:
        List of created clip paths
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Sample Video at 1 FPS")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Clip duration: {clip_duration} seconds (frames at 1fps)")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Could not open video")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps

    print(f"Original: {width}x{height}, {fps:.1f} FPS, {total_frames} frames ({duration_sec:.1f}s)")

    # Calculate frame interval for 1 FPS sampling
    frame_interval = int(fps)  # Sample every fps-th frame = 1 per second
    total_sampled = int(total_frames / frame_interval)
    num_clips = (total_sampled + clip_duration - 1) // clip_duration

    if max_clips:
        num_clips = min(num_clips, max_clips)

    print(f"Sampling every {frame_interval} frames = {total_sampled} frames at 1 FPS")
    print(f"Creating {num_clips} clips of {clip_duration} frames each")
    print(f"{'='*60}\n")

    clip_paths = []

    for clip_idx in range(num_clips):
        # Calculate frame range for this clip
        start_sampled_frame = clip_idx * clip_duration
        end_sampled_frame = min(start_sampled_frame + clip_duration, total_sampled)

        if start_sampled_frame >= total_sampled:
            break

        # Create clip filename with timestamp
        start_sec = start_sampled_frame
        clip_name = f"clip_{clip_idx+1:03d}_{start_sec//60}m{start_sec%60:02d}s.mp4"
        clip_path = output_dir / clip_name

        # Create video writer (1 FPS output)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(clip_path), fourcc, 1.0, (width, height))

        if not writer.isOpened():
            print(f"  Error: Could not create writer for {clip_name}")
            continue

        frames_written = 0

        for sampled_idx in range(start_sampled_frame, end_sampled_frame):
            # Calculate actual frame number in source video
            actual_frame = sampled_idx * frame_interval

            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
            ret, frame = cap.read()

            if not ret:
                break

            writer.write(frame)
            frames_written += 1

        writer.release()

        if frames_written > 0:
            clip_paths.append(clip_path)
            print(f"  Created: {clip_name} ({frames_written} frames)")
        else:
            # Remove empty clip
            clip_path.unlink(missing_ok=True)

    cap.release()

    print(f"\n{'='*60}")
    print(f"Created {len(clip_paths)} clips in {output_dir}")
    print(f"{'='*60}")

    return clip_paths


def main():
    parser = argparse.ArgumentParser(description="Sample video at 1 FPS into clips")
    parser.add_argument("video_path", help="Path to source video")
    parser.add_argument("-o", "--output", required=True, help="Output directory for clips")
    parser.add_argument("--clip-duration", type=int, default=30,
                       help="Seconds per clip (default: 30)")
    parser.add_argument("--max-clips", type=int, default=None,
                       help="Maximum number of clips to create")

    args = parser.parse_args()

    sample_video_1fps(
        video_path=args.video_path,
        output_dir=args.output,
        clip_duration=args.clip_duration,
        max_clips=args.max_clips
    )


if __name__ == "__main__":
    main()
