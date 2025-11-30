"""
Smart Frame Sampler for YOLO Training Data Preparation
======================================================
Efficiently extracts diverse frames from videos while avoiding redundancy.

Features:
- Streaming approach (O(1) memory)
- Lightweight difference detection using downscaled histograms
- Adaptive sampling based on scene changes
- Sharded output directories for file system efficiency
- Supports MP4, MKV, AVI formats
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import json


@dataclass
class SamplerConfig:
    """Configuration for the smart frame sampler."""
    min_interval_sec: float = 1.0  # Minimum seconds between samples
    max_interval_sec: float = 30.0  # Maximum seconds between samples (force sample)
    histogram_threshold: float = 0.3  # Histogram difference threshold (0-1)
    pixel_diff_threshold: float = 0.05  # Mean pixel difference threshold (0-1)
    comparison_size: Tuple[int, int] = (64, 64)  # Downscale size for comparison
    shard_size: int = 1000  # Max images per subdirectory
    output_format: str = "jpg"  # Output image format
    jpg_quality: int = 95  # JPEG quality (if using jpg)

    # Sensitivity presets
    sensitivity: str = "medium"  # "low", "medium", "high", "very_high"

    def apply_sensitivity(self):
        """Apply sensitivity preset to thresholds."""
        presets = {
            "low": (0.4, 0.08),      # Less sensitive - only big changes
            "medium": (0.3, 0.05),   # Default
            "high": (0.15, 0.025),   # More sensitive - catch subtle movements
            "very_high": (0.08, 0.012),  # Very sensitive - for static surveillance
        }
        if self.sensitivity in presets:
            self.histogram_threshold, self.pixel_diff_threshold = presets[self.sensitivity]


class SmartFrameSampler:
    """
    Efficiently samples diverse frames from video streams.

    Uses lightweight histogram and pixel difference comparison
    on downscaled frames to detect scene changes while maintaining
    O(1) memory usage.
    """

    def __init__(self, config: SamplerConfig = None):
        self.config = config or SamplerConfig()
        self.reference_frame = None
        self.reference_hist = None
        self.stats = {
            "frames_scanned": 0,
            "frames_extracted": 0,
            "processing_time": 0,
        }

    def _downsample(self, frame: np.ndarray) -> np.ndarray:
        """Downsample frame for fast comparison."""
        return cv2.resize(frame, self.config.comparison_size, interpolation=cv2.INTER_AREA)

    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute normalized grayscale histogram."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _histogram_difference(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Compute histogram difference using correlation.
        Returns value between 0 (identical) and 1 (completely different).
        """
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        # Convert from [-1, 1] to [0, 1] where 0 = same, 1 = different
        return (1 - correlation) / 2

    def _pixel_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute mean absolute pixel difference.
        Returns value between 0 (identical) and 1 (completely different).
        """
        diff = cv2.absdiff(frame1, frame2)
        return np.mean(diff) / 255.0

    def is_different_enough(self, frame: np.ndarray) -> bool:
        """
        Check if frame is sufficiently different from reference.
        Uses both histogram and pixel difference for robustness.
        """
        small_frame = self._downsample(frame)
        current_hist = self._compute_histogram(small_frame)

        if self.reference_frame is None:
            # First frame is always different
            self.reference_frame = small_frame
            self.reference_hist = current_hist
            return True

        # Check histogram difference (fast, catches color/brightness changes)
        hist_diff = self._histogram_difference(self.reference_hist, current_hist)

        # Check pixel difference (catches spatial changes)
        pixel_diff = self._pixel_difference(self.reference_frame, small_frame)

        # Frame is different if either metric exceeds threshold
        is_different = (hist_diff > self.config.histogram_threshold or
                       pixel_diff > self.config.pixel_diff_threshold)

        if is_different:
            self.reference_frame = small_frame
            self.reference_hist = current_hist

        return is_different

    def _get_shard_path(self, output_dir: Path, frame_count: int) -> Path:
        """Get sharded subdirectory path."""
        shard_num = frame_count // self.config.shard_size
        shard_dir = output_dir / f"{shard_num:04d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        return shard_dir

    def _save_frame(self, frame: np.ndarray, output_dir: Path,
                    frame_idx: int, timestamp: float) -> Path:
        """Save frame to sharded directory."""
        shard_dir = self._get_shard_path(output_dir, self.stats["frames_extracted"])

        # Filename includes original frame index and timestamp for traceability
        filename = f"frame_{frame_idx:08d}_t{timestamp:.2f}.{self.config.output_format}"
        filepath = shard_dir / filename

        if self.config.output_format == "jpg":
            cv2.imwrite(str(filepath), frame,
                       [cv2.IMWRITE_JPEG_QUALITY, self.config.jpg_quality])
        else:
            cv2.imwrite(str(filepath), frame)

        return filepath

    def sample_video(self, video_path: str, output_dir: str = None,
                     progress_callback=None) -> dict:
        """
        Sample diverse frames from a video file.

        Args:
            video_path: Path to input video file
            output_dir: Output directory (default: same as video with _frames suffix)
            progress_callback: Optional callback(current_frame, total_frames, extracted)

        Returns:
            Dictionary with extraction statistics
        """
        video_path = Path(video_path)
        if output_dir is None:
            output_dir = video_path.parent / f"{video_path.stem}_frames"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Reset state
        self.reference_frame = None
        self.reference_hist = None
        self.stats = {
            "frames_scanned": 0,
            "frames_extracted": 0,
            "processing_time": 0,
            "video_path": str(video_path),
            "output_dir": str(output_dir),
        }

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Calculate frame skip based on minimum/maximum interval
        min_frame_skip = int(fps * self.config.min_interval_sec)
        max_frame_skip = int(fps * self.config.max_interval_sec)

        self.stats["video_fps"] = fps
        self.stats["video_total_frames"] = total_frames
        self.stats["video_duration_sec"] = duration
        self.stats["min_frame_skip"] = min_frame_skip

        print(f"Video: {video_path.name}")
        print(f"Duration: {duration:.1f}s, FPS: {fps:.1f}, Total frames: {total_frames}")
        print(f"Min interval: {self.config.min_interval_sec}s ({min_frame_skip} frames)")
        print(f"Output: {output_dir}")
        print("-" * 50)

        start_time = time.time()
        frame_idx = 0
        last_extracted_frame = -min_frame_skip  # Allow first frame
        extracted_timestamps = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.stats["frames_scanned"] += 1
            timestamp = frame_idx / fps

            # Check minimum interval
            if frame_idx - last_extracted_frame >= min_frame_skip:
                frames_since_last = frame_idx - last_extracted_frame
                # Force extract if max interval exceeded, or check difference
                force_extract = frames_since_last >= max_frame_skip

                if force_extract or self.is_different_enough(frame):
                    self._save_frame(frame, output_dir, frame_idx, timestamp)
                    self.stats["frames_extracted"] += 1
                    last_extracted_frame = frame_idx
                    extracted_timestamps.append(timestamp)
                    # Reset reference even on forced extract
                    if force_extract and not self.is_different_enough(frame):
                        small_frame = self._downsample(frame)
                        self.reference_frame = small_frame
                        self.reference_hist = self._compute_histogram(small_frame)

            # Progress update
            if progress_callback:
                progress_callback(frame_idx, total_frames, self.stats["frames_extracted"])
            elif frame_idx % 500 == 0:
                pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                print(f"Progress: {pct:.1f}% ({self.stats['frames_extracted']} extracted)", end="\r")

            frame_idx += 1

        cap.release()

        self.stats["processing_time"] = time.time() - start_time
        self.stats["extraction_rate"] = (self.stats["frames_extracted"] /
                                         self.stats["frames_scanned"] * 100
                                         if self.stats["frames_scanned"] > 0 else 0)
        self.stats["extracted_timestamps"] = extracted_timestamps

        # Save metadata
        metadata_path = output_dir / "extraction_metadata.json"
        with open(metadata_path, "w") as f:
            # Don't save full timestamp list to JSON (can be large)
            stats_for_json = {k: v for k, v in self.stats.items()
                            if k != "extracted_timestamps"}
            stats_for_json["num_timestamps"] = len(extracted_timestamps)
            json.dump(stats_for_json, f, indent=2)

        print(f"\nCompleted in {self.stats['processing_time']:.1f}s")
        print(f"Scanned: {self.stats['frames_scanned']} frames")
        print(f"Extracted: {self.stats['frames_extracted']} frames ({self.stats['extraction_rate']:.1f}%)")

        return self.stats


def main():
    parser = argparse.ArgumentParser(
        description="Smart Frame Sampler - Extract diverse frames from videos"
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("-o", "--output", help="Output directory (default: video_frames)")
    parser.add_argument("--min-interval", type=float, default=1.0,
                       help="Minimum seconds between samples (default: 1.0)")
    parser.add_argument("--max-interval", type=float, default=30.0,
                       help="Maximum seconds between samples - force extract (default: 30.0)")
    parser.add_argument("--sensitivity", choices=["low", "medium", "high", "very_high"],
                       default="medium", help="Detection sensitivity preset (default: medium)")
    parser.add_argument("--hist-threshold", type=float, default=None,
                       help="Histogram difference threshold 0-1 (overrides sensitivity)")
    parser.add_argument("--pixel-threshold", type=float, default=None,
                       help="Pixel difference threshold 0-1 (overrides sensitivity)")
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg",
                       help="Output image format (default: jpg)")
    parser.add_argument("--quality", type=int, default=95,
                       help="JPEG quality 1-100 (default: 95)")
    parser.add_argument("--shard-size", type=int, default=1000,
                       help="Max images per subdirectory (default: 1000)")

    args = parser.parse_args()

    config = SamplerConfig(
        min_interval_sec=args.min_interval,
        max_interval_sec=args.max_interval,
        sensitivity=args.sensitivity,
        output_format=args.format,
        jpg_quality=args.quality,
        shard_size=args.shard_size,
    )

    # Apply sensitivity preset first
    config.apply_sensitivity()

    # Override with explicit thresholds if provided
    if args.hist_threshold is not None:
        config.histogram_threshold = args.hist_threshold
    if args.pixel_threshold is not None:
        config.pixel_diff_threshold = args.pixel_threshold

    sampler = SmartFrameSampler(config)
    sampler.sample_video(args.video, args.output)


if __name__ == "__main__":
    main()
