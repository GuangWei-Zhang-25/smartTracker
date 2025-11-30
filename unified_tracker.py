"""
Unified Mouse Tracker
=====================
Combines YOLO segmentation with SAM3 video tracking for robust
single-target tracking with consistent ID across occlusions.

Tracking Strategy:
1. YOLO provides fast detection in normal conditions
2. When YOLO loses track or detects occlusion, SAM3 takes over
3. SAM3 uses memory-based video tracking to maintain ID through occlusion
4. Track state machine manages transitions between modes

States:
- TRACKING: Normal YOLO tracking with high confidence
- OCCLUSION: SAM3 memory-based tracking during overlap
- RECOVERY: Attempting to re-acquire target after loss
- LOST: Target completely lost, needs re-initialization
"""

import cv2
import numpy as np
from pathlib import Path
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import time

# Try importing models
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import torch
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TrackState(Enum):
    """Tracking state machine states."""
    TRACKING = "tracking"       # Normal YOLO tracking
    OCCLUSION = "occlusion"     # SAM3 handling occlusion
    RECOVERY = "recovery"       # Trying to re-acquire
    LOST = "lost"               # Target lost


@dataclass
class TrackingResult:
    """Result from a single frame tracking."""
    frame_idx: int
    timestamp: float
    state: TrackState
    mask: Optional[np.ndarray] = None
    bbox: Optional[Tuple[float, float, float, float]] = None  # x, y, w, h
    centroid: Optional[Tuple[float, float]] = None
    confidence: float = 0.0
    source: str = "none"  # "yolo", "sam3", "prediction"


@dataclass
class TrackerConfig:
    """Configuration for unified tracker."""
    # YOLO settings
    yolo_conf_threshold: float = 0.5
    yolo_iou_threshold: float = 0.5

    # Target matching
    max_centroid_distance: float = 0.15  # Max normalized distance for match
    min_iou_match: float = 0.3           # Min IoU to consider same target

    # State transitions
    occlusion_frames_threshold: int = 3   # Frames before switching to SAM3
    recovery_frames_timeout: int = 30     # Frames to try recovery before lost
    lost_frames_reinit: int = 60          # Frames before allowing reinit

    # SAM3 settings (when available)
    sam3_memory_frames: int = 6           # Number of frames in SAM3 memory
    sam3_prompt_interval: int = 10        # Re-prompt interval during occlusion


class MotionPredictor:
    """Simple motion model for prediction during brief losses."""

    def __init__(self, history_size: int = 5):
        self.history_size = history_size
        self.positions: List[Tuple[float, float]] = []
        self.velocities: List[Tuple[float, float]] = []

    def update(self, centroid: Tuple[float, float]):
        """Update with new position."""
        if self.positions:
            # Calculate velocity
            last = self.positions[-1]
            vx = centroid[0] - last[0]
            vy = centroid[1] - last[1]
            self.velocities.append((vx, vy))

            # Limit history
            if len(self.velocities) > self.history_size:
                self.velocities.pop(0)

        self.positions.append(centroid)
        if len(self.positions) > self.history_size:
            self.positions.pop(0)

    def predict(self, steps: int = 1) -> Optional[Tuple[float, float]]:
        """Predict position steps frames ahead."""
        if not self.positions or not self.velocities:
            return None

        # Average velocity
        avg_vx = sum(v[0] for v in self.velocities) / len(self.velocities)
        avg_vy = sum(v[1] for v in self.velocities) / len(self.velocities)

        # Predict
        last = self.positions[-1]
        pred_x = last[0] + avg_vx * steps
        pred_y = last[1] + avg_vy * steps

        # Clamp to valid range
        pred_x = max(0.0, min(1.0, pred_x))
        pred_y = max(0.0, min(1.0, pred_y))

        return (pred_x, pred_y)

    def reset(self):
        """Reset predictor state."""
        self.positions.clear()
        self.velocities.clear()


class UnifiedTracker:
    """
    Unified tracker combining YOLO and SAM3 with consistent ID.
    """

    def __init__(self, config: TrackerConfig = None,
                 yolo_weights: str = None, sam3_model: str = None,
                 hf_token: str = None, device: str = "cuda"):
        self.config = config or TrackerConfig()
        self.device = device

        # Models
        self.yolo = None
        self.sam3_model = None
        self.sam3_processor = None

        # Tracking state
        self.state = TrackState.LOST
        self.target_id = 1  # Consistent ID for the target
        self.frames_in_state = 0
        self.last_good_mask = None
        self.last_good_centroid = None
        self.motion_predictor = MotionPredictor()

        # SAM3 memory for video tracking
        self.sam3_memory = []

        # Load models
        if yolo_weights:
            self._load_yolo(yolo_weights)

        if sam3_model:
            self._load_sam3(sam3_model, hf_token)

        # Tracking history
        self.history: List[TrackingResult] = []

    def _load_yolo(self, weights_path: str):
        """Load YOLO segmentation model."""
        if not YOLO_AVAILABLE:
            print("Warning: ultralytics not installed, YOLO disabled")
            return

        print(f"Loading YOLO: {weights_path}")
        self.yolo = YOLO(weights_path)
        print("YOLO loaded")

    def _load_sam3(self, model_name: str, hf_token: str = None):
        """Load SAM3 model for video tracking."""
        if not TORCH_AVAILABLE:
            print("Warning: torch not installed, SAM3 disabled")
            return

        try:
            from transformers import Sam3Model, Sam3Processor
            print(f"Loading SAM3: {model_name}")
            self.sam3_processor = Sam3Processor.from_pretrained(model_name, token=hf_token)
            self.sam3_model = Sam3Model.from_pretrained(model_name, token=hf_token)

            if self.device == "cuda" and torch.cuda.is_available():
                self.sam3_model = self.sam3_model.to("cuda")
            print("SAM3 loaded")
        except Exception as e:
            print(f"Warning: Could not load SAM3: {e}")

    def initialize_target(self, frame: np.ndarray, mask: np.ndarray = None,
                         bbox: Tuple[int, int, int, int] = None,
                         centroid: Tuple[float, float] = None):
        """
        Initialize target for tracking.

        Args:
            frame: First frame (BGR)
            mask: Initial mask (optional)
            bbox: Initial bounding box [x, y, w, h] (optional)
            centroid: Initial centroid [x_norm, y_norm] (optional)
        """
        self.state = TrackState.TRACKING
        self.frames_in_state = 0
        self.motion_predictor.reset()

        if mask is not None:
            self.last_good_mask = mask.copy()
            # Calculate centroid from mask
            if centroid is None:
                coords = np.column_stack(np.where(mask > 0))
                if len(coords) > 0:
                    h, w = mask.shape[:2]
                    cy = coords[:, 0].mean() / h
                    cx = coords[:, 1].mean() / w
                    centroid = (cx, cy)

        if centroid is not None:
            self.last_good_centroid = centroid
            self.motion_predictor.update(centroid)

        # Initialize SAM3 memory if available
        if self.sam3_model is not None and mask is not None:
            self._update_sam3_memory(frame, mask)

        print(f"Target initialized at centroid: {centroid}")

    def _update_sam3_memory(self, frame: np.ndarray, mask: np.ndarray):
        """Update SAM3 memory bank for video tracking."""
        # Convert to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        memory_entry = {
            'image': pil_image,
            'mask': mask.copy(),
        }

        self.sam3_memory.append(memory_entry)

        # Limit memory size
        if len(self.sam3_memory) > self.config.sam3_memory_frames:
            self.sam3_memory.pop(0)

    def _yolo_detect(self, frame: np.ndarray) -> List[Dict]:
        """Run YOLO detection on frame."""
        if self.yolo is None:
            return []

        results = self.yolo(frame, conf=self.config.yolo_conf_threshold,
                           iou=self.config.yolo_iou_threshold, verbose=False)

        detections = []
        for result in results:
            if result.masks is None:
                continue

            for i, mask in enumerate(result.masks.data):
                mask_np = mask.cpu().numpy()
                conf = float(result.boxes.conf[i])

                # Get bbox
                box = result.boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = box
                h, w = frame.shape[:2]

                # Calculate centroid
                coords = np.column_stack(np.where(mask_np > 0.5))
                if len(coords) > 0:
                    cy = coords[:, 0].mean() / mask_np.shape[0]
                    cx = coords[:, 1].mean() / mask_np.shape[1]
                else:
                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h

                detections.append({
                    'mask': mask_np,
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'centroid': (cx, cy),
                    'confidence': conf,
                    'area': np.sum(mask_np > 0.5)
                })

        return detections

    def _match_detection(self, detections: List[Dict]) -> Optional[Dict]:
        """Match detections to target based on position and motion prediction."""
        if not detections:
            return None

        if self.last_good_centroid is None:
            # No reference - return highest confidence
            return max(detections, key=lambda d: d['confidence'])

        # Use motion prediction if available
        predicted = self.motion_predictor.predict(1)
        ref_point = predicted if predicted else self.last_good_centroid

        best_match = None
        best_score = float('inf')

        for det in detections:
            # Distance to reference
            dx = det['centroid'][0] - ref_point[0]
            dy = det['centroid'][1] - ref_point[1]
            dist = (dx * dx + dy * dy) ** 0.5

            if dist < self.config.max_centroid_distance:
                # Score: distance weighted by confidence
                score = dist / (det['confidence'] + 0.1)
                if score < best_score:
                    best_score = score
                    best_match = det

        return best_match

    def _sam3_track(self, frame: np.ndarray) -> Optional[Dict]:
        """Use SAM3 for tracking during occlusion."""
        if self.sam3_model is None or not self.sam3_memory:
            return None

        # This is a simplified SAM3 tracking - actual implementation
        # would use SAM3's video memory feature
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Use last good centroid as point prompt
            if self.last_good_centroid:
                h, w = frame.shape[:2]
                px = int(self.last_good_centroid[0] * w)
                py = int(self.last_good_centroid[1] * h)

                inputs = self.sam3_processor(
                    images=pil_image,
                    input_points=[[[px, py]]],
                    return_tensors="pt"
                )

                if self.device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.sam3_model(**inputs)

                masks = self.sam3_processor.post_process_masks(
                    outputs.pred_masks,
                    inputs["original_sizes"],
                    inputs["reshaped_input_sizes"]
                )

                if masks and len(masks[0]) > 0:
                    mask_np = masks[0][0].cpu().numpy().squeeze()
                    mask_np = (mask_np > 0.5).astype(np.uint8)

                    coords = np.column_stack(np.where(mask_np > 0))
                    if len(coords) > 0:
                        cy = coords[:, 0].mean() / mask_np.shape[0]
                        cx = coords[:, 1].mean() / mask_np.shape[1]

                        return {
                            'mask': mask_np,
                            'centroid': (cx, cy),
                            'confidence': 0.7,  # SAM3 doesn't provide confidence
                            'area': np.sum(mask_np > 0)
                        }
        except Exception as e:
            print(f"SAM3 tracking error: {e}")

        return None

    def _detect_occlusion(self, detections: List[Dict]) -> bool:
        """Detect if target is occluded (merged with another object)."""
        if len(detections) != 2:
            return False

        # Check if two detections are very close or overlapping
        d1, d2 = detections[0], detections[1]

        dx = d1['centroid'][0] - d2['centroid'][0]
        dy = d1['centroid'][1] - d2['centroid'][1]
        dist = (dx * dx + dy * dy) ** 0.5

        # Occlusion if centroids are very close
        return dist < 0.08  # Less than 8% of image dimension

    def track_frame(self, frame: np.ndarray, frame_idx: int = 0,
                   timestamp: float = 0.0) -> TrackingResult:
        """
        Track target in a single frame.

        Args:
            frame: Input frame (BGR)
            frame_idx: Frame index
            timestamp: Frame timestamp

        Returns:
            TrackingResult with tracking info
        """
        self.frames_in_state += 1

        # Run YOLO detection
        detections = self._yolo_detect(frame)

        result = TrackingResult(
            frame_idx=frame_idx,
            timestamp=timestamp,
            state=self.state,
        )

        # State machine logic
        if self.state == TrackState.TRACKING:
            if len(detections) == 0:
                # No detection - might be brief loss
                self.frames_in_state = 0
                self.state = TrackState.RECOVERY
                result.state = self.state

            elif self._detect_occlusion(detections):
                # Occlusion detected - switch to SAM3
                self.frames_in_state = 0
                self.state = TrackState.OCCLUSION
                result.state = self.state

            else:
                # Normal tracking
                match = self._match_detection(detections)
                if match:
                    result.mask = match['mask']
                    result.centroid = match['centroid']
                    result.confidence = match['confidence']
                    result.source = "yolo"

                    # Update state
                    self.last_good_mask = match['mask']
                    self.last_good_centroid = match['centroid']
                    self.motion_predictor.update(match['centroid'])

                    # Update SAM3 memory
                    if self.sam3_model is not None:
                        self._update_sam3_memory(frame, match['mask'])

        elif self.state == TrackState.OCCLUSION:
            # Try SAM3 tracking
            sam_result = self._sam3_track(frame)

            if sam_result:
                result.mask = sam_result['mask']
                result.centroid = sam_result['centroid']
                result.confidence = sam_result['confidence']
                result.source = "sam3"

                self.last_good_centroid = sam_result['centroid']
                self.motion_predictor.update(sam_result['centroid'])

            # Check if occlusion resolved
            if not self._detect_occlusion(detections) and len(detections) > 0:
                # Try to re-match with YOLO
                match = self._match_detection(detections)
                if match and match['confidence'] > 0.6:
                    self.state = TrackState.TRACKING
                    self.frames_in_state = 0

        elif self.state == TrackState.RECOVERY:
            # Try to re-acquire with YOLO
            match = self._match_detection(detections)

            if match:
                result.mask = match['mask']
                result.centroid = match['centroid']
                result.confidence = match['confidence']
                result.source = "yolo"

                self.last_good_mask = match['mask']
                self.last_good_centroid = match['centroid']
                self.motion_predictor.update(match['centroid'])

                self.state = TrackState.TRACKING
                self.frames_in_state = 0

            elif self.frames_in_state > self.config.recovery_frames_timeout:
                # Timeout - mark as lost
                self.state = TrackState.LOST
                self.frames_in_state = 0

            else:
                # Use motion prediction
                predicted = self.motion_predictor.predict(self.frames_in_state)
                if predicted:
                    result.centroid = predicted
                    result.source = "prediction"
                    result.confidence = max(0.1, 0.5 - self.frames_in_state * 0.05)

        elif self.state == TrackState.LOST:
            # Waiting for re-initialization
            result.source = "none"

        # Calculate bbox from centroid if we have one
        if result.centroid and result.mask is not None:
            coords = np.column_stack(np.where(result.mask > 0.5))
            if len(coords) > 0:
                h, w = result.mask.shape[:2]
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                result.bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        # Store in history
        self.history.append(result)

        return result

    def track_video(self, video_path: str, output_path: str = None,
                   init_frame: int = 0, init_bbox: Tuple = None,
                   progress_callback=None) -> List[TrackingResult]:
        """
        Track target through entire video.

        Args:
            video_path: Path to input video
            output_path: Optional path for output video with visualization
            init_frame: Frame to initialize tracking
            init_bbox: Initial bounding box [x, y, w, h]
            progress_callback: Optional callback(frame_idx, total_frames)

        Returns:
            List of TrackingResults
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video: {video_path}")
        print(f"Frames: {total_frames}, FPS: {fps:.1f}, Size: {width}x{height}")

        # Setup output video
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps

            # Initialize on specified frame
            if frame_idx == init_frame and self.state == TrackState.LOST:
                if init_bbox:
                    x, y, w, h = init_bbox
                    cx = (x + w/2) / width
                    cy = (y + h/2) / height
                    self.initialize_target(frame, centroid=(cx, cy))
                else:
                    # Auto-initialize from first YOLO detection
                    detections = self._yolo_detect(frame)
                    if detections:
                        best = max(detections, key=lambda d: d['confidence'])
                        self.initialize_target(frame, mask=best['mask'],
                                             centroid=best['centroid'])

            # Track
            result = self.track_frame(frame, frame_idx, timestamp)
            results.append(result)

            # Visualize
            if writer:
                vis_frame = self._visualize_frame(frame, result)
                writer.write(vis_frame)

            # Progress
            if progress_callback:
                progress_callback(frame_idx, total_frames)
            elif frame_idx % 100 == 0:
                pct = (frame_idx / total_frames) * 100
                print(f"Progress: {pct:.1f}%", end='\r')

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        print(f"\nTracking complete: {len(results)} frames")
        return results

    def _visualize_frame(self, frame: np.ndarray, result: TrackingResult) -> np.ndarray:
        """Draw tracking visualization on frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Draw mask
        if result.mask is not None:
            mask_resized = cv2.resize(result.mask.astype(np.uint8),
                                     (w, h), interpolation=cv2.INTER_NEAREST)
            color = (0, 255, 0) if result.state == TrackState.TRACKING else (0, 255, 255)

            overlay = vis.copy()
            overlay[mask_resized > 0] = color
            vis = cv2.addWeighted(overlay, 0.3, vis, 0.7, 0)

            # Draw contour
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, color, 2)

        # Draw centroid
        if result.centroid:
            cx, cy = int(result.centroid[0] * w), int(result.centroid[1] * h)
            cv2.circle(vis, (cx, cy), 8, (0, 0, 255), -1)
            cv2.circle(vis, (cx, cy), 10, (255, 255, 255), 2)

        # Info bar
        info = f"ID:{self.target_id} | {result.state.value} | {result.source} | conf:{result.confidence:.2f}"
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        frame_info = f"Frame: {result.frame_idx} | t={result.timestamp:.2f}s"
        cv2.putText(vis, frame_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(vis, frame_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return vis

    def save_results(self, output_path: str):
        """Save tracking results to JSON."""
        data = {
            'target_id': self.target_id,
            'total_frames': len(self.history),
            'config': {
                'yolo_conf': self.config.yolo_conf_threshold,
                'max_distance': self.config.max_centroid_distance,
            },
            'tracks': []
        }

        for r in self.history:
            track = {
                'frame_idx': r.frame_idx,
                'timestamp': r.timestamp,
                'state': r.state.value,
                'centroid': r.centroid,
                'confidence': r.confidence,
                'source': r.source,
            }
            if r.bbox:
                track['bbox'] = list(r.bbox)
            data['tracks'].append(track)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Mouse Tracker - YOLO + SAM3 with consistent ID"
    )
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--yolo-weights", required=True, help="YOLO segmentation weights")
    parser.add_argument("--sam3-model", default="facebook/sam3",
                       help="SAM3 model name (default: facebook/sam3)")
    parser.add_argument("--hf-token", help="HuggingFace token for SAM3")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("--results", help="Output JSON results path")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--init-frame", type=int, default=0,
                       help="Frame to initialize tracking (default: 0)")

    args = parser.parse_args()

    config = TrackerConfig()
    tracker = UnifiedTracker(
        config=config,
        yolo_weights=args.yolo_weights,
        sam3_model=args.sam3_model if args.hf_token else None,
        hf_token=args.hf_token,
        device=args.device
    )

    results = tracker.track_video(
        args.video,
        output_path=args.output,
        init_frame=args.init_frame
    )

    if args.results:
        tracker.save_results(args.results)

    # Print summary
    states = {}
    for r in results:
        states[r.state.value] = states.get(r.state.value, 0) + 1

    print("\nTracking Summary:")
    for state, count in sorted(states.items()):
        pct = 100 * count / len(results)
        print(f"  {state}: {count} frames ({pct:.1f}%)")


if __name__ == "__main__":
    main()
