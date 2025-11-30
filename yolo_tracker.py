"""
YOLO Tracker with SAM3 Supervision
===================================
Tracks multiple objects using YOLO segmentation with SAM3 as a supervisor
to verify new ID generation and prevent false positives.

Key Features:
1. SAM3 verifies new IDs before creation
2. Object count is capped by SAM3's majority object count
3. Detects proximity situations for potential ID switches
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed")

try:
    import torch
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not installed, SAM3 supervision disabled")


@dataclass
class TrackedObject:
    """Represents a tracked object with history."""
    id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    centroid: Tuple[float, float]
    mask: Optional[np.ndarray] = None
    confidence: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    velocity: Tuple[float, float] = (0.0, 0.0)
    frames_since_update: int = 0

    def update(self, bbox, centroid, mask=None, confidence=0.0):
        """Update track with new detection."""
        # Calculate velocity from previous position
        if self.history:
            prev_centroid = self.history[-1]
            self.velocity = (centroid[0] - prev_centroid[0],
                           centroid[1] - prev_centroid[1])

        self.bbox = bbox
        self.centroid = centroid
        self.mask = mask
        self.confidence = confidence
        self.history.append(centroid)
        self.frames_since_update = 0

    def predict_position(self) -> Tuple[float, float]:
        """Predict next position based on velocity."""
        return (self.centroid[0] + self.velocity[0],
                self.centroid[1] + self.velocity[1])


@dataclass
class ProximityEvent:
    """Records when objects are close together."""
    frame_idx: int
    timestamp: float
    obj1_id: int
    obj2_id: int
    distance: float
    obj1_centroid: Tuple[float, float]
    obj2_centroid: Tuple[float, float]
    overlap_iou: float = 0.0


class YOLOTracker:
    """
    YOLO-based multi-object tracker with SAM3 supervision.

    Features:
    - SAM3 verifies new ID generation to prevent false positives
    - Object count capped by expected_object_count (from SAM3 majority voting)
    - Detects proximity situations for potential ID switches
    """

    def __init__(self, model_path: str,
                 proximity_threshold: float = 100.0,
                 iou_threshold: float = 0.3,
                 conf_threshold: float = 0.5,
                 max_age: int = 30,
                 expected_object_count: int = None,
                 sam3_model: str = None,
                 sam3_conf_threshold: float = 0.5,
                 hf_token: str = None,
                 device: str = "cuda"):
        """
        Initialize tracker.

        Args:
            model_path: Path to YOLO model weights
            proximity_threshold: Distance threshold (pixels) for close objects
            iou_threshold: IoU threshold for matching detections to tracks
            conf_threshold: Confidence threshold for detections
            max_age: Max frames before track is lost
            expected_object_count: Maximum number of objects (from SAM3 majority voting)
            sam3_model: SAM3 model name for verification (e.g., "facebook/sam3")
            sam3_conf_threshold: Confidence threshold for SAM3 verification
            hf_token: HuggingFace token for SAM3 model access
            device: Device for SAM3 ("cuda" or "cpu")
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics not installed")

        self.model = YOLO(model_path)
        self.proximity_threshold = proximity_threshold
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.max_age = max_age

        # SAM3 supervision settings
        self.expected_object_count = expected_object_count
        self.sam3_conf_threshold = sam3_conf_threshold
        self.device = device
        self.sam3_model = None
        self.sam3_processor = None

        # Load SAM3 if specified
        if sam3_model and TORCH_AVAILABLE:
            self._load_sam3(sam3_model, hf_token)

        self.tracks: Dict[int, TrackedObject] = {}  # Active tracks
        self.lost_tracks: Dict[int, TrackedObject] = {}  # Lost but recoverable tracks
        self.next_id = 0
        self.frame_idx = 0
        self.proximity_events: List[ProximityEvent] = []

        # Statistics for new ID verification
        self.stats = {
            "new_ids_created": 0,
            "new_ids_rejected_count_limit": 0,
            "new_ids_rejected_sam3": 0,
            "sam3_verifications": 0,
            "track_recoveries": 0,  # Times a lost track was recovered
        }

    def _load_sam3(self, model_name: str, hf_token: str = None):
        """Load SAM3 model for verification."""
        try:
            from transformers import Sam3Model, Sam3Processor
            print(f"Loading SAM3 for supervision: {model_name}")

            self.sam3_processor = Sam3Processor.from_pretrained(model_name, token=hf_token)
            self.sam3_model = Sam3Model.from_pretrained(model_name, token=hf_token)

            if self.device == "cuda" and torch.cuda.is_available():
                self.sam3_model = self.sam3_model.to("cuda")
                print(f"SAM3 loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("SAM3 loaded on CPU")
        except Exception as e:
            print(f"Warning: Could not load SAM3: {e}")
            self.sam3_model = None
            self.sam3_processor = None

    def _verify_detection_with_sam3(self, frame: np.ndarray, detection: Dict) -> bool:
        """
        Use SAM3 to verify if a detection is a valid object.

        Args:
            frame: BGR image
            detection: Detection dict with 'centroid', 'bbox', etc.

        Returns:
            True if SAM3 confirms object exists at location, False otherwise
        """
        if self.sam3_model is None:
            # No SAM3 available, allow detection
            return True

        self.stats["sam3_verifications"] += 1

        try:
            # Convert frame to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            h, w = frame.shape[:2]

            # Use centroid as point prompt
            cx, cy = detection['centroid']
            px, py = int(cx), int(cy)

            # Process with point prompt
            inputs = self.sam3_processor(
                images=pil_image,
                input_points=[[[px, py]]],
                return_tensors="pt"
            )

            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.sam3_model(**inputs)

            # Get predicted masks
            masks = self.sam3_processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"],
                inputs["reshaped_input_sizes"]
            )

            if masks and len(masks[0]) > 0:
                # Get the best mask
                mask_np = masks[0][0].cpu().numpy().squeeze()
                mask_area = np.sum(mask_np > 0.5)

                # Check if SAM3 found a meaningful object
                # Compare with YOLO detection area
                yolo_mask = detection.get('mask')
                if yolo_mask is not None:
                    yolo_area = np.sum(yolo_mask > 0.5)
                    # SAM3 should find similar or larger area
                    area_ratio = mask_area / max(yolo_area, 1)

                    # Valid if SAM3 area is at least 30% of YOLO area
                    if area_ratio >= 0.3:
                        return True
                    else:
                        print(f"  SAM3 rejected: area_ratio={area_ratio:.2f} (too small)")
                        return False
                else:
                    # No YOLO mask, just check if SAM3 found something
                    min_area = 100  # minimum pixel area
                    if mask_area >= min_area:
                        return True
                    else:
                        print(f"  SAM3 rejected: mask_area={mask_area} (below minimum)")
                        return False

            print("  SAM3 rejected: no mask found")
            return False

        except Exception as e:
            print(f"  SAM3 verification error: {e}")
            # On error, allow detection (fail-open)
            return True

    def _compute_iou(self, box1, box2) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0
        return inter_area / union_area

    def _compute_distance(self, c1, c2) -> float:
        """Compute Euclidean distance between centroids."""
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def _get_centroid(self, bbox) -> Tuple[float, float]:
        """Get centroid from bbox."""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _match_detections(self, detections: List[Dict]) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Match detections to existing tracks (active and lost).
        Returns:
            - matches: mapping of detection index to active track ID
            - lost_matches: mapping of detection index to lost track ID (for recovery)
        """
        matches = {}
        lost_matches = {}

        if not detections:
            return matches, lost_matches

        # First, match to active tracks
        if self.tracks:
            track_ids = list(self.tracks.keys())
            n_tracks = len(track_ids)
            n_dets = len(detections)

            # Build cost matrix based on distance
            cost_matrix = np.full((n_dets, n_tracks), np.inf)

            for i, det in enumerate(detections):
                det_centroid = det['centroid']
                for j, track_id in enumerate(track_ids):
                    track = self.tracks[track_id]
                    # Use predicted position for matching
                    pred_pos = track.predict_position()
                    dist = self._compute_distance(det_centroid, pred_pos)
                    iou = self._compute_iou(det['bbox'], track.bbox)

                    # Combined cost: distance and inverse IoU
                    if dist < self.proximity_threshold * 2:  # Only consider reasonable matches
                        cost_matrix[i, j] = dist - iou * 50  # IoU bonus

            # Simple greedy matching (could use Hungarian for optimal)
            used_tracks = set()

            # Sort detections by lowest cost match
            det_order = []
            for i in range(n_dets):
                min_cost = np.min(cost_matrix[i])
                det_order.append((min_cost, i))
            det_order.sort()

            for _, det_idx in det_order:
                best_track_idx = -1
                best_cost = np.inf

                for track_idx in range(n_tracks):
                    if track_ids[track_idx] not in used_tracks:
                        if cost_matrix[det_idx, track_idx] < best_cost:
                            best_cost = cost_matrix[det_idx, track_idx]
                            best_track_idx = track_idx

                if best_track_idx >= 0 and best_cost < self.proximity_threshold * 2:
                    matches[det_idx] = track_ids[best_track_idx]
                    used_tracks.add(track_ids[best_track_idx])

        # Second, try to match unmatched detections to lost tracks
        unmatched_det_indices = [i for i in range(len(detections)) if i not in matches]

        if self.lost_tracks and unmatched_det_indices:
            lost_track_ids = list(self.lost_tracks.keys())
            n_lost = len(lost_track_ids)
            n_unmatched = len(unmatched_det_indices)

            # Build cost matrix for lost tracks (use last known position)
            lost_cost_matrix = np.full((n_unmatched, n_lost), np.inf)

            for i, det_idx in enumerate(unmatched_det_indices):
                det = detections[det_idx]
                det_centroid = det['centroid']
                for j, track_id in enumerate(lost_track_ids):
                    track = self.lost_tracks[track_id]
                    # Use last known position (no prediction for lost tracks)
                    dist = self._compute_distance(det_centroid, track.centroid)

                    # More lenient threshold for lost track recovery
                    if dist < self.proximity_threshold * 4:
                        lost_cost_matrix[i, j] = dist

            # Greedy matching for lost tracks
            used_lost_tracks = set()
            lost_det_order = []
            for i in range(n_unmatched):
                min_cost = np.min(lost_cost_matrix[i])
                lost_det_order.append((min_cost, i))
            lost_det_order.sort()

            for _, idx in lost_det_order:
                det_idx = unmatched_det_indices[idx]
                best_lost_idx = -1
                best_cost = np.inf

                for lost_idx in range(n_lost):
                    if lost_track_ids[lost_idx] not in used_lost_tracks:
                        if lost_cost_matrix[idx, lost_idx] < best_cost:
                            best_cost = lost_cost_matrix[idx, lost_idx]
                            best_lost_idx = lost_idx

                if best_lost_idx >= 0 and best_cost < self.proximity_threshold * 4:
                    lost_matches[det_idx] = lost_track_ids[best_lost_idx]
                    used_lost_tracks.add(lost_track_ids[best_lost_idx])

        return matches, lost_matches

    def _check_proximity(self, timestamp: float):
        """Check for close proximity between tracks."""
        track_ids = list(self.tracks.keys())

        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                track1 = self.tracks[track_ids[i]]
                track2 = self.tracks[track_ids[j]]

                dist = self._compute_distance(track1.centroid, track2.centroid)
                iou = self._compute_iou(track1.bbox, track2.bbox)

                # Record proximity event
                if dist < self.proximity_threshold or iou > 0.1:
                    event = ProximityEvent(
                        frame_idx=self.frame_idx,
                        timestamp=timestamp,
                        obj1_id=track_ids[i],
                        obj2_id=track_ids[j],
                        distance=dist,
                        obj1_centroid=track1.centroid,
                        obj2_centroid=track2.centroid,
                        overlap_iou=iou
                    )
                    self.proximity_events.append(event)

    def process_frame(self, frame: np.ndarray, timestamp: float = 0.0) -> Dict:
        """
        Process a single frame.

        Args:
            frame: BGR image
            timestamp: Frame timestamp in seconds

        Returns:
            Dict with tracking results and proximity info
        """
        # Run YOLO inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)

        # Parse detections
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else None

            for idx, box in enumerate(boxes):
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                centroid = self._get_centroid(bbox)

                mask = None
                if masks is not None and idx < len(masks):
                    mask = masks[idx]

                detections.append({
                    'bbox': tuple(bbox),
                    'centroid': centroid,
                    'confidence': conf,
                    'mask': mask
                })

        # Match detections to tracks (active and lost)
        matches, lost_matches = self._match_detections(detections)

        # Update matched active tracks
        matched_det_indices = set()
        for det_idx, track_id in matches.items():
            det = detections[det_idx]
            self.tracks[track_id].update(
                det['bbox'], det['centroid'],
                det['mask'], det['confidence']
            )
            matched_det_indices.add(det_idx)

        # Recover lost tracks that were re-matched
        for det_idx, track_id in lost_matches.items():
            det = detections[det_idx]
            # Move track from lost back to active
            track = self.lost_tracks.pop(track_id)
            track.update(
                det['bbox'], det['centroid'],
                det['mask'], det['confidence']
            )
            self.tracks[track_id] = track
            matched_det_indices.add(det_idx)
            self.stats["track_recoveries"] += 1
            print(f"  Frame {self.frame_idx}: Recovered lost track ID {track_id}")

        # Handle unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_det_indices:
                # Check 1: Current track count limit
                # Count active + lost tracks (lost tracks can be recovered, so they count)
                if self.expected_object_count is not None:
                    current_track_count = len(self.tracks) + len(self.lost_tracks)
                    if current_track_count >= self.expected_object_count:
                        self.stats["new_ids_rejected_count_limit"] += 1
                        # Only print occasionally to avoid spam
                        if self.frame_idx % 100 == 0:
                            print(f"  Frame {self.frame_idx}: Rejected new ID - "
                                  f"max tracks reached ({current_track_count}/{self.expected_object_count})")
                        continue

                # Check 2: SAM3 verification (if enabled)
                if self.sam3_model is not None:
                    if not self._verify_detection_with_sam3(frame, det):
                        self.stats["new_ids_rejected_sam3"] += 1
                        print(f"  Frame {self.frame_idx}: Rejected new ID - SAM3 verification failed")
                        continue

                # Passed all checks - create new track
                # For single-object tracking, reuse ID 0 to maintain consistency
                if self.expected_object_count == 1 and self.next_id > 0:
                    # Reuse ID 0 for single-object scenarios
                    track_id = 0
                    print(f"  Frame {self.frame_idx}: Reusing ID 0 (single-object mode)")
                else:
                    track_id = self.next_id
                    self.next_id += 1
                    print(f"  Frame {self.frame_idx}: Created new ID {track_id} "
                          f"(total unique IDs: {self.next_id})")

                new_track = TrackedObject(
                    id=track_id,
                    bbox=det['bbox'],
                    centroid=det['centroid'],
                    mask=det['mask'],
                    confidence=det['confidence']
                )
                new_track.history.append(det['centroid'])
                self.tracks[track_id] = new_track
                self.stats["new_ids_created"] += 1

        # Update unmatched active tracks
        for track_id in list(self.tracks.keys()):
            if track_id not in matches.values() and track_id not in lost_matches.values():
                self.tracks[track_id].frames_since_update += 1
                track = self.tracks[track_id]

                # Use prediction for short gaps (up to max_age frames)
                if track.frames_since_update <= self.max_age:
                    # Use predicted position to maintain track continuity
                    predicted_pos = track.predict_position()
                    # Update centroid to predicted position (keep other attributes)
                    track.centroid = predicted_pos
                    # Update bbox based on predicted position (maintain size)
                    bbox_w = track.bbox[2] - track.bbox[0]
                    bbox_h = track.bbox[3] - track.bbox[1]
                    track.bbox = (
                        predicted_pos[0] - bbox_w / 2,
                        predicted_pos[1] - bbox_h / 2,
                        predicted_pos[0] + bbox_w / 2,
                        predicted_pos[1] + bbox_h / 2
                    )
                    # Add to history for velocity calculation
                    track.history.append(predicted_pos)
                    if len(track.history) > 30:
                        track.history.pop(0)
                else:
                    # Move to lost tracks after max_age
                    lost_track = self.tracks.pop(track_id)
                    self.lost_tracks[track_id] = lost_track
                    print(f"  Frame {self.frame_idx}: Track ID {track_id} moved to lost")

        # Clean up lost tracks that have been lost too long (3x max_age)
        # This frees up ID slots for new detections
        max_lost_age = self.max_age * 3
        for track_id in list(self.lost_tracks.keys()):
            self.lost_tracks[track_id].frames_since_update += 1
            if self.lost_tracks[track_id].frames_since_update > max_lost_age:
                del self.lost_tracks[track_id]
                print(f"  Frame {self.frame_idx}: Lost track ID {track_id} expired and removed")

        # Check proximity between tracks
        self._check_proximity(timestamp)

        # Prepare result
        result = {
            'frame_idx': self.frame_idx,
            'timestamp': timestamp,
            'num_detections': len(detections),
            'num_tracks': len(self.tracks),
            'tracks': {},
            'proximity_warning': False,
            'proximity_distance': None
        }

        for track_id, track in self.tracks.items():
            result['tracks'][track_id] = {
                'bbox': track.bbox,
                'centroid': track.centroid,
                'confidence': track.confidence,
                'velocity': track.velocity,
                'frames_since_update': track.frames_since_update
            }

        # Check if current frame has proximity warning
        if len(self.tracks) >= 2:
            track_ids = list(self.tracks.keys())
            for i in range(len(track_ids)):
                for j in range(i + 1, len(track_ids)):
                    dist = self._compute_distance(
                        self.tracks[track_ids[i]].centroid,
                        self.tracks[track_ids[j]].centroid
                    )
                    if dist < self.proximity_threshold:
                        result['proximity_warning'] = True
                        result['proximity_distance'] = dist

        self.frame_idx += 1
        return result

    def draw_tracks(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw tracking visualization on frame."""
        vis = frame.copy()

        colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

        for track_id, track_info in result['tracks'].items():
            color = colors[track_id % len(colors)]
            bbox = track_info['bbox']
            centroid = track_info['centroid']
            frames_since_update = track_info.get('frames_since_update', 0)
            is_predicted = frames_since_update > 0  # Track is using predicted position

            # Draw bbox - dashed if predicted
            x1, y1, x2, y2 = map(int, bbox)
            if is_predicted:
                # Draw dashed rectangle for predicted position
                dash_length = 10
                for i in range(x1, x2, dash_length * 2):
                    cv2.line(vis, (i, y1), (min(i + dash_length, x2), y1), color, 2)
                    cv2.line(vis, (i, y2), (min(i + dash_length, x2), y2), color, 2)
                for i in range(y1, y2, dash_length * 2):
                    cv2.line(vis, (x1, i), (x1, min(i + dash_length, y2)), color, 2)
                    cv2.line(vis, (x2, i), (x2, min(i + dash_length, y2)), color, 2)
            else:
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Draw centroid - hollow circle if predicted
            cx, cy = int(centroid[0]), int(centroid[1])
            if is_predicted:
                cv2.circle(vis, (cx, cy), 8, color, 2)  # Hollow, larger
            else:
                cv2.circle(vis, (cx, cy), 8, color, -1)  # Filled, larger

            # Draw ID label at centroid position (above the centroid marker)
            label = f"ID:{track_id}" if not is_predicted else f"ID:{track_id}(P)"
            cv2.putText(vis, label, (cx - 20, cy - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Draw track history if available
            track = self.tracks.get(track_id)
            if track and len(track.history) > 1:
                points = [(int(p[0]), int(p[1])) for p in track.history]
                for k in range(1, len(points)):
                    cv2.line(vis, points[k-1], points[k], color, 2)

        # Draw proximity warning
        if result.get('proximity_warning'):
            dist = result.get('proximity_distance', 0)
            warning = f"PROXIMITY WARNING: {dist:.1f}px"
            cv2.putText(vis, warning, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Draw line between close objects
            track_ids = list(result['tracks'].keys())
            if len(track_ids) >= 2:
                c1 = result['tracks'][track_ids[0]]['centroid']
                c2 = result['tracks'][track_ids[1]]['centroid']
                cv2.line(vis, (int(c1[0]), int(c1[1])),
                        (int(c2[0]), int(c2[1])), (0, 0, 255), 2)

        # Frame info
        info = f"Frame: {result['frame_idx']} | Tracks: {result['num_tracks']}"
        cv2.putText(vis, info, (10, vis.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis

    def get_proximity_summary(self) -> Dict:
        """Get summary of proximity events."""
        if not self.proximity_events:
            return {
                'total_events': 0,
                'events': [],
                'frames_with_proximity': []
            }

        frames = sorted(set(e.frame_idx for e in self.proximity_events))

        # Group consecutive frames into segments
        segments = []
        if frames:
            start = frames[0]
            end = frames[0]
            for f in frames[1:]:
                if f == end + 1:
                    end = f
                else:
                    segments.append((start, end))
                    start = end = f
            segments.append((start, end))

        return {
            'total_events': len(self.proximity_events),
            'unique_frames': len(frames),
            'segments': segments,
            'events': [
                {
                    'frame': int(e.frame_idx),
                    'timestamp': float(e.timestamp),
                    'distance': float(e.distance),
                    'iou': float(e.overlap_iou)
                }
                for e in self.proximity_events
            ]
        }


def track_video(video_path: str, model_path: str, output_dir: str,
               proximity_threshold: float = 100.0,
               conf_threshold: float = 0.5,
               save_video: bool = True,
               show_preview: bool = False,
               expected_object_count: int = None,
               sam3_model: str = None,
               hf_token: str = None,
               device: str = "cuda") -> Dict:
    """
    Track objects in video with SAM3 supervision for new ID verification.

    Args:
        video_path: Path to input video
        model_path: Path to YOLO model
        output_dir: Output directory
        proximity_threshold: Distance for proximity warning
        conf_threshold: Detection confidence threshold
        save_video: Save annotated video
        show_preview: Show live preview
        expected_object_count: Max number of objects (from SAM3 majority voting)
        sam3_model: SAM3 model for verification (e.g., "facebook/sam3")
        hf_token: HuggingFace token for SAM3
        device: Device for SAM3 ("cuda" or "cpu")

    Returns:
        Tracking statistics and proximity events
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path.name}")
    print(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
    print(f"Total frames: {total_frames}")
    print(f"Proximity threshold: {proximity_threshold}px")
    if expected_object_count:
        print(f"Expected object count: {expected_object_count} (SAM3 supervision)")
    if sam3_model:
        print(f"SAM3 model: {sam3_model}")
    print("-" * 50)

    # Initialize tracker with SAM3 supervision
    tracker = YOLOTracker(
        model_path=model_path,
        proximity_threshold=proximity_threshold,
        conf_threshold=conf_threshold,
        expected_object_count=expected_object_count,
        sam3_model=sam3_model,
        hf_token=hf_token,
        device=device
    )

    # Video writer
    writer = None
    if save_video:
        out_path = output_dir / f"{video_path.stem}_tracked.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx = 0
    proximity_frames = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps

            # Process frame
            result = tracker.process_frame(frame, timestamp)

            # Track proximity events
            if result['proximity_warning']:
                proximity_frames.append({
                    'frame': frame_idx,
                    'timestamp': timestamp,
                    'distance': result['proximity_distance']
                })

            # Draw visualization
            vis = tracker.draw_tracks(frame, result)

            # Save frame
            if writer:
                writer.write(vis)

            # Show preview
            if show_preview:
                cv2.imshow('YOLO Tracker', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Progress
            if frame_idx % 100 == 0:
                pct = 100 * frame_idx / total_frames
                prox_count = len(proximity_frames)
                print(f"Frame {frame_idx}/{total_frames} ({pct:.1f}%) - Proximity events: {prox_count}")

            frame_idx += 1

    finally:
        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()

    # Get summary
    summary = tracker.get_proximity_summary()
    summary['total_frames'] = frame_idx
    summary['video_path'] = str(video_path)
    summary['model_path'] = str(model_path)
    summary['proximity_threshold'] = proximity_threshold
    summary['expected_object_count'] = expected_object_count
    summary['sam3_supervision'] = sam3_model is not None
    summary['id_verification_stats'] = tracker.stats

    # Save results
    results_path = output_dir / "tracking_results.json"
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("TRACKING COMPLETE")
    print("=" * 50)
    print(f"Total frames processed: {frame_idx}")
    print(f"Proximity events detected: {summary.get('total_events', 0)}")
    print(f"Unique frames with proximity: {summary.get('unique_frames', 0)}")

    # Print ID verification stats
    stats = tracker.stats
    print(f"\nID Verification Statistics:")
    print(f"  Total unique IDs created: {stats['new_ids_created']}")
    print(f"  Track recoveries: {stats['track_recoveries']}")
    print(f"  Rejected (count limit): {stats['new_ids_rejected_count_limit']}")
    print(f"  Rejected (SAM3): {stats['new_ids_rejected_sam3']}")
    print(f"  SAM3 verifications: {stats['sam3_verifications']}")

    if summary.get('segments'):
        print(f"\nProximity segments (start-end frames):")
        for start, end in summary['segments']:
            duration = (end - start + 1) / fps
            print(f"  Frames {start}-{end} ({duration:.2f}s)")

    if save_video:
        print(f"\nOutput video: {out_path}")
    print(f"Results saved: {results_path}")

    return summary


def main():
    import os

    parser = argparse.ArgumentParser(
        description="YOLO tracker with SAM3 supervision for ID verification"
    )
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--model", required=True, help="YOLO model path")
    parser.add_argument("-o", "--output", default="tracking_output",
                       help="Output directory")
    parser.add_argument("--proximity", type=float, default=100.0,
                       help="Proximity threshold in pixels (default: 100)")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--no-video", action="store_true",
                       help="Don't save output video")
    parser.add_argument("--preview", action="store_true",
                       help="Show live preview")

    # SAM3 supervision arguments
    parser.add_argument("--expected-objects", type=int, default=None,
                       help="Expected number of objects (from SAM3 majority voting)")
    parser.add_argument("--sam3-model", default=None,
                       help="SAM3 model for verification (e.g., facebook/sam3)")
    parser.add_argument("--hf-token", default=None,
                       help="HuggingFace token for SAM3 (or set HF_TOKEN env)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                       help="Device for SAM3 (default: cuda)")

    args = parser.parse_args()

    # Get HF token from args or environment
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    track_video(
        args.video,
        args.model,
        args.output,
        proximity_threshold=args.proximity,
        conf_threshold=args.conf,
        save_video=not args.no_video,
        show_preview=args.preview,
        expected_object_count=args.expected_objects,
        sam3_model=args.sam3_model,
        hf_token=hf_token,
        device=args.device
    )


if __name__ == "__main__":
    main()
