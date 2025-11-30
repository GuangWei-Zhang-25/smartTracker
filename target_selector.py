"""
Target Selector - Human Intervention UI
========================================
Interactive tool to select which mouse to track from multi-object frames.
Allows human to define the target object for consistent tracking.

Features:
- Display frames with multiple detections
- Click to select target object
- Propagate selection using spatial/appearance features
- Export filtered single-target annotations
"""

import cv2
import numpy as np
from pathlib import Path
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import pickle


@dataclass
class Detection:
    """Single detection from YOLO format."""
    class_id: int
    polygon: List[Tuple[float, float]]
    bbox: Tuple[float, float, float, float] = None  # x_center, y_center, w, h
    centroid: Tuple[float, float] = None
    area: float = 0.0

    def __post_init__(self):
        if self.polygon:
            # Compute centroid
            xs = [p[0] for p in self.polygon]
            ys = [p[1] for p in self.polygon]
            self.centroid = (sum(xs) / len(xs), sum(ys) / len(ys))

            # Compute approximate area using shoelace formula
            n = len(self.polygon)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += self.polygon[i][0] * self.polygon[j][1]
                area -= self.polygon[j][0] * self.polygon[i][1]
            self.area = abs(area) / 2.0

            # Compute bbox
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            self.bbox = (
                (min_x + max_x) / 2,
                (min_y + max_y) / 2,
                max_x - min_x,
                max_y - min_y
            )


@dataclass
class FrameAnnotation:
    """Annotations for a single frame."""
    image_path: str
    label_path: str
    timestamp: float
    detections: List[Detection] = field(default_factory=list)
    selected_idx: int = -1  # Index of selected target, -1 = none


def parse_yolo_label(label_path: str) -> List[Detection]:
    """Parse YOLO segmentation format label file."""
    detections = []

    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 7:
                    continue

                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                polygon = []
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        polygon.append((coords[i], coords[i + 1]))

                if len(polygon) >= 3:
                    detections.append(Detection(class_id=class_id, polygon=polygon))
    except Exception as e:
        print(f"Error parsing {label_path}: {e}")

    return detections


def extract_timestamp(filename: str) -> float:
    """Extract timestamp from filename like frame_00001234_t41.23.jpg"""
    try:
        name = Path(filename).stem
        if '_t' in name:
            t_part = name.split('_t')[-1]
            return float(t_part)
    except:
        pass
    return 0.0


def draw_detections(image: np.ndarray, detections: List[Detection],
                   selected_idx: int = -1, highlight_hover: int = -1) -> np.ndarray:
    """Draw detections with selection highlighting."""
    h, w = image.shape[:2]
    result = image.copy()

    colors = [
        (100, 100, 100),  # Gray for unselected
        (0, 255, 0),      # Green for selected
        (0, 255, 255),    # Yellow for hover
    ]

    for i, det in enumerate(detections):
        # Determine color
        if i == selected_idx:
            color = colors[1]  # Green
            alpha = 0.5
            thickness = 3
        elif i == highlight_hover:
            color = colors[2]  # Yellow
            alpha = 0.4
            thickness = 2
        else:
            color = colors[0]  # Gray
            alpha = 0.25
            thickness = 1

        # Convert normalized coords to pixels
        pts = np.array([(int(p[0] * w), int(p[1] * h)) for p in det.polygon], dtype=np.int32)

        # Draw filled polygon with transparency
        overlay = result.copy()
        cv2.fillPoly(overlay, [pts], color)
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

        # Draw outline
        cv2.polylines(result, [pts], True, color, thickness)

        # Draw index number at centroid
        cx, cy = int(det.centroid[0] * w), int(det.centroid[1] * h)
        label = f"#{i+1}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(result, (cx - tw//2 - 5, cy - th//2 - 5),
                     (cx + tw//2 + 5, cy + th//2 + 5), color, -1)
        cv2.putText(result, label, (cx - tw//2, cy + th//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return result


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Check if point is inside polygon using ray casting."""
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def find_detection_at_point(detections: List[Detection],
                           x_norm: float, y_norm: float) -> int:
    """Find which detection contains the given point. Returns index or -1."""
    for i, det in enumerate(detections):
        if point_in_polygon((x_norm, y_norm), det.polygon):
            return i
    return -1


class TargetSelector:
    """Interactive target selection UI."""

    def __init__(self, images_dir: str, labels_dir: str):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.frames: List[FrameAnnotation] = []
        self.current_idx = 0
        self.hover_detection = -1
        self.window_name = "Target Selector - Click to select target mouse"
        self.target_features = None  # For propagation

    def load_frames(self):
        """Load all frame-label pairs."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

        for img_path in sorted(self.images_dir.rglob('*')):
            if img_path.suffix.lower() not in image_extensions:
                continue

            # Find matching label
            relative = img_path.relative_to(self.images_dir)
            label_path = self.labels_dir / relative.with_suffix('.txt')

            if not label_path.exists():
                label_path = self.labels_dir / (img_path.stem + '.txt')

            if not label_path.exists():
                continue

            timestamp = extract_timestamp(img_path.name)
            detections = parse_yolo_label(str(label_path))

            if detections:  # Only include frames with detections
                self.frames.append(FrameAnnotation(
                    image_path=str(img_path),
                    label_path=str(label_path),
                    timestamp=timestamp,
                    detections=detections
                ))

        # Sort by timestamp
        self.frames.sort(key=lambda f: f.timestamp)
        print(f"Loaded {len(self.frames)} frames with detections")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if not self.frames:
            return

        frame = self.frames[self.current_idx]
        img = cv2.imread(frame.image_path)
        if img is None:
            return

        h, w = img.shape[:2]
        x_norm = x / w
        y_norm = y / h

        if event == cv2.EVENT_MOUSEMOVE:
            self.hover_detection = find_detection_at_point(frame.detections, x_norm, y_norm)

        elif event == cv2.EVENT_LBUTTONDOWN:
            clicked_idx = find_detection_at_point(frame.detections, x_norm, y_norm)
            if clicked_idx >= 0:
                frame.selected_idx = clicked_idx
                # Store target features for propagation
                self.target_features = self._extract_features(frame.detections[clicked_idx])
                print(f"Selected detection #{clicked_idx + 1} as target")

    def _extract_features(self, detection: Detection) -> Dict:
        """Extract features for target matching."""
        return {
            'centroid': detection.centroid,
            'area': detection.area,
            'bbox': detection.bbox,
        }

    def _match_target(self, detections: List[Detection], prev_features: Dict) -> int:
        """Find detection that best matches previous target features."""
        if not detections or not prev_features:
            return -1

        best_idx = -1
        best_score = float('inf')

        prev_centroid = prev_features['centroid']
        prev_area = prev_features['area']

        for i, det in enumerate(detections):
            # Distance score (lower is better)
            dx = det.centroid[0] - prev_centroid[0]
            dy = det.centroid[1] - prev_centroid[1]
            dist = (dx * dx + dy * dy) ** 0.5

            # Area similarity (lower is better)
            area_diff = abs(det.area - prev_area) / max(prev_area, 0.001)

            # Combined score
            score = dist + area_diff * 0.5

            if score < best_score:
                best_score = score
                best_idx = i

        return best_idx

    def propagate_selection(self, direction: int = 1):
        """Propagate target selection to adjacent frames."""
        if not self.target_features:
            print("No target selected - click a detection first")
            return

        start_idx = self.current_idx
        propagated = 0

        # Propagate in specified direction
        idx = start_idx + direction
        prev_features = self.target_features

        while 0 <= idx < len(self.frames):
            frame = self.frames[idx]

            if len(frame.detections) == 1:
                # Only one detection - auto-select
                frame.selected_idx = 0
                prev_features = self._extract_features(frame.detections[0])
            else:
                # Match to previous target
                match_idx = self._match_target(frame.detections, prev_features)
                if match_idx >= 0:
                    frame.selected_idx = match_idx
                    prev_features = self._extract_features(frame.detections[match_idx])
                else:
                    break  # Can't match - stop propagation

            propagated += 1
            idx += direction

        print(f"Propagated selection to {propagated} frames {'forward' if direction > 0 else 'backward'}")

    def run(self):
        """Run interactive selection UI."""
        if not self.frames:
            print("No frames loaded!")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n=== Target Selector Controls ===")
        print("Click: Select target detection")
        print("A/D or Left/Right: Navigate frames")
        print("W: Jump forward 10 frames")
        print("S: Jump backward 10 frames")
        print("P: Propagate selection forward")
        print("O: Propagate selection backward")
        print("B: Propagate both directions")
        print("R: Reset current selection")
        print("Space: Auto-select (single detection only)")
        print("Enter: Save and exit")
        print("Q/Esc: Exit without saving")
        print("================================\n")

        while True:
            frame = self.frames[self.current_idx]
            img = cv2.imread(frame.image_path)

            if img is None:
                print(f"Cannot load image: {frame.image_path}")
                self.current_idx = (self.current_idx + 1) % len(self.frames)
                continue

            # Draw detections
            display = draw_detections(img, frame.detections,
                                     frame.selected_idx, self.hover_detection)

            # Add info bar
            h, w = display.shape[:2]
            info_bar = np.zeros((60, w, 3), dtype=np.uint8)

            # Frame info
            info1 = f"Frame {self.current_idx + 1}/{len(self.frames)} | t={frame.timestamp:.2f}s"
            cv2.putText(info_bar, info1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Selection status
            n_det = len(frame.detections)
            if frame.selected_idx >= 0:
                status = f"TARGET: #{frame.selected_idx + 1} of {n_det} | {Path(frame.image_path).name}"
                color = (0, 255, 0)
            else:
                status = f"NO TARGET SELECTED | {n_det} detections | {Path(frame.image_path).name}"
                color = (0, 0, 255)
            cv2.putText(info_bar, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Selection progress
            selected_count = sum(1 for f in self.frames if f.selected_idx >= 0)
            progress = f"Progress: {selected_count}/{len(self.frames)} ({100*selected_count/len(self.frames):.1f}%)"
            cv2.putText(info_bar, progress, (w - 250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Combine
            display = np.vstack([info_bar, display])

            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == 27:  # Q or Esc
                print("Exiting without saving...")
                break

            elif key == 13:  # Enter
                print("Saving selections...")
                self.save_selections()
                break

            elif key == ord('a') or key == 81:  # A or Left
                self.current_idx = max(0, self.current_idx - 1)

            elif key == ord('d') or key == 83:  # D or Right
                self.current_idx = min(len(self.frames) - 1, self.current_idx + 1)

            elif key == ord('w'):  # Jump forward
                self.current_idx = min(len(self.frames) - 1, self.current_idx + 10)

            elif key == ord('s'):  # Jump backward
                self.current_idx = max(0, self.current_idx - 10)

            elif key == ord('p'):  # Propagate forward
                self.propagate_selection(direction=1)

            elif key == ord('o'):  # Propagate backward
                self.propagate_selection(direction=-1)

            elif key == ord('b'):  # Propagate both
                self.propagate_selection(direction=-1)
                self.propagate_selection(direction=1)

            elif key == ord('r'):  # Reset
                frame.selected_idx = -1
                print("Reset current frame selection")

            elif key == ord(' '):  # Space - auto-select single
                if len(frame.detections) == 1:
                    frame.selected_idx = 0
                    self.target_features = self._extract_features(frame.detections[0])
                    print("Auto-selected single detection")
                else:
                    print(f"Cannot auto-select: {len(frame.detections)} detections")

        cv2.destroyAllWindows()

    def save_selections(self):
        """Save selection data and export filtered labels."""
        output_dir = self.labels_dir.parent / "labels_single_target"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save selection metadata
        selections = []
        exported = 0

        for frame in self.frames:
            selection_data = {
                'image_path': frame.image_path,
                'label_path': frame.label_path,
                'timestamp': frame.timestamp,
                'num_detections': len(frame.detections),
                'selected_idx': frame.selected_idx,
            }
            selections.append(selection_data)

            # Export single-target label
            if frame.selected_idx >= 0:
                det = frame.detections[frame.selected_idx]

                # Create YOLO format line
                coords = " ".join(f"{p[0]:.6f} {p[1]:.6f}" for p in det.polygon)
                line = f"{det.class_id} {coords}\n"

                # Save to output
                out_path = output_dir / Path(frame.label_path).name
                with open(out_path, 'w') as f:
                    f.write(line)
                exported += 1

        # Save metadata
        meta_path = output_dir / "selection_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'total_frames': len(self.frames),
                'selected_frames': exported,
                'selections': selections
            }, f, indent=2)

        print(f"\nExported {exported} single-target labels to: {output_dir}")
        print(f"Metadata saved to: {meta_path}")

        return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Interactive target selection for multi-object tracking"
    )
    parser.add_argument("images_dir", help="Directory containing images")
    parser.add_argument("labels_dir", help="Directory containing YOLO labels")
    parser.add_argument("--auto-single", action="store_true",
                       help="Auto-select frames with single detection")

    args = parser.parse_args()

    selector = TargetSelector(args.images_dir, args.labels_dir)
    selector.load_frames()

    if args.auto_single:
        # Pre-select all single-detection frames
        auto_count = 0
        for frame in selector.frames:
            if len(frame.detections) == 1:
                frame.selected_idx = 0
                auto_count += 1
        print(f"Auto-selected {auto_count} single-detection frames")

    selector.run()


if __name__ == "__main__":
    main()
