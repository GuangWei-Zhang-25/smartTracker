# SAM3 + YOLO Segmentation Training Pipeline

Automated pipeline for generating YOLO segmentation training data using Meta's SAM3 (Segment Anything Model 3) for object detection and tracking.

## Overview

This pipeline uses a **two-step tracking approach**:
1. **Step 1 (Detection)**: SAM3 Image Model detects objects on the first frame using text prompts
2. **Step 2 (Propagation)**: SAM3 Video Tracker propagates detections through the video using point prompts (centroids from Step 1)

This approach:
- Uses mask centroids (not bbox centers) for accurate point prompts
- Properly initializes tracker with shared backbone from video model
- Works within 16GB GPU memory constraints
- Exports to YOLO segmentation format with polygon annotations

## Requirements

- Python 3.10+
- CUDA-capable GPU (16GB+ VRAM recommended)
- SAM3 installed from: `git+https://github.com/facebookresearch/sam3.git`

### Dependencies
```bash
pip install torch torchvision opencv-python numpy pillow tqdm
pip install git+https://github.com/facebookresearch/sam3.git
```

## Pipeline Scripts

### 1. `sample_video_1fps.py` - Video Preprocessing
Samples a long video at 1 FPS and creates short clips suitable for SAM3 tracking.

```bash
python sample_video_1fps.py <video_path> -o <output_dir> [--clip-duration 30] [--max-clips N]
```

**Arguments:**
- `video_path`: Path to source video
- `-o, --output`: Output directory for clips
- `--clip-duration`: Seconds per clip (default: 30)
- `--max-clips`: Maximum number of clips to create

**Example:**
```bash
python sample_video_1fps.py "video.mp4" -o "clips_1fps" --clip-duration 30
```

---

### 2. `prepare_yolo_segmentation_data.py` - Full Pipeline (RECOMMENDED)
Complete pipeline that processes video clips and exports YOLO segmentation training data with visualizations.

```bash
python prepare_yolo_segmentation_data.py <clips_dir> -o <output_dir> [options]
```

**Arguments:**
- `clips_dir`: Directory containing video clips
- `-o, --output`: Output directory for YOLO dataset
- `--prompt`: Text prompt for detection (default: "mouse")
- `--expected-objects`: Expected number of objects per frame (default: 2)
- `--val-split`: Validation split ratio (default: 0.2)
- `--min-area`: Minimum mask area in pixels (default: 100)
- `--class-name`: Class name for YOLO config (default: "mouse")
- `--no-vis`: Disable saving visualizations
- `--sample-every`: Sample every Nth clip for even distribution (default: 1 = all clips)

**Example:**
```bash
# Process all clips
python prepare_yolo_segmentation_data.py "clips_1fps" -o "yolo_dataset" --prompt "mouse" --expected-objects 2

# Process every 5th clip for faster testing
python prepare_yolo_segmentation_data.py "clips_1fps" -o "yolo_dataset" --prompt "mouse" --sample-every 5
```

**Output Structure:**
```
yolo_dataset/
├── dataset.yaml          # YOLO configuration file
├── images/
│   ├── train/           # Training images (.jpg)
│   └── val/             # Validation images (.jpg)
├── labels/
│   ├── train/           # Training labels (.txt)
│   └── val/             # Validation labels (.txt)
└── visualizations/
    ├── train/           # Training visualizations with masks
    └── val/             # Validation visualizations with masks
```

---

### 3. `batch_track_clips.py` - Batch Video Tracking
Processes multiple video clips and outputs annotated tracking videos (AVI format).

```bash
python batch_track_clips.py <clips_dir> -o <output_dir> [--prompt "mouse"] [--expected-objects 2]
```

**Example:**
```bash
python batch_track_clips.py "clips_1fps" -o "tracked_videos" --prompt "mouse" --expected-objects 2
```

---

### 4. `step1_detect_first_frame.py` - Detection Only
Standalone script for detecting objects on the first frame of a video.

```bash
python step1_detect_first_frame.py <video_path> -o <output.json> [--prompt "mouse"]
```

**Output:** JSON file with detection centroids and metadata.

---

### 5. `step2_propagate_video_sam2api.py` - Propagation Only
Standalone script for propagating detections through a video using SAM3 Tracker API.

```bash
python step2_propagate_video_sam2api.py <detections.json> -o <output_video.mp4>
```

## YOLO Label Format

Labels are in YOLO segmentation format (polygon coordinates):
```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

Where:
- `class_id`: 0-indexed class (e.g., 0 for "mouse")
- `x1 y1 ... xn yn`: Normalized polygon coordinates [0, 1]

Example label file:
```
0 0.245 0.312 0.256 0.298 0.271 0.305 0.283 0.318 0.278 0.335 0.261 0.340 0.247 0.328
0 0.612 0.445 0.628 0.432 0.645 0.441 0.651 0.458 0.642 0.471 0.625 0.468 0.615 0.456
```

## Technical Notes

### SAM3 Model Building Pattern
The tracker requires proper backbone sharing with the detector:

```python
from sam3.model_builder import build_sam3_video_model

sam3_model = build_sam3_video_model()
predictor = sam3_model.tracker
predictor.backbone = sam3_model.detector.backbone  # Critical!
predictor = predictor.cuda()
predictor.eval()
```

### Sam3Processor API
The processor returns a dictionary, not an object with attributes:

```python
from sam3.model.sam3_image_processor import Sam3Processor

state = processor.set_image(pil_image)
state = processor.set_text_prompt(prompt, state)

# Correct:
masks = state.get('masks', None)
scores = state.get('scores', None)

# Wrong:
# masks = state.masks  # This will fail!
```

### Mask-to-Polygon Conversion
Uses OpenCV contours with polygon approximation:

```python
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
epsilon = 0.002 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)
```

### Memory Management
GPU memory is cleared between clips to prevent OOM errors:

```python
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
```

## Typical Workflow

```bash
# 1. Sample long video into 30-second clips at 1 FPS
python sample_video_1fps.py "experiment_video.mp4" -o "clips_1fps" --clip-duration 30

# 2. Generate YOLO training data (sample every 5th clip for 149 clips)
python prepare_yolo_segmentation_data.py "clips_1fps" -o "yolo_dataset" \
    --prompt "mouse" \
    --expected-objects 2 \
    --val-split 0.2 \
    --sample-every 5

# 3. Train YOLO model
yolo segment train data=yolo_dataset/dataset.yaml model=yolov8n-seg.pt epochs=100
```

## Troubleshooting

### "No detections found"
- Try different prompts: "mouse", "black mouse", "laboratory mouse"
- Lower confidence threshold in processor
- Check if objects are visible in first frame

### Out of Memory
- Reduce clip duration (fewer frames per clip)
- Process fewer clips at once
- Clear GPU cache between clips

### Slow Propagation
- Normal for complex scenes with occlusions
- GPU memory pressure can slow down processing
- Consider using 1 FPS sampling to reduce frame count

## License

This pipeline uses Meta's SAM3 model. See SAM3 license for usage terms.
