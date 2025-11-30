# SmartTracker

An advanced computer vision system for single-target object tracking with robust occlusion handling. Combines YOLO's real-time detection with SAM3's memory-based video tracking to maintain consistent object IDs even when targets are temporarily occluded.

## Features

- **Hybrid YOLO + SAM3 Tracking**: Combines YOLO's speed with SAM3's temporal memory for robust tracking
- **Occlusion Handling**: State machine-based tracking with automatic recovery from occlusions
- **Smart Frame Sampling**: Efficient frame extraction using scene change detection
- **Auto-Annotation**: SAM3-powered automatic segmentation with text prompts
- **YOLO Training Pipeline**: Complete workflow for training custom segmentation models
- **Interactive Target Selection**: UI for selecting specific objects to track
- **Visualization Tools**: Annotation validation and montage generation

## Project Structure

```
smartTracker/
├── unified_tracker.py         # Main hybrid tracker (YOLO + SAM3)
├── yolo_tracker.py            # Multi-object YOLO tracker
├── sam3_annotator.py          # SAM3 auto-annotation
├── smart_frame_sampler.py     # Intelligent frame extraction
├── batch_pipeline.py          # Complete workflow orchestration
├── train_yolo.py              # YOLO training script
├── prepare_yolo_dataset.py    # Dataset preparation (train/val split)
├── filter_labels.py           # Label filtering by object count
├── target_selector.py         # Interactive target selection UI
├── visualize_annotations.py   # Annotation visualization
├── yolo11n.pt                 # Pre-trained YOLO 11 Nano weights
├── yolov8n-seg.pt             # Pre-trained YOLO 8 Nano Seg weights
├── run_training.bat           # Windows training batch script
└── setup_sam3.bat             # Windows SAM3 setup script
```

## Requirements

- Python 3.10+
- PyTorch (with CUDA for GPU acceleration)
- ultralytics (YOLO)
- transformers (HuggingFace)
- OpenCV
- NumPy
- Pillow

## Installation

### Option 1: Using Setup Script (Windows)

```batch
setup_sam3.bat
```

Then authenticate with HuggingFace:
```bash
huggingface-cli login
```

### Option 2: Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision torchaudio
pip install ultralytics opencv-python numpy pillow
pip install transformers huggingface_hub
```

## Usage

### 1. Extract Frames from Video

```bash
python smart_frame_sampler.py video.mp4 -o output_frames
```

Options:
- `--sensitivity`: low, medium, high, very_high (default: medium)
- `--min-interval`: Minimum frames between samples
- `--max-interval`: Maximum frames between samples

### 2. Auto-Annotate with SAM3

```bash
python sam3_annotator.py frames_dir --prompt "mouse"
```

### 3. Filter Labels (Optional)

Filter annotations to keep only frames with a specific object count:

```bash
python filter_labels.py images_dir labels_dir -o filtered_output --num-objects 1
```

### 4. Prepare YOLO Dataset

```bash
python prepare_yolo_dataset.py images_dir labels_dir -o dataset_output
```

### 5. Train YOLO Model

```bash
python train_yolo.py dataset_output/dataset.yaml --model n --epochs 50 --batch 8
```

Model sizes: `n` (nano), `s` (small), `m` (medium), `l` (large), `x` (xlarge)

### 6. Run Tracking

**Unified Tracker** (single target with occlusion handling):
```bash
python unified_tracker.py video.mp4 -m yolo11n.pt --output results.json
```

**YOLO Multi-Object Tracker**:
```bash
python yolo_tracker.py video.mp4 -m yolov8n-seg.pt --output results.json
```

### 7. Complete Batch Pipeline

Run the entire workflow in one command:

```bash
python batch_pipeline.py video.mp4 --output-dir project_output --text-prompt "mouse"
```

### 8. Visualize Annotations

```bash
python visualize_annotations.py images_dir labels_dir -o visualizations
```

### 9. Interactive Target Selection

```bash
python target_selector.py images_dir labels_dir -o selected_targets
```

## Tracker States

The unified tracker uses a state machine with four states:

| State | Description |
|-------|-------------|
| TRACKING | Normal YOLO tracking - target detected |
| OCCLUSION | SAM3 handles temporal tracking during occlusion |
| RECOVERY | Attempting to re-acquire target after loss |
| LOST | Target completely lost |

## Pre-trained Models

- `yolo11n.pt` - YOLOv11 Nano weights
- `yolov8n-seg.pt` - YOLOv8 Nano Segmentation weights

## Use Cases

- Behavioral research (animal tracking)
- Single-target object tracking with occlusions
- Dataset preparation for custom object detection
- Video annotation and segmentation

## License

MIT License
