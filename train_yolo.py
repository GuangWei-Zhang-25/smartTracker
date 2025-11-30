"""
YOLO Segmentation Training Script
=================================
Trains YOLOv8 segmentation model on prepared dataset.
"""

import argparse
from pathlib import Path
import json
import os

# Check for ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Run: pip install ultralytics")


def train_yolo_seg(dataset_yaml: str, model_size: str = "n", epochs: int = 100,
                   imgsz: int = 640, batch: int = 16, device: str = "0",
                   project: str = None, name: str = None, resume: bool = False,
                   workers: int = 0, mosaic: float = 0.0):
    """
    Train YOLOv8 segmentation model.

    Args:
        dataset_yaml: Path to dataset.yaml
        model_size: Model size - n, s, m, l, x (default: n for nano)
        epochs: Number of training epochs
        imgsz: Image size
        batch: Batch size (adjust based on GPU memory)
        device: Device to train on ("0" for GPU, "cpu" for CPU)
        project: Project directory for outputs
        name: Experiment name
        resume: Resume from last checkpoint
        workers: Number of data loader workers (0 for main process only)
        mosaic: Mosaic augmentation (0.0 to disable, 1.0 for full)

    Returns:
        Path to best model weights
    """
    if not YOLO_AVAILABLE:
        raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

    dataset_yaml = Path(dataset_yaml)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_yaml}")

    # Default project/name
    if project is None:
        project = str(dataset_yaml.parent / "runs")
    if name is None:
        name = f"yolo{model_size}_seg"

    # Load base model
    model_name = f"yolov8{model_size}-seg.pt"
    print(f"Loading base model: {model_name}")
    model = YOLO(model_name)

    # Train
    print(f"\nStarting training...")
    print(f"Dataset: {dataset_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    print("-" * 50)

    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        resume=resume,
        # Augmentation settings for small dataset
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=mosaic,
        mixup=0.0 if mosaic == 0.0 else 0.1,
        workers=workers,
        # Save settings
        save=True,
        save_period=10,
        patience=20,  # Early stopping
        # Validation
        val=True,
    )

    # Find best weights
    best_path = Path(project) / name / "weights" / "best.pt"
    if best_path.exists():
        print(f"\nTraining complete!")
        print(f"Best model: {best_path}")
        return str(best_path)
    else:
        print("Training complete but best.pt not found")
        return None


def validate_model(model_path: str, dataset_yaml: str):
    """Run validation on trained model."""
    if not YOLO_AVAILABLE:
        raise RuntimeError("ultralytics not installed")

    model = YOLO(model_path)
    results = model.val(data=dataset_yaml)

    print("\nValidation Results:")
    print(f"  mAP50: {results.seg.map50:.4f}")
    print(f"  mAP50-95: {results.seg.map:.4f}")

    return results


def export_model(model_path: str, format: str = "onnx"):
    """Export model to different format."""
    if not YOLO_AVAILABLE:
        raise RuntimeError("ultralytics not installed")

    model = YOLO(model_path)
    export_path = model.export(format=format)
    print(f"Exported to: {export_path}")
    return export_path


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 segmentation model"
    )
    parser.add_argument("dataset", help="Path to dataset.yaml")
    parser.add_argument("--model", default="n",
                       choices=["n", "s", "m", "l", "x"],
                       help="Model size (default: n)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Training epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size (default: 16)")
    parser.add_argument("--device", default="0",
                       help="Device: 0, 1, cpu (default: 0)")
    parser.add_argument("--project", help="Project directory")
    parser.add_argument("--name", help="Experiment name")
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from last checkpoint")
    parser.add_argument("--workers", type=int, default=0,
                       help="Data loader workers (default: 0)")
    parser.add_argument("--mosaic", type=float, default=0.0,
                       help="Mosaic augmentation (default: 0.0 disabled)")
    parser.add_argument("--validate", action="store_true",
                       help="Run validation only (requires --weights)")
    parser.add_argument("--weights", help="Model weights for validation")
    parser.add_argument("--export", help="Export format (onnx, torchscript, etc)")

    args = parser.parse_args()

    if args.validate:
        if not args.weights:
            print("--weights required for validation")
            return
        validate_model(args.weights, args.dataset)

    elif args.export:
        if not args.weights:
            print("--weights required for export")
            return
        export_model(args.weights, args.export)

    else:
        train_yolo_seg(
            args.dataset,
            model_size=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            resume=args.resume,
            workers=args.workers,
            mosaic=args.mosaic
        )


if __name__ == "__main__":
    main()
