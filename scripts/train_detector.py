"""
Fine-tune YOLOv8 on a LEGO brick detection dataset.

This script downloads a labeled LEGO dataset from Roboflow Universe
and trains YOLOv8 to detect individual LEGO pieces (bounding boxes).

Prerequisites:
    pip install roboflow

Usage:
    # Step 1: Get a free Roboflow API key at https://app.roboflow.com
    # Step 2: Run training
    python scripts/train_detector.py --api-key YOUR_ROBOFLOW_API_KEY

    # Optional: specify epochs and model size
    python scripts/train_detector.py --api-key YOUR_KEY --epochs 50 --model yolov8s.pt

    # The trained model will be saved to runs/detect/lego_detector/weights/best.pt
    # Copy it to the project root and update server.py or pipeline to use it.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def download_dataset(api_key: str, dataset_dir: Path) -> Path:
    """Download a LEGO detection dataset from Roboflow."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow...")
        import subprocess
        subprocess.check_call(["pip", "install", "roboflow"])
        from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)

    # "lego-detection" by Roboflow Universe — a well-labeled LEGO brick dataset
    # with bounding boxes around individual pieces. ~2,000 images.
    # Alternative datasets if this one doesn't work:
    #   - "lego-bricks-kwcvb" (workspace: lego-bricks-kwcvb)
    #   - "lego-piece-detection" (various workspaces)
    print("Downloading LEGO detection dataset from Roboflow...")
    print("(This may take a few minutes on first run)")

    project = rf.workspace("roboflow-100").project("lego-bricks")
    version = project.version(1)
    dataset = version.download("yolov8", location=str(dataset_dir))

    return Path(dataset.location)


def train(
    dataset_path: Path,
    model_path: str = "yolov8m.pt",
    epochs: int = 30,
    image_size: int = 640,
    batch_size: int = 16,
    output_name: str = "lego_detector",
) -> Path:
    """Fine-tune YOLOv8 on the downloaded dataset."""
    from ultralytics import YOLO

    model = YOLO(model_path)

    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {data_yaml}. "
            "Check the dataset download."
        )

    print(f"\nStarting training:")
    print(f"  Model:      {model_path}")
    print(f"  Dataset:    {data_yaml}")
    print(f"  Epochs:     {epochs}")
    print(f"  Image size: {image_size}")
    print(f"  Batch size: {batch_size}")
    print()

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        name=output_name,
        patience=10,       # early stopping if no improvement for 10 epochs
        save=True,
        plots=True,
        verbose=True,
    )

    # Find the best weights
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = Path(results.save_dir) / "weights" / "last.pt"

    return best_weights


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on a LEGO detection dataset"
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="Roboflow API key (free at https://app.roboflow.com)",
    )
    parser.add_argument(
        "--model",
        default="yolov8m.pt",
        help="Base YOLO model to fine-tune (default: yolov8m.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (reduce to 8 or 4 if you run out of GPU memory)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Training image size in pixels (default: 640)",
    )
    args = parser.parse_args()

    # Download dataset
    dataset_dir = PROJECT_ROOT / "data" / "lego_yolo_dataset"
    dataset_path = download_dataset(args.api_key, dataset_dir)

    # Train
    best_weights = train(
        dataset_path=dataset_path,
        model_path=args.model,
        epochs=args.epochs,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    # Copy best weights to project root for easy use
    output_path = PROJECT_ROOT / "lego_yolov8.pt"
    shutil.copy2(best_weights, output_path)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best weights saved to: {output_path}")
    print(f"\nTo use the fine-tuned model, start the server with:")
    print(f"  set YOLO_MODEL=lego_yolov8.pt")
    print(f"  python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
