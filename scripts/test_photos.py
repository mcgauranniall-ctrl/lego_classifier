"""Quick test script: run the pipeline on the 3 sample photos and print results."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from ml.pipeline import LegoPipeline

PHOTOS = [
    ("lego_brick.jfif", "Expected: 1x Brick 2x4"),
    ("lego_brick_2.jfif", "Expected: 1x Brick 2x4 + 1x Brick 2x2"),
    ("lego_brick_3.jfif", "Expected: 9x Brick 1x2"),
]

DOWNLOADS = Path("C:/Users/Owner/Downloads")


def main():
    print("Loading pipeline...")

    # Auto-detect fine-tuned YOLO model
    yolo_model = "yolov8m.pt"
    lego_model = PROJECT_ROOT / "lego_yolov8.pt"
    if lego_model.exists():
        yolo_model = str(lego_model)
        print(f"Using fine-tuned YOLO: {lego_model.name}")
    else:
        print(f"Using default COCO YOLO: {yolo_model}")

    pipeline = LegoPipeline(
        parts_json_path=PROJECT_ROOT / "data" / "parts.json",
        embeddings_npy_path=PROJECT_ROOT / "data" / "embeddings.npy",
        embeddings_index_path=PROJECT_ROOT / "data" / "embeddings_index.json",
        yolo_model_path=yolo_model,
    )
    print(f"Matcher ready: {pipeline.matcher.is_ready}")
    print(f"Parts indexed: {pipeline.matcher.embeddings.shape[0]}")
    print()

    for filename, expected in PHOTOS:
        path = DOWNLOADS / filename
        if not path.exists():
            print(f"SKIP: {filename} not found")
            continue

        print(f"{'='*60}")
        print(f"Photo: {filename}")
        print(f"{expected}")
        print(f"{'='*60}")

        img = Image.open(path).convert("RGB")
        print(f"Image size: {img.size}")

        result = pipeline.analyze_image(img, match_top_k=5)

        print(f"\nDetections: {len(result.detections)}")
        for det in result.detections:
            print(f"  {det.detection_id}: bbox={det.bbox}, "
                  f"det_conf={det.detection_confidence:.3f}")
            for m in det.matches[:3]:
                print(f"    #{m.rank} {m.part_id} {m.name} "
                      f"sim={m.similarity:.4f} ({m.tier})")

        print(f"\nGrouped parts:")
        for g in result.grouped_parts:
            print(f"  {g.count}x {g.part_id} {g.name} "
                  f"(sim={g.best_similarity:.4f}, {g.tier})")

        print(f"\nTotal pieces: {result.total_pieces}, "
              f"Unique parts: {result.unique_parts}")
        print(f"Processing time: {result.processing_time_ms:.0f}ms")
        print()


if __name__ == "__main__":
    main()
