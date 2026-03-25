r"""
Identify LEGO pieces from an image file.

Usage:
    python brickognize/identify.py path/to/photo.jpg
    python brickognize/identify.py photo1.jpg photo2.png photo3.jfif
    python brickognize/identify.py C:\Users\Owner\Downloads\lego_brick.jfif
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from brickognize.pipeline import analyze_image


def run(image_path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  {image_path.name}")
    print(f"{'='*60}")

    img = Image.open(image_path).convert("RGB")
    print(f"Image size: {img.size[0]}x{img.size[1]}")
    print("Analyzing...")

    result = analyze_image(img, top_k=5)

    if not result.grouped_parts:
        print("\nNo LEGO pieces identified.")
        return

    print(f"\nFound {result.total_pieces} piece(s), {result.unique_parts} unique part(s):\n")
    for g in result.grouped_parts:
        print(f"  {g.count}x  {g.name}  (ID: {g.part_id}, confidence: {g.best_score:.0%})")
        print(f"       BrickLink: {g.bricklink_url}")

    print(f"\nProcessing time: {result.processing_time_ms / 1000:.1f}s")

    # Show detailed detections
    print(f"\n--- Detection details ---")
    for det in result.detections:
        top = det.results[0] if det.results else None
        if top:
            print(f"  {det.detection_id}: {top.part_id} {top.name} "
                  f"(score={top.score:.4f}, bbox={det.bbox})")
        else:
            print(f"  {det.detection_id}: no match (bbox={det.bbox})")


def main():
    if len(sys.argv) < 2:
        print("Usage: python brickognize/identify.py <image> [image2] [image3] ...")
        print("\nExamples:")
        print("  python brickognize/identify.py C:\\Users\\Owner\\Downloads\\lego_brick.jfif")
        print("  python brickognize/identify.py photo1.jpg photo2.png")
        sys.exit(1)

    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"\nFile not found: {path}")
            continue
        run(path)

    print()


if __name__ == "__main__":
    main()
