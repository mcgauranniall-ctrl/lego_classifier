"""Test the Brickognize pipeline on sample photos."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from brickognize.pipeline import analyze_image

PHOTOS = [
    ("lego_brick.jfif", "Expected: 1x 3001 Brick 2x4"),
    ("lego_brick_2.jfif", "Expected: 1x 3001 Brick 2x4 + 1x 3003 Brick 2x2"),
    ("lego_brick_3.jfif", "Expected: 9x 3004 Brick 1x2"),
]

DOWNLOADS = Path("C:/Users/Owner/Downloads")


def check_result(filename: str, result) -> tuple[bool, str]:
    """Check if the pipeline result matches expected output."""
    grouped = {g.part_id: g.count for g in result.grouped_parts}

    if filename == "lego_brick.jfif":
        if grouped.get("3001", 0) == 1 and result.total_pieces == 1:
            return True, "PASS"
        for g in result.grouped_parts:
            if "2 x 4" in g.name and g.count == 1:
                return True, f"PASS (matched as {g.part_id})"
        return False, f"FAIL — got {grouped}"

    elif filename == "lego_brick_2.jfif":
        has_2x4 = any("2 x 4" in g.name for g in result.grouped_parts)
        has_2x2 = any("2 x 2" in g.name for g in result.grouped_parts)
        if has_2x4 and has_2x2 and result.total_pieces == 2:
            return True, "PASS"
        return False, f"FAIL — got {grouped}"

    elif filename == "lego_brick_3.jfif":
        has_1x2 = any("1 x 2" in g.name for g in result.grouped_parts)
        if has_1x2 and result.total_pieces >= 7:
            return True, f"PASS ({result.total_pieces}/9 detected)"
        return False, f"FAIL — got {grouped}"

    return False, "UNKNOWN"


def main():
    print("Brickognize Pipeline Test")
    print("YOLO detection -> Brickognize API identification")
    print()

    results_summary = []

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

        result = analyze_image(img, top_k=5)

        print(f"\nDetections: {len(result.detections)}")
        for det in result.detections:
            print(f"  {det.detection_id}: bbox={det.bbox}, "
                  f"det_conf={det.detection_confidence:.3f}")
            for j, r in enumerate(det.results[:3], 1):
                print(f"    #{j} {r.part_id} {r.name} "
                      f"score={r.score:.4f}")

        print(f"\nGrouped parts:")
        for g in result.grouped_parts:
            print(f"  {g.count}x {g.part_id} {g.name} "
                  f"(score={g.best_score:.4f})")

        passed, msg = check_result(filename, result)
        print(f"\nResult: {msg}")
        print(f"Total pieces: {result.total_pieces}, "
              f"Unique parts: {result.unique_parts}")
        print(f"Processing time: {result.processing_time_ms:.0f}ms")
        print()

        results_summary.append((filename, passed, msg))

    # Summary
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for filename, passed, msg in results_summary:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {filename}: {msg}")

    total_pass = sum(1 for _, p, _ in results_summary if p)
    print(f"\n{total_pass}/{len(results_summary)} tests passed")


if __name__ == "__main__":
    main()
