"""
Expand the LEGO parts catalog from 35 → ~1000 parts using Rebrickable bulk data.

Downloads the Rebrickable parts CSV (free, no API key needed), filters to the
most common brick/plate/tile/slope parts, downloads images from the CDN,
and updates data/parts.json.

After running this, run build_embeddings.py to generate CLIP vectors.

Usage:
    python scripts/expand_catalog.py
    python scripts/expand_catalog.py --limit 500   # fewer parts
    python scripts/expand_catalog.py --limit 2000  # more parts
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
PARTS_JSON = DATA_DIR / "parts.json"

# Rebrickable bulk download (free, no API key)
PARTS_CSV_URL = "https://cdn.rebrickable.com/media/downloads/parts.csv.gz"
PART_CATS_CSV_URL = "https://cdn.rebrickable.com/media/downloads/part_categories.csv.gz"

# Rebrickable CDN for LDraw renders (color 0 = neutral)
IMAGE_URL_TEMPLATE = "https://cdn.rebrickable.com/media/parts/ldraw/0/{part_num}.png"

DOWNLOAD_DELAY = 0.3  # seconds between image downloads


# ── Category mapping ─────────────────────────────────────────────────

# Rebrickable category IDs → our simplified categories
# Full list at https://rebrickable.com/api/v3/lego/part_categories/
CATEGORY_MAP: dict[int, str] = {}  # populated from CSV

# Categories we want (by name substring match)
WANTED_CATEGORIES = [
    "brick", "plate", "tile", "slope", "round", "technic",
    "wedge", "arch", "cone", "cylinder", "panel", "hinge",
    "bracket", "angular", "corner",
]

# Categories to skip entirely
SKIP_CATEGORIES = [
    "minifig", "sticker", "duplo", "string", "cloth", "rubber",
    "electric", "pneumatic", "magnet", "spring", "animal",
    "plant", "food", "container", "vehicle", "boat", "aircraft",
    "train", "flag", "baseplate", "znap", "galidor", "scala",
    "belville", "fabuland", "homemaker",
]

# Our simplified category buckets
def classify_category(cat_name: str) -> str | None:
    """Map Rebrickable category name → our simplified category."""
    low = cat_name.lower()

    # Skip unwanted categories
    for skip in SKIP_CATEGORIES:
        if skip in low:
            return None

    if "brick" in low:
        return "Brick"
    if "plate" in low:
        return "Plate"
    if "tile" in low:
        return "Tile"
    if "slope" in low:
        return "Slope"
    if any(w in low for w in ("round", "cone", "cylinder")):
        return "Round"
    if "technic" in low:
        return "Technic"
    if any(w in low for w in ("wedge", "arch", "panel", "hinge",
                               "bracket", "angular", "corner", "wing")):
        return "Special"

    # Check if it's in our wanted list at all
    for want in WANTED_CATEGORIES:
        if want in low:
            return "Special"

    return None  # skip this category


# ── Dimension parsing ─────────────────────────────────────────────────

def parse_dimensions(name: str) -> tuple[int | None, int | None, str | None]:
    """
    Extract stud dimensions from part name.
    e.g. "Brick 2 x 4" → (2, 4, "2 x 4")
         "Plate 1 x 6"  → (1, 6, "1 x 6")
    """
    m = re.search(r"(\d+)\s*x\s*(\d+)", name)
    if m:
        w, l = int(m.group(1)), int(m.group(2))
        return w, l, f"{w} x {l}"
    return None, None, None


def height_plates_for_category(category: str) -> int:
    """Default height in plate units based on category."""
    if category == "Brick":
        return 3
    if category in ("Plate", "Tile"):
        return 1
    if category == "Slope":
        return 3
    return 1


# ── Image download ────────────────────────────────────────────────────

def check_image_exists(part_num: str) -> bool:
    """HEAD request to check if the CDN has an image for this part."""
    url = IMAGE_URL_TEMPLATE.format(part_num=part_num)
    try:
        resp = requests.head(url, timeout=10, allow_redirects=True)
        return resp.status_code == 200
    except Exception:
        return False


def download_image(part_num: str, dest: Path) -> bool:
    """Download part image from Rebrickable CDN."""
    url = IMAGE_URL_TEMPLATE.format(part_num=part_num)
    try:
        resp = requests.get(url, timeout=15, stream=True)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception:
        return False


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Expand LEGO parts catalog")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Max number of parts to add (default: 1000)")
    parser.add_argument("--skip-images", action="store_true",
                        help="Skip image downloads (just build catalog)")
    args = parser.parse_args()

    # Load existing catalog
    if PARTS_JSON.exists():
        with open(PARTS_JSON, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        existing_parts = {p["part_id"] for p in catalog["parts"]}
        print(f"Existing catalog: {len(existing_parts)} parts")
    else:
        catalog = {
            "version": "1.0.0",
            "generated": time.strftime("%Y-%m-%d"),
            "count": 0,
            "description": "LEGO parts catalog with BrickLink IDs",
            "parts": [],
        }
        existing_parts = set()

    # ── Step 1: Download part categories CSV ──
    print("Downloading Rebrickable part categories...")
    resp = requests.get(PART_CATS_CSV_URL, timeout=30)
    resp.raise_for_status()
    raw = gzip.decompress(resp.content).decode("utf-8")
    reader = csv.DictReader(io.StringIO(raw))
    cat_names: dict[int, str] = {}
    for row in reader:
        cat_id = int(row["id"])
        cat_names[cat_id] = row["name"]
    print(f"  Found {len(cat_names)} categories")

    # ── Step 2: Download parts CSV ──
    print("Downloading Rebrickable parts CSV (~3MB compressed)...")
    resp = requests.get(PARTS_CSV_URL, timeout=60)
    resp.raise_for_status()
    raw = gzip.decompress(resp.content).decode("utf-8")
    reader = csv.DictReader(io.StringIO(raw))

    candidates: list[dict] = []
    skipped_cat = 0
    skipped_existing = 0

    for row in reader:
        part_num = row["part_num"]
        name = row["name"]
        cat_id = int(row["part_cat_id"])
        cat_name = cat_names.get(cat_id, "")

        # Skip if already in catalog
        if part_num in existing_parts:
            skipped_existing += 1
            continue

        # Classify into our categories
        our_cat = classify_category(cat_name)
        if our_cat is None:
            skipped_cat += 1
            continue

        # Skip parts with complex names (usually assemblies or printed parts)
        low_name = name.lower()
        if any(skip in low_name for skip in [
            "pattern", "print", "sticker", "assembly", "complete",
            "with ", "and ", "decorated", "molded",
        ]):
            # Allow simple "with groove" or "with lip" but skip complex descriptions
            if "with groove" not in low_name and "with lip" not in low_name:
                continue

        # Skip non-standard part numbers (contain letters other than suffix)
        # Keep things like "3001", "3069b", "92946" but skip "x123" or "bb0001"
        if not re.match(r"^\d+[a-z]?$", part_num):
            continue

        stud_w, stud_l, dims = parse_dimensions(name)

        candidates.append({
            "part_num": part_num,
            "name": name,
            "category": our_cat,
            "rebrickable_cat": cat_name,
            "stud_width": stud_w,
            "stud_length": stud_l,
            "dimensions": dims,
        })

    print(f"  Total parts in CSV: {skipped_cat + skipped_existing + len(candidates) + skipped_existing}")
    print(f"  Skipped (wrong category): {skipped_cat}")
    print(f"  Skipped (already in catalog): {skipped_existing}")
    print(f"  Candidates: {len(candidates)}")

    # ── Step 3: Prioritize parts ──
    # Prefer parts with dimensions (they're standard bricks/plates)
    # and shorter names (simpler parts)
    def sort_key(c: dict) -> tuple:
        has_dims = 0 if c["dimensions"] else 1
        name_len = len(c["name"])
        return (has_dims, name_len)

    candidates.sort(key=sort_key)

    # Take up to the limit
    to_add = candidates[:args.limit]
    print(f"\nWill add {len(to_add)} parts to catalog")

    # ── Step 4: Download images and verify they exist ──
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    new_parts: list[dict] = []
    failed_images = 0

    for i, cand in enumerate(to_add):
        part_num = cand["part_num"]
        image_path = f"data/images/{part_num}.png"
        local_path = PROJECT_ROOT / image_path

        # Download image if needed
        if not local_path.exists() and not args.skip_images:
            success = download_image(part_num, local_path)
            if not success:
                failed_images += 1
                continue
            time.sleep(DOWNLOAD_DELAY)
        elif not local_path.exists() and args.skip_images:
            # Even without downloading, record the part
            pass

        # Build part entry matching our schema
        part_entry = {
            "part_id": part_num,
            "name": cand["name"],
            "category": cand["category"],
            "subcategory": None,
            "dimensions": cand["dimensions"],
            "stud_width": cand["stud_width"],
            "stud_length": cand["stud_length"],
            "height_plates": height_plates_for_category(cand["category"]),
            "image_path": image_path,
            "image_url": IMAGE_URL_TEMPLATE.format(part_num=part_num),
            "embedding": None,
            "aliases": [],
            "tags": [],
            "bricklink_url": f"https://www.bricklink.com/v2/catalog/catalogitem.page?P={part_num}",
            "rebrickable_part_num": part_num,
            "year_introduced": None,
            "is_obsolete": False,
        }

        new_parts.append(part_entry)

        # Progress
        if (i + 1) % 50 == 0 or i == len(to_add) - 1:
            print(f"  Progress: {i + 1}/{len(to_add)} "
                  f"({len(new_parts)} added, {failed_images} failed)")

    # ── Step 5: Update catalog ──
    catalog["parts"].extend(new_parts)
    catalog["count"] = len(catalog["parts"])
    catalog["generated"] = time.strftime("%Y-%m-%d")
    catalog["description"] = (
        f"LEGO parts catalog with {catalog['count']} parts. "
        f"Expanded from Rebrickable bulk data."
    )

    # Write atomically
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(
        "w", dir=str(DATA_DIR), suffix=".json",
        delete=False, encoding="utf-8",
    ) as tmp:
        json.dump(catalog, tmp, indent=2, ensure_ascii=False)
        tmp_path = tmp.name
    os.replace(tmp_path, str(PARTS_JSON))

    print(f"\n{'='*60}")
    print(f"Catalog expanded!")
    print(f"  Previous: {len(existing_parts)} parts")
    print(f"  Added:    {len(new_parts)} parts")
    print(f"  Total:    {catalog['count']} parts")
    print(f"  Failed:   {failed_images} (no image on CDN)")
    print(f"")
    print(f"Next step: rebuild embeddings:")
    print(f"  python scripts/build_embeddings.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
