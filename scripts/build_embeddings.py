"""
Offline build script: download part images and generate CLIP embeddings.

Usage:
    python scripts/build_embeddings.py

Reads data/parts.json, downloads missing images from Rebrickable CDN,
runs CLIP ViT-B/32 on each image, and writes:
  - data/embeddings.npy        (N, 512) float32 matrix
  - data/embeddings_index.json  {part_id: row_index} alignment map
  - Updates data/parts.json with inline embeddings (for seed dataset)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import clip
import numpy as np
import requests
import torch
from PIL import Image

# Project root is one level up from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
PARTS_JSON = DATA_DIR / "parts.json"
EMBEDDINGS_NPY = DATA_DIR / "embeddings.npy"
EMBEDDINGS_INDEX = DATA_DIR / "embeddings_index.json"

CLIP_MODEL = "ViT-B/32"
BATCH_SIZE = 64
DOWNLOAD_DELAY = 0.5  # seconds between CDN requests (polite)


def download_image(url: str, dest: Path) -> bool:
    """Download an image from a URL to a local path. Returns True on success."""
    try:
        resp = requests.get(url, timeout=15, stream=True)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  WARN: Failed to download {url}: {e}")
        return False


def preprocess_for_clip(
    image_path: Path,
    preprocess_fn,
) -> torch.Tensor | None:
    """
    Load an image, composite onto white background, and apply CLIP transform.
    Returns None if the image cannot be processed.
    """
    try:
        img = Image.open(image_path)

        # Composite transparent PNGs onto white background
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        return preprocess_fn(img)
    except Exception as e:
        print(f"  WARN: Could not process {image_path}: {e}")
        return None


def main() -> None:
    print(f"Loading parts catalog from {PARTS_JSON}")
    with open(PARTS_JSON, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    parts = catalog["parts"]
    print(f"Found {len(parts)} parts in catalog")

    # --- Phase 1: Download missing images ---
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0

    for part in parts:
        pid = part["part_id"]
        local_path = PROJECT_ROOT / part["image_path"]

        if local_path.exists():
            skipped += 1
            continue

        url = part["image_url"]
        print(f"  Downloading {pid} from {url}")
        success = download_image(url, local_path)
        if success:
            downloaded += 1
            time.sleep(DOWNLOAD_DELAY)  # Be polite to CDN
        else:
            print(f"  Skipping {pid} — no image available")

    print(f"Images: {downloaded} downloaded, {skipped} already cached")

    # --- Phase 2: Generate CLIP embeddings ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP {CLIP_MODEL} on {device}")
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    model.eval()

    tensors: list[torch.Tensor] = []
    valid_parts: list[dict] = []
    valid_indices: list[int] = []

    for i, part in enumerate(parts):
        local_path = PROJECT_ROOT / part["image_path"]
        if not local_path.exists():
            print(f"  SKIP {part['part_id']} — no local image")
            continue

        tensor = preprocess_for_clip(local_path, preprocess)
        if tensor is None:
            continue

        tensors.append(tensor)
        valid_parts.append(part)
        valid_indices.append(i)

    print(f"Processing {len(tensors)} images through CLIP")

    all_embeddings: list[np.ndarray] = []
    start = time.perf_counter()

    for batch_start in range(0, len(tensors), BATCH_SIZE):
        batch = torch.stack(tensors[batch_start : batch_start + BATCH_SIZE])
        batch = batch.to(device)

        with torch.no_grad():
            features = model.encode_image(batch)

        # L2-normalize to unit vectors (clamp to avoid NaN on zero-norm)
        norms = features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        features = features / norms
        all_embeddings.append(features.cpu().numpy().astype(np.float32))

        processed = min(batch_start + BATCH_SIZE, len(tensors))
        print(f"  Encoded {processed}/{len(tensors)} images")

    elapsed = time.perf_counter() - start
    print(f"CLIP encoding completed in {elapsed:.1f}s")

    if not all_embeddings:
        print("ERROR: No embeddings generated. Check that images exist.")
        sys.exit(1)

    embeddings_matrix = np.concatenate(all_embeddings, axis=0)
    print(f"Embedding matrix shape: {embeddings_matrix.shape}")

    # --- Phase 3: Write outputs ---

    # Write embeddings.npy
    np.save(EMBEDDINGS_NPY, embeddings_matrix)
    print(f"Saved {EMBEDDINGS_NPY}")

    # Write embeddings_index.json
    index_map = {}
    for row_idx, part in enumerate(valid_parts):
        index_map[part["part_id"]] = row_idx

    with open(EMBEDDINGS_INDEX, "w", encoding="utf-8") as f:
        json.dump(index_map, f, indent=2)
    print(f"Saved {EMBEDDINGS_INDEX}")

    # Update parts.json with inline embeddings (atomic write to avoid corruption)
    for row_idx, orig_idx in enumerate(valid_indices):
        parts[orig_idx]["embedding"] = embeddings_matrix[row_idx].tolist()

    catalog["parts"] = parts
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(
        "w", dir=str(DATA_DIR), suffix=".json",
        delete=False, encoding="utf-8",
    ) as tmp:
        json.dump(catalog, tmp, indent=2, ensure_ascii=False)
        tmp_path = tmp.name
    os.replace(tmp_path, str(PARTS_JSON))
    print(f"Updated {PARTS_JSON} with inline embeddings")

    print("\nBuild complete.")
    print(f"  Parts indexed: {len(valid_parts)}/{len(parts)}")
    print(f"  Matrix shape:  {embeddings_matrix.shape}")
    print(f"  .npy size:     {EMBEDDINGS_NPY.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
