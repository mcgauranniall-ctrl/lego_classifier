"""
FastAPI server for the LEGO detection and identification pipeline.

Endpoints:
    POST /api/analyze      — Upload an image, get detection results
    GET  /api/health       — Health check + model status
    GET  /api/parts        — List all indexed parts
    GET  /api/parts/{id}   — Get a single part by BrickLink ID

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Expects data/parts.json (with embeddings) to exist.
Run `python scripts/build_embeddings.py` first to generate them.
"""

from __future__ import annotations

import io
import os
import random
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

DEMO_MODE = os.environ.get("DEMO_MODE", "").lower() in ("1", "true", "yes")

if DEMO_MODE:
    print("[server] DEMO_MODE is ON — returning fake detections, no ML models loaded")

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LEGO Classifier API",
    version="1.0.0",
    description="Detect and identify LEGO pieces from images.",
)

# CORS: allow dev origins + any production origin set via CORS_ORIGINS env var.
# CORS_ORIGINS accepts a comma-separated list, e.g. "https://my-app.vercel.app,https://custom.domain.com"
_cors_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
_extra_origins = os.environ.get("CORS_ORIGINS", "")
if _extra_origins:
    _cors_origins.extend(o.strip() for o in _extra_origins.split(",") if o.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Thread-safe lazy-loaded pipeline singleton
# ---------------------------------------------------------------------------

_pipeline = None
_pipeline_lock = threading.Lock()


def get_pipeline():
    """Lazy-init the pipeline on first request (avoids slow startup)."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from ml.pipeline import LegoPipeline

                parts_path = PROJECT_ROOT / "data" / "parts.json"
                npy_path = PROJECT_ROOT / "data" / "embeddings.npy"
                index_path = PROJECT_ROOT / "data" / "embeddings_index.json"

                # Use fine-tuned LEGO model if available, else fall back to COCO
                yolo_model = os.environ.get("YOLO_MODEL", "yolov8m.pt")
                lego_model = PROJECT_ROOT / "lego_yolov8.pt"
                if lego_model.exists() and yolo_model == "yolov8m.pt":
                    yolo_model = str(lego_model)

                _pipeline = LegoPipeline(
                    parts_json_path=parts_path,
                    embeddings_npy_path=npy_path if npy_path.exists() else None,
                    embeddings_index_path=index_path if index_path.exists() else None,
                    yolo_model_path=yolo_model,
                )
    return _pipeline


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_IMAGE_DIM = 8000  # pixels — guard against decompression bombs
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.post("/api/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    conf_threshold: float = Query(0.10, ge=0.05, le=0.95),
    max_detections: int = Query(50, ge=1, le=100),
    top_k: int = Query(5, ge=1, le=20),
):
    """
    Upload a LEGO image for detection and identification.

    Returns bounding boxes, part matches with BrickLink IDs,
    confidence scores, and a grouped parts list (bill of materials).

    **Request:** multipart/form-data with an `image` file field.

    **Example response:**
    ```json
    {
      "status": "ok",
      "processing_time_ms": 1840.3,
      "total_pieces_detected": 5,
      "unique_parts": 3,
      "detections": [...],
      "grouped_parts": [
        {"part_id": "3001", "name": "Brick 2 x 4", "count": 3, ...}
      ]
    }
    ```
    """
    # Validate content type
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{image.content_type}'. "
            f"Allowed: {', '.join(ALLOWED_TYPES)}",
        )

    # Read and validate size
    contents = await image.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(contents)} bytes). Max {MAX_FILE_SIZE} bytes.",
        )

    # Decode and validate image
    try:
        pil_image = Image.open(io.BytesIO(contents))
        pil_image.verify()  # check for malformed data
        pil_image = Image.open(io.BytesIO(contents))  # reopen after verify
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    if pil_image.width > MAX_IMAGE_DIM or pil_image.height > MAX_IMAGE_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions too large ({pil_image.width}x{pil_image.height}). "
            f"Max {MAX_IMAGE_DIM}px per side.",
        )

    pil_image = pil_image.convert("RGB")

    # Demo mode: return realistic fake detections without loading models
    if DEMO_MODE:
        return JSONResponse(content=_generate_demo_response(pil_image, max_detections, top_k))

    # Run pipeline
    pipeline = get_pipeline()

    if not pipeline.matcher.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Embedding index not loaded. Run build_embeddings.py first.",
        )

    result = pipeline.analyze_image(
        image=pil_image,
        conf_threshold=conf_threshold,
        max_detections=max_detections,
        match_top_k=top_k,
    )

    return JSONResponse(content=result.to_dict())


@app.get("/api/health")
async def health():
    """Health check: reports model load status and indexed part count."""
    if DEMO_MODE:
        return {
            "status": "healthy",
            "demo_mode": True,
            "matcher_ready": True,
            "indexed_parts": len(_DEMO_PARTS),
            "embedding_dim": 512,
        }
    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "matcher_ready": pipeline.matcher.is_ready,
        "indexed_parts": pipeline.matcher.embeddings.shape[0]
        if pipeline.matcher.is_ready
        else 0,
        "embedding_dim": pipeline.matcher.embeddings.shape[1]
        if pipeline.matcher.is_ready
        else 0,
    }


@app.get("/api/parts")
async def list_parts(
    category: str | None = Query(None, description="Filter by category"),
):
    """List all indexed parts, optionally filtered by category."""
    if DEMO_MODE:
        parts = _DEMO_PARTS
        if category:
            parts = [p for p in parts if p["category"].lower() == category.lower()]
        return {
            "count": len(parts),
            "parts": [
                {
                    "part_id": p["part_id"],
                    "name": p["name"],
                    "category": p["category"],
                    "dimensions": p.get("dimensions", ""),
                    "bricklink_url": p.get("bricklink_url", ""),
                    "image_url": p.get("image_url", ""),
                }
                for p in parts
            ],
        }

    pipeline = get_pipeline()
    parts = pipeline.matcher.parts

    if category:
        parts = [p for p in parts if p["category"].lower() == category.lower()]

    return {
        "count": len(parts),
        "parts": [
            {
                "part_id": p["part_id"],
                "name": p["name"],
                "category": p["category"],
                "dimensions": p.get("dimensions", ""),
                "bricklink_url": p.get("bricklink_url", ""),
                "image_url": p.get("image_url", ""),
            }
            for p in parts
        ],
    }


@app.get("/api/parts/{part_id}")
async def get_part(part_id: str):
    """Get a single part by BrickLink ID."""
    if DEMO_MODE:
        for p in _DEMO_PARTS:
            if p["part_id"] == part_id:
                return p
        raise HTTPException(status_code=404, detail=f"Part '{part_id}' not found.")

    pipeline = get_pipeline()

    for p in pipeline.matcher.parts:
        if p["part_id"] == part_id:
            return {
                "part_id": p["part_id"],
                "name": p["name"],
                "category": p["category"],
                "subcategory": p.get("subcategory"),
                "dimensions": p.get("dimensions", ""),
                "stud_width": p.get("stud_width"),
                "stud_length": p.get("stud_length"),
                "height_plates": p.get("height_plates"),
                "bricklink_url": p.get("bricklink_url", ""),
                "image_url": p.get("image_url", ""),
                "aliases": p.get("aliases", []),
                "tags": p.get("tags", []),
                "year_introduced": p.get("year_introduced"),
                "is_obsolete": p.get("is_obsolete", False),
            }

    raise HTTPException(status_code=404, detail=f"Part '{part_id}' not found.")


# ---------------------------------------------------------------------------
# Demo mode — realistic fake data for end-to-end testing
# ---------------------------------------------------------------------------

_DEMO_PARTS = [
    {
        "part_id": "3001",
        "name": "Brick 2 x 4",
        "category": "Brick",
        "subcategory": "Standard",
        "dimensions": "2 x 4 x 1",
        "stud_width": 2,
        "stud_length": 4,
        "height_plates": 3,
        "bricklink_url": "https://www.bricklink.com/v2/catalog/catalogitem.page?P=3001",
        "image_url": "https://img.bricklink.com/ItemImage/PT/0/3001.png",
        "aliases": ["3001old"],
        "tags": ["basic", "brick"],
        "year_introduced": 1958,
        "is_obsolete": False,
    },
    {
        "part_id": "3020",
        "name": "Plate 2 x 4",
        "category": "Plate",
        "subcategory": "Standard",
        "dimensions": "2 x 4 x 0.33",
        "stud_width": 2,
        "stud_length": 4,
        "height_plates": 1,
        "bricklink_url": "https://www.bricklink.com/v2/catalog/catalogitem.page?P=3020",
        "image_url": "https://img.bricklink.com/ItemImage/PT/0/3020.png",
        "aliases": [],
        "tags": ["basic", "plate"],
        "year_introduced": 1965,
        "is_obsolete": False,
    },
    {
        "part_id": "3003",
        "name": "Brick 2 x 2",
        "category": "Brick",
        "subcategory": "Standard",
        "dimensions": "2 x 2 x 1",
        "stud_width": 2,
        "stud_length": 2,
        "height_plates": 3,
        "bricklink_url": "https://www.bricklink.com/v2/catalog/catalogitem.page?P=3003",
        "image_url": "https://img.bricklink.com/ItemImage/PT/0/3003.png",
        "aliases": [],
        "tags": ["basic", "brick"],
        "year_introduced": 1958,
        "is_obsolete": False,
    },
    {
        "part_id": "3069b",
        "name": "Tile 1 x 2",
        "category": "Tile",
        "subcategory": "Standard",
        "dimensions": "1 x 2 x 0.33",
        "stud_width": 1,
        "stud_length": 2,
        "height_plates": 1,
        "bricklink_url": "https://www.bricklink.com/v2/catalog/catalogitem.page?P=3069b",
        "image_url": "https://img.bricklink.com/ItemImage/PT/0/3069b.png",
        "aliases": ["3069"],
        "tags": ["basic", "tile", "smooth"],
        "year_introduced": 1973,
        "is_obsolete": False,
    },
    {
        "part_id": "3039",
        "name": "Slope 45 2 x 2",
        "category": "Slope",
        "subcategory": "45 Degree",
        "dimensions": "2 x 2 x 1.33",
        "stud_width": 2,
        "stud_length": 2,
        "height_plates": 4,
        "bricklink_url": "https://www.bricklink.com/v2/catalog/catalogitem.page?P=3039",
        "image_url": "https://img.bricklink.com/ItemImage/PT/0/3039.png",
        "aliases": [],
        "tags": ["slope", "roof"],
        "year_introduced": 1970,
        "is_obsolete": False,
    },
]


def _generate_demo_response(
    pil_image: Image.Image,
    max_detections: int,
    top_k: int,
) -> dict:
    """Generate a realistic fake analysis response for demo/testing."""
    start = time.perf_counter()

    img_w, img_h = pil_image.size
    num_detections = random.randint(3, min(7, max_detections))

    # Pick random parts for detections (allow repeats)
    chosen = random.choices(_DEMO_PARTS, k=num_detections)

    detections = []
    for i, part in enumerate(chosen):
        # Generate bounding boxes spread across the image
        box_w = random.randint(img_w // 8, img_w // 4)
        box_h = random.randint(img_h // 8, img_h // 4)
        x1 = random.randint(0, max(0, img_w - box_w))
        y1 = random.randint(0, max(0, img_h - box_h))
        x2 = min(x1 + box_w, img_w)
        y2 = min(y1 + box_h, img_h)

        det_conf = round(random.uniform(0.65, 0.98), 4)
        sim = round(random.uniform(0.72, 0.96), 4)
        tier = "high" if sim >= 0.85 else ("medium" if sim >= 0.70 else "uncertain")

        # Build top-k matches (best match + runner-ups)
        matches = [
            {
                "rank": 1,
                "part_id": part["part_id"],
                "part_name": part["name"],
                "category": part["category"],
                "similarity": sim,
                "tier": tier,
                "bricklink_url": part["bricklink_url"],
                "image_url": part["image_url"],
            }
        ]
        others = [p for p in _DEMO_PARTS if p["part_id"] != part["part_id"]]
        for rank, alt in enumerate(random.sample(others, min(top_k - 1, len(others))), start=2):
            alt_sim = round(sim * random.uniform(0.55, 0.85), 4)
            alt_tier = "high" if alt_sim >= 0.85 else ("medium" if alt_sim >= 0.70 else "uncertain")
            matches.append({
                "rank": rank,
                "part_id": alt["part_id"],
                "part_name": alt["name"],
                "category": alt["category"],
                "similarity": alt_sim,
                "tier": alt_tier,
                "bricklink_url": alt["bricklink_url"],
                "image_url": alt["image_url"],
            })

        detections.append({
            "detection_id": f"det_{i}",
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "detection_confidence": det_conf,
            "matches": matches[:top_k],
        })

    # Group by part_id (bill of materials)
    counts: dict[str, dict] = {}
    for det in detections:
        best = det["matches"][0]
        pid = best["part_id"]
        if pid not in counts:
            counts[pid] = {
                "part_id": pid,
                "name": best["part_name"],
                "category": best["category"],
                "count": 0,
                "best_similarity": best["similarity"],
                "tier": best["tier"],
                "bricklink_url": best["bricklink_url"],
                "image_url": best["image_url"],
            }
        counts[pid]["count"] += 1
        if best["similarity"] > counts[pid]["best_similarity"]:
            counts[pid]["best_similarity"] = best["similarity"]
            counts[pid]["tier"] = best["tier"]

    grouped = sorted(counts.values(), key=lambda g: g["count"], reverse=True)
    elapsed = (time.perf_counter() - start) * 1000

    return {
        "status": "ok",
        "processing_time_ms": round(elapsed + random.uniform(800, 2500), 1),
        "image_width": img_w,
        "image_height": img_h,
        "total_pieces_detected": num_detections,
        "unique_parts": len(grouped),
        "detections": detections,
        "grouped_parts": grouped,
    }
