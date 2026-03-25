"""
Web server for LEGO piece identification.

Serves a simple drag-and-drop webpage and processes uploads
through YOLO detection + Brickognize API identification.

Usage:
    python brickognize/server.py
    Then open http://localhost:8000 in your browser.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import uvicorn

from brickognize.pipeline import analyze_image

app = FastAPI(title="LEGO Identifier", version="1.0.0")

YOLO_MODEL = None  # auto-detect: uses SAM if available, falls back to YOLO
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
FEEDBACK_FILE = PROJECT_ROOT / "data" / "feedback.json"
FEEDBACK_CROPS_DIR = PROJECT_ROOT / "data" / "feedback_crops"


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)):
    if image.content_type and image.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Invalid file type: {image.content_type}")

    contents = await image.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large ({len(contents)} bytes)")

    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image")

    result = analyze_image(pil_image, top_k=5)

    return JSONResponse(content={
        "status": "ok",
        "processing_time_ms": round(result.processing_time_ms, 1),
        "image_width": result.image_width,
        "image_height": result.image_height,
        "total_pieces": result.total_pieces,
        "unique_parts": result.unique_parts,
        "detections": [
            {
                "detection_id": d.detection_id,
                "bbox": {"x1": d.bbox[0], "y1": d.bbox[1],
                         "x2": d.bbox[2], "y2": d.bbox[3]},
                "detection_confidence": round(d.detection_confidence, 4),
                "matches": [
                    {
                        "rank": j + 1,
                        "part_id": r.part_id,
                        "name": r.name,
                        "score": round(r.score, 4),
                        "image_url": r.image_url,
                        "bricklink_url": r.bricklink_url,
                    }
                    for j, r in enumerate(d.results)
                ],
            }
            for d in result.detections
        ],
        "grouped_parts": [
            {
                "part_id": g.part_id,
                "name": g.name,
                "count": g.count,
                "best_score": round(g.best_score, 4),
                "image_url": g.image_url,
                "bricklink_url": g.bricklink_url,
            }
            for g in result.grouped_parts
        ],
    })


_feedback_lock = threading.Lock()


@app.post("/api/feedback")
async def feedback(request: Request):
    """Save user feedback on identification accuracy, including cropped image."""
    body = await request.json()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    part_id = body.get("part_id", "unknown")
    safe_part = part_id.replace("/", "_").replace("\\", "_")

    # Save cropped image if provided
    crop_path = ""
    crop_b64 = body.get("crop_image", "")
    if crop_b64:
        FEEDBACK_CROPS_DIR.mkdir(parents=True, exist_ok=True)
        # Strip data URL prefix if present
        if "," in crop_b64:
            crop_b64 = crop_b64.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(crop_b64)
            filename = f"{ts}_{safe_part}.png"
            save_path = FEEDBACK_CROPS_DIR / filename
            save_path.write_bytes(img_bytes)
            crop_path = f"data/feedback_crops/{filename}"
        except Exception:
            pass

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "part_id": part_id,
        "correct": body.get("correct", True),
        "correction": body.get("correction", ""),
        "bbox": body.get("bbox"),
        "image_width": body.get("image_width"),
        "image_height": body.get("image_height"),
        "crop_image": crop_path,
    }

    with _feedback_lock:
        existing = []
        if FEEDBACK_FILE.exists():
            try:
                existing = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        existing.append(entry)
        FEEDBACK_FILE.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    print("\n  LEGO Piece Identifier")
    print("  Open http://localhost:8000 in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
