"""
LEGO piece detection using YOLOv8.

Uses the pretrained COCO model as a starting point.
For production, fine-tune on a LEGO-specific dataset from Roboflow.
The COCO model can detect generic objects — we use it to demonstrate
the pipeline structure and swap in a fine-tuned model later.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO


@dataclass
class Detection:
    """A single detected object in the image."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    confidence: float
    crop: np.ndarray  # RGB uint8 array of the cropped region


# ---------------------------------------------------------------------------
# Thread-safe model cache keyed by model_path
# ---------------------------------------------------------------------------
_models: dict[str, YOLO] = {}
_model_lock = threading.Lock()


def _get_model(model_path: str = "yolov8m.pt") -> YOLO:
    """Load the YOLOv8 model once per model_path and cache it."""
    if model_path not in _models:
        with _model_lock:
            if model_path not in _models:
                _models[model_path] = YOLO(model_path)
    return _models[model_path]


def detect_objects(
    image: Image.Image | str | Path,
    conf_threshold: float = 0.25,
    max_detections: int = 50,
    model_path: str = "yolov8m.pt",
) -> list[Detection]:
    """
    Detect LEGO pieces in an image and return bounding boxes with crops.

    Parameters
    ----------
    image : PIL Image, file path, or Path object
        The input image containing LEGO pieces.
    conf_threshold : float
        Minimum detection confidence (0–1). Lower catches more pieces
        at the cost of more false positives.
    max_detections : int
        Cap on the number of returned detections, sorted by confidence.
    model_path : str
        Path to the YOLO weights file. Default uses the pretrained COCO
        model; replace with a fine-tuned LEGO model for production.

    Returns
    -------
    list[Detection]
        Detected objects sorted by confidence descending, each containing
        the bounding box, confidence score, and cropped image region.

    Example
    -------
    >>> from PIL import Image
    >>> img = Image.open("test_lego.jpg")
    >>> detections = detect_objects(img, conf_threshold=0.3)
    >>> for det in detections:
    ...     print(f"bbox={det.bbox}, conf={det.confidence:.2f}, "
    ...           f"crop_shape={det.crop.shape}")
    bbox=(120, 88, 310, 275), conf=0.94, crop_shape=(187, 190, 3)
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    img_array = np.array(image)
    model = _get_model(model_path)

    # Run inference — returns a list with one Results object per image
    results = model.predict(
        source=img_array,
        conf=conf_threshold,
        max_det=max_detections,
        verbose=False,
    )

    detections: list[Detection] = []
    if not results or results[0].boxes is None:
        return detections

    boxes = results[0].boxes
    # Sort by confidence descending
    confs = boxes.conf.cpu().numpy()
    sorted_indices = np.argsort(-confs)

    for idx in sorted_indices[:max_detections]:
        x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy().astype(int)
        conf = float(confs[idx])

        # Clamp to image bounds
        h, w = img_array.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Skip degenerate boxes
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        crop = img_array[y1:y2, x1:x2].copy()
        detections.append(Detection(
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            confidence=conf,
            crop=crop,
        ))

    return detections
