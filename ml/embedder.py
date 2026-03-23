"""
CLIP-based embedding generation for LEGO piece crops.

Produces 512-dimensional L2-normalized vectors suitable for
cosine similarity matching against the reference part index.
"""

from __future__ import annotations

import threading
from typing import Optional

import clip
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Thread-safe CLIP model cache keyed by model_name
# ---------------------------------------------------------------------------
_clip_cache: dict[str, tuple] = {}  # model_name -> (model, preprocess, device)
_clip_lock = threading.Lock()


def _get_clip(
    model_name: str = "ViT-B/32",
    device: Optional[str] = None,
) -> tuple:
    """Load the CLIP model and preprocessing transform, cached by model_name."""
    if model_name not in _clip_cache:
        with _clip_lock:
            if model_name not in _clip_cache:
                if device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                model, preprocess = clip.load(model_name, device=device)
                model.eval()
                _clip_cache[model_name] = (model, preprocess, device)

    return _clip_cache[model_name]


def extract_embeddings(
    crops: list[np.ndarray | Image.Image],
    model_name: str = "ViT-B/32",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Generate CLIP embeddings for a list of image crops.

    Parameters
    ----------
    crops : list of np.ndarray or PIL.Image
        Cropped LEGO piece images. Arrays must be RGB uint8 (H, W, 3).
    model_name : str
        CLIP model variant. Default "ViT-B/32" produces 512-dim vectors.
    batch_size : int
        Number of images to process per forward pass.

    Returns
    -------
    np.ndarray
        Shape (N, 512) float32 array of L2-normalized embeddings.
        Each row is a unit vector suitable for dot-product similarity.

    Example
    -------
    >>> import numpy as np
    >>> fake_crops = [np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)]
    >>> embeddings = extract_embeddings(fake_crops)
    >>> embeddings.shape
    (1, 512)
    >>> np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)
    True
    """
    if not crops:
        return np.empty((0, 512), dtype=np.float32)

    model, preprocess, device = _get_clip(model_name)

    # Convert all inputs to preprocessed tensors
    tensors = []
    for crop in crops:
        if isinstance(crop, np.ndarray):
            pil_img = Image.fromarray(crop)
        else:
            pil_img = crop
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        tensors.append(preprocess(pil_img))

    all_embeddings = []

    # Process in batches to control memory
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i : i + batch_size]).to(device)
        with torch.no_grad():
            features = model.encode_image(batch)
        # L2-normalize to unit vectors (clamp to avoid NaN on zero-norm)
        norms = features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        features = features / norms
        all_embeddings.append(features.cpu().numpy().astype(np.float32))

    return np.concatenate(all_embeddings, axis=0)


def extract_single_embedding(
    image: np.ndarray | Image.Image,
    model_name: str = "ViT-B/32",
) -> np.ndarray:
    """
    Convenience wrapper for a single image.

    Returns
    -------
    np.ndarray
        Shape (512,) float32 unit vector.
    """
    return extract_embeddings([image], model_name=model_name)[0]
