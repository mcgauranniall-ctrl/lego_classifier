"""
Part matching engine: cosine similarity search against the reference index.

Supports:
- Top-K retrieval with confidence tiers
- Alias collapse for near-duplicate BrickLink parts
- Geometric deduplication of visually identical candidates
- Grouping and counting of repeated pieces in a single image
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PartMatch:
    """A single candidate match from the reference index."""

    rank: int
    part_id: str
    name: str
    category: str
    similarity: float
    tier: str  # "high", "medium", "uncertain"
    bricklink_url: str
    image_url: str


@dataclass
class DetectionResult:
    """Full result for one detected piece: bbox + ranked matches."""

    detection_id: str
    bbox: tuple[int, int, int, int]
    detection_confidence: float
    matches: list[PartMatch]


@dataclass
class GroupedPart:
    """A unique part with a count of how many times it was detected."""

    part_id: str
    name: str
    category: str
    count: int
    best_similarity: float
    tier: str
    bricklink_url: str
    image_url: str
    detection_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------

TIER_HIGH = 0.85
TIER_MEDIUM = 0.70


def _score_to_tier(score: float) -> str:
    if score >= TIER_HIGH:
        return "high"
    elif score >= TIER_MEDIUM:
        return "medium"
    return "uncertain"


# ---------------------------------------------------------------------------
# PartMatcher — the core matching engine
# ---------------------------------------------------------------------------

class PartMatcher:
    """
    Matches CLIP embeddings against a precomputed reference index.

    Parameters
    ----------
    parts_json_path : str or Path
        Path to data/parts.json.
    embeddings_npy_path : str or Path, optional
        Path to data/embeddings.npy. If None, falls back to inline
        embeddings from parts.json (suitable for seed dataset).
    embeddings_index_path : str or Path, optional
        Path to data/embeddings_index.json. Required when using .npy.

    Example
    -------
    >>> matcher = PartMatcher("data/parts.json")
    >>> # query_embedding is a (512,) unit vector from CLIP
    >>> results = matcher.match(query_embedding, top_k=5)
    >>> print(results[0].part_id, results[0].similarity)
    3001 0.912
    """

    def __init__(
        self,
        parts_json_path: str | Path = "data/parts.json",
        embeddings_npy_path: Optional[str | Path] = None,
        embeddings_index_path: Optional[str | Path] = None,
        clip_model_name: str = "ViT-B/32",
    ):
        parts_json_path = Path(parts_json_path)

        with open(parts_json_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)

        self.parts: list[dict] = catalog["parts"]
        self._clip_model_name = clip_model_name

        # Build alias → canonical mapping
        self._alias_map: dict[str, str] = {}
        for part in self.parts:
            for alias in part.get("aliases", []):
                self._alias_map[alias] = part["part_id"]

        # Load image embeddings
        if embeddings_npy_path and Path(embeddings_npy_path).exists():
            self._load_npy_embeddings(embeddings_npy_path, embeddings_index_path)
        else:
            self._load_inline_embeddings()

        # Text embeddings (lazy-loaded on first match call)
        self._text_embeddings: Optional[np.ndarray] = None
        self._text_embeddings_lock = threading.Lock()

    def _load_npy_embeddings(
        self,
        npy_path: str | Path,
        index_path: Optional[str | Path],
    ) -> None:
        """Load embeddings from a separate .npy matrix with index alignment."""
        raw = np.load(npy_path, mmap_mode="r")
        self.embeddings = raw if raw.dtype == np.float32 else raw.astype(np.float32)

        if index_path and Path(index_path).exists():
            with open(index_path, "r", encoding="utf-8") as f:
                idx_map = json.load(f)
            # Reorder self.parts to match the .npy row order
            id_to_part = {p["part_id"]: p for p in self.parts}
            ordered = []
            for pid, row_idx in sorted(idx_map.items(), key=lambda x: x[1]):
                if pid in id_to_part:
                    ordered.append(id_to_part[pid])
            self.parts = ordered
        else:
            # Assume parts.json order matches .npy row order
            pass

        assert len(self.parts) == self.embeddings.shape[0], (
            f"Parts count ({len(self.parts)}) != embeddings rows "
            f"({self.embeddings.shape[0]})"
        )

    def _load_inline_embeddings(self) -> None:
        """Load embeddings from the embedding field inside parts.json."""
        valid_parts = []
        valid_embeddings = []

        for part in self.parts:
            emb = part.get("embedding")
            if emb is not None:
                valid_parts.append(part)
                valid_embeddings.append(emb)

        if not valid_embeddings:
            self.parts = []
            self.embeddings = np.empty((0, 512), dtype=np.float32)
            return

        self.parts = valid_parts
        self.embeddings = np.array(valid_embeddings, dtype=np.float32)

        # Ensure unit-norm
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.embeddings = self.embeddings / norms

    @property
    def is_ready(self) -> bool:
        """True if the matcher has at least one indexed part."""
        return self.embeddings.shape[0] > 0

    def _build_text_embeddings(self) -> np.ndarray:
        """
        Generate CLIP text embeddings for all parts.

        Uses descriptions like "a photo of a LEGO Brick 2 x 4" which
        leverage CLIP's text understanding to bridge the domain gap
        between user photos and LDraw reference renders.
        """
        import clip
        import torch

        if self._text_embeddings is not None:
            return self._text_embeddings

        with self._text_embeddings_lock:
            if self._text_embeddings is not None:
                return self._text_embeddings

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _ = clip.load(self._clip_model_name, device=device)
            model.eval()

            # Generate text descriptions for each part
            texts = []
            for part in self.parts:
                name = part["name"]
                texts.append(f"a photo of a LEGO {name}")

            # Encode in batches
            all_features = []
            batch_size = 64
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                tokens = clip.tokenize(batch_texts, truncate=True).to(device)
                with torch.no_grad():
                    features = model.encode_text(tokens)
                norms = features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                features = features / norms
                all_features.append(features.cpu().numpy().astype(np.float32))

            self._text_embeddings = np.concatenate(all_features, axis=0)
            return self._text_embeddings

    def match(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[PartMatch]:
        """
        Find the top-K most similar parts to a query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray
            A (512,) float32 vector (will be L2-normalized internally).
        top_k : int
            Number of candidates to return.
        min_score : float
            Minimum cosine similarity to include in results.

        Returns
        -------
        list[PartMatch]
            Ranked candidate matches, best first.
        """
        if not self.is_ready:
            return []

        # L2-normalize the query
        query = query_embedding.astype(np.float32).flatten()
        norm = np.linalg.norm(query)
        if norm < 1e-8:
            return []
        query = query / norm

        # Cosine similarity via dot product (embeddings are unit-normed)
        scores = self.embeddings @ query  # shape: (N,)

        # Get top-K indices (argpartition requires k < len; fall back to argsort)
        k = min(top_k, len(scores))
        if k >= len(scores):
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, k)[:k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: list[PartMatch] = []
        seen_canonical: set[str] = set()

        for rank_offset, idx in enumerate(top_indices):
            score = float(scores[idx])
            if score < min_score:
                continue

            part = self.parts[idx]
            pid = part["part_id"]

            # Alias collapse: skip if we already have the canonical version
            canonical = self._alias_map.get(pid, pid)
            if canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)

            results.append(PartMatch(
                rank=len(results) + 1,
                part_id=pid,
                name=part["name"],
                category=part["category"],
                similarity=round(score, 4),
                tier=_score_to_tier(score),
                bricklink_url=part.get("bricklink_url", ""),
                image_url=part.get("image_url", ""),
            ))

        # Geometric deduplication: remove near-identical candidates
        results = self._deduplicate(results)

        # Re-rank after dedup
        for i, m in enumerate(results):
            m.rank = i + 1

        return results[:top_k]

    def _deduplicate(self, matches: list[PartMatch]) -> list[PartMatch]:
        """
        Remove candidates that are geometrically identical
        (same category + dimensions, very high mutual similarity).
        Keep the higher-scoring one.
        """
        if len(matches) <= 1:
            return matches

        # Build a lookup for dimension info
        id_to_part = {p["part_id"]: p for p in self.parts}
        keep = []
        removed: set[int] = set()

        for i, mi in enumerate(matches):
            if i in removed:
                continue
            pi = id_to_part.get(mi.part_id, {})
            for j in range(i + 1, len(matches)):
                if j in removed:
                    continue
                mj = matches[j]
                pj = id_to_part.get(mj.part_id, {})

                same_cat = mi.category == mj.category
                same_w = pi.get("stud_width") == pj.get("stud_width")
                same_l = pi.get("stud_length") == pj.get("stud_length")
                close_score = abs(mi.similarity - mj.similarity) < 0.08

                if same_cat and same_w and same_l and close_score:
                    removed.add(j)

            keep.append(mi)

        return keep


# ---------------------------------------------------------------------------
# Post-processing: group identical pieces and count
# ---------------------------------------------------------------------------

def group_detections(
    detection_results: list[DetectionResult],
    similarity_merge_threshold: float = 0.92,
) -> list[GroupedPart]:
    """
    Group detections by their best-match part_id and count occurrences.

    Pieces are considered the same part if their top match has the same
    part_id. The output is a deduplicated parts list with counts — the
    "bill of materials" view.

    Parameters
    ----------
    detection_results : list[DetectionResult]
        Full detection results from the pipeline.
    similarity_merge_threshold : float
        Unused for now; reserved for merging parts with different IDs
        but near-identical embeddings.

    Returns
    -------
    list[GroupedPart]
        Unique parts sorted by count descending, then by name.

    Example
    -------
    >>> grouped = group_detections(detection_results)
    >>> for g in grouped:
    ...     print(f"{g.count}x {g.part_id} {g.name} ({g.tier})")
    3x 3001 Brick 2 x 4 (high)
    2x 3020 Plate 2 x 4 (high)
    1x 3039 Slope 45 2 x 2 (medium)
    """
    groups: dict[str, GroupedPart] = {}

    for det in detection_results:
        if not det.matches:
            continue

        top = det.matches[0]

        if top.part_id in groups:
            g = groups[top.part_id]
            g.count += 1
            g.detection_ids.append(det.detection_id)
            if top.similarity > g.best_similarity:
                g.best_similarity = top.similarity
                g.tier = top.tier
        else:
            groups[top.part_id] = GroupedPart(
                part_id=top.part_id,
                name=top.name,
                category=top.category,
                count=1,
                best_similarity=top.similarity,
                tier=top.tier,
                bricklink_url=top.bricklink_url,
                image_url=top.image_url,
                detection_ids=[det.detection_id],
            )

    # Sort by count desc, then name asc
    sorted_groups = sorted(
        groups.values(),
        key=lambda g: (-g.count, g.name),
    )
    return sorted_groups


def filter_low_confidence(
    detection_results: list[DetectionResult],
    min_detection_conf: float = 0.3,
    min_match_similarity: float = 0.5,
) -> list[DetectionResult]:
    """
    Remove detections that are likely false positives.

    Filters on two axes:
    1. YOLO detection confidence (was an object actually detected?)
    2. Best CLIP match similarity (is the detected object a known LEGO part?)

    Parameters
    ----------
    detection_results : list[DetectionResult]
        Raw pipeline output.
    min_detection_conf : float
        Minimum YOLO confidence to keep a detection.
    min_match_similarity : float
        Minimum similarity of the best match to keep a detection.

    Returns
    -------
    list[DetectionResult]
        Filtered list with low-confidence entries removed.
    """
    filtered = []
    for det in detection_results:
        if det.detection_confidence < min_detection_conf:
            continue
        if not det.matches or det.matches[0].similarity < min_match_similarity:
            continue
        filtered.append(det)
    return filtered
