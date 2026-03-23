"""
End-to-end LEGO detection and identification pipeline.

Orchestrates: detect_objects → extract_embeddings → match_parts
into a single `analyze_image()` call that takes a PIL Image and
returns structured detection results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ml.detector import Detection, detect_objects
from ml.embedder import extract_embeddings
from ml.matcher import (
    DetectionResult,
    GroupedPart,
    PartMatcher,
    filter_low_confidence,
    group_detections,
)


# ---------------------------------------------------------------------------
# Detection post-processing helpers
# ---------------------------------------------------------------------------

def _iou(a: tuple, b: tuple) -> float:
    """Intersection over union of two (x1,y1,x2,y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _containment(inner: tuple, outer: tuple) -> float:
    """Fraction of inner box area that lies inside outer box."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_inner = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / area_inner if area_inner > 0 else 0.0


def _merge_overlapping_detections(
    detections: list[Detection],
    img_array: np.ndarray,
    iou_threshold: float = 0.3,
    containment_threshold: float = 0.6,
    proximity_ratio: float = 0.5,
    max_area_fraction: float = 0.4,
) -> list[Detection]:
    """
    Merge nearby or overlapping detections into larger bounding boxes.

    This fixes the common problem where YOLO detects individual studs
    or columns of a single brick as separate detections. Uses three
    criteria for merging:
    1. IoU overlap
    2. One box contained inside another
    3. Proximity — boxes whose edges are close relative to their size

    The max_area_fraction prevents cascade merging from combining
    separate bricks into one giant detection.

    Runs multiple passes until no more merges occur.
    """
    if len(detections) <= 1:
        return detections

    h, w = img_array.shape[:2]
    img_area = w * h
    max_merged_area = img_area * max_area_fraction

    boxes = [list(d.bbox) for d in detections]
    confs = [d.confidence for d in detections]
    alive = [True] * len(detections)

    # Run merge passes until stable
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            if not alive[i]:
                continue
            for j in range(i + 1, len(boxes)):
                if not alive[j]:
                    continue

                bi = tuple(boxes[i])
                bj = tuple(boxes[j])

                should_merge = False

                # Check IoU
                if _iou(bi, bj) > iou_threshold:
                    should_merge = True

                # Check containment (either direction)
                if not should_merge:
                    if (_containment(bj, bi) > containment_threshold or
                            _containment(bi, bj) > containment_threshold):
                        should_merge = True

                # Check proximity: are edges close relative to box size?
                if not should_merge and proximity_ratio > 0:
                    avg_w = ((bi[2]-bi[0]) + (bj[2]-bj[0])) / 2
                    avg_h = ((bi[3]-bi[1]) + (bj[3]-bj[1])) / 2
                    avg_size = (avg_w + avg_h) / 2

                    # Gap between boxes
                    gap_x = max(0, max(bi[0], bj[0]) - min(bi[2], bj[2]))
                    gap_y = max(0, max(bi[1], bj[1]) - min(bi[3], bj[3]))
                    gap = max(gap_x, gap_y)

                    if avg_size > 0 and gap / avg_size < proximity_ratio:
                        should_merge = True

                if should_merge:
                    # Check if merged box would be too large
                    merged_x1 = min(boxes[i][0], boxes[j][0])
                    merged_y1 = min(boxes[i][1], boxes[j][1])
                    merged_x2 = max(boxes[i][2], boxes[j][2])
                    merged_y2 = max(boxes[i][3], boxes[j][3])
                    merged_area = (merged_x2 - merged_x1) * (merged_y2 - merged_y1)

                    if merged_area > max_merged_area:
                        continue  # skip — would create too large a box

                    boxes[i] = [merged_x1, merged_y1, merged_x2, merged_y2]
                    confs[i] = max(confs[i], confs[j])
                    alive[j] = False
                    changed = True

    result = []
    for i in range(len(boxes)):
        if not alive[i]:
            continue
        x1, y1, x2, y2 = boxes[i]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img_array[y1:y2, x1:x2].copy()
        result.append(Detection(
            bbox=(x1, y1, x2, y2),
            confidence=confs[i],
            crop=crop,
        ))

    return result


def _pad_crop_to_square(crop: np.ndarray, padding_ratio: float = 0.15) -> np.ndarray:
    """
    Pad a crop to a square with white background and extra margin.

    This makes photo crops more similar to the reference LDraw renders
    (which are centered on white backgrounds), improving CLIP matching.
    """
    h, w = crop.shape[:2]
    # Add padding around the crop
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    new_w = w + 2 * pad_x
    new_h = h + 2 * pad_y

    # Make it square
    side = max(new_w, new_h)
    result = np.full((side, side, 3), 255, dtype=np.uint8)

    # Center the crop
    offset_x = (side - w) // 2
    offset_y = (side - h) // 2
    result[offset_y:offset_y + h, offset_x:offset_x + w] = crop

    return result


def _expand_bbox(
    bbox: tuple, img_w: int, img_h: int, expand_ratio: float = 0.1,
) -> tuple:
    """Expand a bounding box by a fraction of its size, clamped to image bounds."""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    dx = int(bw * expand_ratio)
    dy = int(bh * expand_ratio)
    return (
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(img_w, x2 + dx),
        min(img_h, y2 + dy),
    )


def _suppress_fragments(
    detection_results: list,
    img_array: np.ndarray,
    pipeline_matcher,
    top_k: int,
    img_w: int,
    img_h: int,
    clip_model_name: str,
) -> list:
    """
    Detect and merge fragment detections (e.g. studs on one brick).

    Strategy: find clusters of small, nearby detections whose CLIP
    matches are all "uncertain" or consistently wrong. Replace each
    cluster with a single detection covering the cluster's bounding box.

    Only triggers when there are 4+ detections that overlap or cluster
    in one area, suggesting stud-level fragmentation.
    """
    from ml.matcher import DetectionResult

    if len(detection_results) < 4:
        return detection_results

    # Check if detections look like fragments:
    # - Many detections, each with mediocre match quality
    # - Clustered together (small gaps)
    img_area = img_w * img_h

    # Find clusters of overlapping/nearby detections using union-find
    n = len(detection_results)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for i in range(n):
        bi = detection_results[i].bbox
        area_i = (bi[2] - bi[0]) * (bi[3] - bi[1])
        for j in range(i + 1, n):
            bj = detection_results[j].bbox
            area_j = (bj[2] - bj[0]) * (bj[3] - bj[1])

            # Only cluster small detections (< 15% of image each)
            if area_i > img_area * 0.15 or area_j > img_area * 0.15:
                continue

            # Check overlap or proximity
            iou = _iou(bi, bj)
            avg_size = ((bi[2]-bi[0] + bi[3]-bi[1]) + (bj[2]-bj[0] + bj[3]-bj[1])) / 4
            gap_x = max(0, max(bi[0], bj[0]) - min(bi[2], bj[2]))
            gap_y = max(0, max(bi[1], bj[1]) - min(bi[3], bj[3]))
            gap = max(gap_x, gap_y)

            if iou > 0.1 or (avg_size > 0 and gap / avg_size < 0.3):
                union(i, j)

    # Group clusters
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    # For each cluster with 3+ members, check if replacing with a
    # combined bbox gives better CLIP matches
    kept_indices: set[int] = set(range(n))
    new_detections: list = []

    for root, members in clusters.items():
        if len(members) < 3:
            continue

        # Compute cluster bounding box
        x1 = min(detection_results[m].bbox[0] for m in members)
        y1 = min(detection_results[m].bbox[1] for m in members)
        x2 = max(detection_results[m].bbox[2] for m in members)
        y2 = max(detection_results[m].bbox[3] for m in members)

        cluster_area = (x2 - x1) * (y2 - y1)

        # Don't create a replacement if it would cover too much of the image
        if cluster_area > img_area * 0.5:
            continue

        # Expand slightly and crop
        expanded = _expand_bbox((x1, y1, x2, y2), img_w, img_h, 0.05)
        ex1, ey1, ex2, ey2 = expanded
        crop = img_array[ey1:ey2, ex1:ex2].copy()
        padded = _pad_crop_to_square(crop)

        # Get CLIP embedding for the combined crop
        emb = extract_embeddings([padded], model_name=clip_model_name)[0]
        matches = pipeline_matcher.matcher.match(query_embedding=emb, top_k=top_k * 2)
        matches = _rerank_matches(matches, (ex1, ey1, ex2, ey2))
        matches = matches[:top_k]

        if not matches:
            continue

        # Compare: does the combined crop get a better match than
        # the average of the individual fragments?
        combined_score = matches[0].similarity
        avg_fragment_score = sum(
            detection_results[m].matches[0].similarity
            for m in members if detection_results[m].matches
        ) / max(1, sum(1 for m in members if detection_results[m].matches))

        if combined_score > avg_fragment_score:
            # Combined crop is better — replace fragments
            for m in members:
                kept_indices.discard(m)

            max_conf = max(detection_results[m].detection_confidence for m in members)
            new_detections.append(DetectionResult(
                detection_id=f"det_merged_{root}",
                bbox=(ex1, ey1, ex2, ey2),
                detection_confidence=max_conf,
                matches=matches,
            ))

    # Build final results: kept individual detections + merged clusters
    result = [detection_results[i] for i in sorted(kept_indices)]
    result.extend(new_detections)

    return result


def _rerank_matches(
    matches: list,
    bbox: tuple,
    parts_lookup: dict | None = None,
) -> list:
    """
    Rerank CLIP matches with category, name, and dimension priors.

    CLIP often confuses regular bricks with tiles, round parts, or
    curved pieces because photos taken at angles can look similar.
    """
    from ml.matcher import PartMatch, _score_to_tier
    import re

    if not matches:
        return matches

    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    bbox_aspect = max(bw, bh) / max(min(bw, bh), 1)

    reranked = []
    for m in matches:
        boost = 0.0

        # --- Category boost ---
        if m.category == "Brick":
            boost += 0.04
        elif m.category == "Plate":
            boost += 0.03
        elif m.category == "Tile":
            boost += 0.01
        elif m.category == "Slope":
            boost += 0.0
        elif m.category in ("Round", "Special", "Technic"):
            boost -= 0.02

        # --- Exotic name penalty ---
        name_lower = m.name.lower()
        exotic_words = (
            "curved", "corner", "round", "arch", "wedge", "dish",
            "macaroni", "pentagonal", "turret", "half circle",
            "quarter", "facet", "tapered", "light reflector",
            "no studs", "cut out", "truncated", "cone",
            "bracket", "hinge", "pin hole",
        )
        exotic_count = sum(1 for w in exotic_words if w in name_lower)
        boost -= exotic_count * 0.02

        # --- Simple name bonus ---
        if re.match(r"^(Brick|Plate|Tile|Slope)\s+\d+\s*x\s*\d+$", m.name):
            boost += 0.02

        # --- Aspect ratio hint ---
        # Extract stud dimensions from part name and compare to bbox
        dim_match = re.search(r"(\d+)\s*x\s*(\d+)", m.name)
        if dim_match:
            sw = int(dim_match.group(1))
            sl = int(dim_match.group(2))
            part_aspect = max(sw, sl) / max(min(sw, sl), 1)
            # If bbox and part have similar aspect ratios, small boost
            aspect_diff = abs(bbox_aspect - part_aspect)
            if aspect_diff < 0.5:
                boost += 0.015
            elif aspect_diff > 2.0:
                boost -= 0.01

        new_sim = m.similarity + boost
        reranked.append(PartMatch(
            rank=m.rank,
            part_id=m.part_id,
            name=m.name,
            category=m.category,
            similarity=round(new_sim, 4),
            tier=_score_to_tier(new_sim),
            bricklink_url=m.bricklink_url,
            image_url=m.image_url,
        ))

    # Sort by adjusted similarity
    reranked.sort(key=lambda m: -m.similarity)

    # Re-number ranks
    for i, m in enumerate(reranked):
        m.rank = i + 1

    return reranked


@dataclass
class PipelineResult:
    """Complete output of a single image analysis."""

    detections: list[DetectionResult]
    grouped_parts: list[GroupedPart]
    total_pieces: int
    unique_parts: int
    processing_time_ms: float
    image_width: int
    image_height: int

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "status": "ok",
            "processing_time_ms": round(self.processing_time_ms, 1),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "total_pieces_detected": self.total_pieces,
            "unique_parts": self.unique_parts,
            "detections": [
                {
                    "detection_id": d.detection_id,
                    "bbox": {
                        "x1": d.bbox[0],
                        "y1": d.bbox[1],
                        "x2": d.bbox[2],
                        "y2": d.bbox[3],
                    },
                    "detection_confidence": round(d.detection_confidence, 4),
                    "matches": [
                        {
                            "rank": m.rank,
                            "part_id": m.part_id,
                            "part_name": m.name,
                            "category": m.category,
                            "similarity": m.similarity,
                            "tier": m.tier,
                            "bricklink_url": m.bricklink_url,
                            "image_url": m.image_url,
                        }
                        for m in d.matches
                    ],
                }
                for d in self.detections
            ],
            "grouped_parts": [
                {
                    "part_id": g.part_id,
                    "name": g.name,
                    "category": g.category,
                    "count": g.count,
                    "best_similarity": g.best_similarity,
                    "tier": g.tier,
                    "bricklink_url": g.bricklink_url,
                    "image_url": g.image_url,
                }
                for g in self.grouped_parts
            ],
        }


class LegoPipeline:
    """
    Full LEGO detection + identification pipeline.

    Loads all models on first use (lazy initialization) and caches them
    for subsequent calls. The pipeline is stateless per-request: each
    call to analyze_image() is independent.

    Parameters
    ----------
    parts_json_path : str or Path
        Path to data/parts.json.
    embeddings_npy_path : str or Path, optional
        Path to data/embeddings.npy (if precomputed).
    embeddings_index_path : str or Path, optional
        Path to data/embeddings_index.json.
    yolo_model_path : str
        YOLO weights file. Default "yolov8m.pt" (auto-downloads).
    clip_model_name : str
        CLIP variant. Default "ViT-B/32".
    detection_conf : float
        YOLO confidence threshold.
    match_top_k : int
        Number of candidate matches per detection.

    Example
    -------
    >>> pipeline = LegoPipeline(parts_json_path="data/parts.json")
    >>> result = pipeline.analyze_image("test_photo.jpg")
    >>> print(f"Found {result.total_pieces} pieces, "
    ...       f"{result.unique_parts} unique parts")
    Found 12 pieces, 5 unique parts
    >>> for g in result.grouped_parts:
    ...     print(f"  {g.count}x {g.part_id} {g.name}")
      3x 3001 Brick 2 x 4
      2x 3020 Plate 2 x 4
    """

    def __init__(
        self,
        parts_json_path: str | Path = "data/parts.json",
        embeddings_npy_path: Optional[str | Path] = None,
        embeddings_index_path: Optional[str | Path] = None,
        yolo_model_path: str = "yolov8m.pt",
        clip_model_name: str = "ViT-B/32",
        detection_conf: float = 0.10,
        match_top_k: int = 5,
    ):
        self.yolo_model_path = yolo_model_path
        self.clip_model_name = clip_model_name
        self.detection_conf = detection_conf
        self.match_top_k = match_top_k

        # Initialize the matcher (loads parts catalog + embeddings)
        self.matcher = PartMatcher(
            parts_json_path=parts_json_path,
            embeddings_npy_path=embeddings_npy_path,
            embeddings_index_path=embeddings_index_path,
        )

    def analyze_image(
        self,
        image: Image.Image | str | Path,
        conf_threshold: Optional[float] = None,
        max_detections: int = 50,
        min_match_similarity: float = 0.0,
        match_top_k: Optional[int] = None,
    ) -> PipelineResult:
        """
        Run the full detection → embedding → matching pipeline.

        Parameters
        ----------
        image : PIL Image, str, or Path
            Input image with one or more LEGO pieces.
        conf_threshold : float, optional
            Override the default YOLO confidence threshold.
        max_detections : int
            Maximum number of pieces to detect.
        min_match_similarity : float
            Filter out detections whose best match score is below this.
        match_top_k : int, optional
            Number of candidate matches per detection. Overrides the
            instance default without mutating shared state.

        Returns
        -------
        PipelineResult
            Complete structured output with detections, matches,
            grouped parts, and timing information.
        """
        start_time = time.perf_counter()

        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        img_w, img_h = image.size
        conf = conf_threshold if conf_threshold is not None else self.detection_conf
        top_k = match_top_k if match_top_k is not None else self.match_top_k

        # --- Step 1: Detect objects ---
        detections: list[Detection] = detect_objects(
            image=image,
            conf_threshold=conf,
            max_detections=max_detections,
            model_path=self.yolo_model_path,
        )

        img_array = np.array(image)

        # Fallback: if YOLO finds nothing, treat the whole image as one detection.
        if not detections:
            detections = [Detection(
                bbox=(0, 0, img_w, img_h),
                confidence=1.0,
                crop=img_array,
            )]

        # --- Step 1b: Merge duplicate detections ---
        # Only merge truly overlapping/contained boxes (no proximity).
        detections = _merge_overlapping_detections(
            detections, img_array,
            iou_threshold=0.3,
            containment_threshold=0.6,
            proximity_ratio=0.0,
            max_area_fraction=1.0,
        )

        # --- Step 1c: Expand bboxes and re-crop with padding ---
        for i, det in enumerate(detections):
            expanded = _expand_bbox(det.bbox, img_w, img_h, expand_ratio=0.1)
            x1, y1, x2, y2 = expanded
            detections[i] = Detection(
                bbox=expanded,
                confidence=det.confidence,
                crop=img_array[y1:y2, x1:x2].copy(),
            )

        # --- Step 2: Extract CLIP embeddings for each crop ---
        padded_crops = [_pad_crop_to_square(d.crop) for d in detections]
        embeddings = extract_embeddings(
            crops=padded_crops,
            model_name=self.clip_model_name,
        )

        # --- Step 3: Match each embedding against the reference index ---
        detection_results: list[DetectionResult] = []

        for i, (det, emb) in enumerate(zip(detections, embeddings)):
            matches = self.matcher.match(
                query_embedding=emb,
                top_k=top_k * 2,
            )
            matches = _rerank_matches(matches, det.bbox)
            matches = matches[:top_k]

            detection_results.append(DetectionResult(
                detection_id=f"det_{i}",
                bbox=det.bbox,
                detection_confidence=det.confidence,
                matches=matches,
            ))

        # --- Step 3b: Suppress fragments ---
        # If many small detections are contained within a region and
        # individually score poorly, replace them with one larger detection
        # covering that region. This handles YOLO detecting studs on
        # one brick as separate detections.
        detection_results = _suppress_fragments(
            detection_results, img_array, self, top_k, img_w, img_h,
            self.clip_model_name,
        )

        # --- Step 4: Post-processing ---
        detection_results = filter_low_confidence(
            detection_results,
            min_detection_conf=conf,
            min_match_similarity=min_match_similarity,
        )

        # Group identical parts and count
        grouped = group_detections(detection_results)

        elapsed = (time.perf_counter() - start_time) * 1000

        return PipelineResult(
            detections=detection_results,
            grouped_parts=grouped,
            total_pieces=sum(g.count for g in grouped),
            unique_parts=len(grouped),
            processing_time_ms=elapsed,
            image_width=img_w,
            image_height=img_h,
        )


def _sliding_window_detect(
    image: Image.Image,
    matcher: PartMatcher,
    clip_model_name: str,
    existing_detections: list[Detection],
    min_window_score: float = 0.15,
    max_crops: int = 30,
) -> list[Detection]:
    """
    Scan the image with a coarse grid and keep regions that
    produce a reasonable CLIP match against the parts index.

    Optimized for CPU: limits total crops to ~30 max.
    """
    if not matcher.is_ready:
        return existing_detections or []

    img_w, img_h = image.size
    img_array = np.array(image)

    # Use just 2 window sizes with 25% overlap (not 50%)
    min_dim = min(img_w, img_h)
    window_sizes = sorted(set([
        max(80, min_dim // 2),
        max(80, min_dim // 4),
    ]), reverse=True)

    candidates: list[Detection] = []
    for win_size in window_sizes:
        stride = int(win_size * 0.75)  # 25% overlap
        for y in range(0, img_h - win_size + 1, stride):
            for x in range(0, img_w - win_size + 1, stride):
                x2 = min(x + win_size, img_w)
                y2 = min(y + win_size, img_h)
                crop = img_array[y:y2, x:x2].copy()
                candidates.append(Detection(
                    bbox=(x, y, x2, y2),
                    confidence=0.0,
                    crop=crop,
                ))
        if len(candidates) >= max_crops:
            candidates = candidates[:max_crops]
            break

    if not candidates:
        if not existing_detections:
            return [Detection(bbox=(0, 0, img_w, img_h), confidence=1.0, crop=img_array)]
        return existing_detections

    # Batch-embed all crops at once and score
    crops = [c.crop for c in candidates]
    embeddings = extract_embeddings(crops=crops, model_name=clip_model_name)

    scored: list[tuple[float, Detection]] = []
    for i, det in enumerate(candidates):
        matches = matcher.match(query_embedding=embeddings[i], top_k=1)
        if matches:
            det.confidence = matches[0].similarity
            scored.append((matches[0].similarity, det))

    scored = [(s, d) for s, d in scored if s >= min_window_score]
    if not scored:
        if not existing_detections:
            return [Detection(bbox=(0, 0, img_w, img_h), confidence=1.0, crop=img_array)]
        return existing_detections

    scored.sort(key=lambda x: x[0], reverse=True)

    # Non-maximum suppression
    kept: list[Detection] = []
    for score, det in scored:
        x1, y1, x2, y2 = det.bbox
        is_dup = False
        for ex in kept:
            ex1, ey1, ex2, ey2 = ex.bbox
            ix1, iy1 = max(x1, ex1), max(y1, ey1)
            ix2, iy2 = min(x2, ex2), min(y2, ey2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (x2 - x1) * (y2 - y1) + (ex2 - ex1) * (ey2 - ey1) - inter
            if union > 0 and inter / union > 0.4:
                is_dup = True
                break
        if not is_dup:
            kept.append(det)

    return kept if kept else existing_detections
