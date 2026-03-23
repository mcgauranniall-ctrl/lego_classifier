"""ML pipeline for LEGO piece detection and identification."""

from ml.detector import Detection, detect_objects
from ml.embedder import extract_embeddings, extract_single_embedding
from ml.matcher import (
    DetectionResult,
    GroupedPart,
    PartMatch,
    PartMatcher,
    filter_low_confidence,
    group_detections,
)
from ml.pipeline import LegoPipeline, PipelineResult

__all__ = [
    "Detection",
    "DetectionResult",
    "GroupedPart",
    "LegoPipeline",
    "PartMatch",
    "PartMatcher",
    "PipelineResult",
    "detect_objects",
    "extract_embeddings",
    "extract_single_embedding",
    "filter_low_confidence",
    "group_detections",
]
