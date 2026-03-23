/**
 * Shared TypeScript types for the LEGO classifier API.
 *
 * These mirror the Python pipeline's data structures and define
 * the contract between the Next.js API routes and the frontend.
 */

// ---------------------------------------------------------------------------
// ML Service response types (from Python FastAPI)
// ---------------------------------------------------------------------------

/** A single candidate match from the reference index. */
export interface MLPartMatch {
  rank: number;
  part_id: string;
  part_name: string;
  category: string;
  similarity: number;
  tier: "high" | "medium" | "uncertain";
  bricklink_url: string;
  image_url: string;
}

/** A single detection from the ML service. */
export interface MLDetection {
  detection_id: string;
  bbox: { x1: number; y1: number; x2: number; y2: number };
  detection_confidence: number;
  matches: MLPartMatch[];
}

/** A grouped part (same part_id counted across detections). */
export interface MLGroupedPart {
  part_id: string;
  name: string;
  category: string;
  count: number;
  best_similarity: number;
  tier: "high" | "medium" | "uncertain";
  bricklink_url: string;
  image_url: string;
}

/** Full response from the Python ML service POST /api/analyze. */
export interface MLServiceResponse {
  status: "ok" | "error";
  processing_time_ms: number;
  image_width: number;
  image_height: number;
  total_pieces_detected: number;
  unique_parts: number;
  detections: MLDetection[];
  grouped_parts: MLGroupedPart[];
}

// ---------------------------------------------------------------------------
// Next.js API response types (simplified for frontend)
// ---------------------------------------------------------------------------

/** A single piece in the simplified response. */
export interface Piece {
  name: string;
  part_id: string;
  category: string;
  count: number;
  confidence: number;
  confidence_label: "high" | "medium" | "uncertain";
  bricklink_url: string;
  image_url: string;
}

/** Bounding box detection with its best match. */
export interface Detection {
  id: string;
  bbox: { x1: number; y1: number; x2: number; y2: number };
  detection_confidence: number;
  best_match: {
    part_id: string;
    name: string;
    similarity: number;
    tier: "high" | "medium" | "uncertain";
  } | null;
  all_matches: MLPartMatch[];
}

/** Full response from the Next.js POST /api/analyze endpoint. */
export interface AnalyzeResponse {
  status: "ok";
  processing_time_ms: number;
  image_width: number;
  image_height: number;
  total_pieces: number;
  unique_parts: number;
  pieces: Piece[];
  detections: Detection[];
}

/** Error response. */
export interface ErrorResponse {
  status: "error";
  code: string;
  message: string;
}

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: "healthy" | "degraded" | "unhealthy";
  ml_service: "connected" | "unreachable";
  matcher_ready: boolean;
  indexed_parts: number;
}

// ---------------------------------------------------------------------------
// Parts catalog
// ---------------------------------------------------------------------------

export interface PartSummary {
  part_id: string;
  name: string;
  category: string;
  dimensions: string;
  bricklink_url: string;
  image_url: string;
}

export interface PartDetail extends PartSummary {
  subcategory: string | null;
  stud_width: number;
  stud_length: number;
  height_plates: number;
  aliases: string[];
  tags: string[];
  year_introduced: number | null;
  is_obsolete: boolean;
}
