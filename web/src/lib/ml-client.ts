/**
 * HTTP client for the Python FastAPI ML service.
 *
 * Handles image upload, timeout, error normalization, and response
 * transformation from the verbose ML format to the simplified API format.
 */

import type {
  AnalyzeResponse,
  Detection,
  ErrorResponse,
  HealthResponse,
  MLServiceResponse,
  Piece,
} from "./types";

const ML_SERVICE_URL =
  process.env.ML_SERVICE_URL || "http://localhost:8000";

const ML_TIMEOUT_MS = 120_000; // 120s for CPU inference with sliding window

// ---------------------------------------------------------------------------
// Core client
// ---------------------------------------------------------------------------

class MLServiceError extends Error {
  constructor(
    public code: string,
    message: string,
    public statusCode: number = 502
  ) {
    super(message);
    this.name = "MLServiceError";
  }
}

/**
 * Call the Python ML service POST /api/analyze with an image buffer.
 *
 * @param imageBuffer - Raw image bytes (JPEG/PNG/WebP)
 * @param filename    - Original filename for content-type detection
 * @param options     - Optional query params (conf_threshold, max_detections, top_k)
 * @returns Transformed AnalyzeResponse for the frontend
 *
 * @example
 * ```ts
 * const buffer = await file.arrayBuffer();
 * const result = await analyzeImage(
 *   Buffer.from(buffer),
 *   "photo.jpg",
 *   { conf_threshold: 0.3, top_k: 5 }
 * );
 * console.log(result.pieces);
 * // [{ name: "Brick 2 x 4", part_id: "3001", count: 3, confidence: 0.91, ... }]
 * ```
 */
export async function analyzeImage(
  imageBuffer: Buffer,
  filename: string,
  options: {
    conf_threshold?: number;
    max_detections?: number;
    top_k?: number;
  } = {}
): Promise<AnalyzeResponse> {
  const url = new URL("/api/analyze", ML_SERVICE_URL);
  if (options.conf_threshold !== undefined)
    url.searchParams.set("conf_threshold", String(options.conf_threshold));
  if (options.max_detections !== undefined)
    url.searchParams.set("max_detections", String(options.max_detections));
  if (options.top_k !== undefined)
    url.searchParams.set("top_k", String(options.top_k));

  // Build multipart form data
  const formData = new FormData();
  const blob = new Blob([new Uint8Array(imageBuffer)], {
    type: mimeFromFilename(filename),
  });
  formData.append("image", blob, filename);

  let response: Response;
  try {
    response = await fetch(url.toString(), {
      method: "POST",
      body: formData,
      signal: AbortSignal.timeout(ML_TIMEOUT_MS),
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "TimeoutError") {
      throw new MLServiceError(
        "ML_TIMEOUT",
        `ML service did not respond within ${ML_TIMEOUT_MS / 1000}s`,
        504
      );
    }
    throw new MLServiceError(
      "ML_UNREACHABLE",
      `Could not connect to ML service at ${ML_SERVICE_URL}: ${err instanceof Error ? err.message : String(err)}`,
      502
    );
  }

  if (!response.ok) {
    let detail = `ML service returned ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // response body was not JSON
    }
    throw new MLServiceError("ML_ERROR", detail, response.status);
  }

  const mlResult: MLServiceResponse = await response.json();
  return transformResponse(mlResult);
}

/**
 * Check ML service health.
 */
export async function checkHealth(): Promise<HealthResponse> {
  try {
    const response = await fetch(`${ML_SERVICE_URL}/api/health`, {
      signal: AbortSignal.timeout(5_000),
    });

    if (!response.ok) {
      return {
        status: "degraded",
        ml_service: "connected",
        matcher_ready: false,
        indexed_parts: 0,
      };
    }

    const data = await response.json();
    return {
      status: data.matcher_ready ? "healthy" : "degraded",
      ml_service: "connected",
      matcher_ready: data.matcher_ready,
      indexed_parts: data.indexed_parts,
    };
  } catch {
    return {
      status: "unhealthy",
      ml_service: "unreachable",
      matcher_ready: false,
      indexed_parts: 0,
    };
  }
}

// ---------------------------------------------------------------------------
// Response transformation
// ---------------------------------------------------------------------------

/**
 * Transform the verbose ML service response into the simplified
 * frontend-friendly format.
 *
 * ML response (verbose):
 * ```json
 * {
 *   "grouped_parts": [
 *     { "part_id": "3001", "name": "Brick 2 x 4", "count": 3,
 *       "best_similarity": 0.912, "tier": "high", ... }
 *   ],
 *   "detections": [ ... ]
 * }
 * ```
 *
 * Transformed response (simplified):
 * ```json
 * {
 *   "pieces": [
 *     { "part_id": "3001", "name": "Brick 2 x 4", "count": 3,
 *       "confidence": 0.912, "confidence_label": "high", ... }
 *   ]
 * }
 * ```
 */
function transformResponse(ml: MLServiceResponse): AnalyzeResponse {
  // Transform grouped parts → simplified pieces array
  const pieces: Piece[] = ml.grouped_parts.map((gp) => ({
    name: gp.name,
    part_id: gp.part_id,
    category: gp.category,
    count: gp.count,
    confidence: gp.best_similarity,
    confidence_label: gp.tier,
    bricklink_url: gp.bricklink_url,
    image_url: gp.image_url,
  }));

  // Transform detections with best match extracted
  const detections: Detection[] = ml.detections.map((d) => ({
    id: d.detection_id,
    bbox: d.bbox,
    detection_confidence: d.detection_confidence,
    best_match:
      d.matches.length > 0
        ? {
            part_id: d.matches[0].part_id,
            name: d.matches[0].part_name,
            similarity: d.matches[0].similarity,
            tier: d.matches[0].tier,
          }
        : null,
    all_matches: d.matches,
  }));

  return {
    status: "ok",
    processing_time_ms: ml.processing_time_ms,
    image_width: ml.image_width,
    image_height: ml.image_height,
    total_pieces: ml.total_pieces_detected,
    unique_parts: ml.unique_parts,
    pieces,
    detections,
  };
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function mimeFromFilename(filename: string): string {
  const ext = filename.split(".").pop()?.toLowerCase();
  switch (ext) {
    case "jpg":
    case "jpeg":
      return "image/jpeg";
    case "png":
      return "image/png";
    case "webp":
      return "image/webp";
    default:
      return "image/jpeg";
  }
}

/**
 * Create an ErrorResponse object for consistent error formatting.
 */
export function createError(code: string, message: string): ErrorResponse {
  return { status: "error", code, message };
}

export { MLServiceError };
