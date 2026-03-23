/**
 * POST /api/analyze — Upload an image, detect and identify LEGO pieces.
 *
 * This is the primary endpoint. It accepts a multipart image upload,
 * forwards it to the Python ML service, and returns a simplified
 * response with identified pieces, counts, and confidence scores.
 *
 * Request:
 *   Content-Type: multipart/form-data
 *   Body: { image: File (JPEG/PNG/WebP, ≤5MB) }
 *   Query: ?conf_threshold=0.25&max_detections=50&top_k=5
 *
 * Response (200):
 *   {
 *     "status": "ok",
 *     "processing_time_ms": 1840.3,
 *     "total_pieces": 5,
 *     "unique_parts": 3,
 *     "pieces": [
 *       {
 *         "name": "Brick 2 x 4",
 *         "part_id": "3001",
 *         "category": "Brick",
 *         "count": 3,
 *         "confidence": 0.912,
 *         "confidence_label": "high",
 *         "bricklink_url": "https://www.bricklink.com/v2/catalog/catalogitem.page?P=3001",
 *         "image_url": "https://cdn.rebrickable.com/media/parts/ldraw/0/3001.png"
 *       }
 *     ],
 *     "detections": [...]
 *   }
 *
 * Error (400/502/504):
 *   { "status": "error", "code": "...", "message": "..." }
 */

import { NextRequest, NextResponse } from "next/server";
import {
  analyzeImage,
  createError,
  MLServiceError,
} from "@/lib/ml-client";

const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5 MB
const ALLOWED_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/webp",
]);

export async function POST(request: NextRequest) {
  // --- Parse multipart form data ---
  let formData: FormData;
  try {
    formData = await request.formData();
  } catch {
    return NextResponse.json(
      createError("INVALID_REQUEST", "Expected multipart/form-data with an image field."),
      { status: 400 }
    );
  }

  const file = formData.get("image");
  if (!file || !(file instanceof Blob)) {
    return NextResponse.json(
      createError("MISSING_IMAGE", "No 'image' field found in the upload."),
      { status: 400 }
    );
  }

  // --- Validate file type ---
  if (!ALLOWED_TYPES.has(file.type)) {
    return NextResponse.json(
      createError(
        "INVALID_IMAGE_TYPE",
        `File type '${file.type}' is not supported. Use JPEG, PNG, or WebP.`
      ),
      { status: 400 }
    );
  }

  // --- Validate file size ---
  const buffer = Buffer.from(await file.arrayBuffer());
  if (buffer.length > MAX_FILE_SIZE) {
    return NextResponse.json(
      createError(
        "FILE_TOO_LARGE",
        `File is ${(buffer.length / 1024 / 1024).toFixed(1)}MB. Maximum is 5MB.`
      ),
      { status: 400 }
    );
  }

  // --- Parse optional query parameters ---
  const searchParams = request.nextUrl.searchParams;
  const options = {
    conf_threshold: parseFloat(searchParams.get("conf_threshold") || "0.25"),
    max_detections: parseInt(searchParams.get("max_detections") || "50", 10),
    top_k: parseInt(searchParams.get("top_k") || "5", 10),
  };

  // --- Call ML service ---
  try {
    const filename = file instanceof File ? file.name : "upload.jpg";
    const result = await analyzeImage(buffer, filename, options);
    return NextResponse.json(result);
  } catch (err) {
    if (err instanceof MLServiceError) {
      return NextResponse.json(
        createError(err.code, err.message),
        { status: err.statusCode }
      );
    }
    return NextResponse.json(
      createError("INTERNAL_ERROR", "An unexpected error occurred."),
      { status: 500 }
    );
  }
}
