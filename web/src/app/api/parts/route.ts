/**
 * GET /api/parts — List all indexed LEGO parts.
 *
 * Proxies to the Python ML service parts catalog.
 * Optional query: ?category=Brick
 *
 * Response (200):
 *   {
 *     "count": 35,
 *     "parts": [
 *       { "part_id": "3001", "name": "Brick 2 x 4", "category": "Brick", ... }
 *     ]
 *   }
 */

import { NextRequest, NextResponse } from "next/server";
import { createError } from "@/lib/ml-client";

const ML_SERVICE_URL =
  process.env.ML_SERVICE_URL || "http://localhost:8000";

export async function GET(request: NextRequest) {
  const category = request.nextUrl.searchParams.get("category");
  const url = new URL("/api/parts", ML_SERVICE_URL);
  if (category) url.searchParams.set("category", category);

  try {
    const response = await fetch(url.toString(), {
      signal: AbortSignal.timeout(10_000),
    });

    if (!response.ok) {
      return NextResponse.json(
        createError("ML_ERROR", `ML service returned ${response.status}`),
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      createError("ML_UNREACHABLE", "Could not connect to ML service."),
      { status: 502 }
    );
  }
}
