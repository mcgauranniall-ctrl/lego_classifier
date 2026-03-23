/**
 * GET /api/health — Health check for the full stack.
 *
 * Reports the status of both the Next.js layer and the
 * Python ML service (reachability, model load state, index count).
 *
 * Response (200):
 *   {
 *     "status": "healthy",
 *     "ml_service": "connected",
 *     "matcher_ready": true,
 *     "indexed_parts": 35
 *   }
 */

import { NextResponse } from "next/server";
import { checkHealth } from "@/lib/ml-client";

export async function GET() {
  const health = await checkHealth();
  const statusCode = health.status === "healthy" ? 200 : 503;
  return NextResponse.json(health, { status: statusCode });
}
