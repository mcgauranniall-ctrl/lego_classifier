import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // ML_SERVICE_URL is used server-side by API routes in lib/ml-client.ts.
  // No rewrites needed — Next.js API routes proxy requests to the Python service.
};

export default nextConfig;
