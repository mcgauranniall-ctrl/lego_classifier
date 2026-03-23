"use client";

import { useCallback, useState } from "react";
import type { AnalyzeResponse, ErrorResponse } from "@/lib/types";

type Status = "idle" | "uploading" | "done" | "error";

interface UseAnalyzeReturn {
  status: Status;
  result: AnalyzeResponse | null;
  error: string | null;
  analyze: (file: File) => Promise<void>;
  reset: () => void;
}

export function useAnalyze(): UseAnalyzeReturn {
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const analyze = useCallback(async (file: File) => {
    setStatus("uploading");
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        const err = data as ErrorResponse;
        setError(err.message || `Server error (${res.status})`);
        setStatus("error");
        return;
      }

      setResult(data as AnalyzeResponse);
      setStatus("done");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to connect to server."
      );
      setStatus("error");
    }
  }, []);

  const reset = useCallback(() => {
    setStatus("idle");
    setResult(null);
    setError(null);
  }, []);

  return { status, result, error, analyze, reset };
}
