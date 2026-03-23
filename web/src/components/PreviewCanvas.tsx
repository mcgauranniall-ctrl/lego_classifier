"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Detection } from "@/lib/types";

interface PreviewCanvasProps {
  imageFile: File | null;
  detections: Detection[];
  /** Which detection ID is hovered in the results panel */
  highlightedId: string | null;
  onDetectionHover: (id: string | null) => void;
}

const TIER_COLORS: Record<string, string> = {
  high: "#22c55e",      // green-500
  medium: "#eab308",    // yellow-500
  uncertain: "#ef4444", // red-500
};

export default function PreviewCanvas({
  imageFile,
  detections,
  highlightedId,
  onDetectionHover,
}: PreviewCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [scale, setScale] = useState(1);

  // Load the image when file changes
  useEffect(() => {
    if (!imageFile) {
      setImage(null);
      return;
    }
    const url = URL.createObjectURL(imageFile);
    const img = new Image();
    img.onload = () => {
      setImage(img);
    };
    img.src = url;
    return () => URL.revokeObjectURL(url);
  }, [imageFile]);

  // Draw image + bounding boxes
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !image) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Scale image to fit container width
    const containerWidth = container.clientWidth;
    const s = Math.min(1, containerWidth / image.naturalWidth);
    setScale(s);

    const displayW = Math.round(image.naturalWidth * s);
    const displayH = Math.round(image.naturalHeight * s);

    canvas.width = displayW;
    canvas.height = displayH;

    // Draw image
    ctx.drawImage(image, 0, 0, displayW, displayH);

    // Draw bounding boxes
    for (const det of detections) {
      const tier = det.best_match?.tier || "uncertain";
      const color = TIER_COLORS[tier] || TIER_COLORS.uncertain;
      const isHighlighted = det.id === highlightedId;

      const x = det.bbox.x1 * s;
      const y = det.bbox.y1 * s;
      const w = (det.bbox.x2 - det.bbox.x1) * s;
      const h = (det.bbox.y2 - det.bbox.y1) * s;

      // Box
      ctx.strokeStyle = color;
      ctx.lineWidth = isHighlighted ? 3 : 2;
      ctx.strokeRect(x, y, w, h);

      // Semi-transparent fill on highlight
      if (isHighlighted) {
        ctx.fillStyle = color + "22";
        ctx.fillRect(x, y, w, h);
      }

      // Label background
      const label = det.best_match
        ? `${det.best_match.part_id} (${Math.round(det.best_match.similarity * 100)}%)`
        : "?";

      ctx.font = `${isHighlighted ? "bold " : ""}${Math.max(10, 12 * s)}px system-ui, sans-serif`;
      const textMetrics = ctx.measureText(label);
      const textH = 16 * s;
      const padding = 4 * s;

      ctx.fillStyle = color;
      ctx.fillRect(x, y - textH - padding, textMetrics.width + padding * 2, textH + padding);

      // Label text
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x + padding, y - padding - 2 * s);
    }
  }, [image, detections, highlightedId]);

  useEffect(() => {
    draw();
  }, [draw]);

  // Redraw on window resize
  useEffect(() => {
    const handleResize = () => draw();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [draw]);

  // Handle mouse move to detect hover over bounding boxes
  const onMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!detections.length || !canvasRef.current) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const mx = (e.clientX - rect.left) / scale;
      const my = (e.clientY - rect.top) / scale;

      for (const det of detections) {
        const { x1, y1, x2, y2 } = det.bbox;
        if (mx >= x1 && mx <= x2 && my >= y1 && my <= y2) {
          onDetectionHover(det.id);
          return;
        }
      }
      onDetectionHover(null);
    },
    [detections, scale, onDetectionHover]
  );

  if (!imageFile) {
    return (
      <div className="w-full h-64 rounded-xl bg-gray-100 flex items-center justify-center text-gray-400 text-sm">
        No image loaded
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full">
      <canvas
        ref={canvasRef}
        onMouseMove={onMouseMove}
        onMouseLeave={() => onDetectionHover(null)}
        className="w-full rounded-xl shadow-sm cursor-crosshair"
      />
    </div>
  );
}
