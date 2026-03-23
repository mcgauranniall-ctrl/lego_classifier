"use client";

import type { AnalyzeResponse, Piece, Detection } from "@/lib/types";

interface ResultsPanelProps {
  result: AnalyzeResponse;
  highlightedId: string | null;
  onDetectionHover: (id: string | null) => void;
}

const TIER_BADGE: Record<string, { bg: string; text: string; label: string }> = {
  high:      { bg: "bg-green-100", text: "text-green-800", label: "High" },
  medium:    { bg: "bg-yellow-100", text: "text-yellow-800", label: "Medium" },
  uncertain: { bg: "bg-red-100", text: "text-red-800", label: "Uncertain" },
};

export default function ResultsPanel({
  result,
  highlightedId,
  onDetectionHover,
}: ResultsPanelProps) {
  return (
    <div className="w-full space-y-6">
      {/* Summary bar */}
      <div className="flex items-center gap-4 text-sm text-gray-500">
        <span>
          <strong className="text-gray-900">{result.total_pieces}</strong>{" "}
          piece{result.total_pieces !== 1 && "s"} detected
        </span>
        <span className="text-gray-300">|</span>
        <span>
          <strong className="text-gray-900">{result.unique_parts}</strong>{" "}
          unique part{result.unique_parts !== 1 && "s"}
        </span>
        <span className="text-gray-300">|</span>
        <span>{result.processing_time_ms.toFixed(0)}ms</span>
      </div>

      {/* Parts list (grouped / bill of materials) */}
      <section>
        <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wide mb-3">
          Parts Found
        </h3>

        {result.pieces.length === 0 ? (
          <p className="text-sm text-gray-500">No parts identified.</p>
        ) : (
          <div className="space-y-2">
            {result.pieces.map((piece) => (
              <PieceRow key={piece.part_id} piece={piece} />
            ))}
          </div>
        )}
      </section>

      {/* Individual detections */}
      {result.detections.length > 0 && (
        <section>
          <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wide mb-3">
            Detections
          </h3>
          <div className="space-y-1">
            {result.detections.map((det) => (
              <DetectionRow
                key={det.id}
                detection={det}
                isHighlighted={det.id === highlightedId}
                onHover={onDetectionHover}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Subcomponents
// ---------------------------------------------------------------------------

function PieceRow({ piece }: { piece: Piece }) {
  const badge = TIER_BADGE[piece.confidence_label] || TIER_BADGE.uncertain;

  return (
    <div className="flex items-center gap-3 p-3 rounded-lg bg-white border border-gray-200 hover:border-gray-300 transition-colors">
      {/* Part thumbnail */}
      <div className="w-12 h-12 rounded bg-gray-100 flex-shrink-0 overflow-hidden">
        {piece.image_url && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={piece.image_url}
            alt={piece.name}
            className="w-full h-full object-contain"
            loading="lazy"
          />
        )}
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium text-sm text-gray-900 truncate">
            {piece.name}
          </span>
          <span className={`text-xs px-1.5 py-0.5 rounded-full ${badge.bg} ${badge.text}`}>
            {badge.label}
          </span>
        </div>
        <div className="flex items-center gap-3 mt-0.5 text-xs text-gray-500">
          <span className="font-mono">{piece.part_id}</span>
          <span>{piece.category}</span>
          <span>{Math.round(piece.confidence * 100)}% match</span>
        </div>
      </div>

      {/* Count */}
      <div className="flex-shrink-0 text-right">
        <span className="text-lg font-bold text-gray-900">{piece.count}</span>
        <span className="text-xs text-gray-400 ml-0.5">x</span>
      </div>

      {/* BrickLink link */}
      {piece.bricklink_url && (
        <a
          href={piece.bricklink_url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex-shrink-0 text-gray-400 hover:text-blue-600 transition-colors"
          title="View on BrickLink"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
            />
          </svg>
        </a>
      )}
    </div>
  );
}

function DetectionRow({
  detection,
  isHighlighted,
  onHover,
}: {
  detection: Detection;
  isHighlighted: boolean;
  onHover: (id: string | null) => void;
}) {
  const match = detection.best_match;
  const badge = match
    ? TIER_BADGE[match.tier] || TIER_BADGE.uncertain
    : TIER_BADGE.uncertain;

  return (
    <div
      onMouseEnter={() => onHover(detection.id)}
      onMouseLeave={() => onHover(null)}
      className={`
        flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors cursor-default
        ${isHighlighted ? "bg-blue-50 border border-blue-200" : "bg-gray-50 border border-transparent hover:bg-gray-100"}
      `}
    >
      <span className="text-xs font-mono text-gray-400 w-12">{detection.id}</span>
      <span className="flex-1 truncate text-gray-700">
        {match ? match.name : "Unknown"}
      </span>
      {match && (
        <>
          <span className="font-mono text-xs text-gray-500">{match.part_id}</span>
          <span className={`text-xs px-1.5 py-0.5 rounded-full ${badge.bg} ${badge.text}`}>
            {Math.round(match.similarity * 100)}%
          </span>
        </>
      )}
    </div>
  );
}
