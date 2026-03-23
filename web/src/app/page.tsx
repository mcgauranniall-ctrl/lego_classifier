"use client";

import { useCallback, useState } from "react";
import ImageUploader from "@/components/ImageUploader";
import PreviewCanvas from "@/components/PreviewCanvas";
import ResultsPanel from "@/components/ResultsPanel";
import { useAnalyze } from "@/hooks/useAnalyze";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [highlightedId, setHighlightedId] = useState<string | null>(null);
  const { status, result, error, analyze, reset } = useAnalyze();

  const handleFileSelected = useCallback(
    (f: File) => {
      setFile(f);
      setHighlightedId(null);
      analyze(f);
    },
    [analyze]
  );

  const handleReset = useCallback(() => {
    setFile(null);
    setHighlightedId(null);
    reset();
  }, [reset]);

  return (
    <main className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zm10 0a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zm10 0a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
              </svg>
            </div>
            <h1 className="text-lg font-bold text-gray-900">LEGO Classifier</h1>
          </div>

          {file && (
            <button
              onClick={handleReset}
              className="text-sm text-gray-500 hover:text-gray-700 transition-colors"
            >
              New scan
            </button>
          )}
        </div>
      </header>

      <div className="max-w-5xl mx-auto px-4 py-8">
        {/* Upload area — shown when idle or error with no file */}
        {!file && (
          <div className="max-w-lg mx-auto">
            <div className="text-center mb-6">
              <h2 className="text-2xl font-bold text-gray-900">
                Identify LEGO pieces
              </h2>
              <p className="text-gray-500 mt-1">
                Upload a photo and we'll detect each piece and match it to its BrickLink part ID.
              </p>
            </div>
            <ImageUploader onFileSelected={handleFileSelected} />
          </div>
        )}

        {/* Loading state */}
        {file && status === "uploading" && (
          <div className="space-y-6">
            <PreviewCanvas
              imageFile={file}
              detections={[]}
              highlightedId={null}
              onDetectionHover={() => {}}
            />
            <div className="flex items-center justify-center gap-3 py-8">
              <Spinner />
              <span className="text-sm text-gray-500">
                Detecting and identifying pieces...
              </span>
            </div>
          </div>
        )}

        {/* Error state */}
        {status === "error" && (
          <div className="space-y-6">
            {file && (
              <PreviewCanvas
                imageFile={file}
                detections={[]}
                highlightedId={null}
                onDetectionHover={() => {}}
              />
            )}
            <div className="rounded-lg bg-red-50 border border-red-200 p-4">
              <p className="text-sm text-red-800 font-medium">Analysis failed</p>
              <p className="text-sm text-red-600 mt-1">{error}</p>
              <button
                onClick={() => file && analyze(file)}
                className="mt-3 text-sm text-red-700 underline hover:text-red-900"
              >
                Retry
              </button>
            </div>
          </div>
        )}

        {/* Results */}
        {status === "done" && result && (
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
            {/* Image with overlays — 3/5 width on large screens */}
            <div className="lg:col-span-3">
              <PreviewCanvas
                imageFile={file}
                detections={result.detections}
                highlightedId={highlightedId}
                onDetectionHover={setHighlightedId}
              />

              {/* Upload new image below the preview */}
              <div className="mt-4">
                <ImageUploader
                  onFileSelected={handleFileSelected}
                />
              </div>
            </div>

            {/* Results panel — 2/5 width on large screens */}
            <div className="lg:col-span-2">
              <ResultsPanel
                result={result}
                highlightedId={highlightedId}
                onDetectionHover={setHighlightedId}
              />
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

function Spinner() {
  return (
    <svg
      className="animate-spin h-5 w-5 text-blue-600"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}
