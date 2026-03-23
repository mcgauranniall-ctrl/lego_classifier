"use client";

import { useCallback, useRef, useState } from "react";

interface ImageUploaderProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];
const MAX_SIZE = 5 * 1024 * 1024;

export default function ImageUploader({
  onFileSelected,
  disabled = false,
}: ImageUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  const validate = useCallback((file: File): string | null => {
    if (!ACCEPTED.includes(file.type)) {
      return `Unsupported file type. Use JPEG, PNG, or WebP.`;
    }
    if (file.size > MAX_SIZE) {
      return `File too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Max 5MB.`;
    }
    return null;
  }, []);

  const handleFile = useCallback(
    (file: File) => {
      const err = validate(file);
      if (err) {
        setValidationError(err);
        return;
      }
      setValidationError(null);
      onFileSelected(file);
    },
    [validate, onFileSelected]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (disabled) return;
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [disabled, handleFile]
  );

  const onDragOver = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (!disabled) setDragOver(true);
    },
    [disabled]
  );

  const onDragLeave = useCallback(() => setDragOver(false), []);

  const onChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
      // Reset input so re-selecting the same file triggers onChange
      e.target.value = "";
    },
    [handleFile]
  );

  return (
    <div className="w-full">
      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => !disabled && inputRef.current?.click()}
        className={`
          relative flex flex-col items-center justify-center
          w-full h-48 rounded-xl border-2 border-dashed
          transition-colors cursor-pointer select-none
          ${disabled ? "opacity-50 cursor-not-allowed border-gray-300 bg-gray-50" : ""}
          ${dragOver ? "border-blue-500 bg-blue-50" : "border-gray-300 bg-white hover:border-gray-400 hover:bg-gray-50"}
        `}
      >
        <svg
          className="w-10 h-10 mb-3 text-gray-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M12 16V4m0 0l-4 4m4-4l4 4M4 14v4a2 2 0 002 2h12a2 2 0 002-2v-4"
          />
        </svg>
        <p className="text-sm text-gray-600 font-medium">
          Drop a LEGO photo here or{" "}
          <span className="text-blue-600 underline">browse</span>
        </p>
        <p className="text-xs text-gray-400 mt-1">
          JPEG, PNG, or WebP up to 5MB
        </p>

        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(",")}
          onChange={onChange}
          className="hidden"
          disabled={disabled}
        />
      </div>

      {validationError && (
        <p className="mt-2 text-sm text-red-600">{validationError}</p>
      )}
    </div>
  );
}
