/**
 * Ingest page with drag-and-drop file upload and URL/text input.
 */

import { type DragEvent, type FormEvent, useCallback, useRef, useState } from "react";
import { useIngest } from "../hooks/useApi";
import { apiPost } from "../api/client";
import type { IngestResponse } from "../types";

interface IngestResult {
  filename: string;
  response: IngestResponse | null;
  error: string | null;
}

export default function Ingest() {
  const { ingestText, ingestFile, loading } = useIngest();
  const [dragOver, setDragOver] = useState(false);
  const [results, setResults] = useState<IngestResult[]>([]);
  const [urlInput, setUrlInput] = useState("");
  const [textContent, setTextContent] = useState("");
  const [textFilename, setTextFilename] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // -----------------------------------------------------------------------
  // File upload handlers
  // -----------------------------------------------------------------------

  const handleFiles = useCallback(
    async (files: FileList) => {
      for (const file of Array.from(files)) {
        const result = await ingestFile(file);
        setResults((prev) => [
          {
            filename: file.name,
            response: result,
            error: result ? null : "Upload failed",
          },
          ...prev,
        ]);
      }
    },
    [ingestFile],
  );

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (e.dataTransfer.files.length > 0) {
        handleFiles(e.dataTransfer.files);
      }
    },
    [handleFiles],
  );

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragOver(false);
  }, []);

  // -----------------------------------------------------------------------
  // URL ingest
  // -----------------------------------------------------------------------

  const handleUrlSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      const url = urlInput.trim();
      if (!url) return;

      try {
        const res = await apiPost<IngestResponse>("/v1/clip", { url });
        setResults((prev) => [
          { filename: url, response: res as unknown as IngestResponse, error: null },
          ...prev,
        ]);
        setUrlInput("");
      } catch (err) {
        setResults((prev) => [
          { filename: url, response: null, error: (err as Error).message },
          ...prev,
        ]);
      }
    },
    [urlInput],
  );

  // -----------------------------------------------------------------------
  // Text ingest
  // -----------------------------------------------------------------------

  const handleTextSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      if (!textContent.trim() || !textFilename.trim()) return;

      const result = await ingestText(textContent, textFilename);
      setResults((prev) => [
        {
          filename: textFilename,
          response: result,
          error: result ? null : "Ingest failed",
        },
        ...prev,
      ]);
      setTextContent("");
      setTextFilename("");
    },
    [textContent, textFilename, ingestText],
  );

  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Ingest Content</h1>
        <p className="mt-1 text-sm text-gray-500">
          Add knowledge from files, URLs, or text.
        </p>
      </div>

      {/* Drag-and-drop upload zone */}
      <section>
        <h2 className="mb-3 text-lg font-semibold text-gray-900">
          File Upload
        </h2>
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
          className={`
            flex cursor-pointer flex-col items-center justify-center
            rounded-lg border-2 border-dashed p-8 text-center
            transition-colors
            ${
              dragOver
                ? "border-brand-500 bg-brand-50"
                : "border-gray-300 bg-white hover:border-gray-400"
            }
          `}
          role="button"
          tabIndex={0}
          aria-label="Drop files here or click to upload"
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              fileInputRef.current?.click();
            }
          }}
        >
          <svg
            className="mb-3 h-10 w-10 text-gray-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1}
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
          <p className="text-sm font-medium text-gray-700">
            Drop files here or click to browse
          </p>
          <p className="mt-1 text-xs text-gray-400">
            Supports .md, .txt, .pdf, .docx, .html (max 50MB)
          </p>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => {
              if (e.target.files) handleFiles(e.target.files);
              e.target.value = "";
            }}
            accept=".md,.txt,.pdf,.docx,.html,.csv,.json"
          />
        </div>
      </section>

      {/* URL ingest */}
      <section>
        <h2 className="mb-3 text-lg font-semibold text-gray-900">
          Import from URL
        </h2>
        <form onSubmit={handleUrlSubmit} className="flex gap-2">
          <input
            type="url"
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
            placeholder="https://example.com/article"
            className="input flex-1"
            aria-label="URL to ingest"
          />
          <button type="submit" disabled={loading} className="btn-primary">
            Import
          </button>
        </form>
      </section>

      {/* Text ingest */}
      <section>
        <h2 className="mb-3 text-lg font-semibold text-gray-900">
          Paste Text
        </h2>
        <form onSubmit={handleTextSubmit} className="space-y-3">
          <input
            type="text"
            value={textFilename}
            onChange={(e) => setTextFilename(e.target.value)}
            placeholder="Filename (e.g., notes.md)"
            className="input"
            aria-label="Filename for text content"
          />
          <textarea
            value={textContent}
            onChange={(e) => setTextContent(e.target.value)}
            placeholder="Paste or type content here..."
            rows={6}
            className="input resize-y"
            aria-label="Text content to ingest"
          />
          <button
            type="submit"
            disabled={loading || !textContent.trim() || !textFilename.trim()}
            className="btn-primary"
          >
            Ingest Text
          </button>
        </form>
      </section>

      {/* Results / queue */}
      {results.length > 0 && (
        <section>
          <h2 className="mb-3 text-lg font-semibold text-gray-900">
            Ingest Queue
          </h2>
          <ul className="space-y-2">
            {results.map((r, i) => (
              <li
                key={`${r.filename}-${i}`}
                className="card flex items-center justify-between py-3"
              >
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-gray-900">
                    {r.filename}
                  </p>
                  {r.response && (
                    <p className="text-xs text-gray-500">
                      {r.response.title} -- {r.response.status}
                    </p>
                  )}
                  {r.error && (
                    <p className="text-xs text-red-600">{r.error}</p>
                  )}
                </div>
                <span
                  className={`badge ml-3 ${
                    r.error
                      ? "bg-red-100 text-red-700"
                      : "bg-green-100 text-green-700"
                  }`}
                >
                  {r.error ? "Failed" : "Done"}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
