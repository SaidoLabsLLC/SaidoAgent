/**
 * Lightweight data-fetching hooks for the Saido Agent API.
 *
 * These wrap apiGet/apiPost with React state management so pages can
 * declaratively show loading/error/data states.
 */

import { useCallback, useEffect, useState } from "react";
import { apiGet, apiPost, apiUpload } from "../api/client";
import type {
  DocumentDetail,
  DocumentSummary,
  IngestResponse,
  QueryResponse,
  SearchResult,
  StatsResponse,
} from "../types";

// ---------------------------------------------------------------------------
// Generic fetch hook
// ---------------------------------------------------------------------------

interface UseFetchResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useFetch<T>(path: string | null): UseFetchResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(!!path);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  const refetch = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    if (!path) return;
    let cancelled = false;
    setLoading(true);
    setError(null);

    apiGet<T>(path)
      .then((result) => {
        if (!cancelled) {
          setData(result);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message || "Request failed");
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [path, tick]);

  return { data, loading, error, refetch };
}

// ---------------------------------------------------------------------------
// Domain-specific hooks
// ---------------------------------------------------------------------------

export function useStats() {
  return useFetch<StatsResponse>("/v1/stats");
}

export function useDocuments() {
  return useFetch<DocumentSummary[]>("/v1/documents");
}

export function useDocument(slug: string | undefined) {
  const path = slug ? `/v1/documents/${encodeURIComponent(slug)}` : null;
  return useFetch<DocumentDetail>(path);
}

export function useSearch(query: string) {
  const path = query.trim()
    ? `/v1/search?q=${encodeURIComponent(query.trim())}`
    : null;
  return useFetch<SearchResult[]>(path);
}

// ---------------------------------------------------------------------------
// Mutation hooks
// ---------------------------------------------------------------------------

export function useQuery() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ask = useCallback(
    async (question: string): Promise<QueryResponse | null> => {
      setLoading(true);
      setError(null);
      try {
        const res = await apiPost<QueryResponse>("/v1/query", { question });
        return res;
      } catch (err) {
        setError((err as Error).message);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  return { ask, loading, error };
}

export function useIngest() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ingestText = useCallback(
    async (
      content: string,
      filename: string,
    ): Promise<IngestResponse | null> => {
      setLoading(true);
      setError(null);
      try {
        const res = await apiPost<IngestResponse>("/v1/ingest", {
          content,
          filename,
        });
        return res;
      } catch (err) {
        setError((err as Error).message);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const ingestFile = useCallback(
    async (file: File): Promise<IngestResponse | null> => {
      setLoading(true);
      setError(null);
      try {
        const res = await apiUpload<IngestResponse>("/v1/ingest/upload", file);
        return res;
      } catch (err) {
        setError((err as Error).message);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  return { ingestText, ingestFile, loading, error };
}
