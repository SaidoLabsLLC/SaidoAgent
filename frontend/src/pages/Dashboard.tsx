/**
 * Dashboard page with knowledge store stats, recent activity, and quick search.
 */

import { useCallback, useState } from "react";
import { Link } from "react-router-dom";
import SearchBar from "../components/SearchBar";
import { useSearch, useStats } from "../hooks/useApi";

export default function Dashboard() {
  const { data: stats, loading: statsLoading } = useStats();
  const [searchQuery, setSearchQuery] = useState("");
  const { data: searchResults, loading: searchLoading } =
    useSearch(searchQuery);

  const handleSearch = useCallback((q: string) => {
    setSearchQuery(q);
  }, []);

  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-500">
          Knowledge store overview and quick access.
        </p>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Articles"
          value={stats?.document_count}
          loading={statsLoading}
          icon="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
        />
        <StatCard
          label="Categories"
          value={stats?.category_count}
          loading={statsLoading}
          icon="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
        />
        <StatCard
          label="Concepts"
          value={stats?.concept_count}
          loading={statsLoading}
          icon="M13 10V3L4 14h7v7l9-11h-7z"
        />
        <StatCard
          label="Total Size"
          value={
            stats
              ? `${(stats.total_size_bytes / 1024).toFixed(0)} KB`
              : undefined
          }
          loading={statsLoading}
          icon="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4"
        />
      </div>

      {/* Quick search */}
      <div>
        <h2 className="mb-3 text-lg font-semibold text-gray-900">
          Quick Search
        </h2>
        <SearchBar onChange={handleSearch} />

        {searchLoading && (
          <p className="mt-4 text-sm text-gray-400">Searching...</p>
        )}

        {searchResults && searchResults.length > 0 && (
          <ul className="mt-4 space-y-2">
            {searchResults.slice(0, 5).map((r) => (
              <li key={r.slug}>
                <Link
                  to={`/documents/${encodeURIComponent(r.slug)}`}
                  className="card block transition-shadow hover:shadow-md"
                >
                  <h3 className="text-sm font-medium text-gray-900">
                    {r.title}
                  </h3>
                  <p className="mt-0.5 text-xs text-gray-500 line-clamp-1">
                    {r.snippet}
                  </p>
                  <span className="mt-1 inline-block text-[10px] text-gray-400">
                    Score: {(r.score * 100).toFixed(0)}%
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        )}

        {searchResults && searchResults.length === 0 && searchQuery && (
          <p className="mt-4 text-sm text-gray-400">
            No results found for &quot;{searchQuery}&quot;.
          </p>
        )}
      </div>

      {/* Quick actions */}
      <div>
        <h2 className="mb-3 text-lg font-semibold text-gray-900">
          Quick Actions
        </h2>
        <div className="flex flex-wrap gap-3">
          <Link to="/chat" className="btn-primary">
            Start a conversation
          </Link>
          <Link to="/ingest" className="btn-secondary">
            Add knowledge
          </Link>
          <Link to="/documents" className="btn-secondary">
            Browse articles
          </Link>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stat card sub-component
// ---------------------------------------------------------------------------

function StatCard({
  label,
  value,
  loading,
  icon,
}: {
  label: string;
  value: number | string | undefined;
  loading: boolean;
  icon: string;
}) {
  return (
    <div className="card flex items-center gap-4">
      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-brand-50">
        <svg
          className="h-5 w-5 text-brand-600"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={1.5}
          aria-hidden="true"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d={icon} />
        </svg>
      </div>
      <div>
        <p className="text-sm text-gray-500">{label}</p>
        {loading ? (
          <div className="mt-1 h-6 w-16 animate-pulse rounded bg-gray-200" />
        ) : (
          <p className="text-xl font-semibold text-gray-900">
            {value ?? "--"}
          </p>
        )}
      </div>
    </div>
  );
}
