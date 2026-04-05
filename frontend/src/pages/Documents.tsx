/**
 * Browse knowledge articles in a grid/list view with filtering and search.
 */

import { useCallback, useMemo, useState } from "react";
import ArticleCard from "../components/ArticleCard";
import SearchBar from "../components/SearchBar";
import { useDocuments } from "../hooks/useApi";

export default function Documents() {
  const { data: documents, loading, error } = useDocuments();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const handleSearch = useCallback((q: string) => {
    setSearchQuery(q);
  }, []);

  // Collect all unique categories
  const categories = useMemo(() => {
    if (!documents) return [];
    const set = new Set<string>();
    for (const doc of documents) {
      for (const cat of doc.categories) {
        set.add(cat);
      }
    }
    return Array.from(set).sort();
  }, [documents]);

  // Filter documents by search query and category
  const filtered = useMemo(() => {
    if (!documents) return [];
    let result = documents;

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(
        (d) =>
          d.title.toLowerCase().includes(q) ||
          d.summary.toLowerCase().includes(q),
      );
    }

    if (selectedCategory) {
      result = result.filter((d) => d.categories.includes(selectedCategory));
    }

    return result;
  }, [documents, searchQuery, selectedCategory]);

  if (error) {
    return (
      <div className="mx-auto max-w-5xl">
        <h1 className="text-2xl font-bold text-gray-900">Documents</h1>
        <div className="mt-4 rounded-md bg-red-50 p-4 text-sm text-red-700" role="alert">
          Failed to load documents: {error}
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Documents</h1>
        <p className="mt-1 text-sm text-gray-500">
          Browse and search your knowledge base articles.
        </p>
      </div>

      {/* Search and filter */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
        <SearchBar onChange={handleSearch} className="flex-1" />

        {categories.length > 0 && (
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setSelectedCategory(null)}
              className={`badge cursor-pointer transition-colors ${
                selectedCategory === null
                  ? "bg-brand-100 text-brand-700"
                  : "bg-gray-100 text-gray-500 hover:bg-gray-200"
              }`}
              aria-pressed={selectedCategory === null}
            >
              All
            </button>
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() =>
                  setSelectedCategory(
                    selectedCategory === cat ? null : cat,
                  )
                }
                className={`badge cursor-pointer transition-colors ${
                  selectedCategory === cat
                    ? "bg-brand-100 text-brand-700"
                    : "bg-gray-100 text-gray-500 hover:bg-gray-200"
                }`}
                aria-pressed={selectedCategory === cat}
              >
                {cat}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Loading skeleton */}
      {loading && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="card animate-pulse">
              <div className="h-4 w-3/4 rounded bg-gray-200" />
              <div className="mt-2 h-3 w-full rounded bg-gray-200" />
              <div className="mt-1 h-3 w-2/3 rounded bg-gray-200" />
            </div>
          ))}
        </div>
      )}

      {/* Documents grid */}
      {!loading && filtered.length > 0 && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((doc) => (
            <ArticleCard key={doc.slug} article={doc} />
          ))}
        </div>
      )}

      {/* Empty state */}
      {!loading && filtered.length === 0 && documents && (
        <div className="text-center">
          <p className="text-sm text-gray-400">
            {searchQuery || selectedCategory
              ? "No articles match your filters."
              : "No articles yet. Ingest some content to get started."}
          </p>
        </div>
      )}
    </div>
  );
}
