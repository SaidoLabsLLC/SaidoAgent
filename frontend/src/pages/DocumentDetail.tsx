/**
 * Single article detail view with full body content and metadata.
 */

import { Link, useParams } from "react-router-dom";
import { useDocument } from "../hooks/useApi";

export default function DocumentDetail() {
  const { slug } = useParams<{ slug: string }>();
  const { data: doc, loading, error } = useDocument(slug);

  if (loading) {
    return (
      <div className="mx-auto max-w-3xl animate-pulse space-y-4">
        <div className="h-8 w-3/4 rounded bg-gray-200" />
        <div className="h-4 w-1/2 rounded bg-gray-200" />
        <div className="space-y-2">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="h-3 w-full rounded bg-gray-200" />
          ))}
        </div>
      </div>
    );
  }

  if (error || !doc) {
    return (
      <div className="mx-auto max-w-3xl">
        <div
          className="rounded-md bg-red-50 p-4 text-sm text-red-700"
          role="alert"
        >
          {error || "Document not found."}
        </div>
        <Link to="/documents" className="mt-4 inline-block text-sm text-brand-600 hover:underline">
          Back to documents
        </Link>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl">
      {/* Breadcrumb */}
      <nav className="mb-4 text-sm" aria-label="Breadcrumb">
        <Link to="/documents" className="text-brand-600 hover:underline">
          Documents
        </Link>
        <span className="mx-2 text-gray-400">/</span>
        <span className="text-gray-500">{doc.title}</span>
      </nav>

      {/* Header */}
      <header className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">{doc.title}</h1>
        {doc.summary && (
          <p className="mt-2 text-sm text-gray-500">{doc.summary}</p>
        )}
        {doc.categories.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {doc.categories.map((cat) => (
              <span key={cat} className="badge bg-brand-50 text-brand-700">
                {cat}
              </span>
            ))}
          </div>
        )}
      </header>

      {/* Body */}
      <article className="card prose prose-sm max-w-none">
        <div className="whitespace-pre-wrap text-gray-800">{doc.body}</div>
      </article>

      {/* Frontmatter metadata */}
      {Object.keys(doc.frontmatter).length > 0 && (
        <details className="mt-6">
          <summary className="cursor-pointer text-sm font-medium text-gray-500 hover:text-gray-700">
            Metadata
          </summary>
          <pre className="mt-2 overflow-x-auto rounded-md bg-gray-100 p-4 text-xs text-gray-600">
            {JSON.stringify(doc.frontmatter, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
}
