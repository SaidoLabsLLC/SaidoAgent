/**
 * Knowledge article preview card for the documents grid/list view.
 */

import { Link } from "react-router-dom";
import type { DocumentSummary } from "../types";

interface ArticleCardProps {
  article: DocumentSummary;
}

export default function ArticleCard({ article }: ArticleCardProps) {
  return (
    <Link
      to={`/documents/${encodeURIComponent(article.slug)}`}
      className="card group block transition-shadow hover:shadow-md"
      aria-label={`View article: ${article.title}`}
    >
      <h3 className="text-base font-semibold text-gray-900 group-hover:text-brand-600 line-clamp-1">
        {article.title}
      </h3>

      <p className="mt-1 text-sm text-gray-500 line-clamp-2">
        {article.summary || "No summary available."}
      </p>

      {/* Category badges */}
      {article.categories.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {article.categories.slice(0, 4).map((cat) => (
            <span
              key={cat}
              className="badge bg-brand-50 text-brand-700"
            >
              {cat}
            </span>
          ))}
          {article.categories.length > 4 && (
            <span className="badge bg-gray-100 text-gray-500">
              +{article.categories.length - 4}
            </span>
          )}
        </div>
      )}

      {/* Last updated */}
      {article.updated && (
        <p className="mt-3 text-xs text-gray-400">
          Updated{" "}
          <time dateTime={article.updated}>
            {new Date(article.updated).toLocaleDateString()}
          </time>
        </p>
      )}
    </Link>
  );
}
