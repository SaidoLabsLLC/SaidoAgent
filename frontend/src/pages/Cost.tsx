/**
 * Cost dashboard showing token usage and estimated costs.
 *
 * This is a placeholder that displays stats-based information.
 * A production version would integrate with a cost tracking API.
 */

import { useStats } from "../hooks/useApi";

export default function Cost() {
  const { data: stats, loading } = useStats();

  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Cost Dashboard</h1>
        <p className="mt-1 text-sm text-gray-500">
          Monitor token usage and estimated costs across your knowledge base.
        </p>
      </div>

      {/* Usage overview */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="card">
          <p className="text-sm text-gray-500">Knowledge Store Size</p>
          {loading ? (
            <div className="mt-2 h-8 w-24 animate-pulse rounded bg-gray-200" />
          ) : (
            <p className="mt-1 text-2xl font-semibold text-gray-900">
              {stats
                ? `${(stats.total_size_bytes / 1024).toFixed(1)} KB`
                : "--"}
            </p>
          )}
        </div>

        <div className="card">
          <p className="text-sm text-gray-500">Total Documents</p>
          {loading ? (
            <div className="mt-2 h-8 w-24 animate-pulse rounded bg-gray-200" />
          ) : (
            <p className="mt-1 text-2xl font-semibold text-gray-900">
              {stats?.document_count ?? "--"}
            </p>
          )}
        </div>

        <div className="card">
          <p className="text-sm text-gray-500">Categories</p>
          {loading ? (
            <div className="mt-2 h-8 w-24 animate-pulse rounded bg-gray-200" />
          ) : (
            <p className="mt-1 text-2xl font-semibold text-gray-900">
              {stats?.category_count ?? "--"}
            </p>
          )}
        </div>

        <div className="card">
          <p className="text-sm text-gray-500">Concepts Indexed</p>
          {loading ? (
            <div className="mt-2 h-8 w-24 animate-pulse rounded bg-gray-200" />
          ) : (
            <p className="mt-1 text-2xl font-semibold text-gray-900">
              {stats?.concept_count ?? "--"}
            </p>
          )}
        </div>
      </div>

      {/* Cost estimation note */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900">
          Cost Tracking
        </h2>
        <p className="mt-2 text-sm text-gray-600">
          Detailed token usage and cost tracking per query is available in the
          API response. Each query response includes a{" "}
          <code className="rounded bg-gray-100 px-1 text-xs">tokens_used</code>{" "}
          field and{" "}
          <code className="rounded bg-gray-100 px-1 text-xs">provider</code>{" "}
          identifier.
        </p>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-sm" role="table">
            <thead>
              <tr className="border-b border-gray-200 text-left">
                <th className="pb-2 pr-4 font-medium text-gray-500">
                  Provider
                </th>
                <th className="pb-2 pr-4 font-medium text-gray-500">
                  Input Cost
                </th>
                <th className="pb-2 font-medium text-gray-500">
                  Output Cost
                </th>
              </tr>
            </thead>
            <tbody className="text-gray-600">
              <tr className="border-b border-gray-100">
                <td className="py-2 pr-4">Local (Ollama)</td>
                <td className="py-2 pr-4">Free</td>
                <td className="py-2">Free</td>
              </tr>
              <tr className="border-b border-gray-100">
                <td className="py-2 pr-4">OpenAI GPT-4</td>
                <td className="py-2 pr-4">$0.03 / 1K tokens</td>
                <td className="py-2">$0.06 / 1K tokens</td>
              </tr>
              <tr>
                <td className="py-2 pr-4">Anthropic Claude</td>
                <td className="py-2 pr-4">$0.015 / 1K tokens</td>
                <td className="py-2">$0.075 / 1K tokens</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-gray-400">
          Costs are approximate and based on published pricing as of early 2025.
        </p>
      </div>
    </div>
  );
}
