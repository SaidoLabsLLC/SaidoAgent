/**
 * Chat message bubble with citation links.
 *
 * User messages render right-aligned in brand color.
 * Assistant messages render left-aligned in white with optional citations.
 */

import { Link } from "react-router-dom";
import type { ChatMessage as ChatMessageType } from "../types";

interface ChatMessageProps {
  message: ChatMessageType;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}
      role="listitem"
    >
      <div
        className={`
          max-w-[80%] rounded-lg px-4 py-3 shadow-sm
          ${
            isUser
              ? "bg-brand-600 text-white"
              : "border border-gray-200 bg-white text-gray-800"
          }
        `}
      >
        {/* Message content */}
        <p className="whitespace-pre-wrap text-sm leading-relaxed">
          {message.content}
          {message.streaming && (
            <span className="ml-1 inline-block animate-pulse" aria-label="Typing">
              |
            </span>
          )}
        </p>

        {/* Citations */}
        {message.citations && message.citations.length > 0 && (
          <div className="mt-3 border-t border-gray-100 pt-2">
            <p className="mb-1 text-xs font-medium text-gray-500">Sources:</p>
            <ul className="space-y-1">
              {message.citations.map((citation) => (
                <li key={citation.slug}>
                  <Link
                    to={`/documents/${encodeURIComponent(citation.slug)}`}
                    className="group flex items-start gap-1.5 text-xs"
                  >
                    <svg
                      className="mt-0.5 h-3 w-3 flex-shrink-0 text-brand-500"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                      aria-hidden="true"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
                      />
                    </svg>
                    <span className="text-brand-600 group-hover:underline">
                      {citation.title}
                    </span>
                    {citation.verified && (
                      <span
                        className="badge bg-green-100 text-green-700"
                        aria-label="Verified source"
                      >
                        verified
                      </span>
                    )}
                  </Link>
                  {citation.excerpt && (
                    <p className="ml-4.5 mt-0.5 text-xs text-gray-400 line-clamp-2">
                      {citation.excerpt}
                    </p>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Timestamp */}
        <time
          className={`mt-1 block text-right text-[10px] ${
            isUser ? "text-brand-200" : "text-gray-400"
          }`}
          dateTime={new Date(message.timestamp).toISOString()}
        >
          {new Date(message.timestamp).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </time>
      </div>
    </div>
  );
}
