/**
 * Full chat interface with streaming responses via SSE.
 *
 * Features:
 * - Message input at bottom
 * - Streaming token-by-token responses
 * - Citations displayed as clickable links
 * - Cloud toggle button for provider routing
 */

import { type FormEvent, useCallback, useEffect, useRef, useState } from "react";
import ChatMessageComponent from "../components/ChatMessage";
import { streamQuery } from "../api/client";
import type { ChatMessage, Citation, QueryResponse } from "../types";

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [cloudMode, setCloudMode] = useState(false);
  const cancelRef = useRef<(() => void) | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = useCallback(
    (e: FormEvent) => {
      e.preventDefault();
      const question = input.trim();
      if (!question || streaming) return;

      // Add user message
      const userMsg: ChatMessage = {
        id: `user-${Date.now()}`,
        role: "user",
        content: question,
        timestamp: Date.now(),
      };

      // Add placeholder assistant message
      const assistantId = `assistant-${Date.now()}`;
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: Date.now(),
        streaming: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setInput("");
      setStreaming(true);

      // Prepend /cloud to the query if cloud mode is enabled
      const queryText = cloudMode ? `/cloud ${question}` : question;

      const cancel = streamQuery(
        queryText,
        // onToken
        (token: string) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: m.content + token }
                : m,
            ),
          );
        },
        // onDone
        (result: unknown) => {
          const qr = result as QueryResponse;
          const citations: Citation[] = qr.citations || [];
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: qr.answer, citations, streaming: false }
                : m,
            ),
          );
          setStreaming(false);
        },
        // onError
        (error: Error) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    content: `Error: ${error.message}`,
                    streaming: false,
                  }
                : m,
            ),
          );
          setStreaming(false);
        },
      );

      cancelRef.current = cancel;
    },
    [input, streaming, cloudMode],
  );

  const handleCancel = useCallback(() => {
    cancelRef.current?.();
    setStreaming(false);
    setMessages((prev) =>
      prev.map((m) => (m.streaming ? { ...m, streaming: false } : m)),
    );
  }, []);

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col">
      {/* Header */}
      <div className="flex items-center justify-between pb-4">
        <h1 className="text-2xl font-bold text-gray-900">Chat</h1>
        <button
          onClick={() => setCloudMode(!cloudMode)}
          className={`
            flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium
            transition-colors
            ${
              cloudMode
                ? "bg-brand-100 text-brand-700"
                : "bg-gray-100 text-gray-500"
            }
          `}
          aria-pressed={cloudMode}
          aria-label={`Cloud mode ${cloudMode ? "enabled" : "disabled"}`}
        >
          <svg
            className="h-4 w-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M2.25 15a4.5 4.5 0 004.5 4.5H18a3.75 3.75 0 001.332-7.257 3 3 0 00-3.758-3.848 5.25 5.25 0 00-10.233 2.33A4.502 4.502 0 002.25 15z"
            />
          </svg>
          {cloudMode ? "Cloud ON" : "Cloud OFF"}
        </button>
      </div>

      {/* Message list */}
      <div
        className="flex-1 overflow-y-auto scrollbar-thin"
        role="list"
        aria-label="Chat messages"
      >
        {messages.length === 0 && (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <svg
                className="mx-auto h-12 w-12 text-gray-300"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1}
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                />
              </svg>
              <p className="mt-4 text-sm text-gray-400">
                Ask a question about your knowledge base.
              </p>
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <ChatMessageComponent key={msg.id} message={msg} />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="border-t border-gray-200 pt-4">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question..."
            className="input flex-1"
            disabled={streaming}
            aria-label="Message input"
          />
          {streaming ? (
            <button
              type="button"
              onClick={handleCancel}
              className="btn-secondary"
              aria-label="Cancel streaming"
            >
              Stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              className="btn-primary"
              aria-label="Send message"
            >
              Send
            </button>
          )}
        </form>
      </div>
    </div>
  );
}
