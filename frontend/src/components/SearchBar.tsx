/**
 * Search input with keyboard shortcut support.
 *
 * Debounces input by 300ms before firing the onChange callback.
 */

import { useEffect, useRef, useState } from "react";

interface SearchBarProps {
  placeholder?: string;
  onChange: (query: string) => void;
  className?: string;
}

export default function SearchBar({
  placeholder = "Search knowledge base...",
  onChange,
  className = "",
}: SearchBarProps) {
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();

  // Debounce
  useEffect(() => {
    timerRef.current = setTimeout(() => {
      onChange(value);
    }, 300);
    return () => clearTimeout(timerRef.current);
  }, [value, onChange]);

  // Keyboard shortcut: Ctrl/Cmd + K focuses the search
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        inputRef.current?.focus();
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);

  return (
    <div className={`relative ${className}`}>
      <svg
        className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={2}
        aria-hidden="true"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z"
        />
      </svg>
      <input
        ref={inputRef}
        type="search"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder={placeholder}
        className="input pl-10 pr-16"
        aria-label="Search"
      />
      <kbd className="pointer-events-none absolute right-3 top-1/2 hidden -translate-y-1/2 rounded border border-gray-200 bg-gray-50 px-1.5 py-0.5 text-[10px] font-medium text-gray-400 sm:inline-block">
        Ctrl+K
      </kbd>
    </div>
  );
}
