"""File-based memory storage with user-level and project-level scopes.

Storage layout:
  user scope    : ~/.saido_agent/memory/<slug>.md
  project scope : .saido_agent/memory/<slug>.md  (relative to cwd)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# -- Paths --

USER_MEMORY_DIR = Path.home() / ".saido_agent" / "memory"
INDEX_FILENAME = "MEMORY.md"

MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25_000


def get_project_memory_dir() -> Path:
    return Path.cwd() / ".saido_agent" / "memory"


def get_memory_dir(scope: str = "user") -> Path:
    if scope == "project":
        return get_project_memory_dir()
    return USER_MEMORY_DIR


# -- Data model --

@dataclass
class MemoryEntry:
    name: str
    description: str
    type: str
    content: str
    file_path: str = ""
    created: str = ""
    scope: str = "user"


# -- Helpers --

def _slugify(name: str) -> str:
    s = name.lower().strip().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s[:60]


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta: dict = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()
    return meta, parts[2].strip()


def _format_entry_md(entry: MemoryEntry) -> str:
    return (
        f"---\n"
        f"name: {entry.name}\n"
        f"description: {entry.description}\n"
        f"type: {entry.type}\n"
        f"created: {entry.created}\n"
        f"---\n"
        f"{entry.content}\n"
    )


# -- Core storage operations --

def save_memory(entry: MemoryEntry, scope: str = "user") -> None:
    mem_dir = get_memory_dir(scope)
    mem_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(entry.name)
    fp = mem_dir / f"{slug}.md"
    fp.write_text(_format_entry_md(entry))
    entry.file_path = str(fp)
    entry.scope = scope
    _rewrite_index(scope)


def delete_memory(name: str, scope: str = "user") -> None:
    mem_dir = get_memory_dir(scope)
    slug = _slugify(name)
    fp = mem_dir / f"{slug}.md"
    if fp.exists():
        fp.unlink()
    _rewrite_index(scope)


def load_entries(scope: str = "user") -> list[MemoryEntry]:
    mem_dir = get_memory_dir(scope)
    if not mem_dir.exists():
        return []
    entries: list[MemoryEntry] = []
    for fp in sorted(mem_dir.glob("*.md")):
        if fp.name == INDEX_FILENAME:
            continue
        try:
            text = fp.read_text()
        except Exception:
            continue
        meta, body = parse_frontmatter(text)
        entries.append(MemoryEntry(
            name=meta.get("name", fp.stem),
            description=meta.get("description", ""),
            type=meta.get("type", "user"),
            content=body,
            file_path=str(fp),
            created=meta.get("created", ""),
            scope=scope,
        ))
    return entries


def load_index(scope: str = "all") -> list[MemoryEntry]:
    if scope == "all":
        return load_entries("user") + load_entries("project")
    return load_entries(scope)


def search_memory(query: str, scope: str = "all") -> list[MemoryEntry]:
    q = query.lower()
    results = []
    for entry in load_index(scope):
        haystack = f"{entry.name} {entry.description} {entry.content}".lower()
        if q in haystack:
            results.append(entry)
    return results


def _rewrite_index(scope: str) -> None:
    mem_dir = get_memory_dir(scope)
    if not mem_dir.exists():
        return
    index_path = mem_dir / INDEX_FILENAME
    entries = load_entries(scope)
    lines = [
        f"- [{e.name}]({Path(e.file_path).name}) -- {e.description}"
        for e in entries
    ]
    index_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def get_memory_by_id(memory_id: str, scope: str = "all") -> MemoryEntry | None:
    """Look up a memory by its slug (id). Returns None if not found."""
    scopes = ["user", "project"] if scope == "all" else [scope]
    for s in scopes:
        mem_dir = get_memory_dir(s)
        fp = mem_dir / f"{memory_id}.md"
        if fp.exists():
            try:
                text = fp.read_text()
            except Exception:
                return None
            meta, body = parse_frontmatter(text)
            return MemoryEntry(
                name=meta.get("name", fp.stem),
                description=meta.get("description", ""),
                type=meta.get("type", "user"),
                content=body,
                file_path=str(fp),
                created=meta.get("created", ""),
                scope=s,
            )
    return None


def to_knowledge_article(entry: MemoryEntry) -> dict:
    """Convert a MemoryEntry into a knowledge-store-compatible article dict.

    This prepares the data structure for future KnowledgeBridge integration.
    The actual wiring happens when the knowledge bridge is complete.
    """
    return {
        "type": "memory",
        "category": "agent-memory",
        "title": entry.name,
        "description": entry.description,
        "content": entry.content,
        "metadata": {
            "memory_type": entry.type,
            "scope": entry.scope,
            "created": entry.created,
            "source_file": entry.file_path,
        },
        "tags": [
            f"scope:{entry.scope}",
            f"memory-type:{entry.type}",
        ],
    }


def get_index_content(scope: str = "user") -> str:
    mem_dir = get_memory_dir(scope)
    index_path = mem_dir / INDEX_FILENAME
    if not index_path.exists():
        return ""
    return index_path.read_text().strip()
