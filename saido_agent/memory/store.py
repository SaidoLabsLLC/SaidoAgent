"""File-based memory storage with user-level and project-level scopes.

Storage layout:
  user scope    : ~/.saido_agent/memory/<slug>.md
  project scope : .saido_agent/memory/<slug>.md  (relative to cwd)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path


log = logging.getLogger(__name__)

# -- Paths --

USER_MEMORY_DIR = Path.home() / ".saido_agent" / "memory"
TRUSTED_PROJECTS_FILE = Path.home() / ".saido_agent" / "trusted_projects.json"
INDEX_FILENAME = "MEMORY.md"

MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25_000


# -- MED-4: Memory trust boundary --

def _load_trusted_projects() -> list[str]:
    """Load list of trusted project directories."""
    if not TRUSTED_PROJECTS_FILE.exists():
        return []
    try:
        data = json.loads(TRUSTED_PROJECTS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_trusted_projects(trusted: list[str]) -> None:
    """Persist trusted project directories."""
    TRUSTED_PROJECTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRUSTED_PROJECTS_FILE.write_text(
        json.dumps(sorted(set(trusted)), indent=2), encoding="utf-8"
    )


def is_trusted_project(path: str | Path) -> bool:
    """Check whether a project directory has been explicitly trusted."""
    resolved = str(Path(path).resolve())
    return resolved in _load_trusted_projects()


def trust_project(path: str | Path) -> None:
    """Mark a project directory as trusted for memory loading."""
    resolved = str(Path(path).resolve())
    trusted = _load_trusted_projects()
    if resolved not in trusted:
        trusted.append(resolved)
        _save_trusted_projects(trusted)
        log.info("Trusted project directory: %s", resolved)


def check_project_trust(path: str | Path, *, prompt_fn=None) -> bool:
    """Verify trust for a project directory, prompting if needed.

    Args:
        path: Project directory to check.
        prompt_fn: Callable that takes a message string and returns bool
            (True = user approves). If None, returns False for untrusted dirs.

    Returns:
        True if the directory is trusted (or was just approved by the user).
    """
    resolved = Path(path).resolve()
    resolved_str = str(resolved)

    # User-level memory is always trusted
    user_dir = Path.home() / ".saido_agent"
    if str(resolved).startswith(str(user_dir)):
        return True

    if is_trusted_project(resolved_str):
        return True

    # Directory not yet trusted — prompt
    msg = (
        f"Loading project memories from {resolved_str}. "
        f"This directory was not created by Saido Agent. Continue? [y/N]"
    )

    if prompt_fn is not None:
        approved = prompt_fn(msg)
    else:
        approved = False

    if approved:
        trust_project(resolved_str)
        return True

    log.warning("Untrusted project memory directory rejected: %s", resolved_str)
    return False


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


def load_entries(scope: str = "user", *, prompt_fn=None) -> list[MemoryEntry]:
    mem_dir = get_memory_dir(scope)
    if not mem_dir.exists():
        return []

    # MED-4: Trust boundary for project-scoped memories
    if scope == "project":
        # The project root is the current working directory
        project_root = Path.cwd()
        if not check_project_trust(project_root, prompt_fn=prompt_fn):
            log.warning("Skipping untrusted project memories in %s", mem_dir)
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
