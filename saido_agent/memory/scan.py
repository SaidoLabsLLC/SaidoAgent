"""Memory file scanning with mtime tracking and freshness/age helpers."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

from .store import get_memory_dir, parse_frontmatter, INDEX_FILENAME

MAX_MEMORY_FILES = 200


@dataclass
class MemoryHeader:
    filename: str
    file_path: str
    mtime_s: float
    description: str
    type: str
    scope: str


def scan_memory_dir(mem_dir: Path, scope: str) -> list[MemoryHeader]:
    if not mem_dir.is_dir():
        return []
    headers: list[MemoryHeader] = []
    for fp in mem_dir.glob("*.md"):
        if fp.name == INDEX_FILENAME:
            continue
        try:
            stat = fp.stat()
            lines = fp.read_text(errors="replace").splitlines()[:30]
            snippet = "\n".join(lines)
            meta, _ = parse_frontmatter(snippet)
            headers.append(MemoryHeader(
                filename=fp.name,
                file_path=str(fp),
                mtime_s=stat.st_mtime,
                description=meta.get("description", ""),
                type=meta.get("type", ""),
                scope=scope,
            ))
        except Exception:
            continue
    headers.sort(key=lambda h: h.mtime_s, reverse=True)
    return headers[:MAX_MEMORY_FILES]


def scan_all_memories() -> list[MemoryHeader]:
    user_dir = get_memory_dir("user")
    proj_dir = get_memory_dir("project")
    user_headers = scan_memory_dir(user_dir, "user")
    proj_headers = scan_memory_dir(proj_dir, "project")
    combined = user_headers + proj_headers
    combined.sort(key=lambda h: h.mtime_s, reverse=True)
    return combined[:MAX_MEMORY_FILES]


def memory_age_days(mtime_s: float) -> int:
    return max(0, math.floor((time.time() - mtime_s) / 86_400))


def memory_age_str(mtime_s: float) -> str:
    d = memory_age_days(mtime_s)
    if d == 0: return "today"
    if d == 1: return "yesterday"
    return f"{d} days ago"


def memory_freshness_text(mtime_s: float) -> str:
    d = memory_age_days(mtime_s)
    if d <= 1:
        return ""
    return (
        f"This memory is {d} days old. "
        "Memories are point-in-time observations, not live state -- "
        "claims about code behavior or file:line citations may be outdated. "
        "Verify against current code before asserting as fact."
    )


def format_memory_manifest(headers: list[MemoryHeader]) -> str:
    lines = []
    for h in headers:
        tag = f"[{h.type}/{h.scope}]" if h.type else f"[{h.scope}]"
        age = memory_age_str(h.mtime_s)
        if h.description:
            lines.append(f"- {tag} {h.filename} ({age}): {h.description}")
        else:
            lines.append(f"- {tag} {h.filename} ({age})")
    return "\n".join(lines)
