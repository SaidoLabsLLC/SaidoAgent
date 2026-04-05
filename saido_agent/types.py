"""Public type definitions for Saido Agent SDK.

All public dataclasses used in the SaidoAgent API surface are defined here.
Internal modules should not be imported directly by SDK consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IngestResult:
    """Result of ingesting a file or directory into the knowledge store."""

    slug: str
    title: str
    status: str  # "created" | "updated" | "duplicate" | "split" | "failed"
    children: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class SearchResult:
    """A single search hit from the knowledge store."""

    slug: str
    title: str
    summary: str
    score: float
    snippet: str


@dataclass
class AgentResult:
    """Result of a full agent loop execution."""

    output: str
    tool_calls: list[dict] = field(default_factory=list)
    tokens_used: int = 0


@dataclass
class CompileResult:
    """Result of compiling/enriching a knowledge article."""

    slug: str
    status: str  # "compiled" | "failed" | "skipped"
    summary: str = ""
    concepts: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class StoreStats:
    """Knowledge store statistics."""

    document_count: int
    category_count: int = 0
    concept_count: int = 0
    total_size_bytes: int = 0


# Re-exports from query.py for convenience — these are also part of the
# public API surface.  Import them here so consumers can do:
#   from saido_agent.types import SaidoQueryResult, Citation
from saido_agent.knowledge.query import Citation, SaidoQueryResult

__all__ = [
    "AgentResult",
    "Citation",
    "CompileResult",
    "IngestResult",
    "SaidoQueryResult",
    "SearchResult",
    "StoreStats",
]
