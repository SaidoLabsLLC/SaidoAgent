"""Knowledge lint -- validates knowledge graph consistency and quality.

Runs deterministic checks (dead links, orphans, stale articles) and
LLM-powered checks (duplicates, contradictions, knowledge gaps, missing
data) against the knowledge store via KnowledgeBridge.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Wikilink regex (same as bridge.py for consistency)
# ---------------------------------------------------------------------------
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:#[^\]|]*)?(?:\|[^\]]+)?\]\]")

# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

_DUPLICATE_PROMPT = """You are a knowledge base quality checker.

Given these two articles, determine if they cover substantially the same topic.

Article A — "{title_a}":
{body_a}

Article B — "{title_b}":
{body_b}

Respond with ONLY a JSON object:
{{"duplicate": true/false, "reason": "brief explanation"}}
"""

_CONTRADICTION_PROMPT = """You are a knowledge base quality checker.

Given these two related articles, identify any contradictions between them.

Article A — "{title_a}":
{body_a}

Article B — "{title_b}":
{body_b}

Respond with ONLY a JSON object:
{{"contradicts": true/false, "description": "brief explanation of the contradiction, or empty string if none"}}
"""

_MISSING_DATA_PROMPT = """You are a knowledge base quality checker.

Analyze this article and identify claims made without supporting evidence,
citations, or data. Only flag significant factual claims, not opinions or
general statements.

Article — "{title}":
{body}

Respond with ONLY a JSON array of strings, each describing a claim lacking evidence.
Example: ["Claim X is stated without citation", "Statistic Y has no source"]
If all claims are well-supported, respond with an empty array: []
"""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _extract_json(raw: str) -> Any:
    """Extract and parse JSON from an LLM response."""
    raw = raw.strip()
    fence_match = _JSON_FENCE_RE.search(raw)
    if fence_match:
        raw = fence_match.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    brace_start = raw.find("{")
    bracket_start = raw.find("[")
    # Pick whichever comes first
    starts = [(brace_start, "{", "}"), (bracket_start, "[", "]")]
    starts = [(i, o, c) for i, o, c in starts if i != -1]
    if not starts:
        return None
    starts.sort()
    idx, _, close = starts[0]
    end = raw.rfind(close)
    if end > idx:
        try:
            return json.loads(raw[idx : end + 1])
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# LintReport
# ---------------------------------------------------------------------------


@dataclass
class LintReport:
    """Aggregated results from all lint checks."""

    contradictions: list[tuple] = field(default_factory=list)
    missing_data: list[tuple] = field(default_factory=list)
    knowledge_gaps: list[tuple] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    stale: list[tuple] = field(default_factory=list)
    dead_links: list[tuple] = field(default_factory=list)
    duplicates: list[tuple] = field(default_factory=list)
    overall_health: str = "healthy"

    def issue_count(self) -> int:
        return sum(
            len(getattr(self, f))
            for f in [
                "contradictions",
                "missing_data",
                "knowledge_gaps",
                "orphans",
                "stale",
                "dead_links",
                "duplicates",
            ]
        )

    def compute_health(self) -> str:
        """Compute and set overall_health based on issue count."""
        count = self.issue_count()
        if count <= 2:
            self.overall_health = "healthy"
        elif count <= 10:
            self.overall_health = "needs attention"
        else:
            self.overall_health = "unhealthy"
        return self.overall_health

    def to_dict(self) -> dict[str, Any]:
        """Serialize report for JSON storage."""
        return {
            "contradictions": [
                {"article_a": a, "article_b": b, "description": d}
                for a, b, d in self.contradictions
            ],
            "missing_data": [
                {"article": a, "missing_item": m} for a, m in self.missing_data
            ],
            "knowledge_gaps": [
                {"topic": t, "referencing_articles": r}
                for t, r in self.knowledge_gaps
            ],
            "orphans": self.orphans,
            "stale": [
                {"article": a, "last_updated": u} for a, u in self.stale
            ],
            "dead_links": [
                {"article": a, "broken_link": l} for a, l in self.dead_links
            ],
            "duplicates": [
                {"article_a": a, "article_b": b, "reason": r}
                for a, b, r in self.duplicates
            ],
            "overall_health": self.overall_health,
            "issue_count": self.issue_count(),
        }


# ---------------------------------------------------------------------------
# KnowledgeLinter
# ---------------------------------------------------------------------------


class KnowledgeLinter:
    """Runs health checks on the knowledge store."""

    def __init__(
        self,
        bridge: Any,
        model_router: Any = None,
    ) -> None:
        self._bridge = bridge
        self._router = model_router

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lint(self, scope: str = "all") -> LintReport:
        """Run all health checks.

        Args:
            scope: ``"all"`` for full store, or a specific article slug.

        Returns:
            Aggregated ``LintReport``.
        """
        report = LintReport()

        if scope != "all":
            # Scoped lint: only checks relevant to a single article
            report.dead_links = self._check_dead_links_for_slug(scope)
            report.stale = self.check_stale(slug=scope)
            if self._router is not None:
                report.missing_data = self.check_missing_data(slug=scope)
            report.compute_health()
            self._save_history(report, scope=scope)
            return report

        # Full store lint
        report.dead_links = self.check_dead_links()
        report.orphans = self.check_orphans()
        report.stale = self.check_stale()
        report.knowledge_gaps = self.check_knowledge_gaps()

        if self._router is not None:
            report.duplicates = self.check_duplicates()
            report.contradictions = self.check_contradictions()
            report.missing_data = self.check_missing_data()

        report.compute_health()
        self._save_history(report, scope="all")
        return report

    # ------------------------------------------------------------------
    # Deterministic checks (no LLM)
    # ------------------------------------------------------------------

    def check_dead_links(self) -> list[tuple]:
        """Find ``[[wikilinks]]`` pointing to non-existent articles."""
        if not self._bridge.available:
            return []

        articles = self._bridge.list_articles()
        existing_slugs = {slug for slug, _, _ in articles}
        dead: list[tuple] = []

        for slug, _, _ in articles:
            doc = self._bridge.read_article(slug)
            if doc is None:
                continue
            body = getattr(doc, "body", "") or ""
            linked = {m.group(1) for m in _WIKILINK_RE.finditer(body)}
            for link in linked:
                if link not in existing_slugs:
                    dead.append((slug, link))

        return dead

    def _check_dead_links_for_slug(self, slug: str) -> list[tuple]:
        """Check dead links for a single article."""
        if not self._bridge.available:
            return []

        doc = self._bridge.read_article(slug)
        if doc is None:
            return []

        articles = self._bridge.list_articles()
        existing_slugs = {s for s, _, _ in articles}
        body = getattr(doc, "body", "") or ""
        linked = {m.group(1) for m in _WIKILINK_RE.finditer(body)}
        return [(slug, link) for link in linked if link not in existing_slugs]

    def check_orphans(self) -> list[str]:
        """Find articles with zero inbound backlinks."""
        if not self._bridge.available:
            return []

        articles = self._bridge.list_articles()
        all_slugs = {slug for slug, _, _ in articles}

        # Build inbound link map
        inbound: dict[str, int] = {slug: 0 for slug in all_slugs}

        for slug, _, _ in articles:
            doc = self._bridge.read_article(slug)
            if doc is None:
                continue
            body = getattr(doc, "body", "") or ""
            linked = {m.group(1) for m in _WIKILINK_RE.finditer(body)}
            for link in linked:
                if link in inbound:
                    inbound[link] += 1

        return [slug for slug, count in inbound.items() if count == 0]

    def check_stale(
        self, days: int = 90, slug: str | None = None
    ) -> list[tuple]:
        """Find articles not updated in *days* days.

        Args:
            days: Threshold in days.
            slug: If provided, check only this article.
        """
        if not self._bridge.available:
            return []

        now = datetime.now(timezone.utc)
        stale: list[tuple] = []

        if slug is not None:
            slugs_to_check = [(slug,)]
        else:
            slugs_to_check = [
                (s,) for s, _, _ in self._bridge.list_articles()
            ]

        for (art_slug,) in slugs_to_check:
            fm = self._bridge.read_article_frontmatter(art_slug)
            if fm is None:
                continue
            updated_raw = fm.get("updated") or fm.get("created")
            if updated_raw is None:
                continue
            updated = self._parse_date(updated_raw)
            if updated is None:
                continue
            age_days = (now - updated).days
            if age_days > days:
                stale.append((art_slug, str(updated_raw)))

        return stale

    # ------------------------------------------------------------------
    # LLM-powered checks
    # ------------------------------------------------------------------

    def check_duplicates(self) -> list[tuple]:
        """Find articles covering substantially the same topic (LLM-powered)."""
        if not self._bridge.available:
            return []

        articles = self._bridge.list_articles()
        if len(articles) < 2:
            return []

        # Build concept sets for overlap detection
        concept_map: dict[str, set[str]] = {}
        for slug, _, _ in articles:
            fm = self._bridge.read_article_frontmatter(slug)
            if fm and "concepts" in fm:
                concept_map[slug] = set(fm["concepts"])
            else:
                concept_map[slug] = set()

        # Find candidate pairs: articles sharing concepts
        slugs = [s for s, _, _ in articles]
        candidates: list[tuple[str, str]] = []
        for i in range(len(slugs)):
            for j in range(i + 1, len(slugs)):
                a, b = slugs[i], slugs[j]
                overlap = concept_map.get(a, set()) & concept_map.get(b, set())
                if len(overlap) >= 2:
                    candidates.append((a, b))

        duplicates: list[tuple] = []
        for slug_a, slug_b in candidates:
            doc_a = self._bridge.read_article(slug_a)
            doc_b = self._bridge.read_article(slug_b)
            if doc_a is None or doc_b is None:
                continue

            fm_a = self._bridge.read_article_frontmatter(slug_a) or {}
            fm_b = self._bridge.read_article_frontmatter(slug_b) or {}
            title_a = fm_a.get("title", slug_a)
            title_b = fm_b.get("title", slug_b)

            prompt = _DUPLICATE_PROMPT.format(
                title_a=title_a,
                body_a=(getattr(doc_a, "body", "") or "")[:2000],
                title_b=title_b,
                body_b=(getattr(doc_b, "body", "") or "")[:2000],
            )

            response = self._call_llm(prompt)
            if response is None:
                continue

            parsed = _extract_json(response)
            if isinstance(parsed, dict) and parsed.get("duplicate"):
                reason = parsed.get("reason", "Covers the same topic")
                duplicates.append((slug_a, slug_b, reason))

        return duplicates

    def check_contradictions(
        self, article_pairs: list | None = None
    ) -> list[tuple]:
        """Find contradictions between related articles (LLM-powered).

        Args:
            article_pairs: Optional list of ``(slug_a, slug_b)`` pairs.
                If None, automatically selects pairs sharing categories/concepts.
        """
        if not self._bridge.available:
            return []

        if article_pairs is None:
            article_pairs = self._find_related_pairs()

        contradictions: list[tuple] = []
        for slug_a, slug_b in article_pairs:
            doc_a = self._bridge.read_article(slug_a)
            doc_b = self._bridge.read_article(slug_b)
            if doc_a is None or doc_b is None:
                continue

            fm_a = self._bridge.read_article_frontmatter(slug_a) or {}
            fm_b = self._bridge.read_article_frontmatter(slug_b) or {}
            title_a = fm_a.get("title", slug_a)
            title_b = fm_b.get("title", slug_b)

            prompt = _CONTRADICTION_PROMPT.format(
                title_a=title_a,
                body_a=(getattr(doc_a, "body", "") or "")[:2000],
                title_b=title_b,
                body_b=(getattr(doc_b, "body", "") or "")[:2000],
            )

            response = self._call_llm(prompt)
            if response is None:
                continue

            parsed = _extract_json(response)
            if isinstance(parsed, dict) and parsed.get("contradicts"):
                desc = parsed.get("description", "Contradiction detected")
                contradictions.append((slug_a, slug_b, desc))

        return contradictions

    def check_knowledge_gaps(self) -> list[tuple]:
        """Find concepts referenced in multiple articles but without a dedicated article.

        A concept is a "gap" when it appears in 2+ articles' concept lists but
        no article slug matches the concept name.
        """
        if not self._bridge.available:
            return []

        articles = self._bridge.list_articles()
        existing_slugs = {slug for slug, _, _ in articles}

        # Count concept occurrences across articles
        concept_articles: dict[str, list[str]] = {}
        for slug, _, _ in articles:
            fm = self._bridge.read_article_frontmatter(slug)
            if fm is None:
                continue
            for concept in fm.get("concepts", []):
                concept_lower = concept.lower().replace(" ", "-")
                concept_articles.setdefault(concept, []).append(slug)

        gaps: list[tuple] = []
        for concept, referencing in concept_articles.items():
            if len(referencing) < 2:
                continue
            # Check if a dedicated article exists (slug matches concept)
            concept_slug = concept.lower().replace(" ", "-")
            if concept_slug not in existing_slugs:
                gaps.append((concept, referencing))

        return gaps

    def check_missing_data(self, slug: str | None = None) -> list[tuple]:
        """Find claims stated without supporting evidence (LLM-powered).

        Args:
            slug: If provided, check only this article. Otherwise check all.
        """
        if not self._bridge.available:
            return []

        if slug is not None:
            slugs_to_check = [slug]
        else:
            slugs_to_check = [s for s, _, _ in self._bridge.list_articles()]

        missing: list[tuple] = []
        for art_slug in slugs_to_check:
            doc = self._bridge.read_article(art_slug)
            if doc is None:
                continue

            fm = self._bridge.read_article_frontmatter(art_slug) or {}
            title = fm.get("title", art_slug)
            body = getattr(doc, "body", "") or ""
            if not body.strip():
                continue

            prompt = _MISSING_DATA_PROMPT.format(
                title=title,
                body=body[:3000],
            )

            response = self._call_llm(prompt)
            if response is None:
                continue

            parsed = _extract_json(response)
            if isinstance(parsed, list):
                for item in parsed:
                    missing.append((art_slug, str(item)))

        return missing

    # ------------------------------------------------------------------
    # Lint history persistence
    # ------------------------------------------------------------------

    def _save_history(self, report: LintReport, scope: str = "all") -> None:
        """Append lint results to ``knowledge/saido/lint_history.json``."""
        try:
            root = self._bridge._root if hasattr(self._bridge, "_root") else None
            if root is None:
                return
            hist_path = Path(root) / "saido" / "lint_history.json"
            hist_path.parent.mkdir(parents=True, exist_ok=True)

            entries: list[dict] = []
            if hist_path.exists():
                try:
                    entries = json.loads(hist_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, ValueError):
                    entries = []

            entries.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "scope": scope,
                    "issue_count": report.issue_count(),
                    "overall_health": report.overall_health,
                    "report": report.to_dict(),
                }
            )

            hist_path.write_text(
                json.dumps(entries, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Failed to save lint history: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_related_pairs(self) -> list[tuple[str, str]]:
        """Find article pairs that share categories or concepts."""
        articles = self._bridge.list_articles()
        category_map: dict[str, set[str]] = {}
        concept_map: dict[str, set[str]] = {}

        for slug, _, _ in articles:
            fm = self._bridge.read_article_frontmatter(slug)
            if fm is None:
                continue
            for cat in fm.get("categories", []):
                category_map.setdefault(cat, set()).add(slug)
            for concept in fm.get("concepts", []):
                concept_map.setdefault(concept, set()).add(slug)

        # Collect unique pairs sharing categories or concepts
        pairs: set[tuple[str, str]] = set()
        for group in list(category_map.values()) + list(concept_map.values()):
            slugs = sorted(group)
            for i in range(len(slugs)):
                for j in range(i + 1, len(slugs)):
                    pairs.add((slugs[i], slugs[j]))

        return list(pairs)

    def _call_llm(self, prompt: str) -> str | None:
        """Send a prompt to the LLM via ModelRouter.

        Follows the same provider pattern as WikiCompiler._call_llm.
        """
        if self._router is None:
            logger.warning("No ModelRouter configured -- cannot call LLM")
            return None

        try:
            provider, model = self._router.select_model("lint")
        except Exception as exc:
            logger.error("Model selection failed: %s", exc)
            return None

        try:
            from saido_agent.core.providers import (
                PROVIDERS,
                bare_model,
                get_api_key,
                stream_anthropic,
                stream_openai_compat,
            )

            prov_config = PROVIDERS.get(provider, PROVIDERS.get("openai", {}))
            api_key = get_api_key(provider, {})
            model_name = bare_model(model)

            messages = [{"role": "user", "content": prompt}]
            config: dict[str, Any] = {"max_tokens": 1024, "no_tools": True}

            if prov_config.get("type") == "anthropic":
                gen = stream_anthropic(
                    api_key=api_key,
                    model=model_name,
                    system="You are a knowledge base quality checker. Respond with JSON only.",
                    messages=messages,
                    tool_schemas=[],
                    config=config,
                )
            else:
                base_url = prov_config.get(
                    "base_url", "http://localhost:11434/v1"
                )
                gen = stream_openai_compat(
                    api_key=api_key or "dummy",
                    base_url=base_url,
                    model=model_name,
                    system="You are a knowledge base quality checker. Respond with JSON only.",
                    messages=messages,
                    tool_schemas=[],
                    config=config,
                )

            text_parts: list[str] = []
            for chunk in gen:
                from saido_agent.core.providers import AssistantTurn, TextChunk

                if isinstance(chunk, TextChunk):
                    text_parts.append(chunk.text)
                elif isinstance(chunk, AssistantTurn):
                    if chunk.text:
                        return chunk.text
                    return "".join(text_parts) if text_parts else None

            return "".join(text_parts) if text_parts else None

        except Exception as exc:
            logger.error("LLM call failed for lint: %s", exc)
            return None

    @staticmethod
    def _parse_date(value: Any) -> datetime | None:
        """Parse a date string or datetime into a timezone-aware datetime."""
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        if not isinstance(value, str):
            return None
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(value, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        return None
