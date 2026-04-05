"""Knowledge index -- LLM-powered intelligence layer on top of SmartRAG.

SmartRAG handles master index, FTS5, and backlinks automatically.
WikiIndexer builds higher-order intelligence ON TOP:
- Enriched summaries (LLM-generated 2-3 sentence summaries)
- Concept maps (graph of concepts and their relationships)
- Category trees (hierarchical taxonomy auto-generated from frontmatter)
- Incremental indexing with content-hash tracking
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths for persisted index artifacts (relative to knowledge root)
# ---------------------------------------------------------------------------
_CONCEPT_MAP_REL = Path("saido") / "concept_map.json"
_CATEGORY_TREE_REL = Path("saido") / "category_tree.json"
_INDEX_STATE_REL = Path("saido") / "index_state.json"

# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

_SUMMARY_PROMPT = """\
You are a technical documentation summarizer.
Given the following article body, produce a 2-3 sentence summary that captures \
the key purpose, main concepts, and practical significance. Be specific and \
informative, not generic.

Article title: {title}
Article body:
{body}

Respond with ONLY a JSON object: {{"summary": "your 2-3 sentence summary"}}"""

_CONCEPT_MAP_PROMPT = """\
You are a knowledge graph architect.
Given these concepts extracted from a knowledge base, identify meaningful \
relationships between them. Each concept may relate to others via: \
"is-a", "part-of", "uses", "related-to", "depends-on", "extends", "implements".

Concepts (with article counts):
{concepts_list}

Respond with ONLY a JSON object:
{{
  "edges": [
    {{"source": "concept_id", "target": "concept_id", "relation": "relation_type"}}
  ]
}}

Only include edges where a clear, meaningful relationship exists. \
Use the concept IDs exactly as provided. Limit to the 50 most important edges."""

_CATEGORY_TREE_PROMPT = """\
You are a taxonomy designer.
Given these categories from a knowledge base, organize them into a clean \
hierarchical tree. Group related categories under broader parent categories. \
Create parent categories as needed, but keep the tree shallow (max 3 levels deep).

Categories:
{categories_list}

Respond with ONLY a JSON object:
{{
  "root": [
    {{
      "name": "Parent Category",
      "children": [
        {{"name": "Child Category", "children": ["Leaf1", "Leaf2"]}}
      ]
    }}
  ]
}}

Every input category must appear somewhere in the tree."""


# ---------------------------------------------------------------------------
# JSON extraction (reuse pattern from compile.py)
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> dict[str, Any] | None:
    """Extract and parse a JSON object from an LLM response."""
    import re

    raw = raw.strip()

    # Try extracting from code fences first
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()

    # Try direct parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try finding the first { ... } block
    brace_start = raw.find("{")
    brace_end = raw.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            parsed = json.loads(raw[brace_start : brace_end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# IndexResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class IndexResult:
    """Result of a reindex operation."""

    articles_processed: int = 0
    articles_skipped: int = 0
    concept_map_updated: bool = False
    category_tree_updated: bool = False
    duration_ms: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# WikiIndexer
# ---------------------------------------------------------------------------


class WikiIndexer:
    """LLM-powered indexing layer that builds concept maps, category trees,
    and enriched summaries on top of SmartRAG's base indexes.

    All artifacts are persisted to ``knowledge/saido/`` as JSON files.
    Incremental indexing tracks content hashes in ``index_state.json``
    so only changed articles are re-processed.
    """

    def __init__(
        self,
        bridge: Any,
        model_router: Any | None = None,
    ) -> None:
        self._bridge = bridge
        self._router = model_router
        self._root = Path(getattr(bridge, "_root", "knowledge")).resolve()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reindex(self, scope: str = "all", full: bool = False) -> IndexResult:
        """Run incremental (changed only) or full reindex.

        Args:
            scope: Reserved for future per-category scoping. Currently
                only "all" is supported.
            full: If True, re-process ALL articles regardless of hash.

        Returns:
            IndexResult with counts and timing.
        """
        start_ms = _now_ms()
        result = IndexResult()

        # Load current index state
        state = self._load_index_state()

        # Get all articles
        articles = self._bridge.list_articles()
        if not articles:
            result.duration_ms = _now_ms() - start_ms
            return result

        # Determine which articles need processing
        to_process: list[tuple[str, str]] = []  # (slug, content_hash)
        for slug, _title, _summary in articles:
            content_hash = self._compute_article_hash(slug)
            if content_hash is None:
                continue

            if full or state.get(slug) != content_hash:
                to_process.append((slug, content_hash))
            else:
                result.articles_skipped += 1

        # Generate enriched summaries for changed articles
        for slug, content_hash in to_process:
            try:
                self._generate_single_summary(slug)
                state[slug] = content_hash
                result.articles_processed += 1
            except Exception as exc:
                logger.error("Summary generation failed for %s: %s", slug, exc)
                result.errors.append(f"{slug}: {exc}")
                result.articles_processed += 1

        # Save updated index state
        self._save_index_state(state)

        # Rebuild concept map and category tree if any articles changed
        if to_process or full:
            try:
                self.generate_concept_map()
                result.concept_map_updated = True
            except Exception as exc:
                logger.error("Concept map generation failed: %s", exc)
                result.errors.append(f"concept_map: {exc}")

            try:
                self.generate_category_tree()
                result.category_tree_updated = True
            except Exception as exc:
                logger.error("Category tree generation failed: %s", exc)
                result.errors.append(f"category_tree: {exc}")

        result.duration_ms = _now_ms() - start_ms
        return result

    def generate_concept_map(self) -> dict:
        """Build graph: nodes (concepts) + edges (relationships).

        Collects all concepts from article frontmatter, uses LLM to
        analyze relationships, and persists to concept_map.json.

        Returns:
            The concept map dict with "nodes" and "edges" keys.
        """
        # Collect concepts from all articles
        concept_counts: dict[str, int] = {}
        articles = self._bridge.list_articles()

        for slug, _title, _summary in articles:
            fm = self._bridge.read_article_frontmatter(slug)
            if fm is None:
                continue
            concepts = fm.get("concepts", [])
            if not isinstance(concepts, list):
                continue
            for concept in concepts:
                c = str(concept).strip().lower()
                if c:
                    concept_counts[c] = concept_counts.get(c, 0) + 1

        # Build nodes
        nodes = [
            {"id": cid, "label": cid.replace("-", " ").title(), "articles": count}
            for cid, count in sorted(concept_counts.items())
        ]

        # Use LLM to generate edges if we have concepts and a router
        edges: list[dict[str, str]] = []
        if nodes and self._router is not None:
            concepts_text = "\n".join(
                f"- {n['id']} (appears in {n['articles']} articles)"
                for n in nodes
            )
            prompt = _CONCEPT_MAP_PROMPT.format(concepts_list=concepts_text)
            raw = self._call_llm(prompt)
            if raw:
                parsed = _extract_json(raw)
                if parsed and "edges" in parsed:
                    # Validate edges reference existing concepts
                    valid_ids = {n["id"] for n in nodes}
                    for edge in parsed["edges"]:
                        if (
                            isinstance(edge, dict)
                            and edge.get("source") in valid_ids
                            and edge.get("target") in valid_ids
                            and isinstance(edge.get("relation"), str)
                        ):
                            edges.append({
                                "source": edge["source"],
                                "target": edge["target"],
                                "relation": edge["relation"],
                            })

        concept_map = {"nodes": nodes, "edges": edges}
        self._save_json(_CONCEPT_MAP_REL, concept_map)
        return concept_map

    def generate_category_tree(self) -> dict:
        """Auto-generate hierarchical category taxonomy.

        Collects all categories from frontmatter, uses LLM to organize
        into a hierarchy, and persists to category_tree.json.

        Returns:
            The category tree dict with a "root" key.
        """
        # Collect categories from all articles
        all_categories: set[str] = set()
        articles = self._bridge.list_articles()

        for slug, _title, _summary in articles:
            fm = self._bridge.read_article_frontmatter(slug)
            if fm is None:
                continue
            categories = fm.get("categories", [])
            if isinstance(categories, list):
                for cat in categories:
                    c = str(cat).strip()
                    if c:
                        all_categories.add(c)

        if not all_categories:
            tree: dict[str, Any] = {"root": []}
            self._save_json(_CATEGORY_TREE_REL, tree)
            return tree

        # Use LLM to organize into hierarchy
        if self._router is not None:
            cats_text = "\n".join(f"- {c}" for c in sorted(all_categories))
            prompt = _CATEGORY_TREE_PROMPT.format(categories_list=cats_text)
            raw = self._call_llm(prompt)
            if raw:
                parsed = _extract_json(raw)
                if parsed and "root" in parsed and isinstance(parsed["root"], list):
                    tree = {"root": parsed["root"]}
                    self._save_json(_CATEGORY_TREE_REL, tree)
                    return tree

        # Fallback: flat list if LLM unavailable or fails
        tree = {"root": [{"name": c, "children": []} for c in sorted(all_categories)]}
        self._save_json(_CATEGORY_TREE_REL, tree)
        return tree

    def generate_enriched_summaries(self, slugs: list[str] | None = None) -> int:
        """Generate 2-3 sentence LLM summaries for articles.

        Args:
            slugs: Specific slugs to summarize. If None, all articles.

        Returns:
            Count of articles successfully summarized.
        """
        if slugs is None:
            articles = self._bridge.list_articles()
            slugs = [slug for slug, _, _ in articles]

        count = 0
        for slug in slugs:
            try:
                self._generate_single_summary(slug)
                count += 1
            except Exception as exc:
                logger.error("Summary failed for %s: %s", slug, exc)

        return count

    def load_concept_map(self) -> dict:
        """Load the persisted concept map from disk."""
        return self._load_json(_CONCEPT_MAP_REL, {"nodes": [], "edges": []})

    def load_category_tree(self) -> dict:
        """Load the persisted category tree from disk."""
        return self._load_json(_CATEGORY_TREE_REL, {"root": []})

    # ------------------------------------------------------------------
    # Internal: summary generation
    # ------------------------------------------------------------------

    def _generate_single_summary(self, slug: str) -> None:
        """Generate an enriched summary for a single article and update
        its frontmatter via the bridge."""
        doc = self._bridge.read_article(slug)
        if doc is None:
            raise ValueError(f"Article not found: {slug}")

        fm = self._bridge.read_article_frontmatter(slug) or {}
        title = fm.get("title", slug)
        body = getattr(doc, "body", "") or ""

        if not body.strip():
            return  # Nothing to summarize

        prompt = _SUMMARY_PROMPT.format(title=title, body=body[:4000])
        raw = self._call_llm(prompt)
        if raw is None:
            raise RuntimeError("LLM call returned no response")

        parsed = _extract_json(raw)
        if parsed is None or "summary" not in parsed:
            raise RuntimeError("Could not parse summary from LLM response")

        summary = str(parsed["summary"])[:300]
        self._bridge.update_article(
            slug, frontmatter_updates={"enriched_summary": summary}
        )

    # ------------------------------------------------------------------
    # Internal: content hashing
    # ------------------------------------------------------------------

    def _compute_article_hash(self, slug: str) -> str | None:
        """Compute a SHA-256 hash of an article's body for change detection."""
        doc = self._bridge.read_article(slug)
        if doc is None:
            return None
        body = getattr(doc, "body", "") or ""
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Internal: index state persistence
    # ------------------------------------------------------------------

    def _load_index_state(self) -> dict[str, str]:
        """Load slug -> content_hash mapping from disk."""
        return self._load_json(_INDEX_STATE_REL, {})

    def _save_index_state(self, state: dict[str, str]) -> None:
        """Persist slug -> content_hash mapping to disk."""
        self._save_json(_INDEX_STATE_REL, state)

    # ------------------------------------------------------------------
    # Internal: JSON persistence helpers
    # ------------------------------------------------------------------

    def _load_json(self, rel_path: Path, default: Any) -> Any:
        """Load a JSON file relative to knowledge root, returning default on failure."""
        path = self._root / rel_path
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return default

    def _save_json(self, rel_path: Path, data: Any) -> None:
        """Save a JSON file relative to knowledge root."""
        path = self._root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Internal: LLM interaction
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str | None:
        """Send a prompt to the LLM via ModelRouter.

        Uses task_type='index' for model selection. Falls back to None
        if no router is configured or the call fails.
        """
        if self._router is None:
            logger.warning("No ModelRouter configured -- cannot call LLM")
            return None

        try:
            provider, model = self._router.select_model("index")
        except Exception:
            # Fallback to compile task type if index not configured
            try:
                provider, model = self._router.select_model("compile")
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
            config: dict[str, Any] = {"max_tokens": 2048, "no_tools": True}

            if prov_config.get("type") == "anthropic":
                gen = stream_anthropic(
                    api_key=api_key,
                    model=model_name,
                    system="You are a knowledge indexer. Respond with JSON only.",
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
                    system="You are a knowledge indexer. Respond with JSON only.",
                    messages=messages,
                    tool_schemas=[],
                    config=config,
                )

            # Consume the generator to get the AssistantTurn
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
            logger.error("LLM call failed for index: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    """Current time in milliseconds."""
    return int(time.time() * 1000)
