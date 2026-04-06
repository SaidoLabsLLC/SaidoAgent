"""Knowledge compilation -- enriches SmartRAG documents with LLM-generated metadata.

SmartRAG produces extractive synopses and keyword fingerprints during ingest.
WikiCompiler UPGRADES these with higher-quality, LLM-generated intelligence:
improved summaries, semantic concepts, refined categories, and backlinks.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template (loaded once from prompts/compile.md)
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).parent / "prompts" / "compile.md"

_RETRY_PROMPT = (
    "The previous response was not valid JSON. "
    "Please respond with ONLY a JSON object (no markdown fences, no commentary) "
    "containing these keys: summary, concepts, categories, backlinks, see_also."
)


def _load_prompt_template() -> str:
    """Load the compile prompt template from disk."""
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error("Compile prompt template not found at %s", _PROMPT_PATH)
        raise


# ---------------------------------------------------------------------------
# CompileResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class CompileResult:
    """Result of compiling a single document."""

    slug: str
    status: str  # "compiled", "failed", "skipped"
    summary: str = ""
    concepts: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    backlinks: list[str] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------------

# Matches ```json ... ``` or ``` ... ``` fenced blocks
_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL
)


def _extract_json(raw: str) -> dict[str, Any] | None:
    """Extract and parse a JSON object from an LLM response.

    Handles:
    - Raw JSON
    - JSON wrapped in markdown code fences
    - Leading/trailing whitespace and text
    """
    raw = raw.strip()

    # Try extracting from code fences first
    fence_match = _JSON_FENCE_RE.search(raw)
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


def _validate_compile_response(
    data: dict[str, Any],
    existing_slugs: set[str],
) -> dict[str, Any]:
    """Validate and normalize a parsed compile response.

    Ensures:
    - summary is a string, capped at 200 chars
    - concepts is a non-empty list of strings
    - categories is a list of strings
    - backlinks only reference existing slugs
    - see_also only reference existing slugs
    """
    summary = str(data.get("summary", ""))[:200]

    concepts = data.get("concepts", [])
    if not isinstance(concepts, list) or len(concepts) == 0:
        raise ValueError("concepts must be a non-empty list")
    concepts = [str(c) for c in concepts]

    categories = data.get("categories", [])
    if not isinstance(categories, list):
        categories = []
    categories = [str(c) for c in categories]

    backlinks = data.get("backlinks", [])
    if not isinstance(backlinks, list):
        backlinks = []
    backlinks = [str(b) for b in backlinks if str(b) in existing_slugs]

    see_also = data.get("see_also", [])
    if not isinstance(see_also, list):
        see_also = []
    # Extract slug from wikilink format [[slug]]
    cleaned_see_also: list[str] = []
    for sa in see_also:
        sa_str = str(sa)
        # Strip [[ and ]]
        slug_ref = sa_str.strip("[]")
        if slug_ref in existing_slugs:
            cleaned_see_also.append(f"[[{slug_ref}]]")

    return {
        "summary": summary,
        "concepts": concepts,
        "categories": categories,
        "backlinks": backlinks,
        "see_also": cleaned_see_also,
    }


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_compile_prompt(
    title: str,
    body: str,
    existing_synopsis: str,
    existing_fingerprint: str,
    existing_categories: str,
    existing_articles: list[str],
    code_structure: dict[str, Any] | None = None,
) -> str:
    """Build the compile prompt by filling the template."""
    template = _load_prompt_template()

    prompt = template.replace("{{title}}", title)
    prompt = prompt.replace("{{existing_synopsis}}", existing_synopsis)
    prompt = prompt.replace("{{existing_fingerprint}}", existing_fingerprint)
    prompt = prompt.replace("{{existing_categories}}", existing_categories)
    prompt = prompt.replace("{{document_body}}", body)
    prompt = prompt.replace(
        "{{existing_articles}}", "\n".join(f"- {a}" for a in existing_articles)
    )

    # Handle conditional code_structure block
    if code_structure:
        cs_block = (
            f"Code Structure:\n"
            f"  Language: {code_structure.get('language', 'unknown')}\n"
            f"  Functions: {code_structure.get('functions', [])}\n"
            f"  Classes: {code_structure.get('classes', [])}\n"
            f"  Endpoints: {code_structure.get('endpoints', [])}"
        )
        prompt = re.sub(
            r"\{\{#if code_structure\}\}.*?\{\{/if\}\}",
            cs_block,
            prompt,
            flags=re.DOTALL,
        )
    else:
        prompt = re.sub(
            r"\{\{#if code_structure\}\}.*?\{\{/if\}\}",
            "",
            prompt,
            flags=re.DOTALL,
        )

    return prompt


# ---------------------------------------------------------------------------
# WikiCompiler
# ---------------------------------------------------------------------------


class WikiCompiler:
    """Enriches SmartRAG documents with LLM-generated metadata.

    Uses the ModelRouter to select an LLM (defaults to local qwen3:8b
    via the ``compile`` task type) and sends a structured prompt to
    produce improved summaries, semantic concepts, categories, and
    backlinks.
    """

    def __init__(self, bridge: Any, model_router: Any = None) -> None:
        self._bridge = bridge
        self._router = model_router

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(
        self,
        slug: str,
        code_structure: dict[str, Any] | None = None,
    ) -> CompileResult:
        """Enrich a single document with LLM-generated metadata."""
        # Step a: Read document
        doc = self._bridge.read_article(slug)
        if doc is None:
            return CompileResult(
                slug=slug, status="failed", error="Document not found"
            )

        # Step b: Read existing metadata
        frontmatter = self._bridge.read_article_frontmatter(slug) or {}
        existing_synopsis = frontmatter.get("synopsis", "")
        existing_fingerprint = str(frontmatter.get("fingerprint", ""))
        existing_categories = str(frontmatter.get("categories", ""))
        title = frontmatter.get("title", slug)

        # Check for section-split children
        section_map = frontmatter.get("section_map")
        children = frontmatter.get("children", [])
        if children:
            return self._compile_parent_with_children(
                slug, children, section_map, code_structure
            )

        # Step c: Build the compile prompt
        all_articles = self._get_all_article_slugs()
        existing_slugs = {a for a in all_articles if a != slug}

        body = getattr(doc, "body", "") or ""
        prompt = _build_compile_prompt(
            title=title,
            body=body,
            existing_synopsis=existing_synopsis,
            existing_fingerprint=existing_fingerprint,
            existing_categories=existing_categories,
            existing_articles=list(existing_slugs),
            code_structure=code_structure,
        )

        # Step d: Send to LLM
        raw_response = self._call_llm(prompt)
        if raw_response is None:
            return CompileResult(
                slug=slug, status="failed", error="LLM call returned no response"
            )

        # Step e: Parse and validate
        parsed = _extract_json(raw_response)
        if parsed is None:
            # Retry with simpler prompt
            logger.warning(
                "First compile attempt for %s failed to parse, retrying", slug
            )
            retry_prompt = f"{prompt}\n\n{_RETRY_PROMPT}"
            raw_response = self._call_llm(retry_prompt)
            if raw_response is not None:
                parsed = _extract_json(raw_response)

        if parsed is None:
            logger.error(
                "Compile failed for %s: could not parse LLM response", slug
            )
            return CompileResult(
                slug=slug,
                status="failed",
                error="Could not parse LLM response after retry",
            )

        try:
            validated = _validate_compile_response(parsed, existing_slugs)
        except ValueError as exc:
            logger.error("Validation failed for %s: %s", slug, exc)
            return CompileResult(
                slug=slug, status="failed", error=f"Validation error: {exc}"
            )

        # Step f: Update article via bridge
        fm_updates: dict[str, Any] = {
            "synopsis": validated["summary"],
            "concepts": validated["concepts"],
            "categories": validated["categories"],
            "backlinks": validated["backlinks"],
            "compiled": True,
        }

        # Append see_also to body if present
        updated_body = body
        if validated["see_also"]:
            see_also_section = "\n\n## See Also\n" + "\n".join(
                f"- {sa}" for sa in validated["see_also"]
            )
            # Replace existing See Also section or append
            if "## See Also" in updated_body:
                updated_body = re.sub(
                    r"## See Also.*$",
                    see_also_section.lstrip("\n"),
                    updated_body,
                    flags=re.DOTALL,
                )
            else:
                updated_body += see_also_section

        self._bridge.update_article(
            slug,
            body=updated_body if updated_body != body else None,
            frontmatter_updates=fm_updates,
        )

        # Step g: Return result
        return CompileResult(
            slug=slug,
            status="compiled",
            summary=validated["summary"],
            concepts=validated["concepts"],
            categories=validated["categories"],
            backlinks=validated["backlinks"],
        )

    def compile_batch(
        self,
        slugs: list[str],
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[CompileResult]:
        """Process multiple documents sequentially with progress.

        Args:
            slugs: List of document slugs to compile.
            progress_callback: Optional callback(current, total, slug_title).
        """
        results: list[CompileResult] = []
        total = len(slugs)

        for idx, slug in enumerate(slugs, 1):
            # Progress reporting
            title = slug  # default to slug
            try:
                fm = self._bridge.read_article_frontmatter(slug)
                if fm and "title" in fm:
                    title = fm["title"]
            except Exception:
                pass

            if progress_callback:
                progress_callback(idx, total, title)
            else:
                logger.info("[%d/%d] Compiling: %s", idx, total, title)

            try:
                result = self.compile(slug)
                results.append(result)
            except Exception as exc:
                logger.error("Compile error for %s: %s", slug, exc)
                results.append(
                    CompileResult(
                        slug=slug,
                        status="failed",
                        error=str(exc),
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Section-split handling
    # ------------------------------------------------------------------

    def _compile_parent_with_children(
        self,
        parent_slug: str,
        children: list[str],
        section_map: dict[str, Any] | None,
        code_structure: dict[str, Any] | None = None,
    ) -> CompileResult:
        """Compile each child separately, then update parent's section_map."""
        child_results: list[CompileResult] = []
        updated_section_map = dict(section_map) if section_map else {}

        for child_slug in children:
            result = self.compile(child_slug, code_structure=code_structure)
            child_results.append(result)

            # Update parent section_map with LLM-improved synopsis (<=150 chars)
            if result.status == "compiled" and result.summary:
                synopsis = result.summary[:150]
                updated_section_map[child_slug] = {
                    "synopsis": synopsis,
                    "concepts": result.concepts,
                }

        # Update parent frontmatter with enriched section_map
        all_concepts: list[str] = []
        all_categories: list[str] = []
        all_backlinks: list[str] = []
        combined_summary_parts: list[str] = []

        for cr in child_results:
            if cr.status == "compiled":
                all_concepts.extend(cr.concepts)
                all_categories.extend(cr.categories)
                all_backlinks.extend(cr.backlinks)
                if cr.summary:
                    combined_summary_parts.append(cr.summary)

        # Deduplicate
        unique_concepts = list(dict.fromkeys(all_concepts))
        unique_categories = list(dict.fromkeys(all_categories))
        unique_backlinks = list(dict.fromkeys(all_backlinks))

        parent_summary = "; ".join(combined_summary_parts)[:200]

        fm_updates: dict[str, Any] = {
            "section_map": updated_section_map,
            "synopsis": parent_summary,
            "concepts": unique_concepts,
            "categories": unique_categories,
            "backlinks": unique_backlinks,
            "compiled": True,
        }
        self._bridge.update_article(
            parent_slug, frontmatter_updates=fm_updates
        )

        failed_children = [
            cr.slug for cr in child_results if cr.status == "failed"
        ]
        if failed_children:
            return CompileResult(
                slug=parent_slug,
                status="compiled",
                summary=parent_summary,
                concepts=unique_concepts,
                categories=unique_categories,
                backlinks=unique_backlinks,
                error=f"Some children failed: {failed_children}",
            )

        return CompileResult(
            slug=parent_slug,
            status="compiled",
            summary=parent_summary,
            concepts=unique_concepts,
            categories=unique_categories,
            backlinks=unique_backlinks,
        )

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str | None:
        """Send a prompt to the LLM and return the text response.

        Uses ModelRouter to select provider/model for task_type='compile'.
        Falls back to returning None if no router is configured or the
        call fails.
        """
        if self._router is None:
            logger.warning("No ModelRouter configured -- cannot call LLM")
            return None

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
                    system="You are a knowledge compiler. Respond with JSON only.",
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
                    system="You are a knowledge compiler. Respond with JSON only.",
                    messages=messages,
                    tool_schemas=[],
                    config=config,
                )

            # Consume the generator to get the AssistantTurn
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            for chunk in gen:
                from saido_agent.core.providers import AssistantTurn, TextChunk, ThinkingChunk

                if isinstance(chunk, TextChunk):
                    text_parts.append(chunk.text)
                elif isinstance(chunk, ThinkingChunk):
                    thinking_parts.append(chunk.text)
                elif isinstance(chunk, AssistantTurn):
                    if chunk.text:
                        return chunk.text
                    if text_parts:
                        return "".join(text_parts)
                    # Fallback: use thinking content if no regular content
                    if thinking_parts:
                        return "".join(thinking_parts)
                    return None

            result = "".join(text_parts) if text_parts else ("".join(thinking_parts) if thinking_parts else None)
            return result

        except Exception as exc:
            logger.error("LLM call failed for compile: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_all_article_slugs(self) -> list[str]:
        """Return all article slugs from the bridge."""
        try:
            articles = self._bridge.list_articles()
            return [slug for slug, _, _ in articles]
        except Exception:
            return []
