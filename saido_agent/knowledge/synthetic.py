"""Synthetic training data generation from the knowledge store.

Generates QA pairs, instruction-response pairs, and multi-hop questions
from compiled knowledge articles for fine-tuning downstream models.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saido_agent.knowledge.bridge import KnowledgeBridge
    from saido_agent.core.routing import ModelRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------
_MIN_ANSWER_LEN = 10
_MAX_ANSWER_LEN = 2000
_GENERIC_QUESTIONS = frozenset({
    "what is this about?",
    "what is this?",
    "what does this mean?",
    "can you explain?",
    "tell me more.",
    "what is the topic?",
    "summarize this.",
    "what are the main points?",
})
_FUZZY_MATCH_THRESHOLD = 0.85
_DIVERSITY_THRESHOLD = 0.50  # reject if >50% questions start with same word

# ---------------------------------------------------------------------------
# System prompts for LLM generation
# ---------------------------------------------------------------------------
_QA_SYSTEM_PROMPT = (
    "You are a training data generator. Given the following article, "
    "generate {count} question/answer pairs that can be answered by the "
    "article content. Return ONLY a JSON array of objects with 'question' "
    "and 'answer' keys. No markdown fencing, no explanation."
)

_INSTRUCTION_SYSTEM_PROMPT = (
    "You are a training data generator. Given the following article, "
    "rephrase the key information as {count} instruction-following pairs. "
    "Each pair should have an 'instruction' that asks for specific "
    "information and a 'response' that provides it. Return ONLY a JSON "
    "array of objects with 'instruction' and 'response' keys. "
    "No markdown fencing, no explanation."
)

_MULTI_HOP_SYSTEM_PROMPT = (
    "You are a training data generator. Given the following TWO articles, "
    "generate {count} questions that require combining information from "
    "BOTH articles to answer fully. Return ONLY a JSON array of objects "
    "with 'question' and 'answer' keys. No markdown fencing, no explanation."
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrainingPair:
    """A single training pair for fine-tuning."""

    question: str
    answer: str
    source_slug: str
    pair_type: str  # "qa", "instruction", "multi_hop"
    quality_score: float = 1.0


@dataclass
class SyntheticDataResult:
    """Result of a synthetic data generation run."""

    output_path: str
    format: str
    total_pairs: int
    qa_pairs: int
    instruction_pairs: int
    multi_hop_pairs: int
    filtered_count: int
    status: str  # "generated", "failed"
    error: str | None = None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class SyntheticDataGenerator:
    """Generate synthetic training data from knowledge store articles.

    Uses LLM calls to produce QA pairs, instruction-response pairs, and
    multi-hop questions that require combining information from multiple
    articles. Supports export to OpenAI JSONL, Alpaca, and ShareGPT formats.
    """

    def __init__(
        self,
        bridge: KnowledgeBridge,
        model_router: ModelRouter | None = None,
    ) -> None:
        self._bridge = bridge
        self._router = model_router

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        format: str = "jsonl",
        articles: list[str] | None = None,
        output_dir: str | None = None,
    ) -> SyntheticDataResult:
        """Generate training data from knowledge store.

        Args:
            format: Export format — "jsonl", "alpaca", or "sharegpt".
            articles: Optional list of slugs to use. If None, uses all.
            output_dir: Directory for output file. Defaults to cwd.

        Returns:
            SyntheticDataResult with generation metadata.
        """
        try:
            # Resolve slugs
            if articles:
                slugs = articles
            else:
                all_articles = self._bridge.list_articles()
                slugs = [slug for slug, _title, _summary in all_articles]

            if not slugs:
                return SyntheticDataResult(
                    output_path="",
                    format=format,
                    total_pairs=0,
                    qa_pairs=0,
                    instruction_pairs=0,
                    multi_hop_pairs=0,
                    filtered_count=0,
                    status="failed",
                    error="No articles found in knowledge store.",
                )

            all_pairs: list[TrainingPair] = []

            # Generate QA and instruction pairs per article
            for slug in slugs:
                qa_pairs = self.generate_qa_pairs(slug, count=5)
                all_pairs.extend(qa_pairs)

                inst_pairs = self.generate_instruction_pairs(slug, count=3)
                all_pairs.extend(inst_pairs)

            # Generate multi-hop pairs for consecutive slug pairs
            if len(slugs) >= 2:
                slug_pairs = [
                    (slugs[i], slugs[i + 1]) for i in range(len(slugs) - 1)
                ]
                for pair in slug_pairs:
                    mh_pairs = self.generate_multi_hop([pair], count=2)
                    all_pairs.extend(mh_pairs)

            # Validate
            pre_filter_count = len(all_pairs)
            valid_pairs = self.validate_pairs(all_pairs)
            filtered_count = pre_filter_count - len(valid_pairs)

            # Export
            out_dir = Path(output_dir) if output_dir else Path.cwd()
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}"

            if format == "alpaca":
                out_path = str(out_dir / f"{filename}.json")
                self.export_alpaca(valid_pairs, out_path)
            elif format == "sharegpt":
                out_path = str(out_dir / f"{filename}.json")
                self.export_sharegpt(valid_pairs, out_path)
            else:
                out_path = str(out_dir / f"{filename}.jsonl")
                self.export_jsonl(valid_pairs, out_path)

            qa_count = sum(1 for p in valid_pairs if p.pair_type == "qa")
            inst_count = sum(
                1 for p in valid_pairs if p.pair_type == "instruction"
            )
            mh_count = sum(
                1 for p in valid_pairs if p.pair_type == "multi_hop"
            )

            return SyntheticDataResult(
                output_path=out_path,
                format=format,
                total_pairs=len(valid_pairs),
                qa_pairs=qa_count,
                instruction_pairs=inst_count,
                multi_hop_pairs=mh_count,
                filtered_count=filtered_count,
                status="generated",
            )

        except Exception as exc:
            logger.exception("Synthetic data generation failed")
            return SyntheticDataResult(
                output_path="",
                format=format,
                total_pairs=0,
                qa_pairs=0,
                instruction_pairs=0,
                multi_hop_pairs=0,
                filtered_count=0,
                status="failed",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Per-type generators
    # ------------------------------------------------------------------

    def generate_qa_pairs(
        self, slug: str, count: int = 5
    ) -> list[TrainingPair]:
        """Generate QA pairs from a single article via LLM."""
        doc = self._bridge.read_article(slug)
        if doc is None:
            logger.warning("Article not found: %s", slug)
            return []

        prompt = (
            _QA_SYSTEM_PROMPT.format(count=count)
            + "\n\n## Article\n"
            + f"Title: {doc.title}\n\n{doc.body}"
        )

        raw = self._call_llm(prompt)
        if raw is None:
            return []

        return self._parse_qa_response(raw, slug, "qa")

    def generate_instruction_pairs(
        self, slug: str, count: int = 3
    ) -> list[TrainingPair]:
        """Generate instruction/response pairs from a single article."""
        doc = self._bridge.read_article(slug)
        if doc is None:
            logger.warning("Article not found: %s", slug)
            return []

        prompt = (
            _INSTRUCTION_SYSTEM_PROMPT.format(count=count)
            + "\n\n## Article\n"
            + f"Title: {doc.title}\n\n{doc.body}"
        )

        raw = self._call_llm(prompt)
        if raw is None:
            return []

        return self._parse_instruction_response(raw, slug)

    def generate_multi_hop(
        self, slug_pairs: list[tuple[str, str]], count: int = 2
    ) -> list[TrainingPair]:
        """Generate questions requiring info from multiple articles."""
        pairs: list[TrainingPair] = []

        for slug_a, slug_b in slug_pairs:
            doc_a = self._bridge.read_article(slug_a)
            doc_b = self._bridge.read_article(slug_b)
            if doc_a is None or doc_b is None:
                logger.warning(
                    "Skipping multi-hop for %s + %s: article not found",
                    slug_a,
                    slug_b,
                )
                continue

            prompt = (
                _MULTI_HOP_SYSTEM_PROMPT.format(count=count)
                + "\n\n## Article 1\n"
                + f"Title: {doc_a.title}\n\n{doc_a.body}"
                + "\n\n## Article 2\n"
                + f"Title: {doc_b.title}\n\n{doc_b.body}"
            )

            raw = self._call_llm(prompt)
            if raw is None:
                continue

            source = f"{slug_a}+{slug_b}"
            parsed = self._parse_qa_response(raw, source, "multi_hop")
            pairs.extend(parsed)

        return pairs

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_pairs(
        self, pairs: list[TrainingPair]
    ) -> list[TrainingPair]:
        """Filter out low-quality pairs.

        Filters:
          - Answer too short (<10 chars) or too long (>2000 chars)
          - Question is too generic
          - Duplicate questions (fuzzy match)
          - Diversity: reject batch if >50% of questions start with same word
        """
        valid: list[TrainingPair] = []

        for pair in pairs:
            # Length filters
            if len(pair.answer.strip()) < _MIN_ANSWER_LEN:
                logger.debug("Filtered short answer: %s", pair.question[:50])
                continue
            if len(pair.answer.strip()) > _MAX_ANSWER_LEN:
                logger.debug("Filtered long answer: %s", pair.question[:50])
                continue

            # Generic question filter
            if pair.question.strip().lower().rstrip("?. ") in {
                q.rstrip("?. ") for q in _GENERIC_QUESTIONS
            }:
                logger.debug("Filtered generic question: %s", pair.question)
                continue

            # Fuzzy duplicate filter
            is_dup = False
            for existing in valid:
                ratio = SequenceMatcher(
                    None,
                    pair.question.lower(),
                    existing.question.lower(),
                ).ratio()
                if ratio >= _FUZZY_MATCH_THRESHOLD:
                    is_dup = True
                    break
            if is_dup:
                logger.debug("Filtered duplicate: %s", pair.question[:50])
                continue

            valid.append(pair)

        # Diversity check: reject if >50% start with same word
        if valid:
            first_words = [
                p.question.strip().split()[0].lower()
                for p in valid
                if p.question.strip()
            ]
            if first_words:
                counts = Counter(first_words)
                most_common_count = counts.most_common(1)[0][1]
                if most_common_count / len(first_words) > _DIVERSITY_THRESHOLD:
                    # Penalize quality score but keep the pairs
                    most_common_word = counts.most_common(1)[0][0]
                    for pair in valid:
                        fw = pair.question.strip().split()[0].lower()
                        if fw == most_common_word:
                            pair.quality_score *= 0.7

        return valid

    # ------------------------------------------------------------------
    # Export formats
    # ------------------------------------------------------------------

    def export_jsonl(
        self, pairs: list[TrainingPair], path: str
    ) -> str:
        """Export as OpenAI fine-tuning JSONL format.

        Format: {"messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "<question>"},
            {"role": "assistant", "content": "<answer>"}
        ]}
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for pair in pairs:
                record = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant.",
                        },
                        {"role": "user", "content": pair.question},
                        {"role": "assistant", "content": pair.answer},
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Exported %d pairs to %s (JSONL)", len(pairs), path)
        return path

    def export_alpaca(
        self, pairs: list[TrainingPair], path: str
    ) -> str:
        """Export as Alpaca format.

        Format: [{"instruction": "Q", "input": "", "output": "A"}, ...]
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "instruction": pair.question,
                "input": "",
                "output": pair.answer,
            }
            for pair in pairs
        ]
        out.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Exported %d pairs to %s (Alpaca)", len(pairs), path)
        return path

    def export_sharegpt(
        self, pairs: list[TrainingPair], path: str
    ) -> str:
        """Export as ShareGPT format.

        Format: [{"conversations": [
            {"from": "human", "value": "Q"},
            {"from": "gpt", "value": "A"}
        ]}, ...]
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "conversations": [
                    {"from": "human", "value": pair.question},
                    {"from": "gpt", "value": pair.answer},
                ]
            }
            for pair in pairs
        ]
        out.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Exported %d pairs to %s (ShareGPT)", len(pairs), path)
        return path

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str | None:
        """Send prompt to LLM via ModelRouter. Returns raw text or None."""
        if self._router is None:
            logger.warning("No ModelRouter configured — cannot call LLM")
            return None

        try:
            from saido_agent.core.providers import stream as llm_stream

            provider, model = self._router.select_model("generation")
            messages = [{"role": "user", "content": prompt}]
            total_text = ""

            for chunk in llm_stream(
                model=(
                    f"{provider}/{model}"
                    if provider not in ("anthropic",)
                    else model
                ),
                system="",
                messages=messages,
                tool_schemas=[],
                config={},
            ):
                from saido_agent.core.providers import (
                    AssistantTurn,
                    TextChunk,
                )

                if isinstance(chunk, TextChunk):
                    total_text += chunk.text
                elif isinstance(chunk, AssistantTurn):
                    total_text = chunk.text

            return total_text if total_text else None

        except Exception:
            logger.exception("LLM call failed")
            return None

    # ------------------------------------------------------------------
    # Response parsing helpers
    # ------------------------------------------------------------------

    def _parse_qa_response(
        self, raw: str, source_slug: str, pair_type: str
    ) -> list[TrainingPair]:
        """Parse a JSON array of {question, answer} from LLM response."""
        parsed = self._extract_json_array(raw)
        if parsed is None:
            return []

        pairs: list[TrainingPair] = []
        for item in parsed:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if q and a:
                pairs.append(
                    TrainingPair(
                        question=q,
                        answer=a,
                        source_slug=source_slug,
                        pair_type=pair_type,
                    )
                )
        return pairs

    def _parse_instruction_response(
        self, raw: str, source_slug: str
    ) -> list[TrainingPair]:
        """Parse a JSON array of {instruction, response} from LLM response."""
        parsed = self._extract_json_array(raw)
        if parsed is None:
            return []

        pairs: list[TrainingPair] = []
        for item in parsed:
            q = item.get("instruction", "").strip()
            a = item.get("response", "").strip()
            if q and a:
                pairs.append(
                    TrainingPair(
                        question=q,
                        answer=a,
                        source_slug=source_slug,
                        pair_type="instruction",
                    )
                )
        return pairs

    @staticmethod
    def _extract_json_array(raw: str) -> list[dict] | None:
        """Extract a JSON array from LLM output, handling markdown fencing."""
        text = raw.strip()

        # Strip markdown code fences if present
        fence_match = re.search(
            r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL
        )
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Try to find the array within the text
        bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
        if bracket_match:
            try:
                data = json.loads(bracket_match.group(0))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse JSON array from LLM response")
        return None
