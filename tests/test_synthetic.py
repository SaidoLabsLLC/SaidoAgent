"""Tests for saido_agent.knowledge.synthetic — SyntheticDataGenerator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from saido_agent.knowledge.synthetic import (
    SyntheticDataGenerator,
    SyntheticDataResult,
    TrainingPair,
)


# ---------------------------------------------------------------------------
# Fake SmartRAG types (avoid importing real SmartRAG in unit tests)
# ---------------------------------------------------------------------------


@dataclass
class FakeDocument:
    slug: str
    title: str
    body: str
    frontmatter: dict[str, Any] = field(default_factory=dict)
    word_count: int = 100
    has_children: bool = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ARTICLES = {
    "python-basics": FakeDocument(
        slug="python-basics",
        title="Python Basics",
        body=(
            "Python is a high-level programming language created by "
            "Guido van Rossum. It supports multiple paradigms including "
            "OOP and functional programming. Python uses dynamic typing "
            "and has a large standard library."
        ),
    ),
    "rust-ownership": FakeDocument(
        slug="rust-ownership",
        title="Rust Ownership Model",
        body=(
            "Rust uses an ownership system with borrowing and lifetimes "
            "to guarantee memory safety without a garbage collector. "
            "Each value has a single owner, and ownership can be transferred "
            "or borrowed."
        ),
    ),
    "python-async": FakeDocument(
        slug="python-async",
        title="Python Async Programming",
        body=(
            "Python asyncio provides infrastructure for writing single-threaded "
            "concurrent code using coroutines. Use async/await syntax to define "
            "and run asynchronous functions."
        ),
    ),
}

# Canned LLM responses
QA_LLM_RESPONSE = json.dumps([
    {
        "question": "Who created the Python programming language?",
        "answer": "Python was created by Guido van Rossum as a high-level programming language.",
    },
    {
        "question": "What programming paradigms does Python support?",
        "answer": "Python supports multiple paradigms including object-oriented programming (OOP) and functional programming.",
    },
    {
        "question": "What type system does Python use?",
        "answer": "Python uses dynamic typing, which means variable types are determined at runtime.",
    },
    {
        "question": "Does Python have a standard library?",
        "answer": "Yes, Python has a large standard library that provides many built-in modules and functions.",
    },
    {
        "question": "What level of language is Python?",
        "answer": "Python is a high-level programming language designed for readability and ease of use.",
    },
])

INSTRUCTION_LLM_RESPONSE = json.dumps([
    {
        "instruction": "Explain the ownership model in Rust.",
        "response": "Rust uses an ownership system where each value has a single owner. Ownership can be transferred or borrowed, and this system guarantees memory safety without needing a garbage collector.",
    },
    {
        "instruction": "Describe how Rust achieves memory safety.",
        "response": "Rust achieves memory safety through its ownership system combined with borrowing and lifetimes. This eliminates the need for a garbage collector while preventing common memory bugs.",
    },
    {
        "instruction": "What is borrowing in Rust?",
        "response": "Borrowing in Rust allows code to reference a value without taking ownership of it. This is part of Rust's ownership model that ensures memory safety at compile time.",
    },
])

MULTI_HOP_LLM_RESPONSE = json.dumps([
    {
        "question": "How do Python's dynamic typing and Rust's ownership model differ in handling type safety?",
        "answer": "Python uses dynamic typing where types are checked at runtime, while Rust's ownership model enforces memory safety at compile time through static ownership rules, borrowing, and lifetimes.",
    },
    {
        "question": "Compare the memory management approach of Python and Rust.",
        "answer": "Python relies on a garbage collector for automatic memory management with dynamic typing, whereas Rust uses a compile-time ownership system with borrowing and lifetimes to guarantee memory safety without a garbage collector.",
    },
])


@pytest.fixture()
def mock_bridge() -> MagicMock:
    """Return a mock KnowledgeBridge with canned data."""
    bridge = MagicMock()
    bridge.stats = {"document_count": 3, "index_size_bytes": 1024}
    bridge.list_articles.return_value = [
        ("python-basics", "Python Basics", "A Python primer"),
        ("rust-ownership", "Rust Ownership Model", "Rust ownership"),
        ("python-async", "Python Async Programming", "Python async"),
    ]
    bridge.read_article.side_effect = lambda slug: ARTICLES.get(slug)
    return bridge


@pytest.fixture()
def mock_router() -> MagicMock:
    """Return a mock ModelRouter."""
    router = MagicMock()
    router.select_model.return_value = ("ollama", "qwen3:30b")
    return router


@pytest.fixture()
def generator(
    mock_bridge: MagicMock, mock_router: MagicMock
) -> SyntheticDataGenerator:
    """Return a SyntheticDataGenerator wired to mocked bridge + router."""
    return SyntheticDataGenerator(
        bridge=mock_bridge, model_router=mock_router
    )


# ---------------------------------------------------------------------------
# QA pair generation
# ---------------------------------------------------------------------------


class TestQAPairGeneration:
    """Verify QA pair generation from a single article."""

    def test_generates_qa_pairs(self, generator: SyntheticDataGenerator) -> None:
        generator._call_llm = MagicMock(return_value=QA_LLM_RESPONSE)
        pairs = generator.generate_qa_pairs("python-basics", count=5)
        assert len(pairs) == 5
        assert all(p.pair_type == "qa" for p in pairs)
        assert all(p.source_slug == "python-basics" for p in pairs)

    def test_qa_pairs_have_content(
        self, generator: SyntheticDataGenerator
    ) -> None:
        generator._call_llm = MagicMock(return_value=QA_LLM_RESPONSE)
        pairs = generator.generate_qa_pairs("python-basics")
        for pair in pairs:
            assert pair.question.strip()
            assert pair.answer.strip()

    def test_missing_article_returns_empty(
        self, generator: SyntheticDataGenerator, mock_bridge: MagicMock
    ) -> None:
        mock_bridge.read_article.return_value = None
        pairs = generator.generate_qa_pairs("nonexistent")
        assert pairs == []

    def test_llm_failure_returns_empty(
        self, generator: SyntheticDataGenerator
    ) -> None:
        generator._call_llm = MagicMock(return_value=None)
        pairs = generator.generate_qa_pairs("python-basics")
        assert pairs == []


# ---------------------------------------------------------------------------
# Instruction pair generation
# ---------------------------------------------------------------------------


class TestInstructionPairGeneration:
    """Verify instruction/response pair generation."""

    def test_generates_instruction_pairs(
        self, generator: SyntheticDataGenerator
    ) -> None:
        generator._call_llm = MagicMock(
            return_value=INSTRUCTION_LLM_RESPONSE
        )
        pairs = generator.generate_instruction_pairs("rust-ownership", count=3)
        assert len(pairs) == 3
        assert all(p.pair_type == "instruction" for p in pairs)
        assert all(p.source_slug == "rust-ownership" for p in pairs)

    def test_instruction_pairs_have_content(
        self, generator: SyntheticDataGenerator
    ) -> None:
        generator._call_llm = MagicMock(
            return_value=INSTRUCTION_LLM_RESPONSE
        )
        pairs = generator.generate_instruction_pairs("rust-ownership")
        for pair in pairs:
            assert pair.question.strip()
            assert pair.answer.strip()


# ---------------------------------------------------------------------------
# Multi-hop generation
# ---------------------------------------------------------------------------


class TestMultiHopGeneration:
    """Verify multi-hop question generation across articles."""

    def test_generates_multi_hop_pairs(
        self, generator: SyntheticDataGenerator
    ) -> None:
        generator._call_llm = MagicMock(
            return_value=MULTI_HOP_LLM_RESPONSE
        )
        pairs = generator.generate_multi_hop(
            [("python-basics", "rust-ownership")], count=2
        )
        assert len(pairs) == 2
        assert all(p.pair_type == "multi_hop" for p in pairs)
        assert all(
            "python-basics" in p.source_slug
            and "rust-ownership" in p.source_slug
            for p in pairs
        )

    def test_multi_hop_missing_article_skips(
        self, generator: SyntheticDataGenerator, mock_bridge: MagicMock
    ) -> None:
        mock_bridge.read_article.side_effect = lambda slug: (
            ARTICLES.get(slug) if slug != "nonexistent" else None
        )
        pairs = generator.generate_multi_hop(
            [("python-basics", "nonexistent")], count=2
        )
        assert pairs == []


# ---------------------------------------------------------------------------
# Validation filters
# ---------------------------------------------------------------------------


class TestValidation:
    """Verify validation filters correctly reject low-quality pairs."""

    def _make_pair(
        self,
        question: str = "How does X work?",
        answer: str = "X works by doing Y and Z in a specific way.",
        pair_type: str = "qa",
    ) -> TrainingPair:
        return TrainingPair(
            question=question,
            answer=answer,
            source_slug="test",
            pair_type=pair_type,
        )

    def test_filters_short_answers(
        self, generator: SyntheticDataGenerator
    ) -> None:
        pairs = [self._make_pair(answer="Short")]  # <10 chars
        result = generator.validate_pairs(pairs)
        assert len(result) == 0

    def test_filters_long_answers(
        self, generator: SyntheticDataGenerator
    ) -> None:
        pairs = [self._make_pair(answer="A" * 2001)]
        result = generator.validate_pairs(pairs)
        assert len(result) == 0

    def test_filters_generic_questions(
        self, generator: SyntheticDataGenerator
    ) -> None:
        pairs = [self._make_pair(question="What is this about?")]
        result = generator.validate_pairs(pairs)
        assert len(result) == 0

    def test_filters_duplicate_questions(
        self, generator: SyntheticDataGenerator
    ) -> None:
        pairs = [
            self._make_pair(
                question="How does Python handle memory?",
                answer="Python uses garbage collection for memory management.",
            ),
            self._make_pair(
                question="How does Python handle memory?",
                answer="Python manages memory through a garbage collector.",
            ),
        ]
        result = generator.validate_pairs(pairs)
        assert len(result) == 1

    def test_keeps_valid_pairs(
        self, generator: SyntheticDataGenerator
    ) -> None:
        pairs = [
            self._make_pair(
                question="What is Rust's ownership model?",
                answer="Rust uses an ownership system with borrowing and lifetimes for memory safety.",
            ),
            self._make_pair(
                question="How does Python achieve concurrency?",
                answer="Python uses asyncio and coroutines for single-threaded concurrent code execution.",
            ),
        ]
        result = generator.validate_pairs(pairs)
        assert len(result) == 2

    def test_diversity_penalty_applied(
        self, generator: SyntheticDataGenerator
    ) -> None:
        """When >50% of questions start with 'What', quality scores drop."""
        pairs = [
            self._make_pair(
                question="What is Python?",
                answer="Python is a high-level programming language.",
            ),
            self._make_pair(
                question="What are generics?",
                answer="Generics allow parameterized types in programming.",
            ),
            self._make_pair(
                question="What is Rust?",
                answer="Rust is a systems programming language focused on safety.",
            ),
            self._make_pair(
                question="How does async work?",
                answer="Async uses coroutines to enable concurrent execution.",
            ),
        ]
        result = generator.validate_pairs(pairs)
        # 3/4 start with "What" — should get penalized
        what_pairs = [
            p for p in result if p.question.startswith("What")
        ]
        assert all(p.quality_score < 1.0 for p in what_pairs)

    def test_empty_list_returns_empty(
        self, generator: SyntheticDataGenerator
    ) -> None:
        result = generator.validate_pairs([])
        assert result == []


# ---------------------------------------------------------------------------
# JSONL export format
# ---------------------------------------------------------------------------


class TestJSONLExport:
    """Verify OpenAI fine-tuning JSONL export."""

    def test_jsonl_format(
        self, generator: SyntheticDataGenerator, tmp_path: Path
    ) -> None:
        pairs = [
            TrainingPair(
                question="What is Python?",
                answer="Python is a programming language.",
                source_slug="python-basics",
                pair_type="qa",
            ),
        ]
        out = str(tmp_path / "output.jsonl")
        generator.export_jsonl(pairs, out)

        with open(out, encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "messages" in record
        msgs = record["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "What is Python?"
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"] == "Python is a programming language."

    def test_jsonl_multiple_pairs(
        self, generator: SyntheticDataGenerator, tmp_path: Path
    ) -> None:
        pairs = [
            TrainingPair(
                question=f"Q{i}",
                answer=f"A{i} is a detailed answer.",
                source_slug="test",
                pair_type="qa",
            )
            for i in range(5)
        ]
        out = str(tmp_path / "multi.jsonl")
        generator.export_jsonl(pairs, out)

        with open(out, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 5


# ---------------------------------------------------------------------------
# Alpaca export format
# ---------------------------------------------------------------------------


class TestAlpacaExport:
    """Verify Alpaca format export."""

    def test_alpaca_format(
        self, generator: SyntheticDataGenerator, tmp_path: Path
    ) -> None:
        pairs = [
            TrainingPair(
                question="Explain ownership in Rust.",
                answer="Rust uses an ownership system for memory safety.",
                source_slug="rust-ownership",
                pair_type="instruction",
            ),
        ]
        out = str(tmp_path / "output.json")
        generator.export_alpaca(pairs, out)

        data = json.loads(Path(out).read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["instruction"] == "Explain ownership in Rust."
        assert data[0]["input"] == ""
        assert data[0]["output"] == "Rust uses an ownership system for memory safety."


# ---------------------------------------------------------------------------
# ShareGPT export format
# ---------------------------------------------------------------------------


class TestShareGPTExport:
    """Verify ShareGPT format export."""

    def test_sharegpt_format(
        self, generator: SyntheticDataGenerator, tmp_path: Path
    ) -> None:
        pairs = [
            TrainingPair(
                question="How does async work in Python?",
                answer="Python uses asyncio with async/await syntax.",
                source_slug="python-async",
                pair_type="qa",
            ),
        ]
        out = str(tmp_path / "output.json")
        generator.export_sharegpt(pairs, out)

        data = json.loads(Path(out).read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 1
        conv = data[0]["conversations"]
        assert len(conv) == 2
        assert conv[0]["from"] == "human"
        assert conv[0]["value"] == "How does async work in Python?"
        assert conv[1]["from"] == "gpt"
        assert conv[1]["value"] == "Python uses asyncio with async/await syntax."


# ---------------------------------------------------------------------------
# SyntheticDataResult
# ---------------------------------------------------------------------------


class TestSyntheticDataResult:
    """Verify the result dataclass defaults and structure."""

    def test_defaults(self) -> None:
        r = SyntheticDataResult(
            output_path="/tmp/out.jsonl",
            format="jsonl",
            total_pairs=10,
            qa_pairs=5,
            instruction_pairs=3,
            multi_hop_pairs=2,
            filtered_count=1,
            status="generated",
        )
        assert r.output_path == "/tmp/out.jsonl"
        assert r.format == "jsonl"
        assert r.total_pairs == 10
        assert r.qa_pairs == 5
        assert r.instruction_pairs == 3
        assert r.multi_hop_pairs == 2
        assert r.filtered_count == 1
        assert r.status == "generated"
        assert r.error is None

    def test_error_field(self) -> None:
        r = SyntheticDataResult(
            output_path="",
            format="jsonl",
            total_pairs=0,
            qa_pairs=0,
            instruction_pairs=0,
            multi_hop_pairs=0,
            filtered_count=0,
            status="failed",
            error="No articles found.",
        )
        assert r.status == "failed"
        assert r.error == "No articles found."


# ---------------------------------------------------------------------------
# Empty knowledge store
# ---------------------------------------------------------------------------


class TestEmptyStore:
    """Verify behavior with an empty knowledge store."""

    def test_empty_store_returns_failed_result(
        self, mock_router: MagicMock, tmp_path: Path
    ) -> None:
        bridge = MagicMock()
        bridge.list_articles.return_value = []
        gen = SyntheticDataGenerator(bridge=bridge, model_router=mock_router)
        result = gen.generate(output_dir=str(tmp_path))
        assert result.status == "failed"
        assert result.total_pairs == 0
        assert "No articles" in (result.error or "")

    def test_empty_store_specific_slugs_no_docs(
        self, mock_router: MagicMock, tmp_path: Path
    ) -> None:
        bridge = MagicMock()
        bridge.read_article.return_value = None
        gen = SyntheticDataGenerator(bridge=bridge, model_router=mock_router)
        gen._call_llm = MagicMock(return_value=None)
        result = gen.generate(
            articles=["nonexistent"], output_dir=str(tmp_path)
        )
        # Should generate with 0 pairs (all LLM calls return None)
        assert result.total_pairs == 0


# ---------------------------------------------------------------------------
# Full generate() workflow
# ---------------------------------------------------------------------------


class TestGenerateWorkflow:
    """End-to-end generate() with mocked LLM."""

    def test_generate_produces_output_file(
        self, generator: SyntheticDataGenerator, tmp_path: Path
    ) -> None:
        generator._call_llm = MagicMock(
            side_effect=[
                # QA for python-basics
                QA_LLM_RESPONSE,
                # instruction for python-basics
                INSTRUCTION_LLM_RESPONSE,
                # QA for rust-ownership
                QA_LLM_RESPONSE,
                # instruction for rust-ownership
                INSTRUCTION_LLM_RESPONSE,
                # QA for python-async
                QA_LLM_RESPONSE,
                # instruction for python-async
                INSTRUCTION_LLM_RESPONSE,
                # multi-hop python-basics + rust-ownership
                MULTI_HOP_LLM_RESPONSE,
                # multi-hop rust-ownership + python-async
                MULTI_HOP_LLM_RESPONSE,
            ]
        )
        result = generator.generate(format="jsonl", output_dir=str(tmp_path))
        assert result.status == "generated"
        assert result.total_pairs > 0
        assert Path(result.output_path).exists()

    def test_generate_alpaca_format(
        self, generator: SyntheticDataGenerator, tmp_path: Path
    ) -> None:
        generator._call_llm = MagicMock(
            side_effect=[
                QA_LLM_RESPONSE,
                INSTRUCTION_LLM_RESPONSE,
                QA_LLM_RESPONSE,
                INSTRUCTION_LLM_RESPONSE,
                QA_LLM_RESPONSE,
                INSTRUCTION_LLM_RESPONSE,
                MULTI_HOP_LLM_RESPONSE,
                MULTI_HOP_LLM_RESPONSE,
            ]
        )
        result = generator.generate(format="alpaca", output_dir=str(tmp_path))
        assert result.status == "generated"
        assert result.format == "alpaca"
        data = json.loads(
            Path(result.output_path).read_text(encoding="utf-8")
        )
        assert isinstance(data, list)

    def test_generate_specific_articles(
        self, generator: SyntheticDataGenerator, tmp_path: Path
    ) -> None:
        generator._call_llm = MagicMock(
            side_effect=[
                QA_LLM_RESPONSE,
                INSTRUCTION_LLM_RESPONSE,
            ]
        )
        result = generator.generate(
            articles=["python-basics"], output_dir=str(tmp_path)
        )
        assert result.status == "generated"
        # Only 1 article, no multi-hop possible
        assert result.multi_hop_pairs == 0


# ---------------------------------------------------------------------------
# JSON parsing edge cases
# ---------------------------------------------------------------------------


class TestJSONParsing:
    """Verify robust JSON extraction from LLM output."""

    def test_parses_markdown_fenced_json(
        self, generator: SyntheticDataGenerator
    ) -> None:
        raw = '```json\n[{"question": "Q1?", "answer": "A1 is the answer."}]\n```'
        generator._call_llm = MagicMock(return_value=raw)
        pairs = generator.generate_qa_pairs("python-basics")
        assert len(pairs) == 1

    def test_parses_json_with_surrounding_text(
        self, generator: SyntheticDataGenerator
    ) -> None:
        raw = 'Here are the pairs:\n[{"question": "Q1?", "answer": "A1 is the detailed answer."}]\nDone!'
        generator._call_llm = MagicMock(return_value=raw)
        pairs = generator.generate_qa_pairs("python-basics")
        assert len(pairs) == 1

    def test_handles_unparseable_response(
        self, generator: SyntheticDataGenerator
    ) -> None:
        generator._call_llm = MagicMock(
            return_value="This is not valid JSON at all."
        )
        pairs = generator.generate_qa_pairs("python-basics")
        assert pairs == []
