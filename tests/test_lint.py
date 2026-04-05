"""Tests for KnowledgeLinter and LintReport."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from saido_agent.knowledge.lint import (
    KnowledgeLinter,
    LintReport,
    _extract_json,
)


# ---------------------------------------------------------------------------
# Helpers: mock Document and bridge
# ---------------------------------------------------------------------------


@dataclass
class FakeDocument:
    """Minimal stand-in for SmartRAG Document."""

    slug: str
    body: str
    frontmatter: dict


def _make_bridge(
    articles: dict[str, dict] | None = None,
    all_articles: list[tuple[str, str, str]] | None = None,
) -> MagicMock:
    """Build a mock KnowledgeBridge.

    Args:
        articles: slug -> {"body": ..., "frontmatter": {...}}
        all_articles: list of (slug, title, summary) for list_articles()
    """
    articles = articles or {}
    if all_articles is None:
        all_articles = [
            (slug, meta.get("frontmatter", {}).get("title", slug), "")
            for slug, meta in articles.items()
        ]

    bridge = MagicMock()
    bridge.available = True

    def _read_article(slug):
        if slug in articles:
            return FakeDocument(
                slug=slug,
                body=articles[slug].get("body", ""),
                frontmatter=articles[slug].get("frontmatter", {}),
            )
        return None

    def _read_frontmatter(slug):
        if slug in articles:
            return articles[slug].get("frontmatter", {})
        return None

    bridge.read_article.side_effect = _read_article
    bridge.read_article_frontmatter.side_effect = _read_frontmatter
    bridge.list_articles.return_value = all_articles

    return bridge


# ---------------------------------------------------------------------------
# LintReport tests
# ---------------------------------------------------------------------------


class TestLintReport:
    def test_issue_count_empty(self):
        report = LintReport()
        assert report.issue_count() == 0

    def test_issue_count_aggregates_all_fields(self):
        report = LintReport(
            contradictions=[("a", "b", "desc")],
            missing_data=[("a", "claim")],
            knowledge_gaps=[("topic", ["a", "b"])],
            orphans=["orphan1", "orphan2"],
            stale=[("old", "2020-01-01")],
            dead_links=[("a", "missing")],
            duplicates=[("a", "b", "same topic")],
        )
        assert report.issue_count() == 8

    def test_compute_health_healthy(self):
        report = LintReport()
        assert report.compute_health() == "healthy"

    def test_compute_health_healthy_with_two_issues(self):
        report = LintReport(orphans=["a", "b"])
        assert report.compute_health() == "healthy"

    def test_compute_health_needs_attention(self):
        report = LintReport(orphans=["a", "b", "c"])
        assert report.compute_health() == "needs attention"

    def test_compute_health_needs_attention_at_ten(self):
        report = LintReport(orphans=[f"o{i}" for i in range(10)])
        assert report.compute_health() == "needs attention"

    def test_compute_health_unhealthy(self):
        report = LintReport(orphans=[f"o{i}" for i in range(11)])
        assert report.compute_health() == "unhealthy"

    def test_to_dict_structure(self):
        report = LintReport(
            dead_links=[("art-a", "missing-link")],
            orphans=["lonely"],
            overall_health="needs attention",
        )
        d = report.to_dict()
        assert d["dead_links"] == [
            {"article": "art-a", "broken_link": "missing-link"}
        ]
        assert d["orphans"] == ["lonely"]
        assert d["overall_health"] == "needs attention"
        assert d["issue_count"] == 2


# ---------------------------------------------------------------------------
# Dead links
# ---------------------------------------------------------------------------


class TestDeadLinks:
    def test_detects_dead_links(self):
        articles = {
            "intro": {
                "body": "See [[missing-article]] and [[existing]].",
                "frontmatter": {"title": "Intro"},
            },
            "existing": {
                "body": "No links here.",
                "frontmatter": {"title": "Existing"},
            },
        }
        bridge = _make_bridge(articles)
        linter = KnowledgeLinter(bridge)
        dead = linter.check_dead_links()
        assert len(dead) == 1
        assert dead[0] == ("intro", "missing-article")

    def test_no_dead_links_when_all_exist(self):
        articles = {
            "a": {
                "body": "Link to [[b]].",
                "frontmatter": {"title": "A"},
            },
            "b": {
                "body": "Link to [[a]].",
                "frontmatter": {"title": "B"},
            },
        }
        bridge = _make_bridge(articles)
        linter = KnowledgeLinter(bridge)
        dead = linter.check_dead_links()
        assert dead == []

    def test_no_dead_links_when_no_wikilinks(self):
        articles = {
            "plain": {
                "body": "Just plain text, no links.",
                "frontmatter": {"title": "Plain"},
            },
        }
        bridge = _make_bridge(articles)
        linter = KnowledgeLinter(bridge)
        assert linter.check_dead_links() == []

    def test_handles_wikilink_with_alias(self):
        """[[slug|Display Text]] should extract slug."""
        articles = {
            "a": {
                "body": "See [[missing|the missing page]].",
                "frontmatter": {},
            },
        }
        bridge = _make_bridge(articles)
        linter = KnowledgeLinter(bridge)
        dead = linter.check_dead_links()
        assert len(dead) == 1
        assert dead[0] == ("a", "missing")

    def test_handles_wikilink_with_anchor(self):
        """[[slug#section]] should extract slug."""
        articles = {
            "a": {
                "body": "See [[missing#intro]].",
                "frontmatter": {},
            },
        }
        bridge = _make_bridge(articles)
        linter = KnowledgeLinter(bridge)
        dead = linter.check_dead_links()
        assert len(dead) == 1
        assert dead[0] == ("a", "missing")


# ---------------------------------------------------------------------------
# Orphans
# ---------------------------------------------------------------------------


class TestOrphans:
    def test_detects_orphans(self):
        articles = {
            "linked": {
                "body": "Nothing links to orphan.",
                "frontmatter": {"title": "Linked"},
            },
            "orphan": {
                "body": "See [[linked]].",
                "frontmatter": {"title": "Orphan"},
            },
        }
        bridge = _make_bridge(articles)
        linter = KnowledgeLinter(bridge)
        orphans = linter.check_orphans()
        # "linked" has inbound from "orphan", but "orphan" has no inbound
        # Actually "linked" is linked from orphan, so linked has 1 inbound.
        # "orphan" has 0 inbound from any article.
        assert "orphan" in orphans
        assert "linked" not in orphans

    def test_no_orphans_when_all_linked(self):
        articles = {
            "a": {
                "body": "See [[b]].",
                "frontmatter": {"title": "A"},
            },
            "b": {
                "body": "See [[a]].",
                "frontmatter": {"title": "B"},
            },
        }
        bridge = _make_bridge(articles)
        linter = KnowledgeLinter(bridge)
        orphans = linter.check_orphans()
        assert orphans == []

    def test_all_orphans_when_no_links(self):
        articles = {
            "a": {"body": "No links.", "frontmatter": {}},
            "b": {"body": "No links.", "frontmatter": {}},
        }
        bridge = _make_bridge(articles)
        linter = KnowledgeLinter(bridge)
        orphans = linter.check_orphans()
        assert set(orphans) == {"a", "b"}


# ---------------------------------------------------------------------------
# Stale detection
# ---------------------------------------------------------------------------


class TestStale:
    def test_detects_stale_articles(self):
        old_date = (datetime.now(timezone.utc) - timedelta(days=100)).strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        )
        articles = {
            "old": {
                "body": "Old content.",
                "frontmatter": {"title": "Old", "updated": old_date},
            },
        }
        bridge = _make_bridge(articles)
        linter = KnowledgeLinter(bridge)
        stale = linter.check_stale(days=90)
        assert len(stale) == 1
        assert stale[0][0] == "old"

    def test_fresh_articles_not_flagged(self):
        recent_date = (
            datetime.now(timezone.utc) - timedelta(days=10)
        ).strftime("%Y-%m-%dT%H:%M:%S%z")
        articles = {
            "fresh": {
                "body": "Fresh content.",
                "frontmatter": {"title": "Fresh", "updated": recent_date},
            },
        }
        bridge = _make_bridge(articles)
        linter = KnowledgeLinter(bridge)
        stale = linter.check_stale(days=90)
        assert stale == []

    def test_configurable_threshold(self):
        date_45_days_ago = (
            datetime.now(timezone.utc) - timedelta(days=45)
        ).strftime("%Y-%m-%dT%H:%M:%S%z")
        articles = {
            "medium": {
                "body": "Content.",
                "frontmatter": {"title": "Medium", "updated": date_45_days_ago},
            },
        }
        bridge = _make_bridge(articles)
        linter = KnowledgeLinter(bridge)
        # Not stale at 90 days
        assert linter.check_stale(days=90) == []
        # Stale at 30 days
        stale = linter.check_stale(days=30)
        assert len(stale) == 1

    def test_uses_created_date_fallback(self):
        old_date = (datetime.now(timezone.utc) - timedelta(days=200)).strftime(
            "%Y-%m-%d"
        )
        articles = {
            "created-only": {
                "body": "Content.",
                "frontmatter": {"created": old_date},
            },
        }
        bridge = _make_bridge(articles)
        linter = KnowledgeLinter(bridge)
        stale = linter.check_stale(days=90)
        assert len(stale) == 1

    def test_skips_articles_without_dates(self):
        articles = {
            "no-date": {
                "body": "Content.",
                "frontmatter": {"title": "No Date"},
            },
        }
        bridge = _make_bridge(articles)
        linter = KnowledgeLinter(bridge)
        stale = linter.check_stale(days=90)
        assert stale == []


# ---------------------------------------------------------------------------
# Knowledge gaps
# ---------------------------------------------------------------------------


class TestKnowledgeGaps:
    def test_detects_gaps(self):
        articles = {
            "ml-basics": {
                "body": "Intro to ML.",
                "frontmatter": {
                    "concepts": ["neural networks", "backpropagation"],
                },
            },
            "deep-learning": {
                "body": "Deep learning overview.",
                "frontmatter": {
                    "concepts": ["neural networks", "transformers"],
                },
            },
        }
        bridge = _make_bridge(articles)
        linter = KnowledgeLinter(bridge)
        gaps = linter.check_knowledge_gaps()
        # "neural networks" appears in both but has no dedicated article
        gap_topics = [t for t, _ in gaps]
        assert "neural networks" in gap_topics

    def test_no_gap_when_article_exists(self):
        articles = {
            "neural-networks": {
                "body": "Dedicated article.",
                "frontmatter": {"concepts": ["neural networks"]},
            },
            "ml-basics": {
                "body": "Intro.",
                "frontmatter": {"concepts": ["neural networks", "ml"]},
            },
            "deep-learning": {
                "body": "DL.",
                "frontmatter": {"concepts": ["neural networks"]},
            },
        }
        bridge = _make_bridge(articles)
        linter = KnowledgeLinter(bridge)
        gaps = linter.check_knowledge_gaps()
        gap_topics = [t for t, _ in gaps]
        assert "neural networks" not in gap_topics

    def test_single_reference_not_a_gap(self):
        articles = {
            "only-one": {
                "body": "Only article mentioning this concept.",
                "frontmatter": {"concepts": ["rare-concept"]},
            },
        }
        bridge = _make_bridge(articles)
        linter = KnowledgeLinter(bridge)
        gaps = linter.check_knowledge_gaps()
        assert gaps == []


# ---------------------------------------------------------------------------
# LLM-powered checks (mocked)
# ---------------------------------------------------------------------------


class TestDuplicates:
    def test_detects_duplicates_via_llm(self):
        articles = {
            "python-guide": {
                "body": "Python programming guide.",
                "frontmatter": {
                    "title": "Python Guide",
                    "concepts": ["python", "programming", "scripting"],
                },
            },
            "python-tutorial": {
                "body": "Python programming tutorial.",
                "frontmatter": {
                    "title": "Python Tutorial",
                    "concepts": ["python", "programming", "beginners"],
                },
            },
        }
        bridge = _make_bridge(articles)
        router = MagicMock()
        linter = KnowledgeLinter(bridge, model_router=router)

        with patch.object(
            linter,
            "_call_llm",
            return_value=json.dumps(
                {"duplicate": True, "reason": "Both cover Python basics"}
            ),
        ):
            dupes = linter.check_duplicates()
            assert len(dupes) == 1
            assert dupes[0][2] == "Both cover Python basics"

    def test_no_duplicates_when_llm_says_no(self):
        articles = {
            "a": {
                "body": "Article A.",
                "frontmatter": {"concepts": ["x", "y"]},
            },
            "b": {
                "body": "Article B.",
                "frontmatter": {"concepts": ["x", "y"]},
            },
        }
        bridge = _make_bridge(articles)
        router = MagicMock()
        linter = KnowledgeLinter(bridge, model_router=router)

        with patch.object(
            linter,
            "_call_llm",
            return_value=json.dumps(
                {"duplicate": False, "reason": "Different topics"}
            ),
        ):
            dupes = linter.check_duplicates()
            assert dupes == []

    def test_skips_when_no_concept_overlap(self):
        articles = {
            "a": {
                "body": "A.",
                "frontmatter": {"concepts": ["x"]},
            },
            "b": {
                "body": "B.",
                "frontmatter": {"concepts": ["y"]},
            },
        }
        bridge = _make_bridge(articles)
        router = MagicMock()
        linter = KnowledgeLinter(bridge, model_router=router)

        with patch.object(linter, "_call_llm") as mock_llm:
            dupes = linter.check_duplicates()
            mock_llm.assert_not_called()
            assert dupes == []


class TestContradictions:
    def test_detects_contradictions_via_llm(self):
        articles = {
            "a": {
                "body": "Python is dynamically typed.",
                "frontmatter": {
                    "title": "A",
                    "categories": ["Programming"],
                    "concepts": ["python"],
                },
            },
            "b": {
                "body": "Python is statically typed.",
                "frontmatter": {
                    "title": "B",
                    "categories": ["Programming"],
                    "concepts": ["python"],
                },
            },
        }
        bridge = _make_bridge(articles)
        router = MagicMock()
        linter = KnowledgeLinter(bridge, model_router=router)

        with patch.object(
            linter,
            "_call_llm",
            return_value=json.dumps(
                {
                    "contradicts": True,
                    "description": "Disagree on Python typing",
                }
            ),
        ):
            contras = linter.check_contradictions()
            assert len(contras) == 1
            assert "typing" in contras[0][2].lower()


class TestMissingData:
    def test_detects_missing_evidence(self):
        articles = {
            "claims": {
                "body": "Python is used by 90% of data scientists.",
                "frontmatter": {"title": "Claims"},
            },
        }
        bridge = _make_bridge(articles)
        router = MagicMock()
        linter = KnowledgeLinter(bridge, model_router=router)

        with patch.object(
            linter,
            "_call_llm",
            return_value=json.dumps(
                ["90% statistic lacks citation"]
            ),
        ):
            missing = linter.check_missing_data(slug="claims")
            assert len(missing) == 1
            assert missing[0] == ("claims", "90% statistic lacks citation")

    def test_no_missing_data_returns_empty(self):
        articles = {
            "solid": {
                "body": "Well cited article.",
                "frontmatter": {"title": "Solid"},
            },
        }
        bridge = _make_bridge(articles)
        router = MagicMock()
        linter = KnowledgeLinter(bridge, model_router=router)

        with patch.object(linter, "_call_llm", return_value="[]"):
            missing = linter.check_missing_data(slug="solid")
            assert missing == []


# ---------------------------------------------------------------------------
# Full lint() integration
# ---------------------------------------------------------------------------


class TestLintIntegration:
    def test_full_lint_aggregates(self):
        old_date = (datetime.now(timezone.utc) - timedelta(days=200)).strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        )
        articles = {
            "main": {
                "body": "See [[dead-link]] and [[helper]].",
                "frontmatter": {
                    "title": "Main",
                    "updated": old_date,
                    "concepts": ["testing"],
                },
            },
            "helper": {
                "body": "Helper article, no outbound links.",
                "frontmatter": {
                    "title": "Helper",
                    "updated": old_date,
                    "concepts": ["testing"],
                },
            },
        }
        bridge = _make_bridge(articles)
        # No router = skip LLM checks
        linter = KnowledgeLinter(bridge)

        # Disable history saving (no _root on mock)
        with patch.object(linter, "_save_history"):
            report = linter.lint()

        assert len(report.dead_links) == 1  # [[dead-link]]
        # main links to helper, so helper has 1 inbound link.
        # main has 0 inbound links (helper doesn't link back) -> orphan.
        assert "main" in report.orphans
        assert "helper" not in report.orphans
        assert len(report.stale) == 2  # both are old

    def test_scoped_lint(self):
        articles = {
            "target": {
                "body": "See [[missing]].",
                "frontmatter": {"title": "Target"},
            },
            "other": {
                "body": "No links.",
                "frontmatter": {"title": "Other"},
            },
        }
        bridge = _make_bridge(articles)
        linter = KnowledgeLinter(bridge)

        with patch.object(linter, "_save_history"):
            report = linter.lint(scope="target")

        assert len(report.dead_links) == 1
        assert report.dead_links[0] == ("target", "missing")
        # Scoped lint does not check orphans
        assert report.orphans == []

    def test_empty_store_returns_clean_report(self):
        bridge = _make_bridge({})
        linter = KnowledgeLinter(bridge)

        with patch.object(linter, "_save_history"):
            report = linter.lint()

        assert report.issue_count() == 0
        assert report.overall_health == "healthy"

    def test_overall_health_computed_on_lint(self):
        # Create enough issues to be "unhealthy"
        articles = {}
        for i in range(12):
            articles[f"orphan-{i}"] = {
                "body": "No links.",
                "frontmatter": {"title": f"Orphan {i}"},
            }
        bridge = _make_bridge(articles)
        linter = KnowledgeLinter(bridge)

        with patch.object(linter, "_save_history"):
            report = linter.lint()

        assert report.overall_health == "unhealthy"
        assert report.issue_count() == 12


# ---------------------------------------------------------------------------
# _extract_json helper
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_raw_json(self):
        assert _extract_json('{"key": "value"}') == {"key": "value"}

    def test_fenced_json(self):
        assert _extract_json('```json\n{"key": "val"}\n```') == {"key": "val"}

    def test_json_array(self):
        assert _extract_json('["a", "b"]') == ["a", "b"]

    def test_json_with_surrounding_text(self):
        result = _extract_json('Here is the result: {"ok": true} done.')
        assert result == {"ok": True}

    def test_invalid_returns_none(self):
        assert _extract_json("not json at all") is None
