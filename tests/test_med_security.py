"""Tests for medium-severity security patches (MED-1 through MED-4).

MED-1: SSRF Protection on WebFetch
MED-2: Regex DoS Protection
MED-3: API Rate Limiting / Token Budget
MED-4: Memory Trust Boundary
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# MED-1: SSRF validation blocks private IPs in WebFetch context
# ---------------------------------------------------------------------------

class TestSSRFProtection:
    """Verify that SSRF validation blocks private/reserved IPs."""

    def test_blocks_localhost_127(self):
        from saido_agent.core.ssrf import validate_url_no_resolve
        ok, reason = validate_url_no_resolve("http://127.0.0.1/admin")
        assert not ok
        assert "Blocked" in reason

    def test_blocks_localhost_ipv6(self):
        from saido_agent.core.ssrf import validate_url_no_resolve
        ok, reason = validate_url_no_resolve("http://[::1]/admin")
        assert not ok
        assert "Blocked" in reason

    def test_blocks_private_10(self):
        from saido_agent.core.ssrf import validate_url_no_resolve
        ok, reason = validate_url_no_resolve("http://10.0.0.1/secret")
        assert not ok

    def test_blocks_private_172(self):
        from saido_agent.core.ssrf import validate_url_no_resolve
        ok, reason = validate_url_no_resolve("http://172.16.0.1/internal")
        assert not ok

    def test_blocks_private_192(self):
        from saido_agent.core.ssrf import validate_url_no_resolve
        ok, reason = validate_url_no_resolve("http://192.168.1.1/router")
        assert not ok

    def test_blocks_cloud_metadata(self):
        from saido_agent.core.ssrf import validate_url_no_resolve
        ok, reason = validate_url_no_resolve("http://169.254.169.254/latest/meta-data/")
        assert not ok
        assert "Blocked" in reason

    def test_blocks_gcp_metadata(self):
        from saido_agent.core.ssrf import validate_url
        ok, reason = validate_url("http://metadata.google.internal/computeMetadata/v1/")
        assert not ok

    def test_allows_public_url(self):
        from saido_agent.core.ssrf import validate_url_no_resolve
        ok, reason = validate_url_no_resolve("https://example.com/page")
        assert ok

    def test_blocks_ftp_scheme(self):
        from saido_agent.core.ssrf import validate_url_no_resolve
        ok, reason = validate_url_no_resolve("ftp://files.internal/data")
        assert not ok

    def test_blocks_file_scheme(self):
        from saido_agent.core.ssrf import validate_url_no_resolve
        ok, reason = validate_url_no_resolve("file:///etc/passwd")
        assert not ok

    def test_webfetch_blocks_private_ip(self):
        """Verify _webfetch integrates SSRF check."""
        from saido_agent.core.tools import _webfetch
        result = _webfetch("http://127.0.0.1:8080/admin")
        assert "SSRF" in result or "Blocked" in result

    def test_webfetch_blocks_metadata(self):
        from saido_agent.core.tools import _webfetch
        result = _webfetch("http://169.254.169.254/latest/meta-data/")
        assert "SSRF" in result or "Blocked" in result


# ---------------------------------------------------------------------------
# MED-2: Regex DoS Protection
# ---------------------------------------------------------------------------

class TestRegexDoSProtection:
    """Verify that dangerous regex patterns are rejected."""

    def test_rejects_nested_plus(self):
        from saido_agent.core.tools import _validate_regex
        ok, reason = _validate_regex("(a+)+")
        assert not ok
        assert "backtracking" in reason.lower() or "nested" in reason.lower()

    def test_rejects_nested_star(self):
        from saido_agent.core.tools import _validate_regex
        ok, reason = _validate_regex("(a*)*")
        assert not ok

    def test_rejects_alternation_nested(self):
        from saido_agent.core.tools import _validate_regex
        ok, reason = _validate_regex("(a|b+)+")
        assert not ok

    def test_rejects_star_plus_combo(self):
        from saido_agent.core.tools import _validate_regex
        ok, reason = _validate_regex("(a+)*")
        assert not ok

    def test_allows_simple_pattern(self):
        from saido_agent.core.tools import _validate_regex
        ok, reason = _validate_regex(r"def\s+\w+")
        assert ok

    def test_allows_character_class_quantifier(self):
        from saido_agent.core.tools import _validate_regex
        ok, reason = _validate_regex(r"[a-z]+")
        assert ok

    def test_allows_alternation_without_nesting(self):
        from saido_agent.core.tools import _validate_regex
        ok, reason = _validate_regex(r"foo|bar|baz")
        assert ok

    def test_allows_non_nested_group(self):
        from saido_agent.core.tools import _validate_regex
        ok, reason = _validate_regex(r"(hello)+")
        assert ok

    def test_rejects_invalid_regex(self):
        from saido_agent.core.tools import _validate_regex
        ok, reason = _validate_regex(r"(unclosed")
        assert not ok
        assert "Invalid" in reason

    def test_grep_rejects_redos_pattern(self):
        """Verify _grep integrates regex validation."""
        from saido_agent.core.tools import _grep
        result = _grep("(a+)+")
        assert "Error" in result


# ---------------------------------------------------------------------------
# MED-3: Token Budget
# ---------------------------------------------------------------------------

class TestTokenBudget:
    """Verify token budget enforcement and warnings."""

    def _make_tracker(self, max_tokens=1000, max_turns=10):
        from saido_agent.core.cost_tracker import CostTracker
        tracker = CostTracker()
        tracker.set_budget(max_tokens=max_tokens, max_turns=max_turns)
        return tracker

    def test_warns_at_80_percent_tokens(self):
        tracker = self._make_tracker(max_tokens=1000)
        tracker.record("ollama", "llama3", 800, 0)
        msg, can_continue = tracker.check_budget()
        assert "WARNING" in msg
        assert can_continue is True

    def test_blocks_at_100_percent_tokens(self):
        tracker = self._make_tracker(max_tokens=1000)
        tracker.record("ollama", "llama3", 1000, 0)
        msg, can_continue = tracker.check_budget()
        assert "EXCEEDED" in msg
        assert can_continue is False

    def test_blocks_over_budget(self):
        tracker = self._make_tracker(max_tokens=1000)
        tracker.record("ollama", "llama3", 1200, 0)
        msg, can_continue = tracker.check_budget()
        assert "EXCEEDED" in msg
        assert can_continue is False

    def test_max_turns_enforcement(self):
        tracker = self._make_tracker(max_turns=10)
        for _ in range(10):
            tracker.record_turn()
        msg, can_continue = tracker.check_budget()
        assert "TURN LIMIT" in msg
        assert can_continue is False

    def test_turns_warn_at_80(self):
        tracker = self._make_tracker(max_turns=10)
        for _ in range(8):
            tracker.record_turn()
        msg, can_continue = tracker.check_budget()
        assert "WARNING" in msg
        assert can_continue is True

    def test_under_budget_no_message(self):
        tracker = self._make_tracker(max_tokens=1000, max_turns=10)
        tracker.record("ollama", "llama3", 100, 0)
        tracker.record_turn()
        msg, can_continue = tracker.check_budget()
        assert msg == ""
        assert can_continue is True

    def test_budget_command_output(self):
        tracker = self._make_tracker(max_tokens=10000, max_turns=100)
        tracker.record("ollama", "llama3", 2500, 500)
        for _ in range(25):
            tracker.record_turn()
        report = tracker.format_budget()
        assert "Session budget:" in report
        assert "Tokens:" in report
        assert "Turns:" in report
        assert "remaining" in report

    def test_confirm_budget_override(self):
        tracker = self._make_tracker(max_tokens=1000)
        tracker.record("ollama", "llama3", 1000, 0)
        msg, can_continue = tracker.check_budget()
        assert can_continue is False
        tracker.confirm_budget_override()
        msg, can_continue = tracker.check_budget()
        assert can_continue is True

    def test_set_budget(self):
        tracker = self._make_tracker()
        tracker.set_budget(max_tokens=500, max_turns=50)
        assert tracker._max_tokens == 500
        assert tracker._max_turns == 50


# ---------------------------------------------------------------------------
# MED-4: Memory Trust Boundary
# ---------------------------------------------------------------------------

class TestMemoryTrustBoundary:
    """Verify memory trust boundary for project-scoped memories."""

    def test_is_trusted_returns_false_for_unknown(self):
        from saido_agent.memory.store import is_trusted_project
        with tempfile.TemporaryDirectory() as td:
            # Use a fake trusted_projects file
            with patch("saido_agent.memory.store.TRUSTED_PROJECTS_FILE",
                       Path(td) / "trusted.json"):
                assert is_trusted_project("/some/random/dir") is False

    def test_trust_project_persists(self):
        from saido_agent.memory.store import is_trusted_project, trust_project
        with tempfile.TemporaryDirectory() as td:
            trusted_file = Path(td) / "trusted.json"
            with patch("saido_agent.memory.store.TRUSTED_PROJECTS_FILE", trusted_file):
                project_dir = Path(td) / "my_project"
                project_dir.mkdir()
                trust_project(str(project_dir))
                assert is_trusted_project(str(project_dir)) is True
                # Verify it's persisted to disk
                data = json.loads(trusted_file.read_text())
                assert str(project_dir.resolve()) in data

    def test_check_project_trust_warns_new_dir(self):
        """Untrusted dir with no prompt_fn returns False."""
        from saido_agent.memory.store import check_project_trust
        with tempfile.TemporaryDirectory() as td:
            with patch("saido_agent.memory.store.TRUSTED_PROJECTS_FILE",
                       Path(td) / "trusted.json"):
                result = check_project_trust("/some/new/project")
                assert result is False

    def test_check_project_trust_prompts_and_approves(self):
        """When prompt_fn returns True, directory gets trusted."""
        from saido_agent.memory.store import check_project_trust, is_trusted_project
        with tempfile.TemporaryDirectory() as td:
            trusted_file = Path(td) / "trusted.json"
            project_dir = Path(td) / "new_project"
            project_dir.mkdir()
            with patch("saido_agent.memory.store.TRUSTED_PROJECTS_FILE", trusted_file):
                prompted = []
                def mock_prompt(msg):
                    prompted.append(msg)
                    return True
                result = check_project_trust(str(project_dir), prompt_fn=mock_prompt)
                assert result is True
                assert len(prompted) == 1
                assert "not created by Saido Agent" in prompted[0]
                # Now trusted
                assert is_trusted_project(str(project_dir)) is True

    def test_check_project_trust_prompts_and_rejects(self):
        """When prompt_fn returns False, directory stays untrusted."""
        from saido_agent.memory.store import check_project_trust, is_trusted_project
        with tempfile.TemporaryDirectory() as td:
            trusted_file = Path(td) / "trusted.json"
            project_dir = Path(td) / "untrusted_project"
            project_dir.mkdir()
            with patch("saido_agent.memory.store.TRUSTED_PROJECTS_FILE", trusted_file):
                result = check_project_trust(
                    str(project_dir), prompt_fn=lambda msg: False
                )
                assert result is False
                assert is_trusted_project(str(project_dir)) is False

    def test_trusted_dir_proceeds_without_prompt(self):
        """Already-trusted directory does not trigger prompt."""
        from saido_agent.memory.store import check_project_trust, trust_project
        with tempfile.TemporaryDirectory() as td:
            trusted_file = Path(td) / "trusted.json"
            project_dir = Path(td) / "already_trusted"
            project_dir.mkdir()
            with patch("saido_agent.memory.store.TRUSTED_PROJECTS_FILE", trusted_file):
                trust_project(str(project_dir))
                prompted = []
                result = check_project_trust(
                    str(project_dir),
                    prompt_fn=lambda msg: (prompted.append(msg) or True),
                )
                assert result is True
                assert len(prompted) == 0  # No prompt needed

    def test_user_memory_dir_always_trusted(self):
        """User-level memory directory is always trusted."""
        from saido_agent.memory.store import check_project_trust
        user_dir = Path.home() / ".saido_agent" / "memory"
        with tempfile.TemporaryDirectory() as td:
            with patch("saido_agent.memory.store.TRUSTED_PROJECTS_FILE",
                       Path(td) / "trusted.json"):
                result = check_project_trust(str(user_dir))
                assert result is True
