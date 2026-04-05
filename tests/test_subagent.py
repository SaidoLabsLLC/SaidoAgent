"""Tests for the multi-agent sub-agent system (Phase 3).

Covers:
- HIGH-4 os.chdir guard still holds
- Resource limits enforcement (token budget, turn limit, timeout)
- Inter-agent messaging (send, receive, broadcast)
- Agent type definitions loaded from .md files
- Built-in agent types parseable
- Isolated spawn creates worktree (or falls back)
- Resource usage reported on completion
"""
from __future__ import annotations

import ast
import os
import textwrap
import threading
import time
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# HIGH-4: os.chdir guard still holds
# ---------------------------------------------------------------------------


class TestHigh4ChdirGuard:
    """Verify the os.chdir() thread-safety guard remains installed."""

    def test_no_os_chdir_calls_in_subagent(self):
        """subagent.py must not contain any direct os.chdir() calls
        (other than the guard installation itself)."""
        src = Path("saido_agent/multi_agent/subagent.py").read_text()
        tree = ast.parse(src)

        chdir_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Match os.chdir(...)
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "chdir"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "os"
                ):
                    chdir_calls.append(node.lineno)

        # The only allowed os.chdir references are in the guard itself:
        #   os.chdir = _safe_chdir   (assignment, not a Call node)
        #   _original_chdir = os.chdir  (assignment, not a Call node)
        # There should be zero Call nodes for os.chdir
        assert chdir_calls == [], (
            f"os.chdir() called on lines {chdir_calls} -- "
            "use subprocess.run(cwd=...) instead"
        )

    def test_subprocess_uses_cwd_parameter(self):
        """All subprocess.run calls in subagent.py must pass cwd=."""
        src = Path("saido_agent/multi_agent/subagent.py").read_text()
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "run"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "subprocess"
                ):
                    kw_names = [kw.arg for kw in node.keywords]
                    assert "cwd" in kw_names, (
                        f"subprocess.run() on line {node.lineno} missing cwd= parameter"
                    )

    def test_chdir_guard_installed(self):
        """os.chdir should be the safe wrapper after importing subagent."""
        from saido_agent.multi_agent import subagent  # noqa: F401
        assert os.chdir is not subagent._original_chdir, (
            "os.chdir guard not installed"
        )

    def test_chdir_raises_in_non_main_thread(self):
        """os.chdir() from a non-main thread must raise RuntimeError."""
        from saido_agent.multi_agent import subagent  # noqa: F401

        error = None

        def _try_chdir():
            nonlocal error
            try:
                os.chdir(".")
            except RuntimeError as e:
                error = e

        t = threading.Thread(target=_try_chdir)
        t.start()
        t.join()
        assert error is not None, "os.chdir() did not raise in non-main thread"
        assert "non-main thread" in str(error)


# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------


class TestResourceLimits:
    """Verify resource tracking and limit enforcement."""

    def test_default_limits(self):
        from saido_agent.multi_agent.resources import AgentResourceLimits
        lim = AgentResourceLimits()
        assert lim.max_tokens == 100_000
        assert lim.max_turns == 50
        assert lim.timeout_seconds == 300
        assert lim.max_tool_calls == 100

    def test_custom_limits(self):
        from saido_agent.multi_agent.resources import AgentResourceLimits
        lim = AgentResourceLimits(max_tokens=500, max_turns=3, timeout_seconds=10, max_tool_calls=5)
        assert lim.max_tokens == 500
        assert lim.max_turns == 3

    def test_usage_tracks_tokens(self):
        from saido_agent.multi_agent.resources import AgentResourceUsage
        usage = AgentResourceUsage()
        usage.add_tokens(100)
        usage.add_tokens(200)
        assert usage.tokens_used == 300

    def test_usage_tracks_turns(self):
        from saido_agent.multi_agent.resources import AgentResourceUsage
        usage = AgentResourceUsage()
        usage.add_turn()
        usage.add_turn()
        assert usage.turns_used == 2

    def test_usage_tracks_tool_calls(self):
        from saido_agent.multi_agent.resources import AgentResourceUsage
        usage = AgentResourceUsage()
        usage.add_tool_call()
        assert usage.tool_calls_used == 1

    def test_token_limit_exceeded(self):
        from saido_agent.multi_agent.resources import (
            AgentResourceLimits,
            AgentResourceUsage,
        )
        lim = AgentResourceLimits(max_tokens=100)
        usage = AgentResourceUsage()
        usage.add_tokens(100)
        reason = usage.check_limits(lim)
        assert reason is not None
        assert "token budget exceeded" in reason

    def test_turn_limit_exceeded(self):
        from saido_agent.multi_agent.resources import (
            AgentResourceLimits,
            AgentResourceUsage,
        )
        lim = AgentResourceLimits(max_turns=2)
        usage = AgentResourceUsage()
        usage.add_turn()
        usage.add_turn()
        reason = usage.check_limits(lim)
        assert reason is not None
        assert "turn limit exceeded" in reason

    def test_tool_call_limit_exceeded(self):
        from saido_agent.multi_agent.resources import (
            AgentResourceLimits,
            AgentResourceUsage,
        )
        lim = AgentResourceLimits(max_tool_calls=1)
        usage = AgentResourceUsage()
        usage.add_tool_call()
        reason = usage.check_limits(lim)
        assert reason is not None
        assert "tool-call limit exceeded" in reason

    def test_timeout_exceeded(self):
        from saido_agent.multi_agent.resources import (
            AgentResourceLimits,
            AgentResourceUsage,
        )
        lim = AgentResourceLimits(timeout_seconds=0)
        usage = AgentResourceUsage()
        # start_time is already set, and timeout is 0, so it's exceeded
        time.sleep(0.01)
        reason = usage.check_limits(lim)
        assert reason is not None
        assert "timeout exceeded" in reason

    def test_no_limit_exceeded(self):
        from saido_agent.multi_agent.resources import (
            AgentResourceLimits,
            AgentResourceUsage,
        )
        lim = AgentResourceLimits()
        usage = AgentResourceUsage()
        usage.add_tokens(10)
        usage.add_turn()
        assert usage.check_limits(lim) is None

    def test_resource_tracker_register_and_query(self):
        from saido_agent.multi_agent.resources import (
            AgentResourceLimits,
            ResourceTracker,
        )
        tracker = ResourceTracker()
        usage = tracker.register("agent-1", AgentResourceLimits(max_tokens=50))
        assert usage is not None
        assert tracker.get_usage("agent-1") is usage
        assert tracker.get_limits("agent-1").max_tokens == 50

    def test_resource_tracker_record_and_check(self):
        from saido_agent.multi_agent.resources import (
            AgentResourceLimits,
            ResourceTracker,
        )
        tracker = ResourceTracker()
        tracker.register("agent-1", AgentResourceLimits(max_tokens=50))
        assert tracker.record_tokens("agent-1", 30) is None
        reason = tracker.record_tokens("agent-1", 30)
        assert reason is not None
        assert "token budget exceeded" in reason

    def test_resource_tracker_finish_returns_summary(self):
        from saido_agent.multi_agent.resources import (
            AgentResourceLimits,
            ResourceTracker,
        )
        tracker = ResourceTracker()
        tracker.register("agent-1", AgentResourceLimits(max_tokens=1000))
        tracker.record_tokens("agent-1", 42)
        tracker.record_turn("agent-1")
        summary = tracker.finish("agent-1")
        assert summary is not None
        assert summary["tokens_used"] == 42
        assert summary["turns_used"] == 1
        assert "elapsed_seconds" in summary
        assert "limits" in summary

    def test_usage_summary_includes_exceeded(self):
        from saido_agent.multi_agent.resources import (
            AgentResourceLimits,
            AgentResourceUsage,
        )
        usage = AgentResourceUsage()
        usage.exceeded_limit = "token budget exceeded (100/100)"
        s = usage.summary(AgentResourceLimits())
        assert "exceeded_limit" in s


# ---------------------------------------------------------------------------
# Inter-agent messaging
# ---------------------------------------------------------------------------


class TestAgentMessaging:
    """Verify the AgentInbox send/receive/broadcast system."""

    def test_send_and_receive(self):
        from saido_agent.multi_agent.messaging import AgentInbox
        inbox = AgentInbox()
        inbox.register("a1")
        inbox.register("a2")
        assert inbox.send("a1", "a2", "hello") is True
        msgs = inbox.receive("a2")
        assert len(msgs) == 1
        assert msgs[0]["from"] == "a1"
        assert msgs[0]["body"] == "hello"
        assert msgs[0]["broadcast"] is False

    def test_receive_advances_cursor(self):
        from saido_agent.multi_agent.messaging import AgentInbox
        inbox = AgentInbox()
        inbox.register("a1")
        inbox.send("other", "a1", "msg1")
        inbox.send("other", "a1", "msg2")
        first = inbox.receive("a1")
        assert len(first) == 2
        # Second call returns nothing new
        second = inbox.receive("a1")
        assert len(second) == 0

    def test_send_to_unregistered_returns_false(self):
        from saido_agent.multi_agent.messaging import AgentInbox
        inbox = AgentInbox()
        assert inbox.send("a1", "nonexistent", "hello") is False

    def test_broadcast(self):
        from saido_agent.multi_agent.messaging import AgentInbox
        inbox = AgentInbox()
        inbox.register("a1")
        inbox.register("a2")
        inbox.register("a3")
        count = inbox.broadcast("a1", "announcement")
        assert count == 2  # a2 and a3, not a1
        msgs_a2 = inbox.receive("a2")
        msgs_a3 = inbox.receive("a3")
        msgs_a1 = inbox.receive("a1")
        assert len(msgs_a2) == 1
        assert msgs_a2[0]["broadcast"] is True
        assert len(msgs_a3) == 1
        assert len(msgs_a1) == 0  # sender does not receive own broadcast

    def test_peek_count(self):
        from saido_agent.multi_agent.messaging import AgentInbox
        inbox = AgentInbox()
        inbox.register("a1")
        assert inbox.peek("a1") == 0
        inbox.send("other", "a1", "msg1")
        assert inbox.peek("a1") == 1
        inbox.receive("a1")
        assert inbox.peek("a1") == 0

    def test_unregister_clears_queue(self):
        from saido_agent.multi_agent.messaging import AgentInbox
        inbox = AgentInbox()
        inbox.register("a1")
        inbox.send("other", "a1", "msg1")
        inbox.unregister("a1")
        assert "a1" not in inbox.registered_agents
        # Re-register gets fresh queue
        inbox.register("a1")
        assert inbox.receive("a1") == []

    def test_thread_safety(self):
        """Multiple threads sending concurrently should not corrupt state."""
        from saido_agent.multi_agent.messaging import AgentInbox
        inbox = AgentInbox()
        inbox.register("target")
        num_senders = 10
        msgs_per_sender = 50

        def _send(sender_id):
            for i in range(msgs_per_sender):
                inbox.send(f"sender-{sender_id}", "target", f"msg-{i}")

        threads = [threading.Thread(target=_send, args=(i,)) for i in range(num_senders)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        all_msgs = inbox.receive("target")
        assert len(all_msgs) == num_senders * msgs_per_sender


# ---------------------------------------------------------------------------
# Agent type definitions from .md files
# ---------------------------------------------------------------------------


class TestAgentDefinitions:
    """Verify agent definitions load from .md files with YAML frontmatter."""

    def test_builtin_agents_exist(self):
        from saido_agent.multi_agent.subagent import _BUILTIN_AGENTS
        assert "coder" in _BUILTIN_AGENTS
        assert "reviewer" in _BUILTIN_AGENTS
        assert "researcher" in _BUILTIN_AGENTS

    def test_parse_agent_md_basic(self, tmp_path):
        from saido_agent.multi_agent.subagent import _parse_agent_md
        md_file = tmp_path / "test_agent.md"
        md_file.write_text(textwrap.dedent("""\
            ---
            name: test_agent
            description: A test agent
            tools: [Read, Write]
            max_tokens: 5000
            max_turns: 10
            ---
            You are a test agent.
        """))
        defn = _parse_agent_md(md_file, source="test")
        assert defn.name == "test_agent"
        assert defn.description == "A test agent"
        assert defn.tools == ["Read", "Write"]
        assert defn.source == "test"
        assert "You are a test agent." in defn.system_prompt

    def test_parse_agent_md_resource_limits(self, tmp_path):
        from saido_agent.multi_agent.subagent import _parse_agent_md
        md_file = tmp_path / "limited.md"
        md_file.write_text(textwrap.dedent("""\
            ---
            name: limited
            description: Agent with resource limits
            max_tokens: 2000
            max_turns: 5
            max_tool_calls: 10
            timeout_seconds: 60
            ---
            Limited agent.
        """))
        defn = _parse_agent_md(md_file, source="test")
        assert defn.resource_limits is not None
        assert defn.resource_limits.max_tokens == 2000
        assert defn.resource_limits.max_turns == 5
        assert defn.resource_limits.max_tool_calls == 10
        assert defn.resource_limits.timeout_seconds == 60

    def test_parse_agent_md_no_frontmatter(self, tmp_path):
        from saido_agent.multi_agent.subagent import _parse_agent_md
        md_file = tmp_path / "plain.md"
        md_file.write_text("Just a system prompt with no frontmatter.")
        defn = _parse_agent_md(md_file, source="test")
        assert defn.name == "plain"
        assert defn.description == ""
        assert defn.system_prompt == "Just a system prompt with no frontmatter."
        assert defn.resource_limits is None

    def test_load_agent_definitions_includes_builtins(self):
        from saido_agent.multi_agent.subagent import load_agent_definitions
        defs = load_agent_definitions()
        assert "coder" in defs
        assert "reviewer" in defs
        assert "researcher" in defs

    def test_load_from_user_dir(self, tmp_path, monkeypatch):
        from saido_agent.multi_agent.subagent import load_agent_definitions
        agent_dir = tmp_path / ".saido_agent" / "agents"
        agent_dir.mkdir(parents=True)
        (agent_dir / "custom.md").write_text(textwrap.dedent("""\
            ---
            name: custom
            description: Custom agent
            ---
            Custom system prompt.
        """))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        defs = load_agent_definitions()
        assert "custom" in defs
        assert defs["custom"].source == "user"


# ---------------------------------------------------------------------------
# Built-in agent .md files parseable
# ---------------------------------------------------------------------------


class TestBuiltinAgentFiles:
    """Verify the .md files in saido_agent/multi_agent/agents/ parse correctly."""

    @pytest.mark.parametrize("agent_name", ["researcher", "coder", "reviewer", "writer"])
    def test_builtin_md_parseable(self, agent_name):
        from saido_agent.multi_agent.subagent import _parse_agent_md
        md_path = Path("saido_agent/multi_agent/agents") / f"{agent_name}.md"
        assert md_path.exists(), f"{md_path} does not exist"
        defn = _parse_agent_md(md_path, source="built-in")
        assert defn.name == agent_name
        assert defn.description != ""
        assert len(defn.tools) > 0
        assert defn.system_prompt != ""

    @pytest.mark.parametrize("agent_name", ["researcher", "coder", "reviewer", "writer"])
    def test_builtin_md_has_resource_limits(self, agent_name):
        from saido_agent.multi_agent.subagent import _parse_agent_md
        md_path = Path("saido_agent/multi_agent/agents") / f"{agent_name}.md"
        defn = _parse_agent_md(md_path, source="built-in")
        assert defn.resource_limits is not None
        assert defn.resource_limits.max_tokens > 0
        assert defn.resource_limits.max_turns > 0


# ---------------------------------------------------------------------------
# Isolated spawn (worktree or fallback)
# ---------------------------------------------------------------------------


class TestSpawnIsolated:
    """Verify spawn_isolated creates worktree or falls back."""

    def test_spawn_isolated_falls_back_outside_git(self):
        """When not in a git repo, spawn_isolated should fall back to
        regular spawn (no worktree) without error."""
        from saido_agent.multi_agent.subagent import SubAgentManager

        mgr = SubAgentManager()
        with mock.patch("saido_agent.multi_agent.subagent._git_root", return_value=None):
            with mock.patch.object(mgr, "spawn") as mock_spawn:
                mock_task = mock.MagicMock()
                mock_task.id = "test123"
                mock_spawn.return_value = mock_task
                task_id = mgr.spawn_isolated("coder", "do something")
                assert task_id == "test123"
                # isolation should be "" (no worktree)
                call_kwargs = mock_spawn.call_args
                assert call_kwargs[1].get("isolation", "") == "" or call_kwargs.kwargs.get("isolation", "") == ""

    def test_spawn_isolated_uses_worktree_in_git_repo(self):
        """When in a git repo, spawn_isolated should request worktree isolation."""
        from saido_agent.multi_agent.subagent import SubAgentManager

        mgr = SubAgentManager()
        with mock.patch("saido_agent.multi_agent.subagent._git_root", return_value="/fake/repo"):
            with mock.patch.object(mgr, "spawn") as mock_spawn:
                mock_task = mock.MagicMock()
                mock_task.id = "test456"
                mock_spawn.return_value = mock_task
                task_id = mgr.spawn_isolated("coder", "do something")
                assert task_id == "test456"
                call_kwargs = mock_spawn.call_args
                assert call_kwargs.kwargs.get("isolation") == "worktree"


# ---------------------------------------------------------------------------
# Resource usage reported on completion
# ---------------------------------------------------------------------------


class TestResourceReporting:
    """Verify resource usage is reported when tasks complete."""

    def test_resource_tracker_on_manager(self):
        from saido_agent.multi_agent.subagent import SubAgentManager
        mgr = SubAgentManager()
        assert mgr.resource_tracker is not None
        assert mgr.inbox is not None

    def test_task_gets_resource_limits(self):
        """Spawned tasks should have resource_limits and resource_usage attached."""
        from saido_agent.multi_agent.subagent import SubAgentManager
        from saido_agent.multi_agent.resources import AgentResourceLimits

        mgr = SubAgentManager()

        # Mock spawn internals to avoid needing real agent
        with mock.patch("saido_agent.multi_agent.subagent._agent_run"):
            task = mgr.spawn(
                prompt="test",
                config={},
                system_prompt="test",
                resource_limits=AgentResourceLimits(max_tokens=500),
            )
            assert task.resource_limits is not None
            assert task.resource_limits.max_tokens == 500
            assert task.resource_usage is not None

    def test_max_depth_reports_resources(self):
        """A task that fails due to max_depth should still get a resource summary."""
        from saido_agent.multi_agent.subagent import SubAgentManager

        mgr = SubAgentManager(max_depth=2)
        task = mgr.spawn(
            prompt="test",
            config={},
            system_prompt="test",
            depth=5,
        )
        assert task.status == "failed"
        assert "Max depth" in task.result
        assert task.resource_summary is not None
        assert "tokens_used" in task.resource_summary

    def test_get_resource_summary(self):
        """get_resource_summary returns the task's resource summary."""
        from saido_agent.multi_agent.subagent import SubAgentManager

        mgr = SubAgentManager(max_depth=1)
        task = mgr.spawn(prompt="test", config={}, system_prompt="test", depth=5)
        summary = mgr.get_resource_summary(task.id)
        assert summary is not None
        assert "elapsed_seconds" in summary

    def test_get_resource_summary_missing_task(self):
        from saido_agent.multi_agent.subagent import SubAgentManager
        mgr = SubAgentManager()
        assert mgr.get_resource_summary("nonexistent") is None
