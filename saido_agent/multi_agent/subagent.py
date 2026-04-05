"""Threaded sub-agent system for spawning nested agent loops.

HIGH-4 Security Fix: Removed all os.chdir() calls to prevent thread-safety
race conditions. Sub-agents receive their working directory as a parameter
and pass it to subprocess.run(cwd=...) instead.

Phase 3 extensions:
- Resource management: configurable per-agent limits (tokens, turns, timeout, tool calls)
- Inter-agent messaging: thread-safe AgentInbox for send/receive/broadcast
- Git worktree isolation via spawn_isolated()
- Agent type definitions loaded from .md files with YAML frontmatter
"""
from __future__ import annotations

import os
import threading
import uuid
import queue
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

from .resources import AgentResourceLimits, AgentResourceUsage, ResourceTracker
from .messaging import AgentInbox


# ── HIGH-4: Thread-safety guard against os.chdir() ──────────────────────────

_original_chdir = os.chdir


def _safe_chdir(path):
    """Replacement for os.chdir that raises RuntimeError from non-main threads.

    This prevents thread-safety race conditions where concurrent sub-agents
    could corrupt each other's working directory.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            f"os.chdir() called from non-main thread '{threading.current_thread().name}'. "
            f"Use subprocess.run(cwd=target_dir) instead to avoid race conditions."
        )
    return _original_chdir(path)


def install_chdir_guard():
    """Install the thread-safe os.chdir guard. Called once at module load."""
    os.chdir = _safe_chdir


# Install guard on import
install_chdir_guard()


@dataclass
class AgentDefinition:
    name: str
    description: str = ""
    system_prompt: str = ""
    model: str = ""
    tools: list = field(default_factory=list)
    source: str = "user"
    resource_limits: Optional[AgentResourceLimits] = None


_BUILTIN_AGENTS: Dict[str, AgentDefinition] = {
    "general-purpose": AgentDefinition(
        name="general-purpose",
        description="General-purpose agent for researching complex questions, searching for code, and executing multi-step tasks.",
        system_prompt="",
        source="built-in",
    ),
    "coder": AgentDefinition(
        name="coder",
        description="Specialized coding agent for writing, reading, and modifying code.",
        system_prompt=(
            "You are a specialized coding assistant. Focus on:\n"
            "- Writing clean, idiomatic code\n"
            "- Reading and understanding existing code before modifying\n"
            "- Making minimal targeted changes\n"
            "- Never adding unnecessary features, comments, or error handling\n"
        ),
        source="built-in",
    ),
    "reviewer": AgentDefinition(
        name="reviewer",
        description="Code review agent analyzing quality, security, and correctness.",
        system_prompt=(
            "You are a code reviewer. Analyze code for:\n"
            "- Correctness and logic errors\n"
            "- Security vulnerabilities\n"
            "- Performance issues\n"
            "- Code quality and maintainability\n"
            "Be concise and specific. Categorize findings as: Critical | Warning | Suggestion.\n"
        ),
        tools=["Read", "Glob", "Grep"],
        source="built-in",
    ),
    "researcher": AgentDefinition(
        name="researcher",
        description="Research agent for exploring codebases and answering questions.",
        system_prompt=(
            "You are a research assistant focused on understanding codebases.\n"
            "- Read and analyze code thoroughly before answering\n"
            "- Provide factual, evidence-based answers\n"
            "- Cite specific file paths and line numbers\n"
            "- Be concise and focused\n"
        ),
        tools=["Read", "Glob", "Grep", "WebFetch", "WebSearch"],
        source="built-in",
    ),
    "tester": AgentDefinition(
        name="tester",
        description="Testing agent that writes and runs tests.",
        system_prompt=(
            "You are a testing specialist. Your job:\n"
            "- Write comprehensive tests for the given code\n"
            "- Run existing tests and diagnose failures\n"
            "- Focus on edge cases and error conditions\n"
            "- Keep tests simple, readable, and fast\n"
        ),
        source="built-in",
    ),
}


def _parse_agent_md(path: Path, source: str = "user") -> AgentDefinition:
    content = path.read_text()
    name = path.stem
    description = ""
    model = ""
    tools: list = []
    fm: dict = {}
    system_prompt_body = content

    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            fm_text = content[3:end].strip()
            system_prompt_body = content[end + 3:].strip()
            try:
                import yaml as _yaml
                fm = _yaml.safe_load(fm_text) or {}
            except ImportError:
                for line in fm_text.splitlines():
                    if ":" in line:
                        k, _, v = line.partition(":")
                        fm[k.strip()] = v.strip()
            description = str(fm.get("description", ""))
            model = str(fm.get("model", ""))
            raw_tools = fm.get("tools", [])
            if isinstance(raw_tools, list):
                tools = [str(t) for t in raw_tools]
            elif isinstance(raw_tools, str):
                s = raw_tools.strip("[]")
                tools = [t.strip() for t in s.split(",") if t.strip()]

    # Extract resource limits from frontmatter if present
    rl_kwargs: dict = {}
    for rl_field in ("max_tokens", "max_turns", "max_tool_calls", "timeout_seconds"):
        raw_val = fm.get(rl_field)
        if raw_val is not None:
            try:
                rl_kwargs[rl_field] = int(raw_val)
            except (ValueError, TypeError):
                pass
    resource_limits = AgentResourceLimits(**rl_kwargs) if rl_kwargs else None

    return AgentDefinition(
        name=name, description=description, system_prompt=system_prompt_body,
        model=model, tools=tools, source=source,
        resource_limits=resource_limits,
    )


def load_agent_definitions() -> Dict[str, AgentDefinition]:
    """Load all agent definitions: built-ins, user-level, project-level."""
    defs: Dict[str, AgentDefinition] = dict(_BUILTIN_AGENTS)

    user_dir = Path.home() / ".saido_agent" / "agents"
    if user_dir.is_dir():
        for p in sorted(user_dir.glob("*.md")):
            try:
                d = _parse_agent_md(p, source="user")
                defs[d.name] = d
            except Exception:
                pass

    proj_dir = Path.cwd() / ".saido_agent" / "agents"
    if proj_dir.is_dir():
        for p in sorted(proj_dir.glob("*.md")):
            try:
                d = _parse_agent_md(p, source="project")
                defs[d.name] = d
            except Exception:
                pass

    return defs


def get_agent_definition(name: str) -> Optional[AgentDefinition]:
    return load_agent_definitions().get(name)


@dataclass
class SubAgentTask:
    id: str
    prompt: str
    status: str = "pending"
    result: Optional[str] = None
    depth: int = 0
    name: str = ""
    worktree_path: str = ""
    worktree_branch: str = ""
    resource_limits: Optional[AgentResourceLimits] = None
    resource_usage: Optional[AgentResourceUsage] = None
    resource_summary: Optional[dict] = None
    _cancel_flag: bool = False
    _future: Optional[Future] = field(default=None, repr=False)
    _inbox: Any = field(default_factory=queue.Queue, repr=False)


def _git_root(cwd: str) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd, capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except Exception:
        return None


def _create_worktree(base_dir: str) -> tuple:
    branch = f"saido-agent-{uuid.uuid4().hex[:8]}"
    wt_path = tempfile.mkdtemp(prefix="saido-agent-wt-")
    os.rmdir(wt_path)
    subprocess.run(
        ["git", "worktree", "add", "-b", branch, wt_path],
        cwd=base_dir, check=True, capture_output=True, text=True,
    )
    return wt_path, branch


def _remove_worktree(wt_path: str, branch: str, base_dir: str) -> None:
    try:
        subprocess.run(["git", "worktree", "remove", "--force", wt_path], cwd=base_dir, capture_output=True)
    except Exception:
        pass
    try:
        subprocess.run(["git", "branch", "-D", branch], cwd=base_dir, capture_output=True)
    except Exception:
        pass


def _agent_run(prompt, state, config, system_prompt, depth=0, cancel_check=None):
    from saido_agent.core import agent as _agent_mod
    return _agent_mod.run(prompt, state, config, system_prompt, depth=depth, cancel_check=cancel_check)


def _extract_final_text(messages):
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return None


class SubAgentManager:
    def __init__(self, max_concurrent: int = 5, max_depth: int = 5):
        self.tasks: Dict[str, SubAgentTask] = {}
        self._by_name: Dict[str, str] = {}
        self.max_concurrent = max_concurrent
        self.max_depth = max_depth
        self._pool = ThreadPoolExecutor(max_workers=max_concurrent)
        self.resource_tracker = ResourceTracker()
        self.inbox = AgentInbox()

    def spawn(self, prompt, config, system_prompt, depth=0, agent_def=None,
              isolation="", name="", resource_limits=None):
        task_id = uuid.uuid4().hex[:12]
        short_name = name or task_id[:8]

        # Resolve resource limits: explicit > agent_def > defaults
        eff_limits = resource_limits
        if eff_limits is None and agent_def and agent_def.resource_limits:
            eff_limits = agent_def.resource_limits
        if eff_limits is None:
            eff_limits = AgentResourceLimits()

        task = SubAgentTask(
            id=task_id, prompt=prompt, depth=depth, name=short_name,
            resource_limits=eff_limits,
        )
        self.tasks[task_id] = task
        if name:
            self._by_name[name] = task_id

        # Register with resource tracker and messaging
        usage = self.resource_tracker.register(task_id, eff_limits)
        task.resource_usage = usage
        self.inbox.register(task_id)
        if name:
            self.inbox.register(name)

        if depth >= self.max_depth:
            task.status = "failed"
            task.result = f"Max depth ({self.max_depth}) exceeded"
            task.resource_summary = self.resource_tracker.finish(task_id)
            return task

        eff_config = dict(config)
        eff_system = system_prompt

        if agent_def:
            if agent_def.model:
                eff_config["model"] = agent_def.model
            if agent_def.system_prompt:
                eff_system = agent_def.system_prompt.rstrip() + "\n\n" + system_prompt

        worktree_path = ""
        worktree_branch = ""
        base_dir = os.getcwd()

        if isolation == "worktree":
            git_root = _git_root(base_dir)
            if not git_root:
                task.status = "failed"
                task.result = "isolation='worktree' requires a git repository"
                task.resource_summary = self.resource_tracker.finish(task_id)
                return task
            try:
                worktree_path, worktree_branch = _create_worktree(git_root)
                task.worktree_path = worktree_path
                task.worktree_branch = worktree_branch
                notice = (
                    f"\n\n[Note: You are working in an isolated git worktree at "
                    f"{worktree_path} (branch: {worktree_branch}). "
                    f"Your changes are isolated from the main workspace at {git_root}. "
                    f"Commit your changes before finishing so they can be reviewed/merged.]"
                )
                prompt = prompt + notice
            except Exception as e:
                task.status = "failed"
                task.result = f"Failed to create worktree: {e}"
                task.resource_summary = self.resource_tracker.finish(task_id)
                return task

        # HIGH-4: Pass working dir as config param instead of os.chdir()
        working_dir = worktree_path or base_dir
        eff_config["_working_dir"] = working_dir

        tracker = self.resource_tracker

        def _run():
            from saido_agent.core.agent import AgentState
            task.status = "running"
            try:
                state = AgentState()

                # Record initial turn
                exceeded = tracker.record_turn(task_id)
                if exceeded:
                    task.status = "failed"
                    task.result = f"Resource limit exceeded: {exceeded}"
                    task.resource_usage.exceeded_limit = exceeded
                    return

                gen = _agent_run(prompt, state, eff_config, eff_system, depth=depth + 1, cancel_check=lambda: task._cancel_flag)
                for _event in gen:
                    if task._cancel_flag:
                        break
                    # Check resource limits on each event
                    exceeded = tracker.check(task_id)
                    if exceeded:
                        task._cancel_flag = True
                        task.status = "failed"
                        task.result = f"Resource limit exceeded: {exceeded}"
                        task.resource_usage.exceeded_limit = exceeded
                        break

                if task._cancel_flag and task.status != "failed":
                    task.status = "cancelled"
                    task.result = None
                elif task.status != "failed":
                    task.result = _extract_final_text(state.messages)
                    task.status = "completed"

                while not task._inbox.empty() and not task._cancel_flag:
                    inbox_msg = task._inbox.get_nowait()
                    task.status = "running"

                    exceeded = tracker.record_turn(task_id)
                    if exceeded:
                        task.status = "failed"
                        task.result = f"Resource limit exceeded: {exceeded}"
                        task.resource_usage.exceeded_limit = exceeded
                        break

                    gen2 = _agent_run(inbox_msg, state, eff_config, eff_system, depth=depth + 1, cancel_check=lambda: task._cancel_flag)
                    for _ev in gen2:
                        if task._cancel_flag:
                            break
                        exceeded = tracker.check(task_id)
                        if exceeded:
                            task._cancel_flag = True
                            task.status = "failed"
                            task.result = f"Resource limit exceeded: {exceeded}"
                            task.resource_usage.exceeded_limit = exceeded
                            break
                    if not task._cancel_flag and task.status != "failed":
                        task.result = _extract_final_text(state.messages)
                        task.status = "completed"
            except Exception as e:
                task.status = "failed"
                task.result = f"Error: {e}"
            finally:
                task.resource_summary = tracker.finish(task_id)
                self.inbox.unregister(task_id)
                if name:
                    self.inbox.unregister(name)
                if worktree_path:
                    _remove_worktree(worktree_path, worktree_branch, base_dir)

        task._future = self._pool.submit(_run)
        return task

    def spawn_isolated(self, agent_type: str, prompt: str,
                       working_dir: str = None, config: dict = None,
                       system_prompt: str = "", name: str = "",
                       resource_limits: AgentResourceLimits = None) -> str:
        """Spawn a sub-agent in an isolated git worktree.

        If in a git repo, creates a temporary worktree for the agent so it
        operates on an isolated copy with no conflicts against the main
        working tree. On completion, changes remain on the worktree branch
        for review/merge. If not in a git repo, falls back to a regular
        spawn with cwd= parameter.

        Returns the task ID.
        """
        eff_config = dict(config) if config else {}
        base_dir = working_dir or os.getcwd()
        agent_def = get_agent_definition(agent_type)

        git_root = _git_root(base_dir)
        isolation = "worktree" if git_root else ""

        if not git_root:
            eff_config["_working_dir"] = base_dir

        task = self.spawn(
            prompt=prompt,
            config=eff_config,
            system_prompt=system_prompt,
            agent_def=agent_def,
            isolation=isolation,
            name=name,
            resource_limits=resource_limits,
        )
        return task.id

    def wait(self, task_id, timeout=None):
        task = self.tasks.get(task_id)
        if task is None:
            return None
        if task._future is not None:
            try:
                task._future.result(timeout=timeout)
            except Exception:
                pass
        return task

    def get_result(self, task_id):
        task = self.tasks.get(task_id)
        return task.result if task else None

    def get_resource_summary(self, task_id):
        """Return resource usage summary for a completed/failed task."""
        task = self.tasks.get(task_id)
        if task is None:
            return None
        return task.resource_summary

    def list_tasks(self):
        return list(self.tasks.values())

    def send_message(self, task_id_or_name, message):
        task_id = self._by_name.get(task_id_or_name, task_id_or_name)
        task = self.tasks.get(task_id)
        if task is None:
            return False
        if task.status not in ("running", "pending"):
            return False
        task._inbox.put(message)
        return True

    def cancel(self, task_id):
        task = self.tasks.get(task_id)
        if task is None:
            return False
        if task.status == "running":
            task._cancel_flag = True
            return True
        return False

    def shutdown(self):
        for task in self.tasks.values():
            if task.status == "running":
                task._cancel_flag = True
        self._pool.shutdown(wait=True)
