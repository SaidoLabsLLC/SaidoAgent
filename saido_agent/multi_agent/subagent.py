"""Threaded sub-agent system for spawning nested agent loops."""
from __future__ import annotations

import os
import uuid
import queue
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class AgentDefinition:
    name: str
    description: str = ""
    system_prompt: str = ""
    model: str = ""
    tools: list = field(default_factory=list)
    source: str = "user"


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
                fm: dict = {}
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

    return AgentDefinition(
        name=name, description=description, system_prompt=system_prompt_body,
        model=model, tools=tools, source=source,
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

    def spawn(self, prompt, config, system_prompt, depth=0, agent_def=None, isolation="", name=""):
        task_id = uuid.uuid4().hex[:12]
        short_name = name or task_id[:8]
        task = SubAgentTask(id=task_id, prompt=prompt, depth=depth, name=short_name)
        self.tasks[task_id] = task
        if name:
            self._by_name[name] = task_id

        if depth >= self.max_depth:
            task.status = "failed"
            task.result = f"Max depth ({self.max_depth}) exceeded"
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
                return task

        def _run():
            from saido_agent.core.agent import AgentState
            task.status = "running"
            old_cwd = os.getcwd()
            try:
                if worktree_path:
                    os.chdir(worktree_path)
                state = AgentState()
                gen = _agent_run(prompt, state, eff_config, eff_system, depth=depth + 1, cancel_check=lambda: task._cancel_flag)
                for _event in gen:
                    if task._cancel_flag:
                        break
                if task._cancel_flag:
                    task.status = "cancelled"
                    task.result = None
                else:
                    task.result = _extract_final_text(state.messages)
                    task.status = "completed"
                while not task._inbox.empty() and not task._cancel_flag:
                    inbox_msg = task._inbox.get_nowait()
                    task.status = "running"
                    gen2 = _agent_run(inbox_msg, state, eff_config, eff_system, depth=depth + 1, cancel_check=lambda: task._cancel_flag)
                    for _ev in gen2:
                        if task._cancel_flag:
                            break
                    if not task._cancel_flag:
                        task.result = _extract_final_text(state.messages)
                        task.status = "completed"
            except Exception as e:
                task.status = "failed"
                task.result = f"Error: {e}"
            finally:
                if worktree_path:
                    os.chdir(old_cwd)
                    _remove_worktree(worktree_path, worktree_branch, old_cwd)

        task._future = self._pool.submit(_run)
        return task

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
