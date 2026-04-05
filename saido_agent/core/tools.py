"""Tool definitions and implementations for Saido Agent."""
import json
import os
import re
import glob as _glob
import difflib
import shlex
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from saido_agent.core.tool_registry import ToolDef, register_tool
from saido_agent.core.tool_registry import execute_tool as _registry_execute
from saido_agent.core.permissions import PathSandboxError, get_sandbox

# -- AskUserQuestion state --
_pending_questions: list[dict] = []
_ask_lock = threading.Lock()

# -- Tool JSON schemas (sent to Claude API) --

TOOL_SCHEMAS = [
    {
        "name": "Read",
        "description": (
            "Read a file's contents. Returns content with line numbers "
            "(format: 'N\\tline'). Use limit/offset to read large files in chunks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute file path"},
                "limit":     {"type": "integer", "description": "Max lines to read"},
                "offset":    {"type": "integer", "description": "Start line (0-indexed)"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Write",
        "description": "Write content to a file, creating parent directories as needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content":   {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "Edit",
        "description": (
            "Replace exact text in a file. old_string must match exactly (including whitespace). "
            "If old_string appears multiple times, use replace_all=true or add more context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path":   {"type": "string"},
                "old_string":  {"type": "string", "description": "Exact text to replace"},
                "new_string":  {"type": "string", "description": "Replacement text"},
                "replace_all": {"type": "boolean", "description": "Replace all occurrences"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "Bash",
        "description": "Execute a shell command. Returns stdout+stderr. Stateless (no cd persistence).",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "description": "Seconds before timeout (default 30)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "Glob",
        "description": "Find files matching a glob pattern. Returns sorted list of matching paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern e.g. **/*.py"},
                "path":    {"type": "string", "description": "Base directory (default: cwd)"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "Grep",
        "description": "Search file contents with regex using ripgrep (falls back to grep).",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern":      {"type": "string", "description": "Regex pattern"},
                "path":         {"type": "string", "description": "File or directory to search"},
                "glob":         {"type": "string", "description": "File filter e.g. *.py"},
                "output_mode":  {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "content=matching lines, files_with_matches=file paths, count=match counts",
                },
                "case_insensitive": {"type": "boolean"},
                "context":      {"type": "integer", "description": "Lines of context around matches"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "WebFetch",
        "description": "Fetch a URL and return its text content (HTML stripped).",
        "input_schema": {
            "type": "object",
            "properties": {
                "url":    {"type": "string"},
                "prompt": {"type": "string", "description": "Hint for what to extract"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "WebSearch",
        "description": "Search the web via DuckDuckGo and return top results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    # -- Task tools --
    {
        "name": "TaskCreate",
        "description": (
            "Create a new task in the task list. "
            "Use this to track work items, to-dos, and multi-step plans."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subject":     {"type": "string", "description": "Brief title"},
                "description": {"type": "string", "description": "What needs to be done"},
                "active_form": {"type": "string", "description": "Present-continuous label while in_progress"},
                "metadata":    {"type": "object", "description": "Arbitrary metadata"},
            },
            "required": ["subject", "description"],
        },
    },
    {
        "name": "TaskUpdate",
        "description": (
            "Update a task: change status, subject, description, owner, "
            "dependency edges, or metadata. "
            "Set status='deleted' to remove. "
            "Statuses: pending, in_progress, completed, cancelled, deleted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id":       {"type": "string"},
                "subject":       {"type": "string"},
                "description":   {"type": "string"},
                "status":        {"type": "string", "enum": ["pending","in_progress","completed","cancelled","deleted"]},
                "active_form":   {"type": "string"},
                "owner":         {"type": "string"},
                "add_blocks":    {"type": "array", "items": {"type": "string"}},
                "add_blocked_by":{"type": "array", "items": {"type": "string"}},
                "metadata":      {"type": "object"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "TaskGet",
        "description": "Retrieve full details of a single task by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task ID to retrieve"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "TaskList",
        "description": "List all tasks with their status, owner, and pending blockers.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "NotebookEdit",
        "description": (
            "Edit a Jupyter notebook (.ipynb) cell. "
            "Supports replace (modify existing cell), insert (add new cell after cell_id), "
            "and delete (remove cell) operations. "
            "Read the notebook with the Read tool first to see cell IDs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "notebook_path": {
                    "type": "string",
                    "description": "Absolute path to the .ipynb notebook file",
                },
                "new_source": {
                    "type": "string",
                    "description": "New source code/text for the cell",
                },
                "cell_id": {
                    "type": "string",
                    "description": (
                        "ID of the cell to edit. For insert, the new cell is inserted after this cell "
                        "(or at the beginning if omitted). Use 'cell-N' (0-indexed) if no IDs are set."
                    ),
                },
                "cell_type": {
                    "type": "string",
                    "enum": ["code", "markdown"],
                    "description": "Cell type. Required for insert; defaults to current type for replace.",
                },
                "edit_mode": {
                    "type": "string",
                    "enum": ["replace", "insert", "delete"],
                    "description": "replace (default) / insert / delete",
                },
            },
            "required": ["notebook_path", "new_source"],
        },
    },
    {
        "name": "GetDiagnostics",
        "description": (
            "Get LSP-style diagnostics (errors, warnings, hints) for a source file. "
            "Uses pyright/mypy/flake8 for Python, tsc for TypeScript/JavaScript, "
            "and shellcheck for shell scripts. Returns structured diagnostic output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to diagnose",
                },
                "language": {
                    "type": "string",
                    "description": (
                        "Override auto-detected language: python, javascript, typescript, "
                        "shellscript. Omit to auto-detect from file extension."
                    ),
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "AskUserQuestion",
        "description": (
            "Pause execution and ask the user a clarifying question. "
            "Use this when you need a decision from the user before proceeding. "
            "Returns the user's answer as a string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user.",
                },
                "options": {
                    "type": "array",
                    "description": "Optional list of choices. Each item: {label, description}.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label":       {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["label"],
                    },
                },
                "allow_freetext": {
                    "type": "boolean",
                    "description": "If true (default), user may type a free-text answer instead of selecting an option.",
                },
            },
            "required": ["question"],
        },
    },
    # -- AstGrep tool --
    {
        "name": "AstGrep",
        "description": (
            "Structural code search using AST pattern matching (powered by ast-grep / sg CLI). "
            "Finds code patterns across files using syntax-aware matching, more precise than regex. "
            "Returns matches with file:line references."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "AST pattern to search for (e.g. 'console.log($$$)' or 'def $FUNC($$$): ...')",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search (default: current directory)",
                },
                "language": {
                    "type": "string",
                    "description": "Language to parse as (e.g. python, javascript, typescript, rust, go). Auto-detected if omitted.",
                },
                "rewrite": {
                    "type": "string",
                    "description": "Optional rewrite pattern to transform matches (dry-run preview only).",
                },
            },
            "required": ["pattern"],
        },
    },
]

# -- Shell execution security --

_SAIDO_DIR = Path.home() / ".saido_agent"
_AUDIT_LOG = _SAIDO_DIR / "audit.log"
_BLOCKLIST_FILE = _SAIDO_DIR / "command_blocklist.json"

# Safe binary allowlist -- NO interpreters (python, node, bash, sh, powershell)
_SAFE_BINARIES = frozenset({
    "git", "ls", "cat", "head", "tail", "wc", "find", "grep", "rg", "sg",
    "mkdir", "cp", "mv", "echo", "pip", "npm", "cargo", "make", "docker",
    "curl", "wget", "touch", "chmod", "diff", "sort", "uniq", "tee",
    "which", "env", "pwd", "date", "whoami", "hostname", "df", "du",
    "tree", "zip", "unzip", "tar", "gzip", "printf", "printenv", "uname",
    "id", "ag", "fd", "free", "top", "ps",
})

# Dangerous interpreters that require explicit permission
_BLOCKED_INTERPRETERS = frozenset({
    "python", "python3", "python3.10", "python3.11", "python3.12", "python3.13",
    "node", "nodejs", "deno", "bun",
    "bash", "sh", "zsh", "fish", "csh", "tcsh",
    "powershell", "pwsh", "cmd",
    "ruby", "perl", "php", "lua",
})

# Shell metacharacters that indicate chaining/injection
_SHELL_METACHAR_PATTERN = re.compile(r'[;|&`]|\$\(')

# Sensitive paths that must not be targeted
_SENSITIVE_PATHS = (
    "/etc/", "~/.ssh/", "~/.aws/", "~/.gnupg/",
    "~/.saido_agent/config.json",
)

# Private IP regex patterns for blocklist
_PRIVATE_IP_PATTERNS = [
    re.compile(r'\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
    re.compile(r'\b172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b'),
    re.compile(r'\b192\.168\.\d{1,3}\.\d{1,3}\b'),
    re.compile(r'\b127\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
    re.compile(r'\b169\.254\.\d{1,3}\.\d{1,3}\b'),
]


def _audit_log(command: str, status: str, exit_code: int | None = None) -> None:
    """Append a command execution record to the audit log.

    Args:
        command: The shell command that was executed or attempted.
        status: One of 'auto', 'user', 'denied'.
        exit_code: Process exit code (None if command was denied/not run).
    """
    try:
        _SAIDO_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        entry = {
            "timestamp": ts,
            "command": command,
            "approval_status": status,
            "exit_code": exit_code,
        }
        with open(_AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Audit logging must never break command execution


def _load_command_blocklist() -> list[dict]:
    """Load blocked command patterns from ~/.saido_agent/command_blocklist.json.

    Creates the file with sensible defaults if it does not exist.
    Returns a list of dicts with 'pattern' and 'description' keys.
    """
    default_blocklist = [
        {"pattern": r"rm\s+(-\w*f\w*\s+)*-?\w*r\w*\s+/\s*$", "description": "rm -rf /"},
        {"pattern": r"rm\s+(-\w*f\w*\s+)*-?\w*r\w*\s+~\s*$", "description": "rm -rf ~"},
        {"pattern": r"rm\s+(-\w*r\w*\s+)*-?\w*f\w*\s+/\s*$", "description": "rm -rf /"},
        {"pattern": r"rm\s+(-\w*r\w*\s+)*-?\w*f\w*\s+~\s*$", "description": "rm -rf ~"},
        {"pattern": r"chmod\s+777\b", "description": "chmod 777 (world-writable)"},
        {"pattern": r">\s*/etc/", "description": "redirect to /etc/"},
        {"pattern": r">\s*~/.ssh/", "description": "redirect to ~/.ssh/"},
    ]
    try:
        _SAIDO_DIR.mkdir(parents=True, exist_ok=True)
        if not _BLOCKLIST_FILE.exists():
            _BLOCKLIST_FILE.write_text(
                json.dumps(default_blocklist, indent=2), encoding="utf-8"
            )
            return default_blocklist
        return json.loads(_BLOCKLIST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return default_blocklist


def _check_blocklist(command: str) -> tuple[bool, str]:
    """Check command against blocklist patterns and built-in rules.

    Returns (blocked, reason). blocked=True means the command must be rejected.
    """
    # Check curl/wget to private IP ranges
    cmd_lower = command.lower()
    if "curl" in cmd_lower or "wget" in cmd_lower:
        for pat in _PRIVATE_IP_PATTERNS:
            if pat.search(command):
                return True, "Blocked: network request to private/internal IP range"

    # Check file-based blocklist patterns
    blocklist = _load_command_blocklist()
    for entry in blocklist:
        try:
            if re.search(entry["pattern"], command):
                desc = entry.get("description", entry["pattern"])
                return True, f"Blocked by command blocklist: {desc}"
        except re.error:
            continue

    return False, ""


def _expand_home(path_str: str) -> str:
    """Expand ~ to the actual home directory for path comparison."""
    return path_str.replace("~", str(Path.home()))


def _check_sensitive_paths(command: str) -> tuple[bool, str]:
    """Check if command targets sensitive filesystem paths."""
    expanded = _expand_home(command)
    for sensitive in _SENSITIVE_PATHS:
        expanded_sensitive = _expand_home(sensitive)
        if expanded_sensitive in expanded:
            return True, f"Blocked: command targets sensitive path {sensitive}"
    return False, ""


def _parse_and_validate_command(command: str) -> tuple[bool, str]:
    """Parse and validate a shell command for safety.

    Returns (is_safe, reason) where is_safe=True means the command can run
    without user permission, and reason explains why it was rejected.
    """
    cmd = command.strip()
    if not cmd:
        return False, "Empty command"

    # Step 1: Check blocklist first (always applies)
    blocked, reason = _check_blocklist(cmd)
    if blocked:
        return False, reason

    # Step 2: Check sensitive paths
    blocked, reason = _check_sensitive_paths(cmd)
    if blocked:
        return False, reason

    # Step 3: Check for shell metacharacters
    has_metachar = bool(_SHELL_METACHAR_PATTERN.search(cmd))

    if has_metachar:
        # For pipelines/chains, validate that ALL commands in the pipeline are safe
        # Split on pipe, semicolons, &&, ||
        segments = re.split(r'\s*(?:\|\||&&|[;|])\s*', cmd)
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            # Check for $() and backtick subshells
            if '$(' in segment or '`' in segment:
                return False, "Blocked: command substitution ($() or backticks) requires permission"
            seg_safe, seg_reason = _validate_single_command(segment)
            if not seg_safe:
                return False, f"Blocked: pipeline/chain contains unsafe command: {seg_reason}"
        return True, "Pipeline of safe commands"

    # Step 4: Validate single command
    return _validate_single_command(cmd)


def _validate_single_command(cmd: str) -> tuple[bool, str]:
    """Validate a single command (no pipes/chains) against the safe binary allowlist."""
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return False, "Blocked: malformed command (could not tokenize)"

    if not tokens:
        return False, "Empty command"

    base_cmd = Path(tokens[0]).name  # Handle /usr/bin/ls -> ls

    # Check for blocked interpreters
    if base_cmd in _BLOCKED_INTERPRETERS:
        return False, f"Blocked: '{base_cmd}' is an interpreter and requires explicit permission"

    # Check against safe binary allowlist
    if base_cmd not in _SAFE_BINARIES:
        return False, f"Blocked: '{base_cmd}' is not in the safe command allowlist"

    # Special case: chmod 777 is blocked even though chmod is safe
    if base_cmd == "chmod" and "777" in tokens:
        return False, "Blocked: chmod 777 (world-writable) is not allowed"

    return True, "Safe command"


def _is_safe_bash(cmd: str) -> bool:
    """Check if a bash command is safe to run without user permission."""
    is_safe, _ = _parse_and_validate_command(cmd)
    return is_safe


# -- Diff helpers --

def generate_unified_diff(old, new, filename, context_lines=3):
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines,
        fromfile=f"a/{filename}", tofile=f"b/{filename}", n=context_lines)
    return "".join(diff)

def maybe_truncate_diff(diff_text, max_lines=80):
    lines = diff_text.splitlines()
    if len(lines) <= max_lines:
        return diff_text
    shown = lines[:max_lines]
    remaining = len(lines) - max_lines
    return "\n".join(shown) + f"\n\n[... {remaining} more lines ...]"


# -- Tool implementations --

def _read(file_path: str, limit: int = None, offset: int = None) -> str:
    try:
        file_path = get_sandbox().validate(file_path, "read")
    except PathSandboxError as e:
        return f"Error: {e}"
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"
    if p.is_dir():
        return f"Error: {file_path} is a directory"
    try:
        lines = p.read_text(errors="replace").splitlines(keepends=True)
        start = offset or 0
        chunk = lines[start:start + limit] if limit else lines[start:]
        if not chunk:
            return "(empty file)"
        return "".join(f"{start + i + 1}\t{l}" for i, l in enumerate(chunk))
    except Exception as e:
        return f"Error: {e}"


def _write(file_path: str, content: str) -> str:
    try:
        file_path = get_sandbox().validate(file_path, "write")
    except PathSandboxError as e:
        return f"Error: {e}"
    p = Path(file_path)
    try:
        is_new = not p.exists()
        old_content = "" if is_new else p.read_text(errors="replace")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        if is_new:
            lc = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return f"Created {file_path} ({lc} lines)"
        filename = p.name
        diff = generate_unified_diff(old_content, content, filename)
        if not diff:
            return f"No changes in {file_path}"
        truncated = maybe_truncate_diff(diff)
        return f"File updated -- {file_path}:\n\n{truncated}"
    except Exception as e:
        return f"Error: {e}"


def _edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    try:
        file_path = get_sandbox().validate(file_path, "edit")
    except PathSandboxError as e:
        return f"Error: {e}"
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"
    try:
        content = p.read_text()
        count = content.count(old_string)
        if count == 0:
            return "Error: old_string not found in file"
        if count > 1 and not replace_all:
            return (f"Error: old_string appears {count} times. "
                    "Provide more context to make it unique, or use replace_all=true.")
        old_content = content
        new_content = content.replace(old_string, new_string) if replace_all else \
                      content.replace(old_string, new_string, 1)
        p.write_text(new_content)
        filename = p.name
        diff = generate_unified_diff(old_content, new_content, filename)
        return f"Changes applied to {filename}:\n\n{diff}"
    except Exception as e:
        return f"Error: {e}"


def _bash(command: str, timeout: int = 30, _approval_status: str = "auto") -> str:
    try:
        r = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=os.getcwd(),
        )
        out = r.stdout
        if r.stderr:
            out += ("\n" if out else "") + "[stderr]\n" + r.stderr
        _audit_log(command, _approval_status, r.returncode)
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        _audit_log(command, _approval_status, exit_code=-1)
        return f"Error: timed out after {timeout}s"
    except Exception as e:
        _audit_log(command, _approval_status, exit_code=-1)
        return f"Error: {e}"


def _glob_tool(pattern: str, path: str = None) -> str:
    search_dir = path or str(Path.cwd())
    try:
        search_dir = get_sandbox().validate(search_dir, "glob")
    except PathSandboxError as e:
        return f"Error: {e}"
    base = Path(search_dir)
    try:
        matches = sorted(base.glob(pattern))
        if not matches:
            return "No files matched"
        return "\n".join(str(m) for m in matches[:500])
    except Exception as e:
        return f"Error: {e}"


def _has_rg() -> bool:
    try:
        subprocess.run(["rg", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def _grep(pattern: str, path: str = None, glob: str = None,
          output_mode: str = "files_with_matches",
          case_insensitive: bool = False, context: int = 0) -> str:
    search_path = path or str(Path.cwd())
    try:
        search_path = get_sandbox().validate(search_path, "grep")
    except PathSandboxError as e:
        return f"Error: {e}"
    path = search_path  # Use validated path downstream
    use_rg = _has_rg()
    cmd = ["rg" if use_rg else "grep", "--no-heading"]
    if case_insensitive:
        cmd.append("-i")
    if output_mode == "files_with_matches":
        cmd.append("-l")
    elif output_mode == "count":
        cmd.append("-c")
    else:
        cmd.append("-n")
        if context:
            cmd += ["-C", str(context)]
    if glob:
        cmd += (["--glob", glob] if use_rg else ["--include", glob])
    cmd.append(pattern)
    cmd.append(path or str(Path.cwd()))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = r.stdout.strip()
        return out[:20000] if out else "No matches found"
    except Exception as e:
        return f"Error: {e}"


def _webfetch(url: str, prompt: str = None) -> str:
    try:
        import httpx
        r = httpx.get(url, headers={"User-Agent": "SaidoAgent/0.1"},
                      timeout=30, follow_redirects=True)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "html" in ct:
            text = re.sub(r"<script[^>]*>.*?</script>", "", r.text,
                          flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text,
                          flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
        else:
            text = r.text
        return text[:25000]
    except ImportError:
        return "Error: httpx not installed -- run: pip install httpx"
    except Exception as e:
        return f"Error: {e}"


def _websearch(query: str) -> str:
    try:
        import httpx
        url = "https://html.duckduckgo.com/html/"
        r = httpx.get(url, params={"q": query},
                      headers={"User-Agent": "Mozilla/5.0 (compatible)"},
                      timeout=30, follow_redirects=True)
        titles   = re.findall(r'class="result__title"[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                               r.text, re.DOTALL)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</div>', r.text, re.DOTALL)
        results = []
        for i, (link, title) in enumerate(titles[:8]):
            t = re.sub(r"<[^>]+>", "", title).strip()
            s = re.sub(r"<[^>]+>", "", snippets[i]).strip() if i < len(snippets) else ""
            results.append(f"**{t}**\n{link}\n{s}")
        return "\n\n".join(results) if results else "No results found"
    except ImportError:
        return "Error: httpx not installed -- run: pip install httpx"
    except Exception as e:
        return f"Error: {e}"


# -- NotebookEdit implementation --

def _parse_cell_id(cell_id: str) -> int | None:
    m = re.fullmatch(r"cell-(\d+)", cell_id)
    return int(m.group(1)) if m else None


def _notebook_edit(
    notebook_path: str,
    new_source: str,
    cell_id: str = None,
    cell_type: str = None,
    edit_mode: str = "replace",
) -> str:
    p = Path(notebook_path)
    if p.suffix != ".ipynb":
        return "Error: file must be a Jupyter notebook (.ipynb)"
    if not p.exists():
        return f"Error: notebook not found: {notebook_path}"

    try:
        nb = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return f"Error: notebook is not valid JSON: {e}"

    cells = nb.get("cells", [])

    def _resolve_index(cid: str) -> int | None:
        for i, c in enumerate(cells):
            if c.get("id") == cid:
                return i
        idx = _parse_cell_id(cid)
        if idx is not None and 0 <= idx < len(cells):
            return idx
        return None

    if edit_mode == "replace":
        if not cell_id:
            return "Error: cell_id is required for replace"
        idx = _resolve_index(cell_id)
        if idx is None:
            return f"Error: cell '{cell_id}' not found"
        target = cells[idx]
        target["source"] = new_source
        if cell_type and cell_type != target.get("cell_type"):
            target["cell_type"] = cell_type
        if target.get("cell_type") == "code":
            target["execution_count"] = None
            target["outputs"] = []

    elif edit_mode == "insert":
        if not cell_type:
            return "Error: cell_type is required for insert ('code' or 'markdown')"
        nbformat = nb.get("nbformat", 4)
        nbformat_minor = nb.get("nbformat_minor", 0)
        use_ids = nbformat > 4 or (nbformat == 4 and nbformat_minor >= 5)
        new_id = None
        if use_ids:
            import random, string
            new_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

        if cell_type == "markdown":
            new_cell = {"cell_type": "markdown", "source": new_source, "metadata": {}}
        else:
            new_cell = {
                "cell_type": "code",
                "source": new_source,
                "metadata": {},
                "execution_count": None,
                "outputs": [],
            }
        if use_ids and new_id:
            new_cell["id"] = new_id

        if cell_id:
            idx = _resolve_index(cell_id)
            if idx is None:
                return f"Error: cell '{cell_id}' not found"
            cells.insert(idx + 1, new_cell)
        else:
            cells.insert(0, new_cell)
        nb["cells"] = cells
        cell_id = new_id or cell_id

    elif edit_mode == "delete":
        if not cell_id:
            return "Error: cell_id is required for delete"
        idx = _resolve_index(cell_id)
        if idx is None:
            return f"Error: cell '{cell_id}' not found"
        cells.pop(idx)
        nb["cells"] = cells
        p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        return f"Deleted cell '{cell_id}' from {notebook_path}"
    else:
        return f"Error: unknown edit_mode '{edit_mode}' -- use replace, insert, or delete"

    nb["cells"] = cells
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return f"NotebookEdit({edit_mode}) applied to cell '{cell_id}' in {notebook_path}"


# -- GetDiagnostics implementation --

def _detect_language(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    return {
        ".py":   "python",
        ".js":   "javascript",
        ".mjs":  "javascript",
        ".cjs":  "javascript",
        ".ts":   "typescript",
        ".tsx":  "typescript",
        ".sh":   "shellscript",
        ".bash": "shellscript",
        ".zsh":  "shellscript",
    }.get(ext, "unknown")


def _run_quietly(cmd: list[str], cwd: str | None = None, timeout: int = 30) -> tuple[int, str]:
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=cwd or os.getcwd(),
        )
        out = (r.stdout + ("\n" + r.stderr if r.stderr else "")).strip()
        return r.returncode, out
    except FileNotFoundError:
        return -1, f"(command not found: {cmd[0]})"
    except subprocess.TimeoutExpired:
        return -1, f"(timed out after {timeout}s)"
    except Exception as e:
        return -1, f"(error: {e})"


def _get_diagnostics(file_path: str, language: str = None) -> str:
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"

    lang = language or _detect_language(file_path)
    abs_path = str(p.resolve())
    results: list[str] = []

    if lang == "python":
        rc, out = _run_quietly(["pyright", "--outputjson", abs_path])
        if rc != -1:
            try:
                data = json.loads(out)
                diags = data.get("generalDiagnostics", [])
                if not diags:
                    results.append("pyright: no diagnostics")
                else:
                    lines = [f"pyright ({len(diags)} issue(s)):"]
                    for d in diags[:50]:
                        rng = d.get("range", {}).get("start", {})
                        ln = rng.get("line", 0) + 1
                        ch = rng.get("character", 0) + 1
                        sev = d.get("severity", "error")
                        msg = d.get("message", "")
                        rule = d.get("rule", "")
                        lines.append(f"  {ln}:{ch} [{sev}] {msg}" + (f" ({rule})" if rule else ""))
                    results.append("\n".join(lines))
            except json.JSONDecodeError:
                if out:
                    results.append(f"pyright:\n{out[:3000]}")
        else:
            rc2, out2 = _run_quietly(["mypy", "--no-error-summary", abs_path])
            if rc2 != -1:
                results.append(f"mypy:\n{out2[:3000]}" if out2 else "mypy: no diagnostics")
            else:
                rc3, out3 = _run_quietly(["flake8", abs_path])
                if rc3 != -1:
                    results.append(f"flake8:\n{out3[:3000]}" if out3 else "flake8: no diagnostics")
                else:
                    rc4, out4 = _run_quietly(["python3", "-m", "py_compile", abs_path])
                    if out4:
                        results.append(f"py_compile (syntax check):\n{out4}")
                    else:
                        results.append("py_compile: syntax OK (no further tools available)")

    elif lang in ("javascript", "typescript"):
        rc, out = _run_quietly(["tsc", "--noEmit", "--strict", abs_path])
        if rc != -1:
            results.append(f"tsc:\n{out[:3000]}" if out else "tsc: no errors")
        else:
            rc2, out2 = _run_quietly(["eslint", abs_path])
            if rc2 != -1:
                results.append(f"eslint:\n{out2[:3000]}" if out2 else "eslint: no issues")
            else:
                results.append("No TypeScript/JavaScript checker found (install tsc or eslint)")

    elif lang == "shellscript":
        rc, out = _run_quietly(["shellcheck", abs_path])
        if rc != -1:
            results.append(f"shellcheck:\n{out[:3000]}" if out else "shellcheck: no issues")
        else:
            rc2, out2 = _run_quietly(["bash", "-n", abs_path])
            results.append(f"bash -n (syntax check):\n{out2}" if out2 else "bash -n: syntax OK")

    else:
        results.append(f"No diagnostic tool available for language: {lang or 'unknown'} (ext: {Path(file_path).suffix})")

    return "\n\n".join(results) if results else "(no diagnostics output)"


# -- AstGrep implementation --

def _ast_grep(pattern: str, path: str = None, language: str = None, rewrite: str = None) -> str:
    """Structural code search using the ast-grep (sg) CLI.

    Calls `sg --pattern <pattern> [--lang <language>] [--rewrite <rewrite>] <path>`
    and returns matches with file:line references.
    """
    # Path sandboxing placeholder (CRIT-3 will add full validation)
    search_path = path or "."
    resolved = Path(search_path).resolve()
    cwd = Path.cwd().resolve()
    # Basic containment check: path must be under cwd or be cwd itself
    try:
        resolved.relative_to(cwd)
    except ValueError:
        if resolved != cwd:
            return f"Error: path '{search_path}' is outside the working directory"

    cmd = ["sg", "--pattern", pattern]
    if language:
        cmd += ["--lang", language]
    if rewrite:
        # Dry-run only: show what would be rewritten without modifying files
        cmd += ["--rewrite", rewrite]
    cmd.append(str(resolved))

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=30, cwd=os.getcwd(),
        )
        out = r.stdout
        if r.stderr:
            out += ("\n" if out else "") + "[stderr]\n" + r.stderr
        return out.strip()[:20000] if out.strip() else "No matches found"
    except FileNotFoundError:
        return "Error: ast-grep (sg) CLI not found. Install: pip install ast-grep-cli or cargo install ast-grep"
    except subprocess.TimeoutExpired:
        return "Error: ast-grep search timed out after 30s"
    except Exception as e:
        return f"Error: {e}"


# -- AskUserQuestion implementation --

def _ask_user_question(
    question: str,
    options: list[dict] | None = None,
    allow_freetext: bool = True,
) -> str:
    event = threading.Event()
    result_holder: list[str] = []
    entry = {
        "question": question,
        "options": options or [],
        "allow_freetext": allow_freetext,
        "event": event,
        "result": result_holder,
    }
    with _ask_lock:
        _pending_questions.append(entry)

    event.wait(timeout=300)

    if result_holder:
        return result_holder[0]
    return "(no answer -- timeout)"


def drain_pending_questions() -> bool:
    """Called by the REPL loop after each streaming turn."""
    with _ask_lock:
        pending = list(_pending_questions)
        _pending_questions.clear()

    if not pending:
        return False

    for entry in pending:
        question = entry["question"]
        options  = entry["options"]
        allow_ft = entry["allow_freetext"]
        event    = entry["event"]
        result   = entry["result"]

        print()
        print("\033[1;35m? Question from assistant:\033[0m")
        print(f"   {question}")

        if options:
            print()
            for i, opt in enumerate(options, 1):
                label = opt.get("label", "")
                desc  = opt.get("description", "")
                line  = f"  [{i}] {label}"
                if desc:
                    line += f" -- {desc}"
                print(line)
            if allow_ft:
                print("  [0] Type a custom answer")
            print()

            while True:
                try:
                    raw = input("Your choice (number or text): ").strip()
                except (EOFError, KeyboardInterrupt):
                    raw = ""
                    break

                if raw.isdigit():
                    idx = int(raw)
                    if 1 <= idx <= len(options):
                        raw = options[idx - 1]["label"]
                        break
                    elif idx == 0 and allow_ft:
                        try:
                            raw = input("Your answer: ").strip()
                        except (EOFError, KeyboardInterrupt):
                            raw = ""
                        break
                elif allow_ft:
                    break
        else:
            print()
            try:
                raw = input("Your answer: ").strip()
            except (EOFError, KeyboardInterrupt):
                raw = ""

        result.append(raw)
        event.set()

    return True


# -- Dispatcher (backward-compatible wrapper) --

def execute_tool(
    name: str,
    inputs: dict,
    permission_mode: str = "auto",
    ask_permission: Optional[Callable[[str], bool]] = None,
    config: dict = None,
) -> str:
    """Dispatch tool execution; ask permission for write/destructive ops."""
    cfg = config or {}

    def _check(desc: str) -> bool:
        if permission_mode == "accept-all":
            return True
        if ask_permission:
            return ask_permission(desc)
        return True

    if name == "Write":
        if not _check(f"Write to {inputs['file_path']}"):
            return "Denied: user rejected write operation"
    elif name == "Edit":
        if not _check(f"Edit {inputs['file_path']}"):
            return "Denied: user rejected edit operation"
    elif name == "Bash":
        cmd = inputs["command"]
        is_safe = _is_safe_bash(cmd)

        # Check blocklist -- always enforced, even in accept-all mode
        blocked, block_reason = _check_blocklist(cmd)
        if blocked:
            _audit_log(cmd, "denied")
            return f"Denied: {block_reason}"

        if permission_mode == "accept-all":
            # --accept-all hardening: log all auto-approved commands
            if not is_safe:
                import sys
                print(
                    "\033[1;33m[WARN] --accept-all: auto-approving potentially "
                    f"unsafe command: {cmd}\033[0m",
                    file=sys.stderr,
                )
            # Command runs but is logged as auto-approved
            inputs = {**inputs, "_approval_status": "auto"}
        elif not is_safe:
            if not _check(f"Bash: {cmd}"):
                _audit_log(cmd, "denied")
                return "Denied: user rejected bash command"
            inputs = {**inputs, "_approval_status": "user"}
        else:
            inputs = {**inputs, "_approval_status": "auto"}

    elif name == "NotebookEdit":
        if not _check(f"Edit notebook {inputs['notebook_path']}"):
            return "Denied: user rejected notebook edit operation"

    return _registry_execute(name, inputs, cfg)


# -- Register built-in tools with the plugin registry --

def _register_builtins() -> None:
    """Register all built-in tools into the central registry."""
    _schemas = {s["name"]: s for s in TOOL_SCHEMAS}

    _tool_defs = [
        ToolDef(
            name="Read",
            schema=_schemas["Read"],
            func=lambda p, c: _read(**p),
            read_only=True,
            concurrent_safe=True,
        ),
        ToolDef(
            name="Write",
            schema=_schemas["Write"],
            func=lambda p, c: _write(**p),
            read_only=False,
            concurrent_safe=False,
        ),
        ToolDef(
            name="Edit",
            schema=_schemas["Edit"],
            func=lambda p, c: _edit(**p),
            read_only=False,
            concurrent_safe=False,
        ),
        ToolDef(
            name="Bash",
            schema=_schemas["Bash"],
            func=lambda p, c: _bash(
                p["command"], p.get("timeout", 30),
                _approval_status=p.get("_approval_status", "auto"),
            ),
            read_only=False,
            concurrent_safe=False,
        ),
        ToolDef(
            name="Glob",
            schema=_schemas["Glob"],
            func=lambda p, c: _glob_tool(p["pattern"], p.get("path")),
            read_only=True,
            concurrent_safe=True,
        ),
        ToolDef(
            name="Grep",
            schema=_schemas["Grep"],
            func=lambda p, c: _grep(
                p["pattern"], p.get("path"), p.get("glob"),
                p.get("output_mode", "files_with_matches"),
                p.get("case_insensitive", False),
                p.get("context", 0),
            ),
            read_only=True,
            concurrent_safe=True,
        ),
        ToolDef(
            name="WebFetch",
            schema=_schemas["WebFetch"],
            func=lambda p, c: _webfetch(p["url"], p.get("prompt")),
            read_only=True,
            concurrent_safe=True,
        ),
        ToolDef(
            name="WebSearch",
            schema=_schemas["WebSearch"],
            func=lambda p, c: _websearch(p["query"]),
            read_only=True,
            concurrent_safe=True,
        ),
        ToolDef(
            name="NotebookEdit",
            schema=_schemas["NotebookEdit"],
            func=lambda p, c: _notebook_edit(
                p["notebook_path"],
                p["new_source"],
                p.get("cell_id"),
                p.get("cell_type"),
                p.get("edit_mode", "replace"),
            ),
            read_only=False,
            concurrent_safe=False,
        ),
        ToolDef(
            name="GetDiagnostics",
            schema=_schemas["GetDiagnostics"],
            func=lambda p, c: _get_diagnostics(
                p["file_path"],
                p.get("language"),
            ),
            read_only=True,
            concurrent_safe=True,
        ),
        ToolDef(
            name="AskUserQuestion",
            schema=_schemas["AskUserQuestion"],
            func=lambda p, c: _ask_user_question(
                p["question"],
                p.get("options"),
                p.get("allow_freetext", True),
            ),
            read_only=True,
            concurrent_safe=False,
        ),
        ToolDef(
            name="AstGrep",
            schema=_schemas["AstGrep"],
            func=lambda p, c: _ast_grep(
                p["pattern"],
                p.get("path"),
                p.get("language"),
                p.get("rewrite"),
            ),
            read_only=True,
            concurrent_safe=True,
        ),
    ]
    for td in _tool_defs:
        register_tool(td)


_register_builtins()


# -- Memory tools (MemorySave, MemoryDelete, MemorySearch, MemoryList) --
import saido_agent.memory.tools as _memory_tools  # noqa: F401


# -- Multi-agent tools (Agent, SendMessage, CheckAgentResult, ListAgentTasks, ListAgentTypes) --
import saido_agent.multi_agent.tools as _multiagent_tools  # noqa: F401

# Expose get_agent_manager at module level for backward compatibility
from saido_agent.multi_agent.tools import get_agent_manager as _get_agent_manager  # noqa: F401


# -- Skill tools (Skill, SkillList) --
import saido_agent.skills.tools as _skill_tools  # noqa: F401
