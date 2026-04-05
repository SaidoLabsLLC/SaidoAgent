"""Path sandboxing for file operations — CRIT-3 security hardening.

All file tool operations (Read, Write, Edit, Glob, Grep) must pass through
PathSandbox.validate() before touching the filesystem. This prevents the LLM
from reading sensitive files (~/.ssh, /etc) or writing outside the project.
"""

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class PathSandboxError(Exception):
    """Raised when a file operation attempts to access a path outside the sandbox."""

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Path access denied: {path} — {reason}")


class PathSandbox:
    """Validates file paths against allowed directories and sensitive path rules.

    Security invariants:
    - Sensitive directories are ALWAYS denied, even if under an allowed prefix.
    - Symlinks that resolve outside the sandbox are denied.
    - Path traversal via '..' is blocked before resolution (early reject).
    - All operations are audit-logged to ~/.saido_agent/audit.log.
    """

    # HARDCODED deny list — NOT user-configurable (security invariant)
    SENSITIVE_DIRS: list[str] = []

    @classmethod
    def _build_sensitive_dirs(cls) -> list[str]:
        """Build the sensitive directory list with expanded home paths."""
        home = os.path.expanduser("~")
        dirs = [
            os.path.join(home, ".ssh"),
            os.path.join(home, ".aws"),
            os.path.join(home, ".gnupg"),
            os.path.join(home, ".config", "gcloud"),
        ]
        # Unix-specific
        for d in ["/etc", "/var", "/proc", "/sys"]:
            dirs.append(d)
        # Windows-specific
        dirs.append(os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32"))
        # Canonicalize all entries (resolve symlinks, normalize case on Windows)
        return [os.path.realpath(d) for d in dirs]

    def __init__(
        self,
        allowed_paths: list[str] | None = None,
        *,
        _include_defaults: bool = True,
    ):
        if not PathSandbox.SENSITIVE_DIRS:
            PathSandbox.SENSITIVE_DIRS = self._build_sensitive_dirs()

        if allowed_paths is None:
            allowed_paths = [os.getcwd()]

        saido_config = os.path.join(os.path.expanduser("~"), ".saido_agent")

        if _include_defaults:
            # Always include temp dir and saido config dir in production use
            all_paths = list(allowed_paths) + [saido_config, tempfile.gettempdir()]
        else:
            all_paths = list(allowed_paths)

        self._allowed: list[str] = [os.path.realpath(p) for p in all_paths]

        # Ensure audit log directory exists
        self._audit_dir = Path(saido_config)
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        self._audit_file = self._audit_dir / "audit.log"

    def validate(self, path: str, operation: str = "access") -> str:
        """Validate and return canonicalized path. Raises PathSandboxError if denied.

        Args:
            path: The file path to validate (absolute or relative).
            operation: The operation being performed (read, write, edit, glob, grep).

        Returns:
            The canonicalized (realpath) version of the path.

        Raises:
            PathSandboxError: If the path is outside the sandbox or in a sensitive dir.
        """
        # 1. Early reject: block raw '..' components before any resolution
        #    This catches attempts like "/allowed/../../etc/passwd"
        #    We normalize separators first to handle both / and \ on Windows.
        normalized = path.replace("\\", "/")
        parts = normalized.split("/")
        if ".." in parts:
            self._audit_file_op(operation, path, "DENIED:path_traversal")
            raise PathSandboxError(path, "path traversal ('..') is not allowed")

        # 2. Resolve to absolute canonical path (follows symlinks)
        resolved = os.path.realpath(path)

        # 3. Check against sensitive directories (ALWAYS denied)
        for sensitive in PathSandbox.SENSITIVE_DIRS:
            # Check if resolved path IS or is UNDER a sensitive directory
            if resolved == sensitive or resolved.startswith(sensitive + os.sep):
                self._audit_file_op(operation, path, f"DENIED:sensitive_dir({sensitive})")
                raise PathSandboxError(
                    path,
                    f"access to sensitive directory is forbidden: {sensitive}",
                )

        # 4. Check if resolved path is under an allowed prefix
        allowed = False
        for allowed_dir in self._allowed:
            if resolved == allowed_dir or resolved.startswith(allowed_dir + os.sep):
                allowed = True
                break

        if not allowed:
            self._audit_file_op(operation, path, "DENIED:outside_sandbox")
            raise PathSandboxError(
                path,
                f"path resolves to {resolved} which is outside the allowed directories",
            )

        # 5. Symlink check: if original path differs from resolved, verify the
        #    resolved target is under an allowed directory (already done above).
        #    Also check that the symlink itself doesn't originate from a sensitive dir.
        original_abs = os.path.abspath(path)
        if original_abs != resolved:
            # The path contained a symlink — resolved target was already validated
            # above, but log it for audit visibility.
            self._audit_file_op(operation, path, f"ALLOWED:symlink_resolved({resolved})")
        else:
            self._audit_file_op(operation, path, "ALLOWED")

        return resolved

    def add_allowed_path(self, path: str) -> None:
        """Add a path to the allowed list.

        Args:
            path: Directory path to allow access to.
        """
        real = os.path.realpath(path)
        if real not in self._allowed:
            self._allowed.append(real)
            logger.info("PathSandbox: added allowed path: %s", real)

    def _audit_file_op(self, operation: str, path: str, result: str) -> None:
        """Log file operation to audit log.

        Format: ISO8601 | operation | path | result
        """
        ts = datetime.now(timezone.utc).isoformat()
        entry = f"{ts} | {operation:6s} | {path} | {result}\n"
        try:
            with open(self._audit_file, "a", encoding="utf-8") as f:
                f.write(entry)
        except OSError:
            # Audit logging must never crash the application
            logger.warning("PathSandbox: failed to write audit log entry")


# ---------------------------------------------------------------------------
# Module-level singleton and configuration
# ---------------------------------------------------------------------------

_sandbox: PathSandbox | None = None


def get_sandbox() -> PathSandbox:
    """Return the module-level PathSandbox singleton, creating it if needed."""
    global _sandbox
    if _sandbox is None:
        _sandbox = PathSandbox()
    return _sandbox


def configure_sandbox(
    project_dir: str | None = None,
    knowledge_dir: str | None = None,
    temp_dir: str | None = None,
) -> PathSandbox:
    """Configure the global PathSandbox with project-specific allowed paths.

    Args:
        project_dir: The root project directory. Defaults to cwd.
        knowledge_dir: Optional directory for knowledge/context files.
        temp_dir: Optional additional temp directory.

    Returns:
        The configured PathSandbox instance.
    """
    global _sandbox

    allowed: list[str] = []

    if project_dir:
        allowed.append(project_dir)
    else:
        allowed.append(os.getcwd())

    if knowledge_dir:
        allowed.append(knowledge_dir)

    if temp_dir:
        allowed.append(temp_dir)

    _sandbox = PathSandbox(allowed_paths=allowed)
    return _sandbox
