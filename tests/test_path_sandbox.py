"""Tests for PathSandbox — CRIT-3 path sandboxing security hardening."""

import os
import platform
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from saido_agent.core.permissions import (
    PathSandbox,
    PathSandboxError,
    configure_sandbox,
    get_sandbox,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project directory structure."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "main.py").write_text("print('hello')")
    (project / "data").mkdir()
    (project / "data" / "file.txt").write_text("data content")
    return project


@pytest.fixture
def sandbox(tmp_project):
    """Create a PathSandbox scoped to the tmp project directory only (no defaults)."""
    return PathSandbox(allowed_paths=[str(tmp_project)], _include_defaults=False)


# ---------------------------------------------------------------------------
# 1. Paths under allowed dirs pass validation
# ---------------------------------------------------------------------------


class TestAllowedPaths:
    def test_file_in_allowed_dir(self, sandbox, tmp_project):
        f = tmp_project / "src" / "main.py"
        result = sandbox.validate(str(f), "read")
        assert os.path.realpath(str(f)) == result

    def test_directory_itself_is_allowed(self, sandbox, tmp_project):
        result = sandbox.validate(str(tmp_project), "glob")
        assert os.path.realpath(str(tmp_project)) == result

    def test_nested_file(self, sandbox, tmp_project):
        f = tmp_project / "data" / "file.txt"
        result = sandbox.validate(str(f), "read")
        assert os.path.realpath(str(f)) == result

    def test_temp_dir_always_allowed(self, tmp_project):
        """The system temp directory is always implicitly allowed (with defaults)."""
        sb = PathSandbox(allowed_paths=[str(tmp_project)])  # defaults=True
        tmp = tempfile.gettempdir()
        result = sb.validate(tmp, "read")
        assert os.path.realpath(tmp) == result

    def test_saido_config_dir_always_allowed(self, tmp_project):
        """~/.saido_agent/ is always implicitly allowed (with defaults)."""
        sb = PathSandbox(allowed_paths=[str(tmp_project)])  # defaults=True
        saido_dir = os.path.join(os.path.expanduser("~"), ".saido_agent")
        result = sb.validate(saido_dir, "read")
        assert os.path.realpath(saido_dir) == result


# ---------------------------------------------------------------------------
# 2. Paths outside allowed dirs are rejected
# ---------------------------------------------------------------------------


class TestOutsideSandbox:
    def test_path_outside_sandbox(self, sandbox, tmp_path):
        outside = tmp_path / "outside_project" / "secret.txt"
        outside.parent.mkdir(exist_ok=True)
        outside.write_text("secret")
        with pytest.raises(PathSandboxError) as exc_info:
            sandbox.validate(str(outside), "read")
        assert "outside the allowed directories" in str(exc_info.value)

    def test_root_path_rejected(self, sandbox):
        # Use a path that is guaranteed outside sandbox and not sensitive
        with pytest.raises(PathSandboxError):
            sandbox.validate(os.path.expanduser("~/nonexistent_test_dir_xyz"), "read")

    def test_error_contains_path(self, sandbox, tmp_path):
        outside = str(tmp_path / "nope.txt")
        with pytest.raises(PathSandboxError) as exc_info:
            sandbox.validate(outside, "write")
        assert exc_info.value.path == outside


# ---------------------------------------------------------------------------
# 3. Symlinks outside sandbox are rejected
# ---------------------------------------------------------------------------


class TestSymlinks:
    def test_symlink_pointing_outside_sandbox(self, sandbox, tmp_project, tmp_path):
        """A symlink inside the sandbox that points outside must be rejected."""
        outside_file = tmp_path / "outside_secret.txt"
        outside_file.write_text("secret data")
        symlink = tmp_project / "sneaky_link.txt"
        try:
            symlink.symlink_to(outside_file)
        except OSError:
            pytest.skip("Cannot create symlinks on this system")
        with pytest.raises(PathSandboxError):
            sandbox.validate(str(symlink), "read")

    def test_symlink_within_sandbox_allowed(self, sandbox, tmp_project):
        """A symlink inside the sandbox pointing to another file in the sandbox is OK."""
        target = tmp_project / "src" / "main.py"
        link = tmp_project / "link_to_main.py"
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("Cannot create symlinks on this system")
        result = sandbox.validate(str(link), "read")
        assert result == os.path.realpath(str(target))


# ---------------------------------------------------------------------------
# 4. Path traversal (../) is blocked
# ---------------------------------------------------------------------------


class TestPathTraversal:
    def test_dotdot_in_path(self, sandbox, tmp_project):
        malicious = str(tmp_project / "src" / ".." / ".." / "etc" / "passwd")
        with pytest.raises(PathSandboxError) as exc_info:
            sandbox.validate(malicious, "read")
        assert "path traversal" in str(exc_info.value).lower()

    def test_dotdot_at_start(self, sandbox):
        with pytest.raises(PathSandboxError):
            sandbox.validate("../../../etc/passwd", "read")

    def test_backslash_dotdot_on_windows(self, sandbox, tmp_project):
        malicious = str(tmp_project) + "\\src\\..\\..\\secret"
        with pytest.raises(PathSandboxError):
            sandbox.validate(malicious, "read")


# ---------------------------------------------------------------------------
# 5. Sensitive directories always denied
# ---------------------------------------------------------------------------


class TestSensitiveDirs:
    def test_ssh_dir_denied(self, sandbox):
        ssh_dir = os.path.expanduser("~/.ssh")
        with pytest.raises(PathSandboxError) as exc_info:
            sandbox.validate(ssh_dir, "read")
        assert "sensitive directory" in str(exc_info.value).lower()

    def test_ssh_key_denied(self, sandbox):
        ssh_key = os.path.expanduser("~/.ssh/id_rsa")
        with pytest.raises(PathSandboxError) as exc_info:
            sandbox.validate(ssh_key, "read")
        assert "sensitive" in str(exc_info.value).lower()

    def test_aws_dir_denied(self, sandbox):
        aws_dir = os.path.expanduser("~/.aws")
        with pytest.raises(PathSandboxError) as exc_info:
            sandbox.validate(aws_dir, "read")
        assert "sensitive" in str(exc_info.value).lower()

    def test_gnupg_dir_denied(self, sandbox):
        gnupg_dir = os.path.expanduser("~/.gnupg")
        with pytest.raises(PathSandboxError) as exc_info:
            sandbox.validate(gnupg_dir, "read")
        assert "sensitive" in str(exc_info.value).lower()

    def test_sensitive_dir_denied_even_if_under_allowed(self):
        """If someone passes home as allowed, sensitive subdirs are still blocked."""
        home = os.path.expanduser("~")
        sb = PathSandbox(allowed_paths=[home])
        with pytest.raises(PathSandboxError):
            sb.validate(os.path.join(home, ".ssh", "id_rsa"), "read")

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix path")
    def test_etc_denied(self, sandbox):
        with pytest.raises(PathSandboxError):
            sandbox.validate("/etc/passwd", "read")

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows path")
    def test_system32_denied(self, sandbox):
        sys32 = os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32")
        with pytest.raises(PathSandboxError):
            sandbox.validate(os.path.join(sys32, "cmd.exe"), "read")


# ---------------------------------------------------------------------------
# 6. Audit log entries are created for file ops
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_audit_log_on_allowed(self, sandbox, tmp_project):
        audit_file = sandbox._audit_file
        f = tmp_project / "src" / "main.py"
        sandbox.validate(str(f), "read")
        content = audit_file.read_text()
        assert "read" in content
        assert "ALLOWED" in content

    def test_audit_log_on_denied(self, sandbox):
        audit_file = sandbox._audit_file
        try:
            sandbox.validate(os.path.expanduser("~/.ssh/id_rsa"), "read")
        except PathSandboxError:
            pass
        content = audit_file.read_text()
        assert "DENIED" in content

    def test_audit_log_on_traversal(self, sandbox, tmp_project):
        audit_file = sandbox._audit_file
        try:
            sandbox.validate(str(tmp_project / "src" / ".." / ".." / "x"), "write")
        except PathSandboxError:
            pass
        content = audit_file.read_text()
        assert "path_traversal" in content


# ---------------------------------------------------------------------------
# 7. PathSandboxError contains useful info
# ---------------------------------------------------------------------------


class TestErrorInfo:
    def test_error_has_path_and_reason(self):
        err = PathSandboxError("/etc/passwd", "sensitive directory")
        assert err.path == "/etc/passwd"
        assert err.reason == "sensitive directory"
        assert "/etc/passwd" in str(err)
        assert "sensitive directory" in str(err)

    def test_error_is_exception(self):
        err = PathSandboxError("/x", "test")
        assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# 8. Windows paths handled correctly
# ---------------------------------------------------------------------------


class TestWindowsPaths:
    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")
    def test_windows_backslash_path_in_allowed_dir(self, sandbox, tmp_project):
        f = str(tmp_project) + "\\src\\main.py"
        result = sandbox.validate(f, "read")
        assert os.path.realpath(f) == result

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")
    def test_windows_forward_slash_path(self, sandbox, tmp_project):
        f = str(tmp_project).replace("\\", "/") + "/src/main.py"
        result = sandbox.validate(f, "read")
        assert os.path.exists(result)


# ---------------------------------------------------------------------------
# 9. configure_sandbox and get_sandbox module-level API
# ---------------------------------------------------------------------------


class TestModuleAPI:
    def test_configure_sandbox_sets_project_dir(self, tmp_project):
        sb = configure_sandbox(project_dir=str(tmp_project))
        f = tmp_project / "src" / "main.py"
        result = sb.validate(str(f), "read")
        assert os.path.realpath(str(f)) == result

    def test_configure_sandbox_knowledge_dir(self, tmp_path):
        knowledge = tmp_path / "knowledge"
        knowledge.mkdir()
        (knowledge / "doc.md").write_text("# doc")
        sb = configure_sandbox(
            project_dir=str(tmp_path / "proj"),
            knowledge_dir=str(knowledge),
        )
        result = sb.validate(str(knowledge / "doc.md"), "read")
        assert os.path.realpath(str(knowledge / "doc.md")) == result

    def test_get_sandbox_returns_singleton(self):
        """get_sandbox() returns a PathSandbox instance."""
        sb = get_sandbox()
        assert isinstance(sb, PathSandbox)

    def test_add_allowed_path(self, sandbox, tmp_path):
        new_dir = tmp_path / "extra"
        new_dir.mkdir()
        (new_dir / "file.txt").write_text("x")

        # Should fail before adding
        with pytest.raises(PathSandboxError):
            sandbox.validate(str(new_dir / "file.txt"), "read")

        # Should pass after adding
        sandbox.add_allowed_path(str(new_dir))
        result = sandbox.validate(str(new_dir / "file.txt"), "read")
        assert os.path.realpath(str(new_dir / "file.txt")) == result
