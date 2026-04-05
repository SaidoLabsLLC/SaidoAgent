"""Tests for shell execution security hardening (CRIT-1).

Validates:
- Command parser rejects interpreters (python, node) without explicit permission
- Shell metacharacter injection is blocked for unsafe commands
- Blocklisted commands are rejected (rm -rf /, curl to private IPs)
- Audit log entries are created for executions
- Safe commands pass validation (git status, ls -la, grep, sg)
- Pipeline of safe commands passes (ls | grep foo)
"""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Import the security functions under test
from saido_agent.core.tools import (
    _audit_log,
    _check_blocklist,
    _is_safe_bash,
    _load_command_blocklist,
    _parse_and_validate_command,
    _validate_single_command,
    _AUDIT_LOG,
    _BLOCKLIST_FILE,
    _SAIDO_DIR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_saido_dir(tmp_path, monkeypatch):
    """Redirect _SAIDO_DIR, _AUDIT_LOG, and _BLOCKLIST_FILE to a temp dir."""
    fake_dir = tmp_path / ".saido_agent"
    fake_dir.mkdir()
    monkeypatch.setattr("saido_agent.core.tools._SAIDO_DIR", fake_dir)
    monkeypatch.setattr("saido_agent.core.tools._AUDIT_LOG", fake_dir / "audit.log")
    monkeypatch.setattr("saido_agent.core.tools._BLOCKLIST_FILE", fake_dir / "command_blocklist.json")


# ---------------------------------------------------------------------------
# 1. Interpreters require explicit permission
# ---------------------------------------------------------------------------

class TestInterpreterBlocking:
    """python, node, bash, sh, and other interpreters must be rejected."""

    @pytest.mark.parametrize("cmd", [
        "python script.py",
        "python3 -c 'print(1)'",
        "node -e 'process.exit(0)'",
        "bash -c 'echo pwned'",
        "sh -c 'id'",
        "ruby -e 'puts 1'",
        "perl -e 'print 1'",
        "powershell -Command Get-Process",
    ])
    def test_interpreters_rejected(self, cmd):
        is_safe, reason = _parse_and_validate_command(cmd)
        assert not is_safe, f"Expected '{cmd}' to be rejected"
        assert "interpreter" in reason.lower() or "blocked" in reason.lower()

    def test_python_not_safe(self):
        assert not _is_safe_bash("python script.py")

    def test_node_not_safe(self):
        assert not _is_safe_bash("node index.js")


# ---------------------------------------------------------------------------
# 2. Shell metacharacter injection is blocked
# ---------------------------------------------------------------------------

class TestMetacharInjection:
    """Metacharacters combined with unsafe commands must be blocked."""

    @pytest.mark.parametrize("cmd", [
        "ls; rm -rf /",
        "echo $(cat /etc/passwd)",
        "ls && python evil.py",
        "git status || bash -c 'whoami'",
        "echo `id`",
    ])
    def test_injection_blocked(self, cmd):
        is_safe, reason = _parse_and_validate_command(cmd)
        assert not is_safe, f"Expected injection '{cmd}' to be blocked"

    def test_semicolon_with_unsafe_command(self):
        is_safe, _ = _parse_and_validate_command("ls; rm -rf /")
        assert not is_safe

    def test_command_substitution_dollar_paren(self):
        is_safe, _ = _parse_and_validate_command("echo $(cat /etc/passwd)")
        assert not is_safe

    def test_backtick_substitution(self):
        is_safe, _ = _parse_and_validate_command("echo `id`")
        assert not is_safe


# ---------------------------------------------------------------------------
# 3. Blocklisted commands are rejected
# ---------------------------------------------------------------------------

class TestBlocklist:
    """Dangerous command patterns must be caught by the blocklist."""

    def test_rm_rf_root(self):
        blocked, _ = _check_blocklist("rm -rf /")
        assert blocked

    def test_rm_rf_home(self):
        blocked, _ = _check_blocklist("rm -rf ~")
        assert blocked

    def test_curl_to_private_ip_10(self):
        blocked, _ = _check_blocklist("curl http://10.0.0.1/secret")
        assert blocked

    def test_curl_to_private_ip_192_168(self):
        blocked, _ = _check_blocklist("curl http://192.168.1.1/admin")
        assert blocked

    def test_curl_to_private_ip_172(self):
        blocked, _ = _check_blocklist("curl http://172.16.0.1/api")
        assert blocked

    def test_curl_to_localhost(self):
        blocked, _ = _check_blocklist("curl http://127.0.0.1:8080")
        assert blocked

    def test_wget_to_private_ip(self):
        blocked, _ = _check_blocklist("wget http://10.0.0.5/exfil")
        assert blocked

    def test_curl_to_metadata_ip(self):
        blocked, _ = _check_blocklist("curl http://169.254.169.254/latest/meta-data/")
        assert blocked

    def test_chmod_777(self):
        blocked, _ = _check_blocklist("chmod 777 /tmp/file")
        assert blocked

    def test_redirect_to_etc(self):
        blocked, _ = _check_blocklist("echo 'evil' > /etc/passwd")
        assert blocked

    def test_redirect_to_ssh(self):
        blocked, _ = _check_blocklist("echo 'key' > ~/.ssh/authorized_keys")
        assert blocked

    def test_safe_curl_not_blocked(self):
        blocked, _ = _check_blocklist("curl https://api.github.com/repos")
        assert not blocked

    def test_safe_rm_not_blocked(self):
        blocked, _ = _check_blocklist("rm tempfile.txt")
        assert not blocked

    def test_load_creates_default_blocklist(self, tmp_path, monkeypatch):
        """Default blocklist file is created when missing."""
        fake_dir = tmp_path / ".saido_test"
        fake_dir.mkdir()
        blocklist_file = fake_dir / "command_blocklist.json"
        monkeypatch.setattr("saido_agent.core.tools._SAIDO_DIR", fake_dir)
        monkeypatch.setattr("saido_agent.core.tools._BLOCKLIST_FILE", blocklist_file)
        result = _load_command_blocklist()
        assert len(result) > 0
        assert blocklist_file.exists()


# ---------------------------------------------------------------------------
# 4. Audit log entries are created
# ---------------------------------------------------------------------------

class TestAuditLog:
    """Every command execution must be logged to audit.log."""

    def test_audit_log_creates_entry(self, tmp_path, monkeypatch):
        audit_file = tmp_path / ".saido_agent" / "audit.log"
        monkeypatch.setattr("saido_agent.core.tools._AUDIT_LOG", audit_file)
        monkeypatch.setattr("saido_agent.core.tools._SAIDO_DIR", tmp_path / ".saido_agent")
        (tmp_path / ".saido_agent").mkdir(exist_ok=True)

        _audit_log("git status", "auto", exit_code=0)

        assert audit_file.exists()
        entries = audit_file.read_text().strip().split("\n")
        assert len(entries) == 1
        entry = json.loads(entries[0])
        assert entry["command"] == "git status"
        assert entry["approval_status"] == "auto"
        assert entry["exit_code"] == 0
        assert "timestamp" in entry

    def test_audit_log_denied_entry(self, tmp_path, monkeypatch):
        audit_file = tmp_path / ".saido_agent" / "audit.log"
        monkeypatch.setattr("saido_agent.core.tools._AUDIT_LOG", audit_file)
        monkeypatch.setattr("saido_agent.core.tools._SAIDO_DIR", tmp_path / ".saido_agent")
        (tmp_path / ".saido_agent").mkdir(exist_ok=True)

        _audit_log("python evil.py", "denied", exit_code=None)

        entry = json.loads(audit_file.read_text().strip())
        assert entry["approval_status"] == "denied"
        assert entry["exit_code"] is None

    def test_audit_log_appends(self, tmp_path, monkeypatch):
        audit_file = tmp_path / ".saido_agent" / "audit.log"
        monkeypatch.setattr("saido_agent.core.tools._AUDIT_LOG", audit_file)
        monkeypatch.setattr("saido_agent.core.tools._SAIDO_DIR", tmp_path / ".saido_agent")
        (tmp_path / ".saido_agent").mkdir(exist_ok=True)

        _audit_log("ls", "auto", 0)
        _audit_log("git diff", "auto", 0)

        entries = audit_file.read_text().strip().split("\n")
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# 5. Safe commands pass validation
# ---------------------------------------------------------------------------

class TestSafeCommands:
    """Known-safe commands must be approved without prompting."""

    @pytest.mark.parametrize("cmd", [
        "git status",
        "git log --oneline -10",
        "git diff HEAD~1",
        "git branch -a",
        "ls -la",
        "ls",
        "cat README.md",
        "head -n 20 file.txt",
        "tail -f log.txt",
        "wc -l *.py",
        "grep -rn 'TODO' src/",
        "rg 'pattern' --type py",
        "sg --pattern 'console.log($$$)'",
        "find . -name '*.py'",
        "mkdir -p new_dir",
        "echo hello",
        "pwd",
        "date",
        "whoami",
        "which python",
        "env",
        "df -h",
        "du -sh .",
        "tree -L 2",
        "pip show requests",
        "npm list",
        "cargo metadata",
        "curl https://example.com",
        "wget https://example.com/file.tar.gz",
        "touch newfile.txt",
        "chmod 644 file.txt",
        "diff file1.txt file2.txt",
        "sort data.csv",
        "uniq -c sorted.txt",
        "zip archive.zip file1 file2",
        "unzip archive.zip",
        "tar -xzf archive.tar.gz",
        "gzip file.txt",
        "docker ps",
        "make build",
    ])
    def test_safe_command_passes(self, cmd):
        is_safe, reason = _parse_and_validate_command(cmd)
        assert is_safe, f"Expected '{cmd}' to pass but got: {reason}"

    def test_is_safe_bash_wrapper(self):
        assert _is_safe_bash("git status")
        assert _is_safe_bash("ls -la")
        assert not _is_safe_bash("python script.py")


# ---------------------------------------------------------------------------
# 6. Pipeline of safe commands passes
# ---------------------------------------------------------------------------

class TestSafePipelines:
    """Pipelines composed entirely of safe commands must be allowed."""

    @pytest.mark.parametrize("cmd", [
        "ls | grep foo",
        "cat file.txt | sort | uniq",
        "git log --oneline | head -20",
        "find . -name '*.py' | wc -l",
        "grep -r TODO . | sort",
        "du -sh * | sort -rh | head -10",
    ])
    def test_safe_pipeline_passes(self, cmd):
        is_safe, reason = _parse_and_validate_command(cmd)
        assert is_safe, f"Expected safe pipeline '{cmd}' to pass but got: {reason}"

    def test_mixed_pipeline_blocked(self):
        """Pipeline with an unsafe command in it must be blocked."""
        is_safe, _ = _parse_and_validate_command("ls | python -c 'import sys'")
        assert not is_safe

    def test_safe_chain_with_and(self):
        is_safe, reason = _parse_and_validate_command("mkdir -p dir && ls dir")
        assert is_safe, f"Expected safe && chain to pass but got: {reason}"


# ---------------------------------------------------------------------------
# 7. Sensitive path protection
# ---------------------------------------------------------------------------

class TestSensitivePaths:
    """Commands targeting sensitive filesystem paths must be blocked."""

    @pytest.mark.parametrize("cmd", [
        "cat /etc/shadow",
        "ls ~/.ssh/",
        "cat ~/.aws/credentials",
    ])
    def test_sensitive_path_blocked(self, cmd):
        from saido_agent.core.tools import _check_sensitive_paths
        blocked, _ = _check_sensitive_paths(cmd)
        assert blocked, f"Expected '{cmd}' to be blocked for sensitive path"


# ---------------------------------------------------------------------------
# 8. chmod 777 special case
# ---------------------------------------------------------------------------

class TestChmod777:
    """chmod 777 must be rejected even though chmod is in the safe list."""

    def test_chmod_777_rejected_by_parser(self):
        is_safe, reason = _parse_and_validate_command("chmod 777 /tmp/file")
        assert not is_safe
        assert "777" in reason

    def test_chmod_644_allowed(self):
        is_safe, _ = _parse_and_validate_command("chmod 644 file.txt")
        assert is_safe
