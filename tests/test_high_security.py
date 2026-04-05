"""Tests for HIGH-1 through HIGH-4 security fixes.

HIGH-1: API keys stored in keyring, not in config.json
HIGH-2: MCP server commands require approval
HIGH-3: Session files are encrypted, secrets redacted
HIGH-4: os.chdir() from non-main threads raises RuntimeError
"""
import json
import os
import threading
import tempfile
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# HIGH-1: API Key Secure Storage
# ---------------------------------------------------------------------------

class TestHigh1_APIKeyStorage:
    """API keys must not appear in config.json after migration."""

    def test_migrate_legacy_keys_removes_from_config(self, tmp_path):
        """Plaintext keys in config dict are migrated to secure storage
        and removed from the dict."""
        from saido_agent.core.config import _migrate_legacy_keys, _API_KEY_FIELDS

        cfg = {
            "model": "claude-opus-4-6",
            "anthropic_api_key": "sk-ant-test-key-12345",
            "openai_api_key": "sk-openai-test-key-67890",
        }

        with mock.patch("saido_agent.core.config.store_api_key") as mock_store, \
             mock.patch("saido_agent.core.config._save_config_raw"):
            result = _migrate_legacy_keys(cfg)

        # Keys must be removed from config dict
        for key_field in _API_KEY_FIELDS:
            assert key_field not in result, f"{key_field} still in config after migration"

        # store_api_key must have been called for each key
        assert mock_store.call_count == 2
        mock_store.assert_any_call("anthropic", "sk-ant-test-key-12345")
        mock_store.assert_any_call("openai", "sk-openai-test-key-67890")

    def test_save_config_excludes_secrets_from_json(self, tmp_path):
        """save_config must not write API key fields to the JSON file."""
        from saido_agent.core import config as config_mod

        # Temporarily redirect config paths to tmp_path
        original_dir = config_mod.CONFIG_DIR
        original_file = config_mod.CONFIG_FILE
        config_mod.CONFIG_DIR = tmp_path
        config_mod.CONFIG_FILE = tmp_path / "config.json"

        try:
            cfg = {
                "model": "gpt-4o",
                "anthropic_api_key": "sk-ant-secret",
                "verbose": True,
            }
            with mock.patch("saido_agent.core.config.store_api_key"):
                config_mod.save_config(cfg)

            saved = json.loads((tmp_path / "config.json").read_text())
            assert "anthropic_api_key" not in saved
            assert "api_key" not in saved
            assert saved.get("model") == "gpt-4o"
            assert saved.get("verbose") is True
        finally:
            config_mod.CONFIG_DIR = original_dir
            config_mod.CONFIG_FILE = original_file

    def test_keyring_store_and_retrieve(self):
        """Keyring storage/retrieval round-trips correctly (mocked)."""
        from saido_agent.core.config import store_api_key, retrieve_api_key

        store = {}

        def mock_set(service, key, value):
            store[(service, key)] = value

        def mock_get(service, key):
            return store.get((service, key))

        with mock.patch("saido_agent.core.config._keyring_available", return_value=True), \
             mock.patch("saido_agent.core.config.keyring") as mock_kr:
            mock_kr.set_password = mock_set
            mock_kr.get_password = mock_get

            store_api_key("anthropic", "sk-ant-test-roundtrip")
            result = retrieve_api_key("anthropic")

        assert result == "sk-ant-test-roundtrip"


# ---------------------------------------------------------------------------
# HIGH-2: MCP Server Command Approval
# ---------------------------------------------------------------------------

class TestHigh2_MCPApproval:
    """MCP commands must require explicit approval on first run."""

    def _make_config(self, command="node", args=None, name="test-server"):
        from saido_agent.mcp.types import MCPServerConfig, MCPTransport
        return MCPServerConfig(
            name=name,
            transport=MCPTransport.STDIO,
            command=command,
            args=args or [],
        )

    def test_shell_metacharacters_rejected(self):
        """Commands with shell metacharacters must be rejected."""
        from saido_agent.mcp.client import validate_mcp_command

        dangerous_commands = [
            ("node", ["--eval", "x; rm -rf /"]),
            ("bash", ["-c", "echo && malicious"]),
            ("cmd", ["||", "bad"]),
            ("sh", ["-c", "$(whoami)"]),
            ("node", ["`id`"]),
        ]

        for cmd, args in dangerous_commands:
            config = self._make_config(command=cmd, args=args)
            error = validate_mcp_command(config)
            assert error is not None, f"Should reject: {cmd} {args}"

    def test_safe_command_passes_validation(self):
        """Clean commands without metacharacters pass validation."""
        from saido_agent.mcp.client import validate_mcp_command

        config = self._make_config(command="npx", args=["mcp-server-git"])
        assert validate_mcp_command(config) is None

    def test_first_run_requires_approval(self, tmp_path):
        """Unapproved commands require user approval on first run."""
        from saido_agent.mcp.client import check_mcp_approval, _MCP_APPROVED_FILE
        import saido_agent.mcp.client as mcp_client_mod

        # Redirect approved file to tmp
        original = mcp_client_mod._MCP_APPROVED_FILE
        mcp_client_mod._MCP_APPROVED_FILE = tmp_path / "mcp_approved.json"

        try:
            config = self._make_config(command="npx", args=["my-server"])

            # Without approval, should return False
            result = check_mcp_approval(config, prompt_fn=lambda msg: False)
            assert result is False

            # With approval, should return True and persist
            result = check_mcp_approval(config, prompt_fn=lambda msg: True)
            assert result is True

            # Second run should not need prompt (already approved)
            result = check_mcp_approval(config, prompt_fn=None)
            assert result is True
        finally:
            mcp_client_mod._MCP_APPROVED_FILE = original

    def test_no_prompt_fn_rejects_unapproved(self, tmp_path):
        """Without a prompt function, unapproved commands are rejected."""
        from saido_agent.mcp.client import check_mcp_approval
        import saido_agent.mcp.client as mcp_client_mod

        original = mcp_client_mod._MCP_APPROVED_FILE
        mcp_client_mod._MCP_APPROVED_FILE = tmp_path / "mcp_approved.json"

        try:
            config = self._make_config(command="npx", args=["unknown-server"])
            result = check_mcp_approval(config, prompt_fn=None)
            assert result is False
        finally:
            mcp_client_mod._MCP_APPROVED_FILE = original


# ---------------------------------------------------------------------------
# HIGH-3: Session Encryption
# ---------------------------------------------------------------------------

class TestHigh3_SessionEncryption:
    """Session files must be encrypted, not plain JSON."""

    def test_session_encrypt_decrypt_roundtrip(self):
        """Encrypted session data can be decrypted back."""
        from saido_agent.cli.repl import _encrypt_session, _decrypt_session

        data = {
            "messages": [{"role": "user", "content": "hello"}],
            "turn_count": 1,
            "total_input_tokens": 100,
            "total_output_tokens": 50,
        }
        encrypted = _encrypt_session(data)

        # Encrypted data must not be valid JSON
        with pytest.raises(Exception):
            json.loads(encrypted)

        # Decryption must return original structure
        decrypted = _decrypt_session(encrypted)
        assert decrypted["turn_count"] == 1
        assert len(decrypted["messages"]) == 1

    def test_session_file_not_plain_json(self, tmp_path):
        """Saved session file must not be readable as plain JSON."""
        from saido_agent.cli.repl import _encrypt_session

        data = {"messages": [], "turn_count": 0}
        encrypted = _encrypt_session(data)
        path = tmp_path / "test_session.enc"
        path.write_bytes(encrypted)

        raw = path.read_bytes()
        # Must not be parseable as JSON
        with pytest.raises(Exception):
            json.loads(raw)

    def test_secret_patterns_redacted(self):
        """API key patterns must be redacted before saving."""
        from saido_agent.cli.repl import _redact_secrets, _redact_session_data

        # Test individual pattern redaction
        assert "sk-ant-" not in _redact_secrets("my key is sk-ant-abc123def456")
        assert "sk-" not in _redact_secrets("openai key sk-abcdefghijklmnopqrstuvwxyz")
        assert "key-" not in _redact_secrets("some key-abcdefghijklmnopqrstuvwxyz")

        # Test session data deep redaction
        data = {
            "messages": [
                {"role": "user", "content": "use sk-ant-api03-mykey123456 for auth"},
                {"role": "assistant", "content": "I see your key sk-openai1234567890abcdef"},
            ]
        }
        redacted = _redact_session_data(data)
        full_json = json.dumps(redacted)
        assert "sk-ant-" not in full_json
        assert "sk-openai" not in full_json
        assert "[REDACTED]" in full_json


# ---------------------------------------------------------------------------
# HIGH-4: os.chdir() Thread Safety
# ---------------------------------------------------------------------------

class TestHigh4_ChdirThreadSafety:
    """os.chdir() from non-main threads must raise RuntimeError."""

    def test_chdir_from_non_main_thread_raises(self):
        """os.chdir() called from a worker thread must raise RuntimeError."""
        # Import to ensure the guard is installed
        import saido_agent.multi_agent.subagent  # noqa: F401

        error_holder = {"error": None}

        def _worker():
            try:
                os.chdir(tempfile.gettempdir())
                error_holder["error"] = "Should have raised RuntimeError"
            except RuntimeError as e:
                error_holder["error"] = e

        t = threading.Thread(target=_worker)
        t.start()
        t.join(timeout=5)

        assert isinstance(error_holder["error"], RuntimeError)
        assert "non-main thread" in str(error_holder["error"])

    def test_chdir_from_main_thread_works(self):
        """os.chdir() from the main thread should still work."""
        import saido_agent.multi_agent.subagent  # noqa: F401

        # Only test this from the main thread
        if threading.current_thread() is threading.main_thread():
            original = os.getcwd()
            try:
                os.chdir(tempfile.gettempdir())
                # Should not raise
                assert True
            finally:
                os.chdir(original)

    def test_subagent_no_chdir_calls(self):
        """SubAgentManager.spawn must not contain actual os.chdir() calls
        (comments are OK)."""
        import inspect
        from saido_agent.multi_agent.subagent import SubAgentManager

        source = inspect.getsource(SubAgentManager.spawn)
        # Strip comments and check for actual os.chdir() invocations
        code_lines = [
            line for line in source.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        code_only = "\n".join(code_lines)
        assert "os.chdir(" not in code_only, \
            "SubAgentManager.spawn still contains os.chdir() calls in executable code"
