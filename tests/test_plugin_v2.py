"""Tests for plugin system Phase 2 — production hardening.

Covers:
- Manifest v2 validation (required fields enforcement)
- Version comparison logic
- Update detection
- Dependency resolution with circular dependency detection
- Plugin tool shadow detection
- CLI command registration
- Plugin test runner
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saido_agent.plugins.types import (
    ManifestValidationError,
    MANIFEST_V2_REQUIRED_FIELDS,
    PluginManifest,
    PluginEntry,
    PluginScope,
    PluginSecurityError,
    validate_manifest_v2,
)
from saido_agent.plugins.store import (
    _compare_versions,
    _parse_version_tuple,
    check_for_updates,
    check_plugin_tool_shadows,
    CircularDependencyError,
    get_plugin,
    list_plugins,
    resolve_plugin_dependencies,
    run_plugin_tests,
    update_all_plugins,
    update_plugin,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_v2_manifest(**overrides) -> dict:
    """Return a valid v2 manifest dict with all required fields."""
    defaults = {
        "name": "test-plugin",
        "version": "1.0.0",
        "author": "Test Author",
        "license": "MIT",
        "description": "A test plugin",
    }
    defaults.update(overrides)
    return defaults


def _make_plugin_entry(
    plugin_dir: Path,
    manifest: PluginManifest | None = None,
    name: str = "test_plugin",
    source: str = "https://github.com/saido-labs/test-plugin",
) -> PluginEntry:
    return PluginEntry(
        name=name,
        scope=PluginScope.USER,
        source=source,
        install_dir=plugin_dir,
        enabled=True,
        manifest=manifest,
    )


# =============================================================================
# 1. Manifest v2 Validation
# =============================================================================

class TestManifestV2Validation:
    """Test that v2 manifests enforce required fields."""

    def test_valid_v2_manifest_parses(self):
        """A manifest with all required fields should parse successfully."""
        data = _make_v2_manifest()
        m = PluginManifest.from_dict(data, strict=True)
        assert m.name == "test-plugin"
        assert m.version == "1.0.0"
        assert m.author == "Test Author"
        assert m.license == "MIT"

    def test_missing_name_rejected(self):
        """Manifest missing 'name' should be rejected in strict mode."""
        data = _make_v2_manifest()
        del data["name"]
        with pytest.raises(ManifestValidationError, match="name"):
            PluginManifest.from_dict(data, strict=True)

    def test_missing_version_rejected(self):
        """Manifest missing 'version' should be rejected in strict mode."""
        data = _make_v2_manifest()
        del data["version"]
        with pytest.raises(ManifestValidationError, match="version"):
            PluginManifest.from_dict(data, strict=True)

    def test_missing_author_rejected(self):
        """Manifest missing 'author' should be rejected in strict mode."""
        data = _make_v2_manifest()
        del data["author"]
        with pytest.raises(ManifestValidationError, match="author"):
            PluginManifest.from_dict(data, strict=True)

    def test_missing_license_rejected(self):
        """Manifest missing 'license' should be rejected in strict mode."""
        data = _make_v2_manifest()
        del data["license"]
        with pytest.raises(ManifestValidationError, match="license"):
            PluginManifest.from_dict(data, strict=True)

    def test_empty_required_field_rejected(self):
        """Manifest with empty string for a required field should be rejected."""
        data = _make_v2_manifest(author="")
        with pytest.raises(ManifestValidationError, match="author"):
            PluginManifest.from_dict(data, strict=True)

    def test_whitespace_only_required_field_rejected(self):
        """Manifest with whitespace-only required field should be rejected."""
        data = _make_v2_manifest(license="   ")
        with pytest.raises(ManifestValidationError, match="license"):
            PluginManifest.from_dict(data, strict=True)

    def test_multiple_missing_fields_reported(self):
        """All missing required fields should be listed in the error."""
        data = {"description": "just a description"}
        with pytest.raises(ManifestValidationError) as exc_info:
            PluginManifest.from_dict(data, strict=True)
        msg = str(exc_info.value)
        assert "author" in msg
        assert "license" in msg
        assert "name" in msg
        assert "version" in msg

    def test_non_strict_mode_allows_missing_fields(self):
        """Non-strict mode should allow missing required fields (backward compat)."""
        data = {"name": "legacy-plugin"}
        m = PluginManifest.from_dict(data, strict=False)
        assert m.name == "legacy-plugin"
        assert m.license == ""

    def test_validate_manifest_v2_convenience(self):
        """validate_manifest_v2() should enforce strict mode."""
        data = _make_v2_manifest()
        m = validate_manifest_v2(data)
        assert m.name == "test-plugin"

        data_bad = {"name": "missing-stuff"}
        with pytest.raises(ManifestValidationError):
            validate_manifest_v2(data_bad)

    def test_new_fields_parsed(self):
        """New v2 fields (license, changelog, plugin_dependencies, test_command) should parse."""
        data = _make_v2_manifest(
            changelog="## 1.0.0\n- Initial release",
            homepage="https://example.com",
            plugin_dependencies=["dep-a", "dep-b"],
            test_command="pytest",
        )
        m = PluginManifest.from_dict(data, strict=True)
        assert m.license == "MIT"
        assert m.changelog == "## 1.0.0\n- Initial release"
        assert m.homepage == "https://example.com"
        assert m.plugin_dependencies == ["dep-a", "dep-b"]
        assert m.test_command == "pytest"

    def test_canonical_bytes_includes_new_fields(self):
        """canonical_bytes() should include the new v2 fields."""
        m = PluginManifest.from_dict(_make_v2_manifest(
            changelog="changes",
            plugin_dependencies=["other-plugin"],
            test_command="pytest tests/",
        ))
        canonical = json.loads(m.canonical_bytes().decode("utf-8"))
        assert canonical["license"] == "MIT"
        assert canonical["changelog"] == "changes"
        assert canonical["plugin_dependencies"] == ["other-plugin"]
        assert canonical["test_command"] == "pytest tests/"


# =============================================================================
# 2. Version Comparison
# =============================================================================

class TestVersionComparison:
    """Test semver-like version comparison logic."""

    def test_equal_versions(self):
        assert _compare_versions("1.0.0", "1.0.0") == 0

    def test_greater_major(self):
        assert _compare_versions("2.0.0", "1.0.0") > 0

    def test_lesser_major(self):
        assert _compare_versions("1.0.0", "2.0.0") < 0

    def test_greater_minor(self):
        assert _compare_versions("1.2.0", "1.1.0") > 0

    def test_greater_patch(self):
        assert _compare_versions("1.0.2", "1.0.1") > 0

    def test_different_lengths(self):
        assert _compare_versions("1.0.0", "1.0") == 0
        assert _compare_versions("1.0.1", "1.0") > 0

    def test_prerelease_stripped(self):
        """Pre-release suffix should be stripped for numeric comparison."""
        assert _compare_versions("1.0.0-beta", "1.0.0") == 0

    def test_parse_version_tuple(self):
        assert _parse_version_tuple("1.2.3") == (1, 2, 3)
        assert _parse_version_tuple("0.1.0") == (0, 1, 0)
        assert _parse_version_tuple("2.0") == (2, 0)

    def test_non_numeric_part(self):
        """Non-numeric version parts should be treated as 0."""
        assert _parse_version_tuple("1.abc.3") == (1, 0, 3)


# =============================================================================
# 3. Update Detection
# =============================================================================

class TestUpdateDetection:
    """Test that update checks correctly identify outdated plugins."""

    def test_check_for_updates_plugin_not_found(self):
        """Should report not found for nonexistent plugin."""
        with patch("saido_agent.plugins.store.get_plugin", return_value=None):
            has_update, msg, new_ver = check_for_updates("nonexistent")
            assert not has_update
            assert "not found" in msg

    def test_check_for_updates_local_plugin(self):
        """Should report inability to update local-path plugins."""
        entry = _make_plugin_entry(Path("/tmp/fake"), source="/local/path")
        with patch("saido_agent.plugins.store.get_plugin", return_value=entry):
            has_update, msg, _ = check_for_updates("test_plugin")
            assert not has_update
            assert "local path" in msg

    def test_update_plugin_not_found(self):
        """Update on nonexistent plugin should fail gracefully."""
        with patch("saido_agent.plugins.store.get_plugin", return_value=None):
            success, msg = update_plugin("nonexistent")
            assert not success
            assert "not found" in msg

    def test_update_plugin_local_source(self):
        """Update on local-path plugin should fail with clear message."""
        entry = _make_plugin_entry(Path("/tmp/fake"), source="/local/path")
        with patch("saido_agent.plugins.store.get_plugin", return_value=entry):
            success, msg = update_plugin("test_plugin")
            assert not success
            assert "local path" in msg


# =============================================================================
# 4. Dependency Resolution
# =============================================================================

class TestDependencyResolution:
    """Test topological sort and circular dependency detection."""

    def test_no_dependencies(self):
        """Plugin with no deps should return empty install order."""
        order = resolve_plugin_dependencies("my-plugin", [])
        assert order == []

    def test_already_installed_deps_skipped(self):
        """Dependencies that are already installed should be skipped."""
        installed = _make_plugin_entry(Path("/tmp/dep-a"), name="dep-a")
        with patch("saido_agent.plugins.store.get_plugin", return_value=installed):
            order = resolve_plugin_dependencies("my-plugin", ["dep-a"])
            assert "dep-a" not in order

    def test_uninstalled_deps_in_order(self):
        """Uninstalled dependencies should appear in the install order."""
        with patch("saido_agent.plugins.store.get_plugin", return_value=None):
            order = resolve_plugin_dependencies("my-plugin", ["dep-a", "dep-b"])
            assert "dep-a" in order
            assert "dep-b" in order

    def test_circular_dependency_self(self):
        """A plugin depending on itself should raise CircularDependencyError."""
        with pytest.raises(CircularDependencyError, match="Circular dependency"):
            resolve_plugin_dependencies(
                "plugin-a", ["plugin-a"],
                _visited=set(), _stack=set(),
            )

    def test_circular_dependency_chain(self):
        """A->B->A circular chain should be detected."""
        # Simulate: plugin-a depends on plugin-b, plugin-b depends on plugin-a
        # We test by pre-loading the stack to simulate recursion
        with pytest.raises(CircularDependencyError, match="Circular dependency"):
            resolve_plugin_dependencies(
                "plugin-b", ["plugin-a"],
                _visited=set(),
                _stack={"plugin-a"},  # plugin-a is already in the call stack
            )

    def test_deps_install_order_preserved(self):
        """Dependencies should be returned in the order they are listed."""
        with patch("saido_agent.plugins.store.get_plugin", return_value=None):
            order = resolve_plugin_dependencies("main", ["first", "second", "third"])
            assert order == ["first", "second", "third"]


# =============================================================================
# 5. Plugin Tool Shadow Detection
# =============================================================================

class TestToolShadowDetection:
    """Test that plugin tools shadowing built-in tools are detected."""

    def test_no_shadows_when_no_tools(self):
        """Plugin with no tools should report no shadows."""
        with patch("saido_agent.plugins.store.get_plugin", return_value=None):
            shadows = check_plugin_tool_shadows("missing")
            assert shadows == []

    def test_shadow_detected(self):
        """Plugin tools with same name as built-in should be flagged."""
        from saido_agent.core.tool_registry import ToolDef

        manifest = PluginManifest.from_dict(_make_v2_manifest(tools=["my_tools"]))
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = _make_plugin_entry(Path(tmpdir), manifest=manifest)

            # Mock built-in tools
            builtin = ToolDef(
                name="read_file",
                schema={"name": "read_file"},
                func=lambda p, c: "",
            )

            # Mock plugin module that exports a tool with the same name
            mock_mod = MagicMock()
            plugin_tool = ToolDef(
                name="read_file",
                schema={"name": "read_file"},
                func=lambda p, c: "",
            )
            mock_mod.TOOL_DEFS = [plugin_tool]
            mock_mod.TOOL_SCHEMAS = []

            with (
                patch("saido_agent.plugins.store.get_plugin", return_value=entry),
                patch("saido_agent.core.tool_registry.get_all_tools", return_value=[builtin]),
                patch("saido_agent.plugins.sandbox.sandboxed_import_plugin_module", return_value=mock_mod),
            ):
                shadows = check_plugin_tool_shadows("test_plugin")
                assert "read_file" in shadows

    def test_no_shadow_when_unique(self):
        """Plugin tools with unique names should not be flagged."""
        from saido_agent.core.tool_registry import ToolDef

        manifest = PluginManifest.from_dict(_make_v2_manifest(tools=["my_tools"]))
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = _make_plugin_entry(Path(tmpdir), manifest=manifest)

            builtin = ToolDef(
                name="read_file",
                schema={"name": "read_file"},
                func=lambda p, c: "",
            )
            mock_mod = MagicMock()
            plugin_tool = ToolDef(
                name="my_custom_tool",
                schema={"name": "my_custom_tool"},
                func=lambda p, c: "",
            )
            mock_mod.TOOL_DEFS = [plugin_tool]
            mock_mod.TOOL_SCHEMAS = []

            with (
                patch("saido_agent.plugins.store.get_plugin", return_value=entry),
                patch("saido_agent.core.tool_registry.get_all_tools", return_value=[builtin]),
                patch("saido_agent.plugins.sandbox.sandboxed_import_plugin_module", return_value=mock_mod),
            ):
                shadows = check_plugin_tool_shadows("test_plugin")
                assert shadows == []


# =============================================================================
# 6. Plugin Test Runner
# =============================================================================

class TestPluginTestRunner:
    """Test the plugin test suite runner."""

    def test_no_test_command(self):
        """Plugin without test_command should report no tests defined."""
        manifest = PluginManifest.from_dict(_make_v2_manifest())
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = _make_plugin_entry(Path(tmpdir), manifest=manifest)
            with patch("saido_agent.plugins.store.get_plugin", return_value=entry):
                success, msg = run_plugin_tests("test_plugin")
                assert not success
                assert "no test_command" in msg

    def test_plugin_not_found(self):
        """Running tests on nonexistent plugin should fail."""
        with patch("saido_agent.plugins.store.get_plugin", return_value=None):
            success, msg = run_plugin_tests("nonexistent")
            assert not success
            assert "not found" in msg

    def test_test_command_runs(self):
        """When test_command is defined, it should be executed."""
        manifest = PluginManifest.from_dict(_make_v2_manifest(test_command="echo OK"))
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = _make_plugin_entry(Path(tmpdir), manifest=manifest)
            with patch("saido_agent.plugins.store.get_plugin", return_value=entry):
                success, msg = run_plugin_tests("test_plugin")
                assert success
                assert "Tests passed" in msg

    def test_test_command_failure(self):
        """Failed test command should report failure."""
        # Use a command that exits with non-zero
        fail_cmd = f"{sys.executable} -c \"import sys; sys.exit(1)\""
        manifest = PluginManifest.from_dict(_make_v2_manifest(test_command=fail_cmd))
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = _make_plugin_entry(Path(tmpdir), manifest=manifest)
            with patch("saido_agent.plugins.store.get_plugin", return_value=entry):
                success, msg = run_plugin_tests("test_plugin")
                assert not success
                assert "Tests failed" in msg


# =============================================================================
# 7. CLI Command Registration
# =============================================================================

class TestCLICommandRegistration:
    """Test that new CLI subcommands are registered in cmd_plugin."""

    def test_update_all_subcommand_exists(self):
        """The 'update-all' subcommand should be handled."""
        from saido_agent.cli.repl import cmd_plugin
        # Patch the imports inside cmd_plugin to avoid real plugin operations
        with patch("saido_agent.plugins.store.update_all_plugins", return_value=[]):
            result = cmd_plugin("update-all", None, None)
            assert result is True

    def test_test_subcommand_exists(self):
        """The 'test' subcommand should be handled."""
        from saido_agent.cli.repl import cmd_plugin
        with patch("saido_agent.plugins.store.get_plugin", return_value=None):
            result = cmd_plugin("test my-plugin", None, None)
            assert result is True

    def test_list_subcommand_exists(self):
        """The 'list' subcommand should be handled (alias for default)."""
        from saido_agent.cli.repl import cmd_plugin
        with patch("saido_agent.plugins.store.list_plugins", return_value=[]):
            result = cmd_plugin("list", None, None)
            assert result is True

    def test_info_subcommand_shows_v2_fields(self):
        """The 'info' subcommand should display v2 fields."""
        from saido_agent.cli.repl import cmd_plugin
        manifest = PluginManifest.from_dict(_make_v2_manifest(
            homepage="https://example.com",
            changelog="## 1.0.0",
            plugin_dependencies=["other-plugin"],
            test_command="pytest",
        ))
        entry = _make_plugin_entry(Path("/tmp/fake"), manifest=manifest)
        with patch("saido_agent.plugins.store.get_plugin", return_value=entry):
            result = cmd_plugin("info test_plugin", None, None)
            assert result is True

    def test_update_subcommand_no_args_shows_usage(self, capsys):
        """The 'update' subcommand with no args should show usage."""
        from saido_agent.cli.repl import cmd_plugin
        result = cmd_plugin("update", None, None)
        assert result is True
        captured = capsys.readouterr()
        # err() prints to stderr with ANSI codes
        assert "Usage" in captured.err

    def test_test_subcommand_no_args_shows_usage(self, capsys):
        """The 'test' subcommand with no args should show usage."""
        from saido_agent.cli.repl import cmd_plugin
        result = cmd_plugin("test", None, None)
        assert result is True
        captured = capsys.readouterr()
        assert "Usage" in captured.err


# =============================================================================
# 8. Update All Plugins
# =============================================================================

class TestUpdateAll:
    """Test the update-all functionality."""

    def test_update_all_empty(self):
        """update_all_plugins with no plugins should return empty list."""
        with patch("saido_agent.plugins.store.list_plugins", return_value=[]):
            results = update_all_plugins()
            assert results == []

    def test_update_all_skips_local(self):
        """Local-path plugins should be skipped by update_all."""
        entry = _make_plugin_entry(Path("/tmp/fake"), source="/local/path", name="local_plugin")
        with patch("saido_agent.plugins.store.list_plugins", return_value=[entry]):
            results = update_all_plugins()
            assert len(results) == 1
            name, success, msg = results[0]
            assert name == "local_plugin"
            assert not success
            assert "skipped" in msg.lower() or "local" in msg.lower()
