"""Plugin store: install/uninstall/enable/disable/update + config persistence."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .types import (
    DependencyPin,
    PluginEntry,
    PluginManifest,
    PluginScope,
    PluginSecurityError,
    parse_plugin_identifier,
    sanitize_plugin_name,
)

# ── Config paths ──────────────────────────────────────────────────────────────

USER_PLUGIN_DIR  = Path.home() / ".saido_agent" / "plugins"
USER_PLUGIN_CFG  = Path.home() / ".saido_agent" / "plugins.json"

def _project_plugin_dir() -> Path:
    return Path.cwd() / ".saido_agent" / "plugins"

def _project_plugin_cfg() -> Path:
    return Path.cwd() / ".saido_agent" / "plugins.json"


# ── Pip package name validation ──────────────────────────────────────────────

_VALID_PIP_PACKAGE_RE = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?$")
# Characters that should NEVER appear in a pip dependency string
_DANGEROUS_CHARS_RE = re.compile(r"[;&|`$\n\r\\\"'{}()]")


def validate_pip_package_name(name: str) -> bool:
    """Validate that a pip package name contains only safe characters.

    Rejects names with path traversal, shell metacharacters, or other
    characters that could lead to command injection via pip install.
    """
    if not name or not name.strip():
        return False

    # Reject any string containing shell metacharacters BEFORE parsing
    if _DANGEROUS_CHARS_RE.search(name):
        return False

    # Reject pip flags (starting with -)
    if name.lstrip().startswith("-"):
        return False

    # Reject path traversal
    if ".." in name or "/" in name:
        return False

    # Strip version specifiers for base name validation (e.g., 'requests>=2.0')
    base_name = re.split(r"[><=!~@\[]", name)[0].strip()
    if not base_name:
        return False
    return bool(_VALID_PIP_PACKAGE_RE.match(base_name))


# ── Config read/write ─────────────────────────────────────────────────────────

def _read_cfg(cfg_path: Path) -> dict:
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text())
        except Exception:
            pass
    return {"plugins": {}}


def _write_cfg(cfg_path: Path, data: dict) -> None:
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(data, indent=2))


def _plugin_dir_for(scope: PluginScope) -> Path:
    return USER_PLUGIN_DIR if scope == PluginScope.USER else _project_plugin_dir()


def _plugin_cfg_for(scope: PluginScope) -> Path:
    return USER_PLUGIN_CFG if scope == PluginScope.USER else _project_plugin_cfg()


# ── List ──────────────────────────────────────────────────────────────────────

def list_plugins(scope: PluginScope | None = None) -> list[PluginEntry]:
    """Return all installed plugins (optionally filtered by scope)."""
    entries: list[PluginEntry] = []
    scopes = [PluginScope.USER, PluginScope.PROJECT] if scope is None else [scope]
    for sc in scopes:
        cfg = _read_cfg(_plugin_cfg_for(sc))
        for name, data in cfg.get("plugins", {}).items():
            entry = PluginEntry.from_dict(data)
            entry.manifest = PluginManifest.from_plugin_dir(entry.install_dir)
            entries.append(entry)
    return entries


def get_plugin(name: str, scope: PluginScope | None = None) -> PluginEntry | None:
    for entry in list_plugins(scope):
        if entry.name == name:
            return entry
    return None


# ── Install ───────────────────────────────────────────────────────────────────

def install_plugin(
    identifier: str,
    scope: PluginScope = PluginScope.USER,
    force: bool = False,
    approval_callback: Callable[[str], bool] | None = None,
) -> tuple[bool, str]:
    """
    Install a plugin with security verification.

    Security checks performed:
    1. Manifest signature verification (Ed25519)
    2. Source trust classification
    3. Pip dependency name validation + sha256 hash pinning
    4. Permissions display and approval

    Args:
        identifier: 'name' | 'name@git_url' | 'name@local_path'
        scope: User or project scope.
        force: Reinstall if already present.
        approval_callback: Function that takes a prompt string and returns True/False.
            Used for unsigned plugins, untrusted sources, and permission approval.
            If None, unsigned/untrusted plugins are rejected.

    Returns:
        (success, message) tuple.
    """
    from .verify import classify_source, verify_manifest_signature

    name, source = parse_plugin_identifier(identifier)
    safe_name = sanitize_plugin_name(name)

    # Check if already installed
    existing = get_plugin(safe_name, scope)
    if existing and not force:
        return False, f"Plugin '{safe_name}' is already installed in {scope.value} scope. Use --force to reinstall."

    plugin_dir = _plugin_dir_for(scope) / safe_name

    try:
        if source is None:
            # No source -> treat name as a local path if it exists, else error
            local = Path(name)
            if local.exists() and local.is_dir():
                source = str(local.resolve())
            else:
                return False, (
                    f"No source specified for '{name}'. "
                    "Provide 'name@git_url' or 'name@/local/path'."
                )

        # ── Source verification ───────────────────────────────────────
        source_class = classify_source(source)

        if source_class == "local_path":
            prompt = (
                f"Plugin '{safe_name}' is being installed from a LOCAL PATH: {source}\n"
                "Local path plugins always require explicit approval.\n"
                "Do you want to proceed?"
            )
            if not approval_callback or not approval_callback(prompt):
                return False, f"Installation of local plugin '{safe_name}' rejected: requires user approval."

        elif source_class == "untrusted_git":
            prompt = (
                f"Plugin '{safe_name}' source '{source}' is NOT from a trusted registry.\n"
                "Untrusted git plugins require explicit approval.\n"
                "Do you want to proceed?"
            )
            if not approval_callback or not approval_callback(prompt):
                return False, f"Installation of untrusted plugin '{safe_name}' rejected: source not in trusted registries."

        # ── Clone / copy source ───────────────────────────────────────
        if plugin_dir.exists() and force:
            shutil.rmtree(plugin_dir)

        if _is_git_url(source):
            ok, msg = _clone_plugin(source, plugin_dir)
            if not ok:
                return False, msg
        else:
            local_src = Path(source)
            if not local_src.exists():
                return False, f"Local path not found: {source}"
            shutil.copytree(str(local_src), str(plugin_dir))

        # Load and validate manifest
        manifest = PluginManifest.from_plugin_dir(plugin_dir)
        if manifest is None:
            manifest = PluginManifest(name=safe_name, description="(no manifest)")

        # ── Signature verification ────────────────────────────────────
        if manifest.signature:
            try:
                verify_manifest_signature(manifest)
            except PluginSecurityError as e:
                # Tampered manifest -- reject unconditionally
                if plugin_dir.exists():
                    shutil.rmtree(plugin_dir)
                return False, f"Signature verification failed for '{safe_name}': {e}"
        else:
            # Unsigned plugin requires explicit approval
            prompt = (
                f"Plugin '{safe_name}' is UNSIGNED (no Ed25519 signature).\n"
                "Unsigned plugins have not been verified by Saido Labs.\n"
                "Do you want to proceed with installation?"
            )
            if not approval_callback or not approval_callback(prompt):
                if plugin_dir.exists():
                    shutil.rmtree(plugin_dir)
                return False, f"Installation of unsigned plugin '{safe_name}' rejected: requires user approval."

        # ── Permissions display and approval ──────────────────────────
        if manifest.permissions:
            perms_display = manifest.format_permissions_display()
            prompt = (
                f"Plugin '{safe_name}' requests the following permissions:\n"
                f"{perms_display}\n"
                "Do you approve these permissions?"
            )
            if not approval_callback or not approval_callback(prompt):
                if plugin_dir.exists():
                    shutil.rmtree(plugin_dir)
                return False, f"Installation of '{safe_name}' rejected: permissions not approved."

        # ── Install pip dependencies (with validation) ────────────────
        if manifest.dependencies:
            dep_ok, dep_msg = _install_dependencies_validated(manifest.dependencies)
            if not dep_ok:
                return False, dep_msg

        if manifest.pinned_dependencies:
            dep_ok, dep_msg = _install_pinned_dependencies(manifest.pinned_dependencies)
            if not dep_ok:
                return False, dep_msg

        # Persist to config
        entry = PluginEntry(
            name=safe_name,
            scope=scope,
            source=source,
            install_dir=plugin_dir,
            enabled=True,
            manifest=manifest,
        )
        _save_entry(entry)
        return True, f"Plugin '{safe_name}' installed successfully ({scope.value} scope)."

    except Exception as e:
        return False, f"Install failed: {e}"


def _is_git_url(source: str) -> bool:
    return (
        source.startswith("https://")
        or source.startswith("git@")
        or source.startswith("http://")
        or source.endswith(".git")
    )


def _clone_plugin(url: str, dest: Path) -> tuple[bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False, f"git clone failed: {result.stderr.strip()}"
    return True, "cloned"


def _install_dependencies_validated(deps: list[str]) -> tuple[bool, str]:
    """Install pip dependencies after validating package names.

    Every package name is checked against a strict regex to prevent
    command injection via malicious package name strings.
    """
    validated: list[str] = []
    for dep in deps:
        if not validate_pip_package_name(dep):
            return False, (
                f"Invalid pip package name: '{dep}'. "
                "Package names must contain only alphanumeric characters, hyphens, underscores, and periods."
            )
        validated.append(dep)

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet"] + validated,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False, f"pip install failed: {result.stderr.strip()}"
    return True, "deps installed"


def _install_pinned_dependencies(pins: list[DependencyPin]) -> tuple[bool, str]:
    """Install pip dependencies with sha256 hash verification.

    Uses pip's --require-hashes flag to ensure all packages match their declared hashes.
    """
    pip_args: list[str] = []
    for pin in pins:
        # DependencyPin already validated in __post_init__
        pip_args.append(f"{pin.package}")
        pip_args.append(f"--hash=sha256:{pin.sha256}")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--require-hashes", "--quiet"] + pip_args,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False, f"pip install with hash verification failed: {result.stderr.strip()}"
    return True, "pinned deps installed"


def _save_entry(entry: PluginEntry) -> None:
    cfg_path = _plugin_cfg_for(entry.scope)
    data = _read_cfg(cfg_path)
    data.setdefault("plugins", {})[entry.name] = entry.to_dict()
    _write_cfg(cfg_path, data)


def _remove_entry(name: str, scope: PluginScope) -> None:
    cfg_path = _plugin_cfg_for(scope)
    data = _read_cfg(cfg_path)
    data.get("plugins", {}).pop(name, None)
    _write_cfg(cfg_path, data)


# ── Uninstall ─────────────────────────────────────────────────────────────────

def uninstall_plugin(
    name: str,
    scope: PluginScope | None = None,
    keep_data: bool = False,
) -> tuple[bool, str]:
    entry = get_plugin(name, scope)
    if entry is None:
        return False, f"Plugin '{name}' not found."
    if not keep_data and entry.install_dir.exists():
        shutil.rmtree(entry.install_dir)
    _remove_entry(entry.name, entry.scope)
    return True, f"Plugin '{name}' uninstalled."


# ── Enable / Disable ──────────────────────────────────────────────────────────

def _set_enabled(name: str, scope: PluginScope | None, enabled: bool) -> tuple[bool, str]:
    entry = get_plugin(name, scope)
    if entry is None:
        return False, f"Plugin '{name}' not found."
    entry.enabled = enabled
    _save_entry(entry)
    state = "enabled" if enabled else "disabled"
    return True, f"Plugin '{name}' {state}."


def enable_plugin(name: str, scope: PluginScope | None = None) -> tuple[bool, str]:
    return _set_enabled(name, scope, True)


def disable_plugin(name: str, scope: PluginScope | None = None) -> tuple[bool, str]:
    return _set_enabled(name, scope, False)


def disable_all_plugins(scope: PluginScope | None = None) -> tuple[bool, str]:
    entries = list_plugins(scope)
    if not entries:
        return True, "No plugins to disable."
    for entry in entries:
        entry.enabled = False
        _save_entry(entry)
    return True, f"Disabled {len(entries)} plugin(s)."


# ── Update ────────────────────────────────────────────────────────────────────

def update_plugin(name: str, scope: PluginScope | None = None) -> tuple[bool, str]:
    entry = get_plugin(name, scope)
    if entry is None:
        return False, f"Plugin '{name}' not found."
    if not _is_git_url(entry.source):
        return False, f"Plugin '{name}' was installed from a local path; cannot auto-update."
    if not entry.install_dir.exists():
        return False, f"Install directory missing: {entry.install_dir}"
    result = subprocess.run(
        ["git", "pull", "--ff-only"],
        cwd=str(entry.install_dir),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False, f"git pull failed: {result.stderr.strip()}"
    # Re-install dependencies if manifest changed
    manifest = PluginManifest.from_plugin_dir(entry.install_dir)
    if manifest and manifest.dependencies:
        dep_ok, dep_msg = _install_dependencies_validated(manifest.dependencies)
        if not dep_ok:
            return False, dep_msg
    return True, f"Plugin '{name}' updated. {result.stdout.strip()}"
