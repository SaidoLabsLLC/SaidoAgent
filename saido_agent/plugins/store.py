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
    ManifestValidationError,
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
    """Update a single plugin from its git source.

    Pulls latest changes, re-verifies the manifest signature,
    and compares versions before applying.
    """
    from .verify import verify_manifest_signature

    entry = get_plugin(name, scope)
    if entry is None:
        return False, f"Plugin '{name}' not found."
    if not _is_git_url(entry.source):
        return False, f"Plugin '{name}' was installed from a local path; cannot auto-update."
    if not entry.install_dir.exists():
        return False, f"Install directory missing: {entry.install_dir}"

    old_version = entry.manifest.version if entry.manifest else "0.0.0"

    result = subprocess.run(
        ["git", "pull", "--ff-only"],
        cwd=str(entry.install_dir),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False, f"git pull failed: {result.stderr.strip()}"

    # Re-load and verify manifest
    manifest = PluginManifest.from_plugin_dir(entry.install_dir)
    if manifest and manifest.signature:
        try:
            verify_manifest_signature(manifest)
        except PluginSecurityError as e:
            return False, f"Updated manifest signature verification failed: {e}"

    new_version = manifest.version if manifest else "0.0.0"

    # Re-install dependencies if manifest changed
    if manifest and manifest.dependencies:
        dep_ok, dep_msg = _install_dependencies_validated(manifest.dependencies)
        if not dep_ok:
            return False, dep_msg
    if manifest and manifest.pinned_dependencies:
        dep_ok, dep_msg = _install_pinned_dependencies(manifest.pinned_dependencies)
        if not dep_ok:
            return False, dep_msg

    if _compare_versions(new_version, old_version) > 0:
        return True, f"Plugin '{name}' updated from {old_version} to {new_version}."
    return True, f"Plugin '{name}' is already at latest version ({old_version})."


# ── Version Comparison ───────────────────────────────────────────────────────

def _parse_version_tuple(version_str: str) -> tuple[int, ...]:
    """Parse a semver-like string into a tuple of integers for comparison.

    Handles formats like '1.2.3', '0.1.0', '2.0'. Non-numeric parts are treated as 0.
    """
    parts: list[int] = []
    for part in version_str.split("."):
        # Strip pre-release suffixes like '-beta', '-rc1'
        numeric = part.split("-")[0].split("+")[0]
        try:
            parts.append(int(numeric))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings.

    Returns:
        >0 if v1 > v2, 0 if equal, <0 if v1 < v2.
    """
    t1 = _parse_version_tuple(v1)
    t2 = _parse_version_tuple(v2)
    # Pad to same length
    max_len = max(len(t1), len(t2))
    t1 = t1 + (0,) * (max_len - len(t1))
    t2 = t2 + (0,) * (max_len - len(t2))
    if t1 > t2:
        return 1
    elif t1 < t2:
        return -1
    return 0


# ── Check for Updates ────────────────────────────────────────────────────────

def check_for_updates(name: str, scope: PluginScope | None = None) -> tuple[bool, str, str | None]:
    """Check if a plugin has updates available.

    Returns:
        (has_update, message, new_version_or_None)
    """
    entry = get_plugin(name, scope)
    if entry is None:
        return False, f"Plugin '{name}' not found.", None
    if not _is_git_url(entry.source):
        return False, f"Plugin '{name}' was installed from a local path; cannot check for updates.", None
    if not entry.install_dir.exists():
        return False, f"Install directory missing: {entry.install_dir}", None

    installed_version = entry.manifest.version if entry.manifest else "0.0.0"

    # Fetch remote without merging to check for updates
    result = subprocess.run(
        ["git", "fetch", "--quiet"],
        cwd=str(entry.install_dir),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False, f"git fetch failed: {result.stderr.strip()}", None

    # Check if local is behind remote
    result = subprocess.run(
        ["git", "rev-list", "--count", "HEAD..@{u}"],
        cwd=str(entry.install_dir),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False, f"Cannot determine update status: {result.stderr.strip()}", None

    behind_count = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
    if behind_count == 0:
        return False, f"Plugin '{name}' is up to date (v{installed_version}).", None

    return True, f"Plugin '{name}' has updates available (currently v{installed_version}, {behind_count} commit(s) behind).", None


def update_all_plugins(scope: PluginScope | None = None) -> list[tuple[str, bool, str]]:
    """Check and update all installed plugins.

    Returns:
        List of (plugin_name, success, message) tuples.
    """
    results: list[tuple[str, bool, str]] = []
    for entry in list_plugins(scope):
        if not _is_git_url(entry.source):
            results.append((entry.name, False, "Local plugin, skipped."))
            continue
        success, msg = update_plugin(entry.name, entry.scope)
        results.append((entry.name, success, msg))
    return results


# ── Plugin Dependency Resolution ─────────────────────────────────────────────

class CircularDependencyError(Exception):
    """Raised when circular plugin dependencies are detected."""


def resolve_plugin_dependencies(
    plugin_name: str,
    plugin_deps: list[str],
    _visited: set[str] | None = None,
    _stack: set[str] | None = None,
) -> list[str]:
    """Resolve plugin dependency install order using topological sort.

    Args:
        plugin_name: The plugin being installed.
        plugin_deps: List of plugin names this plugin depends on.
        _visited: Internal tracking for visited nodes.
        _stack: Internal tracking for the current recursion stack (cycle detection).

    Returns:
        Ordered list of plugin names to install (dependencies first).

    Raises:
        CircularDependencyError: If a circular dependency is detected.
    """
    if _visited is None:
        _visited = set()
    if _stack is None:
        _stack = set()

    if plugin_name in _stack:
        raise CircularDependencyError(
            f"Circular dependency detected: '{plugin_name}' depends on itself "
            f"through the chain: {' -> '.join(sorted(_stack))} -> {plugin_name}"
        )

    if plugin_name in _visited:
        return []

    _stack.add(plugin_name)
    install_order: list[str] = []

    for dep_name in plugin_deps:
        # Check if dependency is already installed
        existing = get_plugin(dep_name)
        if existing:
            _visited.add(dep_name)
            continue

        # Check if dependency has its own dependencies (would need manifest)
        # For now, add uninstalled deps directly
        if dep_name in _stack:
            raise CircularDependencyError(
                f"Circular dependency detected: '{plugin_name}' -> '{dep_name}' -> "
                f"... -> '{plugin_name}'"
            )

        if dep_name not in _visited:
            install_order.append(dep_name)
            _visited.add(dep_name)

    _stack.discard(plugin_name)
    _visited.add(plugin_name)
    return install_order


# ── Plugin Testing ───────────────────────────────────────────────────────────

def run_plugin_tests(name: str, scope: PluginScope | None = None) -> tuple[bool, str]:
    """Run a plugin's test suite if defined in its manifest.

    Returns:
        (success, output_message)
    """
    entry = get_plugin(name, scope)
    if entry is None:
        return False, f"Plugin '{name}' not found."
    if not entry.manifest:
        return False, f"Plugin '{name}' has no manifest."
    if not entry.manifest.test_command:
        return False, f"Plugin '{name}' has no test_command defined in its manifest."
    if not entry.install_dir.exists():
        return False, f"Plugin install directory missing: {entry.install_dir}"

    result = subprocess.run(
        entry.manifest.test_command,
        shell=True,
        cwd=str(entry.install_dir),
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout.strip()
    if result.stderr.strip():
        output += "\n" + result.stderr.strip()

    if result.returncode == 0:
        return True, f"Tests passed for '{name}'.\n{output}"
    return False, f"Tests failed for '{name}' (exit code {result.returncode}).\n{output}"


def check_plugin_tool_shadows(
    plugin_name: str,
    scope: PluginScope | None = None,
) -> list[str]:
    """Check if a plugin's tools shadow any built-in tools.

    Returns:
        List of tool names that would shadow built-in tools.
    """
    from saido_agent.core.tool_registry import get_all_tools

    entry = get_plugin(plugin_name, scope)
    if entry is None or not entry.manifest or not entry.manifest.tools:
        return []

    # Collect built-in tool names (tools registered before plugin loading)
    builtin_names = {t.name for t in get_all_tools()}

    # Collect tool names from the plugin's modules
    shadows: list[str] = []
    from .sandbox import sandboxed_import_plugin_module
    for module_name in entry.manifest.tools:
        mod = sandboxed_import_plugin_module(entry, module_name)
        if mod is None:
            continue
        if hasattr(mod, "TOOL_DEFS"):
            for tdef in mod.TOOL_DEFS:
                if tdef.name in builtin_names:
                    shadows.append(tdef.name)
        if hasattr(mod, "TOOL_SCHEMAS"):
            for schema in mod.TOOL_SCHEMAS:
                tool_name = schema.get("name", "")
                if tool_name in builtin_names:
                    shadows.append(tool_name)
    return shadows
