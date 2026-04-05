"""Sandboxed plugin module loader: restricts dangerous imports at load time."""
from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import PluginEntry

from .types import PluginSecurityError

# ── Import Restrictions ──────────────────────────────────────────────────────

# Modules that are ALWAYS blocked for plugin code
BLOCKED_MODULES: frozenset[str] = frozenset({
    "os",
    "subprocess",
    "shutil",
    "socket",
    "http",
    "urllib",
    "ctypes",
})

# Modules that are ALWAYS allowed (safe stdlib)
DEFAULT_ALLOWED_MODULES: frozenset[str] = frozenset({
    "json",
    "re",
    "datetime",
    "math",
    "collections",
    "typing",
    "dataclasses",
    "enum",
    "functools",
    "itertools",
    "abc",
    "copy",
    "hashlib",
    "base64",
    "textwrap",
    "string",
    "logging",
})


def _is_module_blocked(module_name: str, allowed_extras: frozenset[str]) -> bool:
    """Check if a module import should be blocked.

    A module is blocked if its top-level name (or any prefix) matches BLOCKED_MODULES
    and it is NOT explicitly declared as allowed in the manifest.
    """
    # Get the top-level module name (e.g., 'os' from 'os.path')
    top_level = module_name.split(".")[0]

    # Check against blocked list
    if top_level in BLOCKED_MODULES:
        # Check if explicitly allowed by manifest declaration
        if top_level in allowed_extras or module_name in allowed_extras:
            return False
        return True

    # Also block submodules of blocked modules (e.g., http.client)
    parts = module_name.split(".")
    for i in range(len(parts)):
        prefix = ".".join(parts[: i + 1])
        if prefix in BLOCKED_MODULES:
            if prefix in allowed_extras or module_name in allowed_extras:
                return False
            return True

    return False


def _make_restricted_import(
    original_import: builtins.__import__.__class__,  # type: ignore[attr-defined]
    plugin_name: str,
    allowed_extras: frozenset[str],
):
    """Create a restricted __import__ function that blocks dangerous modules."""

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and _is_module_blocked(name, allowed_extras):
            raise PluginSecurityError(
                f"Plugin '{plugin_name}' attempted to import blocked module '{name}'. "
                f"Blocked modules: {sorted(BLOCKED_MODULES)}. "
                f"Declare required modules in manifest 'allowed_imports' for review."
            )
        return original_import(name, globals, locals, fromlist, level)

    return restricted_import


def sandboxed_import_plugin_module(
    entry: "PluginEntry",
    module_name: str,
) -> ModuleType | None:
    """Import a plugin module with import restrictions enforced.

    This replaces the raw importlib loading in loader.py. During module execution,
    the builtin __import__ is temporarily replaced with a restricted version that
    blocks dangerous modules unless explicitly declared in the manifest.

    Args:
        entry: The plugin entry with manifest containing allowed_imports.
        module_name: The module name to import from the plugin directory.

    Returns:
        The loaded module, or None if loading failed.

    Raises:
        PluginSecurityError: If the module tries to import a blocked module.
    """
    plugin_dir_str = str(entry.install_dir)
    if plugin_dir_str not in sys.path:
        sys.path.insert(0, plugin_dir_str)

    unique_name = f"_plugin_{entry.name}_{module_name}"
    if unique_name in sys.modules:
        return sys.modules[unique_name]

    # Determine allowed extras from manifest
    allowed_extras: frozenset[str] = frozenset()
    if entry.manifest and entry.manifest.allowed_imports:
        allowed_extras = frozenset(entry.manifest.allowed_imports)

    # Find the module file
    candidates = [
        entry.install_dir / f"{module_name}.py",
        entry.install_dir / module_name / "__init__.py",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue

        spec = importlib.util.spec_from_file_location(unique_name, candidate)
        if not spec or not spec.loader:
            continue

        mod = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = mod

        # Install restricted import during module execution
        original_import = builtins.__import__
        builtins.__import__ = _make_restricted_import(
            original_import, entry.name, allowed_extras
        )

        try:
            spec.loader.exec_module(mod)
            return mod
        except PluginSecurityError:
            # Re-raise security errors without wrapping
            del sys.modules[unique_name]
            raise
        except Exception as e:
            print(f"[plugin] Failed to load {module_name} from {entry.name}: {e}")
            del sys.modules[unique_name]
            return None
        finally:
            # Always restore original import
            builtins.__import__ = original_import

    return None
