"""Plugin system for Saido Agent."""
from .types import (
    PluginManifest, PluginEntry, PluginScope,
    PluginSecurityError, DependencyPin, VALID_PERMISSIONS,
    parse_plugin_identifier,
)
from .store import (
    install_plugin, uninstall_plugin,
    enable_plugin, disable_plugin, disable_all_plugins,
    update_plugin, list_plugins, get_plugin,
    validate_pip_package_name,
)
from .loader import (
    load_all_plugins, load_plugin_tools, load_plugin_skills,
    load_plugin_mcp_configs, register_plugin_tools,
)
from .sandbox import sandboxed_import_plugin_module, BLOCKED_MODULES, DEFAULT_ALLOWED_MODULES
from .verify import verify_manifest_signature, sign_manifest, is_trusted_source, classify_source
from .recommend import recommend_plugins, recommend_from_files, format_recommendations

__all__ = [
    "PluginManifest", "PluginEntry", "PluginScope",
    "PluginSecurityError", "DependencyPin", "VALID_PERMISSIONS",
    "parse_plugin_identifier",
    "install_plugin", "uninstall_plugin",
    "enable_plugin", "disable_plugin", "disable_all_plugins",
    "update_plugin", "list_plugins", "get_plugin",
    "validate_pip_package_name",
    "load_all_plugins", "load_plugin_tools", "load_plugin_skills",
    "load_plugin_mcp_configs", "register_plugin_tools",
    "sandboxed_import_plugin_module", "BLOCKED_MODULES", "DEFAULT_ALLOWED_MODULES",
    "verify_manifest_signature", "sign_manifest", "is_trusted_source", "classify_source",
    "recommend_plugins", "recommend_from_files", "format_recommendations",
]
