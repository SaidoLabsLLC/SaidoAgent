"""Plugin system types: manifest, entry, scope, security errors."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ── Security Errors ──────────────────────────────────────────────────────────

class PluginSecurityError(Exception):
    """Raised when a plugin violates a security constraint."""


# ── Permission Categories ────────────────────────────────────────────────────

VALID_PERMISSIONS = frozenset({
    "file_read",
    "file_write",
    "network",
    "shell",
    "llm_access",
})


class PluginScope(str, Enum):
    USER    = "user"     # ~/.saido_agent/plugins/
    PROJECT = "project"  # .saido_agent/plugins/ (cwd)


@dataclass
class DependencyPin:
    """A pip dependency with required sha256 hash for integrity verification."""
    package: str
    sha256: str  # hex-encoded sha256 hash

    # Valid pip package name: alphanumeric, hyphens, underscores, periods
    _VALID_PACKAGE_RE = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?$")
    _VALID_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

    def __post_init__(self) -> None:
        if not self._VALID_PACKAGE_RE.match(self.package):
            raise PluginSecurityError(
                f"Invalid pip package name: '{self.package}'. "
                "Only alphanumeric characters, hyphens, underscores, and periods are allowed."
            )
        if not self._VALID_SHA256_RE.match(self.sha256):
            raise PluginSecurityError(
                f"Invalid sha256 hash for package '{self.package}': must be 64 hex characters."
            )

    @classmethod
    def from_str(cls, dep_str: str) -> "DependencyPin":
        """Parse 'package==version --hash=sha256:abc...' or 'package:sha256hash' format."""
        # Support 'package:sha256' shorthand
        if ":" in dep_str and not dep_str.startswith("http"):
            parts = dep_str.rsplit(":", 1)
            return cls(package=parts[0].strip(), sha256=parts[1].strip())
        raise PluginSecurityError(
            f"Dependency '{dep_str}' must include sha256 hash pin in format 'package:sha256hash'."
        )


@dataclass
class PluginManifest:
    """Parsed from PLUGIN.md YAML frontmatter or plugin.json."""
    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    tags: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)    # python modules exporting tools
    skills: list[str] = field(default_factory=list)   # skill .md files
    mcp_servers: dict[str, Any] = field(default_factory=dict)  # name -> mcp server config
    dependencies: list[str] = field(default_factory=list)      # pip packages (legacy, unverified)
    pinned_dependencies: list[DependencyPin] = field(default_factory=list)  # sha256-pinned deps
    homepage: str = ""
    signature: str = ""           # Ed25519 signature (base64-encoded) over canonical manifest
    permissions: list[str] = field(default_factory=list)  # required permission categories
    allowed_imports: list[str] = field(default_factory=list)  # additional allowed module imports

    @classmethod
    def from_dict(cls, data: dict) -> "PluginManifest":
        # Parse pinned dependencies
        pinned: list[DependencyPin] = []
        for dep in data.get("pinned_dependencies", []):
            if isinstance(dep, dict):
                pinned.append(DependencyPin(
                    package=dep.get("package", ""),
                    sha256=dep.get("sha256", ""),
                ))
            elif isinstance(dep, str):
                pinned.append(DependencyPin.from_str(dep))

        # Validate permissions
        permissions = data.get("permissions", [])
        for perm in permissions:
            if perm not in VALID_PERMISSIONS:
                raise PluginSecurityError(
                    f"Unknown permission '{perm}'. Valid permissions: {sorted(VALID_PERMISSIONS)}"
                )

        return cls(
            name=data.get("name", "unknown"),
            version=str(data.get("version", "0.1.0")),
            description=data.get("description", ""),
            author=data.get("author", ""),
            tags=data.get("tags", []),
            tools=data.get("tools", []),
            skills=data.get("skills", []),
            mcp_servers=data.get("mcp_servers", {}),
            dependencies=data.get("dependencies", []),
            pinned_dependencies=pinned,
            homepage=data.get("homepage", ""),
            signature=data.get("signature", ""),
            permissions=permissions,
            allowed_imports=data.get("allowed_imports", []),
        )

    @classmethod
    def from_plugin_dir(cls, plugin_dir: Path) -> "PluginManifest | None":
        """Load manifest from a plugin directory (plugin.json or PLUGIN.md frontmatter)."""
        # Try plugin.json first
        json_file = plugin_dir / "plugin.json"
        if json_file.exists():
            import json
            try:
                return cls.from_dict(json.loads(json_file.read_text()))
            except Exception:
                pass

        # Try PLUGIN.md YAML frontmatter
        md_file = plugin_dir / "PLUGIN.md"
        if md_file.exists():
            return cls._from_md(md_file)

        return None

    @classmethod
    def _from_md(cls, md_file: Path) -> "PluginManifest | None":
        text = md_file.read_text()
        if not text.startswith("---"):
            return None
        end = text.find("---", 3)
        if end == -1:
            return None
        frontmatter = text[3:end].strip()
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(frontmatter)
        except ImportError:
            # Minimal YAML parser for simple key: value pairs
            data = {}
            for line in frontmatter.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    data[k.strip()] = v.strip()
        if isinstance(data, dict):
            return cls.from_dict(data)
        return None

    def canonical_bytes(self) -> bytes:
        """Return a canonical byte representation of the manifest for signature verification.

        This produces a deterministic JSON serialization of all fields except 'signature'
        so that the signature can be verified against this canonical form.
        """
        import json
        canonical = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "tags": sorted(self.tags),
            "tools": self.tools,
            "skills": self.skills,
            "dependencies": self.dependencies,
            "pinned_dependencies": [
                {"package": d.package, "sha256": d.sha256}
                for d in self.pinned_dependencies
            ],
            "homepage": self.homepage,
            "permissions": sorted(self.permissions),
            "allowed_imports": sorted(self.allowed_imports),
        }
        return json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def format_permissions_display(self) -> str:
        """Format permissions for display to user before install."""
        if not self.permissions:
            return "  (no permissions declared)"
        labels = {
            "file_read": "File Read    - Can read files on your system",
            "file_write": "File Write   - Can write/modify files on your system",
            "network": "Network      - Can make network requests",
            "shell": "Shell        - Can execute shell commands",
            "llm_access": "LLM Access   - Can make LLM API calls",
        }
        lines = []
        for perm in sorted(self.permissions):
            label = labels.get(perm, perm)
            lines.append(f"  - {label}")
        return "\n".join(lines)


@dataclass
class PluginEntry:
    """A plugin registered in the config store."""
    name: str
    scope: PluginScope
    source: str          # git URL, local path, or marketplace name@url
    install_dir: Path
    enabled: bool = True
    manifest: PluginManifest | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.name}@{self.scope.value}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "scope": self.scope.value,
            "source": self.source,
            "install_dir": str(self.install_dir),
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PluginEntry":
        return cls(
            name=data["name"],
            scope=PluginScope(data.get("scope", "user")),
            source=data.get("source", ""),
            install_dir=Path(data["install_dir"]),
            enabled=data.get("enabled", True),
        )


def parse_plugin_identifier(identifier: str) -> tuple[str, str | None]:
    """Parse 'name' or 'name@source'. Returns (name, source_or_None)."""
    if "@" in identifier:
        name, _, source = identifier.partition("@")
        return name.strip(), source.strip()
    return identifier.strip(), None


def sanitize_plugin_name(name: str) -> str:
    """Ensure plugin name is safe for use as directory name (alphanumeric + underscore)."""
    return re.sub(r"[^\w]", "_", name)
