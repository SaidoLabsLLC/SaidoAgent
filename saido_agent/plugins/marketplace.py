"""Plugin marketplace: local registry for discovery, submission, and distribution."""
from __future__ import annotations

import ast
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .store import install_plugin as _store_install_plugin, validate_pip_package_name
from .types import (
    MANIFEST_V2_REQUIRED_FIELDS,
    ManifestValidationError,
    PluginManifest,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

VALID_CATEGORIES = frozenset({
    "utility",
    "knowledge",
    "integration",
    "language",
    "security",
    "visualization",
})

VALID_SORT_FIELDS = frozenset({"downloads", "rating", "name", "published_at"})

# Imports considered restricted for plugin code (mirrors sandbox.BLOCKED_MODULES)
RESTRICTED_IMPORTS = frozenset({
    "os",
    "subprocess",
    "shutil",
    "socket",
    "http",
    "urllib",
    "ctypes",
})


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class MarketplaceEntry:
    """A plugin listing in the marketplace registry."""

    name: str
    version: str
    author: str
    description: str
    permissions: list[str] = field(default_factory=list)
    downloads: int = 0
    rating: float = 0.0
    categories: list[str] = field(default_factory=list)
    homepage: str = ""
    signed: bool = False
    published_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> MarketplaceEntry:
        return cls(
            name=data.get("name", ""),
            version=data.get("version", "0.0.0"),
            author=data.get("author", ""),
            description=data.get("description", ""),
            permissions=data.get("permissions", []),
            downloads=data.get("downloads", 0),
            rating=data.get("rating", 0.0),
            categories=data.get("categories", []),
            homepage=data.get("homepage", ""),
            signed=data.get("signed", False),
            published_at=data.get("published_at", ""),
        )


@dataclass
class SubmissionResult:
    """Result of a plugin submission to the marketplace."""

    name: str
    status: str  # "submitted", "approved", "rejected", "pending_review"
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    error: str | None = None


# ── Marketplace ──────────────────────────────────────────────────────────────

class PluginMarketplace:
    """Local plugin marketplace for discovery, submission, and distribution.

    Phase 4 implementation uses a local JSON registry. Cloud marketplace
    integration is planned for a future phase.
    """

    def __init__(self, registry_dir: Path | str | None = None) -> None:
        self._registry_dir = Path(registry_dir) if registry_dir else Path.home() / ".saido_agent" / "marketplace"
        self._registry_dir.mkdir(parents=True, exist_ok=True)
        self._packages_dir = self._registry_dir / "packages"
        self._packages_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self._registry_dir / "index.json"

    # ── Public API ───────────────────────────────────────────────────────

    def search(self, query: str, category: str | None = None) -> list[MarketplaceEntry]:
        """Search marketplace by substring match on name, description, or categories.

        Args:
            query: Substring to match (case-insensitive).
            category: Optional category filter applied after text search.

        Returns:
            List of matching MarketplaceEntry objects.
        """
        entries = self._load_index()
        query_lower = query.lower()
        results: list[MarketplaceEntry] = []

        for entry in entries:
            text_match = (
                query_lower in entry.name.lower()
                or query_lower in entry.description.lower()
                or any(query_lower in cat.lower() for cat in entry.categories)
            )
            if not text_match:
                continue
            if category and category not in entry.categories:
                continue
            results.append(entry)

        return results

    def install(self, name: str) -> dict:
        """Install a plugin from the marketplace registry.

        Locates the plugin package in the local registry, verifies its
        signature if present, and delegates to the standard plugin install
        flow via store.install_plugin.

        Args:
            name: Plugin name as listed in the marketplace.

        Returns:
            Dict with 'success' (bool) and 'message' (str).
        """
        entries = self._load_index()
        entry = next((e for e in entries if e.name == name), None)

        if entry is None:
            return {"success": False, "message": f"Plugin '{name}' not found in marketplace."}

        package_dir = self._packages_dir / name
        if not package_dir.exists():
            return {"success": False, "message": f"Plugin package for '{name}' not found in local registry."}

        # Verify signature if the entry claims to be signed
        if entry.signed:
            manifest = PluginManifest.from_plugin_dir(package_dir)
            if manifest and manifest.signature:
                try:
                    from .verify import verify_manifest_signature
                    verify_manifest_signature(manifest)
                except Exception as exc:
                    return {
                        "success": False,
                        "message": f"Signature verification failed for '{name}': {exc}",
                    }

        # Delegate to standard install flow (local path install with auto-approval)
        success, message = _store_install_plugin(
            f"{name}@{package_dir}",
            approval_callback=lambda _prompt: True,
        )

        # Increment download count on successful install
        if success:
            self._increment_downloads(name)

        return {"success": success, "message": message}

    def publish(self, plugin_dir: str) -> SubmissionResult:
        """Submit a plugin for review and listing in the marketplace.

        Runs automated quality and security checks. Signed plugins that
        pass all checks are auto-approved. Unsigned plugins are placed
        in 'pending_review' status.

        Args:
            plugin_dir: Path to the plugin directory containing plugin.json or PLUGIN.md.

        Returns:
            SubmissionResult with check details and approval status.
        """
        plugin_path = Path(plugin_dir)

        if not plugin_path.exists() or not plugin_path.is_dir():
            return SubmissionResult(
                name="unknown",
                status="rejected",
                error=f"Plugin directory not found: {plugin_dir}",
            )

        # Run automated checks
        result = self._run_automated_checks(plugin_dir)

        if result.checks_failed:
            result.status = "rejected"
            return result

        # Load manifest for metadata
        manifest = PluginManifest.from_plugin_dir(plugin_path)
        if manifest is None:
            result.status = "rejected"
            result.checks_failed.append("manifest_load")
            result.error = "Could not load plugin manifest."
            return result

        # Determine approval status based on signature
        is_signed = bool(manifest.signature)
        if is_signed:
            result.status = "approved"
        else:
            result.status = "pending_review"

        # Copy plugin to packages directory
        dest = self._packages_dir / manifest.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(str(plugin_path), str(dest))

        # Create marketplace entry
        now = datetime.now(timezone.utc).isoformat()
        marketplace_entry = MarketplaceEntry(
            name=manifest.name,
            version=manifest.version,
            author=manifest.author,
            description=manifest.description,
            permissions=manifest.permissions,
            downloads=0,
            rating=0.0,
            categories=[t for t in manifest.tags if t in VALID_CATEGORIES],
            homepage=manifest.homepage,
            signed=is_signed,
            published_at=now,
        )

        # Update index
        self._add_to_index(marketplace_entry)

        result.name = manifest.name
        return result

    def list_all(
        self,
        category: str | None = None,
        sort_by: str = "downloads",
    ) -> list[MarketplaceEntry]:
        """List all available plugins, optionally filtered and sorted.

        Args:
            category: Optional category filter.
            sort_by: Sort field -- one of 'downloads', 'rating', 'name', 'published_at'.

        Returns:
            Sorted list of MarketplaceEntry objects.
        """
        entries = self._load_index()

        if category:
            entries = [e for e in entries if category in e.categories]

        reverse = sort_by in ("downloads", "rating", "published_at")
        if sort_by == "name":
            entries.sort(key=lambda e: e.name.lower())
        elif sort_by == "rating":
            entries.sort(key=lambda e: e.rating, reverse=True)
        elif sort_by == "published_at":
            entries.sort(key=lambda e: e.published_at, reverse=True)
        else:  # downloads (default)
            entries.sort(key=lambda e: e.downloads, reverse=True)

        return entries

    def get_details(self, name: str) -> MarketplaceEntry | None:
        """Get detailed information about a specific plugin.

        Args:
            name: Plugin name.

        Returns:
            MarketplaceEntry if found, None otherwise.
        """
        entries = self._load_index()
        for entry in entries:
            if entry.name == name:
                return entry
        return None

    # ── Automated Checks ─────────────────────────────────────────────────

    def _run_automated_checks(self, plugin_dir: str) -> SubmissionResult:
        """Run automated security and quality checks on a plugin submission.

        Checks performed:
        1. manifest_v2: Manifest exists and has all v2 required fields.
        2. dependency_audit: All pip dependency names are valid.
        3. restricted_imports: No restricted module imports in Python files.
        4. signature: Plugin manifest has an Ed25519 signature (informational).

        Args:
            plugin_dir: Path to the plugin directory.

        Returns:
            SubmissionResult with lists of passed and failed checks.
        """
        plugin_path = Path(plugin_dir)
        passed: list[str] = []
        failed: list[str] = []

        # 1. Manifest v2 validation
        manifest = PluginManifest.from_plugin_dir(plugin_path)
        if manifest is None:
            failed.append("manifest_v2")
        else:
            try:
                # Verify all v2 required fields are present and non-empty
                missing = []
                manifest_data = {
                    "name": manifest.name,
                    "version": manifest.version,
                    "author": manifest.author,
                    "license": manifest.license,
                }
                for field_name in MANIFEST_V2_REQUIRED_FIELDS:
                    value = manifest_data.get(field_name, "")
                    if not value or (isinstance(value, str) and not value.strip()):
                        missing.append(field_name)
                if missing:
                    failed.append("manifest_v2")
                else:
                    passed.append("manifest_v2")
            except ManifestValidationError:
                failed.append("manifest_v2")

        # 2. Dependency audit
        if manifest and manifest.dependencies:
            all_valid = all(
                validate_pip_package_name(dep) for dep in manifest.dependencies
            )
            if all_valid:
                passed.append("dependency_audit")
            else:
                failed.append("dependency_audit")
        else:
            # No dependencies is a pass
            passed.append("dependency_audit")

        # 3. Restricted import scan
        restricted_found = self._scan_restricted_imports(plugin_path)
        if restricted_found:
            failed.append("restricted_imports")
        else:
            passed.append("restricted_imports")

        # 4. Signature check (informational -- not a hard failure)
        if manifest and manifest.signature:
            passed.append("signature")
        else:
            # Signature absence is noted but does not cause rejection
            passed.append("signature_optional")

        name = manifest.name if manifest else "unknown"
        status = "rejected" if failed else "submitted"
        return SubmissionResult(
            name=name,
            status=status,
            checks_passed=passed,
            checks_failed=failed,
        )

    def _scan_restricted_imports(self, plugin_path: Path) -> list[str]:
        """Scan all Python files in a plugin directory for restricted imports.

        Uses AST parsing for accurate detection of import statements.

        Returns:
            List of restricted module names found.
        """
        found: list[str] = []

        for py_file in plugin_path.rglob("*.py"):
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError):
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top_level = alias.name.split(".")[0]
                        if top_level in RESTRICTED_IMPORTS and top_level not in found:
                            found.append(top_level)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        top_level = node.module.split(".")[0]
                        if top_level in RESTRICTED_IMPORTS and top_level not in found:
                            found.append(top_level)

        return found

    # ── Index Persistence ────────────────────────────────────────────────

    def _load_index(self) -> list[MarketplaceEntry]:
        """Load the marketplace index from disk."""
        if not self._index_file.exists():
            return []
        try:
            data = json.loads(self._index_file.read_text(encoding="utf-8"))
            return [MarketplaceEntry.from_dict(item) for item in data.get("plugins", [])]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Corrupted marketplace index at %s, returning empty.", self._index_file)
            return []

    def _save_index(self, entries: list[MarketplaceEntry]) -> None:
        """Persist the marketplace index to disk."""
        data = {"plugins": [e.to_dict() for e in entries]}
        self._index_file.write_text(
            json.dumps(data, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _add_to_index(self, entry: MarketplaceEntry) -> None:
        """Add or update an entry in the marketplace index."""
        entries = self._load_index()
        # Replace existing entry with same name or append new
        entries = [e for e in entries if e.name != entry.name]
        entries.append(entry)
        self._save_index(entries)

    def _increment_downloads(self, name: str) -> None:
        """Increment the download counter for a plugin."""
        entries = self._load_index()
        for entry in entries:
            if entry.name == name:
                entry.downloads += 1
                break
        self._save_index(entries)

    def _update_index(self) -> None:
        """Rebuild the index from packages directory.

        Scans the packages directory and regenerates index entries
        for any plugin that has a valid manifest.
        """
        entries: list[MarketplaceEntry] = []
        existing = {e.name: e for e in self._load_index()}

        for package_dir in self._packages_dir.iterdir():
            if not package_dir.is_dir():
                continue
            manifest = PluginManifest.from_plugin_dir(package_dir)
            if manifest is None:
                continue

            # Preserve download count and rating from existing index
            old = existing.get(manifest.name)
            downloads = old.downloads if old else 0
            rating = old.rating if old else 0.0
            published_at = old.published_at if old else datetime.now(timezone.utc).isoformat()

            entries.append(MarketplaceEntry(
                name=manifest.name,
                version=manifest.version,
                author=manifest.author,
                description=manifest.description,
                permissions=manifest.permissions,
                downloads=downloads,
                rating=rating,
                categories=[t for t in manifest.tags if t in VALID_CATEGORIES],
                homepage=manifest.homepage,
                signed=bool(manifest.signature),
                published_at=published_at,
            ))

        self._save_index(entries)
