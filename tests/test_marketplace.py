"""Tests for the plugin marketplace module."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from saido_agent.plugins.marketplace import (
    MarketplaceEntry,
    PluginMarketplace,
    SubmissionResult,
    VALID_CATEGORIES,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_marketplace(tmp_path: Path) -> PluginMarketplace:
    """Create a marketplace backed by a temporary directory."""
    return PluginMarketplace(registry_dir=tmp_path / "marketplace")


@pytest.fixture
def sample_entries() -> list[dict]:
    """A handful of sample marketplace entries for seeding the index."""
    return [
        {
            "name": "hello-world",
            "version": "1.0.0",
            "author": "Alice",
            "description": "A simple greeting plugin",
            "permissions": [],
            "downloads": 150,
            "rating": 4.5,
            "categories": ["utility"],
            "homepage": "https://example.com/hello",
            "signed": True,
            "published_at": "2026-01-15T00:00:00+00:00",
        },
        {
            "name": "data-viz",
            "version": "2.1.0",
            "author": "Bob",
            "description": "Advanced data visualization tools",
            "permissions": ["file_read"],
            "downloads": 300,
            "rating": 4.8,
            "categories": ["visualization"],
            "homepage": "",
            "signed": False,
            "published_at": "2026-02-20T00:00:00+00:00",
        },
        {
            "name": "slack-connector",
            "version": "0.5.0",
            "author": "Carol",
            "description": "Integration with Slack workspaces",
            "permissions": ["network"],
            "downloads": 80,
            "rating": 3.9,
            "categories": ["integration"],
            "homepage": "https://example.com/slack",
            "signed": True,
            "published_at": "2026-03-10T00:00:00+00:00",
        },
    ]


@pytest.fixture
def seeded_marketplace(
    tmp_marketplace: PluginMarketplace,
    sample_entries: list[dict],
) -> PluginMarketplace:
    """Marketplace with pre-loaded sample entries."""
    entries = [MarketplaceEntry.from_dict(d) for d in sample_entries]
    tmp_marketplace._save_index(entries)
    return tmp_marketplace


def _make_clean_plugin(tmp_path: Path, name: str = "clean-plugin", signed: bool = False) -> Path:
    """Create a minimal valid plugin directory with a v2 manifest."""
    plugin_dir = tmp_path / name
    plugin_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": name,
        "version": "1.0.0",
        "author": "Test Author",
        "license": "MIT",
        "description": "A clean test plugin",
        "tags": ["utility"],
        "dependencies": [],
        "permissions": [],
    }
    if signed:
        manifest["signature"] = "dGVzdHNpZw=="  # dummy base64

    (plugin_dir / "plugin.json").write_text(json.dumps(manifest))
    (plugin_dir / "main.py").write_text("# Clean plugin\nimport json\nimport re\n")
    return plugin_dir


def _make_dirty_plugin(tmp_path: Path, name: str = "dirty-plugin") -> Path:
    """Create a plugin directory with restricted imports."""
    plugin_dir = tmp_path / name
    plugin_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": name,
        "version": "0.1.0",
        "author": "Shady Dev",
        "license": "MIT",
        "description": "Plugin with restricted imports",
        "tags": [],
        "dependencies": [],
    }
    (plugin_dir / "plugin.json").write_text(json.dumps(manifest))
    (plugin_dir / "main.py").write_text(
        "import os\nimport subprocess\nfrom shutil import rmtree\n"
    )
    return plugin_dir


def _make_bad_manifest_plugin(tmp_path: Path, name: str = "bad-manifest") -> Path:
    """Create a plugin with missing v2 required fields."""
    plugin_dir = tmp_path / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    # Missing author and license
    manifest = {
        "name": name,
        "version": "0.1.0",
        "description": "Incomplete manifest",
    }
    (plugin_dir / "plugin.json").write_text(json.dumps(manifest))
    (plugin_dir / "main.py").write_text("# Stub\n")
    return plugin_dir


# ── Search Tests ─────────────────────────────────────────────────────────────

class TestSearch:
    def test_search_by_name(self, seeded_marketplace: PluginMarketplace) -> None:
        results = seeded_marketplace.search("hello")
        assert len(results) == 1
        assert results[0].name == "hello-world"

    def test_search_by_description(self, seeded_marketplace: PluginMarketplace) -> None:
        results = seeded_marketplace.search("visualization")
        assert len(results) == 1
        assert results[0].name == "data-viz"

    def test_search_case_insensitive(self, seeded_marketplace: PluginMarketplace) -> None:
        results = seeded_marketplace.search("SLACK")
        assert len(results) == 1
        assert results[0].name == "slack-connector"

    def test_search_no_results(self, seeded_marketplace: PluginMarketplace) -> None:
        results = seeded_marketplace.search("nonexistent")
        assert results == []

    def test_search_with_category_filter(self, seeded_marketplace: PluginMarketplace) -> None:
        # "integration" query matches description of data-viz too, but category filter narrows it
        results = seeded_marketplace.search("slack", category="integration")
        assert len(results) == 1
        assert results[0].name == "slack-connector"

    def test_search_category_filter_excludes(self, seeded_marketplace: PluginMarketplace) -> None:
        results = seeded_marketplace.search("hello", category="visualization")
        assert results == []

    def test_search_by_category_text(self, seeded_marketplace: PluginMarketplace) -> None:
        """Searching for a category name as a text query matches entries with that category."""
        results = seeded_marketplace.search("utility")
        assert len(results) == 1
        assert results[0].name == "hello-world"


# ── Install Tests ────────────────────────────────────────────────────────────

class TestInstall:
    def test_install_not_found(self, seeded_marketplace: PluginMarketplace) -> None:
        result = seeded_marketplace.install("no-such-plugin")
        assert result["success"] is False
        assert "not found" in result["message"]

    def test_install_missing_package(self, seeded_marketplace: PluginMarketplace) -> None:
        # Entry exists in index but no package directory
        result = seeded_marketplace.install("hello-world")
        assert result["success"] is False
        assert "not found in local registry" in result["message"]

    @patch("saido_agent.plugins.marketplace._store_install_plugin")
    def test_install_success(
        self,
        mock_install: MagicMock,
        seeded_marketplace: PluginMarketplace,
    ) -> None:
        mock_install.return_value = (True, "Installed successfully.")

        # Create the package directory
        pkg_dir = seeded_marketplace._packages_dir / "hello-world"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "plugin.json").write_text(json.dumps({
            "name": "hello-world",
            "version": "1.0.0",
            "author": "Alice",
            "license": "MIT",
        }))

        result = seeded_marketplace.install("hello-world")
        assert result["success"] is True
        mock_install.assert_called_once()

    @patch("saido_agent.plugins.marketplace._store_install_plugin")
    def test_install_increments_downloads(
        self,
        mock_install: MagicMock,
        seeded_marketplace: PluginMarketplace,
    ) -> None:
        mock_install.return_value = (True, "OK")

        pkg_dir = seeded_marketplace._packages_dir / "hello-world"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "plugin.json").write_text(json.dumps({
            "name": "hello-world",
            "version": "1.0.0",
            "author": "Alice",
            "license": "MIT",
        }))

        old_downloads = seeded_marketplace.get_details("hello-world").downloads
        seeded_marketplace.install("hello-world")
        new_downloads = seeded_marketplace.get_details("hello-world").downloads
        assert new_downloads == old_downloads + 1


# ── Publish Tests ────────────────────────────────────────────────────────────

class TestPublish:
    def test_publish_nonexistent_dir(self, tmp_marketplace: PluginMarketplace) -> None:
        result = tmp_marketplace.publish("/no/such/dir")
        assert result.status == "rejected"
        assert result.error is not None

    def test_publish_clean_unsigned_pending(
        self,
        tmp_marketplace: PluginMarketplace,
        tmp_path: Path,
    ) -> None:
        plugin_dir = _make_clean_plugin(tmp_path, signed=False)
        result = tmp_marketplace.publish(str(plugin_dir))
        assert result.status == "pending_review"
        assert "manifest_v2" in result.checks_passed
        assert "restricted_imports" in result.checks_passed
        assert not result.checks_failed

    def test_publish_clean_signed_approved(
        self,
        tmp_marketplace: PluginMarketplace,
        tmp_path: Path,
    ) -> None:
        plugin_dir = _make_clean_plugin(tmp_path, signed=True)
        result = tmp_marketplace.publish(str(plugin_dir))
        assert result.status == "approved"
        assert not result.checks_failed

    def test_publish_adds_to_index(
        self,
        tmp_marketplace: PluginMarketplace,
        tmp_path: Path,
    ) -> None:
        plugin_dir = _make_clean_plugin(tmp_path)
        tmp_marketplace.publish(str(plugin_dir))
        entry = tmp_marketplace.get_details("clean-plugin")
        assert entry is not None
        assert entry.version == "1.0.0"
        assert entry.author == "Test Author"

    def test_publish_copies_package(
        self,
        tmp_marketplace: PluginMarketplace,
        tmp_path: Path,
    ) -> None:
        plugin_dir = _make_clean_plugin(tmp_path)
        tmp_marketplace.publish(str(plugin_dir))
        assert (tmp_marketplace._packages_dir / "clean-plugin" / "plugin.json").exists()


# ── Automated Checks Tests ───────────────────────────────────────────────────

class TestAutomatedChecks:
    def test_catches_restricted_imports(
        self,
        tmp_marketplace: PluginMarketplace,
        tmp_path: Path,
    ) -> None:
        plugin_dir = _make_dirty_plugin(tmp_path)
        result = tmp_marketplace._run_automated_checks(str(plugin_dir))
        assert "restricted_imports" in result.checks_failed

    def test_passes_clean_plugin(
        self,
        tmp_marketplace: PluginMarketplace,
        tmp_path: Path,
    ) -> None:
        plugin_dir = _make_clean_plugin(tmp_path)
        result = tmp_marketplace._run_automated_checks(str(plugin_dir))
        assert "restricted_imports" in result.checks_passed
        assert "manifest_v2" in result.checks_passed
        assert "dependency_audit" in result.checks_passed
        assert not result.checks_failed

    def test_fails_bad_manifest(
        self,
        tmp_marketplace: PluginMarketplace,
        tmp_path: Path,
    ) -> None:
        plugin_dir = _make_bad_manifest_plugin(tmp_path)
        result = tmp_marketplace._run_automated_checks(str(plugin_dir))
        assert "manifest_v2" in result.checks_failed

    def test_fails_bad_dependency_name(
        self,
        tmp_marketplace: PluginMarketplace,
        tmp_path: Path,
    ) -> None:
        plugin_dir = _make_clean_plugin(tmp_path, name="bad-deps")
        manifest_path = plugin_dir / "plugin.json"
        data = json.loads(manifest_path.read_text())
        data["dependencies"] = ["valid-pkg", "../../evil"]
        manifest_path.write_text(json.dumps(data))

        result = tmp_marketplace._run_automated_checks(str(plugin_dir))
        assert "dependency_audit" in result.checks_failed

    def test_publish_rejects_dirty_plugin(
        self,
        tmp_marketplace: PluginMarketplace,
        tmp_path: Path,
    ) -> None:
        plugin_dir = _make_dirty_plugin(tmp_path)
        result = tmp_marketplace.publish(str(plugin_dir))
        assert result.status == "rejected"
        assert "restricted_imports" in result.checks_failed


# ── Category Filtering Tests ─────────────────────────────────────────────────

class TestCategoryFiltering:
    def test_list_all_no_filter(self, seeded_marketplace: PluginMarketplace) -> None:
        entries = seeded_marketplace.list_all()
        assert len(entries) == 3

    def test_list_filter_by_category(self, seeded_marketplace: PluginMarketplace) -> None:
        entries = seeded_marketplace.list_all(category="visualization")
        assert len(entries) == 1
        assert entries[0].name == "data-viz"

    def test_list_filter_empty_category(self, seeded_marketplace: PluginMarketplace) -> None:
        entries = seeded_marketplace.list_all(category="security")
        assert entries == []


# ── Index Persistence Tests ──────────────────────────────────────────────────

class TestIndexPersistence:
    def test_save_and_load_roundtrip(self, tmp_marketplace: PluginMarketplace) -> None:
        entries = [
            MarketplaceEntry(
                name="test-plugin",
                version="1.0.0",
                author="Tester",
                description="A test",
                downloads=42,
                rating=4.2,
                categories=["utility"],
                signed=True,
                published_at="2026-01-01T00:00:00+00:00",
            )
        ]
        tmp_marketplace._save_index(entries)

        loaded = tmp_marketplace._load_index()
        assert len(loaded) == 1
        assert loaded[0].name == "test-plugin"
        assert loaded[0].downloads == 42
        assert loaded[0].rating == 4.2
        assert loaded[0].signed is True

    def test_load_empty_index(self, tmp_marketplace: PluginMarketplace) -> None:
        entries = tmp_marketplace._load_index()
        assert entries == []

    def test_load_corrupted_index(self, tmp_marketplace: PluginMarketplace) -> None:
        tmp_marketplace._index_file.write_text("not valid json{{{")
        entries = tmp_marketplace._load_index()
        assert entries == []

    def test_add_to_index_replaces_existing(self, tmp_marketplace: PluginMarketplace) -> None:
        entry_v1 = MarketplaceEntry(name="plugin-a", version="1.0.0", author="A", description="v1")
        entry_v2 = MarketplaceEntry(name="plugin-a", version="2.0.0", author="A", description="v2")

        tmp_marketplace._add_to_index(entry_v1)
        tmp_marketplace._add_to_index(entry_v2)

        entries = tmp_marketplace._load_index()
        assert len(entries) == 1
        assert entries[0].version == "2.0.0"


# ── Signed vs Unsigned Approval Tests ────────────────────────────────────────

class TestSignedApproval:
    def test_signed_plugin_auto_approved(
        self,
        tmp_marketplace: PluginMarketplace,
        tmp_path: Path,
    ) -> None:
        plugin_dir = _make_clean_plugin(tmp_path, name="signed-ok", signed=True)
        result = tmp_marketplace.publish(str(plugin_dir))
        assert result.status == "approved"

    def test_unsigned_plugin_pending_review(
        self,
        tmp_marketplace: PluginMarketplace,
        tmp_path: Path,
    ) -> None:
        plugin_dir = _make_clean_plugin(tmp_path, name="unsigned-ok", signed=False)
        result = tmp_marketplace.publish(str(plugin_dir))
        assert result.status == "pending_review"


# ── Sort Tests ───────────────────────────────────────────────────────────────

class TestSorting:
    def test_sort_by_downloads(self, seeded_marketplace: PluginMarketplace) -> None:
        entries = seeded_marketplace.list_all(sort_by="downloads")
        downloads = [e.downloads for e in entries]
        assert downloads == sorted(downloads, reverse=True)

    def test_sort_by_rating(self, seeded_marketplace: PluginMarketplace) -> None:
        entries = seeded_marketplace.list_all(sort_by="rating")
        ratings = [e.rating for e in entries]
        assert ratings == sorted(ratings, reverse=True)

    def test_sort_by_name(self, seeded_marketplace: PluginMarketplace) -> None:
        entries = seeded_marketplace.list_all(sort_by="name")
        names = [e.name for e in entries]
        assert names == sorted(names, key=str.lower)

    def test_sort_by_published_at(self, seeded_marketplace: PluginMarketplace) -> None:
        entries = seeded_marketplace.list_all(sort_by="published_at")
        dates = [e.published_at for e in entries]
        assert dates == sorted(dates, reverse=True)


# ── Get Details Tests ────────────────────────────────────────────────────────

class TestGetDetails:
    def test_get_existing(self, seeded_marketplace: PluginMarketplace) -> None:
        entry = seeded_marketplace.get_details("data-viz")
        assert entry is not None
        assert entry.version == "2.1.0"
        assert entry.author == "Bob"

    def test_get_nonexistent(self, seeded_marketplace: PluginMarketplace) -> None:
        entry = seeded_marketplace.get_details("no-plugin")
        assert entry is None


# ── Update Index Tests ───────────────────────────────────────────────────────

class TestUpdateIndex:
    def test_update_index_from_packages(
        self,
        tmp_marketplace: PluginMarketplace,
    ) -> None:
        # Create a package directory with a valid manifest
        pkg_dir = tmp_marketplace._packages_dir / "rebuilder"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "plugin.json").write_text(json.dumps({
            "name": "rebuilder",
            "version": "0.2.0",
            "author": "Indexer",
            "license": "Apache-2.0",
            "description": "Index rebuild test",
            "tags": ["utility"],
        }))

        tmp_marketplace._update_index()

        entries = tmp_marketplace._load_index()
        assert len(entries) == 1
        assert entries[0].name == "rebuilder"
        assert entries[0].categories == ["utility"]
