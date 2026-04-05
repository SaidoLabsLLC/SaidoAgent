"""Tests for the frontend scaffold.

Validates that all required files exist, configs are valid JSON,
and source files contain expected content.
"""

import json
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
SRC = FRONTEND / "src"


# ---------------------------------------------------------------------------
# Config files
# ---------------------------------------------------------------------------


class TestConfigFiles:
    """Verify project configuration files are valid."""

    def test_package_json_is_valid(self):
        """package.json is valid JSON with required deps."""
        pkg = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))
        assert pkg["name"] == "saido-agent-ui"
        assert "react" in pkg["dependencies"]
        assert "react-dom" in pkg["dependencies"]
        assert "react-router-dom" in pkg["dependencies"]
        assert "typescript" in pkg["devDependencies"]
        assert "vite" in pkg["devDependencies"]
        assert "tailwindcss" in pkg["devDependencies"]
        assert "dev" in pkg["scripts"]
        assert "build" in pkg["scripts"]

    def test_tsconfig_is_valid(self):
        """tsconfig.json is valid JSON."""
        cfg = json.loads((FRONTEND / "tsconfig.json").read_text(encoding="utf-8"))
        assert "compilerOptions" in cfg
        assert cfg["compilerOptions"]["jsx"] == "react-jsx"

    def test_vite_config_exists(self):
        """vite.config.ts exists and has plugin-react."""
        path = FRONTEND / "vite.config.ts"
        assert path.is_file()
        content = path.read_text(encoding="utf-8")
        assert "react" in content
        assert "defineConfig" in content

    def test_tailwind_config_exists(self):
        """tailwind.config.js exists and has content rules."""
        path = FRONTEND / "tailwind.config.js"
        assert path.is_file()
        content = path.read_text(encoding="utf-8")
        assert "content" in content
        assert "tsx" in content

    def test_index_html_exists(self):
        """index.html references main.tsx entry point."""
        path = FRONTEND / "index.html"
        assert path.is_file()
        content = path.read_text(encoding="utf-8")
        assert "main.tsx" in content
        assert "root" in content

    def test_postcss_config_exists(self):
        """postcss.config.js exists."""
        assert (FRONTEND / "postcss.config.js").is_file()


# ---------------------------------------------------------------------------
# Source files existence and content
# ---------------------------------------------------------------------------


class TestSourceFiles:
    """Verify all .tsx/.ts source files exist and have meaningful content."""

    @pytest.mark.parametrize(
        "relpath",
        [
            "src/main.tsx",
            "src/App.tsx",
            "src/api/client.ts",
            "src/components/Layout.tsx",
            "src/components/ChatMessage.tsx",
            "src/components/ArticleCard.tsx",
            "src/components/SearchBar.tsx",
            "src/pages/Login.tsx",
            "src/pages/Dashboard.tsx",
            "src/pages/Documents.tsx",
            "src/pages/DocumentDetail.tsx",
            "src/pages/Chat.tsx",
            "src/pages/Ingest.tsx",
            "src/pages/Settings.tsx",
            "src/pages/Cost.tsx",
            "src/hooks/useAuth.ts",
            "src/hooks/useApi.ts",
            "src/types/index.ts",
            "src/styles/globals.css",
        ],
    )
    def test_file_exists_with_content(self, relpath: str):
        """Every required source file exists and is non-empty."""
        path = FRONTEND / relpath
        assert path.is_file(), f"Missing: {relpath}"
        content = path.read_text(encoding="utf-8")
        assert len(content) > 50, f"File too small (likely placeholder): {relpath}"


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


class TestApiClient:
    """Verify the API client has expected auth header logic."""

    def test_auth_header_logic(self):
        """client.ts includes Authorization header injection."""
        content = (SRC / "api" / "client.ts").read_text(encoding="utf-8")
        assert "Authorization" in content
        assert "Bearer" in content
        assert "localStorage" in content
        assert "apiGet" in content
        assert "apiPost" in content
        assert "streamQuery" in content

    def test_api_base_env(self):
        """client.ts reads VITE_API_URL for base URL."""
        content = (SRC / "api" / "client.ts").read_text(encoding="utf-8")
        assert "VITE_API_URL" in content

    def test_stream_query_sse(self):
        """streamQuery uses SSE with Accept: text/event-stream."""
        content = (SRC / "api" / "client.ts").read_text(encoding="utf-8")
        assert "text/event-stream" in content
        assert "EventSource" in content or "event-stream" in content


# ---------------------------------------------------------------------------
# Routes in App.tsx
# ---------------------------------------------------------------------------


class TestAppRoutes:
    """Verify all pages are wired into App.tsx routes."""

    def test_all_pages_in_routes(self):
        """App.tsx imports and routes to every page component."""
        content = (SRC / "App.tsx").read_text(encoding="utf-8")

        # Check imports
        for page in [
            "Login",
            "Dashboard",
            "Chat",
            "Documents",
            "DocumentDetail",
            "Ingest",
            "Settings",
            "Cost",
        ]:
            assert page in content, f"Page '{page}' not found in App.tsx"

        # Check route paths
        for path in ["/chat", "/documents", "/ingest", "/settings", "/costs"]:
            assert path in content, f"Route '{path}' not found in App.tsx"

    def test_layout_used(self):
        """App.tsx uses the Layout component."""
        content = (SRC / "App.tsx").read_text(encoding="utf-8")
        assert "Layout" in content

    def test_auth_gating(self):
        """App.tsx checks auth before showing routes."""
        content = (SRC / "App.tsx").read_text(encoding="utf-8")
        assert "isAuthenticated" in content
        assert "useAuth" in content


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class TestTypes:
    """Verify TypeScript types match the API models."""

    def test_types_have_api_models(self):
        """types/index.ts has interfaces matching key API responses."""
        content = (SRC / "types" / "index.ts").read_text(encoding="utf-8")
        for model in [
            "LoginRequest",
            "LoginResponse",
            "DocumentSummary",
            "DocumentDetail",
            "QueryResponse",
            "Citation",
            "IngestResponse",
            "StatsResponse",
            "ChatMessage",
            "Team",
            "TeamMember",
            "SearchResult",
        ]:
            assert model in content, f"Type '{model}' not found in types/index.ts"


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------


class TestDockerfile:
    """Verify Dockerfile includes frontend build stage."""

    def test_node_build_stage(self):
        """Dockerfile has a node build stage for the frontend."""
        content = (FRONTEND.parent / "Dockerfile").read_text(encoding="utf-8")
        assert "node:" in content.lower() or "FROM node" in content
        assert "npm" in content
        assert "frontend" in content

    def test_copies_dist(self):
        """Dockerfile copies built frontend dist to runtime stage."""
        content = (FRONTEND.parent / "Dockerfile").read_text(encoding="utf-8")
        assert "frontend/dist" in content or "frontend\\dist" in content


# ---------------------------------------------------------------------------
# Server static mount
# ---------------------------------------------------------------------------


class TestServerStaticMount:
    """Verify server.py has static file serving."""

    def test_static_files_mount(self):
        """server.py mounts StaticFiles for frontend."""
        path = FRONTEND.parent / "saido_agent" / "api" / "server.py"
        content = path.read_text(encoding="utf-8")
        assert "StaticFiles" in content
        assert "frontend" in content
