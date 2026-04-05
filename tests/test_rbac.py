"""Tests for RBAC, user management, teams, and WebSocket events.

Validates:
  - User registration and authentication
  - Team creation and member management
  - Role permissions enforced (viewer can't ingest, editor can't manage members)
  - Admin has full access
  - WebSocket connection authenticated
  - Events broadcast to team members
  - Cross-team isolation (user can't access other team's data)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Isolate all auth and DB state to tmp_path for every test."""
    saido_dir = tmp_path / ".saido_agent"
    saido_dir.mkdir()

    # Isolate auth module paths
    import saido_agent.api.auth as auth_mod
    monkeypatch.setattr(auth_mod, "_SAIDO_DIR", saido_dir)
    monkeypatch.setattr(auth_mod, "_API_KEYS_FILE", saido_dir / "api_keys.json")
    monkeypatch.setattr(auth_mod, "_JWT_SECRET_FILE", saido_dir / "jwt_secret")
    auth_mod.reset_rate_limits()

    # Use an in-memory DB path scoped to this test
    db_file = str(tmp_path / "test_rbac.db")

    # Patch get_connection to use our test DB in ALL modules that import it
    import saido_agent.api.db as db_mod
    import saido_agent.api.users as users_mod
    original_get_connection = db_mod.get_connection

    def _patched_get_connection(db_path=None):
        return original_get_connection(db_path or db_file)

    monkeypatch.setattr(db_mod, "get_connection", _patched_get_connection)
    monkeypatch.setattr(db_mod, "_DB_PATH", Path(db_file))
    # Also patch the reference that users.py imported directly
    monkeypatch.setattr(users_mod, "get_connection", _patched_get_connection)

    # Run migrations to set up schema
    from saido_agent.api.db import run_migrations
    run_migrations(db_path=db_file)

    yield db_file


@pytest.fixture(autouse=True)
def _mock_agent_factory(monkeypatch):
    """Replace get_agent_for_tenant with a mock factory."""
    from dataclasses import dataclass, field

    @dataclass
    class _MockStoreStats:
        document_count: int = 10
        category_count: int = 3
        concept_count: int = 5
        total_size_bytes: int = 1024

    @dataclass
    class _MockIngestResult:
        slug: str = "test-doc"
        title: str = "Test Document"
        status: str = "created"
        children: list = field(default_factory=list)
        error: Optional[str] = None

    @dataclass
    class _MockQueryResult:
        answer: str = "This is the answer."
        citations: list = field(default_factory=list)
        confidence: str = "high"
        retrieval_stats: dict = field(default_factory=dict)
        tokens_used: int = 150
        provider: str = "mock/model"

    @dataclass
    class _MockSearchResult:
        slug: str = "found-doc"
        title: str = "Found Document"
        summary: str = "A summary"
        score: float = 0.95
        snippet: str = "A relevant snippet"

    @dataclass
    class _MockAgentResult:
        output: str = "Agent completed the task."
        tool_calls: list = field(default_factory=list)
        tokens_used: int = 200

    class _MockBridge:
        def __init__(self):
            self._articles = {}

        def list_articles(self):
            return []

        def read_article(self, slug):
            return self._articles.get(slug)

        def read_article_frontmatter(self, slug):
            return None

    class _MockAgent:
        def __init__(self, knowledge_dir="./knowledge"):
            self._bridge = _MockBridge()
            self._stats = _MockStoreStats()

        @property
        def bridge(self):
            return self._bridge

        @property
        def stats(self):
            return self._stats

        def ingest(self, path):
            return _MockIngestResult()

        def query(self, question, context=None):
            return _MockQueryResult()

        def search(self, query, top_k=5):
            return [_MockSearchResult()]

        def run(self, instruction):
            return _MockAgentResult()

    _agents = {}

    def _factory(tenant_id):
        if tenant_id not in _agents:
            _agents[tenant_id] = _MockAgent()
        return _agents[tenant_id]

    import saido_agent.api.server as server_mod
    monkeypatch.setattr(server_mod, "get_agent_for_tenant", _factory)
    yield _agents


@pytest.fixture()
def client():
    """FastAPI test client."""
    from saido_agent.api.server import app
    return TestClient(app)


@pytest.fixture()
def api_key():
    """Create and return a valid API key for tenant 'test-tenant'."""
    from saido_agent.api.auth import create_api_key
    return create_api_key("test-tenant")


def _register_user(client, email="alice@example.com", name="Alice", password="SecureP@ss1"):
    """Helper: register a user and return the response data."""
    resp = client.post("/v1/auth/register", json={
        "email": email, "name": name, "password": password,
    })
    assert resp.status_code == 200, f"Registration failed: {resp.text}"
    return resp.json()


def _login_user(client, email="alice@example.com", password="SecureP@ss1", team_id=None):
    """Helper: login a user and return the response data."""
    body = {"email": email, "password": password}
    if team_id:
        body["team_id"] = team_id
    resp = client.post("/v1/auth/login", json=body)
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    return resp.json()


def _auth_header(token):
    """Helper: build an Authorization header from a token."""
    return {"Authorization": f"Bearer {token}"}


# ===========================================================================
# Tests: User Registration and Authentication
# ===========================================================================


class TestUserRegistration:
    """User registration via POST /v1/auth/register."""

    def test_register_new_user(self, client):
        data = _register_user(client)
        assert data["email"] == "alice@example.com"
        assert data["name"] == "Alice"
        assert "user_id" in data

    def test_register_duplicate_email_returns_409(self, client):
        _register_user(client)
        resp = client.post("/v1/auth/register", json={
            "email": "alice@example.com", "name": "Alice2", "password": "AnotherP@ss1",
        })
        assert resp.status_code == 409

    def test_register_email_case_insensitive(self, client):
        _register_user(client, email="BOB@EXAMPLE.COM")
        resp = client.post("/v1/auth/register", json={
            "email": "bob@example.com", "name": "Bob2", "password": "P@ss2",
        })
        assert resp.status_code == 409


class TestUserAuthentication:
    """User login via POST /v1/auth/login."""

    def test_login_success(self, client):
        _register_user(client)
        data = _login_user(client)
        assert data["email"] == "alice@example.com"
        assert "token" in data
        assert data["user_id"]

    def test_login_wrong_password_returns_401(self, client):
        _register_user(client)
        resp = client.post("/v1/auth/login", json={
            "email": "alice@example.com", "password": "WrongPassword",
        })
        assert resp.status_code == 401

    def test_login_nonexistent_user_returns_401(self, client):
        resp = client.post("/v1/auth/login", json={
            "email": "nobody@example.com", "password": "anything",
        })
        assert resp.status_code == 401

    def test_login_jwt_can_access_protected_endpoint(self, client):
        """JWT from login can be used to access authenticated endpoints."""
        _register_user(client)
        login_data = _login_user(client)
        token = login_data["token"]

        # Use token to access stats (a read endpoint)
        resp = client.get("/v1/stats", headers=_auth_header(token))
        assert resp.status_code == 200


# ===========================================================================
# Tests: Team Management
# ===========================================================================


class TestTeamCreation:
    """Team CRUD via /v1/teams endpoints."""

    def test_create_team(self, client):
        _register_user(client)
        login_data = _login_user(client)
        token = login_data["token"]

        resp = client.post(
            "/v1/teams",
            json={"name": "My Team"},
            headers=_auth_header(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "My Team"
        assert data["owner_id"] == login_data["user_id"]
        assert data["role"] == "admin"

    def test_list_teams(self, client):
        _register_user(client)
        login_data = _login_user(client)
        token = login_data["token"]

        # Create two teams
        client.post("/v1/teams", json={"name": "Team A"}, headers=_auth_header(token))
        client.post("/v1/teams", json={"name": "Team B"}, headers=_auth_header(token))

        resp = client.get("/v1/teams", headers=_auth_header(token))
        assert resp.status_code == 200
        teams = resp.json()
        assert len(teams) >= 2
        names = {t["name"] for t in teams}
        assert "Team A" in names
        assert "Team B" in names


class TestTeamMemberManagement:
    """Team member operations: add, update role, remove."""

    def _setup_team(self, client):
        """Create admin user, team, and second user. Returns tokens and IDs."""
        admin = _register_user(client, email="admin@test.com", name="Admin", password="AdminP@ss1")
        viewer = _register_user(client, email="viewer@test.com", name="Viewer", password="ViewerP@ss1")

        admin_login = _login_user(client, email="admin@test.com", password="AdminP@ss1")
        admin_token = admin_login["token"]

        # Create team
        team_resp = client.post(
            "/v1/teams",
            json={"name": "Test Team"},
            headers=_auth_header(admin_token),
        )
        team_id = team_resp.json()["id"]

        # Re-login with team context
        admin_login = _login_user(
            client, email="admin@test.com", password="AdminP@ss1", team_id=team_id,
        )

        return {
            "admin_id": admin["user_id"],
            "admin_token": admin_login["token"],
            "viewer_id": viewer["user_id"],
            "team_id": team_id,
        }

    def test_add_member(self, client):
        setup = self._setup_team(client)

        resp = client.post(
            f"/v1/teams/{setup['team_id']}/members",
            json={"user_id": setup["viewer_id"], "role": "viewer"},
            headers=_auth_header(setup["admin_token"]),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == setup["viewer_id"]
        assert data["role"] == "viewer"

    def test_update_member_role(self, client):
        setup = self._setup_team(client)

        # Add member first
        client.post(
            f"/v1/teams/{setup['team_id']}/members",
            json={"user_id": setup["viewer_id"], "role": "viewer"},
            headers=_auth_header(setup["admin_token"]),
        )

        # Update to editor
        resp = client.patch(
            f"/v1/teams/{setup['team_id']}/members/{setup['viewer_id']}",
            json={"role": "editor"},
            headers=_auth_header(setup["admin_token"]),
        )
        assert resp.status_code == 200
        assert resp.json()["role"] == "editor"

    def test_remove_member(self, client):
        setup = self._setup_team(client)

        # Add then remove
        client.post(
            f"/v1/teams/{setup['team_id']}/members",
            json={"user_id": setup["viewer_id"], "role": "viewer"},
            headers=_auth_header(setup["admin_token"]),
        )

        resp = client.delete(
            f"/v1/teams/{setup['team_id']}/members/{setup['viewer_id']}",
            headers=_auth_header(setup["admin_token"]),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "removed"

    def test_add_duplicate_member_returns_409(self, client):
        setup = self._setup_team(client)

        client.post(
            f"/v1/teams/{setup['team_id']}/members",
            json={"user_id": setup["viewer_id"], "role": "viewer"},
            headers=_auth_header(setup["admin_token"]),
        )

        resp = client.post(
            f"/v1/teams/{setup['team_id']}/members",
            json={"user_id": setup["viewer_id"], "role": "editor"},
            headers=_auth_header(setup["admin_token"]),
        )
        assert resp.status_code == 409


# ===========================================================================
# Tests: Role Permission Enforcement
# ===========================================================================


class TestRolePermissions:
    """Verify RBAC permission checks at the module level."""

    def test_viewer_permissions(self):
        from saido_agent.api.rbac import Role, check_permission

        assert check_permission(Role.VIEWER, "read_articles") is True
        assert check_permission(Role.VIEWER, "query") is True
        assert check_permission(Role.VIEWER, "search") is True
        assert check_permission(Role.VIEWER, "view_stats") is True
        # Viewer should NOT have editor/admin permissions
        assert check_permission(Role.VIEWER, "ingest") is False
        assert check_permission(Role.VIEWER, "run_agent") is False
        assert check_permission(Role.VIEWER, "manage_members") is False
        assert check_permission(Role.VIEWER, "delete_articles") is False

    def test_editor_permissions(self):
        from saido_agent.api.rbac import Role, check_permission

        assert check_permission(Role.EDITOR, "read_articles") is True
        assert check_permission(Role.EDITOR, "ingest") is True
        assert check_permission(Role.EDITOR, "compile") is True
        assert check_permission(Role.EDITOR, "run_agent") is True
        # Editor should NOT have admin permissions
        assert check_permission(Role.EDITOR, "manage_members") is False
        assert check_permission(Role.EDITOR, "manage_keys") is False
        assert check_permission(Role.EDITOR, "delete_articles") is False

    def test_admin_has_all_permissions(self):
        from saido_agent.api.rbac import Role, ROLE_PERMISSIONS, check_permission

        all_perms = set()
        for perms in ROLE_PERMISSIONS.values():
            all_perms.update(perms)

        for perm in all_perms:
            assert check_permission(Role.ADMIN, perm) is True, f"Admin missing permission: {perm}"


class TestRoleEnforcementViaAPI:
    """Test that role permissions are enforced through the API layer."""

    def _setup_team_with_roles(self, client):
        """Create a team with admin, editor, and viewer users."""
        admin = _register_user(client, email="admin@r.com", name="Admin", password="AdminP@ss1")
        editor = _register_user(client, email="editor@r.com", name="Editor", password="EditorP@ss1")
        viewer = _register_user(client, email="viewer@r.com", name="Viewer", password="ViewerP@ss1")

        # Admin logs in and creates team
        admin_login = _login_user(client, email="admin@r.com", password="AdminP@ss1")
        team_resp = client.post(
            "/v1/teams",
            json={"name": "RBAC Team"},
            headers=_auth_header(admin_login["token"]),
        )
        team_id = team_resp.json()["id"]

        # Re-login admin with team context
        admin_login = _login_user(client, email="admin@r.com", password="AdminP@ss1", team_id=team_id)

        # Admin adds editor and viewer
        client.post(
            f"/v1/teams/{team_id}/members",
            json={"user_id": editor["user_id"], "role": "editor"},
            headers=_auth_header(admin_login["token"]),
        )
        client.post(
            f"/v1/teams/{team_id}/members",
            json={"user_id": viewer["user_id"], "role": "viewer"},
            headers=_auth_header(admin_login["token"]),
        )

        # Login each user with team context
        editor_login = _login_user(client, email="editor@r.com", password="EditorP@ss1", team_id=team_id)
        viewer_login = _login_user(client, email="viewer@r.com", password="ViewerP@ss1", team_id=team_id)

        return {
            "team_id": team_id,
            "admin_token": admin_login["token"],
            "admin_id": admin["user_id"],
            "editor_token": editor_login["token"],
            "editor_id": editor["user_id"],
            "viewer_token": viewer_login["token"],
            "viewer_id": viewer["user_id"],
        }

    def test_viewer_cannot_manage_members(self, client):
        """Viewer cannot add/remove team members."""
        setup = self._setup_team_with_roles(client)
        new_user = _register_user(client, email="new@r.com", name="New", password="NewP@ss1")

        resp = client.post(
            f"/v1/teams/{setup['team_id']}/members",
            json={"user_id": new_user["user_id"], "role": "viewer"},
            headers=_auth_header(setup["viewer_token"]),
        )
        assert resp.status_code == 403

    def test_editor_cannot_manage_members(self, client):
        """Editor cannot add/remove team members."""
        setup = self._setup_team_with_roles(client)
        new_user = _register_user(client, email="new2@r.com", name="New2", password="NewP@ss2")

        resp = client.post(
            f"/v1/teams/{setup['team_id']}/members",
            json={"user_id": new_user["user_id"], "role": "viewer"},
            headers=_auth_header(setup["editor_token"]),
        )
        assert resp.status_code == 403

    def test_admin_can_manage_members(self, client):
        """Admin can add team members."""
        setup = self._setup_team_with_roles(client)
        new_user = _register_user(client, email="new3@r.com", name="New3", password="NewP@ss3")

        resp = client.post(
            f"/v1/teams/{setup['team_id']}/members",
            json={"user_id": new_user["user_id"], "role": "viewer"},
            headers=_auth_header(setup["admin_token"]),
        )
        assert resp.status_code == 200

    def test_viewer_can_read_stats(self, client):
        """Viewer can access read-only endpoints like stats."""
        setup = self._setup_team_with_roles(client)
        resp = client.get("/v1/stats", headers=_auth_header(setup["viewer_token"]))
        assert resp.status_code == 200

    def test_editor_can_query(self, client):
        """Editor can access query endpoint."""
        setup = self._setup_team_with_roles(client)
        resp = client.post(
            "/v1/query",
            json={"question": "What is this?"},
            headers=_auth_header(setup["editor_token"]),
        )
        assert resp.status_code == 200


# ===========================================================================
# Tests: Cross-Team Isolation
# ===========================================================================


class TestCrossTeamIsolation:
    """Verify users cannot access other teams' data."""

    def test_user_cannot_manage_other_team(self, client):
        """A user who is admin on Team A cannot manage Team B."""
        # Create two users, each with their own team
        alice = _register_user(client, email="alice@iso.com", name="Alice", password="AliceP@ss1")
        bob = _register_user(client, email="bob@iso.com", name="Bob", password="BobP@ss1")

        alice_login = _login_user(client, email="alice@iso.com", password="AliceP@ss1")
        bob_login = _login_user(client, email="bob@iso.com", password="BobP@ss1")

        # Alice creates her team
        alice_team = client.post(
            "/v1/teams",
            json={"name": "Alice Team"},
            headers=_auth_header(alice_login["token"]),
        ).json()

        # Bob creates his team
        bob_team = client.post(
            "/v1/teams",
            json={"name": "Bob Team"},
            headers=_auth_header(bob_login["token"]),
        ).json()

        # Re-login Alice with her team context
        alice_login = _login_user(
            client, email="alice@iso.com", password="AliceP@ss1",
            team_id=alice_team["id"],
        )

        # Alice tries to add someone to Bob's team -- should fail
        new_user = _register_user(client, email="carol@iso.com", name="Carol", password="CarolP@ss1")
        resp = client.post(
            f"/v1/teams/{bob_team['id']}/members",
            json={"user_id": new_user["user_id"], "role": "viewer"},
            headers=_auth_header(alice_login["token"]),
        )
        # Alice's token is scoped to her team, so she's not admin of Bob's team
        assert resp.status_code == 403

    def test_user_only_sees_own_teams(self, client):
        """Users only see teams they belong to."""
        alice = _register_user(client, email="alice2@iso.com", name="Alice2", password="AliceP@ss2")
        bob = _register_user(client, email="bob2@iso.com", name="Bob2", password="BobP@ss2")

        alice_login = _login_user(client, email="alice2@iso.com", password="AliceP@ss2")
        bob_login = _login_user(client, email="bob2@iso.com", password="BobP@ss2")

        # Each creates a team
        client.post("/v1/teams", json={"name": "Alice Only"}, headers=_auth_header(alice_login["token"]))
        client.post("/v1/teams", json={"name": "Bob Only"}, headers=_auth_header(bob_login["token"]))

        # Alice lists her teams
        resp = client.get("/v1/teams", headers=_auth_header(alice_login["token"]))
        assert resp.status_code == 200
        team_names = {t["name"] for t in resp.json()}
        assert "Alice Only" in team_names
        assert "Bob Only" not in team_names


# ===========================================================================
# Tests: WebSocket Authentication
# ===========================================================================


class TestWebSocket:
    """WebSocket connection and event broadcasting."""

    def _create_user_in_team(self, client, email, name, password):
        """Register user, create team, login with team context."""
        user = _register_user(client, email=email, name=name, password=password)
        login = _login_user(client, email=email, password=password)
        team_resp = client.post(
            "/v1/teams",
            json={"name": f"{name}'s Team"},
            headers=_auth_header(login["token"]),
        )
        team = team_resp.json()
        login = _login_user(client, email=email, password=password, team_id=team["id"])
        return {
            "user_id": user["user_id"],
            "team_id": team["id"],
            "token": login["token"],
        }

    def test_websocket_rejects_missing_token(self, client):
        """WebSocket without token is rejected."""
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises((WebSocketDisconnect, Exception)):
            with client.websocket_connect("/v1/ws?team_id=some-team") as ws:
                ws.receive_text()

    def test_websocket_rejects_invalid_token(self, client):
        """WebSocket with invalid JWT is rejected."""
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises((WebSocketDisconnect, Exception)):
            with client.websocket_connect("/v1/ws?token=invalid.jwt.token&team_id=t") as ws:
                ws.receive_text()

    def test_websocket_rejects_non_member(self, client):
        """WebSocket rejects user not in the requested team."""
        from starlette.websockets import WebSocketDisconnect

        setup = self._create_user_in_team(client, "ws@test.com", "WSUser", "WsP@ss1")
        with pytest.raises((WebSocketDisconnect, Exception)):
            with client.websocket_connect(
                f"/v1/ws?token={setup['token']}&team_id=nonexistent-team"
            ) as ws:
                ws.receive_text()

    def test_websocket_accepts_valid_member(self, client):
        """WebSocket accepts authenticated team member."""
        setup = self._create_user_in_team(client, "ws2@test.com", "WSUser2", "WsP@ss2")

        with client.websocket_connect(
            f"/v1/ws?token={setup['token']}&team_id={setup['team_id']}"
        ) as ws:
            # Send a ping, expect pong
            ws.send_text("ping")
            data = ws.receive_text()
            parsed = json.loads(data)
            assert parsed["event"] == "pong"

    def test_websocket_broadcast_event(self, client):
        """Events are broadcast to connected team members."""
        import asyncio
        from saido_agent.api.websocket import WSEvent, manager, EVENT_ARTICLE_CREATED

        setup = self._create_user_in_team(client, "ws3@test.com", "WSUser3", "WsP@ss3")

        with client.websocket_connect(
            f"/v1/ws?token={setup['token']}&team_id={setup['team_id']}"
        ) as ws:
            # Broadcast an event to the team
            event = WSEvent(
                event_type=EVENT_ARTICLE_CREATED,
                team_id=setup["team_id"],
                data={"slug": "new-article", "title": "New Article"},
            )
            # Run broadcast synchronously via asyncio
            loop = asyncio.new_event_loop()
            loop.run_until_complete(manager.broadcast(event))
            loop.close()

            data = ws.receive_text()
            parsed = json.loads(data)
            assert parsed["event"] == EVENT_ARTICLE_CREATED
            assert parsed["data"]["slug"] == "new-article"


# ===========================================================================
# Tests: Password Hashing
# ===========================================================================


class TestPasswordHashing:
    """Verify password hashing and verification."""

    def test_hash_and_verify(self):
        from saido_agent.api.users import _hash_password, _verify_password

        pw = "MySecretPassword123!"
        hashed = _hash_password(pw)

        assert hashed.startswith("scrypt:")
        assert _verify_password(pw, hashed) is True
        assert _verify_password("WrongPassword", hashed) is False

    def test_different_hashes_for_same_password(self):
        """Each hash should have a unique salt."""
        from saido_agent.api.users import _hash_password

        h1 = _hash_password("same_password")
        h2 = _hash_password("same_password")
        assert h1 != h2  # Different salts

    def test_verify_rejects_corrupt_hash(self):
        from saido_agent.api.users import _verify_password

        assert _verify_password("anything", "not-a-valid-hash") is False
        assert _verify_password("anything", "scrypt:bad:data") is False


# ===========================================================================
# Tests: Legacy API Key Backward Compatibility
# ===========================================================================


class TestLegacyAPIKeyCompat:
    """Verify Phase 2 API keys still work alongside Phase 3 user auth."""

    def test_api_key_still_works_for_existing_endpoints(self, client, api_key):
        """Legacy API keys should continue to work for all endpoints."""
        resp = client.get("/v1/stats", headers={"X-API-Key": api_key})
        assert resp.status_code == 200

    def test_api_key_can_exchange_for_jwt(self, client, api_key):
        """Legacy API key -> JWT exchange still works."""
        resp = client.post("/v1/auth/token", json={"api_key": api_key})
        assert resp.status_code == 200
        token = resp.json()["token"]

        resp = client.get("/v1/stats", headers=_auth_header(token))
        assert resp.status_code == 200
