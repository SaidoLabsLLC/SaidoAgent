"""Tests for enterprise features: audit logging, GDPR compliance, SSO, mobile API.

Validates:
  - Audit log records entries and is append-only
  - Audit log search with filters
  - Audit log export to JSON and CSV
  - Data export creates ZIP with tenant data
  - Data deletion removes all tenant records (with confirmation guard)
  - Consent recording and retrieval
  - SSO SAML/OIDC configuration storage
  - SAML/OIDC validation (mock)
  - Auto-provision creates user with role mapping
  - Token refresh endpoint
  - API versioning header
"""

from __future__ import annotations

import json
import os
import time
import zipfile
from pathlib import Path
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

    # Use a file-based DB scoped to this test
    db_file = str(tmp_path / "test_enterprise.db")

    # Patch get_connection to use our test DB in ALL modules that import it
    import saido_agent.api.db as db_mod
    import saido_agent.api.enterprise as ent_mod

    original_get_connection = db_mod.get_connection

    def _patched_get_connection(db_path=None):
        return original_get_connection(db_path or db_file)

    monkeypatch.setattr(db_mod, "get_connection", _patched_get_connection)
    monkeypatch.setattr(db_mod, "_DB_PATH", Path(db_file))
    monkeypatch.setattr(ent_mod, "get_connection", _patched_get_connection)

    # Also patch users module if available
    try:
        import saido_agent.api.users as users_mod

        monkeypatch.setattr(users_mod, "get_connection", _patched_get_connection)
    except (ImportError, AttributeError):
        pass

    # Also patch billing module if available
    try:
        import saido_agent.api.billing as billing_mod

        monkeypatch.setattr(billing_mod, "get_connection", _patched_get_connection)
    except (ImportError, AttributeError):
        pass

    # Reset enterprise singletons
    ent_mod.reset_enterprise_managers()

    # Run migrations to set up schema
    from saido_agent.api.db import run_migrations

    run_migrations(db_path=db_file)

    yield db_file


@pytest.fixture()
def audit_log(_isolated_env):
    """Return an EnterpriseAuditLog configured for the test database."""
    from saido_agent.api.enterprise import EnterpriseAuditLog

    return EnterpriseAuditLog(db_path=_isolated_env)


@pytest.fixture()
def compliance(_isolated_env):
    """Return a DataComplianceManager configured for the test database."""
    from saido_agent.api.enterprise import DataComplianceManager

    return DataComplianceManager(db_path=_isolated_env)


@pytest.fixture()
def sso_manager(_isolated_env):
    """Return an SSOManager configured for the test database."""
    from saido_agent.api.enterprise import SSOManager

    return SSOManager(db_path=_isolated_env)


@pytest.fixture()
def client(_isolated_env):
    """Return a FastAPI TestClient with mocked agent factory."""
    mock_agent = MagicMock()

    import saido_agent.api.server as server_mod

    with patch.object(
        server_mod,
        "get_agent_for_tenant",
        return_value=mock_agent,
    ):
        from saido_agent.api.server import app

        yield TestClient(app)


@pytest.fixture()
def auth_header(_isolated_env) -> dict[str, str]:
    """Create an API key and return an auth header dict."""
    from saido_agent.api.auth import create_api_key

    key = create_api_key("test-tenant")
    return {"X-API-Key": key}


# ---------------------------------------------------------------------------
# Audit Log Tests
# ---------------------------------------------------------------------------


class TestAuditLog:
    """Test enterprise audit logging."""

    def test_log_records_entry(self, audit_log):
        """Audit log records an entry with all fields."""
        entry = audit_log.log(
            user_id="user-1",
            tenant_id="tenant-1",
            action="create",
            resource="document",
            details="Created a new document",
            ip_address="192.168.1.1",
        )
        assert entry.id > 0
        assert entry.user_id == "user-1"
        assert entry.tenant_id == "tenant-1"
        assert entry.action == "create"
        assert entry.resource == "document"
        assert entry.details == "Created a new document"
        assert entry.ip_address == "192.168.1.1"
        assert entry.timestamp != ""

    def test_log_multiple_entries(self, audit_log):
        """Multiple log entries are recorded with incrementing IDs."""
        e1 = audit_log.log(user_id="u1", tenant_id="t1", action="create")
        e2 = audit_log.log(user_id="u2", tenant_id="t1", action="delete")
        e3 = audit_log.log(user_id="u1", tenant_id="t2", action="update")
        assert e1.id < e2.id < e3.id

    def test_search_all(self, audit_log):
        """Search with no filters returns all entries."""
        audit_log.log(user_id="u1", tenant_id="t1", action="create")
        audit_log.log(user_id="u2", tenant_id="t1", action="delete")
        results = audit_log.search()
        assert len(results) == 2

    def test_search_by_tenant(self, audit_log):
        """Search filters by tenant_id."""
        audit_log.log(user_id="u1", tenant_id="t1", action="create")
        audit_log.log(user_id="u2", tenant_id="t2", action="delete")
        results = audit_log.search(tenant_id="t1")
        assert len(results) == 1
        assert results[0].tenant_id == "t1"

    def test_search_by_action(self, audit_log):
        """Search filters by action."""
        audit_log.log(user_id="u1", tenant_id="t1", action="create")
        audit_log.log(user_id="u2", tenant_id="t1", action="delete")
        results = audit_log.search(action="delete")
        assert len(results) == 1
        assert results[0].action == "delete"

    def test_search_by_user(self, audit_log):
        """Search filters by user_id."""
        audit_log.log(user_id="u1", tenant_id="t1", action="create")
        audit_log.log(user_id="u2", tenant_id="t1", action="delete")
        results = audit_log.search(user_id="u1")
        assert len(results) == 1
        assert results[0].user_id == "u1"

    def test_search_combined_filters(self, audit_log):
        """Search applies multiple filters simultaneously."""
        audit_log.log(user_id="u1", tenant_id="t1", action="create")
        audit_log.log(user_id="u1", tenant_id="t1", action="delete")
        audit_log.log(user_id="u2", tenant_id="t1", action="delete")
        results = audit_log.search(tenant_id="t1", action="delete", user_id="u1")
        assert len(results) == 1
        assert results[0].user_id == "u1"
        assert results[0].action == "delete"

    def test_search_with_limit(self, audit_log):
        """Search respects the limit parameter."""
        for i in range(10):
            audit_log.log(user_id="u1", tenant_id="t1", action=f"action-{i}")
        results = audit_log.search(limit=3)
        assert len(results) == 3

    def test_append_only_no_delete_method(self, audit_log):
        """Audit log class has no delete method -- it is append-only."""
        assert not hasattr(audit_log, "delete")
        assert not hasattr(audit_log, "remove")
        assert not hasattr(audit_log, "update")

    def test_export_json(self, audit_log):
        """Export produces valid JSON with all entries."""
        audit_log.log(user_id="u1", tenant_id="t1", action="create", resource="doc1")
        audit_log.log(user_id="u2", tenant_id="t1", action="delete", resource="doc2")
        data = audit_log.export("t1", format="json")
        parsed = json.loads(data)
        assert len(parsed) == 2
        assert parsed[0]["action"] in ("create", "delete")

    def test_export_csv(self, audit_log):
        """Export produces CSV with header and data rows."""
        audit_log.log(user_id="u1", tenant_id="t1", action="create")
        data = audit_log.export("t1", format="csv")
        lines = data.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row
        assert "id" in lines[0]
        assert "action" in lines[0]

    def test_export_filters_by_tenant(self, audit_log):
        """Export only includes entries for the specified tenant."""
        audit_log.log(user_id="u1", tenant_id="t1", action="create")
        audit_log.log(user_id="u2", tenant_id="t2", action="delete")
        data = audit_log.export("t1")
        parsed = json.loads(data)
        assert len(parsed) == 1
        assert parsed[0]["tenant_id"] == "t1"


# ---------------------------------------------------------------------------
# Data Compliance (GDPR) Tests
# ---------------------------------------------------------------------------


class TestDataCompliance:
    """Test GDPR data compliance features."""

    def test_export_creates_zip(self, compliance, audit_log):
        """Data export creates a ZIP file with tenant data."""
        # Add some audit data first
        audit_log.log(user_id="u1", tenant_id="t1", action="create")

        zip_path = compliance.export_tenant_data("t1")
        assert os.path.exists(zip_path)
        assert zip_path.endswith(".zip")

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert "audit_log.json" in names
            assert "consent_records.json" in names

    def test_export_contains_audit_data(self, compliance, audit_log):
        """Exported ZIP contains actual audit data."""
        audit_log.log(user_id="u1", tenant_id="t1", action="test_action")

        zip_path = compliance.export_tenant_data("t1")
        with zipfile.ZipFile(zip_path, "r") as zf:
            audit_data = json.loads(zf.read("audit_log.json"))
            assert len(audit_data) == 1
            assert audit_data[0]["action"] == "test_action"

    def test_delete_requires_confirmation(self, compliance):
        """Deletion without confirm=True returns an error."""
        result = compliance.delete_tenant_data("t1", confirm=False)
        assert result["status"] == "error"
        assert "Confirmation required" in result["message"]

    def test_delete_removes_consent_records(self, compliance):
        """Deletion with confirm=True removes consent records."""
        compliance.record_consent("t1", "analytics", True)
        consent = compliance.get_consent_record("t1")
        assert len(consent["consents"]) == 1

        result = compliance.delete_tenant_data("t1", confirm=True)
        assert result["status"] == "deleted"
        assert result["records_deleted"]["consent_records"] == 1

        # Verify data is gone
        consent = compliance.get_consent_record("t1")
        assert len(consent["consents"]) == 0

    def test_consent_record_and_retrieve(self, compliance):
        """Consent can be recorded and retrieved."""
        compliance.record_consent("t1", "data_processing", True)
        compliance.record_consent("t1", "marketing", False)

        record = compliance.get_consent_record("t1")
        assert record["tenant_id"] == "t1"
        assert record["consents"]["data_processing"]["granted"] is True
        assert record["consents"]["marketing"]["granted"] is False

    def test_consent_update_overwrites(self, compliance):
        """Re-recording consent for the same purpose updates it."""
        compliance.record_consent("t1", "analytics", True)
        compliance.record_consent("t1", "analytics", False)

        record = compliance.get_consent_record("t1")
        assert record["consents"]["analytics"]["granted"] is False

    def test_consent_isolation_between_tenants(self, compliance):
        """Consent records are isolated between tenants."""
        compliance.record_consent("t1", "analytics", True)
        compliance.record_consent("t2", "analytics", False)

        r1 = compliance.get_consent_record("t1")
        r2 = compliance.get_consent_record("t2")
        assert r1["consents"]["analytics"]["granted"] is True
        assert r2["consents"]["analytics"]["granted"] is False


# ---------------------------------------------------------------------------
# SSO Tests
# ---------------------------------------------------------------------------


class TestSSO:
    """Test SSO configuration and validation."""

    def test_configure_saml(self, sso_manager):
        """SAML configuration is stored successfully."""
        result = sso_manager.configure_saml(
            tenant_id="t1",
            idp_metadata_url="https://idp.example.com/metadata",
            entity_id="saido-sp",
        )
        assert result["status"] == "configured"
        assert result["provider"] == "saml"

    def test_configure_oidc(self, sso_manager):
        """OIDC configuration is stored successfully."""
        result = sso_manager.configure_oidc(
            tenant_id="t1",
            issuer="https://accounts.google.com",
            client_id="client-123",
            client_secret="secret-456",
        )
        assert result["status"] == "configured"
        assert result["provider"] == "oidc"

    def test_get_saml_config(self, sso_manager):
        """Stored SAML config can be retrieved."""
        sso_manager.configure_saml(
            tenant_id="t1",
            idp_metadata_url="https://idp.example.com/metadata",
            entity_id="saido-sp",
        )
        config = sso_manager.get_sso_config("t1", "saml")
        assert config is not None
        assert config["config"]["idp_metadata_url"] == "https://idp.example.com/metadata"
        assert config["config"]["entity_id"] == "saido-sp"

    def test_get_oidc_config(self, sso_manager):
        """Stored OIDC config can be retrieved."""
        sso_manager.configure_oidc(
            tenant_id="t1",
            issuer="https://accounts.google.com",
            client_id="client-123",
            client_secret="secret-456",
        )
        config = sso_manager.get_sso_config("t1", "oidc")
        assert config is not None
        assert config["config"]["issuer"] == "https://accounts.google.com"

    def test_get_nonexistent_config(self, sso_manager):
        """Getting config for unconfigured tenant returns None."""
        result = sso_manager.get_sso_config("t1", "saml")
        assert result is None

    def test_validate_saml_response(self, sso_manager):
        """SAML validation returns user info from mock response."""
        saml_response = json.dumps({
            "user_id": "saml-user-1",
            "email": "user@example.com",
            "name": "Test User",
            "groups": ["editors"],
        })
        result = sso_manager.validate_saml_response(saml_response)
        assert result["valid"] is True
        assert result["user_id"] == "saml-user-1"
        assert result["email"] == "user@example.com"

    def test_validate_saml_invalid(self, sso_manager):
        """Invalid SAML response returns valid=False."""
        result = sso_manager.validate_saml_response("not-valid-xml")
        assert result["valid"] is False

    def test_validate_oidc_token(self, sso_manager):
        """OIDC validation returns user info from mock token."""
        id_token = json.dumps({
            "sub": "oidc-user-1",
            "email": "oidc@example.com",
            "name": "OIDC User",
            "groups": ["admins"],
        })
        result = sso_manager.validate_oidc_token(id_token)
        assert result["valid"] is True
        assert result["user_id"] == "oidc-user-1"

    def test_validate_oidc_invalid(self, sso_manager):
        """Invalid OIDC token returns valid=False."""
        result = sso_manager.validate_oidc_token("garbage")
        assert result["valid"] is False

    def test_auto_provision_user_admin(self, sso_manager):
        """Auto-provision maps 'admins' group to admin role."""
        user_info = {
            "user_id": "u1",
            "email": "admin@example.com",
            "name": "Admin",
            "groups": ["admins"],
        }
        user = sso_manager.auto_provision_user(user_info, team_id="team-1")
        assert user["provisioned"] is True
        assert user["role"] == "admin"
        assert user["team_id"] == "team-1"
        assert user["email"] == "admin@example.com"

    def test_auto_provision_user_editor(self, sso_manager):
        """Auto-provision maps 'editors' group to editor role."""
        user_info = {
            "user_id": "u2",
            "email": "editor@example.com",
            "name": "Editor",
            "groups": ["editors"],
        }
        user = sso_manager.auto_provision_user(user_info)
        assert user["role"] == "editor"

    def test_auto_provision_user_default_viewer(self, sso_manager):
        """Auto-provision defaults to viewer role for unrecognized groups."""
        user_info = {
            "user_id": "u3",
            "email": "viewer@example.com",
            "name": "Viewer",
            "groups": ["users"],
        }
        user = sso_manager.auto_provision_user(user_info)
        assert user["role"] == "viewer"


# ---------------------------------------------------------------------------
# API Route Tests
# ---------------------------------------------------------------------------


class TestAuditAPI:
    """Test audit log API endpoints."""

    def test_search_audit_api(self, client, auth_header, _isolated_env):
        """GET /v1/admin/audit returns audit entries."""
        from saido_agent.api.enterprise import EnterpriseAuditLog

        audit = EnterpriseAuditLog(db_path=_isolated_env)
        audit.log(user_id="u1", tenant_id="test-tenant", action="create")

        resp = client.get("/v1/admin/audit", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data
        assert data["count"] >= 1

    def test_search_audit_with_action_filter(self, client, auth_header, _isolated_env):
        """GET /v1/admin/audit?action=delete filters correctly."""
        from saido_agent.api.enterprise import EnterpriseAuditLog

        audit = EnterpriseAuditLog(db_path=_isolated_env)
        audit.log(user_id="u1", tenant_id="test-tenant", action="create")
        audit.log(user_id="u2", tenant_id="test-tenant", action="delete")

        resp = client.get("/v1/admin/audit?action=delete", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["entries"][0]["action"] == "delete"

    def test_export_audit_api(self, client, auth_header, _isolated_env):
        """GET /v1/admin/audit/export returns JSON data."""
        from saido_agent.api.enterprise import EnterpriseAuditLog

        audit = EnterpriseAuditLog(db_path=_isolated_env)
        audit.log(user_id="u1", tenant_id="test-tenant", action="create")

        resp = client.get("/v1/admin/audit/export", headers=auth_header)
        assert resp.status_code == 200
        data = json.loads(resp.content)
        assert len(data) >= 1


class TestComplianceAPI:
    """Test GDPR compliance API endpoints."""

    def test_get_consent_api(self, client, auth_header):
        """GET /v1/admin/consent returns consent records."""
        resp = client.get("/v1/admin/consent", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] == "test-tenant"

    def test_post_consent_api(self, client, auth_header):
        """POST /v1/admin/consent records consent."""
        resp = client.post(
            "/v1/admin/consent",
            json={"purpose": "analytics", "granted": True},
            headers=auth_header,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "recorded"

        # Verify it was stored
        resp = client.get("/v1/admin/consent", headers=auth_header)
        data = resp.json()
        assert data["consents"]["analytics"]["granted"] is True

    def test_delete_tenant_requires_confirm(self, client, auth_header):
        """DELETE /v1/admin/tenant without confirm returns error."""
        resp = client.delete("/v1/admin/tenant", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"

    def test_delete_tenant_with_confirm(self, client, auth_header):
        """DELETE /v1/admin/tenant?confirm=true deletes data."""
        # Add consent first
        client.post(
            "/v1/admin/consent",
            json={"purpose": "analytics", "granted": True},
            headers=auth_header,
        )

        resp = client.delete("/v1/admin/tenant?confirm=true", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"


class TestSSOAPI:
    """Test SSO API endpoints."""

    def test_saml_login(self, client):
        """POST /v1/auth/sso/saml authenticates via SAML."""
        saml_response = json.dumps({
            "user_id": "saml-user-1",
            "email": "user@example.com",
            "name": "Test User",
            "groups": ["editors"],
        })
        resp = client.post(
            "/v1/auth/sso/saml",
            json={"saml_response": saml_response},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["user"]["provisioned"] is True
        assert data["user"]["role"] == "editor"

    def test_saml_login_invalid(self, client):
        """POST /v1/auth/sso/saml with invalid data returns 401."""
        resp = client.post(
            "/v1/auth/sso/saml",
            json={"saml_response": "invalid"},
        )
        assert resp.status_code == 401

    def test_oidc_login(self, client):
        """POST /v1/auth/sso/oidc authenticates via OIDC."""
        id_token = json.dumps({
            "sub": "oidc-user-1",
            "email": "oidc@example.com",
            "name": "OIDC User",
            "groups": ["admins"],
        })
        resp = client.post(
            "/v1/auth/sso/oidc",
            json={"id_token": id_token},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["user"]["role"] == "admin"

    def test_oidc_login_invalid(self, client):
        """POST /v1/auth/sso/oidc with invalid data returns 401."""
        resp = client.post(
            "/v1/auth/sso/oidc",
            json={"id_token": "bad-token"},
        )
        assert resp.status_code == 401


class TestMobileAPI:
    """Test mobile API support."""

    def test_token_refresh(self, client):
        """POST /v1/auth/refresh issues a new token."""
        from saido_agent.api.auth import create_user_jwt_token

        token = create_user_jwt_token(
            user_id="user-1",
            team_id="team-1",
            role="editor",
        )

        resp = client.post(
            "/v1/auth/refresh",
            json={"token": token},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["expires_in"] == 3600
        # New token should be different
        assert data["token"] != token

    def test_token_refresh_invalid(self, client):
        """POST /v1/auth/refresh with invalid token returns 401."""
        resp = client.post(
            "/v1/auth/refresh",
            json={"token": "invalid-token"},
        )
        assert resp.status_code == 401

    def test_api_version_header(self, client, auth_header):
        """All responses include X-API-Version header."""
        resp = client.get("/health")
        assert resp.headers.get("X-API-Version") == "1"

    def test_api_version_on_authenticated_route(self, client, auth_header):
        """Authenticated routes also include X-API-Version header."""
        resp = client.get("/v1/admin/consent", headers=auth_header)
        assert resp.headers.get("X-API-Version") == "1"
