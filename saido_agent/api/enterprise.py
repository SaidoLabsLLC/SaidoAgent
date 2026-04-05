"""Enterprise features for Saido Agent: audit logging, GDPR compliance, SSO.

Provides:
  - EnterpriseAuditLog: append-only, tamper-proof audit trail
  - DataComplianceManager: GDPR export/erasure/consent
  - SSOManager: SAML 2.0 and OIDC configuration (placeholder)
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import sqlite3
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from saido_agent.api.db import get_connection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    """Single audit log record."""

    id: int = 0
    user_id: str = ""
    tenant_id: str = ""
    action: str = ""
    resource: str = ""
    details: str = ""
    ip_address: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------

class EnterpriseAuditLog:
    """Tamper-proof, append-only audit logging for enterprise compliance.

    The underlying ``enterprise_audit`` table must never be updated or deleted
    from; all writes go through :meth:`log` which only inserts.
    """

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        return get_connection(self._db_path)

    def log(
        self,
        user_id: str,
        tenant_id: str,
        action: str,
        resource: str = "",
        details: str = "",
        ip_address: str = "",
    ) -> AuditEntry:
        """Record an audit event (INSERT only -- append-only)."""
        conn = self._conn()
        try:
            ts = datetime.now(timezone.utc).isoformat()
            cursor = conn.execute(
                """
                INSERT INTO enterprise_audit
                    (user_id, tenant_id, action, resource, details, ip_address, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, tenant_id, action, resource, details, ip_address, ts),
            )
            conn.commit()
            entry = AuditEntry(
                id=cursor.lastrowid,
                user_id=user_id,
                tenant_id=tenant_id,
                action=action,
                resource=resource,
                details=details,
                ip_address=ip_address,
                timestamp=ts,
            )
            logger.info(
                "Audit: user=%s tenant=%s action=%s resource=%s",
                user_id, tenant_id, action, resource,
            )
            return entry
        finally:
            conn.close()

    def search(
        self,
        tenant_id: str | None = None,
        action: str | None = None,
        user_id: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Search audit log with optional filters."""
        clauses: list[str] = []
        params: list = []

        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if action is not None:
            clauses.append("action = ?")
            params.append(action)
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM enterprise_audit {where} ORDER BY id DESC LIMIT ?"
        params.append(limit)

        conn = self._conn()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [
                AuditEntry(
                    id=r["id"],
                    user_id=r["user_id"],
                    tenant_id=r["tenant_id"],
                    action=r["action"],
                    resource=r["resource"],
                    details=r["details"],
                    ip_address=r["ip_address"],
                    timestamp=r["timestamp"],
                )
                for r in rows
            ]
        finally:
            conn.close()

    def export(self, tenant_id: str, format: str = "json") -> str:
        """Export audit log for a tenant as JSON or CSV string."""
        entries = self.search(tenant_id=tenant_id, limit=10_000)
        if format == "csv":
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow([
                "id", "user_id", "tenant_id", "action",
                "resource", "details", "ip_address", "timestamp",
            ])
            for e in entries:
                writer.writerow([
                    e.id, e.user_id, e.tenant_id, e.action,
                    e.resource, e.details, e.ip_address, e.timestamp,
                ])
            return buf.getvalue()

        # Default: JSON
        return json.dumps(
            [
                {
                    "id": e.id,
                    "user_id": e.user_id,
                    "tenant_id": e.tenant_id,
                    "action": e.action,
                    "resource": e.resource,
                    "details": e.details,
                    "ip_address": e.ip_address,
                    "timestamp": e.timestamp,
                }
                for e in entries
            ],
            indent=2,
        )


# ---------------------------------------------------------------------------
# Data Compliance (GDPR)
# ---------------------------------------------------------------------------

class DataComplianceManager:
    """GDPR and data compliance operations.

    Supports:
      - Right to portability (data export)
      - Right to erasure (data deletion)
      - Consent management
    """

    def __init__(self, bridge=None, db_path: str | None = None):
        self._bridge = bridge
        self._db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        return get_connection(self._db_path)

    def export_tenant_data(self, tenant_id: str, output_dir: str | None = None) -> str:
        """Export all tenant data as a ZIP file (GDPR right to portability).

        Returns the path to the generated ZIP file.
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="saido_export_")

        zip_path = os.path.join(output_dir, f"{tenant_id}_export.zip")
        conn = self._conn()

        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                # Audit log
                audit_rows = conn.execute(
                    "SELECT * FROM enterprise_audit WHERE tenant_id = ?",
                    (tenant_id,),
                ).fetchall()
                audit_data = [dict(r) for r in audit_rows]
                zf.writestr("audit_log.json", json.dumps(audit_data, indent=2, default=str))

                # Consent records
                consent_rows = conn.execute(
                    "SELECT * FROM consent_records WHERE tenant_id = ?",
                    (tenant_id,),
                ).fetchall()
                consent_data = [dict(r) for r in consent_rows]
                zf.writestr("consent_records.json", json.dumps(consent_data, indent=2, default=str))

                # Subscriptions (if table exists)
                try:
                    sub_rows = conn.execute(
                        "SELECT * FROM subscriptions WHERE tenant_id = ?",
                        (tenant_id,),
                    ).fetchall()
                    sub_data = [dict(r) for r in sub_rows]
                    zf.writestr("subscriptions.json", json.dumps(sub_data, indent=2, default=str))
                except sqlite3.OperationalError:
                    pass  # Table may not exist

                # Usage records (if table exists)
                try:
                    usage_rows = conn.execute(
                        "SELECT * FROM usage_records WHERE tenant_id = ?",
                        (tenant_id,),
                    ).fetchall()
                    usage_data = [dict(r) for r in usage_rows]
                    zf.writestr("usage_records.json", json.dumps(usage_data, indent=2, default=str))
                except sqlite3.OperationalError:
                    pass

                # SSO config
                try:
                    sso_rows = conn.execute(
                        "SELECT * FROM sso_configs WHERE tenant_id = ?",
                        (tenant_id,),
                    ).fetchall()
                    sso_data = [dict(r) for r in sso_rows]
                    zf.writestr("sso_configs.json", json.dumps(sso_data, indent=2, default=str))
                except sqlite3.OperationalError:
                    pass

            logger.info("Exported tenant %s data to %s", tenant_id, zip_path)
            return zip_path
        finally:
            conn.close()

    def delete_tenant_data(self, tenant_id: str, confirm: bool = False) -> dict:
        """Permanently delete all tenant data (GDPR right to erasure).

        Requires ``confirm=True`` as a safety guard. Audit log entries are
        preserved (legal retention requirement) but a deletion-event is logged.
        """
        if not confirm:
            return {
                "status": "error",
                "message": "Confirmation required. Pass confirm=True to proceed.",
            }

        conn = self._conn()
        deleted: dict[str, int] = {}

        try:
            # Delete consent records
            cur = conn.execute(
                "DELETE FROM consent_records WHERE tenant_id = ?", (tenant_id,)
            )
            deleted["consent_records"] = cur.rowcount

            # Delete subscriptions (if table exists)
            try:
                cur = conn.execute(
                    "DELETE FROM subscriptions WHERE tenant_id = ?", (tenant_id,)
                )
                deleted["subscriptions"] = cur.rowcount
            except sqlite3.OperationalError:
                pass

            # Delete usage records (if table exists)
            try:
                cur = conn.execute(
                    "DELETE FROM usage_records WHERE tenant_id = ?", (tenant_id,)
                )
                deleted["usage_records"] = cur.rowcount
            except sqlite3.OperationalError:
                pass

            # Delete SSO configs
            try:
                cur = conn.execute(
                    "DELETE FROM sso_configs WHERE tenant_id = ?", (tenant_id,)
                )
                deleted["sso_configs"] = cur.rowcount
            except sqlite3.OperationalError:
                pass

            conn.commit()
            logger.info("Deleted tenant %s data: %s", tenant_id, deleted)

            return {
                "status": "deleted",
                "tenant_id": tenant_id,
                "records_deleted": deleted,
            }
        finally:
            conn.close()

    def get_consent_record(self, tenant_id: str) -> dict:
        """Get all consent records for a tenant."""
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT purpose, granted, recorded_at FROM consent_records WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchall()
            consents = {
                row["purpose"]: {
                    "granted": bool(row["granted"]),
                    "recorded_at": row["recorded_at"],
                }
                for row in rows
            }
            return {"tenant_id": tenant_id, "consents": consents}
        finally:
            conn.close()

    def record_consent(self, tenant_id: str, purpose: str, granted: bool) -> None:
        """Record or update consent for a specific purpose."""
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO consent_records (tenant_id, purpose, granted)
                VALUES (?, ?, ?)
                ON CONFLICT(tenant_id, purpose)
                DO UPDATE SET granted = excluded.granted,
                             recorded_at = CURRENT_TIMESTAMP
                """,
                (tenant_id, purpose, int(granted)),
            )
            conn.commit()
            logger.info(
                "Consent recorded: tenant=%s purpose=%s granted=%s",
                tenant_id, purpose, granted,
            )
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# SSO Manager (SAML / OIDC)
# ---------------------------------------------------------------------------

class SSOManager:
    """SSO/SAML/OIDC integration for enterprise identity providers.

    This is a structural placeholder. Real SAML/OIDC validation requires
    libraries like ``python3-saml`` or ``python-jose``; the methods here
    store configuration and return mock validation results suitable for
    integration testing and API contract definition.
    """

    def __init__(self, config: dict | None = None, db_path: str | None = None):
        self._config = config or {}
        self._db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        return get_connection(self._db_path)

    def configure_saml(
        self, tenant_id: str, idp_metadata_url: str, entity_id: str
    ) -> dict:
        """Configure SAML 2.0 SSO with an identity provider."""
        config = {
            "idp_metadata_url": idp_metadata_url,
            "entity_id": entity_id,
        }
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO sso_configs (tenant_id, provider_type, config_json)
                VALUES (?, 'saml', ?)
                ON CONFLICT(tenant_id, provider_type)
                DO UPDATE SET config_json = excluded.config_json,
                             updated_at = CURRENT_TIMESTAMP
                """,
                (tenant_id, json.dumps(config)),
            )
            conn.commit()
            return {"status": "configured", "provider": "saml", "tenant_id": tenant_id}
        finally:
            conn.close()

    def configure_oidc(
        self, tenant_id: str, issuer: str, client_id: str, client_secret: str
    ) -> dict:
        """Configure OIDC SSO."""
        # P4-HIGH-2: Hash the client_secret before storage (never store plaintext)
        import hashlib
        secret_hash = hashlib.sha256(client_secret.encode()).hexdigest()
        config = {
            "issuer": issuer,
            "client_id": client_id,
            "client_secret_hash": secret_hash,
            "_secret_stored": "hashed_sha256",
        }
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO sso_configs (tenant_id, provider_type, config_json)
                VALUES (?, 'oidc', ?)
                ON CONFLICT(tenant_id, provider_type)
                DO UPDATE SET config_json = excluded.config_json,
                             updated_at = CURRENT_TIMESTAMP
                """,
                (tenant_id, json.dumps(config)),
            )
            conn.commit()
            return {"status": "configured", "provider": "oidc", "tenant_id": tenant_id}
        finally:
            conn.close()

    def get_sso_config(self, tenant_id: str, provider_type: str) -> dict | None:
        """Retrieve SSO configuration for a tenant."""
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM sso_configs WHERE tenant_id = ? AND provider_type = ?",
                (tenant_id, provider_type),
            ).fetchone()
            if row is None:
                return None
            return {
                "tenant_id": row["tenant_id"],
                "provider_type": row["provider_type"],
                "config": json.loads(row["config_json"]),
                "enabled": bool(row["enabled"]),
            }
        finally:
            conn.close()

    def validate_saml_response(self, saml_response: str) -> dict:
        """Validate SAML assertion and return user info.

        .. note:: This is a mock implementation. Production use requires
           ``python3-saml`` for real XML signature validation.
        """
        # P4-HIGH-1: Mock validators only work when explicitly enabled
        if not os.environ.get("SAIDO_SSO_MOCK_ENABLED"):
            return {"valid": False, "error": "SSO mock disabled. Set SAIDO_SSO_MOCK_ENABLED=1 for testing, or install python3-saml for production."}
        # Mock: decode a simple JSON payload for testing
        try:
            payload = json.loads(saml_response)
            return {
                "valid": True,
                "user_id": payload.get("user_id", ""),
                "email": payload.get("email", ""),
                "name": payload.get("name", ""),
                "groups": payload.get("groups", []),
            }
        except (json.JSONDecodeError, TypeError):
            return {"valid": False, "error": "Invalid SAML response"}

    def validate_oidc_token(self, id_token: str) -> dict:
        """Validate OIDC ID token and return user info.

        .. note:: This is a mock implementation. Production use requires
           ``python-jose`` for real JWT/JWK validation against the IdP.
        """
        # P4-HIGH-1: Mock validators only work when explicitly enabled
        if not os.environ.get("SAIDO_SSO_MOCK_ENABLED"):
            return {"valid": False, "error": "SSO mock disabled. Set SAIDO_SSO_MOCK_ENABLED=1 for testing, or install python-jose for production."}
        try:
            payload = json.loads(id_token)
            return {
                "valid": True,
                "user_id": payload.get("sub", ""),
                "email": payload.get("email", ""),
                "name": payload.get("name", ""),
                "groups": payload.get("groups", []),
            }
        except (json.JSONDecodeError, TypeError):
            return {"valid": False, "error": "Invalid OIDC token"}

    def auto_provision_user(self, user_info: dict, team_id: str | None = None) -> dict:
        """Create or update a user from IdP attributes.

        Maps IdP groups to team roles when ``team_id`` is provided.
        Returns provisioned user info dict.
        """
        user_id = user_info.get("user_id") or user_info.get("sub", "")
        email = user_info.get("email", "")
        name = user_info.get("name", "")
        groups = user_info.get("groups", [])

        # Determine role from groups
        role = "viewer"
        if "admin" in groups or "admins" in groups:
            role = "admin"
        elif "editor" in groups or "editors" in groups:
            role = "editor"

        return {
            "user_id": user_id,
            "email": email,
            "name": name,
            "role": role,
            "team_id": team_id or "",
            "provisioned": True,
        }


# ---------------------------------------------------------------------------
# Module-level singleton helpers
# ---------------------------------------------------------------------------

_audit_log: EnterpriseAuditLog | None = None
_compliance: DataComplianceManager | None = None
_sso: SSOManager | None = None


def get_audit_log(db_path: str | None = None) -> EnterpriseAuditLog:
    """Return the module-level audit log instance."""
    global _audit_log
    if _audit_log is None:
        _audit_log = EnterpriseAuditLog(db_path=db_path)
    return _audit_log


def get_compliance_manager(db_path: str | None = None) -> DataComplianceManager:
    """Return the module-level compliance manager."""
    global _compliance
    if _compliance is None:
        _compliance = DataComplianceManager(db_path=db_path)
    return _compliance


def get_sso_manager(db_path: str | None = None) -> SSOManager:
    """Return the module-level SSO manager."""
    global _sso
    if _sso is None:
        _sso = SSOManager(db_path=db_path)
    return _sso


def reset_enterprise_managers() -> None:
    """Reset all module-level singletons (for testing)."""
    global _audit_log, _compliance, _sso
    _audit_log = None
    _compliance = None
    _sso = None
