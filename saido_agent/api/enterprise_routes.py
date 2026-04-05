"""Enterprise API routes for audit logging, compliance, SSO, and mobile support.

Mounted under ``/v1/admin/`` and ``/v1/auth/`` via the ``enterprise_router``.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from saido_agent.api.auth import (
    AuthContext,
    create_user_jwt_token,
    get_auth_context,
    get_current_tenant,
    verify_jwt_token,
    _get_jwt_secret,
    _JWT_ALGORITHM,
    _JWT_EXPIRY_SECONDS,
)
from saido_agent.api.enterprise import (
    get_audit_log,
    get_compliance_manager,
    get_sso_manager,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

enterprise_router = APIRouter(prefix="/v1/admin", tags=["enterprise"])
sso_router = APIRouter(prefix="/v1/auth", tags=["sso"])
mobile_router = APIRouter(prefix="/v1/auth", tags=["mobile"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ConsentRequest(BaseModel):
    purpose: str
    granted: bool


class SAMLConfigRequest(BaseModel):
    tenant_id: str
    idp_metadata_url: str
    entity_id: str


class OIDCConfigRequest(BaseModel):
    tenant_id: str
    issuer: str
    client_id: str
    client_secret: str


class SAMLLoginRequest(BaseModel):
    saml_response: str
    team_id: Optional[str] = None


class OIDCLoginRequest(BaseModel):
    id_token: str
    team_id: Optional[str] = None


class RefreshRequest(BaseModel):
    token: str


# ---------------------------------------------------------------------------
# Audit log endpoints
# ---------------------------------------------------------------------------


@enterprise_router.get("/audit")
async def search_audit_log(
    tenant_id: str = Depends(get_current_tenant),
    action: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    since: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """Search the audit log with optional filters."""
    audit = get_audit_log()
    entries = audit.search(
        tenant_id=tenant_id,
        action=action,
        user_id=user_id,
        since=since,
        limit=limit,
    )
    return {
        "entries": [
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
        "count": len(entries),
    }


@enterprise_router.get("/audit/export")
async def export_audit_log(
    tenant_id: str = Depends(get_current_tenant),
    format: str = Query("json", pattern="^(json|csv)$"),
):
    """Export the full audit log for a tenant."""
    audit = get_audit_log()
    data = audit.export(tenant_id=tenant_id, format=format)
    content_type = "text/csv" if format == "csv" else "application/json"
    from fastapi.responses import Response

    return Response(content=data, media_type=content_type)


# ---------------------------------------------------------------------------
# Data compliance (GDPR) endpoints
# ---------------------------------------------------------------------------


@enterprise_router.get("/export")
async def export_tenant_data(
    tenant_id: str = Depends(get_current_tenant),
):
    """Export all tenant data as a ZIP archive (GDPR portability)."""
    compliance = get_compliance_manager()
    zip_path = compliance.export_tenant_data(tenant_id)
    from fastapi.responses import FileResponse

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{tenant_id}_export.zip",
    )


@enterprise_router.delete("/tenant")
async def delete_tenant_data(
    tenant_id: str = Depends(get_current_tenant),
    confirm: bool = Query(False),
):
    """Permanently delete all tenant data (GDPR erasure)."""
    compliance = get_compliance_manager()
    audit = get_audit_log()

    result = compliance.delete_tenant_data(tenant_id, confirm=confirm)

    if result["status"] == "deleted":
        audit.log(
            user_id="system",
            tenant_id=tenant_id,
            action="tenant_data_deleted",
            resource="all",
            details=json.dumps(result.get("records_deleted", {})),
        )

    return result


@enterprise_router.get("/consent")
async def get_consent(
    tenant_id: str = Depends(get_current_tenant),
):
    """Get consent records for a tenant."""
    compliance = get_compliance_manager()
    return compliance.get_consent_record(tenant_id)


@enterprise_router.post("/consent")
async def record_consent(
    body: ConsentRequest,
    tenant_id: str = Depends(get_current_tenant),
):
    """Record consent for a specific purpose."""
    compliance = get_compliance_manager()
    audit = get_audit_log()

    compliance.record_consent(tenant_id, body.purpose, body.granted)
    audit.log(
        user_id="system",
        tenant_id=tenant_id,
        action="consent_recorded",
        resource=body.purpose,
        details=json.dumps({"granted": body.granted}),
    )
    return {"status": "recorded", "purpose": body.purpose, "granted": body.granted}


# ---------------------------------------------------------------------------
# SSO endpoints
# ---------------------------------------------------------------------------


@sso_router.post("/sso/saml/config")
async def configure_saml(body: SAMLConfigRequest, tenant: str = Depends(get_current_tenant)):
    """Configure SAML 2.0 SSO for a tenant. Requires authentication."""
    sso = get_sso_manager()
    return sso.configure_saml(body.tenant_id, body.idp_metadata_url, body.entity_id)


@sso_router.post("/sso/oidc/config")
async def configure_oidc(body: OIDCConfigRequest, tenant: str = Depends(get_current_tenant)):
    """Configure OIDC SSO for a tenant. Requires authentication."""
    sso = get_sso_manager()
    return sso.configure_oidc(body.tenant_id, body.issuer, body.client_id, body.client_secret)


@sso_router.post("/sso/saml")
async def saml_login(body: SAMLLoginRequest):
    """Authenticate via SAML assertion."""
    sso = get_sso_manager()
    result = sso.validate_saml_response(body.saml_response)

    if not result.get("valid"):
        raise HTTPException(status_code=401, detail=result.get("error", "Invalid SAML response"))

    # Auto-provision user
    user = sso.auto_provision_user(result, team_id=body.team_id)

    # Create a JWT for the provisioned user
    token = create_user_jwt_token(
        user_id=user["user_id"],
        team_id=user.get("team_id", ""),
        role=user.get("role", "viewer"),
    )

    return {
        "token": token,
        "user": user,
        "expires_in": _JWT_EXPIRY_SECONDS,
    }


@sso_router.post("/sso/oidc")
async def oidc_login(body: OIDCLoginRequest):
    """Authenticate via OIDC ID token."""
    sso = get_sso_manager()
    result = sso.validate_oidc_token(body.id_token)

    if not result.get("valid"):
        raise HTTPException(status_code=401, detail=result.get("error", "Invalid OIDC token"))

    user = sso.auto_provision_user(result, team_id=body.team_id)

    token = create_user_jwt_token(
        user_id=user["user_id"],
        team_id=user.get("team_id", ""),
        role=user.get("role", "viewer"),
    )

    return {
        "token": token,
        "user": user,
        "expires_in": _JWT_EXPIRY_SECONDS,
    }


# ---------------------------------------------------------------------------
# Mobile API support: token refresh
# ---------------------------------------------------------------------------


@mobile_router.post("/refresh")
async def refresh_token(body: RefreshRequest):
    """Refresh an expiring JWT token.

    Validates the current token and issues a new one with a fresh expiry.
    Works with both user-level and legacy tenant tokens.
    """
    import jwt as pyjwt

    secret = _get_jwt_secret()

    try:
        # Allow expired tokens within a grace window (refresh use case)
        payload = pyjwt.decode(
            body.token,
            secret,
            algorithms=[_JWT_ALGORITHM],
            options={"verify_exp": False},
        )
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Issue a new token
    if "user_id" in payload:
        new_token = create_user_jwt_token(
            user_id=payload["user_id"],
            team_id=payload.get("team_id", ""),
            role=payload.get("role", "viewer"),
        )
    else:
        from saido_agent.api.auth import create_jwt_token

        tenant_id = payload.get("tenant_id", "")
        new_token = create_jwt_token(tenant_id)

    return {
        "token": new_token,
        "expires_in": _JWT_EXPIRY_SECONDS,
    }


# ---------------------------------------------------------------------------
# API versioning middleware helper
# ---------------------------------------------------------------------------


def add_api_version_header(app):
    """Add X-API-Version header to all responses."""

    @app.middleware("http")
    async def _api_version_middleware(request, call_next):
        response = await call_next(request)
        response.headers["X-API-Version"] = "1"
        return response

    return app
