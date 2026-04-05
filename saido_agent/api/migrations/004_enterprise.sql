-- Migration 004: Enterprise audit logging and compliance tables.

CREATE TABLE IF NOT EXISTS enterprise_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT '',
    tenant_id TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL,
    resource TEXT NOT NULL DEFAULT '',
    details TEXT NOT NULL DEFAULT '',
    ip_address TEXT NOT NULL DEFAULT '',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Append-only: application code must never UPDATE or DELETE from this table.
CREATE INDEX IF NOT EXISTS idx_audit_tenant ON enterprise_audit(tenant_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON enterprise_audit(action);
CREATE INDEX IF NOT EXISTS idx_audit_user ON enterprise_audit(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON enterprise_audit(timestamp);

CREATE TABLE IF NOT EXISTS consent_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    purpose TEXT NOT NULL,
    granted INTEGER NOT NULL DEFAULT 0,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tenant_id, purpose)
);

CREATE INDEX IF NOT EXISTS idx_consent_tenant ON consent_records(tenant_id);

CREATE TABLE IF NOT EXISTS sso_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    provider_type TEXT NOT NULL CHECK(provider_type IN ('saml', 'oidc')),
    config_json TEXT NOT NULL DEFAULT '{}',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tenant_id, provider_type)
);

CREATE INDEX IF NOT EXISTS idx_sso_tenant ON sso_configs(tenant_id);
