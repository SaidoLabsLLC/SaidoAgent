"""Plugin manifest signature verification using Ed25519."""
from __future__ import annotations

import base64
from pathlib import Path

from .types import PluginManifest, PluginSecurityError

# ── Public Key Loading ───────────────────────────────────────────────────────

_KEYS_DIR = Path(__file__).parent / "keys"
_SAIDO_LABS_PUBLIC_KEY_PATH = _KEYS_DIR / "saido_labs_public.pem"


def _load_public_key():
    """Load the Saido Labs Ed25519 public key from the bundled PEM file."""
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    if not _SAIDO_LABS_PUBLIC_KEY_PATH.exists():
        raise PluginSecurityError(
            f"Saido Labs public key not found at {_SAIDO_LABS_PUBLIC_KEY_PATH}. "
            "Cannot verify plugin signatures."
        )
    pem_data = _SAIDO_LABS_PUBLIC_KEY_PATH.read_bytes()
    return load_pem_public_key(pem_data)


def verify_manifest_signature(manifest: PluginManifest) -> bool:
    """Verify that a plugin manifest's signature is valid against the Saido Labs public key.

    Args:
        manifest: The plugin manifest with a 'signature' field (base64-encoded Ed25519 signature).

    Returns:
        True if the signature is valid.

    Raises:
        PluginSecurityError: If signature is missing, malformed, or invalid.
    """
    if not manifest.signature:
        return False

    try:
        signature_bytes = base64.b64decode(manifest.signature)
    except Exception:
        raise PluginSecurityError(
            f"Plugin '{manifest.name}' has a malformed signature (not valid base64)."
        )

    canonical = manifest.canonical_bytes()

    try:
        public_key = _load_public_key()
        public_key.verify(signature_bytes, canonical)
        return True
    except PluginSecurityError:
        raise
    except Exception:
        raise PluginSecurityError(
            f"Plugin '{manifest.name}' signature verification FAILED. "
            "The manifest may have been tampered with."
        )


def sign_manifest(manifest: PluginManifest, private_key_pem: bytes) -> str:
    """Sign a manifest with an Ed25519 private key. Returns base64-encoded signature.

    This is a utility for plugin authors / the Saido Labs build system.
    NOT used at runtime -- only for signing during plugin publication.
    """
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    private_key = load_pem_private_key(private_key_pem, password=None)
    canonical = manifest.canonical_bytes()
    signature = private_key.sign(canonical)
    return base64.b64encode(signature).decode("ascii")


# ── Source Verification ──────────────────────────────────────────────────────

# Trusted registries: plugins from these sources do not require additional approval
DEFAULT_TRUSTED_REGISTRIES: list[str] = [
    "https://github.com/saido-agent-plugins/",
    "https://github.com/saido-labs/",
]


def is_trusted_source(source: str, trusted_registries: list[str] | None = None) -> bool:
    """Check if a plugin source URL comes from a trusted registry."""
    registries = trusted_registries or DEFAULT_TRUSTED_REGISTRIES
    for registry in registries:
        if source.startswith(registry):
            return True
    return False


def classify_source(source: str, trusted_registries: list[str] | None = None) -> str:
    """Classify a plugin source for security review.

    Returns:
        'trusted' - from a trusted registry, no approval needed
        'untrusted_git' - git URL from untrusted source, requires approval
        'local_path' - local filesystem path, always requires approval
    """
    if _is_git_url(source):
        if is_trusted_source(source, trusted_registries):
            return "trusted"
        return "untrusted_git"
    return "local_path"


def _is_git_url(source: str) -> bool:
    return (
        source.startswith("https://")
        or source.startswith("git@")
        or source.startswith("http://")
        or source.endswith(".git")
    )
