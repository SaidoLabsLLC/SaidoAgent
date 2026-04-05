"""SSRF protection — validate URLs before fetching external content.

Blocks requests to private IP ranges, loopback addresses, link-local
addresses, and cloud metadata endpoints to prevent server-side request
forgery attacks.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blocked private/reserved networks
# ---------------------------------------------------------------------------

BLOCKED_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    # IPv6 equivalents
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

# Explicit block for cloud metadata endpoints
BLOCKED_HOSTS: frozenset[str] = frozenset({
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.internal",
})

# Allowed schemes
ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})


def validate_url(url: str) -> tuple[bool, str]:
    """Check whether a URL is safe to fetch.

    Returns:
        (is_safe, reason) — ``is_safe`` is True when the URL passes all
        checks; ``reason`` explains rejection when ``is_safe`` is False.
    """
    # 1. Parse and validate scheme
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Malformed URL"

    if not parsed.scheme:
        return False, "Missing URL scheme (http/https required)"

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return False, f"Blocked scheme: {parsed.scheme}"

    hostname = parsed.hostname
    if not hostname:
        return False, "Missing hostname"

    # 2. Check explicit blocked hosts
    if hostname.lower() in BLOCKED_HOSTS:
        return False, f"Blocked host: {hostname}"

    # 3. Resolve hostname to IP and validate
    try:
        addr_infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        return False, f"DNS resolution failed for: {hostname}"

    if not addr_infos:
        return False, f"No DNS records for: {hostname}"

    for family, _type, _proto, _canonname, sockaddr in addr_infos:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False, f"Invalid resolved IP: {ip_str}"

        for network in BLOCKED_NETWORKS:
            if ip in network:
                log.warning(
                    "SSRF blocked: %s resolved to %s (in %s)", url, ip, network
                )
                return False, f"Blocked IP range: {ip} is in {network}"

    return True, "ok"


def validate_url_no_resolve(url: str) -> tuple[bool, str]:
    """Lightweight URL validation without DNS resolution.

    Useful for quick pre-checks before the full ``validate_url`` call.
    Checks scheme, hostname presence, and literal IP addresses against
    blocked ranges. Does NOT resolve hostnames.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Malformed URL"

    if not parsed.scheme or parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return False, f"Blocked or missing scheme: {parsed.scheme}"

    hostname = parsed.hostname
    if not hostname:
        return False, "Missing hostname"

    if hostname.lower() in BLOCKED_HOSTS:
        return False, f"Blocked host: {hostname}"

    # Check if hostname is a literal IP
    try:
        ip = ipaddress.ip_address(hostname)
        for network in BLOCKED_NETWORKS:
            if ip in network:
                return False, f"Blocked IP range: {ip} is in {network}"
    except ValueError:
        # Not a literal IP — that's fine, will be resolved later
        pass

    return True, "ok"
