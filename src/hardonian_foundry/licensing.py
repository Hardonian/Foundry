"""Fail-closed model licensing and provenance validator (policy allowlist).

Decisions are ALLOWED or DENIED with reasons; DENIED never becomes a silent default.
Promotion is NEVER granted here: until a signed promotion mechanism exists,
promotion_allowed is always False, mirroring the fail-closed registry promote.

This validator checks declared metadata; it does not cryptographically verify that a
weight file matches its manifest or license. Treat ALLOWED as "declared fields are on
the allowlist and complete", still requiring human review before any use.
"""
import ipaddress
import re
from urllib.parse import urlsplit

ALLOWED_LICENSES = frozenset({"Apache-2.0", "MIT", "BSD-3-Clause", "CC0-1.0", "OFL-1.1"})
ALLOWED_PROVENANCE = frozenset({"synthetic", "public_research"})
REQUIRED_FIELDS = ("name", "license_id", "provenance", "data_origin")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_NAME = 128


def _clean_str(value, limit=MAX_NAME):
    return isinstance(value, str) and value.strip() == value and bool(value) and len(value) <= limit


def _valid_source(value):
    if not _clean_str(value, 2048) or any(ord(c) < 32 or c.isspace() for c in value):
        return False
    try:
        parsed = urlsplit(value)
        port = parsed.port
        if not parsed.hostname or parsed.username is not None or parsed.password is not None or port == 0:
            return False
        if parsed.scheme == "https":
            return True
        return parsed.scheme == "http" and port is not None and ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        return False


def evaluate(manifest):
    """Return a review-only licensing decision dict; never raises on bad input."""
    reasons = []
    if not isinstance(manifest, dict):
        return {"decision": "DENIED", "reasons": ["manifest must be a dict"], "promotion_allowed": False}
    for field in REQUIRED_FIELDS:
        if field not in manifest:
            reasons.append(f"missing required field: {field}")
    for field in manifest:
        if field not in REQUIRED_FIELDS and field not in ("weights_sha256", "source_url", "notes"):
            reasons.append(f"unknown field: {field}")
    name = manifest.get("name")
    if "name" in manifest and not _clean_str(name):
        reasons.append("name must be a bounded non-space-padded string")
    license_id = manifest.get("license_id")
    if "license_id" in manifest and (not isinstance(license_id, str) or license_id not in ALLOWED_LICENSES):
        reasons.append("license_id not on approved allowlist")
    provenance = manifest.get("provenance")
    if "provenance" in manifest and (not isinstance(provenance, str) or provenance not in ALLOWED_PROVENANCE):
        reasons.append("provenance must be an approved synthetic/public-research value")
    data_origin = manifest.get("data_origin")
    if "data_origin" in manifest and data_origin != "synthetic":
        reasons.append("only synthetic data origins are permitted")
    if "weights_sha256" in manifest and not (isinstance(manifest["weights_sha256"], str)
                                             and HEX64.fullmatch(manifest["weights_sha256"])):
        reasons.append("weights_sha256 must be 64-char lowercase hex when present")
    if "source_url" in manifest:
        source = manifest["source_url"]
        if not _valid_source(source):
            reasons.append("source_url must be https or numeric loopback http")
    decision = "ALLOWED" if not reasons else "DENIED"
    return {
        "decision": decision,
        "reasons": sorted(reasons),
        "promotion_allowed": False,  # fail-closed until signed promotion exists
        "review_required": True,
    }
