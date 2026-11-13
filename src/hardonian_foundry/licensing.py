"""Fail-closed model licensing and provenance validator (policy allowlist).

Decisions are ALLOWED or DENIED with reasons; DENIED never becomes a silent default.
Promotion is NEVER granted here: promotion_allowed is always False; promotion requires
all signed Refinery gates to verify independently.

By default this validator checks declared synthetic metadata; when verify_weights_path
is supplied, it also verifies local artifact SHA-256 against weights_sha256. Non-synthetic
governance tiers (consented_pilot, redacted_eval) are permitted only when explicitly
enabled via allowed_data_origins AND backed by a valid governance_attestation.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ALLOWED_LICENSES = frozenset({"Apache-2.0", "MIT", "BSD-3-Clause", "CC0-1.0", "OFL-1.1"})
ALLOWED_PROVENANCE = frozenset({"synthetic", "public_research"})
GOVERNANCE_DATA_ORIGINS = frozenset({"consented_pilot", "redacted_eval"})
SUPPORTED_DATA_ORIGINS = frozenset({"synthetic"}) | GOVERNANCE_DATA_ORIGINS
REQUIRED_FIELDS = ("name", "license_id", "provenance", "data_origin")
OPTIONAL_FIELDS = frozenset({"weights_sha256", "source_url", "notes", "governance_attestation"})
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_NAME = 128
MAX_NOTES = 1024

_ATTESTATION_REQUIRED = frozenset({"tier", "attestation_id", "approved_by", "approved_at"})
_ATTESTATION_ALLOWED = _ATTESTATION_REQUIRED | frozenset({"redacted", "consent_scope", "notes"})


def _clean_str(value: Any, limit: int = MAX_NAME) -> bool:
    return (
        isinstance(value, str)
        and value.strip() == value
        and bool(value)
        and len(value) <= limit
        and all(ord(ch) >= 32 for ch in value)
    )


def _valid_notes(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) <= MAX_NOTES
        and all(ord(ch) >= 32 or ch in "\n\r\t" for ch in value)
    )


def _valid_source(value: Any) -> bool:
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


def _normalize_allowed_origins(allowed_data_origins: Any) -> frozenset[str] | None:
    if isinstance(allowed_data_origins, (str, bytes)) or not isinstance(allowed_data_origins, Iterable):
        return None
    origins: set[str] = set()
    for item in allowed_data_origins:
        if not isinstance(item, str) or item not in SUPPORTED_DATA_ORIGINS:
            return None
        origins.add(item)
    return frozenset(origins) if origins else None


def _validate_governance_attestation(attestation: Any, *, expected_tier: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(attestation, dict):
        return ["governance_attestation must be a dict"]
    missing = sorted(_ATTESTATION_REQUIRED - set(attestation))
    if missing:
        errors.append(f"governance_attestation missing required fields: {', '.join(missing)}")
    unknown = sorted(set(attestation) - _ATTESTATION_ALLOWED)
    if unknown:
        errors.append(f"governance_attestation has unknown fields: {', '.join(unknown)}")
    if "tier" in attestation and attestation["tier"] != expected_tier:
        errors.append("governance_attestation tier must match data_origin")
    for field, limit in (("attestation_id", 128), ("approved_by", 128), ("approved_at", 64)):
        if field in attestation and not _clean_str(attestation[field], limit):
            errors.append(f"governance_attestation.{field} must be a clean nonempty string <= {limit} chars")
    if "consent_scope" in attestation and not _clean_str(attestation["consent_scope"], 256):
        errors.append("governance_attestation.consent_scope must be a clean nonempty string <= 256 chars")
    if "notes" in attestation and not _valid_notes(attestation["notes"]):
        errors.append("governance_attestation.notes must be a bounded string <= 1024 chars")
    if "redacted" in attestation:
        if type(attestation["redacted"]) is not bool:
            errors.append("governance_attestation.redacted must be boolean")
        elif expected_tier == "redacted_eval" and not attestation["redacted"]:
            errors.append("redacted_eval requires governance_attestation.redacted to be true")
    return errors


def _verify_weights_file(path_like: str | os.PathLike[str], expected_sha256: str) -> tuple[bool, str | None]:
    try:
        target = Path(path_like)
        if not target.is_file():
            return False, "weights artifact not found or is not a regular file"
        with target.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
    except (OSError, TypeError, ValueError):
        return False, "weights artifact could not be read for verification"
    if not hmac.compare_digest(actual, expected_sha256):
        return False, "weights artifact SHA-256 does not match weights_sha256"
    return True, None


def evaluate(
    manifest: Any,
    *,
    allowed_data_origins: Iterable[str] = ("synthetic",),
    verify_weights_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return a review-only licensing decision dict; never raises on bad input."""
    reasons: list[str] = []
    if not isinstance(manifest, dict):
        result: dict[str, Any] = {
            "decision": "DENIED",
            "reasons": ["manifest must be a dict"],
            "promotion_allowed": False,
            "review_required": True,
        }
        if verify_weights_path is not None:
            result["weights_verified"] = False
        return result

    allowed_origins = _normalize_allowed_origins(allowed_data_origins)
    if allowed_origins is None:
        reasons.append("allowed_data_origins must be a nonempty collection of supported data origin tiers")
        allowed_origins = frozenset({"synthetic"})

    for field in REQUIRED_FIELDS:
        if field not in manifest:
            reasons.append(f"missing required field: {field}")
    for field in manifest:
        if field not in REQUIRED_FIELDS and field not in OPTIONAL_FIELDS:
            reasons.append(f"unknown field: {field}")

    name = manifest.get("name")
    if "name" in manifest and not _clean_str(name):
        reasons.append("name must be a bounded non-space-padded string")

    license_id = manifest.get("license_id")
    if "license_id" in manifest and (not isinstance(license_id, str) or license_id not in ALLOWED_LICENSES):
        reasons.append("license_id not on approved allowlist")

    data_origin = manifest.get("data_origin")
    origin_valid = False
    if "data_origin" in manifest:
        if allowed_origins == frozenset({"synthetic"}):
            if data_origin != "synthetic":
                reasons.append("only synthetic data origins are permitted")
            else:
                origin_valid = True
        elif not isinstance(data_origin, str) or data_origin not in allowed_origins:
            reasons.append("data_origin is not permitted by allowed_data_origins")
        else:
            origin_valid = True

    provenance = manifest.get("provenance")
    if "provenance" in manifest:
        extra_prov = (
            {data_origin}
            if origin_valid and isinstance(data_origin, str) and data_origin in GOVERNANCE_DATA_ORIGINS
            else set()
        )
        if not isinstance(provenance, str) or (provenance not in ALLOWED_PROVENANCE and provenance not in extra_prov):
            reasons.append("provenance must be an approved synthetic/public-research value")

    if isinstance(data_origin, str) and data_origin in GOVERNANCE_DATA_ORIGINS and origin_valid:
        if "governance_attestation" not in manifest:
            reasons.append(f"data_origin '{data_origin}' requires explicit governance_attestation")
        else:
            reasons.extend(_validate_governance_attestation(manifest["governance_attestation"], expected_tier=data_origin))
    elif "governance_attestation" in manifest and isinstance(data_origin, str) and origin_valid:
        reasons.extend(_validate_governance_attestation(manifest["governance_attestation"], expected_tier=data_origin))

    weights_hash = manifest.get("weights_sha256")
    weights_hash_valid = isinstance(weights_hash, str) and HEX64.fullmatch(weights_hash) is not None
    if "weights_sha256" in manifest and not weights_hash_valid:
        reasons.append("weights_sha256 must be 64-char lowercase hex when present")

    if "source_url" in manifest and not _valid_source(manifest["source_url"]):
        reasons.append("source_url must be https or numeric loopback http")

    if "notes" in manifest and not _valid_notes(manifest["notes"]):
        reasons.append("notes must be a bounded string <= 1024 chars")

    weights_verified: bool | None = None
    if verify_weights_path is not None:
        if not isinstance(weights_hash, str) or HEX64.fullmatch(weights_hash) is None:
            reasons.append("weights_sha256 is required when verify_weights_path is supplied")
            weights_verified = False
        else:
            ok, err = _verify_weights_file(verify_weights_path, weights_hash)
            weights_verified = ok
            if not ok and err:
                reasons.append(err)

    decision = "ALLOWED" if not reasons else "DENIED"
    result = {
        "decision": decision,
        "reasons": sorted(set(reasons)),
        "promotion_allowed": False,  # fail-closed until signed promotion gates pass
        "review_required": True,
    }
    if weights_verified is not None:
        result["weights_verified"] = weights_verified
    return result
