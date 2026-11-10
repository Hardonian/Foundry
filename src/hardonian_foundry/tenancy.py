"""Tenant identity and isolation checks for synthetic fixtures (fail-closed).

A TenantContext is an immutable, validated holder of tenant and request identifiers.
Reads are authorized by exact tenant match only; every denial returns the same reason
so that response shape cannot leak whether a record exists in another tenant.

This is not authentication: producing a valid TenantContext assumes an upstream,
authenticated layer that does not exist yet. No production API or tenant store is
implemented; do not wire this into any request path with real customer data.
"""
import re
from dataclasses import dataclass

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
DENY_REASON = "access denied"


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str
    request_id: str

    def __post_init__(self):
        for field in ("tenant_id", "request_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not _ID_RE.match(value):
                raise ValueError(f"{field} must match ^[a-z0-9][a-z0-9_-]{{0,63}}$")


def read(ctx, record):
    """Authorize a single-record read; uniform denial, never a partial leak."""
    if not isinstance(ctx, TenantContext):
        return {"status": "DENIED", "reason": DENY_REASON, "review_required": True}
    if not isinstance(record, dict) or "tenant_id" not in record:
        return {"status": "DENIED", "reason": DENY_REASON, "review_required": True}
    if record["tenant_id"] != ctx.tenant_id:
        return {"status": "DENIED", "reason": DENY_REASON, "review_required": True}
    return {"status": "ALLOWED", "reason": "exact tenant match", "review_required": True}


def filter_records(ctx, records):
    """Return only same-tenant records; foreign and malformed entries are dropped."""
    if not isinstance(records, (list, tuple)):
        return []
    allowed = []
    for record in records:
        if read(ctx, record)["status"] == "ALLOWED":
            allowed.append(record)
    return allowed
