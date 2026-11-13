"""Tenant identity and isolation checks for synthetic fixtures (fail-closed).

A TenantContext is an immutable, validated holder of tenant and request identifiers.
Reads are authorized by exact tenant match only; every denial returns the same reason
so that response shape cannot leak whether a record exists in another tenant.

This is not authentication: producing a valid TenantContext assumes an upstream,
authenticated layer that does not exist yet. No production API or tenant store is
implemented; do not wire this into any request path with real customer data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
DENY_REASON = "access denied"


@dataclass(frozen=True)
class TenantContext:
    """Validated immutable tenant and request identifiers."""

    tenant_id: str
    request_id: str

    def __post_init__(self) -> None:
        for field in ("tenant_id", "request_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not _ID_RE.match(value):
                raise ValueError(f"{field} must match ^[a-z0-9][a-z0-9_-]{{0,63}}$")


def _denied() -> dict[str, Any]:
    return {"status": "DENIED", "reason": DENY_REASON, "review_required": True}


def _allowed() -> dict[str, Any]:
    return {"status": "ALLOWED", "reason": "exact tenant match", "review_required": True}


def read(ctx: Any, record: Any) -> dict[str, Any]:
    """Authorize a single-record read; uniform denial, never a partial leak."""
    if not isinstance(ctx, TenantContext):
        return _denied()
    if not isinstance(record, dict) or "tenant_id" not in record:
        return _denied()
    if record["tenant_id"] != ctx.tenant_id:
        return _denied()
    return _allowed()


def filter_records(ctx: Any, records: Any) -> list[dict[str, Any]]:
    """Return only same-tenant records; foreign and malformed entries are dropped."""
    if not isinstance(records, (list, tuple)):
        return []
    allowed: list[dict[str, Any]] = []
    for record in records:
        if read(ctx, record)["status"] == "ALLOWED":
            allowed.append(record)
    return allowed


def authorize_run(ctx: Any, run_record: Any) -> dict[str, Any]:
    """Fail-closed tenant authorization for a tracked run record."""
    if not isinstance(ctx, TenantContext) or not isinstance(run_record, dict):
        return _denied()

    has_top = "tenant_id" in run_record
    top_tid = run_record.get("tenant_id")

    provenance = run_record.get("provenance")
    if "provenance" in run_record and provenance is not None and not isinstance(provenance, dict):
        return _denied()

    has_prov = isinstance(provenance, dict) and "tenant_id" in provenance
    prov_tid = provenance.get("tenant_id") if isinstance(provenance, dict) else None

    if has_top and has_prov and top_tid != prov_tid:
        return _denied()
    if not has_top and not has_prov:
        return _denied()

    candidate = top_tid if has_top else prov_tid
    if not isinstance(candidate, str) or _ID_RE.match(candidate) is None:
        return _denied()
    if candidate != ctx.tenant_id:
        return _denied()
    return _allowed()


def scope_ledger_query(ctx: Any, entries: Any) -> list[dict[str, Any]]:
    """Filter ledger entries strictly to entries owned by ctx.tenant_id."""
    if not isinstance(ctx, TenantContext) or not isinstance(entries, (list, tuple)):
        return []
    scoped: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        record = entry.get("record")
        if not isinstance(record, dict):
            continue
        record_tid = record.get("tenant_id")
        if "tenant_id" in entry and entry.get("tenant_id") != record_tid:
            continue
        if isinstance(record_tid, str) and _ID_RE.match(record_tid) is not None and record_tid == ctx.tenant_id:
            scoped.append(entry)
    return scoped
