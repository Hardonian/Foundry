"""Frozen read-only Proofpack v2 interchange contract for downstream audit consumers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .canonical import digest
from .refinery_runs import _write_json_atomic, generate_audit_report
from .tenancy import TenantContext

PROOFPACK_SCHEMA_VERSION = "2.0"
PROOFPACK_TYPE = "hardonian.foundry.proofpack.v2"
_DOWNSTREAM_CONSUMERS = ["model-tools", "Settler", "SAT-Frontier"]


def build_proofpack(
    run_id: str,
    *,
    output_root: str | Path | None = None,
    keyring: Any = None,
    expected_head: str | None = None,
    expected_count: int | None = None,
    tenant_context: TenantContext | None = None,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a canonical SHA-256-sealed read-only Proofpack v2 for downstream verification."""
    audit = generate_audit_report(
        run_id,
        output_root=output_root,
        keyring=keyring,
        expected_head=expected_head,
        expected_count=expected_count,
        tenant_context=tenant_context,
        fmt="json",
    )
    core: dict[str, Any] = {
        "schema_version": PROOFPACK_SCHEMA_VERSION,
        "proofpack_type": PROOFPACK_TYPE,
        "run_id": audit["run_id"],
        "status": audit["status"],
        "promoted": audit["promoted"],
        "review_required": True,
        "financial_write": False,
        "executed": False,
        "coverage_automatic_decision": 0.0,
        "provenance_status": audit["provenance_status"],
        "anchor_verified": audit["anchor_verified"],
        "dataset_sha256": audit["dataset_sha256"],
        "test_sha256": audit["test_sha256"],
        "ledger_head": audit["ledger_head"],
        "ledger_count": audit["ledger_count"],
        "evaluation_summary": audit["evaluation_summary"],
        "gates": audit["gates"],
        "missing_evidence": audit["missing_evidence"],
        "reasons": audit["reasons"],
        "downstream_contract": {
            "consumers": list(_DOWNSTREAM_CONSUMERS),
            "mode": "read_only_advisory",
            "autonomous_execution_allowed": False,
            "requires_human_review": True,
        },
    }
    pack: dict[str, Any] = {**core, "proofpack_sha256": digest(core)}

    if out_path is not None:
        raw_out = str(out_path)
        if not raw_out or any(ord(ch) < 32 for ch in raw_out):
            raise ValueError("out_path must be a clean non-empty path")
        if any(part == ".." for part in re.split(r"[/\\]", raw_out)):
            raise ValueError("out_path contains path traversal")
        dest = Path(out_path).resolve()
        _write_json_atomic(dest, pack)
        return {**pack, "saved_path": str(dest)}

    return pack


def verify_proofpack(
    proofpack_or_path: dict[str, Any] | str | Path,
    *,
    require_promoted: bool = False,
) -> dict[str, Any]:
    """Verify a Proofpack v2 canonical digest, safety invariants, and optional promotion status."""
    if isinstance(proofpack_or_path, (str, Path)):
        pack_path = Path(proofpack_or_path)
        if not pack_path.is_file():
            raise FileNotFoundError(f"proofpack file not found: {proofpack_or_path}")
        try:
            data = json.loads(pack_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "status": "INVALID",
                "verified": False,
                "run_id": None,
                "promoted": False,
                "proofpack_sha256": None,
                "errors": [f"unreadable or invalid JSON proofpack: {exc}"],
                "review_required": True,
            }
    elif isinstance(proofpack_or_path, dict):
        data = proofpack_or_path
    else:
        raise TypeError("proofpack_or_path must be a dict or path")

    if not isinstance(data, dict):
        return {
            "status": "INVALID",
            "verified": False,
            "run_id": None,
            "promoted": False,
            "proofpack_sha256": None,
            "errors": ["proofpack must be a JSON object"],
            "review_required": True,
        }

    errors: list[str] = []
    if data.get("schema_version") != PROOFPACK_SCHEMA_VERSION:
        errors.append(f"schema_version must equal {PROOFPACK_SCHEMA_VERSION!r}")
    if data.get("proofpack_type") != PROOFPACK_TYPE:
        errors.append(f"proofpack_type must equal {PROOFPACK_TYPE!r}")
    if data.get("review_required") is not True:
        errors.append("safety invariant violation: review_required must be True")
    if data.get("financial_write") is not False:
        errors.append("safety invariant violation: financial_write must be False")
    if data.get("executed") is not False:
        errors.append("safety invariant violation: executed must be False")
    cov = data.get("coverage_automatic_decision")
    if isinstance(cov, bool) or cov != 0.0:
        errors.append("safety invariant violation: coverage_automatic_decision must equal 0.0")

    dc = data.get("downstream_contract")
    if not isinstance(dc, dict):
        errors.append("downstream_contract must be an object")
    else:
        if dc.get("mode") != "read_only_advisory":
            errors.append("downstream_contract.mode must be 'read_only_advisory'")
        if dc.get("autonomous_execution_allowed") is not False:
            errors.append("downstream_contract.autonomous_execution_allowed must be False")
        if dc.get("requires_human_review") is not True:
            errors.append("downstream_contract.requires_human_review must be True")

    core = {k: v for k, v in data.items() if k not in ("proofpack_sha256", "saved_path")}
    expected_sha = digest(core)
    actual_sha = data.get("proofpack_sha256")
    if not isinstance(actual_sha, str) or actual_sha != expected_sha:
        errors.append("proofpack_sha256 mismatch with canonical payload")

    if require_promoted:
        if data.get("promoted") is not True:
            errors.append("require_promoted failed: promoted is not True")
        if data.get("status") != "ALLOWED_REVIEW_REQUIRED":
            errors.append("require_promoted failed: status must be 'ALLOWED_REVIEW_REQUIRED'")
        if data.get("provenance_status") != "PROVENANCE_VERIFIED":
            errors.append("require_promoted failed: provenance_status must be 'PROVENANCE_VERIFIED'")

    ok = not errors
    raw_run_id = data.get("run_id")
    return {
        "status": "VERIFIED" if ok else "INVALID",
        "verified": ok,
        "run_id": raw_run_id if isinstance(raw_run_id, str) else None,
        "promoted": bool(data.get("promoted") is True),
        "proofpack_sha256": actual_sha if isinstance(actual_sha, str) else None,
        "errors": sorted(errors),
        "review_required": True,
    }
