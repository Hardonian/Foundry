"""Frozen read-only Proofpack v2 interchange contract for downstream audit consumers."""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import signing
from .canonical import canonical, digest
from .refinery_attestation import parse_keyring
from .refinery_runs import _write_json_atomic, generate_audit_report, inspect_run
from .tenancy import TenantContext

PROOFPACK_SCHEMA_VERSION = "2.0"
PROOFPACK_TYPE = "hardonian.foundry.proofpack.v2"
_DOWNSTREAM_CONSUMERS = ["model-tools", "Settler", "SAT-Frontier"]


def _normalize_consumers(consumers: Sequence[str] | None) -> list[str]:
    if consumers is None:
        return list(_DOWNSTREAM_CONSUMERS)
    if isinstance(consumers, (str, bytes)) or not isinstance(consumers, Sequence) or not consumers:
        raise ValueError("consumers must be a non-empty sequence of non-empty strings")
    cleaned: list[str] = []
    for item in consumers:
        if not isinstance(item, str) or not item.strip() or item.strip() != item or len(item) > 64:
            raise ValueError(f"invalid consumer identifier: {item!r}")
        if item not in cleaned:
            cleaned.append(item)
    return cleaned


def _extract_cryptographic_receipt(
    run_id: str,
    *,
    output_root: str | Path | None,
    expected_head: str | None,
    expected_count: int | None,
    tenant_context: TenantContext | None,
    ledger_head: Any,
    ledger_count: Any,
) -> dict[str, Any] | None:
    insp = inspect_run(
        run_id,
        output_root=output_root,
        expected_head=expected_head,
        expected_count=expected_count,
        tenant_context=tenant_context,
        _include_snapshot=True,
    )
    if insp.get("provenance_status") != "PROVENANCE_VERIFIED":
        return None
    raw_rec = insp.get("run_record")
    raw_entries = insp.get("ledger_entries")
    if not isinstance(raw_rec, dict) or not isinstance(raw_entries, list):
        return None
    run_sha256 = digest(raw_rec)
    sigs: list[dict[str, Any]] = []
    for entry in raw_entries:
        rec = entry.get("record") if isinstance(entry, dict) else None
        if (
            isinstance(rec, dict)
            and rec.get("type") == "promotion_signature"
            and rec.get("run_id") == run_id
            and rec.get("version") == "2"
            and rec.get("run_sha256") == run_sha256
        ):
            sigs.append({
                "version": "2",
                "entry_seq": entry.get("seq"),
                "entry_hash": entry.get("entry_hash"),
                "signed_head": rec.get("signed_head"),
                "created_at": rec.get("created_at"),
                "run_sha256": run_sha256,
                "public_key_hex": rec.get("public_key_hex"),
                "signature_hex": rec.get("signature_hex"),
                "signing_backend": rec.get("signing_backend"),
            })
    return {
        "ledger_head": ledger_head,
        "ledger_count": ledger_count,
        "run_sha256": run_sha256,
        "signature_count": len(sigs),
        "signatures": sigs,
    }


def _resolve_keyring_policy(keyring: Any) -> dict[str, Any]:
    if isinstance(keyring, (str, Path)):
        raw = json.loads(Path(keyring).read_text(encoding="utf-8"))
        return parse_keyring(raw)
    return parse_keyring(keyring)


def build_proofpack(
    run_id: str,
    *,
    output_root: str | Path | None = None,
    keyring: Any = None,
    expected_head: str | None = None,
    expected_count: int | None = None,
    tenant_context: TenantContext | None = None,
    out_path: str | Path | None = None,
    consumers: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a canonical SHA-256-sealed read-only Proofpack v2 with self-contained signature receipt."""
    consumer_list = _normalize_consumers(consumers)
    audit = generate_audit_report(
        run_id,
        output_root=output_root,
        keyring=keyring,
        expected_head=expected_head,
        expected_count=expected_count,
        tenant_context=tenant_context,
        fmt="json",
    )
    receipt = _extract_cryptographic_receipt(
        audit["run_id"],
        output_root=output_root,
        expected_head=expected_head,
        expected_count=expected_count,
        tenant_context=tenant_context,
        ledger_head=audit["ledger_head"],
        ledger_count=audit["ledger_count"],
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
            "consumers": consumer_list,
            "mode": "read_only_advisory",
            "autonomous_execution_allowed": False,
            "requires_human_review": True,
        },
    }
    if receipt is not None:
        core["cryptographic_receipt"] = receipt

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
    keyring: Any = None,
    expected_head: str | None = None,
) -> dict[str, Any]:
    """Verify a Proofpack v2 canonical digest, safety invariants, anchor head, and Ed25519 receipt."""
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

    if expected_head is not None and data.get("ledger_head") != expected_head:
        errors.append(f"expected_head mismatch: expected {expected_head}, got {data.get('ledger_head')}")

    if require_promoted:
        if data.get("promoted") is not True:
            errors.append("require_promoted failed: promoted is not True")
        if data.get("status") != "ALLOWED_REVIEW_REQUIRED":
            errors.append("require_promoted failed: status must be 'ALLOWED_REVIEW_REQUIRED'")
        if data.get("provenance_status") != "PROVENANCE_VERIFIED":
            errors.append("require_promoted failed: provenance_status must be 'PROVENANCE_VERIFIED'")

    signature_verified = False
    verified_signers: list[str] = []
    if keyring is not None:
        if not signing.backend_available():
            errors.append("signing backend unavailable: install hardonian-foundry[signing] to verify signatures")
        else:
            try:
                policy = _resolve_keyring_policy(keyring)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                policy = None
                errors.append(f"invalid keyring for offline proofpack verification: {exc}")

            if policy is not None:
                receipt = data.get("cryptographic_receipt")
                if not isinstance(receipt, dict) or not isinstance(receipt.get("signatures"), list) or not receipt["signatures"]:
                    errors.append("cryptographic_receipt missing embedded signatures")
                else:
                    if receipt.get("ledger_head") != data.get("ledger_head"):
                        errors.append("cryptographic_receipt ledger_head mismatch")
                    key_by_hex = {k["public_key_hex"]: k for k in policy["keys"]}
                    verified_roles: set[str] = set()
                    for sig in receipt["signatures"]:
                        if not isinstance(sig, dict):
                            continue
                        pub = sig.get("public_key_hex")
                        if not isinstance(pub, str) or pub not in key_by_hex:
                            continue
                        kmeta = key_by_hex[pub]
                        if kmeta.get("revoked"):
                            errors.append(f"embedded signature key {pub[:12]} has been revoked")
                            continue
                        if sig.get("run_sha256") != receipt.get("run_sha256"):
                            errors.append("embedded signature run_sha256 mismatch")
                            continue
                        msg = canonical({
                            "version": sig.get("version", "2"),
                            "run_id": data.get("run_id"),
                            "signed_head": sig.get("signed_head"),
                            "created_at": sig.get("created_at"),
                            "run_sha256": sig.get("run_sha256"),
                        }).encode("utf-8")
                        if signing.verify(msg, sig.get("signature_hex"), [pub]) and pub not in verified_signers:
                            verified_signers.append(pub)
                            if kmeta.get("role"):
                                verified_roles.add(kmeta["role"])

                    req_sigs = int(policy["required_signatures"])
                    missing_roles = sorted(set(policy["required_roles"]) - verified_roles)
                    if len(verified_signers) < req_sigs:
                        errors.append(
                            f"offline signature quorum not met: verified {len(verified_signers)}, required {req_sigs}"
                        )
                    if missing_roles:
                        errors.append(f"offline signature missing required roles: {', '.join(missing_roles)}")
                    if len(verified_signers) >= req_sigs and not missing_roles:
                        signature_verified = True

    ok = not errors
    raw_run_id = data.get("run_id")
    out: dict[str, Any] = {
        "status": "VERIFIED" if ok else "INVALID",
        "verified": ok,
        "run_id": raw_run_id if isinstance(raw_run_id, str) else None,
        "promoted": bool(data.get("promoted") is True),
        "proofpack_sha256": actual_sha if isinstance(actual_sha, str) else None,
        "errors": sorted(errors),
        "review_required": True,
    }
    if keyring is not None:
        out["signature_verified"] = signature_verified
        out["verified_signers"] = verified_signers
    return out
