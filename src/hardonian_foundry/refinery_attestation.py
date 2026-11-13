"""Refinery attestations: ledger attachments, Ed25519 signing, keyrings, and external anchors."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import signing
from .canonical import canonical, digest
from .evidence import _append_with_builder as evidence_append_with_builder
from .evidence import append as evidence_append
from .evidence import verify_entries as evidence_verify_entries
from .licensing import evaluate as evaluate_license
from .refinery_runs import (
    _verify_seq1_metadata,
    _write_json_atomic,
    inspect_run,
)
from .refinery_validation import (
    DEFAULT_RUNS_DIR,
    facade_attr,
    resolve_ledger_path,
    resolve_run_dir,
    validate_run_id,
)
from .types import AnchorRecord


def _iter_ledger_entries(path: str | Path, run_id: str, entry_type: str) -> list[dict[str, Any]]:
    """Collect ledger entries of a given type bound to run_id. Chain must be verified elsewhere."""
    out: list[dict[str, Any]] = []
    if not path or not os.path.isfile(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                record = entry.get("record", {})
                if isinstance(record, dict) and record.get("type") == entry_type and record.get("run_id") == run_id:
                    out.append(entry)
    except (OSError, json.JSONDecodeError):
        return []
    return out


def _find_ledger_entry(path: str | Path, run_id: str, entry_type: str) -> dict[str, Any] | None:
    entries = _iter_ledger_entries(path, run_id, entry_type)
    return entries[-1] if entries else None


def extract_verified_license_attachment(
    entries: Sequence[dict[str, Any]],
    run_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool]:
    """Find the latest model_licensing_manifest entry for run_id and verify its self-digest."""
    attachments = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("record"), dict)
        and entry["record"].get("type") == "model_licensing_manifest"
        and entry["record"].get("run_id") == run_id
    ]
    if not attachments:
        return None, None, False
    attached = attachments[-1]
    record = attached.get("record", {})
    stored = record.get("manifest")
    if isinstance(stored, dict) and record.get("manifest_sha256") == digest(stored):
        return attached, stored, True
    return attached, None, False


def _evaluate_license_compat(
    manifest: dict[str, Any] | None,
    *,
    allowed_data_origins: Sequence[str] | None = None,
) -> dict[str, Any]:
    eval_fn = facade_attr("evaluate_license", evaluate_license)
    if allowed_data_origins is not None:
        try:
            return eval_fn(manifest, allowed_data_origins=tuple(allowed_data_origins))
        except TypeError:
            pass
    return eval_fn(manifest)


def parse_keyring(data: Any) -> dict[str, Any]:
    """Parse a keyring list or policy object into normalized M-of-N key policy.

    Supports backward-compatible hex lists / key-dict lists as well as policy dicts:
      {
        "trusted_keys": [{"public_key_hex": "...", "key_id"?: "...", "role"?: "...", "revoked_at"?: "..."}],
        "required_signatures": 1,
        "required_roles": ["security"]
      }
    """
    signing_mod = facade_attr("signing", signing)

    # Extract raw entries and policy controls from list or dict
    if isinstance(data, dict):
        if "trusted_keys" not in data or not isinstance(data["trusted_keys"], list):
            raise TypeError("keyring must be an object with a trusted_keys list")
        allowed_top = {"trusted_keys", "required_signatures", "required_roles"}
        unknown_top = set(data) - allowed_top
        if unknown_top:
            raise ValueError(f"unknown keyring fields: {sorted(unknown_top)}")
        raw_entries = data["trusted_keys"]
        req_sigs = data.get("required_signatures", 1)
        req_roles = data.get("required_roles", [])
    elif isinstance(data, list):
        raw_entries = data
        req_sigs = 1
        req_roles = []
    else:
        raise TypeError("keyring must be a list of keys or a policy object")

    if isinstance(req_sigs, bool) or not isinstance(req_sigs, int) or req_sigs < 1:
        raise ValueError("keyring required_signatures must be an integer >= 1")
    if not isinstance(req_roles, list) or any(
        not isinstance(r, str) or not r.strip() or r.strip() != r for r in req_roles
    ):
        raise ValueError("keyring required_roles must be a list of clean nonempty strings")

    # Validate via signing.parse_keyring when present post-merge
    if hasattr(signing_mod, "parse_keyring") and isinstance(data, dict):
        signing_mod.parse_keyring(data)

    key_records: dict[str, dict[str, Any]] = {}
    for entry in raw_entries:
        if isinstance(entry, str):
            hex_key = signing_mod.normalize_trusted_keys([entry])[0]
            meta: dict[str, Any] = {
                "public_key_hex": hex_key,
                "key_id": None,
                "label": None,
                "role": None,
                "revoked_at": None,
                "revoked": False,
            }
        elif isinstance(entry, dict):
            allowed_entry = {"public_key_hex", "key", "public_key", "key_id", "label", "role", "revoked_at"}
            unknown_entry = set(entry) - allowed_entry
            if unknown_entry:
                raise ValueError(f"unknown keyring entry fields: {sorted(unknown_entry)}")
            hex_key = signing_mod.normalize_trusted_keys([entry])[0]
            role = entry.get("role")
            if role is not None and (not isinstance(role, str) or not role.strip() or role.strip() != role):
                raise ValueError("keyring entry role must be a clean nonempty string")
            key_id = entry.get("key_id")
            if key_id is not None and (not isinstance(key_id, str) or not key_id.strip()):
                raise ValueError("keyring entry key_id must be a nonempty string")
            label = entry.get("label")
            if label is not None and not isinstance(label, str):
                raise ValueError("keyring entry label must be a string")
            revoked_at = entry.get("revoked_at")
            if revoked_at is not None and (not isinstance(revoked_at, str) or not revoked_at.strip()):
                raise ValueError("keyring entry revoked_at must be a nonempty timestamp string")
            meta = {
                "public_key_hex": hex_key,
                "key_id": key_id,
                "label": label,
                "role": role,
                "revoked_at": revoked_at,
                "revoked": revoked_at is not None,
            }
        else:
            raise TypeError("trusted key entry must be a hex string or object")

        if hex_key in key_records:
            existing = key_records[hex_key]
            existing["revoked"] = existing["revoked"] or meta["revoked"]
            existing["revoked_at"] = existing["revoked_at"] or meta["revoked_at"]
            existing["role"] = existing["role"] or meta["role"]
            existing["key_id"] = existing["key_id"] or meta["key_id"]
            existing["label"] = existing["label"] or meta["label"]
        else:
            key_records[hex_key] = meta

    return {
        "keys": list(key_records.values()),
        "trusted_keys": list(key_records.keys()),
        "required_signatures": req_sigs,
        "required_roles": list(dict.fromkeys(req_roles)),
    }


def load_keyring_policy(path: str | Path) -> dict[str, Any]:
    """Load a trusted keyring JSON file and return its normalized M-of-N policy object."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("trusted_keys"), list):
        raise TypeError("keyring must be an object with a trusted_keys list")
    return parse_keyring(data)


def load_keyring(path: str | Path) -> list[str] | dict[str, Any]:
    """Load a trusted keyring JSON file.

    Returns a list of public key hex strings for simple keyrings (preserving legacy
    API compatibility) or a policy dict preserving M-of-N quorum, role, and revocation
    metadata for evaluate_promotion().
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("trusted_keys"), list):
        raise TypeError("keyring must be an object with a trusted_keys list")
    policy = parse_keyring(data)
    has_policy_extras = (
        "required_signatures" in data
        or "required_roles" in data
        or any(
            isinstance(item, dict) and any(k in item for k in ("role", "revoked_at", "key_id", "label"))
            for item in data["trusted_keys"]
        )
    )
    if has_policy_extras:
        return {
            "trusted_keys": [
                {
                    k: m[k]
                    for k in ("public_key_hex", "key_id", "role", "revoked_at", "label")
                    if m.get(k) is not None
                }
                for m in policy["keys"]
            ],
            "required_signatures": policy["required_signatures"],
            "required_roles": list(policy["required_roles"]),
        }
    return list(policy["trusted_keys"])


def attach_license(
    run_id: str,
    *,
    manifest: dict[str, Any],
    output_root: str | Path | None = None,
    allowed_data_origins: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Bind a licensing decision (from the SPDX allowlist validator) into the evidence ledger."""
    clean_id = validate_run_id(run_id)
    root_dir = Path(output_root or DEFAULT_RUNS_DIR).resolve()
    run_dir = resolve_run_dir(root_dir, clean_id)
    run_file = run_dir / "run.json"
    if not run_file.is_file():
        raise FileNotFoundError(f"run not found: {clean_id}")
    with open(run_file, "r", encoding="utf-8") as handle:
        run_record = json.load(handle)

    raw_ledger = run_record.get("provenance", {}).get("evidence_ledger_path")
    ledger_file = resolve_ledger_path(root_dir, raw_ledger, allow_legacy_outside_fallback=False)
    if not ledger_file.is_file():
        raise ValueError("evidence ledger not found for run")
    if _find_ledger_entry(str(ledger_file), clean_id, "model_licensing_manifest"):
        raise ValueError(f"license manifest already attached for {clean_id}")

    lic_result = _evaluate_license_compat(manifest, allowed_data_origins=allowed_data_origins)
    payload: dict[str, Any] = {
        "data_origin": "synthetic",
        "type": "model_licensing_manifest",
        "run_id": clean_id,
        "manifest_sha256": digest(manifest),
        "decision": lic_result["decision"],
        "reasons": lic_result["reasons"],
        "manifest": manifest,
    }
    if "tenant_id" in run_record:
        payload["tenant_id"] = run_record["tenant_id"]

    append_fn = facade_attr("evidence_append", evidence_append)
    entry = append_fn(str(ledger_file), payload, key=f"{clean_id}#license")
    return {
        "run_id": clean_id,
        "decision": lic_result["decision"],
        "reasons": lic_result["reasons"],
        "manifest_sha256": entry["record"]["manifest_sha256"],
        "entry_hash": entry["entry_hash"],
        "entry_seq": entry["seq"],
    }


def sign_run(
    run_id: str,
    *,
    private_key_path: str | Path,
    output_root: str | Path | None = None,
    cosign: bool = False,
    allowed_data_origins: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Sign the current ledger head, binding an Ed25519 promotion attestation to the run.

    Verifies run.json integrity against the seq-1 ledger record BEFORE signing so
    pre-sign tampering of thresholds, model_manifest, or evaluation metrics fails closed.
    """
    clean_id = validate_run_id(run_id)
    root_dir = Path(output_root or DEFAULT_RUNS_DIR).resolve()
    run_dir = resolve_run_dir(root_dir, clean_id)
    run_file = run_dir / "run.json"
    if not run_file.is_file():
        raise FileNotFoundError(f"run not found: {clean_id}")
    with open(run_file, "r", encoding="utf-8") as handle:
        run_record = json.load(handle)
    if not isinstance(run_record, dict) or not isinstance(run_record.get("provenance"), dict):
        raise TypeError("run metadata requires provenance object")

    # Ensure run_record is canonical before entering ledger lock
    run_sha256 = digest(run_record)

    raw_ledger = run_record.get("provenance", {}).get("evidence_ledger_path")
    ledger_file = resolve_ledger_path(root_dir, raw_ledger, allow_legacy_outside_fallback=False)
    if not ledger_file.is_file():
        raise ValueError("evidence ledger not found for run")

    signing_mod = facade_attr("signing", signing)
    expected_seq = run_record["provenance"].get("evidence_seq")
    expected_hash = run_record["provenance"].get("evidence_entry_hash")

    idem_key = f"{clean_id}#sig"
    if cosign:
        pub_preview = signing_mod.public_key_hex(str(private_key_path))
        idem_key = f"{clean_id}#sig#{pub_preview[:16]}"

    def build_signature(entries: list[dict[str, Any]], signed_head: str) -> dict[str, Any]:
        # 1. Pre-sign integrity check against run's initial ledger entry
        matched = next(
            (
                entry
                for entry in entries
                if entry.get("seq") == expected_seq and entry.get("entry_hash") == expected_hash
            ),
            None,
        )
        if matched is None or not _verify_seq1_metadata(clean_id, run_record, matched):
            raise ValueError(
                f"run.json integrity verification against ledger failed before signing for {clean_id}"
            )

        # 2. Enforce single-sign default or distinct-key v2 cosign
        existing_sigs = [
            entry
            for entry in entries
            if isinstance(entry.get("record"), dict)
            and entry["record"].get("type") == "promotion_signature"
            and entry["record"].get("run_id") == clean_id
        ]
        if existing_sigs and not cosign:
            raise ValueError(
                f"run already signed: {clean_id}; recreate the run, attach an allowed license, and sign the new run"
            )

        # 3. License validation before touching private key material
        manifest = run_record.get("model_manifest")
        if manifest is None:
            _, attached_manifest, valid_att = extract_verified_license_attachment(entries, clean_id)
            if valid_att:
                manifest = attached_manifest
        if _evaluate_license_compat(manifest, allowed_data_origins=allowed_data_origins)["decision"] != "ALLOWED":
            raise ValueError(
                "signing requires an allowed license first; attach an allowed license or recreate the run"
            )

        public_hex = signing_mod.public_key_hex(str(private_key_path))
        if cosign and existing_sigs:
            for sig_entry in existing_sigs:
                srec = sig_entry["record"]
                if srec.get("version") != "2" or srec.get("run_sha256") != run_sha256:
                    raise ValueError(
                        "existing signature does not match current run; recreate the run, attach an allowed license, and sign the new run"
                    )
                if srec.get("public_key_hex") == public_hex:
                    raise ValueError(f"key has already signed {clean_id}")

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        message = canonical({
            "version": "2",
            "run_id": clean_id,
            "signed_head": signed_head,
            "created_at": timestamp,
            "run_sha256": run_sha256,
        }).encode("utf-8")
        signature_hex = signing_mod.sign(message, str(private_key_path))
        sig_record: dict[str, Any] = {
            "data_origin": "synthetic",
            "type": "promotion_signature",
            "version": "2",
            "run_id": clean_id,
            "signed_head": signed_head,
            "run_sha256": run_sha256,
            "public_key_hex": public_hex,
            "signature_hex": signature_hex,
            "created_at": timestamp,
            "signing_backend": signing_mod.SIGNING_BACKEND,
        }
        if "tenant_id" in run_record:
            sig_record["tenant_id"] = run_record["tenant_id"]
        return sig_record

    builder_fn = facade_attr("evidence_append_with_builder", evidence_append_with_builder)
    try:
        entry = builder_fn(str(ledger_file), build_signature, key=idem_key)
    except ValueError as exc:
        if str(exc) == "duplicate idempotency key":
            raise ValueError(
                "run already signed or signature key occupied; recreate the run, attach an allowed license, and sign the new run"
            ) from exc
        raise
    return {
        "run_id": clean_id,
        "signed_head": entry["record"]["signed_head"],
        "public_key_hex": entry["record"]["public_key_hex"],
        "entry_hash": entry["entry_hash"],
        "entry_seq": entry["seq"],
    }


def anchor_run(
    run_id: str | None = None,
    *,
    output_root: str | Path | None = None,
    anchor_path: str | Path | None = None,
) -> AnchorRecord:
    """Emit (and optionally persist) an external ledger checkpoint anchor to detect rollback/truncation."""
    root_dir = Path(output_root or DEFAULT_RUNS_DIR).resolve()
    if run_id is not None:
        clean_id = validate_run_id(run_id)
        insp = inspect_run(clean_id, output_root=str(root_dir), _include_snapshot=True)
        if insp["provenance_status"] != "PROVENANCE_VERIFIED":
            raise ValueError(f"cannot anchor unverified run {clean_id}: {insp['provenance_status']}")
        resolved_lp = insp["resolved_ledger_path"]
        assert resolved_lp is not None
        ledger_file = Path(resolved_lp)
        ledger_report = insp["ledger_report"]
    else:
        clean_id = None
        ledger_file = (root_dir / "evidence.jsonl").resolve()
        if not ledger_file.is_relative_to(root_dir):
            raise ValueError("evidence ledger symlink escapes output_root")
        if not ledger_file.is_file():
            raise FileNotFoundError(f"evidence ledger not found at {ledger_file}")
        entries = [
            json.loads(line)
            for line in ledger_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        verify_fn = facade_attr("evidence_verify_entries", evidence_verify_entries)
        ledger_report = verify_fn(entries)
        if ledger_report["status"] != "OK":
            raise ValueError(f"cannot anchor tampered ledger: {ledger_report['status']}")

    assert ledger_report is not None
    anchor: AnchorRecord = {
        "schema_version": "1.0",
        "data_origin": "synthetic",
        "run_id": clean_id,
        "head": ledger_report["head"],
        "count": ledger_report["count"],
        "anchored_at": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "ledger_sha256": hashlib.sha256(ledger_file.read_bytes()).hexdigest(),
    }
    if anchor_path is not None:
        dest = Path(anchor_path).resolve()
        _write_json_atomic(dest, dict(anchor))
        anchor["anchor_path"] = str(dest)
    return anchor
