"""Fail-closed, SHA-256-verified local model registry for Foundry runs."""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .canonical import digest
from .refinery_gates import evaluate_promotion
from .refinery_runs import _write_json_atomic, inspect_run, list_runs
from .refinery_validation import DEFAULT_RUNS_DIR, validate_run_id
from .tenancy import DENY_REASON, TenantContext

_REGISTRY_SCHEMA_VERSION = "1.0"
_MAX_NOTES_LEN = 1024


def _validate_tenant_context(tenant_context: Any) -> None:
    if tenant_context is not None and not isinstance(tenant_context, TenantContext):
        raise ValueError("tenant_context must be a TenantContext instance")


def _validate_notes(notes: Any) -> str | None:
    if notes is None:
        return None
    if (
        not isinstance(notes, str)
        or len(notes) > _MAX_NOTES_LEN
        or any(ord(ch) < 32 and ch not in "\n\r\t" for ch in notes)
    ):
        raise ValueError(f"notes must be a clean string <= {_MAX_NOTES_LEN} chars")
    return notes


def resolve_registry_path(
    output_root: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> Path:
    """Resolve the local registry.json path safely without traversal or control characters."""
    if registry_path is not None:
        raw = str(registry_path)
        if not raw or any(ord(ch) < 32 for ch in raw):
            raise ValueError("registry_path must be a non-empty clean path")
        if any(part == ".." for part in re.split(r"[/\\]", raw)):
            raise ValueError("registry_path contains path traversal")
        return Path(registry_path).resolve()

    raw_root = str(output_root) if output_root is not None else DEFAULT_RUNS_DIR
    if not raw_root or any(ord(ch) < 32 for ch in raw_root):
        raise ValueError("output_root must be a clean path")
    if any(part == ".." for part in re.split(r"[/\\]", raw_root)):
        raise ValueError("output_root contains path traversal")

    root_dir = Path(raw_root).resolve()
    candidate = (root_dir / "registry.json").resolve()
    if not candidate.is_relative_to(root_dir):
        raise ValueError("registry_path escapes output_root")
    return candidate


def _empty_registry() -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    return {
        "schema_version": _REGISTRY_SCHEMA_VERSION,
        "models": models,
        "registry_sha256": digest({"schema_version": _REGISTRY_SCHEMA_VERSION, "models": models}),
    }


def _load_verified_registry_data(reg_file: Path) -> dict[str, Any]:
    if not reg_file.is_file():
        return _empty_registry()
    with open(reg_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("registry file must contain a JSON object")
    schema_version = data.get("schema_version")
    models = data.get("models")
    if not isinstance(schema_version, str) or not isinstance(models, list):
        raise ValueError("registry file missing valid schema_version or models list")
    expected_sha = digest({"schema_version": schema_version, "models": models})
    if data.get("registry_sha256") != expected_sha:
        raise ValueError("registry integrity check failed: registry_sha256 mismatch")
    return data


def register_model(
    run_id: str,
    *,
    model_id: str | None = None,
    output_root: str | Path | None = None,
    registry_path: str | Path | None = None,
    keyring: Any = None,
    expected_head: str | None = None,
    expected_count: int | None = None,
    tenant_context: TenantContext | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Evaluate promotion gates and, only if ALLOWED_REVIEW_REQUIRED, register the model atomically."""
    _validate_tenant_context(tenant_context)
    clean_run_id = validate_run_id(run_id)
    clean_model_id = validate_run_id(model_id) if model_id is not None else clean_run_id
    clean_notes = _validate_notes(notes)

    promo = evaluate_promotion(
        clean_run_id,
        output_root=output_root,
        keyring=keyring,
        expected_head=expected_head,
        expected_count=expected_count,
        tenant_context=tenant_context,
    )
    if not promo["promoted"]:
        return {
            **promo,
            "registered": False,
            "registry_entry": None,
        }

    reg_file = resolve_registry_path(output_root=output_root, registry_path=registry_path)
    insp = inspect_run(
        clean_run_id,
        output_root=output_root,
        expected_head=expected_head,
        expected_count=expected_count,
        tenant_context=tenant_context,
        _include_snapshot=True,
    )
    raw_record = insp.get("run_record")
    run_record: dict[str, Any] = raw_record if isinstance(raw_record, dict) else {}
    raw_eval = run_record.get("evaluation")
    eval_obj: dict[str, Any] = raw_eval if isinstance(raw_eval, dict) else {}
    raw_task = eval_obj.get("task")
    task_block: dict[str, Any] = raw_task if isinstance(raw_task, dict) else {}
    raw_report = insp.get("ledger_report")
    ledger_report: dict[str, Any] = raw_report if isinstance(raw_report, dict) else {}

    model_identity = task_block.get("model_identity") or run_record.get("model_manifest")
    registered_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    registry_entry: dict[str, Any] = {
        "model_id": clean_model_id,
        "run_id": clean_run_id,
        "registered_at": registered_at,
        "status": "REGISTERED_REVIEW_REQUIRED",
        "review_required": True,
        "task_id": eval_obj.get("task_id"),
        "dataset_sha256": run_record.get("dataset_sha256"),
        "test_sha256": run_record.get("test_sha256"),
        "ledger_head": ledger_report.get("head"),
        "f1": eval_obj.get("f1"),
        "model_identity": model_identity,
        "notes": clean_notes,
    }
    if tenant_context is not None:
        registry_entry["tenant_id"] = tenant_context.tenant_id
    elif run_record.get("tenant_id") is not None:
        registry_entry["tenant_id"] = run_record.get("tenant_id")

    existing = _load_verified_registry_data(reg_file)
    models_by_id: dict[str, dict[str, Any]] = {}
    for entry in existing["models"]:
        if isinstance(entry, dict) and isinstance(entry.get("model_id"), str):
            if (
                entry["model_id"] == clean_model_id
                and tenant_context is not None
                and entry.get("tenant_id") != tenant_context.tenant_id
            ):
                raise PermissionError(DENY_REASON)
            models_by_id[entry["model_id"]] = entry

    models_by_id[clean_model_id] = registry_entry
    sorted_models = [models_by_id[k] for k in sorted(models_by_id)]
    payload_core = {
        "schema_version": _REGISTRY_SCHEMA_VERSION,
        "models": sorted_models,
    }
    registry_sha256 = digest(payload_core)
    _write_json_atomic(reg_file, {**payload_core, "registry_sha256": registry_sha256})

    return {
        **promo,
        "registered": True,
        "registry_entry": registry_entry,
        "registry_path": str(reg_file),
        "registry_sha256": registry_sha256,
    }


def list_models(
    *,
    output_root: str | Path | None = None,
    registry_path: str | Path | None = None,
    tenant_context: TenantContext | None = None,
    include_runs: bool = True,
) -> dict[str, Any]:
    """List verified registered models (tenant-scoped when TenantContext is provided)."""
    _validate_tenant_context(tenant_context)
    reg_file = resolve_registry_path(output_root=output_root, registry_path=registry_path)
    data = _load_verified_registry_data(reg_file)
    all_models: list[dict[str, Any]] = [m for m in data["models"] if isinstance(m, dict)]

    if tenant_context is not None:
        scoped_models = [m for m in all_models if m.get("tenant_id") == tenant_context.tenant_id]
    else:
        scoped_models = all_models

    runs = list_runs(output_root=output_root, tenant_context=tenant_context) if (include_runs and output_root is not None) else []
    registry_status = "VERIFIED" if reg_file.is_file() else "EMPTY"
    message = (
        f"{len(scoped_models)} model(s) registered (review required)."
        if scoped_models
        else "No model registry is configured."
    )
    return {
        "models": scoped_models,
        "runs": runs,
        "count": len(scoped_models),
        "registry_status": registry_status,
        "message": message,
    }


def get_model(
    model_id: str,
    *,
    output_root: str | Path | None = None,
    registry_path: str | Path | None = None,
    tenant_context: TenantContext | None = None,
) -> dict[str, Any]:
    """Fetch a registered model entry by model_id or run_id after verifying registry integrity."""
    _validate_tenant_context(tenant_context)
    clean_id = validate_run_id(model_id)
    reg_file = resolve_registry_path(output_root=output_root, registry_path=registry_path)
    if not reg_file.is_file():
        raise KeyError(f"model not found in registry: {clean_id}")

    data = _load_verified_registry_data(reg_file)
    match: dict[str, Any] | None = None
    for entry in data["models"]:
        if isinstance(entry, dict) and (entry.get("model_id") == clean_id or entry.get("run_id") == clean_id):
            match = entry
            break

    if match is None:
        raise KeyError(f"model not found in registry: {clean_id}")
    if tenant_context is not None and match.get("tenant_id") != tenant_context.tenant_id:
        raise PermissionError(DENY_REASON)
    return match


def verify_registry(
    *,
    output_root: str | Path | None = None,
    registry_path: str | Path | None = None,
    keyring: Any = None,
    tenant_context: TenantContext | None = None,
) -> dict[str, Any]:
    """Verify registry SHA-256 digest and re-verify every registered run's provenance (and keyring gates)."""
    _validate_tenant_context(tenant_context)
    reg_file = resolve_registry_path(output_root=output_root, registry_path=registry_path)
    try:
        data = _load_verified_registry_data(reg_file)
    except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
        return {
            "status": "TAMPERED",
            "verified": False,
            "model_count": 0,
            "errors": [str(exc)],
            "review_required": True,
        }

    all_models = [m for m in data["models"] if isinstance(m, dict)]
    scoped_models = (
        [m for m in all_models if m.get("tenant_id") == tenant_context.tenant_id]
        if tenant_context is not None
        else all_models
    )

    eff_root = output_root if output_root is not None else reg_file.parent
    errors: list[str] = []
    for entry in scoped_models:
        mid = entry.get("model_id")
        rid = entry.get("run_id")
        if not isinstance(rid, str) or not rid:
            errors.append(f"model {mid!r}: missing valid run_id")
            continue
        try:
            insp = inspect_run(
                rid,
                output_root=eff_root,
                tenant_context=tenant_context,
                _include_snapshot=True,
            )
            if insp["provenance_status"] != "PROVENANCE_VERIFIED":
                errors.append(f"model {mid!r} (run {rid}): provenance status is {insp['provenance_status']}")
            run_rec = insp.get("run_record") or {}
            if run_rec.get("dataset_sha256") != entry.get("dataset_sha256"):
                errors.append(f"model {mid!r} (run {rid}): dataset_sha256 mismatch with registry")
            if run_rec.get("test_sha256") != entry.get("test_sha256"):
                errors.append(f"model {mid!r} (run {rid}): test_sha256 mismatch with registry")
            if keyring is not None:
                promo = evaluate_promotion(
                    rid,
                    output_root=eff_root,
                    keyring=keyring,
                    tenant_context=tenant_context,
                )
                if not promo["promoted"]:
                    errors.append(
                        f"model {mid!r} (run {rid}): promotion gates blocked ({', '.join(promo['missing_evidence'])})"
                    )
        except (FileNotFoundError, PermissionError, ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
            errors.append(f"model {mid!r} (run {rid}): verification failed ({exc})")

    return {
        "status": "VERIFIED" if not errors else "TAMPERED",
        "verified": not errors,
        "model_count": len(scoped_models),
        "errors": sorted(errors),
        "review_required": True,
    }
