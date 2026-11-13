"""Refinery run execution, inspection, listing, diffing, and bundle export."""
from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import tarfile
import tempfile
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

try:
    from .canonical import digest  # type: ignore[import-not-found,import-untyped]
except ImportError:  # pragma: no cover - pre-merge fallback
    from .baseline import digest

from . import tenancy
from .baseline import dataset as default_dataset
from .baseline import reconciliation
from .evidence import append as evidence_append
from .evidence import verify_entries as evidence_verify_entries
from .refinery_preflight import run_preflight
from .refinery_validation import (
    DEFAULT_RUNS_DIR,
    _code_digest,
    facade_attr,
    resolve_ledger_path,
    resolve_run_dir,
    validate_and_split_dataset,
    validate_run_id,
)
from .task_eval import (
    score_predictions,
    split_task_dataset,
    validate_spec,
    validate_task_dataset,
)
from .tenancy import DENY_REASON, TenantContext
from .types import BundleManifest, InspectResult, RunDiff, RunRecord, RunSummary

_ALLOWED_BUNDLE_FORMATS = frozenset({"dir", "zip", "tar", "tar.gz", "tgz"})


def _run_body_digest(run_record: dict[str, Any]) -> str:
    """Compute canonical digest of run_record excluding post-append ledger coordinates."""
    prov = run_record.get("provenance") or {}
    body: dict[str, Any] = {k: v for k, v in run_record.items() if k != "provenance"}
    body["provenance"] = {
        k: v for k, v in prov.items()
        if k not in ("evidence_entry_hash", "evidence_seq")
    }
    return digest(body)


def _authorize_tenant_run(ctx: TenantContext, run_record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ctx, TenantContext):
        return {"status": "DENIED", "reason": DENY_REASON, "review_required": True}
    if hasattr(tenancy, "authorize_run"):
        return tenancy.authorize_run(ctx, run_record)
    return tenancy.read(ctx, run_record)


def _scope_ledger_entries(ctx: TenantContext, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(ctx, TenantContext):
        return []
    if hasattr(tenancy, "scope_ledger_query"):
        return tenancy.scope_ledger_query(ctx, entries)
    return [
        entry for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("record"), dict)
        and tenancy.read(ctx, entry["record"])["status"] == "ALLOWED"
    ]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".foundry-json-", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def evaluate_dataset(
    records: list[dict[str, Any]] | None = None,
    *,
    seed: int = 42,
    _splits: dict[str, list[dict[str, Any]]] | None = None,
    _dataset_sha256: str | None = None,
) -> dict[str, Any]:
    """Execute reproducible held-out evaluation on the test split of a synthetic dataset."""
    data = records if records is not None else default_dataset()
    if _splits is None or _dataset_sha256 is None:
        val, splits = validate_and_split_dataset(data, seed=seed)
        if val["status"] != "VALID" or splits is None:
            raise ValueError(f"cannot evaluate invalid dataset: {'; '.join(val['errors'])}")
        dataset_sha256 = val["dataset_sha256"]
    else:
        splits = _splits
        dataset_sha256 = _dataset_sha256

    test_rows = splits["test"]

    tp = fp = fn = critical_fn = critical_total = 0
    for row in test_rows:
        pred = reconciliation(row)["category"] == "amount_mismatch"
        truth = row["expected_minor"] != row["actual_minor"]
        tp += int(pred and truth)
        fp += int(pred and not truth)
        fn += int(not pred and truth)
        critical_total += int(truth and row["critical"])
        critical_fn += int(not pred and truth and row["critical"])

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision is not None and recall is not None and precision + recall)
        else None
    )
    critical_fn_rate = (critical_fn / critical_total) if critical_total else None

    return {
        "data_origin": "synthetic",
        "mode": "deterministic_heuristic",
        "test_count": len(test_rows),
        "dataset_sha256": dataset_sha256,
        "split_seed": seed,
        "test_sha256": digest(test_rows),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "critical_false_negative_rate": critical_fn_rate,
        "coverage_automatic_decision": 0.0,
        "review_rate": 1.0,
        "cost_per_case_usd": None,
        "latency_p95_ms": None,
        "note": "Synthetic self-consistency only; no trained model or pilot accuracy.",
    }


def create_run(
    *,
    output_root: str | Path | None = None,
    dataset: list[dict[str, Any]] | None = None,
    model_manifest: dict[str, Any] | None = None,
    seed: int = 42,
    tenant_context: TenantContext | None = None,
    allowed_data_origins: Sequence[str] | None = None,
) -> RunRecord:
    """Execute the vertical Refinery workflow: validate, preflight, evaluate, and track run with evidence."""
    if tenant_context is not None and not isinstance(tenant_context, TenantContext):
        raise ValueError("tenant_context must be a TenantContext instance")

    root_dir = Path(output_root or DEFAULT_RUNS_DIR).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    data = dataset if dataset is not None else default_dataset()

    # Single dataset validation + split pass
    val_result, splits = validate_and_split_dataset(data, seed=seed)

    # Step 1: Preflight (reusing validation result)
    preflight = run_preflight(
        dataset=data,
        output_root=str(root_dir),
        model_manifest=model_manifest,
        allowed_data_origins=allowed_data_origins,
        _validation_result=val_result,
    )
    if preflight["status"] != "PASSED":
        raise ValueError(f"refinery run aborted: preflight failed ({'; '.join(preflight['errors'])})")

    # Step 2: Evaluation (reusing splits + dataset digest)
    eval_result = evaluate_dataset(
        data,
        seed=seed,
        _splits=splits,
        _dataset_sha256=val_result["dataset_sha256"],
    )

    # Step 3: Run identity & metadata (CSPRNG entropy prevents concurrent collisions)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_hash = hashlib.sha256(
        f"{timestamp}:{eval_result['test_sha256']}:{secrets.token_hex(6)}".encode()
    ).hexdigest()[:12]
    run_id = f"run-{timestamp}-{run_hash}"
    run_dir = resolve_run_dir(root_dir, run_id)
    run_dir.mkdir(mode=0o700, exist_ok=False)

    # Step 4: Construct run record base (portable relative ledger path)
    prov: dict[str, Any] = {
        "code_digest": _code_digest(),
        "runtime": "offline_cpu",
        "evidence_ledger_path": "evidence.jsonl",
        "evidence_key": run_id,
    }
    if tenant_context is not None:
        prov["request_id"] = tenant_context.request_id

    run_record: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": timestamp,
        "status": "COMPLETED",
        "dataset_sha256": eval_result["dataset_sha256"],
        "test_sha256": eval_result["test_sha256"],
        "split_seed": seed,
        "preflight": preflight,
        "evaluation": eval_result,
        "model_manifest": model_manifest,
        "provenance": prov,
    }
    if tenant_context is not None:
        run_record["tenant_id"] = tenant_context.tenant_id

    run_record_sha256 = _run_body_digest(run_record)

    # Step 5: Record in tamper-evident evidence ledger
    ledger_file = root_dir / "evidence.jsonl"
    evidence_payload: dict[str, Any] = {
        "data_origin": "synthetic",
        "run_id": run_id,
        "dataset_sha256": eval_result["dataset_sha256"],
        "test_sha256": eval_result["test_sha256"],
        "f1": eval_result["f1"],
        "critical_false_negative_rate": eval_result["critical_false_negative_rate"],
        "preflight_status": preflight["status"],
        "created_at": timestamp,
        "run_record_sha256": run_record_sha256,
    }
    if tenant_context is not None:
        evidence_payload["tenant_id"] = tenant_context.tenant_id

    append_fn = facade_attr("evidence_append", evidence_append)
    evidence_entry = append_fn(str(ledger_file), evidence_payload, key=run_id)

    run_record["provenance"]["evidence_entry_hash"] = evidence_entry["entry_hash"]
    run_record["provenance"]["evidence_seq"] = evidence_entry["seq"]

    _write_json_atomic(run_dir / "run.json", run_record)
    return run_record  # type: ignore[return-value]


def create_task_run(
    *,
    task_spec: dict[str, Any],
    dataset_records: list[dict[str, Any]],
    baseline_predictions: Sequence[dict[str, Any]],
    model_predictions: Sequence[dict[str, Any]],
    output_root: str | Path | None = None,
    model_manifest: dict[str, Any] | None = None,
    ood_challenge_predictions: Sequence[dict[str, Any]] | None = None,
    tenant_context: TenantContext | None = None,
    allowed_data_origins: Sequence[str] | None = None,
) -> RunRecord:
    """Execute the task-mode vertical: spec-validated dataset, frozen family-disjoint test
    split, dual-predictor scoring, and tamper-evident run tracking. Predictions come from
    external harnesses; Foundry only validates, scores, and binds evidence."""
    if tenant_context is not None and not isinstance(tenant_context, TenantContext):
        raise ValueError("tenant_context must be a TenantContext instance")

    spec_errors = validate_spec(task_spec)
    if spec_errors:
        raise ValueError(f"invalid task spec: {'; '.join(spec_errors)}")

    root_dir = Path(output_root or DEFAULT_RUNS_DIR).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    # Single validation pass
    val_result = validate_task_dataset(dataset_records, task_spec)
    if val_result["status"] != "VALID":
        raise ValueError(f"invalid task dataset: {'; '.join(val_result['errors'])}")

    preflight = run_preflight(
        dataset=dataset_records,
        output_root=str(root_dir),
        model_manifest=model_manifest,
        task_spec=task_spec,
        allowed_data_origins=allowed_data_origins,
        _validation_result=val_result,
    )
    if preflight["status"] != "PASSED":
        raise ValueError(f"refinery run aborted: preflight failed ({'; '.join(preflight['errors'])})")

    seed = task_spec.get("split_seed", 42)
    splits = split_task_dataset(dataset_records, task_spec)
    test_rows = splits["test"]
    baseline_metrics = score_predictions(test_rows, baseline_predictions, task_spec)
    model_metrics = score_predictions(test_rows, model_predictions, task_spec)

    ood_summary: dict[str, Any] | None = None
    if ood_challenge_predictions is not None:
        if not isinstance(ood_challenge_predictions, (list, tuple)) or not ood_challenge_predictions:
            raise ValueError("ood_challenge_predictions must be a non-empty sequence of predictions")
        label_set = set(task_spec["labels"])
        ood_invalid = 0
        ood_correct = 0
        ood_has_gold = True
        for idx, row in enumerate(ood_challenge_predictions):
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row.get("label") not in label_set:
                raise ValueError(f"invalid OOD prediction at index {idx}")
            if row.get("raw_invalid", False):
                ood_invalid += 1
            if "gold_label" in row:
                if row["label"] == row["gold_label"]:
                    ood_correct += 1
            else:
                ood_has_gold = False
        ood_summary = {
            "count": len(ood_challenge_predictions),
            "predictions_sha256": digest(list(ood_challenge_predictions)),
            "invalid_output_rate": ood_invalid / len(ood_challenge_predictions),
            "accuracy": (ood_correct / len(ood_challenge_predictions)) if ood_has_gold else None,
        }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_hash = hashlib.sha256(
        f"{timestamp}:{digest(test_rows)}:{model_metrics['predictions_sha256']}:{secrets.token_hex(6)}".encode()
    ).hexdigest()[:12]
    run_id = f"run-{timestamp}-{run_hash}"
    run_dir = resolve_run_dir(root_dir, run_id)
    run_dir.mkdir(mode=0o700, exist_ok=False)

    dataset_sha256 = val_result["dataset_sha256"]
    assert dataset_sha256 is not None
    test_sha256 = digest(test_rows)

    task_block: dict[str, Any] = {
        "labels": task_spec["labels"],
        "thresholds": task_spec["thresholds"],
        "baseline_must_improve": task_spec.get("baseline_must_improve", True),
        "family_disjoint": val_result["family_disjoint"],
        "family_count": val_result["family_count"],
        "split_counts": {k: len(v) for k, v in splits.items()},
        "model_identity": task_spec["model"],
        "baseline": baseline_metrics,
        "model_metrics": model_metrics,
        "baseline_predictions_sha256": baseline_metrics["predictions_sha256"],
        "model_predictions_sha256": model_metrics["predictions_sha256"],
    }
    if ood_summary is not None:
        task_block["ood_challenge"] = ood_summary

    evaluation: dict[str, Any] = {
        "data_origin": "synthetic",
        "mode": "task_predictor_pair",
        "task_id": task_spec["task_id"],
        "test_count": len(test_rows),
        "dataset_sha256": dataset_sha256,
        "split_seed": seed,
        "test_sha256": test_sha256,
        "f1": model_metrics["macro_f1"],
        "critical_false_negative_rate": model_metrics["critical_false_negative_rate"],
        "coverage_automatic_decision": 0.0,
        "review_rate": 1.0,
        "cost_per_case_usd": None,
        "latency_p95_ms": model_metrics.get("latency_p95_ms"),
        "task": task_block,
        "note": (
            "External predictions scored against a frozen synthetic split. "
            "Synthetic self-consistency is not pilot accuracy; outcomes require human review."
        ),
    }

    prov: dict[str, Any] = {
        "code_digest": _code_digest(),
        "runtime": "offline_cpu",
        "evidence_ledger_path": "evidence.jsonl",
        "evidence_key": run_id,
    }
    if tenant_context is not None:
        prov["request_id"] = tenant_context.request_id

    run_record: dict[str, Any] = {
        "schema_version": "1.1",
        "run_id": run_id,
        "created_at": timestamp,
        "status": "COMPLETED",
        "dataset_sha256": dataset_sha256,
        "test_sha256": test_sha256,
        "split_seed": seed,
        "preflight": preflight,
        "evaluation": evaluation,
        "model_manifest": model_manifest,
        "provenance": prov,
    }
    if tenant_context is not None:
        run_record["tenant_id"] = tenant_context.tenant_id

    run_record_sha256 = _run_body_digest(run_record)

    ledger_file = root_dir / "evidence.jsonl"
    evidence_payload: dict[str, Any] = {
        "data_origin": "synthetic",
        "run_id": run_id,
        "task_id": task_spec["task_id"],
        "dataset_sha256": dataset_sha256,
        "test_sha256": test_sha256,
        "f1": model_metrics["macro_f1"],
        "critical_false_negative_rate": model_metrics["critical_false_negative_rate"],
        "preflight_status": preflight["status"],
        "created_at": timestamp,
        "run_record_sha256": run_record_sha256,
    }
    if tenant_context is not None:
        evidence_payload["tenant_id"] = tenant_context.tenant_id

    append_fn = facade_attr("evidence_append", evidence_append)
    evidence_entry = append_fn(str(ledger_file), evidence_payload, key=run_id)

    run_record["provenance"]["evidence_entry_hash"] = evidence_entry["entry_hash"]
    run_record["provenance"]["evidence_seq"] = evidence_entry["seq"]

    _write_json_atomic(run_dir / "run.json", run_record)
    return run_record  # type: ignore[return-value]


def _verify_seq1_metadata(run_id: str, run_record: dict[str, Any], matched_entry: dict[str, Any]) -> bool:
    rec = matched_entry.get("record", {})
    eval_obj = run_record.get("evaluation", {})
    if (
        rec.get("run_id") != run_id
        or rec.get("dataset_sha256") != run_record.get("dataset_sha256")
        or rec.get("test_sha256") != run_record.get("test_sha256")
        or rec.get("f1") != eval_obj.get("f1")
        or rec.get("critical_false_negative_rate") != eval_obj.get("critical_false_negative_rate")
        or rec.get("preflight_status") != run_record.get("preflight", {}).get("status")
    ):
        return False
    if "task_id" in rec and rec.get("task_id") != eval_obj.get("task_id"):
        return False
    return not ("run_record_sha256" in rec and rec.get("run_record_sha256") != _run_body_digest(run_record))


def inspect_run(
    run_id: str,
    *,
    output_root: str | Path | None = None,
    expected_head: str | None = None,
    expected_count: int | None = None,
    tenant_context: TenantContext | None = None,
    _include_snapshot: bool = False,
) -> InspectResult:
    """Inspect a tracked run and cryptographically verify its tamper-evident evidence chain."""
    if tenant_context is not None and not isinstance(tenant_context, TenantContext):
        raise ValueError("tenant_context must be a TenantContext instance")

    root_dir = Path(output_root or DEFAULT_RUNS_DIR).resolve()
    run_dir = resolve_run_dir(root_dir, run_id)
    run_file = run_dir / "run.json"
    if not run_file.is_file():
        if tenant_context is not None:
            raise PermissionError(DENY_REASON)
        raise FileNotFoundError(f"run not found: {run_id} (expected at {run_file})")

    with open(run_file, "r", encoding="utf-8") as handle:
        run_record = json.load(handle)

    if tenant_context is not None:
        auth = _authorize_tenant_run(tenant_context, run_record)
        if auth.get("status") != "ALLOWED":
            raise PermissionError(DENY_REASON)

    if (
        not isinstance(run_record, dict)
        or not isinstance(run_record.get("provenance"), dict)
        or not isinstance(run_record.get("preflight"), dict)
    ):
        raise TypeError("run metadata requires provenance and preflight objects")

    raw_ledger_path = run_record.get("provenance", {}).get("evidence_ledger_path")
    ledger_file = resolve_ledger_path(
        root_dir,
        raw_ledger_path,
        allow_legacy_outside_fallback=True,
    )
    expected_entry_hash = run_record.get("provenance", {}).get("evidence_entry_hash")
    expected_seq = run_record.get("provenance", {}).get("evidence_seq")

    provenance_status = "UNKNOWN"
    ledger_report: dict[str, Any] | None = None
    matched_entry: dict[str, Any] | None = None
    ledger_entries: list[dict[str, Any]] = []
    anchor_verified = True
    anchor_error: str | None = None

    if ledger_file.is_file():
        with open(ledger_file, "r", encoding="utf-8") as handle:
            ledger_entries = [json.loads(line) for line in handle if line.strip()]
        verify_fn = facade_attr("evidence_verify_entries", evidence_verify_entries)
        ledger_report = verify_fn(ledger_entries)
        if ledger_report["status"] == "OK":
            for entry in ledger_entries:
                if entry.get("key") == run_id or entry.get("entry_hash") == expected_entry_hash:
                    matched_entry = entry
                    break
            if (
                matched_entry
                and matched_entry.get("entry_hash") == expected_entry_hash
                and matched_entry.get("seq") == expected_seq
            ):
                if not _verify_seq1_metadata(run_id, run_record, matched_entry):
                    provenance_status = "TAMPERED_METADATA_MISMATCH"
                else:
                    provenance_status = "PROVENANCE_VERIFIED"
            else:
                provenance_status = "TAMPERED_ENTRY_MISMATCH"

            if provenance_status == "PROVENANCE_VERIFIED":
                if expected_head is not None and ledger_report.get("head") != expected_head:
                    provenance_status = "TAMPERED_ANCHOR_MISMATCH"
                    anchor_verified = False
                    anchor_error = f"expected_head mismatch: expected {expected_head}, got {ledger_report.get('head')}"
                elif expected_count is not None and ledger_report.get("count") != expected_count:
                    provenance_status = "TAMPERED_ANCHOR_MISMATCH"
                    anchor_verified = False
                    anchor_error = f"expected_count mismatch: expected {expected_count}, got {ledger_report.get('count')}"
        else:
            provenance_status = f"TAMPERED_CHAIN_{ledger_report['status']}"
    else:
        provenance_status = "LEDGER_NOT_FOUND"

    result: InspectResult = {
        "run_id": run_id,
        "status": run_record.get("status"),
        "created_at": run_record.get("created_at"),
        "provenance_status": provenance_status,
        "dataset_sha256": run_record.get("dataset_sha256"),
        "evaluation": run_record.get("evaluation"),
        "preflight_status": run_record.get("preflight", {}).get("status"),
        "provenance": run_record.get("provenance"),
        "ledger_report": ledger_report,
        "matched_entry": matched_entry,
        "model_manifest": run_record.get("model_manifest"),
        "resolved_ledger_path": str(ledger_file),
    }
    if expected_head is not None or expected_count is not None:
        result["anchor_verified"] = anchor_verified
        result["anchor_error"] = anchor_error
    if _include_snapshot:
        result["run_record"] = run_record
        result["ledger_entries"] = (
            _scope_ledger_entries(tenant_context, ledger_entries)
            if tenant_context is not None
            else ledger_entries
        )
    return result


def list_runs(
    *,
    output_root: str | Path | None = None,
    tenant_context: TenantContext | None = None,
    status: str | None = None,
    task_id: str | None = None,
    verify_provenance: bool = False,
) -> list[RunSummary]:
    """List tracked runs under output_root, optionally filtered by tenant, status, or task_id."""
    if tenant_context is not None and not isinstance(tenant_context, TenantContext):
        raise ValueError("tenant_context must be a TenantContext instance")

    root_dir = Path(output_root or DEFAULT_RUNS_DIR).resolve()
    if not root_dir.is_dir():
        return []

    runs: list[RunSummary] = []
    for child in sorted(root_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        try:
            rid = validate_run_id(child.name)
        except ValueError:
            continue
        run_file = child / "run.json"
        if not run_file.is_file():
            continue
        try:
            record = json.loads(run_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        if tenant_context is not None and _authorize_tenant_run(tenant_context, record).get("status") != "ALLOWED":
            continue

        raw_eval = record.get("evaluation")
        eval_obj: dict[str, Any] = raw_eval if isinstance(raw_eval, dict) else {}
        rec_status = record.get("status")
        rec_task_id = eval_obj.get("task_id")
        if status is not None and rec_status != status:
            continue
        if task_id is not None and rec_task_id != task_id:
            continue

        summary: RunSummary = {
            "run_id": rid,
            "created_at": record.get("created_at"),
            "status": rec_status,
            "schema_version": record.get("schema_version"),
            "mode": eval_obj.get("mode"),
            "task_id": rec_task_id,
            "dataset_sha256": record.get("dataset_sha256"),
            "test_sha256": record.get("test_sha256"),
            "f1": eval_obj.get("f1"),
            "critical_false_negative_rate": eval_obj.get("critical_false_negative_rate"),
        }
        if "tenant_id" in record:
            summary["tenant_id"] = record.get("tenant_id")
        if verify_provenance:
            try:
                insp = inspect_run(rid, output_root=str(root_dir), tenant_context=tenant_context)
                summary["provenance_status"] = insp["provenance_status"]
            except (OSError, ValueError, TypeError):
                summary["provenance_status"] = "INSPECTION_FAILED"
        runs.append(summary)

    runs.sort(key=lambda r: (r.get("created_at") or "", r["run_id"]), reverse=True)
    return runs


def diff_runs(
    run_id_a: str,
    run_id_b: str,
    *,
    output_root: str | Path | None = None,
    tenant_context: TenantContext | None = None,
) -> RunDiff:
    """Compare two tracked runs across provenance, dataset/code digests, and evaluation metrics."""
    insp_a = inspect_run(run_id_a, output_root=output_root, tenant_context=tenant_context, _include_snapshot=True)
    insp_b = inspect_run(run_id_b, output_root=output_root, tenant_context=tenant_context, _include_snapshot=True)

    rec_a = insp_a["run_record"]
    rec_b = insp_b["run_record"]
    raw_eval_a = rec_a.get("evaluation")
    eval_a: dict[str, Any] = raw_eval_a if isinstance(raw_eval_a, dict) else {}
    raw_eval_b = rec_b.get("evaluation")
    eval_b: dict[str, Any] = raw_eval_b if isinstance(raw_eval_b, dict) else {}

    def _delta(va: Any, vb: Any) -> float | None:
        if isinstance(va, (int, float)) and not isinstance(va, bool) and isinstance(vb, (int, float)) and not isinstance(vb, bool):
            return float(vb) - float(va)
        return None

    metrics_delta: dict[str, float | None] = {
        "f1_delta": _delta(eval_a.get("f1"), eval_b.get("f1")),
        "critical_fn_rate_delta": _delta(
            eval_a.get("critical_false_negative_rate"),
            eval_b.get("critical_false_negative_rate"),
        ),
        "latency_p95_ms_delta": _delta(eval_a.get("latency_p95_ms"), eval_b.get("latency_p95_ms")),
    }

    raw_task_a = eval_a.get("task")
    task_a: dict[str, Any] = raw_task_a if isinstance(raw_task_a, dict) else {}
    raw_task_b = eval_b.get("task")
    task_b: dict[str, Any] = raw_task_b if isinstance(raw_task_b, dict) else {}
    raw_mm_a = task_a.get("model_metrics")
    mm_a: dict[str, Any] = raw_mm_a if isinstance(raw_mm_a, dict) else {}
    raw_mm_b = task_b.get("model_metrics")
    mm_b: dict[str, Any] = raw_mm_b if isinstance(raw_mm_b, dict) else {}

    if mm_a or mm_b:
        metrics_delta["accuracy_delta"] = _delta(mm_a.get("accuracy"), mm_b.get("accuracy"))
        metrics_delta["macro_f1_delta"] = _delta(mm_a.get("macro_f1"), mm_b.get("macro_f1"))
        raw_cal_a = mm_a.get("calibration")
        cal_a: dict[str, Any] = raw_cal_a if isinstance(raw_cal_a, dict) else {}
        raw_cal_b = mm_b.get("calibration")
        cal_b: dict[str, Any] = raw_cal_b if isinstance(raw_cal_b, dict) else {}
        metrics_delta["ece_delta"] = _delta(cal_a.get("ece"), cal_b.get("ece"))
        metrics_delta["brier_score_delta"] = _delta(cal_a.get("brier_score"), cal_b.get("brier_score"))

    raw_pc_a = mm_a.get("per_class")
    pc_a: dict[str, Any] = raw_pc_a if isinstance(raw_pc_a, dict) else {}
    raw_pc_b = mm_b.get("per_class")
    pc_b: dict[str, Any] = raw_pc_b if isinstance(raw_pc_b, dict) else {}
    per_class_f1_delta: dict[str, float | None] = {}
    for label in sorted(set(pc_a) | set(pc_b)):
        entry_a = pc_a.get(label)
        entry_b = pc_b.get(label)
        fa = entry_a.get("f1") if isinstance(entry_a, dict) else None
        fb = entry_b.get("f1") if isinstance(entry_b, dict) else None
        per_class_f1_delta[label] = _delta(fa, fb)

    raw_prov_a = rec_a.get("provenance")
    prov_a: dict[str, Any] = raw_prov_a if isinstance(raw_prov_a, dict) else {}
    raw_prov_b = rec_b.get("provenance")
    prov_b: dict[str, Any] = raw_prov_b if isinstance(raw_prov_b, dict) else {}

    return {
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
        "provenance_a": insp_a["provenance_status"],
        "provenance_b": insp_b["provenance_status"],
        "same_dataset": rec_a.get("dataset_sha256") == rec_b.get("dataset_sha256"),
        "same_test_split": rec_a.get("test_sha256") == rec_b.get("test_sha256"),
        "same_code_digest": prov_a.get("code_digest") == prov_b.get("code_digest"),
        "same_model_identity": task_a.get("model_identity") == task_b.get("model_identity"),
        "metrics_delta": metrics_delta,
        "per_class_f1_delta": per_class_f1_delta,
        "thresholds_changed": task_a.get("thresholds") != task_b.get("thresholds"),
    }


def export_bundle(
    run_id: str,
    *,
    output_root: str | Path | None = None,
    dest_path: str | Path,
    fmt: str = "dir",
    tenant_context: TenantContext | None = None,
) -> dict[str, Any]:
    """Export a self-contained, SHA-256-manifested verification bundle (dir, zip, tar, tar.gz)."""
    if fmt not in _ALLOWED_BUNDLE_FORMATS:
        raise ValueError(f"unsupported bundle format {fmt!r}; expected one of {sorted(_ALLOWED_BUNDLE_FORMATS)}")

    insp = inspect_run(run_id, output_root=output_root, tenant_context=tenant_context, _include_snapshot=True)
    if insp["provenance_status"] != "PROVENANCE_VERIFIED":
        raise ValueError(f"refusing to export unverified run {run_id}: {insp['provenance_status']}")

    root_dir = Path(output_root or DEFAULT_RUNS_DIR).resolve()
    run_dir = resolve_run_dir(root_dir, run_id)
    run_bytes = (run_dir / "run.json").read_bytes()
    resolved_lp = insp["resolved_ledger_path"]
    assert resolved_lp is not None
    ledger_path = Path(resolved_lp)  # verified inside root_dir by inspect_run

    # Fail closed if exporting under a tenant_context and the ledger contains another tenant's records.
    if tenant_context is not None:
        raw_lines = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for entry in raw_lines:
            rec = entry.get("record") if isinstance(entry, dict) else None
            if not isinstance(rec, dict) or rec.get("tenant_id") != tenant_context.tenant_id:
                raise PermissionError(DENY_REASON)

    ledger_bytes = ledger_path.read_bytes()
    inspect_summary = {k: v for k, v in insp.items() if k not in ("run_record", "ledger_entries", "resolved_ledger_path")}
    inspect_bytes = (json.dumps(inspect_summary, indent=2, sort_keys=True) + "\n").encode("utf-8")

    payload_files: list[tuple[str, bytes]] = [
        ("run.json", run_bytes),
        ("evidence.jsonl", ledger_bytes),
        ("inspect.json", inspect_bytes),
    ]

    # Include any artifacts registered for this run_id if present in root_dir/artifacts
    art_meta_dir = root_dir / "artifacts" / "metadata"
    if art_meta_dir.is_dir():
        for meta_path in sorted(art_meta_dir.glob("*.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(meta, dict) and meta.get("run_id") == run_id and isinstance(meta.get("sha256"), str):
                blob = (root_dir / "artifacts" / "blobs" / "sha256" / meta["sha256"]).resolve()
                if blob.is_file() and blob.is_relative_to(root_dir):
                    payload_files.append((f"artifacts/blobs/sha256/{meta['sha256']}", blob.read_bytes()))
                    payload_files.append((f"artifacts/metadata/{meta['sha256']}.json", meta_path.read_bytes()))

    manifest_files = [
        {
            "path": rel,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
        for rel, data in sorted(payload_files, key=lambda item: item[0])
    ]
    manifest_core = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "format": fmt,
        "head": (insp.get("ledger_report") or {}).get("head"),
        "count": (insp.get("ledger_report") or {}).get("count"),
        "files": manifest_files,
    }
    bundle_manifest: BundleManifest = {
        **manifest_core,  # type: ignore[typeddict-item]
        "bundle_sha256": digest(manifest_core),
    }
    manifest_bytes = (json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    all_files = sorted([*payload_files, ("manifest.json", manifest_bytes)], key=lambda item: item[0])

    dest = Path(dest_path).resolve()
    if fmt == "dir":
        dest.mkdir(mode=0o700, parents=True, exist_ok=True)
        for rel, data in all_files:
            target = (dest / rel).resolve()
            if not target.is_relative_to(dest):
                raise ValueError("bundle entry escapes destination directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".foundry-bundle-", dir=str(target.parent))
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, target)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
    elif fmt == "zip":
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_zip = tempfile.mkstemp(prefix=".foundry-bundle-", dir=str(dest.parent), suffix=".zip")
        os.close(fd)
        try:
            with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for rel, data in all_files:
                    info = zipfile.ZipInfo(filename=rel, date_time=(2026, 1, 1, 0, 0, 0))
                    info.external_attr = 0o644 << 16
                    zf.writestr(info, data)
            os.replace(tmp_zip, dest)
        finally:
            if os.path.exists(tmp_zip):
                os.unlink(tmp_zip)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tar_mode: Literal["w:gz", "w"] = "w:gz" if fmt in ("tar.gz", "tgz") else "w"
        fd, tmp_tar = tempfile.mkstemp(prefix=".foundry-bundle-", dir=str(dest.parent))
        os.close(fd)
        try:
            with tarfile.open(tmp_tar, tar_mode) as tf:
                for rel, data in all_files:
                    ti = tarfile.TarInfo(name=rel)
                    ti.size = len(data)
                    ti.mtime = 0
                    ti.mode = 0o644
                    ti.uid = 0
                    ti.gid = 0
                    tf.addfile(ti, io.BytesIO(data))
            os.replace(tmp_tar, dest)
        finally:
            if os.path.exists(tmp_tar):
                os.unlink(tmp_tar)

    return {
        "status": "EXPORTED",
        "run_id": run_id,
        "format": fmt,
        "dest_path": str(dest),
        "bundle_sha256": bundle_manifest["bundle_sha256"],
        "file_count": len(all_files),
        "head": bundle_manifest["head"],
        "count": bundle_manifest["count"],
    }


def verify_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Verify an exported verification bundle (dir, zip, or tar/tar.gz) fail-closed."""
    target = Path(bundle_path).resolve()
    files_map: dict[str, bytes] = {}
    if target.is_dir():
        for path in target.rglob("*"):
            if path.is_file():
                rel = path.relative_to(target).as_posix()
                files_map[rel] = path.read_bytes()
    elif target.is_file() and zipfile.is_zipfile(target):
        with zipfile.ZipFile(target, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                if member.filename.startswith("/") or ".." in member.filename.split("/"):
                    raise ValueError(f"unsafe zip member path: {member.filename}")
                files_map[member.filename] = zf.read(member.filename)
    elif target.is_file():
        with tarfile.open(target, "r:*") as tf:
            for tar_member in tf.getmembers():
                if not tar_member.isfile():
                    continue
                if tar_member.name.startswith("/") or ".." in tar_member.name.split("/"):
                    raise ValueError(f"unsafe tar member path: {tar_member.name}")
                extracted = tf.extractfile(tar_member)
                if extracted is None:
                    raise ValueError(f"unreadable tar member: {tar_member.name}")
                files_map[tar_member.name] = extracted.read()
    else:
        raise FileNotFoundError(f"bundle not found: {bundle_path}")

    if "manifest.json" not in files_map or "run.json" not in files_map or "evidence.jsonl" not in files_map:
        raise ValueError("bundle missing required manifest.json, run.json, or evidence.jsonl")

    manifest = json.loads(files_map["manifest.json"].decode("utf-8"))
    core = {k: v for k, v in manifest.items() if k != "bundle_sha256"}
    if manifest.get("bundle_sha256") != digest(core):
        raise ValueError("bundle manifest_sha256 mismatch")

    for entry in manifest.get("files", []):
        rel = entry["path"]
        if rel not in files_map:
            raise ValueError(f"bundle missing declared file: {rel}")
        actual_sha = hashlib.sha256(files_map[rel]).hexdigest()
        if actual_sha != entry["sha256"]:
            raise ValueError(f"bundle file sha256 mismatch for {rel}")

    run_record = json.loads(files_map["run.json"].decode("utf-8"))
    ledger_entries = [
        json.loads(line)
        for line in files_map["evidence.jsonl"].decode("utf-8").splitlines()
        if line.strip()
    ]
    verify_fn = facade_attr("evidence_verify_entries", evidence_verify_entries)
    report = verify_fn(ledger_entries)
    if report["status"] != "OK":
        raise ValueError(f"bundled ledger verification failed: {report['status']}")
    if manifest.get("head") != report["head"] or manifest.get("count") != report["count"]:
        raise ValueError("bundled ledger head/count does not match manifest")

    run_id = manifest["run_id"]
    if run_record.get("run_id") != run_id:
        raise ValueError("bundled run.json run_id does not match manifest")

    expected_seq = (run_record.get("provenance") or {}).get("evidence_seq")
    expected_hash = (run_record.get("provenance") or {}).get("evidence_entry_hash")
    matched = next(
        (e for e in ledger_entries if e.get("seq") == expected_seq and e.get("entry_hash") == expected_hash),
        None,
    )
    if matched is None or not _verify_seq1_metadata(run_id, run_record, matched):
        raise ValueError("bundled run.json metadata does not match verified ledger entry")

    return {
        "status": "VERIFIED",
        "verified": True,
        "run_id": run_id,
        "bundle_sha256": manifest["bundle_sha256"],
        "head": report["head"],
        "count": report["count"],
        "file_count": len(manifest.get("files", [])),
    }
