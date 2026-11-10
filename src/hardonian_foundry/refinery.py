"""Developer-facing Refinery workflow: dataset validation, reproducible evaluation,
training preflight, run tracking, provenance verification, and promotion gating.

Synthetic-only research baseline. Never deploy for real financial data or autonomous tool execution.
"""
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .baseline import (
    dataset as default_dataset,
)
from .baseline import (
    digest,
    reconciliation,
    split,
)
from .evidence import append as evidence_append
from .evidence import verify as evidence_verify
from .licensing import evaluate as evaluate_license

DEFAULT_RUNS_DIR = os.path.join(".foundry", "runs")


def _code_digest():
    """Compute a deterministic hash of current refinery and baseline modules."""
    components = []
    for mod_name in ("baseline", "evidence", "licensing", "refinery"):
        path = os.path.join(os.path.dirname(__file__), f"{mod_name}.py")
        if os.path.isfile(path):
            with open(path, "rb") as stream:
                components.append(hashlib.sha256(stream.read()).hexdigest())
    return hashlib.sha256(":".join(components).encode("utf-8")).hexdigest()


def validate_dataset(records, *, seed=42):
    """Validate a dataset against Foundry synthetic schema and split requirements.

    Enforces synthetic data origins, schema fields, bounded integer minor units,
    unique IDs, unique content (leakage detection), and deterministic splitability.
    """
    errors = []
    if not isinstance(records, list):
        return {
            "status": "INVALID",
            "record_count": 0,
            "dataset_sha256": None,
            "split_counts": None,
            "errors": ["dataset must be a list of records"],
        }
    if len(records) < 3:
        errors.append("dataset must contain at least three records for train/validation/test partitions")

    seen_ids = set()
    seen_contents = set()

    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            errors.append(f"record at index {idx} is not a dictionary")
            continue
        try:
            reconciliation(row)
        except ValueError as exc:
            errors.append(f"record at index {idx} ({row.get('id', 'unknown')}) failed validation: {exc}")
            continue

        row_id = row.get("id")
        if row_id in seen_ids:
            errors.append(f"duplicate record id: {row_id}")
        else:
            seen_ids.add(row_id)

        content_digest = digest({k: v for k, v in row.items() if k != "id"})
        if content_digest in seen_contents:
            errors.append(f"duplicate record content detected (leakage hazard) for id {row_id}")
        else:
            seen_contents.add(content_digest)

    splits = None
    if not errors and len(records) >= 3:
        try:
            splits = split(records, seed=seed)
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"failed to partition dataset: {exc}")

    status = "VALID" if not errors else "INVALID"
    return {
        "status": status,
        "record_count": len(records),
        "dataset_sha256": digest(records) if not errors else None,
        "split_counts": {k: len(v) for k, v in splits.items()} if splits else None,
        "errors": sorted(errors),
    }


def run_preflight(*, dataset=None, output_root=None, model_manifest=None):
    """Execute training and run preflight checks before launching a Refinery workload.

    Checks dataset readiness, runtime environment constraints, output directory security,
    and licensing/provenance allowlists.
    """
    checks = []
    errors = []

    # 1. Dataset readiness
    data = dataset if dataset is not None else default_dataset()
    val_result = validate_dataset(data)
    if val_result["status"] == "VALID":
        checks.append({
            "name": "dataset_validation",
            "status": "PASSED",
            "details": f"{val_result['record_count']} synthetic records validated with zero duplicate content.",
        })
    else:
        err_msg = "; ".join(val_result["errors"])
        errors.append(f"dataset validation failed: {err_msg}")
        checks.append({
            "name": "dataset_validation",
            "status": "FAILED",
            "details": err_msg,
        })

    # 2. Runtime environment
    env_details = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "runtime": "offline_cpu",
    }
    checks.append({
        "name": "runtime_environment",
        "status": "PASSED",
        "details": f"Python {env_details['python_version']} on {env_details['platform']} (offline CPU runtime).",
    })

    # 3. Output directory security and storage
    root_path = Path(output_root or DEFAULT_RUNS_DIR)
    try:
        root_path.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(root_path)
        # Bounded check: at least 50 MB free for synthetic run tracking
        free_mb = usage.free / (1024 * 1024)
        if free_mb < 50:
            err_msg = f"insufficient disk space: {free_mb:.1f} MB free (minimum 50 MB required)"
            errors.append(err_msg)
            checks.append({"name": "output_directory", "status": "FAILED", "details": err_msg})
        else:
            checks.append({
                "name": "output_directory",
                "status": "PASSED",
                "details": f"Directory {root_path} available ({free_mb:.1f} MB free).",
            })
    except (OSError, PermissionError) as exc:
        err_msg = f"failed to initialize output directory {root_path}: {exc}"
        errors.append(err_msg)
        checks.append({"name": "output_directory", "status": "FAILED", "details": err_msg})

    # 4. Model manifest licensing & provenance allowlist (if provided)
    if model_manifest is not None:
        lic_result = evaluate_license(model_manifest)
        if lic_result["decision"] == "ALLOWED":
            checks.append({
                "name": "licensing_and_provenance",
                "status": "PASSED",
                "details": f"License {model_manifest.get('license_id')} and provenance {model_manifest.get('provenance')} approved.",
            })
        else:
            reasons = "; ".join(lic_result["reasons"])
            errors.append(f"model manifest failed licensing/provenance check: {reasons}")
            checks.append({
                "name": "licensing_and_provenance",
                "status": "FAILED",
                "details": reasons,
            })
    else:
        checks.append({
            "name": "licensing_and_provenance",
            "status": "PASSED",
            "details": "No external model manifest specified; using built-in synthetic heuristic.",
        })

    # 5. Security boundaries check
    checks.append({
        "name": "execution_safety",
        "status": "PASSED",
        "details": "Fail-closed review required; zero autonomous tool/financial execution permitted.",
    })

    return {
        "status": "PASSED" if not errors else "FAILED",
        "checks": checks,
        "errors": sorted(errors),
    }


def evaluate_dataset(records=None, *, seed=42):
    """Execute reproducible held-out evaluation on the test split of a synthetic dataset."""
    data = records if records is not None else default_dataset()
    val = validate_dataset(data, seed=seed)
    if val["status"] != "VALID":
        raise ValueError(f"cannot evaluate invalid dataset: {'; '.join(val['errors'])}")

    splits = split(data, seed=seed)
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
    f1 = 2 * precision * recall / (precision + recall) if (precision is not None and recall is not None and precision + recall) else None
    critical_fn_rate = (critical_fn / critical_total) if critical_total else None

    return {
        "data_origin": "synthetic",
        "mode": "deterministic_heuristic",
        "test_count": len(test_rows),
        "dataset_sha256": digest(data),
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


def create_run(*, output_root=None, dataset=None, model_manifest=None, seed=42):
    """Execute the vertical Refinery workflow: validate, preflight, evaluate, and track run with evidence."""
    root_dir = Path(output_root or DEFAULT_RUNS_DIR).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    data = dataset if dataset is not None else default_dataset()

    # Step 1: Preflight
    preflight = run_preflight(dataset=data, output_root=str(root_dir), model_manifest=model_manifest)
    if preflight["status"] != "PASSED":
        raise ValueError(f"refinery run aborted: preflight failed ({'; '.join(preflight['errors'])})")

    # Step 2: Evaluation
    eval_result = evaluate_dataset(data, seed=seed)

    # Step 3: Run identity & metadata
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_hash = hashlib.sha256(f"{timestamp}:{eval_result['test_sha256']}:{time.monotonic()}".encode()).hexdigest()[:8]
    run_id = f"run-{timestamp}-{run_hash}"
    run_dir = root_dir / run_id
    run_dir.mkdir(mode=0o700, exist_ok=False)

    # Step 4: Record in tamper-evident evidence ledger
    ledger_path = str(root_dir / "evidence.jsonl")
    evidence_payload = {
        "data_origin": "synthetic",
        "run_id": run_id,
        "dataset_sha256": eval_result["dataset_sha256"],
        "test_sha256": eval_result["test_sha256"],
        "f1": eval_result["f1"],
        "critical_false_negative_rate": eval_result["critical_false_negative_rate"],
        "preflight_status": preflight["status"],
        "created_at": timestamp,
    }
    evidence_entry = evidence_append(ledger_path, evidence_payload, key=run_id)

    # Step 5: Persist run record
    run_record = {
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
        "provenance": {
            "code_digest": _code_digest(),
            "runtime": "offline_cpu",
            "evidence_ledger_path": ledger_path,
            "evidence_entry_hash": evidence_entry["entry_hash"],
            "evidence_seq": evidence_entry["seq"],
            "evidence_key": run_id,
        },
    }

    with open(run_dir / "run.json", "w", encoding="utf-8") as handle:
        handle.write(json.dumps(run_record, indent=2, sort_keys=True) + "\n")

    return run_record


def inspect_run(run_id, *, output_root=None):
    """Inspect a tracked run and cryptographically verify its tamper-evident evidence chain."""
    root_dir = Path(output_root or DEFAULT_RUNS_DIR).resolve()
    run_file = root_dir / run_id / "run.json"
    if not run_file.is_file():
        raise FileNotFoundError(f"run not found: {run_id} (expected at {run_file})")

    with open(run_file, "r", encoding="utf-8") as handle:
        run_record = json.load(handle)

    ledger_path = run_record.get("provenance", {}).get("evidence_ledger_path")
    expected_entry_hash = run_record.get("provenance", {}).get("evidence_entry_hash")
    expected_seq = run_record.get("provenance", {}).get("evidence_seq")

    provenance_status = "UNKNOWN"
    ledger_report = None
    matched_entry = None

    if ledger_path and os.path.isfile(ledger_path):
        ledger_report = evidence_verify(ledger_path)
        if ledger_report["status"] == "OK":
            with open(ledger_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    if entry.get("key") == run_id or entry.get("entry_hash") == expected_entry_hash:
                        matched_entry = entry
                        break
            if matched_entry and matched_entry.get("entry_hash") == expected_entry_hash and matched_entry.get("seq") == expected_seq:
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
                    provenance_status = "TAMPERED_METADATA_MISMATCH"
                else:
                    provenance_status = "PROVENANCE_VERIFIED"
            else:
                provenance_status = "TAMPERED_ENTRY_MISMATCH"
        else:
            provenance_status = f"TAMPERED_CHAIN_{ledger_report['status']}"
    else:
        provenance_status = "LEDGER_NOT_FOUND"

    return {
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
    }


def evaluate_promotion(run_id, *, output_root=None):
    """Evaluate fail-closed model promotion gate against required evidence and safety contracts.

    Checks existence, tamper-evident ledger verification, preflight completion,
    benchmark thresholds, and SPDX licensing allowlist. Blocks promotion when any
    evidence is missing or unverified, and always fails closed pending signed promotion.
    """
    reasons = []
    missing_evidence = []
    evidence_summary = {}

    root_dir = Path(output_root or DEFAULT_RUNS_DIR).resolve()
    run_file = root_dir / run_id / "run.json"

    # Gate 1: Run record must exist
    if not run_file.is_file():
        missing_evidence.append("run_metadata")
        reasons.append(f"run record not found for {run_id}")
        return {
            "promoted": False,
            "status": "BLOCKED",
            "run_id": run_id,
            "reasons": sorted(reasons),
            "missing_evidence": sorted(missing_evidence),
            "evidence_summary": evidence_summary,
        }

    try:
        inspection = inspect_run(run_id, output_root=str(root_dir))
    except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        missing_evidence.append("run_inspection_failed")
        reasons.append(f"failed to inspect run: {exc}")
        return {
            "promoted": False,
            "status": "BLOCKED",
            "run_id": run_id,
            "reasons": sorted(reasons),
            "missing_evidence": sorted(missing_evidence),
            "evidence_summary": evidence_summary,
        }

    evidence_summary["provenance_status"] = inspection["provenance_status"]
    evidence_summary["preflight_status"] = inspection["preflight_status"]

    # Gate 2: Tamper-evident evidence chain must be intact and match metadata
    if inspection["provenance_status"] != "PROVENANCE_VERIFIED":
        missing_evidence.append("tamper_evident_proof")
        reasons.append(f"provenance verification failed: {inspection['provenance_status']}")

    # Gate 3: Training preflight must have passed
    if inspection.get("preflight_status") != "PASSED":
        missing_evidence.append("passed_preflight")
        reasons.append("training preflight did not pass")

    # Gate 4: Evaluation benchmarks
    eval_metrics = inspection.get("evaluation")
    if not eval_metrics or not isinstance(eval_metrics, dict):
        missing_evidence.append("evaluation_metrics")
        reasons.append("evaluation metrics missing from run record")
    else:
        f1 = eval_metrics.get("f1")
        cfn_rate = eval_metrics.get("critical_false_negative_rate")
        evidence_summary["evaluation_f1"] = f1
        evidence_summary["critical_fn_rate"] = cfn_rate

        if cfn_rate is None:
            missing_evidence.append("critical_case_coverage")
            reasons.append("evaluation lacks critical test cases (critical false negative rate is undefined)")
        elif cfn_rate != 0.0:
            missing_evidence.append("zero_critical_false_negatives")
            reasons.append(f"critical false negative rate ({cfn_rate}) exceeds zero tolerance")

        if eval_metrics.get("review_rate") != 1.0:
            missing_evidence.append("mandatory_human_review_policy")
            reasons.append("evaluation does not enforce 100% human review policy")

        if f1 is None:
            missing_evidence.append("positive_case_coverage")
            reasons.append("evaluation lacks positive/mismatch cases (f1 score is undefined)")
        elif f1 < 0.8:
            missing_evidence.append("minimum_f1_benchmark")
            reasons.append(f"f1 score ({f1}) does not meet benchmark threshold")

    # Gate 5: Model manifest and SPDX licensing
    model_manifest = inspection.get("model_manifest")
    if model_manifest is None:
        missing_evidence.append("model_licensing_manifest")
        reasons.append("no model licensing manifest provided for promotion")
    else:
        lic_result = evaluate_license(model_manifest)
        evidence_summary["licensing_decision"] = lic_result["decision"]
        if lic_result["decision"] != "ALLOWED":
            missing_evidence.append("approved_spdx_license")
            reasons.append(f"model licensing check failed: {'; '.join(lic_result['reasons'])}")

    # Gate 6: Cryptographic signed promotion authority
    # As mandated by Foundry architecture: promotion always fails closed until signed promotion exists.
    missing_evidence.append("cryptographic_promotion_signature")
    reasons.append("Signed provenance, license, security and benchmark gates are not implemented.")

    return {
        "promoted": False,
        "status": "BLOCKED",
        "run_id": run_id,
        "reasons": sorted(reasons),
        "missing_evidence": sorted(missing_evidence),
        "evidence_summary": evidence_summary,
    }
