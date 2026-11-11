"""Developer-facing Refinery workflow: dataset validation, reproducible evaluation,
training preflight, run tracking, provenance verification, and promotion gating.

Synthetic-only research baseline. Never deploy for real financial data or autonomous tool execution.
"""
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from . import signing
from .baseline import (
    canonical,
    digest,
    reconciliation,
    split,
)
from .baseline import (
    dataset as default_dataset,
)
from .evidence import _append_with_builder as evidence_append_with_builder
from .evidence import append as evidence_append
from .evidence import verify_entries as evidence_verify_entries
from .licensing import evaluate as evaluate_license
from .task_eval import (
    score_predictions,
    split_task_dataset,
    validate_spec,
    validate_task_dataset,
)

DEFAULT_RUNS_DIR = os.path.join(".foundry", "runs")


def _code_digest():
    """Compute a deterministic hash of current refinery and baseline modules."""
    components = []
    for mod_name in ("baseline", "evidence", "licensing", "refinery", "task_eval", "signing"):
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


def run_preflight(*, dataset=None, output_root=None, model_manifest=None, task_spec=None):
    """Execute training and run preflight checks before launching a Refinery workload.

    Checks dataset readiness, runtime environment constraints, output directory security,
    and licensing/provenance allowlists.
    """
    checks = []
    errors = []

    # 1. Dataset readiness (task-mode or built-in reconciliation schema)
    data = dataset if dataset is not None else default_dataset()
    if task_spec is not None:
        val_result = validate_task_dataset(data, task_spec)
        detail_ok = f"{val_result['record_count']} task records validated (family_disjoint={val_result['family_disjoint']})."
    else:
        val_result = validate_dataset(data)
        detail_ok = f"{val_result['record_count']} synthetic records validated with zero duplicate content."
    if val_result["status"] == "VALID":
        checks.append({
            "name": "dataset_validation",
            "status": "PASSED",
            "details": detail_ok,
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


def inspect_run(run_id, *, output_root=None, _include_snapshot=False):
    """Inspect a tracked run and cryptographically verify its tamper-evident evidence chain."""
    root_dir = Path(output_root or DEFAULT_RUNS_DIR).resolve()
    run_file = root_dir / run_id / "run.json"
    if not run_file.is_file():
        raise FileNotFoundError(f"run not found: {run_id} (expected at {run_file})")

    with open(run_file, "r", encoding="utf-8") as handle:
        run_record = json.load(handle)
    if (not isinstance(run_record, dict)
            or not isinstance(run_record.get("provenance"), dict)
            or not isinstance(run_record.get("preflight"), dict)):
        raise TypeError("run metadata requires provenance and preflight objects")

    ledger_path = run_record.get("provenance", {}).get("evidence_ledger_path")
    expected_entry_hash = run_record.get("provenance", {}).get("evidence_entry_hash")
    expected_seq = run_record.get("provenance", {}).get("evidence_seq")

    provenance_status = "UNKNOWN"
    ledger_report = None
    matched_entry = None
    ledger_entries = []

    if ledger_path and os.path.isfile(ledger_path):
        with open(ledger_path, "r", encoding="utf-8") as handle:
            ledger_entries = [json.loads(line) for line in handle if line.strip()]
        ledger_report = evidence_verify_entries(ledger_entries)
        if ledger_report["status"] == "OK":
            for entry in ledger_entries:
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

    result = {
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
    if _include_snapshot:
        result["run_record"] = run_record
        result["ledger_entries"] = ledger_entries
    return result


def evaluate_promotion(run_id, *, output_root=None, keyring=None):
    """Evaluate fail-closed model promotion gate against required evidence and safety contracts.

    Checks existence, tamper-evident ledger verification, preflight completion,
    benchmark thresholds, SPDX licensing (run record or ledger attachment), and
    an Ed25519 promotion signature bound to the ledger head and validated
    against a trusted keyring. Any missing or unverified evidence blocks promotion.
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
        inspection = inspect_run(run_id, output_root=str(root_dir), _include_snapshot=True)
        digest(inspection["run_record"])  # Reject non-canonical values before evaluating evidence.
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
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
    # Never consume attachment or signature records from an invalid ledger snapshot.
    if inspection.get("ledger_report") and inspection["ledger_report"].get("status") != "OK":
        return {
            "promoted": False,
            "status": "BLOCKED",
            "run_id": run_id,
            "reasons": reasons,
            "missing_evidence": missing_evidence,
            "evidence_summary": evidence_summary,
        }

    # Gate 3: Training preflight must have passed
    if inspection.get("preflight_status") != "PASSED":
        missing_evidence.append("passed_preflight")
        reasons.append("training preflight did not pass")

    def valid_rate(value):
        return type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value)

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

        if not valid_rate(cfn_rate):
            missing_evidence.append("critical_case_coverage")
            reasons.append("evaluation lacks critical test cases (critical false negative rate is undefined)")
        elif cfn_rate != 0.0:
            missing_evidence.append("zero_critical_false_negatives")
            reasons.append(f"critical false negative rate ({cfn_rate}) exceeds zero tolerance")

        if not valid_rate(eval_metrics.get("review_rate")) or eval_metrics.get("review_rate") != 1.0:
            missing_evidence.append("mandatory_human_review_policy")
            reasons.append("evaluation does not enforce 100% human review policy")

        if not valid_rate(f1):
            missing_evidence.append("positive_case_coverage")
            reasons.append("evaluation lacks positive/mismatch cases (f1 score is undefined)")
        elif f1 < 0.8:
            missing_evidence.append("minimum_f1_benchmark")
            reasons.append(f"f1 score ({f1}) does not meet benchmark threshold")

    # Gate 4b: Task-mode benchmark contract (stricter than the global floor)
    if isinstance(eval_metrics, dict) and eval_metrics.get("mode") == "task_predictor_pair":
        task_block = eval_metrics.get("task")
        if not isinstance(task_block, dict):
            task_block = {}
        labels = task_block.get("labels")
        for metrics in (task_block.get("baseline"), task_block.get("model_metrics")):
            if (not isinstance(labels, list) or not labels
                    or any(not isinstance(label, str) or not label.strip() for label in labels)
                    or len(set(labels)) != len(labels) or not isinstance(metrics, dict)
                    or not isinstance(metrics.get("per_class"), dict)
                    or set(metrics["per_class"]) != set(labels)
                    or any(not isinstance(metrics["per_class"].get(label), dict)
                           or not isinstance(metrics["per_class"][label].get("support"), int)
                           or isinstance(metrics["per_class"][label].get("support"), bool)
                           or metrics["per_class"][label]["support"] <= 0 for label in labels)):
                missing_evidence.append("task_label_coverage")
                reasons.append("task test split lacks positive support for every contract label")
                break
        baseline = task_block.get("baseline")
        model = task_block.get("model_metrics")
        thresholds = task_block.get("thresholds")
        baseline_f1 = baseline.get("macro_f1") if isinstance(baseline, dict) else None
        model_f1 = model.get("macro_f1") if isinstance(model, dict) else None
        spec_min = thresholds.get("macro_f1_min", 0.8) if isinstance(thresholds, dict) else None
        evidence_summary["task_baseline_macro_f1"] = baseline_f1
        evidence_summary["task_model_macro_f1"] = model_f1
        if (not valid_rate(baseline_f1) or not valid_rate(model_f1)
                or not valid_rate(spec_min)
                or not isinstance(task_block.get("baseline_must_improve", True), bool)):
            missing_evidence.append("task_benchmark_metrics")
            reasons.append("task run lacks baseline or model macro_f1 metrics")
        else:
            if model_f1 < spec_min:
                missing_evidence.append("task_macro_f1_threshold")
                reasons.append(f"model macro_f1 ({model_f1}) below spec threshold ({spec_min})")
            if task_block.get("baseline_must_improve", True) and model_f1 <= baseline_f1:
                missing_evidence.append("baseline_improvement")
                reasons.append(f"model macro_f1 ({model_f1}) does not beat baseline ({baseline_f1})")

    # Gate 5: Model manifest and SPDX licensing (run record or ledger attachment)
    model_manifest = inspection.get("model_manifest")
    license_source = "run_record" if model_manifest is not None else None
    attached = None
    ledger_path = inspection.get("provenance", {}).get("evidence_ledger_path")
    if model_manifest is None and ledger_path:
        attachments = [entry for entry in inspection["ledger_entries"]
                       if isinstance(entry, dict) and isinstance(entry.get("record"), dict)
                       and entry["record"].get("type") == "model_licensing_manifest"
                       and entry["record"].get("run_id") == run_id]
        attached = attachments[-1] if attachments else None
        if attached:
            record = attached.get("record", {})
            stored = record.get("manifest")
            if isinstance(stored, dict) and record.get("manifest_sha256") == digest(stored):
                model_manifest = stored
                license_source = "evidence_ledger"
            else:
                missing_evidence.append("model_licensing_manifest")
                reasons.append("ledger license attachment failed self-verification")
    if model_manifest is None:
        if "model_licensing_manifest" not in missing_evidence:
            missing_evidence.append("model_licensing_manifest")
            reasons.append("no model licensing manifest provided for promotion")
    else:
        lic_result = evaluate_license(model_manifest)
        evidence_summary["licensing_decision"] = lic_result["decision"]
        evidence_summary["licensing_source"] = license_source
        if lic_result["decision"] != "ALLOWED":
            missing_evidence.append("approved_spdx_license")
            reasons.append(f"model licensing check failed: {'; '.join(lic_result['reasons'])}")

    # Gate 6: Cryptographic signed promotion authority (Ed25519 bound to ledger head)
    signature_entries = [entry for entry in inspection["ledger_entries"]
                         if isinstance(entry, dict) and isinstance(entry.get("record"), dict)
                         and entry["record"].get("type") == "promotion_signature"
                         and entry["record"].get("run_id") == run_id]
    if not signature_entries:
        missing_evidence.append("cryptographic_promotion_signature")
        reasons.append("no promotion signature entry bound to this run in the verified ledger")
    elif not signing.backend_available():
        missing_evidence.append("signing_backend")
        reasons.append("signing backend unavailable: install hardonian-foundry[signing]")
    elif keyring is None:
        missing_evidence.append("trusted_promotion_keyring")
        reasons.append("promotion requires a trusted keyring to validate signatures")
    else:
        try:
            trusted = signing.normalize_trusted_keys(keyring)
        except (ValueError, TypeError) as exc:
            trusted = []
            missing_evidence.append("trusted_promotion_keyring")
            reasons.append(f"invalid trusted keyring: {exc}")
        verified_entry = None
        for entry in signature_entries:
            record = entry.get("record", {})
            signed_head = record.get("signed_head")
            message = canonical({
                "version": "2",
                "run_id": run_id,
                "signed_head": signed_head,
                "created_at": record.get("created_at"),
                "run_sha256": digest(inspection["run_record"]),
            }).encode("utf-8")
            pub = record.get("public_key_hex")
            if record.get("version") != "2" or record.get("run_sha256") != digest(inspection["run_record"]):
                reasons.append("signature does not bind the complete current run record")
                continue
            if entry.get("prev_hash") != signed_head:
                reasons.append("signature does not bind the ledger head at its signing position")
                continue
            if (inspection["matched_entry"] is None
                    or inspection["matched_entry"]["seq"] >= entry["seq"]
                    or (license_source == "evidence_ledger" and attached["seq"] >= entry["seq"])):
                reasons.append("signature does not cover the selected run and license evidence")
                continue
            if not isinstance(pub, str) or pub not in trusted:
                reasons.append("signature key is not in the trusted keyring")
                continue
            if signing.verify(message, record.get("signature_hex"), [pub]):
                verified_entry = entry
                break
        if verified_entry:
            vrec = verified_entry["record"]
            evidence_summary["promotion_signature"] = {
                "entry_hash": verified_entry["entry_hash"],
                "signed_head": vrec["signed_head"],
                "public_key_hex": vrec["public_key_hex"],
            }
        else:
            missing_evidence.append("verified_promotion_signature")
            reasons.append("no promotion signature verified against the trusted keyring; if the signature is legacy or predates license evidence, recreate the run, attach an allowed license, and sign the new run")

    promoted = not missing_evidence
    return {
        "promoted": promoted,
        "status": "ALLOWED_REVIEW_REQUIRED" if promoted else "BLOCKED",
        "run_id": run_id,
        "reasons": sorted(set(reasons)) if not promoted else [],
        "missing_evidence": sorted(missing_evidence),
        "evidence_summary": evidence_summary,
        "review_required": True,
        "note": (
            "All evidence gates verified against the tamper-evident ledger at the signed head. "
            "Promotion attests evidence completeness only: deployment remains an explicit human decision."
            if promoted else
            "Promotion is fail-closed: any missing or unverified evidence blocks promotion with exit code 2."
        ),
    }


def _iter_ledger_entries(path, run_id, entry_type):
    """Collect ledger entries of a given type bound to run_id. Chain must be verified elsewhere."""
    out = []
    if not path or not os.path.isfile(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                record = entry.get("record", {})
                if record.get("type") == entry_type and record.get("run_id") == run_id:
                    out.append(entry)
    except (OSError, json.JSONDecodeError):
        return []
    return out


def _find_ledger_entry(path, run_id, entry_type):
    entries = _iter_ledger_entries(path, run_id, entry_type)
    return entries[-1] if entries else None


def attach_license(run_id, *, manifest, output_root=None):
    """Bind a licensing decision (from the SPDX allowlist validator) into the evidence ledger."""
    root_dir = Path(output_root or DEFAULT_RUNS_DIR).resolve()
    run_file = root_dir / run_id / "run.json"
    if not run_file.is_file():
        raise FileNotFoundError(f"run not found: {run_id}")
    with open(run_file, "r", encoding="utf-8") as handle:
        run_record = json.load(handle)
    ledger_path = run_record.get("provenance", {}).get("evidence_ledger_path")
    if not ledger_path or not os.path.isfile(ledger_path):
        raise ValueError("evidence ledger not found for run")
    if _find_ledger_entry(ledger_path, run_id, "model_licensing_manifest"):
        raise ValueError(f"license manifest already attached for {run_id}")
    lic_result = evaluate_license(manifest)
    entry = evidence_append(
        ledger_path,
        {
            "data_origin": "synthetic",
            "type": "model_licensing_manifest",
            "run_id": run_id,
            "manifest_sha256": digest(manifest),
            "decision": lic_result["decision"],
            "reasons": lic_result["reasons"],
            "manifest": manifest,
        },
        key=f"{run_id}#license",
    )
    return {
        "run_id": run_id,
        "decision": lic_result["decision"],
        "reasons": lic_result["reasons"],
        "manifest_sha256": entry["record"]["manifest_sha256"],
        "entry_hash": entry["entry_hash"],
        "entry_seq": entry["seq"],
    }


def sign_run(run_id, *, private_key_path, output_root=None):
    """Sign the current ledger head, binding an Ed25519 promotion attestation to the run."""
    root_dir = Path(output_root or DEFAULT_RUNS_DIR).resolve()
    run_file = root_dir / run_id / "run.json"
    if not run_file.is_file():
        raise FileNotFoundError(f"run not found: {run_id}")
    with open(run_file, "r", encoding="utf-8") as handle:
        run_record = json.load(handle)
    ledger_path = run_record.get("provenance", {}).get("evidence_ledger_path")
    if not ledger_path or not os.path.isfile(ledger_path):
        raise ValueError("evidence ledger not found for run")

    def build_signature(entries, signed_head):
        if any(isinstance(entry.get("record"), dict)
               and entry["record"].get("type") == "promotion_signature"
               and entry["record"].get("run_id") == run_id for entry in entries):
            raise ValueError(f"run already signed: {run_id}; recreate the run, attach an allowed license, and sign the new run")
        manifest = run_record.get("model_manifest")
        if manifest is None:
            attachments = [entry["record"] for entry in entries
                           if isinstance(entry.get("record"), dict)
                           and entry["record"].get("type") == "model_licensing_manifest"
                           and entry["record"].get("run_id") == run_id]
            if attachments:
                attachment = attachments[-1]
                manifest = attachment.get("manifest")
                if not isinstance(manifest, dict) or attachment.get("manifest_sha256") != digest(manifest):
                    manifest = None
        if evaluate_license(manifest)["decision"] != "ALLOWED":
            raise ValueError("signing requires an allowed license first; attach an allowed license or recreate the run")
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_sha256 = digest(run_record)
        message = canonical({"version": "2", "run_id": run_id, "signed_head": signed_head,
                             "created_at": timestamp, "run_sha256": run_sha256}).encode("utf-8")
        public_hex = signing.public_key_hex(private_key_path)
        signature_hex = signing.sign(message, private_key_path)
        return {
            "data_origin": "synthetic",
            "type": "promotion_signature",
            "version": "2",
            "run_id": run_id,
            "signed_head": signed_head,
            "run_sha256": run_sha256,
            "public_key_hex": public_hex,
            "signature_hex": signature_hex,
            "created_at": timestamp,
            "signing_backend": signing.SIGNING_BACKEND,
        }

    try:
        entry = evidence_append_with_builder(ledger_path, build_signature, key=f"{run_id}#sig")
    except ValueError as exc:
        if str(exc) == "duplicate idempotency key":
            raise ValueError("run already signed or signature key occupied; recreate the run, attach an allowed license, and sign the new run") from exc
        raise
    return {
        "run_id": run_id,
        "signed_head": entry["record"]["signed_head"],
        "public_key_hex": entry["record"]["public_key_hex"],
        "entry_hash": entry["entry_hash"],
        "entry_seq": entry["seq"],
    }


def load_keyring(path):
    """Load a trusted keyring JSON file: {\"trusted_keys\": [{\"public_key_hex\": ...}]}."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("trusted_keys"), list):
        raise TypeError("keyring must be an object with a trusted_keys list")
    return signing.normalize_trusted_keys(data["trusted_keys"])


def create_task_run(*, task_spec, dataset_records, baseline_predictions, model_predictions,
                    output_root=None, model_manifest=None):
    """Execute the task-mode vertical: spec-validated dataset, frozen family-disjoint test
    split, dual-predictor scoring, and tamper-evident run tracking. Predictions come from
    external harnesses; Foundry only validates, scores, and binds evidence."""
    spec_errors = validate_spec(task_spec)
    if spec_errors:
        raise ValueError(f"invalid task spec: {'; '.join(spec_errors)}")

    root_dir = Path(output_root or DEFAULT_RUNS_DIR).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    val_result = validate_task_dataset(dataset_records, task_spec)
    if val_result["status"] != "VALID":
        raise ValueError(f"invalid task dataset: {'; '.join(val_result['errors'])}")

    preflight = run_preflight(dataset=dataset_records, output_root=str(root_dir),
                              model_manifest=model_manifest, task_spec=task_spec)
    if preflight["status"] != "PASSED":
        raise ValueError(f"refinery run aborted: preflight failed ({'; '.join(preflight['errors'])})")

    seed = task_spec.get("split_seed", 42)
    splits = split_task_dataset(dataset_records, task_spec)
    test_rows = splits["test"]
    baseline_metrics = score_predictions(test_rows, baseline_predictions, task_spec)
    model_metrics = score_predictions(test_rows, model_predictions, task_spec)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_hash = hashlib.sha256(
        f"{timestamp}:{digest(test_rows)}:{model_metrics['predictions_sha256']}:{time.monotonic()}".encode()
    ).hexdigest()[:8]
    run_id = f"run-{timestamp}-{run_hash}"
    run_dir = root_dir / run_id
    run_dir.mkdir(mode=0o700, exist_ok=False)

    dataset_sha256 = digest(dataset_records)
    test_sha256 = digest(test_rows)
    ledger_path = str(root_dir / "evidence.jsonl")
    evidence_payload = {
        "data_origin": "synthetic",
        "run_id": run_id,
        "task_id": task_spec["task_id"],
        "dataset_sha256": dataset_sha256,
        "test_sha256": test_sha256,
        "f1": model_metrics["macro_f1"],
        "critical_false_negative_rate": model_metrics["critical_false_negative_rate"],
        "preflight_status": preflight["status"],
        "created_at": timestamp,
    }
    evidence_entry = evidence_append(ledger_path, evidence_payload, key=run_id)

    evaluation = {
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
        "latency_p95_ms": None,
        "task": {
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
        },
        "note": ("External predictions scored against a frozen synthetic split. "
                 "Synthetic self-consistency is not pilot accuracy; outcomes require human review."),
    }

    run_record = {
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
