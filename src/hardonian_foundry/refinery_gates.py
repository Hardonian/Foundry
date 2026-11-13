"""Refinery promotion gate evaluation: fail-closed benchmarks, licensing, quorum, and anchors."""
from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from .canonical import canonical, digest  # type: ignore[import-not-found,import-untyped]
except ImportError:  # pragma: no cover - pre-merge fallback
    from .baseline import canonical, digest

from . import signing
from .refinery_attestation import (
    _evaluate_license_compat,
    extract_verified_license_attachment,
    parse_keyring,
)
from .refinery_runs import inspect_run
from .refinery_validation import DEFAULT_RUNS_DIR, facade_attr, validate_run_id
from .tenancy import TenantContext
from .types import PromotionResult


def _promotion_response(
    run_id: str,
    *,
    promoted: bool,
    reasons: Sequence[str],
    missing_evidence: Sequence[str],
    evidence_summary: dict[str, Any],
) -> PromotionResult:
    return {
        "promoted": promoted,
        "status": "ALLOWED_REVIEW_REQUIRED" if promoted else "BLOCKED",
        "run_id": run_id,
        "reasons": sorted(set(reasons)) if not promoted else [],
        "missing_evidence": sorted(set(missing_evidence)),
        "evidence_summary": evidence_summary,
        "review_required": True,
        "note": (
            "All evidence gates verified against the tamper-evident ledger at the signed head. "
            "Promotion attests evidence completeness only: deployment remains an explicit human decision."
            if promoted
            else "Promotion is fail-closed: any missing or unverified evidence blocks promotion with exit code 2."
        ),
    }


def _as_rate(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        fval = float(value)
        if math.isfinite(fval) and 0.0 <= fval <= 1.0:
            return fval
    return None


def _valid_rate(value: Any) -> bool:
    return _as_rate(value) is not None


def _evaluate_task_thresholds(
    task_block: dict[str, Any],
    missing_evidence: list[str],
    reasons: list[str],
    evidence_summary: dict[str, Any],
) -> None:
    labels = task_block.get("labels")
    for metrics in (task_block.get("baseline"), task_block.get("model_metrics")):
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)
            or not isinstance(metrics, dict)
            or not isinstance(metrics.get("per_class"), dict)
            or set(metrics["per_class"]) != set(labels)
            or any(
                not isinstance(metrics["per_class"].get(label), dict)
                or not isinstance(metrics["per_class"][label].get("support"), int)
                or isinstance(metrics["per_class"][label].get("support"), bool)
                or metrics["per_class"][label]["support"] <= 0
                for label in labels
            )
        ):
            missing_evidence.append("task_label_coverage")
            reasons.append("task test split lacks positive support for every contract label")
            break

    baseline = task_block.get("baseline")
    model = task_block.get("model_metrics")
    thresholds = task_block.get("thresholds")
    baseline_f1 = _as_rate(baseline.get("macro_f1") if isinstance(baseline, dict) else None)
    model_f1 = _as_rate(model.get("macro_f1") if isinstance(model, dict) else None)
    spec_min = _as_rate(thresholds.get("macro_f1_min", 0.8) if isinstance(thresholds, dict) else None)
    evidence_summary["task_baseline_macro_f1"] = baseline_f1
    evidence_summary["task_model_macro_f1"] = model_f1

    if (
        baseline_f1 is None
        or model_f1 is None
        or spec_min is None
        or not isinstance(task_block.get("baseline_must_improve", True), bool)
    ):
        missing_evidence.append("task_benchmark_metrics")
        reasons.append("task run lacks baseline or model macro_f1 metrics")
        return

    if model_f1 < spec_min:
        missing_evidence.append("task_macro_f1_threshold")
        reasons.append(f"model macro_f1 ({model_f1}) below spec threshold ({spec_min})")
    if task_block.get("baseline_must_improve", True) and model_f1 <= baseline_f1:
        missing_evidence.append("baseline_improvement")
        reasons.append(f"model macro_f1 ({model_f1}) does not beat baseline ({baseline_f1})")

    if not isinstance(thresholds, dict) or not isinstance(model, dict):
        return

    # Optional Sprint 2-4 task thresholds (strictly opt-in via spec.thresholds)
    if "critical_fn_rate_max" in thresholds:
        cfn_max = _as_rate(thresholds["critical_fn_rate_max"])
        cfn_val = _as_rate(model.get("critical_false_negative_rate"))
        if cfn_max is None or cfn_val is None or cfn_val > cfn_max:
            missing_evidence.append("task_critical_fn_rate_threshold")
            reasons.append(f"model critical_false_negative_rate ({model.get('critical_false_negative_rate')}) exceeds max ({thresholds['critical_fn_rate_max']})")

    if "per_class_recall_min" in thresholds and isinstance(model.get("per_class"), dict) and isinstance(labels, list):
        pcr_spec = thresholds["per_class_recall_min"]
        per_class = model["per_class"]
        if isinstance(pcr_spec, (int, float)) and not isinstance(pcr_spec, bool):
            for lbl in labels:
                rec = _as_rate((per_class.get(lbl) or {}).get("recall"))
                if rec is None or rec < float(pcr_spec):
                    missing_evidence.append("task_per_class_recall")
                    reasons.append(f"class {lbl} recall ({rec}) below minimum ({pcr_spec})")
                    break
        elif isinstance(pcr_spec, dict):
            for lbl, rmin in pcr_spec.items():
                rec = _as_rate((per_class.get(lbl) or {}).get("recall"))
                if rec is None or rec < float(rmin):
                    missing_evidence.append("task_per_class_recall")
                    reasons.append(f"class {lbl} recall ({rec}) below minimum ({rmin})")
                    break

    if "invalid_output_rate_max" in thresholds:
        ior_max = _as_rate(thresholds["invalid_output_rate_max"])
        ior_val = _as_rate(model.get("invalid_output_rate"))
        if ior_max is None or ior_val is None or ior_val > ior_max:
            missing_evidence.append("task_invalid_output_rate")
            reasons.append(f"model invalid_output_rate ({model.get('invalid_output_rate')}) exceeds max ({thresholds['invalid_output_rate_max']})")

    if "latency_p95_ms_max" in thresholds:
        lat_max = thresholds["latency_p95_ms_max"]
        lat_val = model.get("latency_p95_ms")
        if (
            isinstance(lat_max, bool)
            or not isinstance(lat_max, (int, float))
            or isinstance(lat_val, bool)
            or not isinstance(lat_val, (int, float))
            or not math.isfinite(float(lat_val))
            or float(lat_val) > float(lat_max)
        ):
            missing_evidence.append("task_latency_p95_ms")
            reasons.append(f"model latency_p95_ms ({lat_val}) missing or exceeds max ({lat_max})")

    if thresholds.get("ood_challenge_predictions"):
        ood_req = thresholds["ood_challenge_predictions"]
        ood_block = task_block.get("ood_challenge")
        if not isinstance(ood_block, dict):
            missing_evidence.append("ood_challenge_predictions")
            reasons.append("required OOD challenge predictions missing from run record")
        else:
            min_count = 1
            if isinstance(ood_req, int) and not isinstance(ood_req, bool):
                min_count = ood_req
            elif isinstance(ood_req, dict) and isinstance(ood_req.get("min_count"), int):
                min_count = ood_req["min_count"]
            if (ood_block.get("count") or 0) < min_count:
                missing_evidence.append("ood_challenge_predictions")
                reasons.append(f"OOD challenge count ({ood_block.get('count')}) below required ({min_count})")
            if isinstance(ood_req, dict) and ("min_accuracy" in ood_req or "macro_f1_min" in ood_req):
                req_acc = ood_req.get("min_accuracy", ood_req.get("macro_f1_min"))
                ood_acc = _as_rate(ood_block.get("accuracy"))
                if not isinstance(req_acc, (int, float)) or isinstance(req_acc, bool) or ood_acc is None or ood_acc < float(req_acc):
                    missing_evidence.append("ood_challenge_predictions")
                    reasons.append(
                        f"OOD challenge accuracy ({ood_block.get('accuracy')}) below required ({req_acc})"
                    )


def evaluate_promotion(
    run_id: str,
    *,
    output_root: str | Path | None = None,
    keyring: Any = None,
    expected_head: str | None = None,
    expected_count: int | None = None,
    tenant_context: TenantContext | None = None,
    allowed_data_origins: Sequence[str] | None = None,
) -> PromotionResult:
    """Evaluate fail-closed model promotion gate against required evidence and safety contracts.

    Checks existence, tamper-evident ledger verification, optional external anchor,
    optional tenant authorization, preflight completion, benchmark thresholds, SPDX
    licensing (run record or ledger attachment), and M-of-N Ed25519 promotion signatures
    bound to the ledger head and validated against a non-revoked trusted keyring.
    """
    clean_id = validate_run_id(run_id)
    reasons: list[str] = []
    missing_evidence: list[str] = []
    evidence_summary: dict[str, Any] = {}

    if tenant_context is not None and not isinstance(tenant_context, TenantContext):
        missing_evidence.append("tenant_authorization")
        reasons.append("invalid tenant_context")
        return _promotion_response(
            clean_id,
            promoted=False,
            reasons=reasons,
            missing_evidence=missing_evidence,
            evidence_summary=evidence_summary,
        )

    root_dir = Path(output_root or DEFAULT_RUNS_DIR).resolve()
    run_file = (root_dir / clean_id / "run.json").resolve()
    if not run_file.is_relative_to(root_dir):
        raise ValueError(f"run_id escapes output_root: {run_id!r}")

    # Gate 1: Run record must exist
    if not run_file.is_file():
        missing_evidence.append("run_metadata")
        reasons.append(f"run record not found for {clean_id}")
        return _promotion_response(
            clean_id,
            promoted=False,
            reasons=reasons,
            missing_evidence=missing_evidence,
            evidence_summary=evidence_summary,
        )

    try:
        inspection = inspect_run(
            clean_id,
            output_root=str(root_dir),
            expected_head=expected_head,
            expected_count=expected_count,
            tenant_context=tenant_context,
            _include_snapshot=True,
        )
        run_record = inspection["run_record"]
        run_sha256 = digest(run_record)  # Reject non-canonical values & hoist outside sig loop.
    except PermissionError as exc:
        missing_evidence.append("tenant_authorization")
        reasons.append(str(exc))
        return _promotion_response(
            clean_id,
            promoted=False,
            reasons=reasons,
            missing_evidence=missing_evidence,
            evidence_summary=evidence_summary,
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
        missing_evidence.append("run_inspection_failed")
        reasons.append(f"failed to inspect run: {exc}")
        return _promotion_response(
            clean_id,
            promoted=False,
            reasons=reasons,
            missing_evidence=missing_evidence,
            evidence_summary=evidence_summary,
        )

    evidence_summary["provenance_status"] = inspection["provenance_status"]
    evidence_summary["preflight_status"] = inspection["preflight_status"]
    if expected_head is not None or expected_count is not None:
        evidence_summary["anchor_verified"] = inspection.get("anchor_verified", False)

    # Gate 2: Tamper-evident evidence chain must be intact and match metadata (and anchor if supplied)
    if inspection["provenance_status"] != "PROVENANCE_VERIFIED":
        missing_evidence.append("tamper_evident_proof")
        reasons.append(f"provenance verification failed: {inspection['provenance_status']}")
        if inspection["provenance_status"] == "TAMPERED_ANCHOR_MISMATCH":
            missing_evidence.append("anchor_checkpoint")
            if inspection.get("anchor_error"):
                reasons.append(str(inspection["anchor_error"]))

    # Never consume attachment or signature records from an invalid ledger snapshot.
    ledger_rep = inspection.get("ledger_report")
    if isinstance(ledger_rep, dict) and ledger_rep.get("status") != "OK":
        return _promotion_response(
            clean_id,
            promoted=False,
            reasons=reasons,
            missing_evidence=missing_evidence,
            evidence_summary=evidence_summary,
        )

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
        f1 = _as_rate(eval_metrics.get("f1"))
        cfn_rate = _as_rate(eval_metrics.get("critical_false_negative_rate"))
        review_rate = _as_rate(eval_metrics.get("review_rate"))
        evidence_summary["evaluation_f1"] = eval_metrics.get("f1")
        evidence_summary["critical_fn_rate"] = eval_metrics.get("critical_false_negative_rate")

        if cfn_rate is None:
            missing_evidence.append("critical_case_coverage")
            reasons.append("evaluation lacks critical test cases (critical false negative rate is undefined)")
        elif cfn_rate != 0.0:
            missing_evidence.append("zero_critical_false_negatives")
            reasons.append(f"critical false negative rate ({cfn_rate}) exceeds zero tolerance")

        if review_rate is None or review_rate != 1.0:
            missing_evidence.append("mandatory_human_review_policy")
            reasons.append("evaluation does not enforce 100% human review policy")

        if f1 is None:
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
        _evaluate_task_thresholds(task_block, missing_evidence, reasons, evidence_summary)

    # Gate 5: Model manifest and SPDX licensing (run record or deduplicated ledger attachment)
    ledger_entries = inspection.get("ledger_entries") or []
    model_manifest = inspection.get("model_manifest")
    license_source = "run_record" if model_manifest is not None else None
    attached = None
    if model_manifest is None and inspection.get("resolved_ledger_path"):
        attached, stored_manifest, valid_att = extract_verified_license_attachment(ledger_entries, clean_id)
        if attached is not None:
            if valid_att and stored_manifest is not None:
                model_manifest = stored_manifest
                license_source = "evidence_ledger"
            else:
                missing_evidence.append("model_licensing_manifest")
                reasons.append("ledger license attachment failed self-verification")

    if model_manifest is None:
        if "model_licensing_manifest" not in missing_evidence:
            missing_evidence.append("model_licensing_manifest")
            reasons.append("no model licensing manifest provided for promotion")
    else:
        lic_result = _evaluate_license_compat(model_manifest, allowed_data_origins=allowed_data_origins)
        evidence_summary["licensing_decision"] = lic_result["decision"]
        evidence_summary["licensing_source"] = license_source
        if lic_result["decision"] != "ALLOWED":
            missing_evidence.append("approved_spdx_license")
            reasons.append(f"model licensing check failed: {'; '.join(lic_result['reasons'])}")

    # Gate 6: Cryptographic signed promotion authority (Ed25519 M-of-N quorum + revocation check)
    signing_mod = facade_attr("signing", signing)
    signature_entries = [
        entry
        for entry in ledger_entries
        if isinstance(entry, dict)
        and isinstance(entry.get("record"), dict)
        and entry["record"].get("type") == "promotion_signature"
        and entry["record"].get("run_id") == clean_id
    ]
    if not signature_entries:
        missing_evidence.append("cryptographic_promotion_signature")
        reasons.append("no promotion signature entry bound to this run in the verified ledger")
    elif not signing_mod.backend_available():
        missing_evidence.append("signing_backend")
        reasons.append("signing backend unavailable: install hardonian-foundry[signing]")
    elif keyring is None:
        missing_evidence.append("trusted_promotion_keyring")
        reasons.append("promotion requires a trusted keyring to validate signatures")
    else:
        policy: dict[str, Any] | None = None
        try:
            policy = parse_keyring(keyring)
        except (ValueError, TypeError) as exc:
            missing_evidence.append("trusted_promotion_keyring")
            reasons.append(f"invalid trusted keyring: {exc}")

        if policy is not None:
            key_by_hex = {k["public_key_hex"]: k for k in policy["keys"]}
            required_sigs = policy["required_signatures"]
            required_roles = set(policy["required_roles"])
            matched_entry = inspection.get("matched_entry")

            verified_entries_by_pub: dict[str, dict[str, Any]] = {}
            saw_revoked = False

            for entry in signature_entries:
                record = entry.get("record", {})
                signed_head = record.get("signed_head")
                pub = record.get("public_key_hex")
                if record.get("version") != "2" or record.get("run_sha256") != run_sha256:
                    reasons.append("signature does not bind the complete current run record")
                    continue
                if entry.get("prev_hash") != signed_head:
                    reasons.append("signature does not bind the ledger head at its signing position")
                    continue
                if (
                    matched_entry is None
                    or matched_entry["seq"] >= entry["seq"]
                    or (license_source == "evidence_ledger" and attached is not None and attached["seq"] >= entry["seq"])
                ):
                    reasons.append("signature does not cover the selected run and license evidence")
                    continue
                if not isinstance(pub, str) or pub not in key_by_hex:
                    reasons.append("signature key is not in the trusted keyring")
                    continue
                key_meta = key_by_hex[pub]
                if key_meta.get("revoked"):
                    saw_revoked = True
                    reasons.append(f"signature key {pub[:12]} has been revoked")
                    continue
                if pub in verified_entries_by_pub:
                    continue
                message = canonical({
                    "version": "2",
                    "run_id": clean_id,
                    "signed_head": signed_head,
                    "created_at": record.get("created_at"),
                    "run_sha256": run_sha256,
                }).encode("utf-8")
                if signing_mod.verify(message, record.get("signature_hex"), [pub]):
                    verified_entries_by_pub[pub] = entry

            verified_pubs = list(verified_entries_by_pub.keys())
            verified_roles = {
                key_by_hex[pub]["role"]
                for pub in verified_pubs
                if key_by_hex[pub].get("role")
            }
            quorum_met = len(verified_pubs) >= required_sigs
            roles_missing = sorted(required_roles - verified_roles)

            if verified_pubs and quorum_met and not roles_missing:
                sig_details = []
                for pub in verified_pubs:
                    ventry = verified_entries_by_pub[pub]
                    vrec = ventry["record"]
                    kmeta = key_by_hex[pub]
                    sig_details.append({
                        "entry_hash": ventry["entry_hash"],
                        "signed_head": vrec["signed_head"],
                        "public_key_hex": vrec["public_key_hex"],
                        "role": kmeta.get("role"),
                        "key_id": kmeta.get("key_id"),
                    })
                evidence_summary["promotion_signature"] = sig_details[0]
                evidence_summary["promotion_signatures"] = sig_details
                evidence_summary["quorum"] = {
                    "verified": len(verified_pubs),
                    "required": required_sigs,
                    "roles_verified": sorted(verified_roles),
                    "roles_required": sorted(required_roles),
                }
            else:
                missing_evidence.append("verified_promotion_signature")
                if saw_revoked and not verified_pubs:
                    missing_evidence.append("revoked_signing_key")
                if not quorum_met:
                    if len(verified_pubs) > 0:
                        missing_evidence.append("signature_quorum")
                        reasons.append(
                            f"promotion requires {required_sigs} distinct valid signatures, found {len(verified_pubs)}"
                        )
                    else:
                        reasons.append(
                            "no promotion signature verified against the trusted keyring; if the signature is legacy or predates license evidence, recreate the run, attach an allowed license, and sign the new run"
                        )
                if roles_missing:
                    missing_evidence.append("required_signer_roles")
                    reasons.append(f"promotion missing required signer roles: {', '.join(roles_missing)}")

    promoted = not missing_evidence
    return _promotion_response(
        clean_id,
        promoted=promoted,
        reasons=reasons,
        missing_evidence=missing_evidence,
        evidence_summary=evidence_summary,
    )
