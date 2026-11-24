"""Regulatory compliance control mapping, HTML audit dossiers, and policy-as-code gates."""
from __future__ import annotations

import html
import math
from typing import Any

from .canonical import digest

COMPLIANCE_PROFILES: tuple[str, ...] = (
    "eu-ai-act",
    "sr-11-7",
    "nist-ai-rmf",
    "iso-42001",
)

_PROFILE_CONTROLS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "eu-ai-act": (
        ("Art-9", "Risk Management System & Zero Critical FN Gate", "critical_fn_zero"),
        ("Art-10", "Data Governance, Split Disjointness & Licensing", "licensing_allowed"),
        ("Art-12", "Automatic Record-Keeping & Tamper-Evident Logging", "provenance_verified"),
        ("Art-14", "Mandatory Human Oversight & Review-Only Enforcement", "human_review"),
        ("Art-15", "Accuracy, Robustness & Cryptographic Provenance", "promotion_signed"),
    ),
    "sr-11-7": (
        ("SR11-7-Sec3", "Conceptual Soundness & Benchmark Separation", "f1_threshold"),
        ("SR11-7-Sec4", "Ongoing Monitoring, Zero Critical Misses & Drift Readiness", "critical_fn_zero"),
        ("SR11-7-Sec5", "Governance, Independent Review & Promotion Sign-Off", "promotion_signed"),
        ("SR11-7-Sec6", "Documentation, Audit Trail & Reproducibility", "provenance_verified"),
        ("SR11-7-Oversight", "Human Decision Authority (No Autonomous Execution)", "human_review"),
    ),
    "nist-ai-rmf": (
        ("GOVERN-1.2", "Roles, Keyring Authority & Accountable Sign-Off", "promotion_signed"),
        ("MAP-2.3", "License, Origin & Technical Boundaries Documented", "licensing_allowed"),
        ("MEASURE-2.5", "Safety & Critical False-Negative Verification", "critical_fn_zero"),
        ("MEASURE-2.7", "Benchmark Metric & Threshold Verification", "f1_threshold"),
        ("MANAGE-4.1", "Human-in-the-Loop Monitoring & Fail-Closed Escalation", "human_review"),
    ),
    "iso-42001": (
        ("A.5.4", "Auditability & Cryptographic Chain of Custody", "provenance_verified"),
        ("A.6.2", "System Lifecycle Verification & Quality Gates", "f1_threshold"),
        ("A.7.3", "Data Provenance & Licensing Allowlist Conformance", "licensing_allowed"),
        ("A.8.4", "Information for Users & Human Review Enforcement", "human_review"),
        ("A.9.3", "Pre-Release Management Approval & Signature Quorum", "promotion_signed"),
    ),
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_finite_float(value: Any) -> float | None:
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        fval = float(value)
        if math.isfinite(fval):
            return fval
    return None


def _check_control_condition(audit_report: dict[str, Any], check_key: str) -> tuple[str, str]:
    provenance_ok = audit_report.get("provenance_status") == "PROVENANCE_VERIFIED"
    gates = _as_dict(audit_report.get("gates"))
    eval_sum = _as_dict(audit_report.get("evaluation_summary"))
    missing = set(audit_report.get("missing_evidence") or [])

    if check_key == "provenance_verified":
        ok = provenance_ok and bool(audit_report.get("anchor_verified", False))
        return ("PASS" if ok else "FAIL"), f"provenance={audit_report.get('provenance_status')}, anchor_verified={audit_report.get('anchor_verified')}"

    if check_key == "human_review":
        rev = audit_report.get("review_required") is True and eval_sum.get("review_rate") == 1.0
        auto_num = _as_finite_float(eval_sum.get("coverage_automatic_decision"))
        auto_zero = eval_sum.get("coverage_automatic_decision") is None or auto_num == 0.0
        ok = rev and auto_zero
        return ("PASS" if ok else "FAIL"), f"review_required={audit_report.get('review_required')}, review_rate={eval_sum.get('review_rate')}, auto_decision={eval_sum.get('coverage_automatic_decision')}"

    if check_key == "critical_fn_zero":
        cfn = _as_finite_float(eval_sum.get("critical_false_negative_rate"))
        ok = cfn == 0.0 and "zero_critical_false_negatives" not in missing
        return ("PASS" if ok else "FAIL"), f"critical_false_negative_rate={cfn}"

    if check_key == "f1_threshold":
        f1 = _as_finite_float(eval_sum.get("f1"))
        ok = (
            f1 is not None
            and f1 >= 0.8
            and "minimum_f1_benchmark" not in missing
            and "task_macro_f1_threshold" not in missing
        )
        return ("PASS" if ok else "FAIL"), f"f1={f1}, preflight_status={gates.get('preflight_status')}"

    if check_key == "licensing_allowed":
        dec = gates.get("licensing_decision")
        ok = dec == "ALLOWED" and "approved_spdx_license" not in missing and "model_licensing_manifest" not in missing
        return ("PASS" if ok else "FAIL"), f"licensing_decision={dec}, source={gates.get('licensing_source')}"

    if check_key == "promotion_signed":
        sig_ok = (
            audit_report.get("promoted") is True
            and bool(gates.get("promotion_signature") or gates.get("promotion_signatures"))
        )
        status = "PASS" if sig_ok else ("REVIEW_REQUIRED" if audit_report.get("status") == "ALLOWED_REVIEW_REQUIRED" else "FAIL")
        return status, f"status={audit_report.get('status')}, promoted={audit_report.get('promoted')}"

    return "FAIL", "unknown control condition"


def evaluate_compliance_profile(audit_report: dict[str, Any], profile: str) -> dict[str, Any]:
    """Map a Foundry audit report to regulatory framework control checks (advisory only)."""
    if not isinstance(audit_report, dict):
        raise TypeError("audit_report must be a dict")
    if not isinstance(profile, str) or profile not in COMPLIANCE_PROFILES:
        supported = ", ".join(COMPLIANCE_PROFILES)
        raise ValueError(f"unknown compliance profile {profile!r}; expected one of: {supported}")

    controls: list[dict[str, Any]] = []
    passed = 0
    for control_id, title, condition_key in _PROFILE_CONTROLS[profile]:
        status, evidence_ref = _check_control_condition(audit_report, condition_key)
        if status == "PASS":
            passed += 1
        controls.append({
            "control_id": control_id,
            "title": title,
            "status": status,
            "evidence_ref": evidence_ref,
        })

    return {
        "profile": profile,
        "run_id": audit_report.get("run_id"),
        "all_controls_passed": passed == len(controls),
        "passed_count": passed,
        "total_controls": len(controls),
        "advisory_only": True,
        "review_required": True,
        "controls": controls,
    }


def render_audit_html(audit_report: dict[str, Any]) -> str:
    """Render a self-contained, XSS-safe single-file HTML5 compliance audit dossier."""
    if not isinstance(audit_report, dict):
        raise TypeError("audit_report must be a dict")

    def esc(val: Any) -> str:
        return html.escape("" if val is None else str(val), quote=True)

    run_id = esc(audit_report.get("run_id", "unknown"))
    status = str(audit_report.get("status", "UNKNOWN"))
    promoted = audit_report.get("promoted") is True
    badge_bg = "#137333" if promoted else "#b3261e"
    eval_sum = _as_dict(audit_report.get("evaluation_summary"))
    gates = _as_dict(audit_report.get("gates"))
    missing = audit_report.get("missing_evidence") or []
    reasons = audit_report.get("reasons") or []
    raw_compliance = audit_report.get("compliance")
    compliance = raw_compliance if isinstance(raw_compliance, dict) else None

    gate_rows = "\n".join(
        f"<tr><td><code>{esc(k)}</code></td><td><code>{esc(v)}</code></td></tr>"
        for k, v in sorted(gates.items())
    )
    missing_items = "".join(f"<li><code>{esc(item)}</code></li>" for item in missing) or "<li><em>None</em></li>"
    reason_items = "".join(f"<li>{esc(r)}</li>" for r in reasons) or "<li><em>None</em></li>"

    compliance_html = ""
    if compliance is not None:
        ctrl_rows = "\n".join(
            f"<tr><td><code>{esc(c.get('control_id'))}</code></td>"
            f"<td>{esc(c.get('title'))}</td>"
            f"<td><strong>{esc(c.get('status'))}</strong></td>"
            f"<td><code>{esc(c.get('evidence_ref'))}</code></td></tr>"
            for c in (compliance.get("controls") or [])
        )
        compliance_html = (
            f"<h2>Compliance Profile: {esc(compliance.get('profile'))} "
            f"({esc(compliance.get('passed_count'))}/{esc(compliance.get('total_controls'))} passed)</h2>"
            "<table><thead><tr><th>Control</th><th>Requirement</th><th>Status</th><th>Evidence</th></tr></thead>"
            f"<tbody>{ctrl_rows}</tbody></table>"
        )

    return "\n".join([
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>Foundry Audit Report — {run_id}</title>",
        "<style>",
        "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:960px;margin:2rem auto;padding:0 1.25rem;color:#1f2937;background:#f9fafb;line-height:1.5;}",
        "h1,h2{color:#111827;} .badge{display:inline-block;padding:0.25rem 0.7rem;border-radius:9999px;color:#fff;font-weight:600;font-size:0.875rem;}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:0.75rem;margin:1rem 0;}",
        ".card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:0.85rem;} .label{font-size:0.75rem;color:#6b7280;text-transform:uppercase;}",
        ".val{font-size:1.05rem;font-weight:600;word-break:break-all;} table{width:100%;border-collapse:collapse;background:#fff;margin:0.75rem 0;}",
        "th,td{border:1px solid #e5e7eb;padding:0.5rem 0.65rem;text-align:left;vertical-align:top;} th{background:#f3f4f6;}",
        "code{background:#f3f4f6;padding:0.1rem 0.3rem;border-radius:4px;font-size:0.85em;}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Hardonian Foundry Audit Dossier: <code>{run_id}</code></h1>",
        f'<p><span class="badge" style="background:{badge_bg}">{esc(status)}</span> '
        f"<strong>Mandatory Human Review:</strong> <code>review_required={esc(audit_report.get('review_required'))}</code></p>",
        '<div class="grid">',
        f'<div class="card"><div class="label">Macro-F1 / F1</div><div class="val">{esc(eval_sum.get("f1"))}</div></div>',
        f'<div class="card"><div class="label">Critical FN Rate</div><div class="val">{esc(eval_sum.get("critical_false_negative_rate"))}</div></div>',
        f'<div class="card"><div class="label">Provenance</div><div class="val">{esc(audit_report.get("provenance_status"))}</div></div>',
        f'<div class="card"><div class="label">Ledger Head</div><div class="val"><code>{esc(audit_report.get("ledger_head"))}</code></div></div>',
        "</div>",
        "<h2>Provenance &amp; Split Digests</h2>",
        "<table><tbody>",
        f"<tr><th>Dataset SHA-256</th><td><code>{esc(audit_report.get('dataset_sha256'))}</code></td></tr>",
        f"<tr><th>Test Split SHA-256</th><td><code>{esc(audit_report.get('test_sha256'))}</code></td></tr>",
        f"<tr><th>Anchor Verified</th><td><code>{esc(audit_report.get('anchor_verified'))}</code></td></tr>",
        f"<tr><th>Data Origin</th><td><code>{esc(eval_sum.get('data_origin'))}</code></td></tr>",
        "</tbody></table>",
        compliance_html,
        "<h2>Missing / Blocking Evidence</h2>",
        f"<ul>{missing_items}</ul>",
        "<h2>Gate Reasons</h2>",
        f"<ul>{reason_items}</ul>",
        "<h2>Evidence Summary</h2>",
        f"<table><thead><tr><th>Key</th><th>Value</th></tr></thead><tbody>{gate_rows}</tbody></table>",
        "</body>",
        "</html>",
    ])


def evaluate_policy_bundle(run_record: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a declarative policy-as-code bundle fail-closed against a run record."""
    if not isinstance(run_record, dict) or not isinstance(policy, dict):
        raise TypeError("run_record and policy must be dicts")

    allowed_fields = {
        "schema_version",
        "policy_id",
        "min_macro_f1",
        "max_critical_fn_rate",
        "max_latency_p95_ms",
        "min_slice_recall",
        "require_zero_false_tool_exec",
        "allowed_data_origins",
        "notes",
    }
    unknown = sorted(set(policy) - allowed_fields)
    if unknown:
        raise ValueError(f"unknown policy bundle fields: {', '.join(unknown)}")

    eval_obj = _as_dict(run_record.get("evaluation"))
    task_block = _as_dict(eval_obj.get("task"))
    model_metrics = _as_dict(task_block.get("model_metrics"))
    violations: list[str] = []

    if "min_macro_f1" in policy:
        req_f1 = _as_finite_float(policy["min_macro_f1"])
        if req_f1 is None or not (0.0 <= req_f1 <= 1.0):
            raise ValueError("min_macro_f1 must be in [0.0, 1.0]")
        actual_f1 = _as_finite_float(eval_obj.get("f1"))
        if actual_f1 is None or actual_f1 < req_f1:
            violations.append(f"f1 ({eval_obj.get('f1')}) below policy min_macro_f1 ({req_f1})")

    if "max_critical_fn_rate" in policy:
        max_cfn = _as_finite_float(policy["max_critical_fn_rate"])
        if max_cfn is None or not (0.0 <= max_cfn <= 1.0):
            raise ValueError("max_critical_fn_rate must be in [0.0, 1.0]")
        actual_cfn = _as_finite_float(eval_obj.get("critical_false_negative_rate"))
        if actual_cfn is None or actual_cfn > max_cfn:
            violations.append(f"critical_false_negative_rate ({eval_obj.get('critical_false_negative_rate')}) exceeds policy max ({max_cfn})")

    if "max_latency_p95_ms" in policy:
        max_lat = _as_finite_float(policy["max_latency_p95_ms"])
        if max_lat is None or max_lat <= 0.0:
            raise ValueError("max_latency_p95_ms must be > 0")
        raw_lat = eval_obj.get("latency_p95_ms")
        if raw_lat is None:
            raw_lat = model_metrics.get("latency_p95_ms")
        actual_lat = _as_finite_float(raw_lat)
        if actual_lat is None or actual_lat > max_lat:
            violations.append(f"latency_p95_ms ({raw_lat}) missing or exceeds policy max ({max_lat})")

    if "min_slice_recall" in policy:
        min_slice = _as_finite_float(policy["min_slice_recall"])
        if min_slice is None or not (0.0 <= min_slice <= 1.0):
            raise ValueError("min_slice_recall must be in [0.0, 1.0]")
        per_class = _as_dict(model_metrics.get("per_class"))
        if not per_class:
            violations.append("per_class slice metrics missing for min_slice_recall check")
        else:
            for lbl, stats in sorted(per_class.items()):
                rec = _as_finite_float(stats.get("recall") if isinstance(stats, dict) else None)
                if rec is None or rec < min_slice:
                    violations.append(f"slice {lbl} recall ({rec}) below policy min_slice_recall ({min_slice})")

    if policy.get("require_zero_false_tool_exec"):
        traj = run_record.get("trajectory_eval")
        if not isinstance(traj, dict) or traj.get("false_tool_execution_count") != 0 or traj.get("passed") is not True:
            violations.append("trajectory_eval missing or failed zero false tool execution check")

    if "allowed_data_origins" in policy:
        origins = policy["allowed_data_origins"]
        if not isinstance(origins, list) or not origins or any(not isinstance(o, str) for o in origins):
            raise ValueError("allowed_data_origins must be a non-empty list of strings")
        origin = eval_obj.get("data_origin")
        if origin not in set(origins):
            violations.append(f"data_origin ({origin!r}) not in allowed_data_origins ({origins})")

    return {
        "policy_id": policy.get("policy_id", "foundry-policy"),
        "passed": not violations,
        "status": "PASSED" if not violations else "BLOCKED",
        "violations": violations,
        "policy_sha256": digest(policy),
        "review_required": True,
    }
