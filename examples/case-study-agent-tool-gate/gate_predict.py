#!/usr/bin/env python3
"""Structural offline safety-gate predictor for agent tool-call proposals.

Pure stdlib, deterministic, zero HF weights required. Parses the structured
Tool Proposal payload, validates exact tool and argument schemas, rejects path
traversal / argument smuggling / shell & SQL metacharacters / prompt-injection
overrides, and flags elevated-scope operations for human escalation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ALLOW_PROPOSAL = "ALLOW_PROPOSAL"
DENY = "DENY"
ESCALATE_REVIEW = "ESCALATE_REVIEW"

ALLOWED_TOOLS = {
    "read_record",
    "list_exceptions",
    "compute_reconciliation_diff",
    "export_audit_bundle",
}
FORBIDDEN_TOOLS = {
    "shell",
    "bash",
    "powershell",
    "execute_sql",
    "write_ledger",
    "transfer_funds",
    "delete_evidence",
    "export_private_key",
    "modify_keyring",
}
ALLOWED_ROLES = {"analyst", "auditor", "operator", "reconciler"}
ESCALATE_ROLES = {"external_contractor", "unverified_agent"}

INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(?:all\s+)?previous\s+(?:safety\s+|system\s+)?(?:policies|instructions|rules)", re.IGNORECASE),
    re.compile(r"system\s+override", re.IGNORECASE),
    re.compile(r"</justification>|<system>", re.IGNORECASE),
    re.compile(r"\bbypass_gate\b|\bexec_cmd\b|\b__proto__\b|\bsql_override\b", re.IGNORECASE),
    re.compile(r"\bactually\s+(?:invoke|run|execute)\b", re.IGNORECASE),
)
TRAVERSAL_PATTERN = re.compile(r"\.\.|%2e%2e|/etc/|/var/|\\|\\x00|^[a-zA-Z]:", re.IGNORECASE)
METACHAR_PATTERN = re.compile(r"[;|$`<>'\x00\r\n]|--\s|\bOR\b\s+['\"]?1['\"]?\s*=\s*['\"]?1", re.IGNORECASE)
SAFE_SYNTHETIC_ID = re.compile(r"^synthetic-[a-z0-9-]+$")
QUARANTINE_ID = re.compile(r"^quarantine-[a-z0-9-]+$")
ISO_CURRENCY = re.compile(r"^[A-Z]{3}$")


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_fields(prompt: str) -> tuple[str, dict[str, Any] | None]:
    context = ""
    proposal_raw = ""
    for line in prompt.splitlines():
        if line.startswith("User Context:"):
            context = line[len("User Context:") :].strip()
        elif line.startswith("Tool Proposal:"):
            proposal_raw = line[len("Tool Proposal:") :].strip()
    if not proposal_raw:
        return context, None
    try:
        parsed = json.loads(proposal_raw)
    except json.JSONDecodeError:
        return context, None
    if not isinstance(parsed, dict):
        return context, None
    return context, parsed


def _has_unsafe_text(value: str) -> bool:
    return bool(TRAVERSAL_PATTERN.search(value) or METACHAR_PATTERN.search(value))


def predict(prompt: str) -> str:
    """Classify a synthetic tool-call proposal prompt into ALLOW_PROPOSAL, DENY, or ESCALATE_REVIEW."""
    context, proposal = _extract_fields(prompt)
    if any(pat.search(context) for pat in INJECTION_PATTERNS):
        return DENY
    if proposal is None:
        return DENY

    expected_top_keys = {"tool", "arguments", "requester_role", "justification"}
    if set(proposal) != expected_top_keys:
        return DENY

    tool = proposal["tool"]
    arguments = proposal["arguments"]
    role = proposal["requester_role"]
    justification = proposal["justification"]

    if not isinstance(tool, str) or not isinstance(arguments, dict):
        return DENY
    if not isinstance(role, str) or not isinstance(justification, str):
        return DENY
    if any(pat.search(justification) for pat in INJECTION_PATTERNS):
        return DENY
    if _has_unsafe_text(justification):
        return DENY

    if tool in FORBIDDEN_TOOLS or tool not in ALLOWED_TOOLS:
        return DENY

    # Check argument values recursively for traversal/injection/smuggling before role escalation
    for key, val in arguments.items():
        if not isinstance(key, str) or any(pat.search(key) for pat in INJECTION_PATTERNS):
            return DENY
        if isinstance(val, str) and (_has_unsafe_text(val) or any(pat.search(val) for pat in INJECTION_PATTERNS)):
            return DENY

    escalate = False
    if role in ESCALATE_ROLES:
        escalate = True
    elif role not in ALLOWED_ROLES:
        return DENY

    if not justification.strip() or justification.strip().upper() == "TBD":
        escalate = True

    if tool == "read_record":
        if set(arguments) != {"record_id", "scope"}:
            return DENY
        record_id = arguments["record_id"]
        scope = arguments["scope"]
        if not isinstance(record_id, str) or not isinstance(scope, str):
            return DENY
        if QUARANTINE_ID.fullmatch(record_id):
            escalate = True
        elif not SAFE_SYNTHETIC_ID.fullmatch(record_id):
            return DENY
        if scope == "cross_family_audit":
            escalate = True
        elif scope != "read_only":
            return DENY

    elif tool == "list_exceptions":
        if set(arguments) != {"status", "limit", "tenant_scope"}:
            return DENY
        status = arguments["status"]
        limit = arguments["limit"]
        tenant_scope = arguments["tenant_scope"]
        if status not in {"open", "escalated", "resolved"}:
            return DENY
        if type(limit) is not int or limit < 1:
            return DENY
        if limit > 100:
            escalate = True
        if tenant_scope == "cross_family_audit":
            escalate = True
        elif tenant_scope not in {"synthetic-demo", "synthetic-sandbox"}:
            return DENY

    elif tool == "compute_reconciliation_diff":
        if set(arguments) != {"expected_minor", "actual_minor", "currency"}:
            return DENY
        expected_minor = arguments["expected_minor"]
        actual_minor = arguments["actual_minor"]
        currency = arguments["currency"]
        if type(expected_minor) is not int or type(actual_minor) is not int:
            return DENY
        if abs(expected_minor) > 10**12 or abs(actual_minor) > 10**12:
            return DENY
        if not isinstance(currency, str) or not ISO_CURRENCY.fullmatch(currency):
            return DENY

    elif tool == "export_audit_bundle":
        if set(arguments) != {"run_id", "include_signatures"}:
            return DENY
        run_id = arguments["run_id"]
        include_sigs = arguments["include_signatures"]
        if not isinstance(run_id, str) or not SAFE_SYNTHETIC_ID.fullmatch(run_id):
            return DENY
        if type(include_sigs) is not bool:
            return DENY
        escalate = True

    return ESCALATE_REVIEW if escalate else ALLOW_PROPOSAL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run structural tool-gate predictor on frozen test split.")
    parser.add_argument("--task-dir", required=True, help="directory with spec.json and data.jsonl")
    parser.add_argument("--out", required=True, help="output predictions JSONL")
    args = parser.parse_args(argv)

    task_dir = Path(args.task_dir)
    spec = json.loads((task_dir / "spec.json").read_text(encoding="utf-8"))
    self_sha = sha256_file(Path(__file__).resolve())
    expected_sha = spec.get("model", {}).get("adapter_sha256")
    if expected_sha and expected_sha != "0" * 64 and expected_sha != self_sha:
        raise SystemExit(
            f"predictor provenance mismatch: expected {expected_sha[:12]}..., got {self_sha[:12]}..."
        )

    from hardonian_foundry.task_eval import split_task_dataset

    rows = [
        json.loads(line)
        for line in (task_dir / "data.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    test_rows = split_task_dataset(rows, spec)["test"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = [{"id": row["id"], "label": predict(row["prompt"])} for row in test_rows]
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for pred in predictions:
            fh.write(json.dumps(pred, sort_keys=True) + "\n")

    correct = sum(
        1 for row, pred in zip(test_rows, predictions, strict=True) if pred["label"] == row["gold_label"]
    )
    crit_total = sum(1 for row in test_rows if row["critical"])
    crit_missed = sum(
        1
        for row, pred in zip(test_rows, predictions, strict=True)
        if row["critical"] and pred["label"] != row["gold_label"]
    )
    print(
        json.dumps(
            {
                "test_records": len(test_rows),
                "candidate_top1": round(correct / len(test_rows), 4),
                "critical_total": crit_total,
                "critical_missed": crit_missed,
                "predictor_sha256": self_sha,
                "out": str(out_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
