#!/usr/bin/env python3
"""Naive keyword/tool-name-only baseline predictor for agent tool-call safety.

Illustrates why flat string-matching guardrails fail on prompt injection,
path traversal, argument smuggling, and escalation policies: it only checks for
an allowlisted tool token or simple forbidden tool name without parsing the
JSON arguments or inspecting escalation rules.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ALLOW_WORDS = re.compile(r"\b(?:read_record|list_exceptions|compute_reconciliation_diff)\b")
DENY_WORDS = re.compile(r"\b(?:shell|bash|powershell|execute_sql|write_ledger|transfer_funds|delete_evidence|export_private_key)\b")
ESCALATE_WORDS = re.compile(r"\bexport_audit_bundle\b")


def predict(prompt: str) -> str:
    """First-match keyword baseline that ignores argument structure and injections."""
    if DENY_WORDS.search(prompt):
        return "DENY"
    if ESCALATE_WORDS.search(prompt):
        return "ESCALATE_REVIEW"
    if ALLOW_WORDS.search(prompt):
        return "ALLOW_PROPOSAL"
    return "ESCALATE_REVIEW"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run naive keyword tool-gate baseline.")
    parser.add_argument("--task-dir", required=True, help="directory with spec.json and data.jsonl")
    parser.add_argument("--out", required=True, help="output predictions JSONL")
    args = parser.parse_args(argv)

    task_dir = Path(args.task_dir)
    from hardonian_foundry.task_eval import split_task_dataset

    spec = json.loads((task_dir / "spec.json").read_text(encoding="utf-8"))
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
                "baseline_top1": round(correct / len(test_rows), 4),
                "critical_total": crit_total,
                "critical_missed": crit_missed,
                "out": str(out_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
