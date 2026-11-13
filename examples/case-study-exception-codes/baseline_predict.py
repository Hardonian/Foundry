#!/usr/bin/env python3
"""Baseline predictor: a faithful Python port of Settler's first-match-wins
keyword rules (reconciliation-core/src/exception-intelligence.ts, same owner).

Reads the frozen spec + dataset, recomputes the deterministic group-disjoint
test split, and writes baseline.jsonl predictions. No rule matches ->
UNDETERMINED (the honest recall gap; Settler leaves such items for review).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from hardonian_foundry.task_eval import split_task_dataset, validate_spec
except ImportError:
    sys.modules.pop("hardonian_foundry", None)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from hardonian_foundry.task_eval import split_task_dataset, validate_spec

# Ported verbatim (semantics) from Settler KEYWORD_RULES, first-match-wins order.
KEYWORD_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "DUPLICATE_RECORD_CONFIRMED",
        re.compile(
            r"\bduplicate\b|\bduplicated\b|\btwice\b|\bdouble(?:\s|-)?count(?:ed)?\b|\breimport(?:ed)?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "TIMING_DIFFERENCE_ACCEPTED",
        re.compile(
            r"\btiming\b|\bdate drift\b|\bdate difference\b|\bsettlement window\b|\bsettlement delay\b|\bcutoff\b",
            re.IGNORECASE,
        ),
    ),
    (
        "AMOUNT_VARIANCE_ACCEPTED",
        re.compile(r"\bamount\b|\bvariance\b|\bfee\b|\btax\b|\bround(?:ing)?\b", re.IGNORECASE),
    ),
    (
        "CURRENCY_VARIANCE_ACCEPTED",
        re.compile(r"\bcurrency\b|\bfx\b|\bforeign exchange\b|\bexchange rate\b", re.IGNORECASE),
    ),
    (
        "MISSING_COUNTERPART_CONFIRMED",
        re.compile(
            r"\bmissing\b|\bno counterpart\b|\bnot found\b|\bnot posted\b|\bpending capture\b",
            re.IGNORECASE,
        ),
    ),
    (
        "FALSE_POSITIVE_CONFIRMED",
        re.compile(
            r"\bfalse positive(?:s)?\b|\bnot actionable\b|\bexpected gap\b|\bbenign\b|\bexpected\b",
            re.IGNORECASE,
        ),
    ),
    (
        "REOPENED_FOR_INVESTIGATION",
        re.compile(
            r"\bre-?open(?:ed)?\b|\b[i]nvestigat(?:e|ion)\b|\bfollow(?:\s|-)?up\b|\b[e]scalat",
            re.IGNORECASE,
        ),
    ),
]


def predict(note: str) -> str:
    """Classify an operator note using first-match-wins keyword rules."""
    if not isinstance(note, str):
        raise TypeError("note must be a string")
    for code, pattern in KEYWORD_RULES:
        if pattern.search(note):
            return code
    return "UNDETERMINED"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True, type=Path, help="directory with spec.json and data.jsonl")
    parser.add_argument("--out", required=True, type=Path, help="output predictions JSONL")
    parser.add_argument("--json", action="store_true", help="emit compact JSON summary / structured error")
    args = parser.parse_args(argv)

    task_dir = args.task_dir
    spec_path = task_dir / "spec.json"
    data_path = task_dir / "data.jsonl"
    try:
        if not spec_path.is_file() or not data_path.is_file():
            raise FileNotFoundError(f"missing spec.json or data.jsonl in {task_dir}")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec_errors = validate_spec(spec)
        if spec_errors:
            raise ValueError(f"invalid task spec: {'; '.join(spec_errors)}")
        rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"no records found in {data_path}")
        test = split_task_dataset(rows, spec)["test"]
        if not test:
            raise ValueError("test split is empty")

        preds = [(row, predict(row["prompt"])) for row in test]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            for row, label in preds:
                fh.write(json.dumps({"id": row["id"], "label": label}, sort_keys=True) + "\n")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        if args.json:
            print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        else:
            print(f"baseline_predict error: {exc}", file=sys.stderr)
        return 1

    correct = sum(1 for row, label in preds if label == row["gold_label"])
    crit_total = sum(1 for row, _ in preds if row.get("critical"))
    crit_missed = sum(1 for row, label in preds if row.get("critical") and label != row["gold_label"])
    summary = {
        "status": "completed",
        "test_records": len(test),
        "baseline_top1": round(correct / len(test), 4),
        "critical_total": crit_total,
        "critical_missed": crit_missed,
        "out": str(args.out),
    }
    print(json.dumps(summary, indent=2 if not args.json else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
