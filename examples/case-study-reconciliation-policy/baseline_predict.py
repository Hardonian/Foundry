#!/usr/bin/env python3
"""Naive baseline that trusts an untrusted model suggestion without arithmetic checks.

Demonstrates what happens when a model's output is accepted directly without a
deterministic policy gate: 1-cent rounding errors, sign flips, float-precision
truncations, currency mismatches, and missing fields slip through as ``matched``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ALLOWED_LABELS = {"matched", "amount_mismatch", "review_required"}


def extract_suggestion(prompt: str) -> Any:
    """Extract the raw Untrusted Model Suggestion value from a prompt."""
    for line in prompt.splitlines():
        if line.startswith("Untrusted Model Suggestion:"):
            return json.loads(line[len("Untrusted Model Suggestion:") :].strip())
    return None


def predict(prompt: str) -> str:
    """Trust the untrusted suggestion whenever it matches an allowlisted label."""
    suggestion = extract_suggestion(prompt)
    if isinstance(suggestion, str) and suggestion in ALLOWED_LABELS:
        return suggestion
    return "review_required"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run untrusted-suggestion baseline predictor.")
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
