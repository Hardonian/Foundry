#!/usr/bin/env python3
"""Deterministic policy-override predictor powered by synthetic_policy.assess_with_suggestion.

Demonstrates Foundry's core safety invariant: an untrusted model suggestion can
be audited for agreement (model_agrees, model_status) but can NEVER alter the
deterministic minor-unit arithmetic category or bypass human review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from hardonian_foundry.synthetic_policy import assess_with_suggestion


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_prompt(prompt: str) -> tuple[dict[str, Any], Any]:
    """Extract structured Case JSON and Untrusted Model Suggestion from prompt text."""
    case_obj: dict[str, Any] | None = None
    suggestion: Any = None
    for line in prompt.splitlines():
        if line.startswith("Case JSON:"):
            case_obj = json.loads(line[len("Case JSON:") :].strip())
        elif line.startswith("Untrusted Model Suggestion:"):
            suggestion = json.loads(line[len("Untrusted Model Suggestion:") :].strip())
    if case_obj is None:
        raise ValueError("prompt missing Case JSON line")
    return case_obj, suggestion


def predict_with_audit(prompt: str) -> dict[str, Any]:
    """Evaluate deterministic arithmetic policy alongside the untrusted suggestion."""
    case_obj, suggestion = parse_prompt(prompt)
    return assess_with_suggestion(case_obj, suggestion)


def predict(prompt: str) -> str:
    """Return the deterministic policy category, never overridden by untrusted suggestions."""
    return str(predict_with_audit(prompt)["category"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic policy-override predictor.")
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

    predictions: list[dict[str, str]] = []
    agrees = 0
    overridden = 0
    unavailable = 0
    all_review_required = True
    any_financial_write = False

    for row in test_rows:
        audit = predict_with_audit(row["prompt"])
        predictions.append({"id": row["id"], "label": audit["category"]})
        if audit["model_status"] == "UNAVAILABLE":
            unavailable += 1
            overridden += 1
        elif audit["model_agrees"] is True:
            agrees += 1
        else:
            overridden += 1
        all_review_required = all_review_required and bool(audit["review_required"])
        any_financial_write = any_financial_write or bool(audit["financial_write"])

    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for pred in predictions:
            fh.write(json.dumps(pred, sort_keys=True) + "\n")

    correct = sum(
        1 for row, pred in zip(test_rows, predictions, strict=True) if pred["label"] == row["gold_label"]
    )
    crit_missed = sum(
        1
        for row, pred in zip(test_rows, predictions, strict=True)
        if row["critical"] and pred["label"] != row["gold_label"]
    )
    print(
        json.dumps(
            {
                "test_records": len(test_rows),
                "policy_top1": round(correct / len(test_rows), 4),
                "critical_missed": crit_missed,
                "untrusted_model_agreed": agrees,
                "untrusted_model_overridden": overridden,
                "untrusted_model_unavailable": unavailable,
                "all_review_required": all_review_required,
                "any_financial_write": any_financial_write,
                "predictor_sha256": self_sha,
                "out": str(out_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
