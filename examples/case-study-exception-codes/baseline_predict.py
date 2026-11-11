#!/usr/bin/env python3
"""Baseline predictor: a faithful Python port of Settler's first-match-wins
keyword rules (reconciliation-core/src/exception-intelligence.ts, same owner).

Reads the frozen spec + dataset, recomputes the deterministic group-disjoint
test split, and writes baseline.jsonl predictions. No rule matches ->
UNDETERMINED (the honest recall gap; Settler leaves such items for review).
"""
import argparse
import json
import re
from pathlib import Path

# Ported verbatim (semantics) from Settler KEYWORD_RULES, first-match-wins order.
KEYWORD_RULES = [
    ("DUPLICATE_RECORD_CONFIRMED",
     re.compile(r"\bduplicate\b|\bduplicated\b|\btwice\b|\bdouble(?:\s|-)?count(?:ed)?\b|\breimport(?:ed)?\b", re.IGNORECASE)),
    ("TIMING_DIFFERENCE_ACCEPTED",
     re.compile(r"\btiming\b|\bdate drift\b|\bdate difference\b|\bsettlement window\b|\bsettlement delay\b|\bcutoff\b", re.IGNORECASE)),
    ("AMOUNT_VARIANCE_ACCEPTED",
     re.compile(r"\bamount\b|\bvariance\b|\bfee\b|\btax\b|\bround(?:ing)?\b", re.IGNORECASE)),
    ("CURRENCY_VARIANCE_ACCEPTED",
     re.compile(r"\bcurrency\b|\bfx\b|\bforeign exchange\b|\bexchange rate\b", re.IGNORECASE)),
    ("MISSING_COUNTERPART_CONFIRMED",
     re.compile(r"\bmissing\b|\bno counterpart\b|\bnot found\b|\bnot posted\b|\bpending capture\b", re.IGNORECASE)),
    ("FALSE_POSITIVE_CONFIRMED",
     re.compile(r"\bfalse positive(?:s)?\b|\bnot actionable\b|\bexpected gap\b|\bbenign\b|\bexpected\b", re.IGNORECASE)),
    ("REOPENED_FOR_INVESTIGATION",
     re.compile(r"\bre-?open(?:ed)?\b|\b[i]nvestigat(?:e|ion)\b|\bfollow(?:\s|-)?up\b|\b[e]scalat", re.IGNORECASE)),
]


def predict(note):
    for code, pattern in KEYWORD_RULES:
        if pattern.search(note):
            return code
    return "UNDETERMINED"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", required=True, help="directory with spec.json and data.jsonl")
    parser.add_argument("--out", required=True, help="output predictions JSONL")
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    from hardonian_foundry.task_eval import split_task_dataset

    spec = json.loads((task_dir / "spec.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (task_dir / "data.jsonl").read_text(encoding="utf-8").splitlines() if line]
    test = split_task_dataset(rows, spec)["test"]

    with Path(args.out).open("w", encoding="utf-8") as fh:
        fh.writelines(json.dumps({"id": row["id"], "label": predict(row["prompt"])}, sort_keys=True) + "\n" for row in test)

    correct = sum(1 for row in test if predict(row["prompt"]) == row["gold_label"])
    crit_total = sum(1 for row in test if row["critical"])
    crit_missed = sum(1 for row in test if row["critical"] and predict(row["prompt"]) != row["gold_label"])
    print(json.dumps({"test_records": len(test), "baseline_top1": round(correct / len(test), 4),
                      "critical_total": crit_total, "critical_missed": crit_missed,
                      "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
