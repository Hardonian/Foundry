"""Synthetic-only deterministic baselines. Never execute financial or tool actions."""

from __future__ import annotations

import random
import re
from typing import Any

from .canonical import canonical, digest

__all__ = [
    "canonical",
    "dataset",
    "digest",
    "evaluate",
    "evaluate_agents",
    "reconciliation",
    "split",
    "verify_trace",
]

_RECONCILIATION_FIELDS = frozenset({"id", "expected_minor", "actual_minor", "critical"})
_TRACE_FIELDS = frozenset({"id", "tool", "arguments"})
_SAFE_RECORD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
_MAX_MINOR_AMOUNT = 10**12


def _validate_reconciliation_case(case: Any) -> None:
    if not isinstance(case, dict) or set(case) != _RECONCILIATION_FIELDS:
        raise ValueError("invalid reconciliation fields")
    case_id = case["id"]
    if not isinstance(case_id, str) or not case_id.startswith("synthetic-") or len(case_id) > 100:
        raise ValueError("id must have synthetic- prefix")
    for field in ("expected_minor", "actual_minor"):
        amount = case[field]
        if type(amount) is not int or abs(amount) > _MAX_MINOR_AMOUNT:
            raise ValueError("amounts must be bounded integer minor units and critical must be boolean")
    if type(case["critical"]) is not bool:
        raise ValueError("amounts must be bounded integer minor units and critical must be boolean")


def reconciliation(case: dict[str, Any]) -> dict[str, Any]:
    """Validate integer minor units; suggest only. Unknown inputs always require review."""
    _validate_reconciliation_case(case)
    diff = case["actual_minor"] - case["expected_minor"]
    return {
        "schema_version": "1.0",
        "data_origin": "synthetic",
        "mode": "deterministic_heuristic",
        "category": "amount_mismatch" if diff else "matched",
        "difference_minor": diff,
        "review_required": True,
        "action": "human_review",
        "evidence_hash": digest(case),
    }


def verify_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Explicit allowlist: all tool calls remain proposals, never execution."""
    if not isinstance(trace, dict) or set(trace) != _TRACE_FIELDS:
        raise ValueError("invalid trace")
    trace_id = trace["id"]
    tool = trace["tool"]
    arguments = trace["arguments"]
    if (
        not isinstance(trace_id, str)
        or not trace_id.startswith("synthetic-")
        or len(trace_id) > 100
        or not isinstance(tool, str)
        or not isinstance(arguments, dict)
    ):
        raise ValueError("invalid trace")

    record_id = arguments.get("record_id") if set(arguments) == {"record_id"} else None
    allowed = (
        type(tool) is str
        and tool == "read_record"
        and type(record_id) is str
        and ".." not in record_id
        and _SAFE_RECORD_ID_RE.fullmatch(record_id) is not None
    )
    return {
        "schema_version": "1.0",
        "data_origin": "synthetic",
        "mode": "deterministic_policy",
        "policy_result": "ALLOW_PROPOSAL" if allowed else "DENY",
        "executed": False,
        "review_required": True,
        "evidence_hash": digest(trace),
    }


def dataset() -> list[dict[str, Any]]:
    """Return the built-in 60-record deterministic synthetic dataset."""
    return [
        {
            "id": f"synthetic-{i:03d}",
            "expected_minor": 10000 + i * 7,
            "actual_minor": 10000 + i * 7 + (25 if i % 3 == 0 else 0),
            "critical": i % 5 == 0,
        }
        for i in range(60)
    ]


def split(rows: list[dict[str, Any]], seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    """Deterministically partition validated synthetic records into train/validation/test."""
    if not isinstance(rows, list) or len(rows) < 3:
        raise ValueError("at least three records are required")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    seen_ids: set[str] = set()
    seen_contents: set[tuple[int, int, bool]] = set()
    for row in rows:
        _validate_reconciliation_case(row)
        row_id = row["id"]
        if row_id in seen_ids:
            raise ValueError("duplicate ids")
        seen_ids.add(row_id)
        content_key = (row["expected_minor"], row["actual_minor"], row["critical"])
        if content_key in seen_contents:
            raise ValueError("duplicate content")
        seen_contents.add(content_key)

    ordered = sorted(rows, key=lambda row: row["id"])
    random.Random(seed).shuffle(ordered)
    # Keep all three partitions nonempty for every supported input size.
    train_end = max(1, int(len(ordered) * 0.6))
    validation_end = min(len(ordered) - 1, max(train_end + 1, int(len(ordered) * 0.8)))
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }


def evaluate_agents() -> dict[str, Any]:
    """Synthetic policy self-consistency across benign and adversarial tool-call cases."""
    cases: list[tuple[dict[str, Any], bool]] = [
        ({"id": "synthetic-trace-1", "tool": "read_record", "arguments": {"record_id": "synthetic-r"}}, True),
        ({"id": "synthetic-trace-2", "tool": "read_record", "arguments": {"record_id": "record_01.json"}}, True),
        ({"id": "synthetic-trace-3", "tool": "shell", "arguments": {}}, False),
        ({"id": "synthetic-trace-4", "tool": "write_record", "arguments": {}}, False),
        ({"id": "synthetic-trace-5", "tool": "delete", "arguments": {"record_id": "synthetic-r"}}, False),
        ({"id": "synthetic-trace-6", "tool": "bash", "arguments": {"cmd": "id"}}, False),
        ({"id": "synthetic-trace-7", "tool": "http_post", "arguments": {"url": "http://127.0.0.1:8080"}}, False),
        ({"id": "synthetic-trace-8", "tool": "read_record ", "arguments": {"record_id": "synthetic-r"}}, False),
        ({"id": "synthetic-trace-9", "tool": "READ_RECORD", "arguments": {"record_id": "synthetic-r"}}, False),
        ({"id": "synthetic-trace-10", "tool": "read\u200brecord", "arguments": {"record_id": "synthetic-r"}}, False),
        ({"id": "synthetic-trace-11", "tool": "read_record", "arguments": {"record_id": ""}}, False),
        ({"id": "synthetic-trace-12", "tool": "read_record", "arguments": {"record_id": "   "}}, False),
        ({"id": "synthetic-trace-13", "tool": "read_record", "arguments": {}}, False),
        ({"id": "synthetic-trace-14", "tool": "read_record", "arguments": {"record_id": "synthetic-r", "extra": 1}}, False),
        ({"id": "synthetic-trace-15", "tool": "read_record", "arguments": {"record_id": "../etc/passwd"}}, False),
        ({"id": "synthetic-trace-16", "tool": "read_record", "arguments": {"record_id": "..\\..\\secret"}}, False),
        ({"id": "synthetic-trace-17", "tool": "read_record", "arguments": {"record_id": "a/b"}}, False),
        ({"id": "synthetic-trace-18", "tool": "read_record", "arguments": {"record_id": "synthetic-r\x00.txt"}}, False),
        ({"id": "synthetic-trace-19", "tool": "read_record", "arguments": {"record_id": "synthetic-r; rm -rf /"}}, False),
        ({"id": "synthetic-trace-20", "tool": "read_record", "arguments": {"record_id": 123}}, False),
        ({"id": "synthetic-trace-21", "tool": "read_record", "arguments": {"record_id": "x" * 101}}, False),
        ({"id": "synthetic-trace-22", "tool": "read_record", "arguments": {"record_id": ".hidden"}}, False),
    ]
    unsafe_total = sum(not expected for _, expected in cases)
    unsafe_fn = sum(
        not expected and verify_trace(row)["policy_result"] == "ALLOW_PROPOSAL"
        for row, expected in cases
    )
    return {
        "data_origin": "synthetic",
        "mode": "deterministic_policy",
        "case_count": len(cases),
        "unsafe_count": unsafe_total,
        "unsafe_action_false_negative_rate": unsafe_fn / unsafe_total if unsafe_total else None,
        "review_required": True,
        "cases_sha256": digest(cases),
        "note": "Self-consistency of toy and adversarial cases; not a security certification.",
    }


def evaluate() -> dict[str, Any]:
    """Run held-out deterministic baseline evaluation on the built-in synthetic dataset."""
    data = dataset()
    rows = split(data)["test"]
    tp = fp = fn = critical_fn = critical_total = 0
    for row in rows:
        pred = reconciliation(row)["category"] == "amount_mismatch"
        truth = row["expected_minor"] != row["actual_minor"]
        tp += int(pred and truth)
        fp += int(pred and not truth)
        fn += int(not pred and truth)
        critical_total += int(truth and row["critical"])
        critical_fn += int(not pred and truth and row["critical"])

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )
    return {
        "data_origin": "synthetic",
        "mode": "deterministic_heuristic",
        "test_count": len(rows),
        "dataset_sha256": digest(data),
        "split_seed": 42,
        "test_sha256": digest(rows),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "critical_false_negative_rate": critical_fn / critical_total if critical_total else None,
        "coverage_automatic_decision": 0.0,
        "review_rate": 1.0,
        "cost_per_case_usd": None,
        "latency_p95_ms": None,
        "note": "Synthetic self-consistency only; no trained model or pilot accuracy.",
    }
