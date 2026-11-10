"""Synthetic-only deterministic baselines. Never execute financial or tool actions."""
from decimal import Decimal, InvalidOperation
import hashlib
import json
import random


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def reconciliation(case):
    """Validate integer minor units; suggest only. Unknown inputs always require review."""
    if set(case) != {"id", "expected_minor", "actual_minor", "critical"}:
        raise ValueError("invalid reconciliation fields")
    if not isinstance(case["id"], str) or not case["id"].startswith("synthetic-") or len(case["id"]) > 100:
        raise ValueError("id must have synthetic- prefix")
    if any(type(case[k]) is not int or abs(case[k]) > 10**12 for k in ("expected_minor", "actual_minor")) or type(case["critical"]) is not bool:
        raise ValueError("amounts must be bounded integer minor units and critical must be boolean")
    diff = case["actual_minor"] - case["expected_minor"]
    return {"schema_version": "1.0", "data_origin": "synthetic", "mode": "deterministic_heuristic", "category": "amount_mismatch" if diff else "matched", "difference_minor": diff, "review_required": True, "action": "human_review", "evidence_hash": digest(case)}


def verify_trace(trace):
    """Explicit allowlist: all tool calls remain proposals, never execution."""
    if set(trace) != {"id", "tool", "arguments"} or not isinstance(trace["id"], str) or not trace["id"].startswith("synthetic-") or len(trace["id"]) > 100 or not isinstance(trace["tool"], str) or not isinstance(trace["arguments"], dict):
        raise ValueError("invalid trace")
    allowed = trace["tool"] == "read_record" and set(trace["arguments"]) == {"record_id"} and isinstance(trace["arguments"]["record_id"], str) and 0 < len(trace["arguments"]["record_id"]) <= 100
    return {"schema_version": "1.0", "data_origin": "synthetic", "mode": "deterministic_policy", "policy_result": "ALLOW_PROPOSAL" if allowed else "DENY", "executed": False, "review_required": True, "evidence_hash": digest(trace)}


def dataset():
    return [{"id": f"synthetic-{i:03d}", "expected_minor": 10000+i*7, "actual_minor": 10000+i*7+(25 if i % 3 == 0 else 0), "critical": i % 5 == 0} for i in range(60)]


def split(rows, seed=42):
    identities = [row["id"] for row in rows]
    if len(set(identities)) != len(identities): raise ValueError("duplicate ids")
    contents = [digest({k:v for k,v in row.items() if k != "id"}) for row in rows]
    if len(set(contents)) != len(contents): raise ValueError("duplicate content")
    ordered = list(rows); random.Random(seed).shuffle(ordered)
    return {"train": ordered[:36], "validation": ordered[36:48], "test": ordered[48:]}


def evaluate():
    rows = split(dataset())["test"]
    tp=fp=fn=critical_fn=critical_total=0
    for row in rows:
        pred = reconciliation(row)["category"] == "amount_mismatch"
        truth = row["expected_minor"] != row["actual_minor"]
        tp += int(pred and truth); fp += int(pred and not truth); fn += int(not pred and truth)
        critical_total += int(truth and row["critical"]); critical_fn += int(not pred and truth and row["critical"])
    precision = tp/(tp+fp) if tp+fp else None; recall = tp/(tp+fn) if tp+fn else None
    f1 = 2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else None
    return {"data_origin":"synthetic", "mode":"deterministic_heuristic", "test_count":len(rows), "dataset_sha256":digest(dataset()), "split_seed":42, "test_sha256":digest(rows), "precision":precision, "recall":recall, "f1":f1, "critical_false_negative_rate":critical_fn/critical_total if critical_total else None, "coverage_automatic_decision":0.0, "review_rate":1.0, "cost_per_case_usd":None, "latency_p95_ms":None, "note":"Synthetic self-consistency only; no trained model or pilot accuracy."}
