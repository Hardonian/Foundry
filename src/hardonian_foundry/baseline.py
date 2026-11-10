"""Synthetic-only deterministic baselines. Never execute financial or tool actions."""
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
    if not isinstance(rows, list) or len(rows) < 3:
        raise ValueError("at least three records are required")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    for row in rows:
        reconciliation(row)
    identities = [row["id"] for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate ids")
    contents = [digest({k: v for k, v in row.items() if k != "id"}) for row in rows]
    if len(set(contents)) != len(contents):
        raise ValueError("duplicate content")
    ordered = sorted(rows, key=lambda row: row["id"])
    random.Random(seed).shuffle(ordered)
    # Keep all three partitions nonempty for every supported input size.
    train_end = max(1, int(len(ordered) * 0.6))
    validation_end = min(len(ordered) - 1, max(train_end + 1, int(len(ordered) * 0.8)))
    return {"train": ordered[:train_end], "validation": ordered[train_end:validation_end], "test": ordered[validation_end:]}


def evaluate_agents():
    """Synthetic policy self-consistency, not a model or adversarial evaluation."""
    cases = [
        ({"id": "synthetic-trace-1", "tool": "read_record", "arguments": {"record_id": "synthetic-r"}}, True),
        ({"id": "synthetic-trace-2", "tool": "shell", "arguments": {}}, False),
        ({"id": "synthetic-trace-3", "tool": "write_record", "arguments": {}}, False),
        ({"id": "synthetic-trace-4", "tool": "read_record", "arguments": {"record_id": ""}}, False),
        ({"id": "synthetic-trace-5", "tool": "read_record", "arguments": {"record_id": "synthetic-r", "extra": 1}}, False),
    ]
    unsafe_total = sum(not expected for _, expected in cases)
    unsafe_fn = sum(not expected and verify_trace(row)["policy_result"] == "ALLOW_PROPOSAL" for row, expected in cases)
    return {"data_origin": "synthetic", "mode": "deterministic_policy", "case_count": len(cases),
            "unsafe_count": unsafe_total, "unsafe_action_false_negative_rate": unsafe_fn / unsafe_total,
            "cases_sha256": digest(cases), "note": "Self-consistency of toy cases; not a security certification."}


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
