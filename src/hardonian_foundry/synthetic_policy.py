"""Synthetic-only deterministic reconciliation safety gate, never an executor."""
import hashlib
import json
import re


def assess(case):
    """Classify valid synthetic cases using integer arithmetic, always requiring review.

    A synthetic ID is an input convention, not proof of data provenance. Do not send
    real or customer records to this demonstrator. Missing values or currency
    disagreement require review; invalid types are rejected.
    """
    fields = {"id", "expected_minor", "actual_minor", "expected_currency", "actual_currency"}
    if not isinstance(case, dict) or set(case) != fields:
        raise ValueError("expected exact reconciliation fields")
    if not isinstance(case["id"], str) or not case["id"].startswith("synthetic-") or len(case["id"]) > 100:
        raise ValueError("invalid synthetic id")
    for key in ("expected_minor", "actual_minor"):
        value = case[key]
        if value is not None and (type(value) is not int or abs(value) > 10**12):
            raise ValueError("amounts must be bounded integer minor units or null")
    for key in ("expected_currency", "actual_currency"):
        value = case[key]
        if value is not None and (not isinstance(value, str) or re.fullmatch(r"[A-Z]{3}", value) is None):
            raise ValueError("currency must be a three-letter uppercase code or null")
    e, a = case["expected_minor"], case["actual_minor"]
    ec, ac = case["expected_currency"], case["actual_currency"]
    if e is None or a is None or ec is None or ac is None or ec != ac:
        category = "review_required"
    else:
        category = "matched" if e == a else "amount_mismatch"
    encoded = json.dumps(case, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"schema_version": "1.0", "data_origin": "synthetic", "mode": "deterministic_policy",
            "category": category, "review_required": True, "financial_write": False,
            "evidence_hash": hashlib.sha256(encoded).hexdigest()}


def assess_with_suggestion(case, suggestion):
    """Compare untrusted suggestions to arithmetic; suggestions never authorize actions."""
    result = assess(case)
    allowed = {"matched", "amount_mismatch", "review_required"}
    if not isinstance(suggestion, str) or suggestion not in allowed:
        result.update({"model_status": "UNAVAILABLE", "model_agrees": None})
    else:
        result.update({"model_status": "REVIEW_REQUIRED", "model_agrees": suggestion == result["category"]})
    return result
