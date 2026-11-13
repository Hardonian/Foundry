"""Synthetic-only deterministic reconciliation safety gate, never an executor."""

from __future__ import annotations

import re
from typing import Any

from .canonical import digest

_REQUIRED_FIELDS = frozenset({"id", "expected_minor", "actual_minor", "expected_currency", "actual_currency"})
_CURRENCY_RE = re.compile(r"[A-Z]{3}")
_ALLOWED_SUGGESTIONS = frozenset({"matched", "amount_mismatch", "review_required"})


def assess(case: Any) -> dict[str, Any]:
    """Classify valid synthetic cases using integer arithmetic, always requiring review.

    A synthetic ID is an input convention, not proof of data provenance. Do not send
    real or customer records to this demonstrator. Missing values or currency
    disagreement require review; invalid types are rejected.
    """
    if not isinstance(case, dict) or set(case) != _REQUIRED_FIELDS:
        raise ValueError("expected exact reconciliation fields")
    case_id = case["id"]
    if not isinstance(case_id, str) or not case_id.startswith("synthetic-") or len(case_id) > 100:
        raise ValueError("invalid synthetic id")
    for key in ("expected_minor", "actual_minor"):
        value = case[key]
        if value is not None and (type(value) is not int or abs(value) > 10**12):
            raise ValueError("amounts must be bounded integer minor units or null")
    for key in ("expected_currency", "actual_currency"):
        value = case[key]
        if value is not None and (not isinstance(value, str) or _CURRENCY_RE.fullmatch(value) is None):
            raise ValueError("currency must be a three-letter uppercase code or null")
    expected_minor, actual_minor = case["expected_minor"], case["actual_minor"]
    expected_currency, actual_currency = case["expected_currency"], case["actual_currency"]
    if (
        expected_minor is None
        or actual_minor is None
        or expected_currency is None
        or actual_currency is None
        or expected_currency != actual_currency
    ):
        category = "review_required"
    else:
        category = "matched" if expected_minor == actual_minor else "amount_mismatch"
    return {
        "schema_version": "1.0",
        "data_origin": "synthetic",
        "mode": "deterministic_policy",
        "category": category,
        "review_required": True,
        "financial_write": False,
        "evidence_hash": digest(case),
    }


def assess_with_suggestion(case: Any, suggestion: Any) -> dict[str, Any]:
    """Compare untrusted suggestions to arithmetic; suggestions never authorize actions."""
    result = assess(case)
    if not isinstance(suggestion, str) or suggestion not in _ALLOWED_SUGGESTIONS:
        result.update({"model_status": "UNAVAILABLE", "model_agrees": None})
    else:
        result.update({"model_status": "REVIEW_REQUIRED", "model_agrees": suggestion == result["category"]})
    return result
