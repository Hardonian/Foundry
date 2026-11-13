"""Offline tests for synthetic deterministic reconciliation safety policy."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from hardonian_foundry.canonical import digest
from hardonian_foundry.synthetic_policy import assess, assess_with_suggestion

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("challenge_policy_fixture", SCRIPTS / "challenge_lora.py")
assert spec is not None and spec.loader is not None
challenge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(challenge)


class SyntheticPolicyTest(unittest.TestCase):
    def test_every_challenge_case_uses_deterministic_oracle_and_review(self) -> None:
        rows = challenge.challenge_cases()
        for row in rows:
            record = {key: row[key] for key in ("id", "expected_minor", "actual_minor", "expected_currency", "actual_currency")}
            out = assess(record)
            self.assertEqual(out["category"], row["expected_label"], row["id"])
            self.assertTrue(out["review_required"])
            self.assertFalse(out["financial_write"])
            self.assertEqual(out["evidence_hash"], digest(record))
        self.assertGreaterEqual(len(rows), 24)

    def test_untrusted_suggestions_cannot_change_policy_or_approve(self) -> None:
        case = {
            "id": "synthetic-risk",
            "expected_minor": 10,
            "actual_minor": 11,
            "expected_currency": "USD",
            "actual_currency": "USD",
        }
        suggestions: tuple[object, ...] = (
            "matched",
            "amount_mismatch",
            "review_required",
            None,
            {},
            "Matched",
            "",
            float("nan"),
        )
        for suggestion in suggestions:
            with self.subTest(suggestion=suggestion):
                out = assess_with_suggestion(case, suggestion)
                self.assertEqual(out["category"], "amount_mismatch")
                self.assertTrue(out["review_required"])
                self.assertFalse(out["financial_write"])
                if suggestion == "matched":
                    self.assertFalse(out["model_agrees"])
                elif suggestion == "review_required":
                    self.assertEqual(out["model_status"], "REVIEW_REQUIRED")
                    self.assertFalse(out["model_agrees"])
                if not isinstance(suggestion, str) or suggestion not in {"matched", "amount_mismatch", "review_required"}:
                    self.assertEqual(out["model_status"], "UNAVAILABLE")
                    self.assertIsNone(out["model_agrees"])

    def test_malformed_amounts_and_currencies_reject(self) -> None:
        case = {
            "id": "synthetic-case",
            "expected_minor": 10,
            "actual_minor": 10,
            "expected_currency": "USD",
            "actual_currency": "USD",
        }
        for changed in (
            {"actual_minor": True},
            {"actual_minor": 1.5},
            {"actual_minor": 10**12 + 1},
            {"actual_minor": -(10**12) - 1},
            {"actual_currency": "usd"},
            {"actual_currency": "US"},
            {"actual_currency": "USD\n"},
            {"actual_currency": "US1"},
            {"id": "real-record"},
            {"id": "synthetic-" + "x" * 95},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                assess({**case, **changed})
        non_dicts: tuple[object, ...] = (None, [], "case")
        for non_dict in non_dicts:
            with self.assertRaises(ValueError):
                assess(non_dict)
        with self.assertRaises(ValueError):
            assess({**case, "unknown": 1})

    def test_boundary_amounts_and_missing_currency_mismatch_require_review(self) -> None:
        base = {
            "id": "synthetic-case",
            "expected_minor": 10,
            "actual_minor": 10,
            "expected_currency": "USD",
            "actual_currency": "USD",
        }
        self.assertEqual(assess({**base, "expected_minor": 10**12, "actual_minor": 10**12})["category"], "matched")
        self.assertEqual(assess({**base, "expected_minor": -(10**12), "actual_minor": 0})["category"], "amount_mismatch")
        for changed in (
            {"expected_minor": None},
            {"actual_minor": None},
            {"expected_currency": None},
            {"actual_currency": None},
            {"actual_currency": "EUR"},
        ):
            self.assertEqual(assess({**base, **changed})["category"], "review_required")


if __name__ == "__main__":
    unittest.main()
