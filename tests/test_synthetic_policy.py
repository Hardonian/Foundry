import importlib.util
import sys
import unittest
from pathlib import Path

from hardonian_foundry.synthetic_policy import assess, assess_with_suggestion

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("challenge_policy_fixture", SCRIPTS / "challenge_lora.py")
challenge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(challenge)


class SyntheticPolicyTest(unittest.TestCase):
    def test_every_challenge_case_uses_deterministic_oracle_and_review(self):
        rows = challenge.challenge_cases()
        for row in rows:
            record = {key: row[key] for key in ("id", "expected_minor", "actual_minor", "expected_currency", "actual_currency")}
            out = assess(record)
            self.assertEqual(out["category"], row["expected_label"], row["id"])
            self.assertTrue(out["review_required"])
            self.assertFalse(out["financial_write"])
        self.assertEqual(len(rows), 24)

    def test_untrusted_suggestions_cannot_change_policy_or_approve(self):
        case = {"id": "synthetic-risk", "expected_minor": 10, "actual_minor": 11,
                "expected_currency": "USD", "actual_currency": "USD"}
        for suggestion in ("matched", "amount_mismatch", "review_required", None, {}, "Matched", ""):
            with self.subTest(suggestion=suggestion):
                out = assess_with_suggestion(case, suggestion)
                self.assertEqual(out["category"], "amount_mismatch")
                self.assertTrue(out["review_required"])
                self.assertFalse(out["financial_write"])
                if suggestion == "matched":
                    self.assertFalse(out["model_agrees"])
                if not isinstance(suggestion, str) or suggestion not in {"matched", "amount_mismatch", "review_required"}:
                    self.assertEqual(out["model_status"], "UNAVAILABLE")
                    self.assertIsNone(out["model_agrees"])

    def test_malformed_amounts_and_currencies_reject(self):
        case = {"id": "synthetic-case", "expected_minor": 10, "actual_minor": 10,
                "expected_currency": "USD", "actual_currency": "USD"}
        for changed in ({"actual_minor": True}, {"actual_minor": 1.5}, {"actual_minor": 10**12+1},
                        {"actual_currency": "usd"}, {"actual_currency": "US"}, {"id": "real-record"}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                assess({**case, **changed})
        with self.assertRaises(ValueError):
            assess({**case, "unknown": 1})

    def test_missing_and_currency_mismatch_require_review(self):
        base = {"id": "synthetic-case", "expected_minor": 10, "actual_minor": 10,
                "expected_currency": "USD", "actual_currency": "USD"}
        for changed in ({"expected_minor": None}, {"actual_minor": None}, {"expected_currency": None},
                        {"actual_currency": None}, {"actual_currency": "EUR"}):
            self.assertEqual(assess({**base, **changed})["category"], "review_required")


if __name__ == "__main__":
    unittest.main()
