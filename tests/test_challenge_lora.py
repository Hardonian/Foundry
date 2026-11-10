import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("challenge_lora", SCRIPTS / "challenge_lora.py")
challenge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(challenge)


class ChallengeTest(unittest.TestCase):
    def test_oracle_branches(self):
        self.assertEqual(challenge.oracle(-1, -1, "USD", "USD"), "matched")
        self.assertEqual(challenge.oracle(-1, 0, "USD", "USD"), "amount_mismatch")
        self.assertEqual(challenge.oracle(None, 1, "USD", "USD"), "review_required")
        self.assertEqual(challenge.oracle(1, None, "USD", "USD"), "review_required")
        self.assertEqual(challenge.oracle(1, 1, "USD", None), "review_required")
        self.assertEqual(challenge.oracle(1, 1, "USD", "EUR"), "review_required")

    def test_unique_and_expected_labels(self):
        rows = challenge.challenge_cases()
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({x["id"] for x in rows}), len(rows))
        self.assertEqual(len({x["prompt"] for x in rows}), len(rows))
        self.assertEqual(Counter(x["expected_label"] for x in rows),
                         {"matched": 8, "amount_mismatch": 8, "review_required": 8})
        self.assertTrue(any(x["expected_minor"] is not None and x["expected_minor"] < 0 for x in rows))
        self.assertTrue(any(x["expected_minor"] is not None and x["expected_minor"] > 1000000 for x in rows))

    def test_scoring_does_not_treat_invalid_as_safe(self):
        rows = [{"expected": "review_required", "output": "Matched", "max_token_limit_hit": False},
                {"expected": "amount_mismatch", "output": "matched", "max_token_limit_hit": False},
                {"expected": "matched", "output": "matched", "max_token_limit_hit": False},
                {"expected": "review_required", "output": "review_required", "max_token_limit_hit": False}]
        out = challenge.summarize(rows)
        self.assertEqual(out["strict_correct"], 2)
        self.assertEqual(out["unsafe_matched"], 2)
        self.assertEqual(out["invalid_outputs"], 1)
        self.assertEqual(out["by_expected"]["review_required"]["correct"], 1)


if __name__ == "__main__":
    unittest.main()
