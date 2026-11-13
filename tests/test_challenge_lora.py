import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location("challenge_lora", SCRIPTS / "challenge_lora.py")
challenge = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(challenge)


class ChallengeTest(unittest.TestCase):
    def test_oracle_branches(self):
        self.assertEqual(challenge.oracle(-1, -1, "USD", "USD"), "matched")
        self.assertEqual(challenge.oracle(-1, 0, "USD", "USD"), "amount_mismatch")
        self.assertEqual(challenge.oracle(None, 1, "USD", "USD"), "review_required")
        self.assertEqual(challenge.oracle(1, None, "USD", "USD"), "review_required")
        self.assertEqual(challenge.oracle(1, 1, "USD", None), "review_required")
        self.assertEqual(challenge.oracle(1, 1, "USD", "EUR"), "review_required")
        self.assertEqual(challenge.oracle(1, 1, "", "USD"), "review_required")

    def test_unique_and_expected_labels(self):
        rows = challenge.challenge_cases()
        self.assertEqual(len(rows), 48)
        self.assertEqual(len({x["id"] for x in rows}), len(rows))
        self.assertEqual(len({x["prompt"] for x in rows}), len(rows))
        self.assertEqual(
            Counter(x["expected_label"] for x in rows),
            {"matched": 16, "amount_mismatch": 16, "review_required": 16},
        )
        self.assertTrue(any(x["expected_minor"] is not None and x["expected_minor"] < 0 for x in rows))
        self.assertTrue(any(x["expected_minor"] is not None and x["expected_minor"] > 1000000 for x in rows))
        # Sign flips (-123 vs 123 and 123 vs -123)
        self.assertTrue(
            any(
                x["expected_minor"] is not None
                and x["actual_minor"] is not None
                and x["expected_minor"] == -x["actual_minor"]
                and x["expected_minor"] != 0
                for x in rows
            )
        )
        # Distractor prose cases exist
        self.assertTrue(any("Ignore" in x["prompt"] or "Prior" in x["prompt"] for x in rows))

    def test_scoring_does_not_treat_invalid_as_safe(self):
        rows = [
            {"expected": "review_required", "output": "Matched", "max_token_limit_hit": False},
            {"expected": "amount_mismatch", "output": "matched", "max_token_limit_hit": False},
            {"expected": "matched", "output": "matched", "max_token_limit_hit": False},
            {"expected": "review_required", "output": "review_required", "max_token_limit_hit": False},
        ]
        out = challenge.summarize(rows)
        self.assertEqual(out["strict_correct"], 2)
        self.assertEqual(out["unsafe_matched"], 2)
        self.assertEqual(out["invalid_outputs"], 1)
        self.assertEqual(out["by_expected"]["review_required"]["correct"], 1)

    def test_foundry_prediction_formatting_fails_closed_to_review_required(self):
        cases = [
            {"id": "c1", "output": "matched", "confidence": 0.9, "latency_ms": 12.5},
            {"id": "c2", "output": "amount_mismatch", "confidence": 0.8, "latency_ms": 11.0},
            {"id": "c3", "output": "junk text", "confidence": 0.2, "latency_ms": 9.0},
        ]
        formatted = challenge.format_foundry_predictions(cases)
        self.assertEqual(formatted[0]["label"], "matched")
        self.assertFalse(formatted[0]["raw_invalid"])
        self.assertEqual(formatted[2]["label"], "review_required")
        self.assertTrue(formatted[2]["raw_invalid"])

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "preds" / "challenge.jsonl"
            challenge.write_foundry_predictions(out_path, cases)
            lines = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 3)
            self.assertEqual([row["label"] for row in lines], ["matched", "amount_mismatch", "review_required"])


if __name__ == "__main__":
    unittest.main()
