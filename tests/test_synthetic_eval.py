import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("synthetic_eval", SCRIPTS / "evaluate_synthetic_lora.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SyntheticEvaluationTest(unittest.TestCase):
    def test_metrics_distinguish_false_negative_and_invalid_output(self):
        rows = [
            {"expected": "amount_mismatch", "output": "matched", "max_token_limit_hit": False},
            {"expected": "amount_mismatch", "output": "unexpected", "max_token_limit_hit": True},
            {"expected": "matched", "output": "amount_mismatch", "max_token_limit_hit": False},
            {"expected": "matched", "output": "matched", "max_token_limit_hit": False},
        ]
        result = module.metrics(rows)
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["recall"], 0)
        self.assertEqual(result["false_negative_rate"], 1)
        self.assertEqual(result["f1"], 0)
        self.assertEqual(result["precision"], 0)
        self.assertEqual(result["invalid_outputs"], 1)
        self.assertEqual(result["max_token_limit_hits"], 1)

    def test_undefined_precision_and_recall_are_null(self):
        row = {"expected": "matched", "output": "matched", "max_token_limit_hit": False}
        result = module.metrics([row])
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])


if __name__ == "__main__":
    unittest.main()
