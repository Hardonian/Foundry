"""Deep unit tests for hardonian_foundry.refinery_gates covering threshold evaluation and fail-closed promotion gates."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from hardonian_foundry.refinery_gates import (
    _as_rate,
    _evaluate_task_thresholds,
    evaluate_promotion,
)


class RefineryGatesDeepTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_as_rate_helper(self):
        self.assertEqual(_as_rate(0.5), 0.5)
        self.assertEqual(_as_rate(1), 1.0)
        self.assertEqual(_as_rate(0), 0.0)
        self.assertIsNone(_as_rate(True))
        self.assertIsNone(_as_rate(-0.1))
        self.assertIsNone(_as_rate(1.1))
        self.assertIsNone(_as_rate("not-a-number"))
        self.assertIsNone(_as_rate(float("nan")))
        self.assertIsNone(_as_rate(float("inf")))

    def test_evaluate_task_thresholds(self):
        missing: list[str] = []
        reasons: list[str] = []
        summary: dict[str, Any] = {}

        task_block = {
            "labels": ["A", "B"],
            "baseline": {
                "macro_f1": 0.70,
                "per_class": {
                    "A": {"recall": 0.7, "support": 10},
                    "B": {"recall": 0.7, "support": 10},
                },
            },
            "model_metrics": {
                "macro_f1": 0.75,
                "critical_false_negative_rate": 0.05,
                "per_class": {
                    "A": {"recall": 0.5, "support": 10},
                    "B": {"recall": 0.9, "support": 10},
                },
                "invalid_output_rate": 0.02,
                "latency_p95_ms": 150.0,
            },
            "thresholds": {
                "macro_f1_min": 0.80,
                "critical_fn_rate_max": 0.0,
                "per_class_recall_min": 0.80,
                "invalid_output_rate_max": 0.01,
                "latency_p95_ms_max": 100.0,
                "ood_challenge_predictions": 5,
            },
            "baseline_must_improve": True,
        }

        _evaluate_task_thresholds(task_block, missing, reasons, summary)
        self.assertIn("task_macro_f1_threshold", missing)
        self.assertIn("task_critical_fn_rate_threshold", missing)
        self.assertIn("task_per_class_recall", missing)
        self.assertIn("task_invalid_output_rate", missing)
        self.assertIn("task_latency_p95_ms", missing)
        self.assertIn("ood_challenge_predictions", missing)

    def test_evaluate_promotion_missing_run(self):
        res = evaluate_promotion("nonexistent-run", output_root=self.root)
        self.assertFalse(res["promoted"])
        self.assertIn("run_metadata", res["missing_evidence"])

    def test_evaluate_promotion_invalid_tenant_context(self):
        res = evaluate_promotion("any-run", output_root=self.root, tenant_context="not-a-tenant-context")  # type: ignore[arg-type]
        self.assertFalse(res["promoted"])
        self.assertIn("tenant_authorization", res["missing_evidence"])


if __name__ == "__main__":
    unittest.main()
