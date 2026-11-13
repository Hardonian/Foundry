"""Deep unit tests for hardonian_foundry.refinery_gates covering threshold evaluation and fail-closed promotion gates."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

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

        # Dict per-class recall and dict OOD challenge threshold
        missing2: list[str] = []
        reasons2: list[str] = []
        summary2: dict[str, Any] = {}
        task_block2 = {
            "labels": ["A", "B"],
            "baseline": {
                "macro_f1": 0.70,
                "labels_scored": ["A", "B"],
                "per_class": {
                    "A": {"recall": 0.7, "support": 10},
                    "B": {"recall": 0.7, "support": 10},
                },
            },
            "model_metrics": {
                "macro_f1": 0.85,
                "labels_scored": ["A", "B"],
                "per_class": {
                    "A": {"recall": 0.5, "support": 10},
                    "B": {"recall": 0.9, "support": 10},
                },
            },
            "ood_challenge": {"count": 2, "accuracy": 0.6},
            "thresholds": {
                "per_class_recall_min": {"A": 0.8},
                "ood_challenge_predictions": {"min_count": 5, "min_accuracy": 0.85},
            },
            "baseline_must_improve": True,
        }
        _evaluate_task_thresholds(task_block2, missing2, reasons2, summary2)
        self.assertIn("task_per_class_recall", missing2)
        self.assertIn("ood_challenge_predictions", missing2)

    def test_evaluate_promotion_gate_failures(self):
        run_dir = self.root / "run-test"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text('{"provenance": {}, "preflight": {}}', encoding="utf-8")

        # 1. Provenance anchor mismatch failure
        mock_insp = {
            "status": "COMPLETED",
            "run_id": "run-test",
            "run_record": {"run_id": "run-test"},
            "provenance_status": "TAMPERED_ANCHOR_MISMATCH",
            "preflight_status": "PASSED",
            "anchor_error": "head mismatch",
            "ledger_report": {"status": "OK"},
        }
        with mock.patch("hardonian_foundry.refinery_gates.inspect_run", return_value=mock_insp):
            res = evaluate_promotion("run-test", output_root=self.root)
            self.assertFalse(res["promoted"])
            self.assertIn("anchor_checkpoint", res["missing_evidence"])

        # 2. Failed preflight
        mock_insp["provenance_status"] = "PROVENANCE_VERIFIED"
        mock_insp["preflight_status"] = "FAILED"
        mock_insp["evaluation"] = None
        with mock.patch("hardonian_foundry.refinery_gates.inspect_run", return_value=mock_insp):
            res = evaluate_promotion("run-test", output_root=self.root)
            self.assertFalse(res["promoted"])
            self.assertIn("passed_preflight", res["missing_evidence"])
            self.assertIn("evaluation_metrics", res["missing_evidence"])

        # 3. Missing 100% human review policy
        mock_insp["preflight_status"] = "PASSED"
        mock_insp["evaluation"] = {
            "f1": 0.9,
            "critical_false_negative_rate": 0.0,
            "review_rate": 0.5,
        }
        mock_insp["model_manifest"] = None
        mock_insp["resolved_ledger_path"] = "some_ledger.jsonl"
        with mock.patch("hardonian_foundry.refinery_gates.inspect_run", return_value=mock_insp):
            with mock.patch("hardonian_foundry.refinery_gates.extract_verified_license_attachment", return_value=(None, None, False)):
                res = evaluate_promotion("run-test", output_root=self.root)
                self.assertFalse(res["promoted"])
                self.assertIn("mandatory_human_review_policy", res["missing_evidence"])
                self.assertIn("model_licensing_manifest", res["missing_evidence"])

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
