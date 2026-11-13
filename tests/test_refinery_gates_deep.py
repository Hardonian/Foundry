"""Deep unit tests for hardonian_foundry.refinery_gates covering threshold evaluation and fail-closed promotion gates."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from hardonian_foundry.canonical import digest
from hardonian_foundry.refinery_gates import (
    _as_rate,
    _evaluate_task_thresholds,
    _valid_rate,
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

    def test_valid_rate_and_threshold_branches(self):
        self.assertTrue(_valid_rate(0.5))
        self.assertFalse(_valid_rate(-0.1))
        self.assertFalse(_valid_rate("bad"))

        # Valid metrics but thresholds is None -> exercises line 114
        miss: list[str] = []
        reas: list[str] = []
        task_no_thresh = {
            "labels": ["A"],
            "baseline": {"macro_f1": 0.7, "per_class": {"A": {"support": 5}}},
            "model_metrics": {"macro_f1": 0.85, "per_class": {"A": {"support": 5}}},
            "thresholds": None,
        }
        _evaluate_task_thresholds(task_no_thresh, miss, reas, {})
        self.assertEqual(len(miss), 0)

        # OOD integer min_count and dict with macro_f1_min
        task_ood = {
            "labels": ["A"],
            "baseline": {"macro_f1": 0.7},
            "model_metrics": {"macro_f1": 0.8},
            "ood_challenge": {"count": 3, "accuracy": 0.5},
            "thresholds": {
                "ood_challenge_predictions": {"min_count": 5, "macro_f1_min": 0.8},
            },
        }
        miss_ood: list[str] = []
        _evaluate_task_thresholds(task_ood, miss_ood, [], {})
        self.assertIn("ood_challenge_predictions", miss_ood)

        # OOD integer count below required
        task_ood_int = {
            "labels": ["A"],
            "baseline": {"macro_f1": 0.7},
            "model_metrics": {"macro_f1": 0.8},
            "ood_challenge": {"count": 2},
            "thresholds": {"ood_challenge_predictions": 5},
        }
        miss_ood_int: list[str] = []
        _evaluate_task_thresholds(task_ood_int, miss_ood_int, [], {})
        self.assertIn("ood_challenge_predictions", miss_ood_int)

    def test_evaluate_promotion_advanced_gate_branches(self):
        adv_dir = self.root / "run-test-adv"
        adv_dir.mkdir(parents=True, exist_ok=True)
        (adv_dir / "run.json").write_text('{"run_id": "run-test-adv"}', encoding="utf-8")

        # 1. Attached license failed self-verification
        mock_insp1 = {
            "status": "COMPLETED",
            "run_id": "run-test-adv",
            "run_record": {"run_id": "run-test-adv"},
            "provenance_status": "PROVENANCE_VERIFIED",
            "preflight_status": "PASSED",
            "evaluation": {"f1": 0.9, "review_rate": 1.0, "critical_false_negative_rate": 0.0},
            "model_manifest": None,
            "resolved_ledger_path": "evidence.jsonl",
            "ledger_report": {"status": "OK", "entries": []},
        }
        dummy_entry = {"record": {"type": "model_licensing_manifest"}}
        with mock.patch("hardonian_foundry.refinery_gates.inspect_run", return_value=mock_insp1):
            with mock.patch("hardonian_foundry.refinery_gates.extract_verified_license_attachment", return_value=(dummy_entry, None, False)):
                res = evaluate_promotion("run-test-adv", output_root=self.root)
                self.assertFalse(res["promoted"])
                self.assertIn("model_licensing_manifest", res["missing_evidence"])
                self.assertTrue(any("ledger license attachment failed self-verification" in r for r in res["reasons"]))

        # 2. Signing backend unavailable
        sig_entry = {
            "seq": 2,
            "prev_hash": "h0",
            "entry_hash": "h1",
            "record": {
                "type": "promotion_signature",
                "run_id": "run-test-adv",
                "public_key_hex": "a" * 64,
                "signed_head": "h0",
                "version": "2",
                "run_sha256": "0" * 64,
            },
        }
        mock_insp2 = dict(mock_insp1)
        mock_insp2["model_manifest"] = {"license_id": "Apache-2.0"}
        mock_insp2["ledger_entries"] = [sig_entry]
        with mock.patch("hardonian_foundry.refinery_gates.inspect_run", return_value=mock_insp2):
            with mock.patch("hardonian_foundry.signing.backend_available", return_value=False):
                res = evaluate_promotion("run-test-adv", output_root=self.root)
                self.assertFalse(res["promoted"])
                self.assertIn("signing_backend", res["missing_evidence"])

        # 3. Signature prev_hash != signed_head and revoked key
        sig_bad_head = {
            "seq": 2,
            "prev_hash": "h_different",
            "entry_hash": "h1",
            "record": {
                "type": "promotion_signature",
                "run_id": "run-test-adv",
                "public_key_hex": "a" * 64,
                "signed_head": "h_expected",
                "version": "2",
                "run_sha256": "0" * 64,
            },
        }
        mock_insp3 = dict(mock_insp1)
        mock_insp3["model_manifest"] = {"license_id": "Apache-2.0"}
        mock_insp3["ledger_entries"] = [sig_bad_head]
        sig_bad_head["record"]["run_sha256"] = digest(mock_insp3["run_record"])
        with mock.patch("hardonian_foundry.refinery_gates.inspect_run", return_value=mock_insp3):
            with mock.patch("hardonian_foundry.signing.backend_available", return_value=True):
                res = evaluate_promotion(
                    "run-test-adv",
                    output_root=self.root,
                    keyring={"trusted_keys": [{"public_key_hex": "a" * 64, "revoked_at": "2026-01-01T00:00:00Z"}]},
                )
                self.assertFalse(res["promoted"])
                self.assertTrue(any("signature does not bind the ledger head" in r for r in res["reasons"]))


if __name__ == "__main__":
    unittest.main()
