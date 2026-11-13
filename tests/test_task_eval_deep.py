"""Deep unit tests for hardonian_foundry.task_eval covering spec validation,
path hygiene, threshold validation, dataset validation, JSONL loading, and prediction map checks."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hardonian_foundry.task_eval import (
    _is_clean_relative_path,
    _validate_predictions_map,
    _validate_thresholds,
    bootstrap_ci,
    load_jsonl,
    load_predictor_dataset,
    mcnemar_test,
    validate_spec,
    validate_task_dataset,
)


class TaskEvalDeepTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_is_clean_relative_path(self):
        self.assertTrue(_is_clean_relative_path("data/dataset.jsonl"))
        self.assertTrue(_is_clean_relative_path("spec.json"))

        # Rejections
        self.assertFalse(_is_clean_relative_path(""))
        self.assertFalse(_is_clean_relative_path(123))
        self.assertFalse(_is_clean_relative_path("/abs/posix/path"))
        self.assertFalse(_is_clean_relative_path("\\abs\\win\\path"))
        self.assertFalse(_is_clean_relative_path("C:\\drive\\path"))
        self.assertFalse(_is_clean_relative_path("traversal/../escape"))
        self.assertFalse(_is_clean_relative_path("D:relative_drive_path"))
        self.assertFalse(_is_clean_relative_path("bad\x00null"))
        self.assertFalse(_is_clean_relative_path("bad\x01ctrl"))

    def test_validate_spec_errors(self):
        # Non-dict
        self.assertIn("task spec must be a JSON object", validate_spec("not-a-dict"))

        valid_spec = {
            "schema_version": "1.0",
            "task_id": "test-task",
            "description": "Valid task description.",
            "labels": ["PASS", "FAIL"],
            "thresholds": {"macro_f1_min": 0.85},
            "model": {
                "base_model_id": "org/model",
                "revision": "rev1",
                "adapter_sha256": "0" * 64,
                "adapter_name": "adapter-v1",
            },
        }
        self.assertEqual(validate_spec(valid_spec), [])

        # Unknown spec field
        errs_unknown = validate_spec({**valid_spec, "rogue_field": "val"})
        self.assertTrue(any("unknown spec field" in e for e in errs_unknown))

        # Bad schema version
        errs_ver = validate_spec({**valid_spec, "schema_version": "2.0"})
        self.assertTrue(any("schema_version" in e for e in errs_ver))

        # Bad task_id
        errs_tid = validate_spec({**valid_spec, "task_id": "has space"})
        self.assertTrue(any("task_id" in e for e in errs_tid))

        # Bad description
        errs_desc = validate_spec({**valid_spec, "description": ""})
        self.assertTrue(any("description" in e for e in errs_desc))

        # Bad labels
        errs_lbl = validate_spec({**valid_spec, "labels": ["ONLY_ONE"]})
        self.assertTrue(any("labels" in e for e in errs_lbl))

        # Bad model subfields
        bad_model = dict(valid_spec["model"])
        bad_model["adapter_sha256"] = "not-64-hex"
        errs_sha = validate_spec({**valid_spec, "model": bad_model})
        self.assertTrue(any("adapter_sha256" in e for e in errs_sha))

    def test_validate_thresholds(self):
        # Missing macro_f1_min
        self.assertTrue(any("macro_f1_min" in e for e in _validate_thresholds({}, ["A", "B"])))

        # Bad macro_f1_min range (< 0.8)
        self.assertTrue(any("macro_f1_min" in e for e in _validate_thresholds({"macro_f1_min": 0.5}, ["A", "B"])))

        # Unknown threshold fields
        self.assertTrue(any("unknown" in e for e in _validate_thresholds({"macro_f1_min": 0.85, "extra": 1}, ["A", "B"])))

        # Per class recall min bad label
        errs_pcr = _validate_thresholds(
            {"macro_f1_min": 0.85, "per_class_recall_min": {"INVALID_LABEL": 0.9}},
            ["A", "B"],
        )
        self.assertTrue(any("INVALID_LABEL" in e for e in errs_pcr))

    def test_load_jsonl(self):
        f = self.root / "data.jsonl"
        f.write_text('{"id": "1", "val": "a"}\n\n{"id": "2", "val": "b"}\n', encoding="utf-8")
        rows = load_jsonl(f)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], "1")

        # Invalid JSON line
        bad_f = self.root / "bad.jsonl"
        bad_f.write_text('{"valid": 1}\nnot-json\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_jsonl(bad_f)

        # Non-dict line
        nondict_f = self.root / "nondict.jsonl"
        nondict_f.write_text("[1, 2, 3]\n", encoding="utf-8")
        with self.assertRaises(TypeError):
            load_jsonl(nondict_f)

        # Empty file
        empty_f = self.root / "empty.jsonl"
        empty_f.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_jsonl(empty_f)

    def test_validate_task_dataset(self):
        spec = {
            "schema_version": "1.0",
            "task_id": "test-task",
            "labels": ["PASS", "FAIL"],
        }
        # Non-list
        res_nonlist = validate_task_dataset("not-a-list", spec)
        self.assertEqual(res_nonlist["status"], "INVALID")

        # Fewer than 9 records
        res_few = validate_task_dataset([{"id": "1"}], spec)
        self.assertEqual(res_few["status"], "INVALID")
        self.assertTrue(any("at least 9 records" in e for e in res_few["errors"]))

    def test_validate_predictions_map(self):
        test_rows = [
            {"id": "r1", "gold_label": "PASS"},
            {"id": "r2", "gold_label": "FAIL"},
        ]
        label_set = {"PASS", "FAIL"}

        # Non-list predictions
        with self.assertRaises(TypeError):
            _validate_predictions_map(test_rows, "not-a-list", label_set)

        # Missing prediction for r2
        with self.assertRaises(ValueError):
            _validate_predictions_map(test_rows, [{"id": "r1", "label": "PASS"}], label_set)

        # Unknown label in prediction
        with self.assertRaises(ValueError):
            _validate_predictions_map(
                test_rows,
                [{"id": "r1", "label": "PASS"}, {"id": "r2", "label": "UNKNOWN_LABEL"}],
                label_set,
            )

        # Missing required label field
        with self.assertRaises(ValueError):
            _validate_predictions_map(test_rows, [{"id": "r1"}, {"id": "r2", "label": "FAIL"}], label_set)

        # Extra unknown field in prediction
        with self.assertRaises(ValueError):
            _validate_predictions_map(
                test_rows,
                [{"id": "r1", "label": "PASS", "rogue": 1}, {"id": "r2", "label": "FAIL"}],
                label_set,
            )

        # Bad confidence values (> 1.0)
        with self.assertRaises(ValueError):
            _validate_predictions_map(
                test_rows,
                [{"id": "r1", "label": "PASS", "confidence": 1.5}, {"id": "r2", "label": "FAIL", "confidence": 0.5}],
                label_set,
            )

        # Bad latency values (< 0)
        with self.assertRaises(ValueError):
            _validate_predictions_map(
                test_rows,
                [{"id": "r1", "label": "PASS", "latency_ms": -1.0}, {"id": "r2", "label": "FAIL", "latency_ms": 10.0}],
                label_set,
            )

        # Non-boolean raw_invalid and abstained
        with self.assertRaises(ValueError):
            _validate_predictions_map(
                test_rows,
                [{"id": "r1", "label": "PASS", "raw_invalid": "not-bool"}, {"id": "r2", "label": "FAIL"}],
                label_set,
            )
        with self.assertRaises(ValueError):
            _validate_predictions_map(
                test_rows,
                [{"id": "r1", "label": "PASS", "abstained": "not-bool"}, {"id": "r2", "label": "FAIL"}],
                label_set,
            )

        # Inconsistent confidence (only on 1 prediction)
        with self.assertRaises(ValueError):
            _validate_predictions_map(
                test_rows,
                [{"id": "r1", "label": "PASS", "confidence": 0.9}, {"id": "r2", "label": "FAIL"}],
                label_set,
            )

        # Inconsistent latency_ms (only on 1 prediction)
        with self.assertRaises(ValueError):
            _validate_predictions_map(
                test_rows,
                [{"id": "r1", "label": "PASS", "latency_ms": 10.0}, {"id": "r2", "label": "FAIL"}],
                label_set,
            )

        # Non-string or empty prediction id
        with self.assertRaises(ValueError):
            _validate_predictions_map(
                test_rows,
                [{"id": "", "label": "PASS"}, {"id": "r2", "label": "FAIL"}],
                label_set,
            )

        # Duplicate prediction id
        with self.assertRaises(ValueError):
            _validate_predictions_map(
                test_rows,
                [{"id": "r1", "label": "PASS"}, {"id": "r1", "label": "FAIL"}],
                label_set,
            )

        # Valid predictions map
        valid_preds = [
            {"id": "r1", "label": "PASS", "confidence": 0.95, "latency_ms": 10.0},
            {"id": "r2", "label": "FAIL", "confidence": 0.88, "latency_ms": 12.0},
        ]
        pmap, has_conf, has_lat = _validate_predictions_map(test_rows, valid_preds, label_set)
        self.assertEqual(len(pmap), 2)
        self.assertTrue(has_conf)
        self.assertTrue(has_lat)

    def test_bootstrap_ci_valid_and_errors(self):
        spec = {"labels": ["PASS", "FAIL"]}
        test_rows = [
            {"id": f"r{i}", "gold_label": "PASS" if i % 2 == 0 else "FAIL", "critical": False}
            for i in range(12)
        ]
        preds = [
            {"id": f"r{i}", "label": "PASS" if i % 2 == 0 else "FAIL"}
            for i in range(12)
        ]

        # Valid run
        ci = bootstrap_ci(test_rows, preds, spec, n_bootstrap=20, confidence=0.90, seeds=[42, 43])
        self.assertEqual(ci["n_test"], 12)
        self.assertEqual(ci["n_bootstrap"], 20)
        self.assertEqual(ci["macro_f1"]["point"], 1.0)
        self.assertIn("42", ci["per_seed"])

        # Error cases
        with self.assertRaises(ValueError):
            bootstrap_ci(test_rows, preds, spec, n_bootstrap=5)
        with self.assertRaises(ValueError):
            bootstrap_ci(test_rows, preds, spec, confidence=1.2)
        with self.assertRaises(ValueError):
            bootstrap_ci(test_rows, preds, spec, confidence=0.4)
        with self.assertRaises(ValueError):
            bootstrap_ci(test_rows, preds, spec, seeds=[])
        with self.assertRaises(ValueError):
            bootstrap_ci(test_rows, preds, spec, seeds=["bad"])  # type: ignore[list-item]
        with self.assertRaises(ValueError):
            bootstrap_ci([], preds, spec)

    def test_mcnemar_test_branches(self):
        test_rows = [
            {"id": f"r{i}", "gold_label": "A"}
            for i in range(30)
        ]
        # Concordant: both correct
        preds_all_correct = [{"id": f"r{i}", "label": "A"} for i in range(30)]
        res_zero_disc = mcnemar_test(test_rows, preds_all_correct, preds_all_correct)
        self.assertEqual(res_zero_disc["discordant"], 0)
        self.assertEqual(res_zero_disc["p_value"], 1.0)
        self.assertTrue(res_zero_disc["exact"])

        # Exact binomial path: <= 25 discordant
        base_preds = [{"id": f"r{i}", "label": "A" if i < 15 else "B"} for i in range(30)]
        mod_preds = [{"id": f"r{i}", "label": "A" if i < 25 else "B"} for i in range(30)]
        res_exact = mcnemar_test(test_rows, base_preds, mod_preds, exact_threshold=25)
        self.assertTrue(res_exact["exact"])
        self.assertLessEqual(res_exact["discordant"], 25)

        # Chi-square approximation path: > 25 discordant
        base_preds_chi = [{"id": f"r{i}", "label": "B"} for i in range(30)]  # all wrong
        mod_preds_chi = [{"id": f"r{i}", "label": "A"} for i in range(30)]   # all right -> 30 discordant
        res_chi = mcnemar_test(test_rows, base_preds_chi, mod_preds_chi, exact_threshold=20)
        self.assertFalse(res_chi["exact"])
        self.assertGreater(res_chi["discordant"], 20)
        self.assertTrue(res_chi["significant_at_0_05"])

        # Validation errors
        with self.assertRaises(ValueError):
            mcnemar_test([], base_preds, mod_preds)
        with self.assertRaises(TypeError):
            mcnemar_test(test_rows, "not-list", mod_preds)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            mcnemar_test(test_rows, [{"id": "r1"}], mod_preds)
        with self.assertRaises(ValueError):
            mcnemar_test(test_rows, [{"id": "r1", "label": "A"}, {"id": "r1", "label": "A"}], mod_preds)
        with self.assertRaises(ValueError):
            mcnemar_test(test_rows, [{"id": "other_id", "label": "A"}], mod_preds)

    def test_load_predictor_dataset(self):
        data_file = self.root / "dataset.jsonl"
        data_file.write_text('{"id": "r1", "prompt": "p", "gold_label": "PASS"}\n', encoding="utf-8")

        # Valid load
        spec = {"dataset_path": "dataset.jsonl"}
        records, resolved = load_predictor_dataset(spec, self.root)
        self.assertEqual(len(records), 1)
        self.assertEqual(resolved, data_file.resolve())

        # Missing or non-clean relative path
        with self.assertRaises(ValueError):
            load_predictor_dataset({}, self.root)
        with self.assertRaises(ValueError):
            load_predictor_dataset({"dataset_path": "../escapes.jsonl"}, self.root)


if __name__ == "__main__":
    unittest.main()
