"""Deep unit tests for hardonian_foundry.task_eval covering spec validation,
path hygiene, threshold validation, dataset validation, JSONL loading,
prediction map checks, dataset splits, scoring, and statistical evaluation."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hardonian_foundry.task_eval import (
    _is_clean_relative_path,
    _validate_predictions_map,
    _validate_thresholds,
    bootstrap_ci,
    load_jsonl,
    load_predictor_dataset,
    mcnemar_test,
    score_predictions,
    split_task_dataset,
    validate_spec,
    validate_task_dataset,
)


class TaskEvalDeepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_is_clean_relative_path(self) -> None:
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

    def test_validate_spec_errors(self) -> None:
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

        # Duplicate or whitespace labels
        errs_lbl_dup = validate_spec({**valid_spec, "labels": ["PASS", "PASS"]})
        self.assertTrue(any("labels must be unique" in e for e in errs_lbl_dup))

        errs_lbl_ws = validate_spec({**valid_spec, "labels": ["PASS ", "FAIL"]})
        self.assertTrue(any("clean nonempty string" in e for e in errs_lbl_ws))

        # Bad baseline_must_improve
        errs_bmi = validate_spec({**valid_spec, "baseline_must_improve": "not-bool"})
        self.assertTrue(any("baseline_must_improve must be a boolean" in e for e in errs_bmi))

        # Bad split_seed
        errs_seed = validate_spec({**valid_spec, "split_seed": True})
        self.assertTrue(any("split_seed must be an integer" in e for e in errs_seed))

        # Bad model subfields
        errs_mod_empty = validate_spec({**valid_spec, "model": {}})
        self.assertTrue(any("model must contain exactly" in e for e in errs_mod_empty))

        bad_model = dict(valid_spec["model"])
        bad_model["adapter_sha256"] = "not-64-hex"
        errs_sha = validate_spec({**valid_spec, "model": bad_model})
        self.assertTrue(any("adapter_sha256" in e for e in errs_sha))

        bad_model2 = dict(valid_spec["model"])
        bad_model2["base_model_id"] = "has\x00null"
        errs_id = validate_spec({**valid_spec, "model": bad_model2})
        self.assertTrue(any("base_model_id" in e for e in errs_id))

        # Bad dataset_path
        errs_dp = validate_spec({**valid_spec, "dataset_path": "../escape.jsonl"})
        self.assertTrue(any("dataset_path must be a relative" in e for e in errs_dp))

        # Bad notes
        errs_notes = validate_spec({**valid_spec, "notes": "bad\x01char"})
        self.assertTrue(any("notes must be a clean string" in e for e in errs_notes))

    def test_validate_thresholds(self) -> None:
        # Missing macro_f1_min
        self.assertTrue(any("macro_f1_min" in e for e in _validate_thresholds({}, ["A", "B"])))

        # Bad macro_f1_min range (< 0.8)
        self.assertTrue(any("macro_f1_min" in e for e in _validate_thresholds({"macro_f1_min": 0.5}, ["A", "B"])))

        # Unknown threshold fields
        self.assertTrue(any("unknown" in e for e in _validate_thresholds({"macro_f1_min": 0.85, "extra": 1}, ["A", "B"])))

        # Rate keys invalid
        errs_rates = _validate_thresholds(
            {"macro_f1_min": 0.85, "critical_fn_rate_max": 1.5, "invalid_output_rate_max": float("nan")},
            ["A", "B"],
        )
        self.assertTrue(any("critical_fn_rate_max" in e for e in errs_rates))
        self.assertTrue(any("invalid_output_rate_max" in e for e in errs_rates))

        # Latency p95 invalid
        errs_lat = _validate_thresholds({"macro_f1_min": 0.85, "latency_p95_ms_max": -1.0}, ["A", "B"])
        self.assertTrue(any("latency_p95_ms_max" in e for e in errs_lat))

        # Per class recall min bad label
        errs_pcr = _validate_thresholds(
            {"macro_f1_min": 0.85, "per_class_recall_min": {"INVALID_LABEL": 0.9}},
            ["A", "B"],
        )
        self.assertTrue(any("INVALID_LABEL" in e for e in errs_pcr))

        # Per class recall min float out of range
        errs_pcr_num = _validate_thresholds(
            {"macro_f1_min": 0.85, "per_class_recall_min": 1.5},
            ["A", "B"],
        )
        self.assertTrue(any("per_class_recall_min must be in [0.0, 1.0]" in e for e in errs_pcr_num))

        # Per class recall min dict val out of range
        errs_pcr_dict = _validate_thresholds(
            {"macro_f1_min": 0.85, "per_class_recall_min": {"A": 1.5}},
            ["A", "B"],
        )
        self.assertTrue(any("must be a number in [0.0, 1.0]" in e for e in errs_pcr_dict))

        # Per class recall min bad type
        errs_pcr_type = _validate_thresholds(
            {"macro_f1_min": 0.85, "per_class_recall_min": "invalid-type"},
            ["A", "B"],
        )
        self.assertTrue(any("must be a number in [0.0, 1.0] or a label map" in e for e in errs_pcr_type))

        # ood_challenge_predictions branches
        # Valid bool and int
        self.assertEqual(_validate_thresholds({"macro_f1_min": 0.85, "ood_challenge_predictions": True}, ["A", "B"]), [])
        self.assertEqual(_validate_thresholds({"macro_f1_min": 0.85, "ood_challenge_predictions": 10}, ["A", "B"]), [])

        # Bad string path
        errs_ood_str = _validate_thresholds({"macro_f1_min": 0.85, "ood_challenge_predictions": "../bad.jsonl"}, ["A", "B"])
        self.assertTrue(any("ood_challenge_predictions path must be relative" in e for e in errs_ood_str))

        # Dict ood
        errs_ood_dict = _validate_thresholds(
            {
                "macro_f1_min": 0.85,
                "ood_challenge_predictions": {
                    "unknown_key": 1,
                    "min_count": 0,
                    "macro_f1_min": 2.0,
                    "dataset_path": "../bad.jsonl",
                },
            },
            ["A", "B"],
        )
        self.assertTrue(any("unknown thresholds.ood_challenge_predictions fields" in e for e in errs_ood_dict))
        self.assertTrue(any("min_count must be an integer >= 1" in e for e in errs_ood_dict))
        self.assertTrue(any("macro_f1_min must be in [0.0, 1.0]" in e for e in errs_ood_dict))
        self.assertTrue(any("dataset_path must be relative" in e for e in errs_ood_dict))

        # Bad ood type
        errs_ood_type = _validate_thresholds({"macro_f1_min": 0.85, "ood_challenge_predictions": 12.34}, ["A", "B"])
        self.assertTrue(any("must be bool, positive int, relative path, or object" in e for e in errs_ood_type))

    def test_load_jsonl(self) -> None:
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

    def test_validate_task_dataset(self) -> None:
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

        # Non-dict row, missing fields, unknown fields, invalid types
        bad_records = [
            "not-a-dict",  # index 0: non-dict
            {"id": "r1"},  # index 1: missing required fields
            {"id": "r2", "prompt": "p2", "gold_label": "PASS", "critical": False, "extra_col": 1},
            {"id": "", "prompt": "p3", "gold_label": "PASS", "critical": False},  # bad id
            {"id": "r4", "prompt": "", "gold_label": "PASS", "critical": False},  # bad prompt
            {"id": "r5", "prompt": "p5", "gold_label": "UNKNOWN_LBL", "critical": False},  # bad gold
            {"id": "r6", "prompt": "p6", "gold_label": "PASS", "critical": "not-bool"},  # bad critical
            {"id": "r7", "prompt": "p7", "gold_label": "PASS", "critical": False, "family": ""},  # bad family
            {"id": "r8", "prompt": "p8", "gold_label": "PASS", "critical": False, "family": "famA"},
            {"id": "r9", "prompt": "p9", "gold_label": "PASS", "critical": False},  # family inconsistency
        ]
        res_bad = validate_task_dataset(bad_records, spec)
        self.assertEqual(res_bad["status"], "INVALID")
        self.assertTrue(any("is not an object" in e for e in res_bad["errors"]))
        self.assertTrue(any("missing required fields" in e for e in res_bad["errors"]))
        self.assertTrue(any("unknown fields" in e for e in res_bad["errors"]))
        self.assertTrue(any("id must be a" in e for e in res_bad["errors"]))
        self.assertTrue(any("prompt must be a" in e for e in res_bad["errors"]))
        self.assertTrue(any("not in spec labels" in e for e in res_bad["errors"]))
        self.assertTrue(any("critical must be boolean" in e for e in res_bad["errors"]))
        self.assertTrue(any("family must be a 1-64 char" in e for e in res_bad["errors"]))
        self.assertTrue(any("must be present on all records or none" in e for e in res_bad["errors"]))

    def test_split_task_dataset_branches(self) -> None:
        spec = {"split_seed": 42}

        # Records without family (use_family=False)
        records_no_fam = [
            {"id": f"r{i}", "prompt": f"prompt {i}", "gold_label": "PASS", "critical": False}
            for i in range(10)
        ]
        splits = split_task_dataset(records_no_fam, spec)
        self.assertIn("train", splits)
        self.assertIn("validation", splits)
        self.assertIn("test", splits)
        self.assertEqual(len(splits["train"]) + len(splits["validation"]) + len(splits["test"]), 10)

        # Under 3 groups -> raises ValueError
        under_three = [
            {"id": "r1", "prompt": "p1"},
            {"id": "r2", "prompt": "p2"},
        ]
        with self.assertRaises(ValueError) as ctx:
            split_task_dataset(under_three, spec)
        self.assertIn("need at least three", str(ctx.exception))

    def test_validate_predictions_map(self) -> None:
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

        # Predictions covering unknown extra id
        with self.assertRaises(ValueError) as ctx_extra:
            _validate_predictions_map(
                test_rows,
                [{"id": "r1", "label": "PASS"}, {"id": "r2", "label": "FAIL"}, {"id": "extra_r3", "label": "PASS"}],
                label_set,
            )
        self.assertIn("unknown ids", str(ctx_extra.exception))

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

    def test_score_predictions_type_error(self) -> None:
        with self.assertRaises(TypeError):
            score_predictions("not-a-sequence", [], {"labels": ["PASS", "FAIL"]})  # type: ignore[arg-type]

    def test_bootstrap_ci_valid_and_errors(self) -> None:
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

    def test_mcnemar_test_branches(self) -> None:
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

        # Exact binomial path: <= 25 discordant (with baseline-only correct)
        base_preds = [{"id": f"r{i}", "label": "A" if i < 20 else "B"} for i in range(30)]
        mod_preds = [{"id": f"r{i}", "label": "A" if i < 15 else "B"} for i in range(30)]
        res_exact = mcnemar_test(test_rows, base_preds, mod_preds, exact_threshold=25)
        self.assertTrue(res_exact["exact"])
        self.assertGreater(res_exact["baseline_only_correct"], 0)
        self.assertEqual(res_exact["model_only_correct"], 0)

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

    def test_load_predictor_dataset(self) -> None:
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

        # Escapes check via is_relative_to
        with mock.patch.object(Path, "is_relative_to", return_value=False):
            with self.assertRaises(ValueError) as ctx_esc:
                load_predictor_dataset(spec, self.root)
            self.assertIn("escapes the spec directory", str(ctx_esc.exception))


if __name__ == "__main__":
    unittest.main()
