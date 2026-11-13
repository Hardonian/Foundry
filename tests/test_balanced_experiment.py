import importlib
import importlib.util
import re
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

refinery = importlib.import_module("hardonian_foundry.refinery")
task_eval = importlib.import_module("hardonian_foundry.task_eval")

spec = importlib.util.spec_from_file_location("balanced_experiment", SCRIPTS / "experiment_balanced_lora.py")
experiment = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(experiment)


class BalancedExperimentTest(unittest.TestCase):
    def test_reproducible_balanced_group_disjoint_splits(self):
        data = experiment.build_datasets()
        self.assertEqual(data, experiment.build_datasets())
        self.assertEqual([len(data[k]) for k in ("train", "validation", "test")], [48, 16, 24])
        groups = [{r["group"] for r in data[k]} for k in ("train", "validation", "test")]
        self.assertTrue(all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)))
        for name, rows in data.items():
            counts = Counter(r["answer"] for r in rows)
            self.assertEqual(counts["matched"], counts["amount_mismatch"], name)
            self.assertEqual(len({r["prompt"] for r in rows}), len(rows))
            self.assertEqual(len({r["id"] for r in rows}), len(rows))
            self.assertTrue(all(r["id"].startswith("synthetic-") for r in rows))
            for row in rows:
                amounts = [int(value) for value in re.findall(r"\d+", row["prompt"])]
                self.assertEqual(len(amounts), 2)
                self.assertEqual(row["answer"], "matched" if amounts[0] == amounts[1] else "amount_mismatch")

    def test_numeric_ranges_overlap_across_splits(self):
        data = experiment.build_datasets()
        train_vals = [int(r["group"]) for r in data["train"]]
        val_vals = [int(r["group"]) for r in data["validation"]]
        test_vals = [int(r["group"]) for r in data["test"]]
        # Ensures the old 1100/4100/7100 range shortcut leak is fixed
        self.assertLess(min(test_vals), max(train_vals))
        self.assertLess(min(train_vals), max(test_vals))
        self.assertLess(min(val_vals), max(train_vals))
        self.assertLess(min(train_vals), max(val_vals))

    def test_missing_model_is_explicit_preflight_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            usage = type("Usage", (), {"free": 100 * 1024**3, "used": 1, "total": 100 * 1024**3})()
            with (
                patch.object(experiment.shutil, "disk_usage", return_value=usage),
                patch.object(experiment, "available_ram_bytes", return_value=10 * 1024**3),
                self.assertRaisesRegex(ValueError, "required pinned model files are missing"),
            ):
                experiment.preflight(root, root)

    def test_data_hash_is_content_sensitive(self):
        data = experiment.build_datasets()
        old = experiment.digest_json(data["test"])
        data["test"][0]["answer"] = "changed"
        self.assertNotEqual(old, experiment.digest_json(data["test"]))

    def test_emit_task_artifacts_is_leak_free_and_compatible_with_refinery(self):
        datasets = experiment.build_datasets()
        base_cases = [
            {"id": row["id"], "expected": row["answer"], "output": "matched"}
            for row in datasets["test"]
        ]
        adapter_cases = [
            {"id": row["id"], "expected": row["answer"], "output": row["answer"]}
            for row in datasets["test"]
        ]
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "task"
            out = experiment.emit_task_artifacts(
                task_dir,
                datasets,
                base_cases=base_cases,
                adapter_cases=adapter_cases,
                adapter_sha256="c" * 64,
            )
            self.assertTrue((task_dir / "spec.json").is_file())
            self.assertTrue((task_dir / "data.jsonl").is_file())
            self.assertTrue((task_dir / "baseline.jsonl").is_file())
            self.assertTrue((task_dir / "model.jsonl").is_file())

            splits = task_eval.split_task_dataset(out["dataset_records"], out["spec"])
            test_ids = {r["id"] for r in splits["test"]}
            orig_test_ids = {r["id"] for r in datasets["test"]}
            self.assertTrue(test_ids <= orig_test_ids)
            self.assertFalse(test_ids & {r["id"] for r in datasets["train"]})

            runs_root = str(Path(tmp) / "runs")
            run = refinery.create_task_run(
                task_spec=out["spec"],
                dataset_records=out["dataset_records"],
                baseline_predictions=out["baseline_predictions"],
                model_predictions=out["model_predictions"],
                output_root=runs_root,
            )
            self.assertEqual(run["status"], "COMPLETED")
            self.assertEqual(run["evaluation"]["task"]["model_metrics"]["macro_f1"], 1.0)
            promo = refinery.evaluate_promotion(run["run_id"], output_root=runs_root)
            self.assertEqual(
                promo["missing_evidence"],
                ["cryptographic_promotion_signature", "model_licensing_manifest"],
            )


if __name__ == "__main__":
    unittest.main()
