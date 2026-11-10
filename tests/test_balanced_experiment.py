import importlib.util
import re
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("balanced_experiment", SCRIPTS / "experiment_balanced_lora.py")
experiment = importlib.util.module_from_spec(spec)
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

    def test_missing_model_is_explicit_preflight_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            usage = type("Usage", (), {"free": 100 * 1024**3, "used": 1, "total": 100 * 1024**3})()
            with (patch.object(experiment.shutil, "disk_usage", return_value=usage),
                  patch.object(experiment.os, "sysconf", return_value=10 * 1024**3),
                  self.assertRaisesRegex(ValueError, "required pinned model files are missing")):
                experiment.preflight(root, root)

    def test_data_hash_is_content_sensitive(self):
        data = experiment.build_datasets()
        old = experiment.digest_json(data["test"])
        data["test"][0]["answer"] = "changed"
        self.assertNotEqual(old, experiment.digest_json(data["test"]))


if __name__ == "__main__":
    unittest.main()
