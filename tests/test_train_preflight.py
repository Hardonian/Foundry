import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_synthetic_lora.py"
spec = importlib.util.spec_from_file_location("synthetic_lora", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TrainingPreflightTest(unittest.TestCase):
    def test_synthetic_holdout_is_disjoint_and_deterministic(self):
        data = module.examples()
        self.assertEqual(data, module.examples())
        self.assertEqual((len(data["train"]), len(data["holdout"])), (12, 4))
        self.assertFalse({x["prompt"] for x in data["train"]} & {x["prompt"] for x in data["holdout"]})
        self.assertTrue(all(x["id"].startswith("synthetic-") for rows in data.values() for x in rows))
        self.assertEqual(len(module.sha_bytes(json.dumps(data).encode())), 64)

    def test_insecure_output_fails_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "open"
            root.mkdir(mode=0o755)
            root.chmod(0o755)
            result = subprocess.run([sys.executable, str(SCRIPT), "--hf-home", tmp, "--output-root", str(root)],
                                    capture_output=True, text=True, timeout=10, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("owner-only", result.stderr)
            self.assertEqual(list(root.iterdir()), [])

    def test_missing_model_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(SCRIPT), "--hf-home", tmp, "--output-root", tmp],
                                    capture_output=True, text=True, timeout=10, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no download attempted", result.stderr)
            self.assertEqual(os.listdir(tmp), [])

    def test_steps_out_of_range_rejected(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--hf-home", ".", "--output-root", ".", "--steps", "21"],
                                capture_output=True, text=True, timeout=10, check=False)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
