import hashlib
import importlib.util
import sys
import tempfile
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

    def test_adapter_hashes_reject_modified_configuration_and_legacy_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            weights = p / "adapter_model.safetensors"
            config = p / "adapter_config.json"
            weights.write_bytes(b"synthetic-weights")
            config.write_bytes(b'{"use_rslora":false}')
            manifest = {"adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                        "adapter_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}
            module.verify_adapter_hashes(p, manifest)
            config.write_bytes(b'{"use_rslora":true}')
            with self.assertRaises(ValueError):
                module.verify_adapter_hashes(p, manifest)
            with self.assertRaises(ValueError):
                module.verify_adapter_hashes(p, {"adapter_sha256": manifest["adapter_sha256"]})

    def test_undefined_precision_and_recall_are_null(self):
        row = {"expected": "matched", "output": "matched", "max_token_limit_hit": False}
        result = module.metrics([row])
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])


if __name__ == "__main__":
    unittest.main()
