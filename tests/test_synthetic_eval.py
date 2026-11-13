import hashlib
import importlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

batched_greedy_generate = importlib.import_module("hardonian_foundry.training_utils").batched_greedy_generate

SCRIPT = SCRIPTS / "evaluate_synthetic_lora.py"
spec = importlib.util.spec_from_file_location("synthetic_eval", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
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
            manifest = {
                "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "adapter_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            }
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

    def test_batched_greedy_generate_handles_mixed_lengths_and_confidence(self):
        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 2
            vocab: ClassVar[dict[int, str]] = {10: "matched", 11: "amount", 12: "_mismatch"}

            def decode(self, ids, skip_special_tokens=True):
                return "".join(self.vocab.get(int(t), "") for t in ids if int(t) not in (0, 2))

        class FakeGenerateOutput:
            def __init__(self, sequences, scores):
                self.sequences = sequences
                self.scores = scores

        class FakeModel:
            def eval(self):
                return self

            def generate(self, input_ids, attention_mask=None, max_new_tokens=4, **_kwargs):
                batch_size = input_ids.shape[0]
                # First sequence stops early with EOS; second sequence exhausts max_new_tokens=2
                extra = torch.tensor([[10, 2], [11, 12]], dtype=torch.long)[:batch_size, :max_new_tokens]
                sequences = torch.cat([input_ids, extra], dim=1)
                scores = []
                for step in range(extra.shape[1]):
                    logits = torch.full((batch_size, 15), -10.0)
                    for row_idx in range(batch_size):
                        logits[row_idx, int(extra[row_idx, step])] = 5.0
                    scores.append(logits)
                return FakeGenerateOutput(sequences=sequences, scores=scores)

        results = batched_greedy_generate(
            FakeModel(),
            FakeTokenizer(),
            [[5, 6], [7, 8, 9]],
            max_new_tokens=2,
            batch_size=2,
            return_confidence=True,
            torch_module=torch,
            device="cpu",
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].text, "matched")
        self.assertFalse(results[0].max_token_limit_hit)
        self.assertEqual(results[1].text, "amount_mismatch")
        self.assertTrue(results[1].max_token_limit_hit)
        self.assertGreater(results[0].confidence, 0.99)
        self.assertGreater(results[1].confidence, 0.99)
        self.assertGreaterEqual(results[0].latency_ms, 0.0)

    def test_evaluate_cli_rejects_nonpositive_batch_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--hf-home",
                    tmp,
                    "--run-dir",
                    tmp,
                    "--output-root",
                    tmp,
                    "--batch-size",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("--batch-size", res.stderr)


if __name__ == "__main__":
    unittest.main()
