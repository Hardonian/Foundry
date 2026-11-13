"""Deep unit tests for hardonian_foundry.training_utils covering device selection,
training hooks, batch collation, gradient accumulation windows, prefix tries, and percentiles."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from hardonian_foundry.training_utils import (
    DeviceSpec,
    DPOAdapterHook,
    ExactMatchVerifier,
    QLoRAAdapterHook,
    RLVRAdapterHook,
    SFTAdapterHook,
    get_training_hook,
    grad_accum_window_size,
    iter_micro_batches,
    percentile,
    select_device,
    should_step_grad_accum,
)


class TrainingUtilsDeepTest(unittest.TestCase):
    def test_select_device_cpu_and_invalid(self):
        spec = select_device("cpu")
        self.assertEqual(spec.device, "cpu")
        self.assertEqual(spec.backend, "cpu")
        self.assertEqual(spec.torch_dtype_name, "float32")
        self.assertFalse(spec.allow_gpu)

        with self.assertRaises(ValueError):
            select_device("unknown_backend_xyz")

    def test_select_device_cuda_mocked(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.is_bf16_supported.return_value = True

        spec = select_device("cuda", torch_module=mock_torch)
        self.assertEqual(spec.device, "cuda")
        self.assertEqual(spec.backend, "cuda")
        self.assertEqual(spec.torch_dtype_name, "bfloat16")
        self.assertTrue(spec.allow_gpu)

    def test_select_device_cuda_unavailable_raises(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with self.assertRaises(RuntimeError):
            select_device("cuda", torch_module=mock_torch)

    def test_training_hooks(self):
        sft = get_training_hook("sft", r=16, lora_alpha=32)
        self.assertIsInstance(sft, SFTAdapterHook)
        self.assertEqual(sft.r, 16)
        self.assertEqual(sft.lora_alpha, 32)
        kwargs = sft.lora_config_kwargs()
        self.assertEqual(kwargs["r"], 16)
        self.assertEqual(kwargs["bias"], "none")

        qlora = get_training_hook("qlora")
        self.assertIsInstance(qlora, QLoRAAdapterHook)
        self.assertTrue(qlora.load_in_4bit)

        # QLoRA on CPU device raises RuntimeError
        cpu_spec = DeviceSpec(device="cpu", backend="cpu", torch_dtype_name="float32", allow_gpu=False)
        with self.assertRaises(RuntimeError):
            qlora.ensure_supported(cpu_spec)

        dpo = get_training_hook("dpo", beta=0.2)
        self.assertIsInstance(dpo, DPOAdapterHook)
        self.assertEqual(dpo.beta, 0.2)
        with self.assertRaises(ValueError):
            dpo.validate_preferences([])
        with self.assertRaises(ValueError):
            dpo.validate_preferences([{"prompt": "p", "chosen": "", "rejected": "r"}])
        dpo.validate_preferences([{"prompt": "p", "chosen": "c", "rejected": "r"}])

        rlvr = get_training_hook("rlvr")
        self.assertIsInstance(rlvr, RLVRAdapterHook)
        scores = rlvr.score_completions(["p1"], ["LABEL_A"], ["LABEL_A"])
        self.assertEqual(scores, [1.0])
        scores_miss = rlvr.score_completions(["p1"], ["WRONG"], ["LABEL_A"])
        self.assertEqual(scores_miss, [0.0])

        with self.assertRaises(ValueError):
            get_training_hook("unsupported_method")

    def test_exact_match_verifier(self):
        v = ExactMatchVerifier()
        self.assertEqual(v.score("prompt", "  ALPHA. ", "ALPHA"), 1.0)
        self.assertEqual(v.score("prompt", "BETA", "ALPHA"), 0.0)
        self.assertEqual(v.score("prompt", "", "ALPHA"), 0.0)

    def test_grad_accum_helpers(self):
        # total=10, grad_accum=4 -> windows: [0..3](4), [4..7](4), [8..9](2)
        self.assertEqual(grad_accum_window_size(0, 10, 4), 4)
        self.assertEqual(grad_accum_window_size(3, 10, 4), 4)
        self.assertEqual(grad_accum_window_size(8, 10, 4), 2)
        self.assertEqual(grad_accum_window_size(9, 10, 4), 2)

        with self.assertRaises(ValueError):
            grad_accum_window_size(0, 10, 0)
        with self.assertRaises(ValueError):
            grad_accum_window_size(-1, 10, 4)

        # should_step_grad_accum
        self.assertFalse(should_step_grad_accum(0, 10, 4))
        self.assertFalse(should_step_grad_accum(2, 10, 4))
        self.assertTrue(should_step_grad_accum(3, 10, 4))  # 4th item (boundary)
        self.assertTrue(should_step_grad_accum(7, 10, 4))  # 8th item (boundary)
        self.assertTrue(should_step_grad_accum(9, 10, 4))  # 10th item (final batch)

    def test_iter_micro_batches(self):
        items = list(range(10))
        batches = list(iter_micro_batches(items, 4))
        self.assertEqual(batches, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])
        with self.assertRaises(ValueError):
            list(iter_micro_batches(items, 0))

    def test_percentile(self):
        self.assertIsNone(percentile([], 50))
        self.assertEqual(percentile([42.0], 50), 42.0)
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertEqual(percentile(vals, 0), 10.0)
        self.assertEqual(percentile(vals, 50), 30.0)
        self.assertEqual(percentile(vals, 100), 50.0)
        self.assertAlmostEqual(percentile(vals, 25), 20.0)
        with self.assertRaises(ValueError):
            percentile(vals, -1)
        with self.assertRaises(ValueError):
            percentile(vals, 101)


if __name__ == "__main__":
    unittest.main()
