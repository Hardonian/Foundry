"""Deep unit tests for hardonian_foundry.training_utils covering device selection,
training hooks, batch collation, gradient accumulation windows, prefix tries, and percentiles."""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Dual-runner torch guard (see test_case_study_scripts.py): plain unittest discover cannot
# honor pytest.importorskip; module-level unittest.SkipTest works under both runners.
if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch not installed (optional [train] extra)")

import torch  # noqa: E402

from hardonian_foundry.training_utils import (  # noqa: E402
    DeviceSpec,
    DPOAdapterHook,
    ExactMatchVerifier,
    LabelPrefixTrie,
    QLoRAAdapterHook,
    RLVRAdapterHook,
    SFTAdapterHook,
    acquire_training_lock,
    available_ram_bytes,
    batched_greedy_generate,
    check_disk_space,
    collate_causal_lm_batch,
    collate_left_padded_prompts,
    compute_dpo_loss,
    get_training_hook,
    git_blob_sha1,
    grad_accum_window_size,
    iter_micro_batches,
    model_registry,
    percentile,
    resolve_model_spec,
    select_device,
    set_offline_env,
    sha_bytes,
    sha_file,
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

    def test_hashing_and_env_helpers(self):
        self.assertEqual(sha_bytes(b"test"), "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08")
        self.assertEqual(git_blob_sha1(b"hello"), "b6fc4c620b67d95f953a5c1c1230aaab5db5a1b0")

        with tempfile.TemporaryDirectory() as tmp:
            sample_file = Path(tmp) / "sample.bin"
            sample_file.write_bytes(b"file content")
            self.assertEqual(sha_file(sample_file), sha_bytes(b"file content"))

        # set_offline_env
        with tempfile.TemporaryDirectory() as tmp_hf:
            updates = set_offline_env(tmp_hf, threads=2, allow_gpu=False)
            self.assertEqual(updates["HF_HUB_OFFLINE"], "1")  # returned dict matches applied env
            self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
            self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
            self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "2")

    def test_system_resource_checks(self):
        # available_ram_bytes on live platform
        ram = available_ram_bytes()
        self.assertIsInstance(ram, int)
        self.assertGreater(ram, 0)

        # check_disk_space
        with tempfile.TemporaryDirectory() as tmp:
            # Valid space requirement
            check_disk_space(tmp, min_free_bytes=1024)

            # Excessive space requirement raises ValueError
            with self.assertRaises(ValueError):
                check_disk_space(tmp, min_free_bytes=10**18)

            # acquire_training_lock
            lock_file = Path(tmp) / "train.lock"
            with acquire_training_lock(lock_file) as handle:
                self.assertIsNotNone(handle)

    def test_model_registry_and_resolution(self):
        reg = model_registry()
        self.assertIn("Qwen/Qwen2.5-Coder-0.5B-Instruct", reg)
        spec = resolve_model_spec("Qwen/Qwen2.5-Coder-0.5B-Instruct")
        self.assertEqual(spec.model_id, "Qwen/Qwen2.5-Coder-0.5B-Instruct")

        with self.assertRaises(ValueError):
            resolve_model_spec("nonexistent_model_id_xyz")

    def test_compute_dpo_loss(self):
        pi_c = torch.tensor([0.5, 0.8])
        pi_r = torch.tensor([0.2, 0.3])
        ref_c = torch.tensor([0.4, 0.7])
        ref_r = torch.tensor([0.3, 0.4])

        loss, c_rew, r_rew = compute_dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(len(c_rew), 2)
        self.assertEqual(len(r_rew), 2)

        with self.assertRaises(ValueError):
            compute_dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=-0.1)

    def test_collate_batches(self):
        # collate_causal_lm_batch
        examples = [
            ([1, 2, 3], [1, 2, 3]),
            ([4, 5], [4, 5]),
        ]
        batch = collate_causal_lm_batch(examples, pad_token_id=0)
        self.assertEqual(batch["input_ids"].shape, (2, 3))
        self.assertEqual(batch["labels"].shape, (2, 3))
        self.assertEqual(batch["attention_mask"].shape, (2, 3))

        with self.assertRaises(ValueError):
            collate_causal_lm_batch([], pad_token_id=0)
        with self.assertRaises(ValueError):
            collate_causal_lm_batch([([1, 2], [1])], pad_token_id=0)

        # collate_left_padded_prompts
        prompts = [[1, 2], [3, 4, 5]]
        p_ids, p_masks = collate_left_padded_prompts(prompts, pad_token_id=0)
        self.assertEqual(p_ids.shape, (2, 3))
        self.assertEqual(p_masks.shape, (2, 3))
        # Left-padding check
        self.assertEqual(p_ids[0, 0].item(), 0)

        with self.assertRaises(ValueError):
            collate_left_padded_prompts([], pad_token_id=0)

    def test_prefix_trie_and_greedy_generate(self):
        class MockTokenizer:
            eos_token_id = 99
            pad_token_id = 0

            def encode(self, text, add_special_tokens=False):
                if text == "PASS":
                    return [10, 11]
                if text == "FAIL":
                    return [20, 21]
                return [1]

            def decode(self, token_ids, skip_special_tokens=True):
                if token_ids == [10, 11]:
                    return "PASS"
                if token_ids == [20, 21]:
                    return "FAIL"
                return "OTHER"

        tok = MockTokenizer()
        trie = LabelPrefixTrie(["PASS", "FAIL"], tok, eos_token_id=99)
        self.assertEqual(sorted(trie.allowed_next_tokens([])), [10, 20])
        self.assertEqual(trie.allowed_next_tokens([10]), [11])
        self.assertEqual(trie.allowed_next_tokens([10, 11]), [99])
        self.assertEqual(trie.allowed_next_tokens([99]), [99])

        # Empty prompts list returns []
        self.assertEqual(batched_greedy_generate(None, tok, []), [])

        # Mock model for greedy generate
        class MockModel:
            def eval(self):
                pass

            def parameters(self):
                yield torch.nn.Parameter(torch.zeros(1))

            def generate(self, **kwargs):
                batch_size = kwargs["input_ids"].shape[0]
                # Return sequences: prompt + [10, 11, 99]
                seqs = torch.cat([kwargs["input_ids"], torch.tensor([[10, 11, 99]] * batch_size)], dim=1)
                mock_out = MagicMock()
                mock_out.sequences = seqs
                mock_out.scores = [torch.zeros(batch_size, 100)] * 3
                return mock_out

        gen_results = batched_greedy_generate(
            MockModel(),
            tok,
            [[1, 2], [3, 4, 5]],
            max_new_tokens=4,
            batch_size=2,
            label_trie=trie,
            return_confidence=True,
        )
        self.assertEqual(len(gen_results), 2)
        self.assertEqual(gen_results[0].text, "PASS")


if __name__ == "__main__":
    unittest.main()
