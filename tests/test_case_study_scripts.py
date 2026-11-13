import contextlib
import importlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")  # optional train extra; skip module without torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CASE_DIR = ROOT / "examples" / "case-study-exception-codes"
for candidate in (SRC, CASE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

signing = importlib.import_module("hardonian_foundry.signing")
task_eval = importlib.import_module("hardonian_foundry.task_eval")
_tu = importlib.import_module("hardonian_foundry.training_utils")
ExactMatchVerifier = _tu.ExactMatchVerifier
LabelPrefixTrie = _tu.LabelPrefixTrie
PrefixTrieLogitsProcessor = _tu.PrefixTrieLogitsProcessor
collate_causal_lm_batch = _tu.collate_causal_lm_batch
collate_left_padded_prompts = _tu.collate_left_padded_prompts
compute_dpo_loss = _tu.compute_dpo_loss
get_training_hook = _tu.get_training_hook
git_blob_sha1 = _tu.git_blob_sha1
grad_accum_window_size = _tu.grad_accum_window_size
percentile = _tu.percentile
select_device = _tu.select_device
sha_bytes = _tu.sha_bytes
should_step_grad_accum = _tu.should_step_grad_accum


def _load_case_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CASE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


train_lora = _load_case_module("case_train_lora", "train_lora.py")
model_predict = _load_case_module("case_model_predict", "model_predict.py")
finalize_spec_mod = _load_case_module("case_finalize_spec", "finalize_spec.py")
baseline_predict = _load_case_module("case_baseline_predict", "baseline_predict.py")
show_pubkey = _load_case_module("case_show_pubkey", "show_pubkey.py")


class _FakeTokenizer:
    """Deterministic lightweight tokenizer supporting chat templates and multi-token labels."""

    pad_token_id = 0
    eos_token_id = 2

    _SPECIAL: ClassVar[dict[str, list[int]]] = {
        "DUPLICATE_RECORD_CONFIRMED": [10, 11],
        "AMOUNT_VARIANCE_ACCEPTED": [12, 13],
        "UNDETERMINED": [14],
        "junk_output": [19],
    }
    _DECODE: ClassVar[dict[tuple[int, ...], str]] = {
        (10, 11): "DUPLICATE_RECORD_CONFIRMED",
        (12, 13): "AMOUNT_VARIANCE_ACCEPTED",
        (14,): "UNDETERMINED",
        (19,): "junk_output",
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        stripped = text.strip()
        if stripped in self._SPECIAL:
            return list(self._SPECIAL[stripped])
        tokens: list[int] = []
        for word in stripped.split():
            if word in self._SPECIAL:
                tokens.extend(self._SPECIAL[word])
            else:
                tokens.append(20 + (sum(ord(ch) for ch in word) % 35))
        return tokens if tokens else [20]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        ids = [3]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                ids.extend([4, *self.encode(content)])
            elif role == "user":
                ids.extend([5, *self.encode(content)])
            elif role == "assistant":
                ids.extend([6, *self.encode(content), self.eos_token_id])
        if add_generation_prompt:
            ids.append(6)
        return ids

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        filtered = [int(t) for t in token_ids if not (skip_special_tokens and int(t) in (0, 2))]
        tup = tuple(filtered)
        if tup in self._DECODE:
            return self._DECODE[tup]
        parts: list[str] = []
        idx = 0
        while idx < len(filtered):
            if idx + 1 < len(filtered) and (filtered[idx], filtered[idx + 1]) in self._DECODE:
                parts.append(self._DECODE[(filtered[idx], filtered[idx + 1])])
                idx += 2
            elif (filtered[idx],) in self._DECODE:
                parts.append(self._DECODE[(filtered[idx],)])
                idx += 1
            else:
                parts.append(f"tok{filtered[idx]}")
                idx += 1
        return " ".join(parts)


class _FakeCausalLM(torch.nn.Module):
    """Tiny real-tensor causal LM preferring junk token 19 unless constrained by logits_processor."""

    def __init__(self, target_modules: tuple[str, ...] = ("q_proj", "v_proj")) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(64, 8)
        self.proj = torch.nn.Linear(8, 64)
        self.target_modules = list(target_modules)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.embed(input_ids)
        logits = self.proj(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
            shift_labels = labels[:, 1:].contiguous().view(-1)
            mask = shift_labels != -100
            if mask.any():
                loss = torch.nn.functional.cross_entropy(shift_logits[mask], shift_labels[mask])
            else:
                loss = logits.sum() * 0.0
        return SimpleNamespace(loss=loss, logits=logits)

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens: int = 12,
        do_sample: bool = False,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        logits_processor=None,
        return_dict_in_generate: bool = True,
        output_scores: bool = True,
        **_kwargs,
    ):
        batch_size = int(input_ids.shape[0])
        cur = input_ids.clone()
        scores: list[torch.Tensor] = []
        finished = [False] * batch_size

        for step in range(max_new_tokens):
            step_logits = torch.full((batch_size, 64), -8.0, device=cur.device)
            if step == 0:
                # Junk token 19 beats allowlist starts (10, 12, 14) unless prefix-trie masks 19 to -inf
                step_logits[:, 19] = 6.0
                step_logits[:, 10] = 4.0
                step_logits[:, 12] = 2.0
                step_logits[:, 14] = 1.0
            elif step == 1:
                step_logits[:, 11] = 5.0
                step_logits[:, 13] = 5.0
                step_logits[:, eos_token_id] = 4.5
            else:
                step_logits[:, eos_token_id] = 7.0

            if logits_processor is not None:
                for proc in logits_processor:
                    step_logits = proc(cur, step_logits)

            scores.append(step_logits)
            next_tok = step_logits.argmax(dim=-1, keepdim=True)
            for row_idx in range(batch_size):
                if finished[row_idx]:
                    next_tok[row_idx, 0] = eos_token_id
                elif int(next_tok[row_idx, 0]) == eos_token_id:
                    finished[row_idx] = True
            cur = torch.cat([cur, next_tok], dim=1)
            if all(finished):
                break

        return SimpleNamespace(sequences=cur, scores=scores)

    def save_pretrained(self, out_dir: str, safe_serialization: bool = True) -> None:
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        weight_sum = float(self.proj.weight.detach().sum())
        (target / "adapter_model.safetensors").write_bytes(f"adapter:{weight_sum:.6f}".encode())
        cfg = {"target_modules": self.target_modules, "safe_serialization": safe_serialization}
        (target / "adapter_config.json").write_text(json.dumps(cfg, sort_keys=True), encoding="utf-8")


@contextlib.contextmanager
def _patch_modules(overrides: dict[str, types.ModuleType]):
    """Temporarily replace only explicit keys in `sys.modules` without removing lazy Torch submodules."""
    sentinel = object()
    saved = {name: sys.modules.get(name, sentinel) for name in overrides}
    sys.modules.update(overrides)
    try:
        yield
    finally:
        for name, orig in saved.items():
            if orig is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig  # type: ignore[assignment]


def _build_fake_ml_modules():
    fake_transformers = types.ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **_kwargs):
            return _FakeTokenizer()

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path, **_kwargs):
            return _FakeCausalLM()

    class _LogitsProcessorList(list):
        pass

    fake_transformers.AutoTokenizer = _AutoTokenizer
    fake_transformers.AutoModelForCausalLM = _AutoModelForCausalLM
    fake_transformers.LogitsProcessorList = _LogitsProcessorList

    fake_peft = types.ModuleType("peft")

    class _LoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _PeftModel:
        @staticmethod
        def from_pretrained(model, adapter_dir, is_trainable=False, local_files_only=True):
            for param in model.parameters():
                param.requires_grad = bool(is_trainable)
            model.loaded_adapter = str(adapter_dir)
            return model

    def _get_peft_model(model, peft_cfg):
        model.target_modules = list(peft_cfg.kwargs.get("target_modules", ["q_proj", "v_proj"]))
        return model

    fake_peft.LoraConfig = _LoraConfig
    fake_peft.PeftModel = _PeftModel
    fake_peft.get_peft_model = _get_peft_model
    fake_peft.__version__ = "0.0-mock"
    fake_transformers.__version__ = "0.0-mock"
    return {"transformers": fake_transformers, "peft": fake_peft}


def _make_mock_snapshot_and_registry(root: Path) -> tuple[Path, Path, dict[str, str]]:
    hf_home = root / "hf"
    model_id = "org/mock-tiny-coder"
    revision = "rev-mock-1"
    snap = hf_home / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    weights = b"mock-base-safetensors-weights-v1"
    cfg_bytes = b'{"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"}\n'
    (snap / "model.safetensors").write_bytes(weights)
    (snap / "config.json").write_bytes(cfg_bytes)

    registry_path = root / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "models": {
                    model_id: {
                        "revision": revision,
                        "model_sha256": sha_bytes(weights),
                        "blobs": {"config.json": git_blob_sha1(cfg_bytes)},
                        "lora_target_modules": ["q_proj", "v_proj"],
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    meta = {"model_id": model_id, "revision": revision, "weights_sha256": sha_bytes(weights)}
    return hf_home, registry_path, meta


def _make_case_task_dir(root: Path, model_id: str, revision: str) -> Path:
    task_dir = root / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    labels = ["DUPLICATE_RECORD_CONFIRMED", "AMOUNT_VARIANCE_ACCEPTED", "UNDETERMINED"]
    spec = {
        "schema_version": "1.0",
        "task_id": "mock-exception-codes",
        "description": "Synthetic mock exception-code task for unit tests.",
        "labels": labels,
        "thresholds": {"macro_f1_min": 0.8},
        "baseline_must_improve": False,
        "split_seed": 20261110,
        "model": {
            "base_model_id": model_id,
            "revision": revision,
            "adapter_sha256": "0" * 64,
            "adapter_name": "mock-adapter",
        },
        "dataset_path": "data.jsonl",
    }
    (task_dir / "spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    notes = [
        ("Duplicate settlement entry posted twice", "DUPLICATE_RECORD_CONFIRMED"),
        ("Minor fee rounding difference of two cents", "AMOUNT_VARIANCE_ACCEPTED"),
        ("Ambiguous batch memo without clearing detail", "UNDETERMINED"),
    ]
    rows = []
    for idx in range(18):
        text, label = notes[idx % len(notes)]
        rows.append(
            {
                "id": f"ex-{idx:03d}",
                "prompt": f"Item {idx}: {text}.",
                "gold_label": label,
                "critical": label == "DUPLICATE_RECORD_CONFIRMED",
                "family": f"fam-{idx:02d}",
            }
        )
    with (task_dir / "data.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return task_dir


class CaseStudyToLabelTest(unittest.TestCase):
    def test_to_label_exact_normalized_and_prefix_collision_guard(self):
        allowlist = ["FOO", "FOO_BAR", "FOO_BAR_BAZ", "UNDETERMINED"]
        self.assertEqual(model_predict.to_label("FOO", allowlist), ("FOO", "exact"))
        self.assertEqual(model_predict.to_label("FOO_BAR", allowlist), ("FOO_BAR", "exact"))
        self.assertEqual(model_predict.to_label("UNDETERMINED", allowlist), ("UNDETERMINED", "exact"))

        # Longest-first prevents FOO stealing FOO_BAR prose, and separator boundary rejects FOO_BAZ/FOOBAR
        self.assertEqual(model_predict.to_label("FOO_BAR because reason", allowlist), ("FOO_BAR", "normalized"))
        self.assertEqual(model_predict.to_label("foo-bar-baz", allowlist), ("FOO_BAR_BAZ", "normalized"))
        self.assertEqual(model_predict.to_label("FOO - short explanation", allowlist), ("FOO", "normalized"))
        self.assertEqual(model_predict.to_label("FOO_BAZ", allowlist), ("UNDETERMINED", "undetermined"))
        self.assertEqual(model_predict.to_label("FOOBAR", allowlist), ("UNDETERMINED", "undetermined"))
        self.assertEqual(model_predict.to_label("", allowlist), ("UNDETERMINED", "undetermined"))
        self.assertEqual(model_predict.to_label("not sure at all", allowlist), ("UNDETERMINED", "undetermined"))


class TrainingHelpersAndHooksTest(unittest.TestCase):
    def test_grad_accum_window_math_and_numerical_equivalence(self):
        # total_batches=5, grad_accum=2 -> windows (2, 2, 1), steps on idx 1, 3, 4
        windows = [grad_accum_window_size(i, 5, 2) for i in range(5)]
        steps = [should_step_grad_accum(i, 5, 2) for i in range(5)]
        self.assertEqual(windows, [2, 2, 2, 2, 1])
        self.assertEqual(steps, [False, True, False, True, True])

        # Exact equivalence of rescaled remainder window vs direct batch-mean SGD
        x = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0], [-2.0, 1.0], [2.0, 3.0]])
        y = torch.tensor([[1.0], [2.0], [3.0], [0.0], [-1.0]])

        direct = torch.nn.Linear(2, 1, bias=False)
        accum = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            direct.weight.copy_(torch.tensor([[0.25, -0.5]]))
            accum.weight.copy_(direct.weight)

        opt_direct = torch.optim.SGD(direct.parameters(), lr=0.1)
        opt_accum = torch.optim.SGD(accum.parameters(), lr=0.1)

        # Direct updates over [0:2], [2:4], [4:5]
        for start, end in ((0, 2), (2, 4), (4, 5)):
            opt_direct.zero_grad()
            loss = ((direct(x[start:end]) - y[start:end]) ** 2).mean()
            loss.backward()
            opt_direct.step()

        # Micro-batch=1, grad_accum=2 over 5 items using grad_accum_window_size
        opt_accum.zero_grad()
        for b_idx in range(5):
            win = grad_accum_window_size(b_idx, 5, 2)
            loss = ((accum(x[b_idx : b_idx + 1]) - y[b_idx : b_idx + 1]) ** 2).mean()
            (loss / float(win)).backward()
            if should_step_grad_accum(b_idx, 5, 2):
                opt_accum.step()
                opt_accum.zero_grad()

        self.assertTrue(torch.allclose(direct.weight, accum.weight, atol=1e-6))

    def test_encode_classification_example_masks_prompt_and_fails_on_truncation(self):
        tok = _FakeTokenizer()
        row = {"id": "ex-1", "prompt": "Duplicate check was posted twice.", "gold_label": "DUPLICATE_RECORD_CONFIRMED"}
        menu = "DUPLICATE_RECORD_CONFIRMED, UNDETERMINED"
        enc = train_lora.encode_classification_example(tok, row, menu, max_seq=64)
        self.assertEqual(len(enc["input_ids"]), len(enc["labels"]))
        self.assertEqual(enc["labels"][-3:], [10, 11, 2])
        self.assertTrue(all(lbl == -100 for lbl in enc["labels"][:-3]))

        with self.assertRaisesRegex(ValueError, "exceeds max_seq"):
            train_lora.encode_classification_example(tok, row, menu, max_seq=6)

    def test_lr_schedulers_padding_trie_percentile_and_training_hooks(self):
        param = torch.nn.Parameter(torch.tensor([1.0]))
        for sched_name in ("cosine", "linear", "constant"):
            opt = torch.optim.SGD([param], lr=1.0)
            sched = train_lora.build_lr_scheduler(
                opt, total_steps=10, scheduler_type=sched_name, warmup_ratio=0.2, torch_module=torch
            )
            lrs = []
            for _ in range(10):
                lrs.append(opt.param_groups[0]["lr"])
                opt.step()
                sched.step()
            self.assertGreater(lrs[1], lrs[0])
            if sched_name in ("cosine", "linear"):
                self.assertLess(lrs[-1], lrs[1])
            else:
                self.assertAlmostEqual(lrs[-1], 1.0, places=5)

        causal = collate_causal_lm_batch(
            [([1, 2, 3], [-100, 2, 3]), ([4, 5], [-100, 5])],
            pad_token_id=0,
            torch_module=torch,
        )
        self.assertEqual(causal["input_ids"].tolist(), [[1, 2, 3], [4, 5, 0]])
        self.assertEqual(causal["labels"].tolist(), [[-100, 2, 3], [-100, 5, -100]])
        self.assertEqual(causal["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])

        left_ids, left_mask = collate_left_padded_prompts([[5, 6], [7, 8, 9]], pad_token_id=0, torch_module=torch)
        self.assertEqual(left_ids.tolist(), [[0, 5, 6], [7, 8, 9]])
        self.assertEqual(left_mask.tolist(), [[0, 1, 1], [1, 1, 1]])

        tok = _FakeTokenizer()
        trie = LabelPrefixTrie(["DUPLICATE_RECORD_CONFIRMED", "UNDETERMINED"], tok)
        self.assertEqual(trie.allowed_next_tokens([]), [10, 14])
        self.assertEqual(trie.allowed_next_tokens([10]), [11])
        self.assertEqual(trie.allowed_next_tokens([10, 11]), [2])
        proc = PrefixTrieLogitsProcessor(trie, prompt_length=2)
        scores = torch.zeros((1, 64))
        masked = proc(torch.tensor([[3, 4]]), scores)
        self.assertEqual(float(masked[0, 19]), float("-inf"))
        self.assertEqual(float(masked[0, 10]), 0.0)

        self.assertIsNone(percentile([], 95.0))
        self.assertAlmostEqual(percentile([10.0, 20.0, 30.0, 40.0], 50.0), 25.0)

        qlora_hook = get_training_hook("qlora")
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            qlora_hook.ensure_supported(select_device("cpu"))

        dpo_hook = get_training_hook("dpo", beta=0.2)
        dpo_hook.validate_preferences([{"prompt": "p", "chosen": "c", "rejected": "r"}])
        loss, c_rew, r_rew = compute_dpo_loss(
            torch.tensor([-0.5]), torch.tensor([-2.0]), torch.tensor([-1.0]), torch.tensor([-1.0]), beta=0.2
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(c_rew[0]), float(r_rew[0]))

        rlvr_hook = get_training_hook("rlvr", verifier=ExactMatchVerifier())
        self.assertEqual(rlvr_hook.score_completions(["p1", "p2"], ["FOO", "BAR"], ["FOO", "FOO"]), [1.0, 0.0])
        pg_loss = rlvr_hook.compute_policy_gradient_loss(
            torch.tensor([-0.3, -1.2], requires_grad=True),
            torch.tensor([1.0, 0.0]),
            ref_logps=torch.tensor([-0.4, -1.0]),
        )
        self.assertTrue(torch.isfinite(pg_loss))


class CaseStudyScriptsEndToEndTest(unittest.TestCase):
    def test_train_finalize_predict_provenance_and_resume_with_mocked_models(self):
        fake_mods = _build_fake_ml_modules()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_home, registry_path, _meta = _make_mock_snapshot_and_registry(root)
            task_dir = _make_case_task_dir(root, "org/mock-tiny-coder", "rev-mock-1")
            adapter_dir = root / "adapter_out"
            pred_out = root / "model_preds.jsonl"

            with _patch_modules(fake_mods), contextlib.redirect_stdout(io.StringIO()):
                # 1. QLoRA must fail closed on CPU without silent fallback
                with self.assertRaises(SystemExit):
                    train_lora.main(
                        [
                            "--hf-home",
                            str(hf_home),
                            "--task-dir",
                            str(task_dir),
                            "--output",
                            str(root / "qlora_out"),
                            "--method",
                            "qlora",
                            "--device",
                            "cpu",
                            "--model-registry",
                            str(registry_path),
                        ]
                    )

                # 2. Train LoRA for 1 epoch with checkpoints saved
                rc = train_lora.main(
                    [
                        "--hf-home",
                        str(hf_home),
                        "--task-dir",
                        str(task_dir),
                        "--output",
                        str(adapter_dir),
                        "--epochs",
                        "1",
                        "--batch-size",
                        "2",
                        "--grad-accum",
                        "2",
                        "--max-seq",
                        "64",
                        "--threads",
                        "1",
                        "--save-checkpoints",
                        "--model-registry",
                        str(registry_path),
                    ]
                )
                self.assertEqual(rc, 0)
                manifest = json.loads((adapter_dir / "train_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["best_epoch"], 1)
                self.assertGreater(manifest["optimizer_steps"], 0)
                self.assertTrue((adapter_dir / "checkpoints" / "epoch-1" / "checkpoint_state.json").is_file())

                # 3. Prediction before finalizing spec must fail closed
                with self.assertRaisesRegex(SystemExit, "unfinalized"):
                    model_predict.main(
                        [
                            "--hf-home",
                            str(hf_home),
                            "--task-dir",
                            str(task_dir),
                            "--adapter-dir",
                            str(adapter_dir),
                            "--out",
                            str(pred_out),
                            "--threads",
                            "1",
                            "--model-registry",
                            str(registry_path),
                        ]
                    )

                # 4. Finalize spec, verify idempotency and --verify
                fin1 = finalize_spec_mod.finalize_spec(task_dir, adapter_dir)
                self.assertEqual(fin1["status"], "finalized")
                fin2 = finalize_spec_mod.finalize_spec(task_dir, adapter_dir)
                self.assertEqual(fin2["status"], "already_finalized")
                self.assertEqual(
                    finalize_spec_mod.finalize_spec(task_dir, adapter_dir, verify_only=True)["status"],
                    "verified",
                )

                # 5. Unconstrained prediction maps junk token 19 cleanly to UNDETERMINED
                rc_unc = model_predict.main(
                    [
                        "--hf-home",
                        str(hf_home),
                        "--task-dir",
                        str(task_dir),
                        "--adapter-dir",
                        str(adapter_dir),
                        "--out",
                        str(pred_out),
                        "--batch-size",
                        "2",
                        "--threads",
                        "1",
                        "--model-registry",
                        str(registry_path),
                    ]
                )
                self.assertEqual(rc_unc, 0)
                pm_unc = json.loads((pred_out.parent / "predict_manifest.json").read_text(encoding="utf-8"))
                self.assertGreater(pm_unc["mapped_to_undetermined"], 0)
                self.assertEqual(pm_unc["exact_allowlist_outputs"], 0)
                self.assertIsNotNone(pm_unc["latency_p95_ms"])

                # 6. Constrained prefix-trie prediction yields exact allowlist predictions
                rc_con = model_predict.main(
                    [
                        "--hf-home",
                        str(hf_home),
                        "--task-dir",
                        str(task_dir),
                        "--adapter-dir",
                        str(adapter_dir),
                        "--out",
                        str(pred_out),
                        "--batch-size",
                        "2",
                        "--threads",
                        "1",
                        "--constrained",
                        "--model-registry",
                        str(registry_path),
                    ]
                )
                self.assertEqual(rc_con, 0)
                pm_con = json.loads((pred_out.parent / "predict_manifest.json").read_text(encoding="utf-8"))
                self.assertTrue(pm_con["constrained"])
                self.assertEqual(pm_con["exact_allowlist_outputs"], pm_con["test_records"])
                self.assertEqual(pm_con["mapped_to_undetermined"], 0)
                pred_rows = [json.loads(ln) for ln in pred_out.read_text(encoding="utf-8").splitlines() if ln.strip()]
                self.assertEqual(len(pred_rows), pm_con["test_records"])
                self.assertTrue(
                    all(
                        0.0 <= row["confidence"] <= 1.0
                        and row["latency_ms"] >= 0.0
                        and isinstance(row["raw_invalid"], bool)
                        for row in pred_rows
                    )
                )
                spec = json.loads((task_dir / "spec.json").read_text(encoding="utf-8"))
                all_rows = task_eval.load_jsonl(task_dir / "data.jsonl")
                test_split_rows = task_eval.split_task_dataset(all_rows, spec)["test"]
                core_preds = [{"id": row["id"], "label": row["label"]} for row in pred_rows]
                score = task_eval.score_predictions(test_split_rows, core_preds, spec)
                self.assertEqual(score["test_count"], len(pred_rows))

                # 7. --base-only zero-shot mode works without an adapter
                rc_base = model_predict.main(
                    [
                        "--hf-home",
                        str(hf_home),
                        "--task-dir",
                        str(task_dir),
                        "--base-only",
                        "--constrained",
                        "--out",
                        str(root / "base_preds.jsonl"),
                        "--threads",
                        "1",
                        "--model-registry",
                        str(registry_path),
                    ]
                )
                self.assertEqual(rc_base, 0)

                # 8. Tampering with spec thresholds after training fails provenance check
                tampered = json.loads((task_dir / "spec.json").read_text(encoding="utf-8"))
                tampered["thresholds"]["macro_f1_min"] = 0.99
                (task_dir / "spec.json").write_text(json.dumps(tampered, indent=2), encoding="utf-8")
                with self.assertRaisesRegex(SystemExit, "spec changed"):
                    model_predict.main(
                        [
                            "--hf-home",
                            str(hf_home),
                            "--task-dir",
                            str(task_dir),
                            "--adapter-dir",
                            str(adapter_dir),
                            "--out",
                            str(pred_out),
                            "--threads",
                            "1",
                            "--model-registry",
                            str(registry_path),
                        ]
                    )

                # 9. Resume refuses tampered spec, then succeeds once frozen spec is restored
                (adapter_dir / "train_manifest.json").unlink()
                with self.assertRaisesRegex(SystemExit, "provenance mismatch"):
                    train_lora.main(
                        [
                            "--hf-home",
                            str(hf_home),
                            "--task-dir",
                            str(task_dir),
                            "--output",
                            str(adapter_dir),
                            "--resume",
                            str(adapter_dir / "checkpoints" / "epoch-1"),
                            "--epochs",
                            "2",
                            "--batch-size",
                            "2",
                            "--grad-accum",
                            "2",
                            "--max-seq",
                            "64",
                            "--threads",
                            "1",
                            "--model-registry",
                            str(registry_path),
                        ]
                    )

                tampered["thresholds"]["macro_f1_min"] = 0.8
                (task_dir / "spec.json").write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                rc_res = train_lora.main(
                    [
                        "--hf-home",
                        str(hf_home),
                        "--task-dir",
                        str(task_dir),
                        "--output",
                        str(adapter_dir),
                        "--resume",
                        str(adapter_dir / "checkpoints" / "epoch-1"),
                        "--epochs",
                        "2",
                        "--batch-size",
                        "2",
                        "--grad-accum",
                        "2",
                        "--max-seq",
                        "64",
                        "--threads",
                        "1",
                        "--model-registry",
                        str(registry_path),
                    ]
                )
                self.assertEqual(rc_res, 0)
                resumed_manifest = json.loads((adapter_dir / "train_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(resumed_manifest["epochs"], 2)
                self.assertEqual(len(resumed_manifest["validation_history"]), 2)

                # 10. --method dpo requires preference data and executes real DPO optimization
                with self.assertRaisesRegex(SystemExit, "--preference-data"):
                    train_lora.main(
                        [
                            "--hf-home",
                            str(hf_home),
                            "--task-dir",
                            str(task_dir),
                            "--output",
                            str(root / "dpo_fail"),
                            "--method",
                            "dpo",
                            "--model-registry",
                            str(registry_path),
                        ]
                    )

                pref_path = root / "prefs.jsonl"
                pref_rows = [
                    {
                        "prompt": "Duplicate entry posted twice in settlement batch",
                        "chosen": "DUPLICATE_RECORD_CONFIRMED",
                        "rejected": "UNDETERMINED",
                    },
                    {
                        "prompt": "Two cent fee rounding tolerance difference",
                        "chosen": "AMOUNT_VARIANCE_ACCEPTED",
                        "rejected": "DUPLICATE_RECORD_CONFIRMED",
                    },
                ]
                pref_path.write_text("\n".join(json.dumps(r) for r in pref_rows) + "\n", encoding="utf-8")
                dpo_dir = root / "dpo_out"
                rc_dpo = train_lora.main(
                    [
                        "--hf-home",
                        str(hf_home),
                        "--task-dir",
                        str(task_dir),
                        "--output",
                        str(dpo_dir),
                        "--method",
                        "dpo",
                        "--preference-data",
                        str(pref_path),
                        "--epochs",
                        "1",
                        "--batch-size",
                        "1",
                        "--grad-accum",
                        "1",
                        "--max-seq",
                        "64",
                        "--threads",
                        "1",
                        "--model-registry",
                        str(registry_path),
                    ]
                )
                self.assertEqual(rc_dpo, 0)
                dpo_manifest = json.loads((dpo_dir / "train_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(dpo_manifest["method"], "dpo")
                self.assertEqual(dpo_manifest["train_records"], 2)
                self.assertIsNotNone(dpo_manifest["preference_data_sha256"])
                self.assertEqual(dpo_manifest["optimizer_steps"], 2)


class BaselineFinalizeShowPubkeyCliTest(unittest.TestCase):
    def test_baseline_predict_rules_and_cli(self):
        self.assertEqual(baseline_predict.predict("Posted twice by batch reimport"), "DUPLICATE_RECORD_CONFIRMED")
        self.assertEqual(baseline_predict.predict("Arrived after cutoff timing window"), "TIMING_DIFFERENCE_ACCEPTED")
        self.assertEqual(baseline_predict.predict("Wire fee rounding tax discrepancy"), "AMOUNT_VARIANCE_ACCEPTED")
        self.assertEqual(baseline_predict.predict("GBP vs EUR exchange rate drift"), "CURRENCY_VARIANCE_ACCEPTED")
        self.assertEqual(baseline_predict.predict("Counterpart not found in ledger"), "MISSING_COUNTERPART_CONFIRMED")
        self.assertEqual(baseline_predict.predict("Benign expected gap false positive"), "FALSE_POSITIVE_CONFIRMED")
        self.assertEqual(baseline_predict.predict("Reopen and escalate for review"), "REOPENED_FOR_INVESTIGATION")
        self.assertEqual(baseline_predict.predict("Unclear memo line 42"), "UNDETERMINED")
        # First-match-wins: DUPLICATE precedes TIMING
        self.assertEqual(baseline_predict.predict("Duplicated due to cutoff timing"), "DUPLICATE_RECORD_CONFIRMED")
        with self.assertRaises(TypeError):
            baseline_predict.predict(123)  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = _make_case_task_dir(root, "org/mock-tiny-coder", "rev-mock-1")
            out_path = root / "baseline.jsonl"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = baseline_predict.main(["--task-dir", str(task_dir), "--out", str(out_path), "--json"])
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "completed")
            self.assertTrue(out_path.is_file())

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                bad_rc = baseline_predict.main(["--task-dir", str(root / "missing"), "--out", str(out_path), "--json"])
            self.assertEqual(bad_rc, 1)
            self.assertEqual(json.loads(stderr.getvalue())["status"], "error")

    def test_finalize_spec_refuses_mismatch_and_supports_json_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = _make_case_task_dir(root, "org/mock-tiny-coder", "rev-mock-1")
            adapter_dir = root / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "train_manifest.json").write_text(
                json.dumps(
                    {
                        "adapter_sha256": "a" * 64,
                        "base_model_id": "org/mock-tiny-coder",
                        "revision": "rev-mock-1",
                    }
                ),
                encoding="utf-8",
            )
            # --verify fails on placeholder before finalizing
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = finalize_spec_mod.main(
                    ["--task-dir", str(task_dir), "--adapter-dir", str(adapter_dir), "--verify", "--json"]
                )
            self.assertEqual(rc, 1)
            self.assertIn("not finalized", json.loads(stderr.getvalue())["error"])

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc_fin = finalize_spec_mod.main(
                    ["--task-dir", str(task_dir), "--adapter-dir", str(adapter_dir), "--json"]
                )
            self.assertEqual(rc_fin, 0)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "finalized")

            # Refuses overwriting with a different non-placeholder adapter hash
            other_dir = root / "other_adapter"
            other_dir.mkdir()
            (other_dir / "train_manifest.json").write_text(
                json.dumps(
                    {
                        "adapter_sha256": "b" * 64,
                        "base_model_id": "org/mock-tiny-coder",
                        "revision": "rev-mock-1",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "refusing to rewrite"):
                finalize_spec_mod.finalize_spec(task_dir, other_dir)

    def test_show_pubkey_cli_plain_json_and_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_path = Path(tmp) / "ed25519.pem"
            key_path.write_text("placeholder-pem", encoding="utf-8")
            fake_hex = "ab" * 32

            with patch.object(show_pubkey.signing, "public_key_hex", return_value=fake_hex):
                stdout_plain = io.StringIO()
                with contextlib.redirect_stdout(stdout_plain):
                    rc_plain = show_pubkey.main([str(key_path)])
                self.assertEqual(rc_plain, 0)
                self.assertEqual(stdout_plain.getvalue().strip(), fake_hex)

                stdout_json = io.StringIO()
                with contextlib.redirect_stdout(stdout_json):
                    rc_json = show_pubkey.main([str(key_path), "--json"])
                self.assertEqual(rc_json, 0)
                parsed = json.loads(stdout_json.getvalue())
                self.assertEqual(parsed["public_key_hex"], fake_hex)
                self.assertEqual(parsed["fingerprint_sha256"], signing.fingerprint(fake_hex))

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc_err = show_pubkey.main([str(Path(tmp) / "missing.pem"), "--json"])
            self.assertEqual(rc_err, 1)
            self.assertEqual(json.loads(stderr.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
