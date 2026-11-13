#!/usr/bin/env python3
"""LoRA fine-tune a pinned local snapshot on the frozen train split of an exception-code task.

Offline, synthetic. The base snapshot must already exist under --hf-home in the hub
cache layout at the pinned revision. Verifies snapshot SHA-256, Git blobs, and
absence of `auto_map` remote code, enforces non-truncating `max_seq`, uses padded
micro-batching and rescaled gradient-accumulation boundaries, and writes an adapter
directory plus `train_manifest.json`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

try:
    from hardonian_foundry.baseline import digest
    from hardonian_foundry.task_eval import split_task_dataset, validate_spec
    from hardonian_foundry.training_utils import (
        DEFAULT_MODEL_ID,
        DEFAULT_REVISION,
        SUPPORTED_DEVICES,
        SUPPORTED_METHODS,
        DPOAdapterHook,
        QLoRAAdapterHook,
        RLVRAdapterHook,
        TrainingLockError,
        acquire_training_lock,
        batched_greedy_generate,
        chmod_private,
        collate_causal_lm_batch,
        get_training_hook,
        grad_accum_window_size,
        iter_micro_batches,
        load_verified_snapshot,
        mkdir_private,
        resolve_model_spec,
        select_device,
        set_offline_env,
        sha_file,
        should_step_grad_accum,
        snapshot_path,
        verify_private_dir,
        verify_snapshot,
    )
except ImportError:
    sys.modules.pop("hardonian_foundry", None)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from hardonian_foundry.baseline import digest
    from hardonian_foundry.task_eval import split_task_dataset, validate_spec
    from hardonian_foundry.training_utils import (
        DEFAULT_MODEL_ID,
        DEFAULT_REVISION,
        SUPPORTED_DEVICES,
        SUPPORTED_METHODS,
        DPOAdapterHook,
        QLoRAAdapterHook,
        RLVRAdapterHook,
        TrainingLockError,
        acquire_training_lock,
        batched_greedy_generate,
        chmod_private,
        collate_causal_lm_batch,
        get_training_hook,
        grad_accum_window_size,
        iter_micro_batches,
        load_verified_snapshot,
        mkdir_private,
        resolve_model_spec,
        select_device,
        set_offline_env,
        sha_file,
        should_step_grad_accum,
        snapshot_path,
        verify_private_dir,
        verify_snapshot,
    )

MODEL_ID = DEFAULT_MODEL_ID
REVISION = DEFAULT_REVISION
SEED = 20261110
PROMPT_SYS = (
    "You classify reconciliation exception notes into exactly one resolution code. "
    "Reply with only the code, nothing else."
)


def compute_spec_core_sha256(spec: dict[str, Any]) -> str:
    """Hash canonical spec with `model.adapter_sha256` normalized to 64 zeros."""
    normalized = json.loads(json.dumps(spec))
    if isinstance(normalized.get("model"), dict):
        normalized["model"]["adapter_sha256"] = "0" * 64
    return digest(normalized)


def _normalize_token_list(raw_ids: Any) -> list[int]:
    if isinstance(raw_ids, dict) and "input_ids" in raw_ids:
        raw_ids = raw_ids["input_ids"]
    elif hasattr(raw_ids, "get"):
        maybe = raw_ids.get("input_ids")
        if maybe is not None:
            raw_ids = maybe
    if hasattr(raw_ids, "tolist"):
        raw_ids = raw_ids.tolist()
    if raw_ids and isinstance(raw_ids[0], list):
        raw_ids = raw_ids[0]
    return [int(t) for t in raw_ids]


def encode_classification_example(
    tokenizer: Any,
    row: dict[str, Any],
    label_menu: str,
    max_seq: int,
    *,
    prompt_sys: str = PROMPT_SYS,
) -> dict[str, list[int]]:
    """Tokenize a classification example; fail closed if `max_seq` would truncate it."""
    if max_seq <= 0:
        raise ValueError("max_seq must be positive")
    messages = [
        {"role": "system", "content": prompt_sys},
        {"role": "user", "content": f"Codes: {label_menu}\nNote: {row['prompt']}\nCode:"},
    ]
    prompt_ids = _normalize_token_list(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    )
    full_ids = _normalize_token_list(
        tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": row["gold_label"]}],
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"example {row.get('id', '?')}: chat template prefix mismatch")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError(f"example {row.get('id', '?')}: assistant target produced zero tokens")
    if len(full_ids) > max_seq:
        raise ValueError(
            f"example {row.get('id', '?')} length {len(full_ids)} exceeds max_seq={max_seq}; "
            "refusing silent truncation (increase --max-seq)"
        )
    labels_ids = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {"input_ids": full_ids, "labels": labels_ids}


def encode_preference_pair(
    tokenizer: Any,
    row: dict[str, Any],
    label_menu: str,
    max_seq: int,
    *,
    prompt_sys: str = PROMPT_SYS,
) -> dict[str, list[int]]:
    """Tokenize a DPO `(prompt, chosen, rejected)` pair; fail closed if `max_seq` would truncate."""
    if max_seq <= 0:
        raise ValueError("max_seq must be positive")
    messages = [
        {"role": "system", "content": prompt_sys},
        {"role": "user", "content": f"Codes: {label_menu}\nNote: {row['prompt']}\nCode:"},
    ]
    prompt_ids = _normalize_token_list(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    )
    encoded: dict[str, list[int]] = {}
    for key in ("chosen", "rejected"):
        full_ids = _normalize_token_list(
            tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": row[key]}],
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"preference pair {key} chat template prefix mismatch")
        if len(full_ids) <= len(prompt_ids):
            raise ValueError(f"preference pair {key} produced zero assistant tokens")
        if len(full_ids) > max_seq:
            raise ValueError(
                f"preference pair {key} length {len(full_ids)} exceeds max_seq={max_seq}; "
                "refusing silent truncation (increase --max-seq)"
            )
        encoded[f"{key}_input_ids"] = full_ids
        encoded[f"{key}_labels"] = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return encoded


def _sequence_completion_logps(model: Any, batch: dict[str, Any], torch_mod: Any) -> Any:
    """Sum token log-probabilities over assistant completion tokens (`labels != -100`)."""
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[:, :-1, :]
    targets = batch["input_ids"][:, 1:]
    mask = batch["labels"][:, 1:] != -100
    log_probs = torch_mod.nn.functional.log_softmax(logits.float(), dim=-1)
    token_logps = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask.to(token_logps.dtype)).sum(dim=-1)


def build_lr_scheduler(
    optimizer: Any,
    total_steps: int,
    *,
    scheduler_type: str = "cosine",
    warmup_ratio: float = 0.1,
    torch_module: Any | None = None,
) -> Any:
    """Create cosine, linear, or constant LR schedule with linear warmup."""
    if torch_module is None:
        import torch

        torch_mod: Any = torch
    else:
        torch_mod = torch_module
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not (0.0 <= warmup_ratio <= 0.5):
        raise ValueError("warmup_ratio must be in [0.0, 0.5]")
    if scheduler_type not in ("cosine", "linear", "constant"):
        raise ValueError(f"unsupported scheduler {scheduler_type!r}")

    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        if scheduler_type == "constant":
            return 1.0
        denom = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(current_step - warmup_steps) / float(denom)))
        if scheduler_type == "linear":
            return max(0.0, 1.0 - progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch_mod.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise TypeError(f"line {lineno} of {path} is not a JSON object")
        rows.append(obj)
    if not rows:
        raise ValueError(f"no records found in {path}")
    return rows


def main(argv: list[str] | None = None) -> int:
    if os.name != "nt":
        os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--task-dir", required=True, help="directory containing spec.json and data.jsonl")
    parser.add_argument("--output", required=True, help="adapter output directory (must not exist unless resuming inside it)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq", type=int, default=160)
    parser.add_argument("--scheduler", default="cosine", choices=("cosine", "linear", "constant"))
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--resume", type=Path, default=None, help="optional checkpoint dir to resume from")
    parser.add_argument("--save-checkpoints", action="store_true", help="persist per-epoch resume checkpoints")
    parser.add_argument("--eval-val", dest="eval_val", action="store_true", default=True)
    parser.add_argument("--no-eval-val", dest="eval_val", action="store_false")
    parser.add_argument("--device", default="cpu", choices=SUPPORTED_DEVICES)
    parser.add_argument("--method", default="lora", choices=SUPPORTED_METHODS)
    parser.add_argument("--preference-data", type=Path, default=None, help="JSONL preference pairs for --method dpo")
    parser.add_argument("--model-registry", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.epochs <= 0 or args.batch_size <= 0 or args.grad_accum <= 0 or args.max_seq <= 0 or args.threads <= 0:
        raise SystemExit("epochs, batch-size, grad-accum, max-seq, and threads must be positive")
    if args.lr <= 0 or args.max_grad_norm < 0 or not (0.0 <= args.warmup_ratio <= 0.5):
        raise SystemExit("invalid lr, max-grad-norm, or warmup-ratio")

    out_dir = Path(args.output).expanduser().resolve()
    resume_dir = args.resume.expanduser().resolve() if args.resume is not None else None
    if out_dir.exists() and (
        resume_dir is None or not resume_dir.is_relative_to(out_dir) or (out_dir / "train_manifest.json").exists()
    ):
        raise SystemExit(f"output directory already exists: {out_dir}")

    task_dir = Path(args.task_dir).expanduser().resolve()
    spec_path = task_dir / "spec.json"
    data_path = task_dir / "data.jsonl"
    if not spec_path.is_file() or not data_path.is_file():
        raise SystemExit(f"task directory missing spec.json or data.jsonl: {task_dir}")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec_errors = validate_spec(spec)
    if spec_errors:
        raise SystemExit(f"invalid task spec: {'; '.join(spec_errors)}")

    hf_home = Path(args.hf_home).expanduser().resolve()
    try:
        model_spec = resolve_model_spec(
            spec["model"]["base_model_id"],
            revision=spec["model"].get("revision"),
            registry_path=args.model_registry,
        )
        snap = snapshot_path(hf_home, model_spec.model_id, model_spec.revision)
        snap_info = verify_snapshot(
            snap,
            model_sha256=model_spec.model_sha256,
            blobs=model_spec.blobs or None,
            weight_files=model_spec.weight_files or None,
        )
    except (ValueError, OSError) as exc:
        raise SystemExit(f"model preflight failed: {exc}") from exc

    parent_dir = out_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    try:
        verify_private_dir(parent_dir, label="output parent directory")
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc

    try:
        with acquire_training_lock(parent_dir / ".training.lock"):
            return _run_training(args, out_dir, resume_dir, task_dir, spec, hf_home, snap, snap_info, model_spec)
    except TrainingLockError as exc:
        raise SystemExit(str(exc)) from exc


def _run_training(
    args: argparse.Namespace,
    out_dir: Path,
    resume_dir: Path | None,
    task_dir: Path,
    spec: dict[str, Any],
    hf_home: Path,
    snap: Path,
    snap_info: dict[str, Any],
    model_spec: Any,
) -> int:
    set_offline_env(hf_home, threads=args.threads, allow_gpu=(args.device != "cpu"))
    import peft
    import torch
    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        device_spec = select_device(args.device, torch)
        hook = get_training_hook(args.method)
        if isinstance(hook, QLoRAAdapterHook):
            hook.ensure_supported(device_spec)
            quant_cfg = hook.build_quantization_config(torch)
        else:
            quant_cfg = None

        pref_rows: list[dict[str, Any]] = []
        if isinstance(hook, DPOAdapterHook):
            if args.preference_data is None:
                raise ValueError("training method 'dpo' requires --preference-data JSONL with prompt/chosen/rejected")
            pref_rows = _load_jsonl(args.preference_data)
            hook.validate_preferences(pref_rows)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)

    rows = _load_jsonl(task_dir / "data.jsonl")
    splits = split_task_dataset(rows, spec)
    train_rows = splits["train"]
    val_rows = splits["validation"]
    labels = list(spec["labels"])
    label_menu = ", ".join(labels)

    dtype = getattr(torch, device_spec.torch_dtype_name, torch.float32)
    tokenizer, model = load_verified_snapshot(
        snap,
        AutoTokenizer,
        AutoModelForCausalLM,
        torch,
        device=device_spec.device,
        dtype=dtype,
        quantization_config=quant_cfg,
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    try:
        encoded_val = (
            [encode_classification_example(tokenizer, row, label_menu, args.max_seq) for row in val_rows]
            if args.eval_val and val_rows
            else []
        )
        encoded_train: list[dict[str, Any]] = []
        encoded_prefs: list[dict[str, Any]] = []
        if args.method == "dpo":
            encoded_prefs = [
                encode_preference_pair(tokenizer, row, label_menu, args.max_seq) for row in pref_rows
            ]
        else:
            encoded_train = [
                encode_classification_example(tokenizer, row, label_menu, args.max_seq) for row in train_rows
            ]
    except ValueError as exc:
        raise SystemExit(f"dataset tokenization failed closed: {exc}") from exc

    if args.method == "dpo":
        model.eval()
        with torch.inference_mode():
            for pref_item in encoded_prefs:
                c_batch = collate_causal_lm_batch(
                    [{"input_ids": pref_item["chosen_input_ids"], "labels": pref_item["chosen_labels"]}],
                    pad_token_id=int(pad_id),
                    torch_module=torch,
                    device=device_spec.device,
                )
                r_batch = collate_causal_lm_batch(
                    [{"input_ids": pref_item["rejected_input_ids"], "labels": pref_item["rejected_labels"]}],
                    pad_token_id=int(pad_id),
                    torch_module=torch,
                    device=device_spec.device,
                )
                pref_item["ref_chosen_logp"] = float(_sequence_completion_logps(model, c_batch, torch)[0].item())
                pref_item["ref_rejected_logp"] = float(_sequence_completion_logps(model, r_batch, torch)[0].item())

    train_pool: list[dict[str, Any]] = encoded_prefs if args.method == "dpo" else encoded_train
    dataset_sha = sha_file(task_dir / "data.jsonl")
    pref_sha = sha_file(args.preference_data) if args.method == "dpo" and args.preference_data is not None else None
    spec_sha = sha_file(task_dir / "spec.json")
    spec_core_sha = compute_spec_core_sha256(spec)
    base_weights_sha = snap_info["weights_sha256"]

    start_epoch = 0
    step = 0
    losses: list[float] = []
    val_history: list[dict[str, Any]] = []
    best_val_loss: float | None = None
    best_epoch = 0
    ckpt_opt_state: dict[str, Any] | None = None
    ckpt_sched_state: dict[str, Any] | None = None

    if resume_dir is not None:
        state_file = resume_dir / "checkpoint_state.json"
        ckpt_adapter = resume_dir / "adapter"
        if not state_file.is_file() or not ckpt_adapter.is_dir():
            raise SystemExit(f"invalid resume checkpoint directory: {resume_dir}")
        ckpt_state = json.loads(state_file.read_text(encoding="utf-8"))
        if (
            ckpt_state.get("dataset_sha256") != dataset_sha
            or ckpt_state.get("spec_core_sha256") != spec_core_sha
            or ckpt_state.get("base_weights_sha256") != base_weights_sha
            or ckpt_state.get("method") != args.method
            or ckpt_state.get("preference_data_sha256") != pref_sha
        ):
            raise SystemExit("resume checkpoint provenance mismatch; refusing to resume")
        model = PeftModel.from_pretrained(model, str(ckpt_adapter), is_trainable=True, local_files_only=True)
        start_epoch = int(ckpt_state["epoch"])
        step = int(ckpt_state["optimizer_steps"])
        best_val_loss = ckpt_state.get("best_val_loss")
        best_epoch = int(ckpt_state.get("best_epoch", 0))
        losses = list(ckpt_state.get("losses", []))
        val_history = list(ckpt_state.get("val_history", []))
        opt_path = resume_dir / "optimizer.pt"
        if opt_path.is_file():
            ckpt_opt_state = torch.load(opt_path, map_location=device_spec.device, weights_only=True)
        sched_path = resume_dir / "scheduler.pt"
        if sched_path.is_file():
            ckpt_sched_state = torch.load(sched_path, map_location=device_spec.device, weights_only=True)
    else:
        peft_config = LoraConfig(
            r=model_spec.default_lora_r,
            lora_alpha=model_spec.default_lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(model_spec.lora_target_modules),
        )
        model = get_peft_model(model, peft_config)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"trainable params: {trainable_count}", flush=True)

    micro_batches_per_epoch = math.ceil(len(train_pool) / args.batch_size)
    steps_per_epoch = math.ceil(micro_batches_per_epoch / args.grad_accum)
    total_opt_steps = max(1, args.epochs * steps_per_epoch)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    scheduler = build_lr_scheduler(
        optimizer,
        total_opt_steps,
        scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        torch_module=torch,
    )
    if ckpt_opt_state is not None:
        optimizer.load_state_dict(ckpt_opt_state)
    if ckpt_sched_state is not None:
        scheduler.load_state_dict(ckpt_sched_state)

    mkdir_private(out_dir, exist_ok=True)
    t0 = time.time()
    gen = torch.Generator().manual_seed(SEED)

    def eval_validation_loss() -> float | None:
        if not encoded_val:
            return None
        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            for mb in iter_micro_batches(encoded_val, args.batch_size):
                batch = collate_causal_lm_batch(
                    mb,
                    pad_token_id=int(pad_id),
                    torch_module=torch,
                    device=device_spec.device,
                )
                vloss = float(model(**batch).loss)
                if not math.isfinite(vloss):
                    raise RuntimeError("non-finite validation loss")
                val_losses.append(vloss)
        return sum(val_losses) / len(val_losses)

    for epoch in range(args.epochs):
        order = torch.randperm(len(train_pool), generator=gen).tolist()
        if epoch < start_epoch:
            continue
        ordered_items = [train_pool[i] for i in order]
        epoch_batches = list(iter_micro_batches(ordered_items, args.batch_size))
        index_batches = list(iter_micro_batches(order, args.batch_size))
        total_batches = len(epoch_batches)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        for b_idx, mb in enumerate(epoch_batches):
            if isinstance(hook, DPOAdapterHook):
                chosen_batch = collate_causal_lm_batch(
                    [{"input_ids": x["chosen_input_ids"], "labels": x["chosen_labels"]} for x in mb],
                    pad_token_id=int(pad_id),
                    torch_module=torch,
                    device=device_spec.device,
                )
                rejected_batch = collate_causal_lm_batch(
                    [{"input_ids": x["rejected_input_ids"], "labels": x["rejected_labels"]} for x in mb],
                    pad_token_id=int(pad_id),
                    torch_module=torch,
                    device=device_spec.device,
                )
                pi_chosen = _sequence_completion_logps(model, chosen_batch, torch)
                pi_rejected = _sequence_completion_logps(model, rejected_batch, torch)
                ref_chosen = torch.tensor(
                    [float(x["ref_chosen_logp"]) for x in mb],
                    dtype=pi_chosen.dtype,
                    device=pi_chosen.device,
                )
                ref_rejected = torch.tensor(
                    [float(x["ref_rejected_logp"]) for x in mb],
                    dtype=pi_rejected.dtype,
                    device=pi_rejected.device,
                )
                raw_loss, _, _ = hook.loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected)
            elif isinstance(hook, RLVRAdapterHook):
                # Execute deterministic verifier hook on generated batch outputs
                mb_rows = [train_rows[idx] for idx in index_batches[b_idx]]
                val_prompts = [
                    _normalize_token_list(
                        tokenizer.apply_chat_template(
                            [
                                {"role": "system", "content": PROMPT_SYS},
                                {"role": "user", "content": f"Codes: {label_menu}\nNote: {r['prompt']}\nCode:"},
                            ],
                            tokenize=True,
                            add_generation_prompt=True,
                        )
                    )
                    for r in mb_rows
                ]
                gens = batched_greedy_generate(
                    model,
                    tokenizer,
                    val_prompts,
                    max_new_tokens=12,
                    batch_size=len(val_prompts),
                    return_confidence=False,
                    torch_module=torch,
                    device=device_spec.device,
                )
                rewards = hook.score_completions(
                    [r["prompt"] for r in mb_rows],
                    [g.text for g in gens],
                    [r["gold_label"] for r in mb_rows],
                )
                batch = collate_causal_lm_batch(
                    mb, pad_token_id=int(pad_id), torch_module=torch, device=device_spec.device
                )
                raw_loss = hook.compute_loss(model, batch)
                reward_tensor = torch.tensor(rewards, dtype=raw_loss.dtype, device=raw_loss.device)
                raw_loss = raw_loss * (1.5 - 0.5 * reward_tensor.mean())
            else:
                batch = collate_causal_lm_batch(
                    mb,
                    pad_token_id=int(pad_id),
                    torch_module=torch,
                    device=device_spec.device,
                )
                raw_loss = hook.compute_loss(model, batch)

            if not torch.isfinite(raw_loss):
                raise RuntimeError("non-finite training loss")

            win_size = grad_accum_window_size(b_idx, total_batches, args.grad_accum)
            (raw_loss / float(win_size)).backward()
            losses.append(float(raw_loss.detach()))

            if should_step_grad_accum(b_idx, total_batches, args.grad_accum):
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainable_params):
                    raise RuntimeError("non-finite gradient")
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    recent = sum(losses[-100:]) / min(100, len(losses[-100:]))
                    print(
                        f"epoch {epoch} step {step} loss {recent:.4f} elapsed {time.time() - t0:.0f}s",
                        flush=True,
                    )

        val_loss = eval_validation_loss()
        improved = val_loss is None or best_val_loss is None or val_loss < best_val_loss
        if val_loss is not None:
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                model.save_pretrained(str(out_dir), safe_serialization=True)
            val_history.append({"epoch": epoch + 1, "val_loss": round(val_loss, 6), "improved": improved})
            print(f"epoch {epoch + 1} val_loss {val_loss:.4f} improved={improved}", flush=True)

        if args.save_checkpoints:
            mkdir_private(out_dir / "checkpoints", exist_ok=True)
            ckpt_dir = mkdir_private(out_dir / "checkpoints" / f"epoch-{epoch + 1}", exist_ok=True)
            model.save_pretrained(str(ckpt_dir / "adapter"), safe_serialization=True)
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
            torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
            ckpt_meta = {
                "epoch": epoch + 1,
                "optimizer_steps": step,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "dataset_sha256": dataset_sha,
                "spec_core_sha256": spec_core_sha,
                "base_weights_sha256": base_weights_sha,
                "method": args.method,
                "preference_data_sha256": pref_sha,
                "losses": losses,
                "val_history": val_history,
            }
            (ckpt_dir / "checkpoint_state.json").write_text(
                json.dumps(ckpt_meta, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    if not encoded_val or best_epoch == 0:
        model.save_pretrained(str(out_dir), safe_serialization=True)

    duration = time.time() - t0
    adapter_file = out_dir / "adapter_model.safetensors"
    manifest = {
        "status": "completed",
        "data_origin": "synthetic",
        "base_model_id": model_spec.model_id,
        "revision": model_spec.revision,
        "base_weights_sha256": base_weights_sha,
        "adapter_sha256": sha_file(adapter_file),
        "adapter_config_sha256": sha_file(out_dir / "adapter_config.json"),
        "dataset_sha256": dataset_sha,
        "preference_data_sha256": pref_sha,
        "spec_sha256": spec_sha,
        "spec_core_sha256": spec_core_sha,
        "train_records": len(train_pool),
        "validation_records": len(encoded_val),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "scheduler": args.scheduler,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "grad_accum": args.grad_accum,
        "max_seq": args.max_seq,
        "optimizer_steps": step,
        "final_train_loss_mean100": round(sum(losses[-100:]) / min(100, len(losses[-100:])), 6),
        "validation_history": val_history,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6) if best_val_loss is not None else None,
        "resumed_from": str(resume_dir) if resume_dir is not None else None,
        "device": device_spec.device,
        "method": args.method,
        "seed": SEED,
        "threads": args.threads,
        "duration_seconds": round(duration, 1),
        "lora": {
            "r": model_spec.default_lora_r,
            "alpha": model_spec.default_lora_alpha,
            "dropout": 0.0,
            "target_modules": list(model_spec.lora_target_modules),
        },
        "trainable_params": trainable_count,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "host": platform.node(),
        "limitations": (
            "Teacher-forced LoRA on synthetic in-style notes; train loss is not a "
            "generalization metric. Test scoring happens only via model_predict.py "
            "through the Foundry task gate."
        ),
    }
    manifest_path = out_dir / "train_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    chmod_private(manifest_path)
    print(
        json.dumps(
            {
                "status": "completed",
                "adapter_dir": str(out_dir),
                "adapter_sha256": manifest["adapter_sha256"],
                "optimizer_steps": step,
                "best_epoch": best_epoch,
                "duration_seconds": manifest["duration_seconds"],
                "final_train_loss_mean100": manifest["final_train_loss_mean100"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
