"""Bounded offline CPU LoRA smoke; synthetic data, no production promotion.

Requires an existing compatible PyTorch/Transformers/PEFT environment and the
pinned model in a local Hugging Face cache. Never downloads models.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

try:
    from hardonian_foundry.training_utils import (
        DEFAULT_BLOBS,
        DEFAULT_MODEL_ID,
        DEFAULT_MODEL_SHA256,
        DEFAULT_REVISION,
        MIN_AVAILABLE_RAM_BYTES,
        SUPPORTED_DEVICES,
        TrainingLockError,
        acquire_training_lock,
        available_ram_bytes,
        check_disk_space,
        chmod_private,
        collate_causal_lm_batch,
        iter_micro_batches,
        load_verified_snapshot,
        mkdir_private,
        resolve_model_spec,
        select_device,
        set_offline_env,
        sha_bytes,
        sha_file,
        snapshot_path,
        verify_private_dir,
        verify_snapshot,
    )
except ImportError:
    sys.modules.pop("hardonian_foundry", None)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hardonian_foundry.training_utils import (
        DEFAULT_BLOBS,
        DEFAULT_MODEL_ID,
        DEFAULT_MODEL_SHA256,
        DEFAULT_REVISION,
        MIN_AVAILABLE_RAM_BYTES,
        SUPPORTED_DEVICES,
        TrainingLockError,
        acquire_training_lock,
        available_ram_bytes,
        check_disk_space,
        chmod_private,
        collate_causal_lm_batch,
        iter_micro_batches,
        load_verified_snapshot,
        mkdir_private,
        resolve_model_spec,
        select_device,
        set_offline_env,
        sha_bytes,
        sha_file,
        snapshot_path,
        verify_private_dir,
        verify_snapshot,
    )

MODEL = DEFAULT_MODEL_ID
REVISION = DEFAULT_REVISION
SEED = 1729
MODEL_SHA256 = DEFAULT_MODEL_SHA256
BLOBS = dict(DEFAULT_BLOBS)


def examples() -> dict[str, list[dict[str, str]]]:
    """Generate deterministic synthetic train and holdout classification items."""

    def item(i: int) -> dict[str, str]:
        expected = 1000 + i * 37
        actual = expected + (5 if i % 2 else 0)
        answer = "amount_mismatch" if actual != expected else "matched"
        prompt = (
            f"Synthetic example {i}. Expected minor units: {expected}. "
            f"Actual minor units: {actual}. Classify as matched or amount_mismatch; "
            "return the label only."
        )
        return {"id": f"synthetic-{i:03d}", "prompt": prompt, "answer": answer}

    train = [item(i) for i in range(12)]
    holdout = [item(i) for i in range(100, 104)]
    if {x["prompt"] for x in train} & {x["prompt"] for x in holdout}:
        raise RuntimeError("train/holdout prompt overlap detected")
    return {"train": train, "holdout": holdout}


def main() -> None:
    if os.name != "nt":
        os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-home",
        type=Path,
        required=True,
        help="existing HF_HOME containing the pinned cached model",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="existing private directory on local storage",
    )
    parser.add_argument("--steps", type=int, default=4, choices=range(1, 21), metavar="1..20")
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 5), metavar="1..4")
    parser.add_argument("--batch-size", type=int, default=1, help="micro-batch size (default: 1)")
    parser.add_argument("--device", default="cpu", choices=SUPPORTED_DEVICES, help="execution device (default: cpu)")
    parser.add_argument("--model-registry", type=Path, default=None, help="optional model registry JSON")
    parser.add_argument("--model-id", default=None, help="model ID from registry (default: pinned Qwen 0.5B)")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    try:
        cache = args.hf_home.expanduser().resolve(strict=True)
        root = verify_private_dir(args.output_root, label="output root")
        model_spec = resolve_model_spec(args.model_id, registry_path=args.model_registry)
        snap = snapshot_path(cache, model_spec.model_id, model_spec.revision)
        verify_snapshot(
            snap,
            model_sha256=model_spec.model_sha256,
            blobs=model_spec.blobs or None,
            weight_files=model_spec.weight_files or None,
        )
        check_disk_space(root)
        if available_ram_bytes() < MIN_AVAILABLE_RAM_BYTES:
            raise ValueError("insufficient available physical memory (minimum 8 GiB)")
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc

    try:
        with acquire_training_lock(root / ".training.lock"):
            run(args, cache, root, snap, model_spec=model_spec)
    except TrainingLockError as exc:
        raise SystemExit(str(exc)) from exc


def run(
    args: argparse.Namespace,
    cache: Path,
    root: Path,
    snap: Path,
    *,
    model_spec: Any | None = None,
) -> None:
    spec = model_spec or resolve_model_spec()
    requested_device = getattr(args, "device", "cpu")
    set_offline_env(cache, threads=args.threads, allow_gpu=(requested_device != "cpu"))
    import peft
    import torch
    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_spec = select_device(requested_device, torch)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    random.seed(SEED)
    started = time.monotonic()
    run_id = time.strftime("synthetic-lora-%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
    output = mkdir_private(root / run_id, exist_ok=False)
    data = examples()
    data_bytes = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    (output / "synthetic-dataset.json").write_bytes(data_bytes)

    dtype = getattr(torch, device_spec.torch_dtype_name, torch.float32)
    tokenizer, model = load_verified_snapshot(
        snap,
        AutoTokenizer,
        AutoModelForCausalLM,
        torch,
        device=device_spec.device,
        dtype=dtype,
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    def encode(row: dict[str, str]) -> tuple[list[int], list[int]]:
        messages = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["answer"]}]
        prompt = tokenizer.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if full[: len(prompt)] != prompt or not len(prompt) < len(full) <= spec.max_seq:
            raise RuntimeError("chat template mismatch or token limit exceeded")
        return list(full), [-100] * len(prompt) + list(full[len(prompt) :])

    train_encoded = [encode(row) for row in data["train"]]
    holdout_encoded = [encode(row) for row in data["holdout"]]

    def losses(m: Any) -> list[float]:
        m.eval()
        values: list[float] = []
        with torch.inference_mode():
            for ids_seq, lbl_seq in holdout_encoded:
                batch = collate_causal_lm_batch(
                    [(ids_seq, lbl_seq)],
                    pad_token_id=int(pad_id),
                    torch_module=torch,
                    device=device_spec.device,
                )
                values.append(float(m(**batch).loss))
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError("non-finite evaluation loss")
        return values

    base_loss = losses(model)
    target_modules = (
        ["q_proj", "v_proj"] if spec.model_id == DEFAULT_MODEL_ID else list(spec.lora_target_modules)
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0002)
    training_losses: list[float] = []
    batch_size = getattr(args, "batch_size", 1)
    batches = list(iter_micro_batches(train_encoded, batch_size))
    model.train()
    for step in range(args.steps):
        batch = collate_causal_lm_batch(
            batches[step % len(batches)],
            pad_token_id=int(pad_id),
            torch_module=torch,
            device=device_spec.device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("non-finite gradient")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
            raise RuntimeError("non-finite adapter weights")
        training_losses.append(float(loss.detach()))
        print(f"step={step + 1} loss={training_losses[-1]:.6f}", flush=True)

    trained_loss = losses(model)
    adapter = output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    base_model = model.unload()
    reloaded = PeftModel.from_pretrained(base_model, adapter, is_trainable=False, local_files_only=True)
    reloaded_loss = losses(reloaded)
    delta = max(abs(a - b) for a, b in zip(trained_loss, reloaded_loss, strict=True))
    if delta > 1e-6:
        raise RuntimeError(f"offline adapter reload mismatch: {delta}")

    weights_file = snap / "model.safetensors"
    base_sha = sha_file(weights_file) if weights_file.is_file() else spec.model_sha256
    result = {
        "status": "completed",
        "data_origin": "synthetic",
        "model": spec.model_id,
        "revision": spec.revision,
        "model_license": f"{spec.license_id} (review upstream license before distribution)",
        "base_model_sha256": base_sha,
        "dataset_sha256": sha_bytes(data_bytes),
        "adapter_sha256": sha_file(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": sha_file(adapter / "adapter_config.json"),
        "script_sha256": sha_file(Path(__file__)),
        "device": device_spec.device,
        "seed": SEED,
        "steps": args.steps,
        "batch_size": batch_size,
        "threads": args.threads,
        "trainable_parameters": trainable,
        "train_count": len(train_encoded),
        "holdout_count": len(holdout_encoded),
        "train_losses": training_losses,
        "base_holdout_loss": base_loss,
        "adapter_holdout_loss": trained_loss,
        "reload_holdout_loss": reloaded_loss,
        "max_reload_delta": delta,
        "duration_seconds": round(time.monotonic() - started, 2),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "limitations": [
            "synthetic smoke, teacher-forced loss only",
            "not a production or quality benchmark",
            "not approved for promotion",
            "no checkpoint resume",
        ],
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    chmod_private(result_path)
    print(json.dumps({"run_dir": str(output), **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
