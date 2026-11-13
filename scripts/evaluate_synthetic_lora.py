"""Compare base and adapter generation on the existing synthetic holdout.

This is an in-domain smoke, not a pristine or commercially valid benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
        batched_greedy_generate,
        check_disk_space,
        chmod_private,
        load_verified_snapshot,
        mkdir_private,
        resolve_model_spec,
        select_device,
        set_offline_env,
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
        batched_greedy_generate,
        check_disk_space,
        chmod_private,
        load_verified_snapshot,
        mkdir_private,
        resolve_model_spec,
        select_device,
        set_offline_env,
        sha_file,
        snapshot_path,
        verify_private_dir,
        verify_snapshot,
    )

try:
    from train_synthetic_lora import examples
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_synthetic_lora import examples

MODEL = DEFAULT_MODEL_ID
REVISION = DEFAULT_REVISION
MODEL_SHA256 = DEFAULT_MODEL_SHA256
BLOBS = dict(DEFAULT_BLOBS)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute binary mismatch detection metrics plus invalid-output tracking."""
    tp = sum(r["expected"] == "amount_mismatch" and r["output"] == "amount_mismatch" for r in rows)
    fp = sum(r["expected"] == "matched" and r["output"] == "amount_mismatch" for r in rows)
    fn = sum(r["expected"] == "amount_mismatch" and r["output"] != "amount_mismatch" for r in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {
        "correct": sum(r["output"] == r["expected"] for r in rows),
        "total": len(rows),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_negative_rate": fn / (tp + fn) if tp + fn else None,
        "invalid_outputs": sum(r["output"] not in ("matched", "amount_mismatch") for r in rows),
        "max_token_limit_hits": sum(r["max_token_limit_hit"] for r in rows),
    }


def verify_adapter_hashes(adapter: Path, manifest: dict[str, Any]) -> None:
    """Reject missing or modified adapter weights or configuration."""
    for key, filename in (
        ("adapter_sha256", "adapter_model.safetensors"),
        ("adapter_config_sha256", "adapter_config.json"),
    ):
        path = adapter / filename
        if not path.is_file() or not manifest.get(key) or manifest[key] != sha_file(path):
            raise ValueError(f"missing or mismatched {key}; older manifests cannot be safely evaluated")


def main() -> None:
    if os.name != "nt":
        os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu", choices=SUPPORTED_DEVICES)
    parser.add_argument("--model-registry", type=Path, default=None)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    try:
        root = verify_private_dir(args.output_root, label="output root")
        run = verify_private_dir(args.run_dir, label="run directory")
        cache = args.hf_home.expanduser().resolve(strict=True)
        check_disk_space(root)
        if available_ram_bytes() < MIN_AVAILABLE_RAM_BYTES:
            raise ValueError("less than 8 GiB free physical memory")

        data_path = run / "synthetic-dataset.json"
        if not data_path.is_file() or not (run / "result.json").is_file():
            raise ValueError("missing synthetic-dataset.json or result.json in run directory")
        data = data_path.read_bytes()
        expected_data = json.dumps(examples(), sort_keys=True, separators=(",", ":")).encode()
        if data != expected_data:
            raise ValueError("only the built-in synthetic dataset is supported")
        manifest = json.loads((run / "result.json").read_text(encoding="utf-8"))
        spec = resolve_model_spec(manifest.get("model"), registry_path=args.model_registry)
        snap = snapshot_path(cache, spec.model_id, spec.revision)
        verify_snapshot(
            snap,
            model_sha256=spec.model_sha256,
            blobs=spec.blobs or None,
            weight_files=spec.weight_files or None,
        )
        adapter = run / "adapter"
        if (
            manifest.get("status") != "completed"
            or manifest.get("model") != spec.model_id
            or manifest.get("revision") != spec.revision
            or manifest.get("dataset_sha256") != hashlib.sha256(data).hexdigest()
        ):
            raise ValueError("training run manifest mismatch")
        verify_adapter_hashes(adapter, manifest)
        cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
        expected_targets = {"q_proj", "v_proj"} if spec.model_id == DEFAULT_MODEL_ID else set(spec.lora_target_modules)
        if (
            cfg.get("peft_type") != "LORA"
            or cfg.get("task_type") != "CAUSAL_LM"
            or cfg.get("r") != 4
            or cfg.get("lora_alpha") != 8
            or set(cfg.get("target_modules", [])) != expected_targets
            or cfg.get("auto_mapping") is not None
            or cfg.get("modules_to_save") is not None
        ):
            raise ValueError("unsupported adapter configuration")
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    try:
        with acquire_training_lock(root / ".training.lock"):
            run_eval(args, root, cache, snap, adapter, data, manifest, model_spec=spec)
    except TrainingLockError as exc:
        raise SystemExit(str(exc)) from exc


def run_eval(
    args: argparse.Namespace,
    root: Path,
    cache: Path,
    snap: Path,
    adapter: Path,
    data: bytes,
    manifest: dict[str, Any],
    *,
    model_spec: Any | None = None,
) -> None:
    spec = model_spec or resolve_model_spec(manifest.get("model"))
    requested_device = getattr(args, "device", "cpu")
    set_offline_env(cache, threads=args.threads, allow_gpu=(requested_device != "cpu"))
    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_spec = select_device(requested_device, torch)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1729)
    started = time.monotonic()
    output = mkdir_private(
        root / (time.strftime("synthetic-generation-eval-%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"),
        exist_ok=False,
    )
    dtype = getattr(torch, device_spec.torch_dtype_name, torch.float32)
    tokenizer, model = load_verified_snapshot(
        snap,
        AutoTokenizer,
        AutoModelForCausalLM,
        torch,
        device=device_spec.device,
        dtype=dtype,
    )
    holdout = json.loads(data)["holdout"]
    prompt_ids = [
        list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        for row in holdout
    ]
    batch_size = getattr(args, "batch_size", 4)

    def generate(m: Any) -> list[dict[str, Any]]:
        outputs = batched_greedy_generate(
            m,
            tokenizer,
            prompt_ids,
            max_new_tokens=8,
            batch_size=batch_size,
            return_confidence=False,
            torch_module=torch,
            device=device_spec.device,
        )
        return [
            {
                "id": row["id"],
                "expected": row["answer"],
                "output": res.text,
                "max_token_limit_hit": res.max_token_limit_hit,
            }
            for row, res in zip(holdout, outputs, strict=True)
        ]

    base = generate(model)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
    adapted = generate(model)
    result = {
        "status": "completed",
        "data_origin": "synthetic",
        "model": spec.model_id,
        "revision": spec.revision,
        "base_model_sha256": manifest.get("base_model_sha256", spec.model_sha256),
        "adapter_sha256": manifest["adapter_sha256"],
        "adapter_config_sha256": manifest["adapter_config_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "training_script_sha256": manifest["script_sha256"],
        "evaluation_script_sha256": sha_file(Path(__file__)),
        "device": device_spec.device,
        "batch_size": batch_size,
        "decoding": {"do_sample": False, "max_new_tokens": 8},
        "base": {"metrics": metrics(base), "cases": base},
        "adapter": {"metrics": metrics(adapted), "cases": adapted},
        "regressions": sum(
            b["output"] == b["expected"] and a["output"] != a["expected"]
            for b, a in zip(base, adapted, strict=True)
        ),
        "improvements": sum(
            b["output"] != b["expected"] and a["output"] == a["expected"]
            for b, a in zip(base, adapted, strict=True)
        ),
        "duration_seconds": round(time.monotonic() - started, 2),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "limitations": [
            "four authored synthetic examples",
            "holdout already used for teacher-forced evaluation; not pristine",
            "not customer data or commercial validation",
            "no promotion",
        ],
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    chmod_private(result_path)
    print(json.dumps({"evaluation_dir": str(output), **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
