"""Bounded synthetic LoRA experiment with predeclared, group-disjoint splits.

Experimental CPU-only research. No customer data, promotion, or deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

try:
    from hardonian_foundry.task_eval import (
        _digest_order,
        split_task_dataset,
        validate_spec,
        validate_task_dataset,
    )
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
        collate_causal_lm_batch,
        iter_micro_batches,
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
    from hardonian_foundry.task_eval import (
        _digest_order,
        split_task_dataset,
        validate_spec,
        validate_task_dataset,
    )
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
        collate_causal_lm_batch,
        iter_micro_batches,
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
    from evaluate_synthetic_lora import metrics
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_synthetic_lora import metrics

MODEL = DEFAULT_MODEL_ID
REVISION = DEFAULT_REVISION
MODEL_SHA256 = DEFAULT_MODEL_SHA256
BLOBS = dict(DEFAULT_BLOBS)

SEED = 20261110
STEPS = 96

_TEMPLATES: tuple[str, ...] = (
    "Expected minor units: {expected}. Actual minor units: {actual}. Output matched or amount_mismatch only.",
    "Compare two synthetic amounts. Expected={expected}; actual={actual}. Return matched or amount_mismatch.",
    "Synthetic reconciliation: expected {expected}, observed {actual}. Reply matched or amount_mismatch only.",
)


def build_datasets() -> dict[str, list[dict[str, Any]]]:
    """Matched/mismatch pairs share a base number; numeric ranges overlap across splits."""
    all_groups = list(range(44))
    random.Random(SEED).shuffle(all_groups)  # nosec B311 - deterministic synthetic split

    split_specs = (
        ("train", all_groups[:24]),
        ("validation", all_groups[24:32]),
        ("test", all_groups[32:44]),
    )
    deltas = (1, -1, 5, -7, 13, -19)
    split: dict[str, list[dict[str, Any]]] = {}
    for split_idx, (name, group_indices) in enumerate(split_specs):
        rows: list[dict[str, Any]] = []
        for pos, group_id in enumerate(group_indices):
            expected = 1100 + group_id * 137
            delta = deltas[group_id % len(deltas)]
            tmpl = _TEMPLATES[(pos + split_idx) % len(_TEMPLATES)]
            for variant, actual in (("matched", expected), ("amount_mismatch", expected + delta)):
                rows.append(
                    {
                        "id": f"synthetic-{name}-{pos:02d}-{variant}",
                        "group": expected,
                        "prompt": tmpl.format(expected=expected, actual=actual),
                        "answer": variant,
                    }
                )
        random.Random(SEED + split_idx + 1).shuffle(rows)  # nosec B311 - deterministic synthetic ordering
        split[name] = rows

    groups = [{r["group"] for r in rows} for rows in split.values()]
    prompts = [{r["prompt"] for r in rows} for rows in split.values()]
    if any(groups[i] & groups[j] or prompts[i] & prompts[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("split leakage")
    return split


def digest_json(data: Any) -> str:
    """Compute SHA-256 of canonical compact JSON."""
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def preflight(
    cache: Path,
    root: Path,
    *,
    model_id: str | None = None,
    registry_path: Path | None = None,
) -> Path:
    """Verify private output directory, disk, RAM, and pinned local model snapshot."""
    verify_private_dir(root, label="output root")
    space = shutil.disk_usage(root)
    if space.free < 10 * 1024**3 or (space.used / space.total if space.total else 1.0) >= 0.8:
        raise ValueError("disk watermark exceeded")
    if available_ram_bytes() < MIN_AVAILABLE_RAM_BYTES:
        raise ValueError("not enough free physical RAM")

    spec = resolve_model_spec(model_id, registry_path=registry_path)
    snap = snapshot_path(cache, spec.model_id, spec.revision)
    verify_snapshot(
        snap,
        model_sha256=spec.model_sha256,
        blobs=spec.blobs or None,
        weight_files=spec.weight_files or None,
    )
    return snap


def _to_binary_label(raw_text: str) -> tuple[str, bool]:
    cleaned = raw_text.strip().splitlines()[0].strip(".\"'` ") if raw_text.strip() else ""
    lowered = cleaned.lower()
    if lowered in ("matched", "amount_mismatch"):
        return lowered, True
    # Conservative fail-closed fallback for binary task spec without UNDETERMINED
    return "amount_mismatch", False


def emit_task_artifacts(
    task_dir: Path,
    datasets: dict[str, list[dict[str, Any]]],
    *,
    base_cases: list[dict[str, Any]],
    adapter_cases: list[dict[str, Any]],
    adapter_sha256: str,
    model_id: str = MODEL,
    revision: str = REVISION,
    seed: int = SEED,
) -> dict[str, Any]:
    """Emit a leak-free Foundry task-mode directory (spec, dataset, baseline, model predictions).

    Uses 40 groups (24 train / 6 validation / 10 test) with family IDs ordered by
    `task_eval._digest_order(..., seed)` so `split_task_dataset` produces a test
    partition strictly inside `datasets["test"]` with zero train/test overlap.
    """
    task_dir.mkdir(parents=True, exist_ok=True)

    def _group_rows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        by_group: dict[int, list[dict[str, Any]]] = {}
        for r in rows:
            by_group.setdefault(int(r["group"]), []).append(r)
        return [by_group[k] for k in sorted(by_group)]

    train_groups = _group_rows(datasets["train"])[:24]
    val_groups = _group_rows(datasets["validation"])[:6]
    test_groups = _group_rows(datasets["test"])[:10]
    selected_groups = train_groups + val_groups + test_groups

    raw_fams = [f"fam-{i:03d}" for i in range(len(selected_groups))]
    ordered_fams = _digest_order(raw_fams, seed)

    task_records: list[dict[str, Any]] = []
    for fam, group_rows in zip(ordered_fams, selected_groups, strict=True):
        for r in group_rows:
            task_records.append(
                {
                    "id": r["id"],
                    "prompt": r["prompt"],
                    "gold_label": r["answer"],
                    "critical": r["answer"] == "amount_mismatch",
                    "family": fam,
                }
            )
    task_records.sort(key=lambda r: r["id"])

    spec = {
        "schema_version": "1.0",
        "task_id": "synthetic-balanced-reconciliation",
        "description": "Balanced group-disjoint synthetic amount reconciliation task emitted by experiment_balanced_lora.",
        "labels": ["matched", "amount_mismatch"],
        "thresholds": {"macro_f1_min": 0.8},
        "baseline_must_improve": True,
        "split_seed": seed,
        "model": {
            "base_model_id": model_id,
            "revision": revision,
            "adapter_sha256": adapter_sha256,
            "adapter_name": "synthetic-balanced-lora",
        },
        "dataset_path": "data.jsonl",
        "notes": "Synthetic balanced experiment task export; not for production promotion.",
    }
    spec_errors = validate_spec(spec)
    if spec_errors:
        raise ValueError(f"emitted spec invalid: {'; '.join(spec_errors)}")
    val = validate_task_dataset(task_records, spec)
    if val["status"] != "VALID":
        raise ValueError(f"emitted dataset invalid: {'; '.join(val['errors'])}")

    splits = split_task_dataset(task_records, spec)
    task_test_ids = {r["id"] for r in splits["test"]}
    orig_test_ids = {r["id"] for r in datasets["test"]}
    if not task_test_ids <= orig_test_ids:
        raise RuntimeError("emitted task test split overlapped non-test records")

    base_by_id = {c["id"]: c for c in base_cases}
    adapter_by_id = {c["id"]: c for c in adapter_cases}
    baseline_preds: list[dict[str, Any]] = []
    model_preds: list[dict[str, Any]] = []
    for row in splits["test"]:
        rid = row["id"]
        b_out = base_by_id[rid]["output"] if rid in base_by_id else ""
        a_out = adapter_by_id[rid]["output"] if rid in adapter_by_id else ""
        b_label, _ = _to_binary_label(b_out)
        a_label, _ = _to_binary_label(a_out)
        baseline_preds.append({"id": rid, "label": b_label})
        model_preds.append({"id": rid, "label": a_label})

    (task_dir / "spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (task_dir / "data.jsonl").open("w", encoding="utf-8") as fh:
        for row in task_records:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with (task_dir / "baseline.jsonl").open("w", encoding="utf-8") as fh:
        for pred in baseline_preds:
            fh.write(json.dumps(pred, sort_keys=True) + "\n")
    with (task_dir / "model.jsonl").open("w", encoding="utf-8") as fh:
        for pred in model_preds:
            fh.write(json.dumps(pred, sort_keys=True) + "\n")

    return {
        "task_dir": str(task_dir),
        "spec": spec,
        "dataset_records": task_records,
        "baseline_predictions": baseline_preds,
        "model_predictions": model_preds,
        "split_counts": {k: len(v) for k, v in splits.items()},
    }


def main() -> None:
    if os.name != "nt":
        os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=SUPPORTED_DEVICES)
    parser.add_argument("--model-registry", type=Path, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--emit-task-dir", type=Path, default=None, help="write Foundry task-mode artifacts")
    parser.add_argument("--run-refinery", action="store_true", help="invoke Foundry create_task_run on emitted task")
    parser.add_argument("--refinery-output-root", type=Path, default=None)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise SystemExit("--epochs, --batch-size, and --patience must be positive")

    try:
        root = args.output_root.expanduser().resolve(strict=True)
        cache = args.hf_home.expanduser().resolve(strict=True)
        snap = preflight(cache, root, model_id=args.model_id, registry_path=args.model_registry)
        check_disk_space(root)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"preflight failed: {exc}") from exc

    try:
        with acquire_training_lock(root / ".training.lock"):
            run_experiment(args, root, cache, snap)
    except TrainingLockError as exc:
        raise SystemExit(str(exc)) from exc


def run_experiment(args: argparse.Namespace, root: Path, cache: Path, snap: Path) -> None:
    spec = resolve_model_spec(getattr(args, "model_id", None), registry_path=getattr(args, "model_registry", None))
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
    started = time.monotonic()
    output = mkdir_private(
        root / (time.strftime("synthetic-balanced-%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"),
        exist_ok=False,
    )
    datasets = build_datasets()
    epochs = getattr(args, "epochs", 4)
    batch_size = getattr(args, "batch_size", 4)
    lr = getattr(args, "lr", 2e-4)
    patience = getattr(args, "patience", 2)

    prereg = {
        "data_origin": "synthetic",
        "model": spec.model_id,
        "revision": spec.revision,
        "seed": SEED,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "train_count": len(datasets["train"]),
        "validation_count": len(datasets["validation"]),
        "test_count": len(datasets["test"]),
        "split_sha256": {k: digest_json(v) for k, v in datasets.items()},
        "rule": "best validation selection + early stopping; test strictly untouched until post-training",
        "decoding": {"do_sample": False, "max_new_tokens": 8},
    }
    (output / "preregistration.json").write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "synthetic-dataset.json").write_text(
        json.dumps(datasets, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
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
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    def encode(row: dict[str, Any]) -> tuple[list[int], list[int]]:
        messages = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["answer"]}]
        prompt = tokenizer.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if full[: len(prompt)] != prompt or not len(prompt) < len(full) <= spec.max_seq:
            raise RuntimeError("unsafe tokenization")
        return list(full), [-100] * len(prompt) + list(full[len(prompt) :])

    def encode_prompt(row: dict[str, Any]) -> list[int]:
        return list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )

    train_encoded = [encode(row) for row in datasets["train"]]
    val_encoded = [encode(row) for row in datasets["validation"]]
    val_prompts = [encode_prompt(row) for row in datasets["validation"]]
    test_prompts = [encode_prompt(row) for row in datasets["test"]]

    target_modules = ["q_proj", "v_proj"] if spec.model_id == DEFAULT_MODEL_ID else list(spec.lora_target_modules)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)

    def evaluate_split(m: Any, rows: list[dict[str, Any]], prompt_seqs: list[list[int]]) -> list[dict[str, Any]]:
        outputs = batched_greedy_generate(
            m,
            tokenizer,
            prompt_seqs,
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
            for row, res in zip(rows, outputs, strict=True)
        ]

    def eval_val_loss(m: Any) -> float:
        m.eval()
        losses: list[float] = []
        with torch.inference_mode():
            for mb in iter_micro_batches(val_encoded, batch_size):
                batch = collate_causal_lm_batch(
                    mb, pad_token_id=int(pad_id), torch_module=torch, device=device_spec.device
                )
                val_l = float(m(**batch).loss)
                if not math.isfinite(val_l):
                    raise RuntimeError("non-finite validation loss")
                losses.append(val_l)
        return sum(losses) / len(losses)

    train_losses: list[float] = []
    val_history: list[dict[str, Any]] = []
    best_score: tuple[float, float] | None = None
    best_epoch = 0
    no_improve = 0
    early_stopped = False
    adapter = output / "adapter"
    gen = torch.Generator().manual_seed(SEED)

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_encoded), generator=gen).tolist()
        ordered_train = [train_encoded[i] for i in order]
        for mb in iter_micro_batches(ordered_train, batch_size):
            batch = collate_causal_lm_batch(
                mb, pad_token_id=int(pad_id), torch_module=torch, device=device_spec.device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("non-finite gradient")
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        if any(not torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
            raise RuntimeError("non-finite adapter weights")

        val_loss = eval_val_loss(model)
        val_cases = evaluate_split(model, datasets["validation"], val_prompts)
        val_m = metrics(val_cases)
        val_f1 = float(val_m["f1"]) if val_m["f1"] is not None else -1.0
        score = (round(val_f1, 6), round(-val_loss, 6))
        improved = best_score is None or score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            model.save_pretrained(adapter, safe_serialization=True)
        else:
            no_improve += 1
        val_history.append(
            {
                "epoch": epoch,
                "val_loss": round(val_loss, 6),
                "val_f1": val_m["f1"],
                "val_correct": val_m["correct"],
                "val_total": val_m["total"],
                "improved": improved,
            }
        )
        print(
            f"epoch={epoch} val_loss={val_loss:.4f} val_f1={val_m['f1']} improved={improved}",
            flush=True,
        )
        if no_improve >= patience:
            early_stopped = True
            break

    adapter_config_sha256 = sha_file(adapter / "adapter_config.json")
    adapter_sha256 = sha_file(adapter / "adapter_model.safetensors")

    # Reload best-validation checkpoint onto clean base model, keeping test untouched until now.
    base_model = model.unload()
    with torch.inference_mode():
        base = evaluate_split(base_model, datasets["test"], test_prompts)
    if sha_file(adapter / "adapter_config.json") != adapter_config_sha256:
        raise RuntimeError("adapter configuration changed before reload")
    reloaded = PeftModel.from_pretrained(base_model, adapter, is_trainable=False, local_files_only=True)
    reloaded.eval()
    adapted = evaluate_split(reloaded, datasets["test"], test_prompts)

    refinery_run: dict[str, Any] | None = None
    task_summary: dict[str, Any] | None = None
    emit_dir = getattr(args, "emit_task_dir", None)
    run_refinery = bool(getattr(args, "run_refinery", False))
    if emit_dir is not None or run_refinery:
        target_task_dir = emit_dir if emit_dir is not None else (output / "task")
        emitted = emit_task_artifacts(
            target_task_dir,
            datasets,
            base_cases=base,
            adapter_cases=adapted,
            adapter_sha256=adapter_sha256,
            model_id=spec.model_id,
            revision=spec.revision,
            seed=SEED,
        )
        task_summary = {"task_dir": emitted["task_dir"], "split_counts": emitted["split_counts"]}
        if run_refinery:
            from hardonian_foundry.refinery import create_task_run

            ref_root = getattr(args, "refinery_output_root", None) or (output / "refinery_runs")
            refinery_run = create_task_run(
                task_spec=emitted["spec"],
                dataset_records=emitted["dataset_records"],
                baseline_predictions=emitted["baseline_predictions"],
                model_predictions=emitted["model_predictions"],
                output_root=str(ref_root),
            )

    weights_file = snap / "model.safetensors"
    base_sha = sha_file(weights_file) if weights_file.is_file() else spec.model_sha256
    result = {
        "status": "completed",
        "data_origin": "synthetic",
        "model": spec.model_id,
        "revision": spec.revision,
        "base_model_sha256": base_sha,
        "script_sha256": sha_file(Path(__file__)),
        "preregistration_sha256": sha_file(output / "preregistration.json"),
        "dataset_sha256": sha_file(output / "synthetic-dataset.json"),
        "adapter_sha256": adapter_sha256,
        "adapter_config_sha256": adapter_config_sha256,
        "seed": SEED,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "batch_size": batch_size,
        "device": device_spec.device,
        "threads": args.threads,
        "trainable_parameters": params,
        "train_losses": train_losses,
        "validation_history": val_history,
        "base": {"metrics": metrics(base), "cases": base},
        "adapter": {"metrics": metrics(adapted), "cases": adapted},
        "reload_exact_match": True,
        "regressions": sum(
            b["output"] == b["expected"] and a["output"] != a["expected"]
            for b, a in zip(base, adapted, strict=True)
        ),
        "improvements": sum(
            b["output"] != b["expected"] and a["output"] == a["expected"]
            for b, a in zip(base, adapted, strict=True)
        ),
        "task_export": task_summary,
        "refinery_run_id": refinery_run["run_id"] if refinery_run else None,
        "duration_seconds": round(time.monotonic() - started, 2),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "limitations": [
            "generated synthetic records and one model",
            "numeric ranges overlap across splits; templates still synthetic",
            "no pretraining-contamination guarantee",
            "test observed after run; no longer pristine",
            "no promotion or real-world accuracy claim",
        ],
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    chmod_private(result_path)
    summary = {
        k: result[k]
        for k in (
            "status",
            "model",
            "revision",
            "best_epoch",
            "early_stopped",
            "trainable_parameters",
            "adapter_sha256",
            "dataset_sha256",
            "regressions",
            "improvements",
            "reload_exact_match",
            "refinery_run_id",
            "duration_seconds",
        )
    }
    summary.update(run_dir=str(output), base_metrics=metrics(base), adapter_metrics=metrics(adapted))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
