"""Synthetic out-of-template challenge for a frozen balanced LoRA adapter.

Never train or select checkpoints on this challenge. No promotion or production use.
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
        DEFAULT_MODEL_ID,
        DEFAULT_MODEL_SHA256,
        DEFAULT_REVISION,
        SUPPORTED_DEVICES,
        TrainingLockError,
        acquire_training_lock,
        batched_greedy_generate,
        chmod_private,
        load_verified_snapshot,
        mkdir_private,
        resolve_model_spec,
        select_device,
        set_offline_env,
        sha_file,
        verify_private_dir,
    )
except ImportError:
    sys.modules.pop("hardonian_foundry", None)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hardonian_foundry.training_utils import (
        DEFAULT_MODEL_ID,
        DEFAULT_MODEL_SHA256,
        DEFAULT_REVISION,
        SUPPORTED_DEVICES,
        TrainingLockError,
        acquire_training_lock,
        batched_greedy_generate,
        chmod_private,
        load_verified_snapshot,
        mkdir_private,
        resolve_model_spec,
        select_device,
        set_offline_env,
        sha_file,
        verify_private_dir,
    )

try:
    from evaluate_synthetic_lora import verify_adapter_hashes
    from experiment_balanced_lora import preflight
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_synthetic_lora import verify_adapter_hashes
    from experiment_balanced_lora import preflight

MODEL = DEFAULT_MODEL_ID
REVISION = DEFAULT_REVISION
MODEL_SHA256 = DEFAULT_MODEL_SHA256

ALLOWED_CHALLENGE_LABELS: tuple[str, ...] = ("matched", "amount_mismatch", "review_required")


def oracle(
    expected: int | None,
    actual: int | None,
    expected_ccy: str | None,
    actual_ccy: str | None,
) -> str:
    """Deterministic minor-unit oracle requiring review on missing fields or FX mismatch."""
    if (
        expected is None
        or actual is None
        or not expected_ccy
        or not actual_ccy
        or expected_ccy != actual_ccy
    ):
        return "review_required"
    return "matched" if expected == actual else "amount_mismatch"


def challenge_cases() -> list[dict[str, Any]]:
    """Return 48 balanced OOD synthetic challenge cases across 3 classes (16 each)."""
    # Only fictional minor-unit values; covers negatives, sign flips, zero-crossing,
    # large values, commas, multiple currencies, missing fields, and distractor clauses.
    specs: list[tuple[int | None, int | None, str | None, str | None, str]] = [
        # 16 matched
        (-24000, -24000, "USD", "USD", ""),
        (-35000, -35000, "EUR", "EUR", ""),
        (0, 0, "USD", "USD", ""),
        (999999, 999999, "USD", "USD", ""),
        (90000000, 90000000, "USD", "USD", ""),
        (43210, 43210, "USD", "USD", ""),
        (88888, 88888, "EUR", "EUR", ""),
        (-1, -1, "USD", "USD", ""),
        (5000000000, 5000000000, "GBP", "GBP", ""),
        (100, 100, "JPY", "JPY", ""),
        (-99999999, -99999999, "CHF", "CHF", ""),
        (123456789, 123456789, "CAD", "CAD", " Prior memo flagged variance, verify current totals."),
        (77, 77, "GBP", "GBP", " Ignore earlier draft amount 999."),
        (-500, -500, "EUR", "EUR", " Note: fee dispute closed without change."),
        (4200000, 4200000, "AUD", "AUD", ""),
        (1, 1, "CHF", "CHF", ""),
        # 16 amount_mismatch
        (-120000, -120001, "USD", "USD", ""),
        (-88000, -87999, "USD", "USD", ""),
        (0, -1, "USD", "USD", ""),
        (0, 1, "EUR", "EUR", ""),
        (1000000, 1000001, "USD", "USD", ""),
        (9999999, 9999998, "EUR", "EUR", ""),
        (75000000, 75000005, "EUR", "EUR", ""),
        (54321, 54320, "EUR", "EUR", ""),
        (-123, 123, "USD", "USD", ""),
        (123, -123, "GBP", "GBP", ""),
        (5000000000, 5000000001, "CHF", "CHF", ""),
        (10000, 9999, "JPY", "JPY", " Prior draft assumed matched."),
        (-42, -43, "CAD", "CAD", " Ignore comment saying balanced."),
        (250000, 250100, "AUD", "AUD", ""),
        (999, 1000, "GBP", "GBP", ""),
        (-9000000, -8999999, "USD", "USD", ""),
        # 16 review_required
        (None, 42000, "USD", "USD", ""),
        (42000, None, "USD", "USD", ""),
        (None, None, "EUR", "EUR", ""),
        (125000, 125000, "USD", "EUR", ""),
        (48000, 48001, "EUR", "USD", ""),
        (0, 0, None, "USD", ""),
        (-24000, -24000, "EUR", None, ""),
        (99999, None, "USD", "EUR", ""),
        (15000, 15000, "GBP", "CHF", " Amounts match numerically; currencies differ."),
        (None, 0, "JPY", "JPY", ""),
        (0, None, "CAD", "CAD", ""),
        (64000, 64000, None, "USD", ""),
        (64000, 64000, "USD", None, ""),
        (-1000, -1000, "GBP", "USD", ""),
        (None, None, None, None, ""),
        (88888, 88888, "AUD", "NZD", " Treat equal minor units as matched only if currencies agree."),
    ]
    templates = [
        "Synthetic check: expected {e} {ec}; actual {a} {ac}.",
        "Fictional ledger: actual={a} {ac} versus expected={e} {ec}.",
        "Compare expected ({ec} {e}) with observed ({ac} {a}) for this made-up record.",
        "Adjudication probe -> expected={ec} {e} | observed={ac} {a}.",
    ]
    instructions = (
        " If either amount or either currency is missing, or currencies differ, return review_required. "
        "Otherwise return matched if the integer minor-unit amounts are equal, or amount_mismatch if different. "
        "Return exactly one label: matched, amount_mismatch, or review_required."
    )

    def fmt(value: int | None) -> str:
        return "missing" if value is None else format(value, ",d")

    rows: list[dict[str, Any]] = []
    for i, (expected, actual, ec, ac, note) in enumerate(specs):
        ec_disp = ec if ec else "missing"
        ac_disp = ac if ac else "missing"
        prompt = (
            templates[i % len(templates)].format(
                e=fmt(expected), a=fmt(actual), ec=ec_disp, ac=ac_disp
            )
            + note
            + instructions
        )
        rows.append(
            {
                "id": f"synthetic-challenge-{i:03d}",
                "expected_minor": expected,
                "actual_minor": actual,
                "expected_currency": ec,
                "actual_currency": ac,
                "expected_label": oracle(expected, actual, ec, ac),
                "prompt": prompt,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize challenge accuracy, invalid outputs, and unsafe matched suggestions."""
    labels = ALLOWED_CHALLENGE_LABELS
    by_label: dict[str, dict[str, int]] = {}
    for label in labels:
        group = [r for r in rows if r["expected"] == label]
        by_label[label] = {
            "total": len(group),
            "correct": sum(r["output"] == label for r in group),
            "normalized_correct": sum(r["output"].lower() == label for r in group),
        }
    return {
        "total": len(rows),
        "strict_correct": sum(r["output"] == r["expected"] for r in rows),
        "normalized_correct": sum(r["output"].lower() == r["expected"] for r in rows),
        "invalid_outputs": sum(r["output"] not in labels for r in rows),
        "unsafe_matched": sum(r["expected"] != "matched" and r["output"].lower() == "matched" for r in rows),
        "token_limit_hits": sum(r["max_token_limit_hit"] for r in rows),
        "by_expected": by_label,
    }


def format_foundry_predictions(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map raw challenge outputs to Foundry `{id, label}` records, failing closed to `review_required`."""
    preds: list[dict[str, Any]] = []
    allowed = set(ALLOWED_CHALLENGE_LABELS)
    for row in cases:
        raw = str(row.get("output", "")).strip().lower()
        is_valid = raw in allowed
        record: dict[str, Any] = {
            "id": row["id"],
            "label": raw if is_valid else "review_required",
            "raw_invalid": not is_valid,
        }
        if row.get("confidence") is not None:
            record["confidence"] = row["confidence"]
        if row.get("latency_ms") is not None:
            record["latency_ms"] = row["latency_ms"]
        preds.append(record)
    return preds


def write_foundry_predictions(path: Path, cases: list[dict[str, Any]]) -> Path:
    """Write Foundry-compatible prediction JSONL for `task_eval.score_predictions`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    preds = format_foundry_predictions(cases)
    with path.open("w", encoding="utf-8") as fh:
        for pred in preds:
            fh.write(json.dumps(pred, sort_keys=True) + "\n")
    return path


def main() -> None:
    if os.name != "nt":
        os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-home", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--threads", type=int, default=4, choices=range(1, 5))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cpu", choices=SUPPORTED_DEVICES)
    p.add_argument("--model-registry", type=Path, default=None)
    p.add_argument("--emit-predictions", type=Path, default=None, help="optional Foundry JSONL output path")
    args = p.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    try:
        root = args.output_root.expanduser().resolve(strict=True)
        cache = args.hf_home.expanduser().resolve(strict=True)
        run = verify_private_dir(args.run_dir, label="training run")
        snap = preflight(cache, root, registry_path=args.model_registry)
        manifest = json.loads((run / "result.json").read_text(encoding="utf-8"))
        spec = resolve_model_spec(manifest.get("model"), registry_path=args.model_registry)
        if manifest.get("status") != "completed" or manifest.get("model") != spec.model_id or manifest.get("revision") != spec.revision:
            raise ValueError("training manifest mismatch")
        adapter = run / "adapter"
        verify_adapter_hashes(adapter, manifest)
        data_path = run / "synthetic-dataset.json"
        if sha_file(data_path) != manifest.get("dataset_sha256"):
            raise ValueError("training dataset hash mismatch")
        dataset = json.loads(data_path.read_text(encoding="utf-8"))
        challenge = challenge_cases()
        prior_prompts = {r["prompt"] for rows in dataset.values() for r in rows}
        if any(row["prompt"] in prior_prompts for row in challenge):
            raise ValueError("challenge overlaps previous prompts")
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"preflight failed: {exc}") from exc

    try:
        with acquire_training_lock(root / ".training.lock"):
            run_eval(args, root, cache, snap, adapter, manifest, challenge, model_spec=spec)
    except TrainingLockError as exc:
        raise SystemExit(str(exc)) from exc


def run_eval(
    args: argparse.Namespace,
    root: Path,
    cache: Path,
    snap: Path,
    adapter: Path,
    manifest: dict[str, Any],
    challenge: list[dict[str, Any]],
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
    started = time.monotonic()
    output = mkdir_private(
        root / (time.strftime("synthetic-challenge-%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"),
        exist_ok=False,
    )
    content = json.dumps(challenge, sort_keys=True, separators=(",", ":")).encode()
    (output / "synthetic-challenge.json").write_bytes(content)
    protocol = {
        "challenge_sha256": hashlib.sha256(content).hexdigest(),
        "adapter_sha256": manifest["adapter_sha256"],
        "adapter_config_sha256": manifest["adapter_config_sha256"],
        "model": spec.model_id,
        "revision": spec.revision,
        "scoring": "exact and case-normalized labels; missing/different currency requires review",
        "do_sample": False,
        "max_new_tokens": 8,
        "training_on_challenge": False,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    dtype = getattr(torch, device_spec.torch_dtype_name, torch.float32)
    tokenizer, model = load_verified_snapshot(
        snap,
        AutoTokenizer,
        AutoModelForCausalLM,
        torch,
        device=device_spec.device,
        dtype=dtype,
    )
    prompt_seqs = [
        list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        for row in challenge
    ]
    batch_size = getattr(args, "batch_size", 4)

    def generate(m: Any) -> list[dict[str, Any]]:
        outputs = batched_greedy_generate(
            m,
            tokenizer,
            prompt_seqs,
            max_new_tokens=8,
            batch_size=batch_size,
            return_confidence=True,
            torch_module=torch,
            device=device_spec.device,
        )
        records: list[dict[str, Any]] = []
        for row, res in zip(challenge, outputs, strict=True):
            rec: dict[str, Any] = {
                "id": row["id"],
                "expected": row["expected_label"],
                "output": res.text,
                "max_token_limit_hit": res.max_token_limit_hit,
                "latency_ms": res.latency_ms,
            }
            if res.confidence is not None:
                rec["confidence"] = round(res.confidence, 6)
            records.append(rec)
        return records

    base = generate(model)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
    adapted = generate(model)
    write_foundry_predictions(output / "challenge_predictions.jsonl", adapted)
    emit_path = getattr(args, "emit_predictions", None)
    if emit_path is not None:
        write_foundry_predictions(Path(emit_path), adapted)

    result = {
        "status": "completed",
        "data_origin": "synthetic",
        "model": spec.model_id,
        "revision": spec.revision,
        "base_model_sha256": manifest.get("base_model_sha256", spec.model_sha256),
        "adapter_sha256": manifest["adapter_sha256"],
        "adapter_config_sha256": manifest["adapter_config_sha256"],
        "challenge_sha256": protocol["challenge_sha256"],
        "training_dataset_sha256": manifest["dataset_sha256"],
        "protocol_sha256": sha_file(output / "protocol.json"),
        "script_sha256": sha_file(Path(__file__)),
        "device": device_spec.device,
        "batch_size": batch_size,
        "base": {"metrics": summarize(base), "cases": base},
        "adapter": {"metrics": summarize(adapted), "cases": adapted},
        "duration_seconds": round(time.monotonic() - started, 2),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "limitations": [
            "authored synthetic challenge not representative of production",
            "no retraining on challenge",
            "observed test is no longer pristine",
            "unsigned metadata",
            "all production outcomes require deterministic policy and human review",
        ],
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    chmod_private(result_path)
    print(
        json.dumps(
            {
                "evaluation_dir": str(output),
                "challenge_sha256": result["challenge_sha256"],
                "base": result["base"]["metrics"],
                "adapter": result["adapter"]["metrics"],
                "duration_seconds": result["duration_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
