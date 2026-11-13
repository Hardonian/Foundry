#!/usr/bin/env python3
"""Score the frozen test split with the fine-tuned adapter (batched greedy decoding).

Offline, CPU-by-default. Enforces full provenance verification (`spec.json`,
`train_manifest.json`, `adapter_model.safetensors`, `base_weights_sha256`, and
snapshot Git blobs/`auto_map`), uses `torch.inference_mode()` with left-padded
batched generation, optionally restricts outputs via a token prefix-trie, and
maps any off-allowlist generation safely to UNDETERMINED.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from hardonian_foundry.baseline import digest
    from hardonian_foundry.task_eval import split_task_dataset, validate_spec
    from hardonian_foundry.training_utils import (
        DEFAULT_REVISION,
        SUPPORTED_DEVICES,
        LabelPrefixTrie,
        batched_greedy_generate,
        load_verified_snapshot,
        percentile,
        resolve_model_spec,
        select_device,
        set_offline_env,
        sha_bytes,
        sha_file,
        snapshot_path,
        verify_snapshot,
    )
except ImportError:
    sys.modules.pop("hardonian_foundry", None)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from hardonian_foundry.baseline import digest
    from hardonian_foundry.task_eval import split_task_dataset, validate_spec
    from hardonian_foundry.training_utils import (
        DEFAULT_REVISION,
        SUPPORTED_DEVICES,
        LabelPrefixTrie,
        batched_greedy_generate,
        load_verified_snapshot,
        percentile,
        resolve_model_spec,
        select_device,
        set_offline_env,
        sha_bytes,
        sha_file,
        snapshot_path,
        verify_snapshot,
    )

REVISION = DEFAULT_REVISION
PROMPT_SYS = (
    "You classify reconciliation exception notes into exactly one resolution code. "
    "Reply with only the code, nothing else."
)


def sha256_file(path: str | Path) -> str:
    """Backward-compatible alias for `sha_file`."""
    return sha_file(path)


def compute_spec_core_sha256(spec: dict[str, Any]) -> str:
    """Hash canonical spec with `model.adapter_sha256` normalized to 64 zeros."""
    normalized = json.loads(json.dumps(spec))
    if isinstance(normalized.get("model"), dict):
        normalized["model"]["adapter_sha256"] = "0" * 64
    return digest(normalized)


def to_label(
    text: str,
    allowlist: Sequence[str],
    *,
    fallback: str = "UNDETERMINED",
) -> tuple[str, str]:
    """Parse generated text into an allowlist label and match classification.

    Returns `(label, match_type)` where `match_type` is one of:
    - `"exact"`: trimmed first line matches an allowlist entry verbatim
    - `"normalized"`: matched after case/separator normalization or longest-first
      boundary-aware prefix matching (prevents `FOO` colliding with `FOO_BAR` or `FOO_BAZ`)
    - `"undetermined"`: off-allowlist output mapped fail-closed to `fallback`
    """
    if not allowlist:
        raise ValueError("allowlist must be nonempty")
    allowset = set(allowlist)
    resolved_fallback = fallback if fallback in allowset else allowlist[-1]

    raw = text.strip().splitlines()[0].strip() if text and text.strip() else ""
    if not raw:
        return resolved_fallback, "undetermined"

    cleaned = raw.strip(".\"'` \t")
    if cleaned in allowset:
        return cleaned, "exact"

    norm_lookup = {re.sub(r"[\s-]+", "_", lbl.strip().upper()): lbl for lbl in allowlist}
    norm_cleaned = re.sub(r"[\s-]+", "_", cleaned.upper())
    if norm_cleaned in norm_lookup:
        return norm_lookup[norm_cleaned], "normalized"

    # Longest allowlist entry first, with a strict non-identifier boundary so
    # "FOO" never steals "FOO_BAR ..." or falsely accepts "FOO_BAZ".
    upper_cleaned = cleaned.upper()
    for lbl in sorted(allowlist, key=len, reverse=True):
        parts = [re.escape(part) for part in re.split(r"[\s_-]+", lbl.upper()) if part]
        if not parts:
            continue
        pattern = r"^" + r"[\s_-]+".join(parts) + r"(?:$|[^A-Z0-9_])"
        if re.match(pattern, upper_cleaned):
            return lbl, "normalized"

    return resolved_fallback, "undetermined"


def verify_prediction_provenance(
    *,
    task_dir: Path,
    spec: dict[str, Any],
    snap_info: dict[str, Any],
    adapter_dir: Path | None,
    base_only: bool = False,
) -> dict[str, Any]:
    """Verify dataset, spec, base weights, and adapter provenance hashes fail-closed."""
    dataset_sha = sha_file(task_dir / "data.jsonl")
    spec_sha = sha_file(task_dir / "spec.json")
    spec_core_sha = compute_spec_core_sha256(spec)
    base_weights_sha = snap_info.get("weights_sha256")

    if base_only:
        return {
            "dataset_sha256": dataset_sha,
            "spec_sha256": spec_sha,
            "base_weights_sha256": base_weights_sha,
            "adapter_sha256": None,
            "manifest": None,
        }

    if adapter_dir is None:
        raise ValueError("--adapter-dir is required unless --base-only is set")
    manifest_path = adapter_dir / "train_manifest.json"
    adapter_weights_path = adapter_dir / "adapter_model.safetensors"
    if not manifest_path.is_file() or not adapter_weights_path.is_file():
        raise ValueError(f"missing train_manifest.json or adapter_model.safetensors in {adapter_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_sha256") != dataset_sha:
        raise ValueError("dataset changed since training; refusing to predict")

    actual_adapter_sha = sha_file(adapter_weights_path)
    if manifest.get("adapter_sha256") != actual_adapter_sha:
        raise ValueError("adapter weights changed since training; refusing to predict")

    spec_model = spec.get("model", {})
    spec_adapter_sha = spec_model.get("adapter_sha256")
    if not spec_adapter_sha or spec_adapter_sha == "0" * 64:
        raise ValueError("spec.json has unfinalized adapter_sha256; run finalize_spec.py first")
    if spec_adapter_sha != actual_adapter_sha:
        raise ValueError("spec.model.adapter_sha256 does not match adapter weights")
    if spec_model.get("base_model_id") != manifest.get("base_model_id"):
        raise ValueError("spec.model.base_model_id does not match train_manifest.json")
    if spec_model.get("revision") != manifest.get("revision"):
        raise ValueError("spec.model.revision does not match train_manifest.json")

    if not manifest.get("base_weights_sha256") or manifest.get("base_weights_sha256") != base_weights_sha:
        raise ValueError("base model weights SHA-256 mismatch against train_manifest.json")

    manifest_spec_sha = manifest.get("spec_sha256")
    if not manifest_spec_sha:
        raise ValueError("train_manifest.json missing spec_sha256")
    pre_finalized_spec = json.loads(json.dumps(spec))
    pre_finalized_spec["model"]["adapter_sha256"] = "0" * 64
    pre_finalized_bytes = (json.dumps(pre_finalized_spec, indent=2, sort_keys=True) + "\n").encode("utf-8")
    spec_matches = (
        manifest_spec_sha == spec_sha
        or (manifest.get("spec_core_sha256") is not None and manifest.get("spec_core_sha256") == spec_core_sha)
        or (manifest_spec_sha == sha_bytes(pre_finalized_bytes))
    )
    if not spec_matches:
        raise ValueError("spec changed since training (train_manifest spec_sha256 mismatch); refusing to predict")

    return {
        "dataset_sha256": dataset_sha,
        "spec_sha256": spec_sha,
        "base_weights_sha256": base_weights_sha,
        "adapter_sha256": actual_adapter_sha,
        "manifest": manifest,
    }


def _normalize_prompt_ids(raw_ids: Any) -> list[int]:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--adapter-dir", default=None, help="trained adapter directory (required unless --base-only)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--constrained", action="store_true", help="restrict generation to spec label prefix-trie")
    parser.add_argument("--base-only", action="store_true", help="score zero-shot base model without an adapter")
    parser.add_argument("--device", default="cpu", choices=SUPPORTED_DEVICES)
    parser.add_argument("--model-registry", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.batch_size <= 0 or args.max_new_tokens <= 0 or args.threads <= 0:
        raise SystemExit("--batch-size, --max-new-tokens, and --threads must be positive")
    if not args.base_only and not args.adapter_dir:
        raise SystemExit("--adapter-dir is required unless --base-only is set")

    task_dir = Path(args.task_dir).expanduser().resolve()
    adapter_dir = Path(args.adapter_dir).expanduser().resolve() if args.adapter_dir else None
    hf_home = Path(args.hf_home).expanduser().resolve()

    spec_path = task_dir / "spec.json"
    data_path = task_dir / "data.jsonl"
    if not spec_path.is_file() or not data_path.is_file():
        raise SystemExit(f"missing spec.json or data.jsonl in {task_dir}")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec_errors = validate_spec(spec)
    if spec_errors:
        raise SystemExit(f"invalid task spec: {'; '.join(spec_errors)}")

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
        prov = verify_prediction_provenance(
            task_dir=task_dir,
            spec=spec,
            snap_info=snap_info,
            adapter_dir=adapter_dir,
            base_only=args.base_only,
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    set_offline_env(hf_home, threads=args.threads, allow_gpu=(args.device != "cpu"))
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_spec = select_device(args.device, torch)
    torch.set_num_threads(args.threads)

    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    test_rows = split_task_dataset(rows, spec)["test"]
    allowlist = list(spec["labels"])
    label_menu = ", ".join(allowlist)

    dtype = getattr(torch, device_spec.torch_dtype_name, torch.float32)
    tokenizer, model = load_verified_snapshot(
        snap,
        AutoTokenizer,
        AutoModelForCausalLM,
        torch,
        device=device_spec.device,
        dtype=dtype,
    )
    if not args.base_only and adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False, local_files_only=True)
    model.eval()

    prompt_seqs = [
        _normalize_prompt_ids(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": PROMPT_SYS},
                    {"role": "user", "content": f"Codes: {label_menu}\nNote: {row['prompt']}\nCode:"},
                ],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        for row in test_rows
    ]

    trie = LabelPrefixTrie(allowlist, tokenizer) if args.constrained else None
    t0 = time.time()
    gen_results = batched_greedy_generate(
        model,
        tokenizer,
        prompt_seqs,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        label_trie=trie,
        return_confidence=True,
        torch_module=torch,
        device=device_spec.device,
    )

    predictions: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    exact = 0
    normalized = 0
    undetermined = 0
    latencies: list[float] = []
    for row, res in zip(test_rows, gen_results, strict=True):
        label, match_kind = to_label(res.text, allowlist)
        if match_kind == "exact":
            exact += 1
        elif match_kind == "normalized":
            normalized += 1
        else:
            undetermined += 1
        latencies.append(res.latency_ms)

        pred_rec: dict[str, Any] = {
            "id": row["id"],
            "label": label,
            "latency_ms": res.latency_ms,
            "raw_invalid": match_kind == "undetermined",
        }
        if res.confidence is not None:
            pred_rec["confidence"] = round(res.confidence, 6)
        predictions.append(pred_rec)

        case_summaries.append(
            {
                "id": row["id"],
                "label": label,
                "match_type": match_kind,
                "confidence": pred_rec.get("confidence"),
                "latency_ms": res.latency_ms,
                "max_token_limit_hit": res.max_token_limit_hit,
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for pred in predictions:
            fh.write(json.dumps(pred, sort_keys=True) + "\n")

    p95 = percentile(latencies, 95.0)
    mean_lat = (sum(latencies) / len(latencies)) if latencies else None
    predict_manifest = {
        "test_records": len(test_rows),
        "exact_allowlist_outputs": exact,
        "normalized_outputs": normalized,
        "mapped_to_undetermined": undetermined,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "decoding": "greedy",
        "constrained": bool(args.constrained),
        "base_only": bool(args.base_only),
        "device": device_spec.device,
        "adapter_sha256": prov["adapter_sha256"],
        "base_weights_sha256": prov["base_weights_sha256"],
        "spec_sha256": prov["spec_sha256"],
        "dataset_sha256": prov["dataset_sha256"],
        "duration_seconds": round(time.time() - t0, 2),
        "mean_latency_ms": round(mean_lat, 3) if mean_lat is not None else None,
        "latency_p95_ms": round(p95, 3) if p95 is not None else None,
        "threads": args.threads,
        "cases": case_summaries,
    }
    (out_path.parent / "predict_manifest.json").write_text(
        json.dumps(predict_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in predict_manifest.items() if k != "cases"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
