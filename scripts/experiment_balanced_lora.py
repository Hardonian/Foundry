"""Bounded synthetic LoRA experiment with predeclared, group-disjoint test cases.

Experimental CPU-only research. No customer data, promotion, or deployment.
"""
import argparse
import fcntl
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

from evaluate_synthetic_lora import metrics
from train_synthetic_lora import (
    BLOBS,
    MODEL,
    MODEL_SHA256,
    REVISION,
    load_verified_snapshot,
    sha_file,
)

SEED = 20261110
STEPS = 96


def build_datasets():
    """Matched/mismatch pairs share a base number; base groups never cross splits."""
    templates = {
        "train": "Expected minor units: {expected}. Actual minor units: {actual}. Output matched or amount_mismatch only.",
        "validation": "Compare two synthetic amounts. Expected={expected}; actual={actual}. Return matched or amount_mismatch.",
        "test": "Synthetic reconciliation: expected {expected}, observed {actual}. Reply matched or amount_mismatch only.",
    }
    split = {}
    for name, count, start in (("train", 24, 1100), ("validation", 8, 4100), ("test", 12, 7100)):
        rows = []
        for group in range(count):
            expected = start + group * 31
            delta = (1, -1, 5, -7)[group % 4]
            for variant, actual in (("matched", expected), ("amount_mismatch", expected + delta)):
                rows.append({"id": f"synthetic-{name}-{group:02d}-{variant}", "group": expected,
                             "prompt": templates[name].format(expected=expected, actual=actual), "answer": variant})
        random.Random(SEED + start).shuffle(rows)
        split[name] = rows
    groups = [{r["group"] for r in rows} for rows in split.values()]
    prompts = [{r["prompt"] for r in rows} for rows in split.values()]
    if any(groups[i] & groups[j] or prompts[i] & prompts[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("split leakage")
    return split


def digest_json(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def preflight(cache, root):
    if not root.is_dir() or root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
        raise ValueError("output root must be an owner-only directory")
    space = shutil.disk_usage(root)
    if space.free < 10 * 1024**3 or space.used / space.total >= 0.8:
        raise ValueError("disk watermark exceeded")
    if os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") < 8 * 1024**3:
        raise ValueError("not enough free physical RAM")
    snap = cache / "hub" / "models--Qwen--Qwen2.5-Coder-0.5B-Instruct" / "snapshots" / REVISION
    if any(not (snap / name).is_file() for name in ("model.safetensors", *BLOBS)):
        raise ValueError("required pinned model files are missing; no download attempted")
    if sha_file(snap / "model.safetensors") != MODEL_SHA256:
        raise ValueError("model hash mismatch")
    for name, expected in BLOBS.items():
        content = (snap / name).read_bytes()
        if hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest() != expected:
            raise ValueError(f"model metadata hash mismatch: {name}")
    config = json.loads((snap / "config.json").read_text())
    if config.get("auto_map"):
        raise ValueError("remote code is not allowed")
    return snap


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    root = args.output_root.resolve(strict=True)
    cache = args.hf_home.resolve(strict=True)
    try:
        snap = preflight(cache, root)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"preflight failed: {exc}") from exc
    fd = os.open(root / ".training.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another participating training job holds the lock") from exc
        run_experiment(args, root, cache, snap)


def run_experiment(args, root, cache, snap):
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HOME=str(cache), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", OMP_NUM_THREADS=str(args.threads),
                      MKL_NUM_THREADS=str(args.threads), TOKENIZERS_PARALLELISM="false")
    import peft
    import torch
    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    started = time.monotonic()
    output = root / (time.strftime("synthetic-balanced-%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}")
    output.mkdir(mode=0o700, exist_ok=False)
    datasets = build_datasets()
    prereg = {"data_origin": "synthetic", "model": MODEL, "revision": REVISION, "seed": SEED,
              "steps": STEPS, "train_count": len(datasets["train"]), "validation_count": len(datasets["validation"]),
              "test_count": len(datasets["test"]), "split_sha256": {k: digest_json(v) for k, v in datasets.items()},
              "rule": "fixed 96 steps; no checkpoint selection; test evaluated after training",
              "decoding": {"do_sample": False, "max_new_tokens": 8}}
    (output / "preregistration.json").write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    (output / "synthetic-dataset.json").write_text(json.dumps(datasets, sort_keys=True, separators=(",", ":")))
    tokenizer, model = load_verified_snapshot(snap, AutoTokenizer, AutoModelForCausalLM, torch)

    def encode(row):
        messages = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["answer"]}]
        prompt = tokenizer.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if full[:len(prompt)] != prompt or not len(prompt) < len(full) <= 256:
            raise RuntimeError("unsafe tokenization")
        return torch.tensor([full]), torch.tensor([[-100] * len(prompt) + full[len(prompt):]])

    train = [encode(row) for row in datasets["train"]]
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"))
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0002)
    train_losses = []
    model.train()
    for step in range(STEPS):
        ids, labels = train[step % len(train)]
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=labels).loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("non-finite gradient")
        optimizer.step()
        train_losses.append(float(loss.detach()))
        if (step + 1) % 24 == 0:
            print(f"step={step+1} loss={train_losses[-1]:.6f}", flush=True)
    if any(not torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
        raise RuntimeError("non-finite adapter weights")
    adapter = output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    adapter_config_sha256 = sha_file(adapter / "adapter_config.json")
    model.eval()
    gen_config = GenerationConfig(do_sample=False, max_new_tokens=8, pad_token_id=tokenizer.eos_token_id,
                                  eos_token_id=tokenizer.eos_token_id)

    def evaluate(m, rows):
        out = []
        for row in rows:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt")
            with torch.no_grad():
                generated = m.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), generation_config=gen_config)
            tokens = generated[0, prompt.shape[-1]:]
            output_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            out.append({"id": row["id"], "expected": row["answer"], "output": output_text,
                        "max_token_limit_hit": len(tokens) == 8 and int(tokens[-1]) != tokenizer.eos_token_id})
        return out

    # Only after the fixed training budget is complete do we score the held-out test.
    with model.disable_adapter():
        base = evaluate(model, datasets["test"])
    adapted = evaluate(model, datasets["test"])
    base_model = model.unload()
    if sha_file(adapter / "adapter_config.json") != adapter_config_sha256:
        raise RuntimeError("adapter configuration changed before reload")
    reloaded = PeftModel.from_pretrained(base_model, adapter, is_trainable=False, local_files_only=True)
    reload_results = evaluate(reloaded, datasets["test"])
    if adapted != reload_results:
        raise RuntimeError("reloaded adapter generation differs")
    result = {"status": "completed", "data_origin": "synthetic", "model": MODEL, "revision": REVISION,
              "base_model_sha256": MODEL_SHA256, "script_sha256": sha_file(Path(__file__)),
              "preregistration_sha256": sha_file(output / "preregistration.json"),
              "dataset_sha256": sha_file(output / "synthetic-dataset.json"),
              "adapter_sha256": sha_file(adapter / "adapter_model.safetensors"),
              "adapter_config_sha256": adapter_config_sha256,
              "seed": SEED, "steps": STEPS, "threads": args.threads, "trainable_parameters": params,
              "train_losses": train_losses, "base": {"metrics": metrics(base), "cases": base},
              "adapter": {"metrics": metrics(adapted), "cases": adapted}, "reload_exact_match": True,
              "regressions": sum(b["output"] == b["expected"] and a["output"] != a["expected"] for b, a in zip(base, adapted)),
              "improvements": sum(b["output"] != b["expected"] and a["output"] == a["expected"] for b, a in zip(base, adapted)),
              "duration_seconds": round(time.monotonic() - started, 2),
              "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
              "limitations": ["generated synthetic records and one model", "numeric range and wording differ across splits", "no pretraining-contamination guarantee", "test observed after run; no longer pristine", "no promotion or real-world accuracy claim"]}
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {k: result[k] for k in ("status", "model", "revision", "steps", "trainable_parameters", "adapter_sha256", "dataset_sha256", "regressions", "improvements", "reload_exact_match", "duration_seconds")}
    summary.update(run_dir=str(output), base_metrics=metrics(base), adapter_metrics=metrics(adapted))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
