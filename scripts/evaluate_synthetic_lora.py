"""Compare base and adapter generation on the existing synthetic holdout.

This is an in-domain smoke, not a pristine or commercially valid benchmark.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from train_synthetic_lora import (
    BLOBS,
    MODEL,
    MODEL_SHA256,
    REVISION,
    examples,
    load_verified_snapshot,
    sha_file,
)


def metrics(rows):
    tp = sum(r["expected"] == "amount_mismatch" and r["output"] == "amount_mismatch" for r in rows)
    fp = sum(r["expected"] == "matched" and r["output"] == "amount_mismatch" for r in rows)
    fn = sum(r["expected"] == "amount_mismatch" and r["output"] != "amount_mismatch" for r in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"correct": sum(r["output"] == r["expected"] for r in rows), "total": len(rows),
            "precision": precision, "recall": recall, "f1": f1,
            "false_negative_rate": fn / (tp + fn) if tp + fn else None,
            "invalid_outputs": sum(r["output"] not in ("matched", "amount_mismatch") for r in rows),
            "max_token_limit_hits": sum(r["max_token_limit_hit"] for r in rows)}


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    root = args.output_root.resolve(strict=True)
    run = args.run_dir.resolve(strict=True)
    cache = args.hf_home.resolve(strict=True)
    for p in (root, run):
        if not p.is_dir() or p.stat().st_uid != os.getuid() or p.stat().st_mode & 0o077:
            raise SystemExit("run and output directories must be owner-only")
    disk = shutil.disk_usage(root)
    if disk.free < 10 * 1024**3 or disk.used / disk.total >= 0.80:
        raise SystemExit("output filesystem below 10 GiB free or at least 80% utilized")
    if os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") < 8 * 1024**3:
        raise SystemExit("less than 8 GiB free physical memory")
    snap = cache / "hub" / "models--Qwen--Qwen2.5-Coder-0.5B-Instruct" / "snapshots" / REVISION
    if sha_file(snap / "model.safetensors") != MODEL_SHA256:
        raise SystemExit("base model hash mismatch")
    for name, expected in BLOBS.items():
        payload = (snap / name).read_bytes()
        if hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest() != expected:
            raise SystemExit(f"model metadata mismatch: {name}")
    data = (run / "synthetic-dataset.json").read_bytes()
    expected_data = json.dumps(examples(), sort_keys=True, separators=(",", ":")).encode()
    if data != expected_data:
        raise SystemExit("only the built-in synthetic dataset is supported")
    manifest = json.loads((run / "result.json").read_text())
    adapter = run / "adapter"
    if (manifest.get("status") != "completed" or manifest.get("model") != MODEL or
        manifest.get("revision") != REVISION or manifest.get("dataset_sha256") != hashlib.sha256(data).hexdigest() or
        manifest.get("adapter_sha256") != sha_file(adapter / "adapter_model.safetensors")):
        raise SystemExit("training run manifest mismatch")
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    if (cfg.get("peft_type") != "LORA" or cfg.get("task_type") != "CAUSAL_LM" or cfg.get("r") != 4 or
        cfg.get("lora_alpha") != 8 or set(cfg.get("target_modules", [])) != {"q_proj", "v_proj"} or
        cfg.get("auto_mapping") is not None or cfg.get("modules_to_save") is not None):
        raise SystemExit("unsupported adapter configuration")
    lock_file = root / ".training.lock"
    fd = os.open(lock_file, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another participating Foundry training job holds the lock") from exc
        run_eval(args, root, cache, snap, adapter, data, manifest)


def run_eval(args, root, cache, snap, adapter, data, manifest):
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HOME=str(cache), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                      OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads), TOKENIZERS_PARALLELISM="false")
    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1729)
    started = time.monotonic()
    output = root / (time.strftime("synthetic-generation-eval-%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}")
    output.mkdir(mode=0o700, exist_ok=False)
    tokenizer, model = load_verified_snapshot(snap, AutoTokenizer, AutoModelForCausalLM, torch)
    holdout = json.loads(data)["holdout"]
    generation_config = GenerationConfig(do_sample=False, max_new_tokens=8,
                                         pad_token_id=tokenizer.eos_token_id,
                                         eos_token_id=tokenizer.eos_token_id)

    def generate(m):
        m.eval()
        results = []
        for row in holdout:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt")
            with torch.no_grad():
                generated = m.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                       generation_config=generation_config)
            tokens = generated[0, prompt.shape[-1]:]
            answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            hit = len(tokens) == 8 and int(tokens[-1]) != tokenizer.eos_token_id
            results.append({"id": row["id"], "expected": row["answer"], "output": answer,
                            "max_token_limit_hit": hit})
        return results

    base = generate(model)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
    adapted = generate(model)
    result = {"status": "completed", "data_origin": "synthetic", "model": MODEL, "revision": REVISION,
              "base_model_sha256": MODEL_SHA256, "adapter_sha256": manifest["adapter_sha256"],
              "dataset_sha256": manifest["dataset_sha256"], "training_script_sha256": manifest["script_sha256"],
              "evaluation_script_sha256": sha_file(Path(__file__)), "decoding": {"do_sample": False, "max_new_tokens": 8},
              "base": {"metrics": metrics(base), "cases": base},
              "adapter": {"metrics": metrics(adapted), "cases": adapted},
              "regressions": sum(b["output"] == b["expected"] and a["output"] != a["expected"] for b, a in zip(base, adapted)),
              "improvements": sum(b["output"] != b["expected"] and a["output"] == a["expected"] for b, a in zip(base, adapted)),
              "duration_seconds": round(time.monotonic() - started, 2),
              "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
              "limitations": ["four authored synthetic examples", "holdout already used for teacher-forced evaluation; not pristine", "not customer data or commercial validation", "no promotion"]}
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"evaluation_dir": str(output), **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
