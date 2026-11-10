"""Bounded offline CPU LoRA smoke; synthetic data, no production promotion.

Requires an existing compatible CPU-only PyTorch/Transformers/PEFT environment
and the pinned model in a local Hugging Face cache. Never downloads models.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
REVISION = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
SEED = 1729
MODEL_SHA256 = "f9523886352217ded3aeeef552b381af79d568c6d49a4b9e423288cea56b0a44"
# Git blob IDs published by the model repository for this immutable revision.
BLOBS = {"config.json": "e2aa18293b0dd66539341467fa454878a5adb2be",
         "generation_config.json": "c28f9c697cbc09f047434efec57556505af31111",
         "merges.txt": "20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0",
         "tokenizer.json": (
             "443909a61d429dff23010e5bddd28ff530edda00"
         ),
         "tokenizer_config.json": (
             "acee076f49bf3c0298e15de0909d1da7b392f0c3"
         ),
         "vocab.json": "4783fe10ac3adce15ac8f358ef5462739852c569"}


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def examples():
    # Hand-generated records are deliberately synthetic, not financial records.
    def item(i):
        expected = 1000 + i * 37
        actual = expected + (5 if i % 2 else 0)
        answer = "amount_mismatch" if actual != expected else "matched"
        prompt = f"Synthetic example {i}. Expected minor units: {expected}. Actual minor units: {actual}. Classify as matched or amount_mismatch; return the label only."
        return {"id": f"synthetic-{i:03d}", "prompt": prompt, "answer": answer}
    train = [item(i) for i in range(12)]
    holdout = [item(i) for i in range(100, 104)]
    assert not ({x["prompt"] for x in train} & {x["prompt"] for x in holdout})
    return {"train": train, "holdout": holdout}


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", type=Path, required=True, help="existing HF_HOME containing the pinned cached model")
    parser.add_argument("--output-root", type=Path, required=True, help="existing private directory on Linux storage")
    parser.add_argument("--steps", type=int, default=4, choices=range(1, 21), metavar="1..20")
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 5), metavar="1..4")
    args = parser.parse_args()
    cache = args.hf_home.expanduser().resolve(strict=True)
    root = args.output_root.expanduser().resolve(strict=True)
    if not root.is_dir() or root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
        raise SystemExit("output root must be an existing owner-only directory")
    snap = cache / "hub" / "models--Qwen--Qwen2.5-Coder-0.5B-Instruct" / "snapshots" / REVISION
    if not (snap / "model.safetensors").is_file() or not (snap / "config.json").is_file():
        raise SystemExit("pinned safetensors model/config unavailable in cache; no download attempted")
    if sha_file(snap / "model.safetensors") != MODEL_SHA256:
        raise SystemExit("cached model SHA-256 mismatch")
    for filename, expected in BLOBS.items():
        payload = (snap / filename).read_bytes()
        git_blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        if git_blob != expected:
            raise SystemExit(f"cached model metadata mismatch: {filename}")
    config = json.loads((snap / "config.json").read_text())
    if config.get("auto_map"):
        raise SystemExit("remote custom model code is not permitted")
    free = shutil.disk_usage(root)
    if free.free < 10 * 1024**3 or free.used / free.total >= 0.80:
        raise SystemExit("output filesystem below 10 GiB free or at least 80% utilized")
    if os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") < 8 * 1024**3:
        raise SystemExit("insufficient available physical memory (minimum 8 GiB)")
    lock_path = root / ".training.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another participating Foundry training job holds the lock") from exc
        run(args, cache, root, snap)


def load_verified_snapshot(snap, tokenizer_cls, model_cls, torch_module):
    """Load only the preflight-verified snapshot, independent of Hub cache overrides."""
    tokenizer = tokenizer_cls.from_pretrained(str(snap), local_files_only=True, trust_remote_code=False, token=False)
    model = model_cls.from_pretrained(str(snap), local_files_only=True, trust_remote_code=False, token=False,
                                      torch_dtype=torch_module.float32, attn_implementation="eager").to("cpu")
    return tokenizer, model


def run(args, cache, root, snap):
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HOME=str(cache), HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      HF_HUB_DISABLE_IMPLICIT_TOKEN="1", OMP_NUM_THREADS=str(args.threads),
                      MKL_NUM_THREADS=str(args.threads), TOKENIZERS_PARALLELISM="false")
    import peft
    import torch
    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    random.seed(SEED)
    started = time.monotonic()
    run_id = time.strftime("synthetic-lora-%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
    output = root / run_id
    output.mkdir(mode=0o700, exist_ok=False)
    data = examples()
    data_bytes = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    (output / "synthetic-dataset.json").write_bytes(data_bytes)
    tokenizer, model = load_verified_snapshot(snap, AutoTokenizer, AutoModelForCausalLM, torch)

    def encode(row):
        messages = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["answer"]}]
        prompt = tokenizer.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if full[:len(prompt)] != prompt or not len(prompt) < len(full) <= 256:
            raise RuntimeError("chat template mismatch or token limit exceeded")
        return torch.tensor([full]), torch.tensor([[-100] * len(prompt) + full[len(prompt):]])

    train = [encode(row) for row in data["train"]]
    holdout = [encode(row) for row in data["holdout"]]

    def losses(m):
        m.eval()
        with torch.no_grad():
            values = [float(m(input_ids=ids, labels=labels).loss) for ids, labels in holdout]
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError("non-finite evaluation loss")
        return values

    base_loss = losses(model)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0002)
    training_losses = []
    model.train()
    for step in range(args.steps):
        ids, labels = train[step % len(train)]
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=labels).loss
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
    # Reload into the same base, without duplicate model allocation.
    base_model = model.unload()
    reloaded = PeftModel.from_pretrained(base_model, adapter, is_trainable=False, local_files_only=True)
    reloaded_loss = losses(reloaded)
    delta = max(abs(a - b) for a, b in zip(trained_loss, reloaded_loss))
    if delta > 1e-6:
        raise RuntimeError(f"offline adapter reload mismatch: {delta}")
    result = {"status": "completed", "data_origin": "synthetic", "model": MODEL, "revision": REVISION,
              "model_license": "Apache-2.0 (review upstream license before distribution)",
              "base_model_sha256": sha_file(snap / "model.safetensors"), "dataset_sha256": sha_bytes(data_bytes),
              "adapter_sha256": sha_file(adapter / "adapter_model.safetensors"),
              "adapter_config_sha256": sha_file(adapter / "adapter_config.json"),
              "script_sha256": sha_file(Path(__file__)), "device": "cpu", "seed": SEED,
              "steps": args.steps, "threads": args.threads, "trainable_parameters": trainable,
              "train_count": len(train), "holdout_count": len(holdout), "train_losses": training_losses,
              "base_holdout_loss": base_loss, "adapter_holdout_loss": trained_loss,
              "reload_holdout_loss": reloaded_loss, "max_reload_delta": delta,
              "duration_seconds": round(time.monotonic() - started, 2),
              "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
              "limitations": ["synthetic smoke, teacher-forced loss only", "not a production or quality benchmark", "not approved for promotion", "no checkpoint resume"]}
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run_dir": str(output), **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
