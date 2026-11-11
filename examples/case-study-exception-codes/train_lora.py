#!/usr/bin/env python3
"""LoRA fine-tune the pinned Qwen2.5-Coder-0.5B-Instruct snapshot on the frozen
train split of the exception-resolution-code task. Offline, CPU-only, synthetic.

The base snapshot must already exist under --hf-home in the hub cache layout at
the pinned revision. Nothing is downloaded; Hub networking and implicit
credentials are disabled. Writes an adapter directory (mode 0700) plus a
train_manifest.json recording every input hash needed for provenance.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
REVISION = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
PROMPT_SYS = ("You classify reconciliation exception notes into exactly one resolution code. "
              "Reply with only the code, nothing else.")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--task-dir", required=True, help="directory containing spec.json and data.jsonl")
    parser.add_argument("--output", required=True, help="adapter output directory (must not exist)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq", type=int, default=160)
    args = parser.parse_args()

    out_dir = Path(args.output)
    if out_dir.exists():
        raise SystemExit(f"output directory already exists: {out_dir}")

    hf_home = Path(args.hf_home).expanduser().resolve()
    snap = hf_home / "hub" / "models--Qwen--Qwen2.5-Coder-0.5B-Instruct" / "snapshots" / REVISION
    weights = snap / "model.safetensors"
    if not weights.is_file():
        raise SystemExit(f"pinned snapshot missing: {snap}")

    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HOME=str(hf_home), HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")  # nosec B106 - env opt-out flags, not credentials
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.manual_seed(20261110)

    task_dir = Path(args.task_dir)
    spec = json.loads((task_dir / "spec.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in
            (task_dir / "data.jsonl").read_text(encoding="utf-8").splitlines() if line]
    from hardonian_foundry.task_eval import split_task_dataset
    train_rows = split_task_dataset(rows, spec)["train"]
    labels = list(spec["labels"])  # UNDETERMINED is a legitimate gold/pred class
    label_menu = ", ".join(labels)

    tokenizer = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)  # nosec B615 - local pinned snapshot dir, no hub download
    model = AutoModelForCausalLM.from_pretrained(str(snap), local_files_only=True,  # nosec B615 - local pinned snapshot dir, no hub download
                                                 torch_dtype=torch.float32)
    peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                             task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable}", flush=True)

    def encode(row):
        messages = [
            {"role": "system", "content": PROMPT_SYS},
            {"role": "user", "content": f"Codes: {label_menu}\nNote: {row['prompt']}\nCode:"},
        ]
        prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        full_ids = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": row["gold_label"]}],
            add_generation_prompt=False)
        prompt_ids = prompt_ids[:args.max_seq]
        full_ids = full_ids[:args.max_seq]
        if len(full_ids) <= len(prompt_ids):
            return None
        labels_ids = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        return {"input_ids": full_ids, "labels": labels_ids}

    encoded = [item for item in (encode(row) for row in train_rows) if item is not None]
    if len(encoded) < len(train_rows):
        print(f"warning: {len(train_rows) - len(encoded)} rows truncated away by max-seq", flush=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    step, losses, t0 = 0, [], time.time()
    gen = torch.Generator().manual_seed(20261110)
    for epoch in range(args.epochs):
        order = torch.randperm(len(encoded), generator=gen).tolist()
        optimizer.zero_grad()
        for i, idx in enumerate(order):
            item = encoded[idx]
            input_ids = torch.tensor([item["input_ids"]])
            labels_t = torch.tensor([item["labels"]])
            loss = model(input_ids=input_ids, labels=labels_t).loss / args.grad_accum
            loss.backward()
            losses.append(float(loss) * args.grad_accum)
            if (i + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step % 25 == 0:
                    recent = sum(losses[-100:]) / min(100, len(losses[-100:]))
                    print(f"epoch {epoch} step {step} loss {recent:.4f} "
                          f"elapsed {time.time() - t0:.0f}s", flush=True)
    optimizer.step()

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    os.makedirs(out_dir, mode=0o700)
    model.save_pretrained(str(out_dir))
    duration = time.time() - t0
    adapter_file = out_dir / "adapter_model.safetensors"
    manifest = {
        "status": "completed",
        "data_origin": "synthetic",
        "base_model_id": MODEL_ID,
        "revision": REVISION,
        "base_weights_sha256": sha256_file(weights),
        "adapter_sha256": sha256_file(adapter_file),
        "dataset_sha256": sha256_file(task_dir / "data.jsonl"),
        "spec_sha256": sha256_file(task_dir / "spec.json"),
        "train_records": len(encoded),
        "epochs": args.epochs,
        "lr": args.lr,
        "grad_accum": args.grad_accum,
        "max_seq": args.max_seq,
        "optimizer_steps": step,
        "final_train_loss_mean100": round(sum(losses[-100:]) / min(100, len(losses[-100:])), 6),
        "seed": 20261110,
        "threads": args.threads,
        "duration_seconds": round(duration, 1),
        "lora": {"r": 8, "alpha": 16, "dropout": 0.0,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "trainable_params": trainable,
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "host": platform.node(),
        "limitations": ("Teacher-forced LoRA on synthetic in-style notes; train loss is not a "
                        "generalization metric. Test scoring happens only via model_predict.py "
                        "through the Foundry task gate."),
    }
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                 encoding="utf-8")
    os.chmod(out_dir / "train_manifest.json", 0o600)
    print(json.dumps({"status": "completed", "adapter_dir": str(out_dir),
                      "adapter_sha256": manifest["adapter_sha256"],
                      "duration_seconds": manifest["duration_seconds"],
                      "final_train_loss_mean100": manifest["final_train_loss_mean100"]}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
