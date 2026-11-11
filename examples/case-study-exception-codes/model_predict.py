#!/usr/bin/env python3
"""Score the frozen test split with the fine-tuned adapter (greedy decoding).

Offline, CPU-only. Maps any off-allowlist generation to UNDETERMINED (the
honest, safe fallback: a garbage prediction must never masquerade as a code).
Writes model.jsonl plus predict_manifest.json with decode statistics.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

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
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    adapter_dir = Path(args.adapter_dir)
    hf_home = Path(args.hf_home).expanduser().resolve()
    snap = hf_home / "hub" / "models--Qwen--Qwen2.5-Coder-0.5B-Instruct" / "snapshots" / REVISION
    manifest = json.loads((adapter_dir / "train_manifest.json").read_text(encoding="utf-8"))
    dataset_sha = sha256_file(task_dir / "data.jsonl")
    if manifest["dataset_sha256"] != dataset_sha:
        raise SystemExit("dataset changed since training; refusing to predict")
    if manifest["adapter_sha256"] != sha256_file(adapter_dir / "adapter_model.safetensors"):
        raise SystemExit("adapter weights changed since training; refusing to predict")

    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HOME=str(hf_home), HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")  # nosec B106 - env opt-out flags, not credentials
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    spec = json.loads((task_dir / "spec.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in
            (task_dir / "data.jsonl").read_text(encoding="utf-8").splitlines() if line]
    from hardonian_foundry.task_eval import split_task_dataset
    test_rows = split_task_dataset(rows, spec)["test"]
    allowlist = list(spec["labels"])  # UNDETERMINED is a legitimate class
    label_menu = ", ".join(allowlist)

    tokenizer = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)  # nosec B615 - local pinned snapshot dir, no hub download
    base = AutoModelForCausalLM.from_pretrained(str(snap), local_files_only=True,  # nosec B615 - local pinned snapshot dir, no hub download
                                                torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    def to_label(text):
        cleaned = text.strip().splitlines()[0].strip() if text.strip() else ""
        cleaned = cleaned.strip(".\"'` ")
        if cleaned in allowlist:
            return cleaned, True
        upper = cleaned.upper().replace(" ", "_")
        if upper in allowlist:
            return upper, True
        for label in allowlist:  # tolerate trailing prose: first allowlist token wins
            if upper.startswith(label):
                return label, True
        return "UNDETERMINED", False

    predictions, exact, invalid, t0 = [], 0, 0, time.time()
    with torch.no_grad():
        for i, row in enumerate(test_rows):
            messages = [
                {"role": "system", "content": PROMPT_SYS},
                {"role": "user", "content": f"Codes: {label_menu}\nNote: {row['prompt']}\nCode:"},
            ]
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                      return_tensors="pt")
            attention_mask = torch.ones_like(input_ids)
            out = model.generate(input_ids, attention_mask=attention_mask,
                                 max_new_tokens=args.max_new_tokens,
                                 do_sample=False, num_beams=1,
                                 pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
            label, ok = to_label(text)
            exact += int(ok)
            invalid += int(not ok)
            predictions.append({"id": row["id"], "label": label})
            if (i + 1) % 50 == 0:
                print(f"predicted {i + 1}/{len(test_rows)} elapsed {time.time() - t0:.0f}s", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for pred in predictions:
            fh.write(json.dumps(pred, sort_keys=True) + "\n")
    predict_manifest = {
        "test_records": len(test_rows),
        "exact_allowlist_outputs": exact,
        "mapped_to_undetermined": invalid,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "adapter_sha256": manifest["adapter_sha256"],
        "dataset_sha256": dataset_sha,
        "duration_seconds": round(time.time() - t0, 1),
        "threads": args.threads,
    }
    (out_path.parent / "predict_manifest.json").write_text(
        json.dumps(predict_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(predict_manifest, indent=2))


if __name__ == "__main__":
    sys.exit(main())
