"""Synthetic out-of-template challenge for a frozen balanced LoRA adapter.

Never train or select checkpoints on this challenge. No promotion or production use.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from hardonian_foundry import synthetic_policy

try:
    import fcntl
except ImportError:
    fcntl = None

from evaluate_synthetic_lora import verify_adapter_hashes
from experiment_balanced_lora import preflight
from train_synthetic_lora import (
    MODEL,
    MODEL_SHA256,
    REVISION,
    load_verified_snapshot,
    sha_file,
)


def oracle(expected, actual, expected_ccy, actual_ccy):
    if expected is None or actual is None or expected_ccy is None or actual_ccy is None or expected_ccy != actual_ccy:
        return "review_required"
    return "matched" if expected == actual else "amount_mismatch"


def challenge_cases():
    # Only fictional minor-unit values. Values and templates differ from prior training/test.
    specs = [
        (-24000,-24000,"USD","USD"), (-120000,-120001,"USD","USD"),
        (-88000,-87999,"USD","USD"), (-35000,-35000,"EUR","EUR"),
        (0,0,"USD","USD"), (0,-1,"USD","USD"), (0,1,"EUR","EUR"),
        (999999,999999,"USD","USD"), (1000000,1000001,"USD","USD"),
        (9999999,9999998,"EUR","EUR"), (90000000,90000000,"USD","USD"),
        (75000000,75000005,"EUR","EUR"), (43210,43210,"USD","USD"),
        (54321,54320,"EUR","EUR"), (88888,88888,"EUR","EUR"),
        (-1,-1,"USD","USD"),
        (None,42000,"USD","USD"), (42000,None,"USD","USD"),
        (None,None,"EUR","EUR"), (125000,125000,"USD","EUR"),
        (48000,48001,"EUR","USD"), (0,0,None,"USD"),
        (-24000,-24000,"EUR",None), (99999,None,"USD","EUR"),
    ]
    templates = [
        "Synthetic check: expected {e} {ec}; actual {a} {ac}.",
        "Fictional ledger: actual={a} {ac} versus expected={e} {ec}.",
        "Compare expected ({ec} {e}) with observed ({ac} {a}) for this made-up record.",
    ]
    instructions = (" If either amount or either currency is missing, or currencies differ, return review_required. "
                    "Otherwise return matched if the integer minor-unit amounts are equal, or amount_mismatch if different. "
                    "Return exactly one label: matched, amount_mismatch, or review_required.")
    rows = []
    for i, (expected, actual, ec, ac) in enumerate(specs):
        # Comma separators test formatting without changing numerical meaning.
        fmt = lambda value: "missing" if value is None else format(value, ",d")
        prompt = templates[i % len(templates)].format(e=fmt(expected), a=fmt(actual), ec=ec or "missing", ac=ac or "missing") + instructions
        rows.append({"id": f"synthetic-challenge-{i:03d}", "expected_minor": expected, "actual_minor": actual,
                     "expected_currency": ec, "actual_currency": ac,
                     "expected_label": oracle(expected, actual, ec, ac), "prompt": prompt})
    return rows


def summarize(rows):
    labels = ("matched", "amount_mismatch", "review_required")
    by_label = {}
    for label in labels:
        group = [r for r in rows if r["expected"] == label]
        by_label[label] = {"total": len(group), "correct": sum(r["output"] == label for r in group),
                           "normalized_correct": sum(r["output"].lower() == label for r in group)}
    return {"total": len(rows), "strict_correct": sum(r["output"] == r["expected"] for r in rows),
            "normalized_correct": sum(r["output"].lower() == r["expected"] for r in rows),
            "invalid_outputs": sum(r["output"] not in labels for r in rows),
            "unsafe_matched": sum(r["expected"] != "matched" and r["output"].lower() == "matched" for r in rows),
            "token_limit_hits": sum(r["max_token_limit_hit"] for r in rows), "by_expected": by_label}


def deterministic_baseline(challenge):
    """Run the existing policy on structured fields, not the model prompt."""
    fields = ("id", "expected_minor", "actual_minor", "expected_currency", "actual_currency")
    return [{"id": row["id"], "expected": row["expected_label"],
             "output": synthetic_policy.assess({key: row[key] for key in fields})["category"],
             "max_token_limit_hit": False} for row in challenge]


def compare_to_baseline(baseline, model):
    """Count paired strict wins and losses, rejecting mismatched case sequences."""
    if len(baseline) != len(model) or any(
        b["id"] != m["id"] or b["expected"] != m["expected"]
        for b, m in zip(baseline, model)
    ):
        raise ValueError("baseline and model cases do not align")
    return {"model_only_correct": sum(m["output"] == m["expected"] and b["output"] != b["expected"]
                                      for b, m in zip(baseline, model)),
            "baseline_only_correct": sum(b["output"] == b["expected"] and m["output"] != m["expected"]
                                         for b, m in zip(baseline, model))}


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-home", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--threads", type=int, default=4, choices=range(1, 5))
    args = p.parse_args()
    root = args.output_root.resolve(strict=True)
    cache = args.hf_home.resolve(strict=True)
    run = args.run_dir.resolve(strict=True)
    if not run.is_dir() or run.stat().st_uid != os.getuid() or run.stat().st_mode & 0o077:
        raise SystemExit("training run must be an owner-only directory")
    try:
        snap = preflight(cache, root)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"preflight failed: {exc}") from exc
    manifest = json.loads((run / "result.json").read_text())
    if manifest.get("status") != "completed" or manifest.get("model") != MODEL or manifest.get("revision") != REVISION:
        raise SystemExit("training manifest mismatch")
    adapter = run / "adapter"
    try:
        verify_adapter_hashes(adapter, manifest)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    data_path = run / "synthetic-dataset.json"
    if sha_file(data_path) != manifest.get("dataset_sha256"):
        raise SystemExit("training dataset hash mismatch")
    dataset = json.loads(data_path.read_text())
    challenge = challenge_cases()
    prior_prompts = {r["prompt"] for rows in dataset.values() for r in rows}
    if any(row["prompt"] in prior_prompts for row in challenge):
        raise SystemExit("challenge overlaps previous prompts")
    fd = os.open(root / ".training.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as lock:
        try:
            if fcntl is not None:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another participating job holds the lock") from exc
        run_eval(args, root, cache, snap, adapter, manifest, challenge)


def run_eval(args, root, cache, snap, adapter, manifest, challenge):
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HOME=str(cache), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", OMP_NUM_THREADS=str(args.threads),
                      MKL_NUM_THREADS=str(args.threads), TOKENIZERS_PARALLELISM="false")
    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    output = root / (time.strftime("synthetic-challenge-%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}")
    output.mkdir(mode=0o700, exist_ok=False)
    content = json.dumps(challenge, sort_keys=True, separators=(",", ":")).encode()
    (output / "synthetic-challenge.json").write_bytes(content)
    protocol = {"challenge_sha256": hashlib.sha256(content).hexdigest(), "adapter_sha256": manifest["adapter_sha256"],
                "adapter_config_sha256": manifest["adapter_config_sha256"], "model": MODEL, "revision": REVISION,
                "scoring": "exact and case-normalized labels; missing/different currency requires review",
                "baseline": "synthetic_policy.assess on structured fields; challenge oracle encodes the same rules",
                "do_sample": False, "max_new_tokens": 8,
                "training_on_challenge": None,
                "training_on_challenge_assessment": "not verified; exact prompt overlap checked against supplied dataset"}
    (output / "protocol.json").write_text(json.dumps(protocol, sort_keys=True, indent=2) + "\n")
    tokenizer, model = load_verified_snapshot(snap, AutoTokenizer, AutoModelForCausalLM, torch)
    config = GenerationConfig(do_sample=False, max_new_tokens=8, pad_token_id=tokenizer.eos_token_id,
                              eos_token_id=tokenizer.eos_token_id)

    def generate(m):
        m.eval()
        records = []
        for row in challenge:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt")
            with torch.no_grad():
                tokens = m.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), generation_config=config)[0, prompt.shape[-1]:]
            text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            records.append({"id": row["id"], "expected": row["expected_label"], "output": text,
                            "max_token_limit_hit": len(tokens) == 8 and int(tokens[-1]) != tokenizer.eos_token_id})
        return records

    base = generate(model)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
    adapted = generate(model)
    baseline = deterministic_baseline(challenge)
    result = {"status": "completed", "data_origin": "synthetic", "model": MODEL, "revision": REVISION,
              "base_model_sha256": MODEL_SHA256, "adapter_sha256": manifest["adapter_sha256"],
              "adapter_config_sha256": manifest["adapter_config_sha256"], "challenge_sha256": protocol["challenge_sha256"],
              "training_dataset_sha256": manifest["dataset_sha256"], "protocol_sha256": sha_file(output / "protocol.json"),
              "baseline_script_sha256": sha_file(Path(synthetic_policy.__file__)),
              "script_sha256": sha_file(Path(__file__)), "base": {"metrics": summarize(base), "cases": base},
              "adapter": {"metrics": summarize(adapted), "cases": adapted},
              "deterministic_baseline": {"metrics": summarize(baseline), "cases": baseline},
              "versus_deterministic_baseline": {"base": compare_to_baseline(baseline, base),
                                                 "adapter": compare_to_baseline(baseline, adapted)},
              "duration_seconds": round(time.monotonic() - started, 2),
              "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
              "limitations": ["authored synthetic challenge not representative of production", "challenge labels and deterministic baseline encode the same rules; baseline score is rule self-consistency", "baseline receives structured fields while models receive text", "no retraining on challenge is an unverified claim", "observed test is no longer pristine", "unsigned metadata and hashes are not authenticated provenance", "no promotion decision", "all production outcomes require deterministic policy and human review"]}
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"evaluation_dir": str(output), "challenge_sha256": result["challenge_sha256"],
                      "base": result["base"]["metrics"], "adapter": result["adapter"]["metrics"],
                      "deterministic_baseline": result["deterministic_baseline"]["metrics"],
                      "versus_deterministic_baseline": result["versus_deterministic_baseline"],
                      "limitations": result["limitations"],
                      "duration_seconds": result["duration_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
