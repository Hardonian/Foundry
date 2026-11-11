#!/usr/bin/env python3
"""Generate the synthetic exception-resolution-code dataset for the Foundry case study.

1152 records = 12 writing-style families x 12 scenario archetypes x 8 slot variants.
Gold labels are Settler's seven resolution codes plus UNDETERMINED (notes with no
codeable content - the honest recall case, and the same fallback the keyword
baseline produces). Every family carries all eight labels so ANY family-disjoint
test split satisfies Foundry's task_label_coverage gate.

Four of the twelve archetypes are adversarial: they deliberately contain a keyword
that Settler's first-match-wins regex latches onto before the true rule - the same
structural failure measured in the read-only probe (0.438 top-1 on adversarial
notes). `critical` marks REOPENED_FOR_INVESTIGATION / MISSING_COUNTERPART_CONFIRMED:
codes that must never be silently downgraded to a benign closure.

Everything is synthetic; no real or customer data. Deterministic for a fixed seed.
"""
import argparse
import json
import random
from pathlib import Path

BASE_SEED = 20261110

D, T, A, C, M, F, R, U = (
    "DUPLICATE_RECORD_CONFIRMED",
    "TIMING_DIFFERENCE_ACCEPTED",
    "AMOUNT_VARIANCE_ACCEPTED",
    "CURRENCY_VARIANCE_ACCEPTED",
    "MISSING_COUNTERPART_CONFIRMED",
    "FALSE_POSITIVE_CONFIRMED",
    "REOPENED_FOR_INVESTIGATION",
    "UNDETERMINED",
)
LABELS = [D, T, A, C, M, F, R, U]
CRITICAL_GOLD = {R, M}

PARTIES = ["Northwind Logistics", "Kestrel Payments", "Bluepeak Energy", "Cobalt Freight",
           "Alderpoint Bank", "Silverline Telecom", "Harborview Foods", "Ironvale Mining"]
AMOUNTS = ["USD 1,204.55", "EUR 870.10", "USD 42.99", "GBP 15,300.00", "USD 0.02",
           "CAD 998.14", "EUR 12,450.00", "USD 617.33", "JPY 84,200", "AUD 2,055.80",
           "USD 133.07", "CHF 4,910.25"]
DATES = ["Mar 3", "Mar 11", "Apr 2", "Apr 19", "May 6", "May 21", "Jun 1", "Jun 17",
         "Jul 8", "Jul 23", "Aug 5", "Aug 30"]
REFS = ["INV-8841", "INV-2079", "PAY-5510", "PAY-6634", "BATCH-118", "BATCH-907",
        "TXN-4402", "TXN-7713", "WIRE-221", "WIRE-860"]

# (gold, template, baseline_first_match_is_gold). Templates are worded so the first
# regex rule that fires either IS the gold rule (True) or a trap earlier in the rule
# order than the gold rule (False).
ARCHETYPES = [
    (D, "same invoice {ref} was imported twice on {date}; duplicate confirmed against {party}", True),
    (T, "wire from {party} landed after the cutoff on {date}; the settlement window explains the {amount} gap", True),
    (A, "processing fee of {amount} from {party} on {date} explains the variance; accepted", True),
    (C, "foreign exchange revaluation on the {party} leg moved the rate; fx delta of {amount} accepted", True),
    (M, "the {party} leg for {ref} is not posted anywhere in the ledger; no counterpart found, entry confirmed missing", True),
    (F, "alert fired on an expected gap in the feed for {party}; not actionable, false positive", True),
    (R, "escalate to tier 2 and re-open {ref}; we must investigate why {party} came up short on {date}", True),
    (U, "referred to {party} on {date}; awaiting their response on {ref} before anything can be decided", True),
    (R, "earlier notes suspected a duplicate on {ref}, but the {party} figures still do not tie out; reopen for investigation", False),
    (R, "the fx angle on {ref} was a red herring; investigate the {party} batch on {date} before anything else, do not close", False),
    (M, "this is not a rounding problem; the {party} leg for {amount} was never sent and nothing is pending capture, the record is missing entirely", False),
    (D, "it looked like a cutoff artifact, but {party} submitted the same payment twice on {date}; duplicate confirmed", False),
]

OPENERS = ["Note:", "Triage:", "Ops log:", "Reviewer:", "Desk memo:", "Ticket update:",
           "Shift handover:", "Queue comment:", "Audit trail:", "Ops:", "Analyst:", "Case file:"]
PREAMBLES = [["", "", ""],
             ["After review, ", "Following the queue item, ", "Per the runbook, "],
             ["Quick one - ", "FYI ", "Heads up: "],
             ["", "Checking in: ", "As discussed, "]]
SUFFIXES = [["", "", ""],
            [" (desk {k})", " (desk {k})", " [auto]"],
            [" - ref {ref2}", " - ref {ref2}", ""],
            ["", " [verified]", " [verified]"]]


def compose(family, archetype, variant_seed):
    rng = random.Random(variant_seed)  # nosec B311 - deterministic synthetic-data styling, not security randomness
    opener = OPENERS[family]
    preamble = rng.choice(PREAMBLES[family % len(PREAMBLES)])
    suffix = rng.choice(SUFFIXES[family % len(SUFFIXES)])
    template = archetype[1]
    core = template.format(party=rng.choice(PARTIES), amount=rng.choice(AMOUNTS),
                           date=rng.choice(DATES), ref=rng.choice(REFS))
    if family % 3 == 1:  # terse style drops the opener
        opener = ""
    head = f"{opener} " if opener else ""
    text = f"{head}{preamble}{core}{suffix.format(k=family + 1, ref2=rng.choice(REFS))}"
    return " ".join(text.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--records-per-archetype", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records, seen_prompts, idx = [], set(), 0
    for family in range(12):
        for s_pos in range(12):
            archetype = ARCHETYPES[(s_pos + family) % len(ARCHETYPES)]
            base_seed = BASE_SEED + family * 1000 + s_pos * 100
            made, attempt = 0, 0
            while made < args.records_per_archetype:
                prompt = compose(family, archetype, base_seed + attempt)
                attempt += 1
                if prompt in seen_prompts:
                    continue
                seen_prompts.add(prompt)
                gold = archetype[0]
                records.append({
                    "id": f"synthetic-exc-{idx:04d}",
                    "prompt": prompt,
                    "gold_label": gold,
                    "critical": gold in CRITICAL_GOLD,
                    "family": f"style-{family:02d}",
                })
                idx += 1
                made += 1

    with (out / "data.jsonl").open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    spec = {
        "schema_version": "1.0",
        "task_id": "settler-exception-resolution-codes",
        "description": ("Classify operator adjudication notes into Settler resolution codes. "
                        "Synthetic, family-disjoint. Critical codes REOPENED_FOR_INVESTIGATION "
                        "and MISSING_COUNTERPART_CONFIRMED must never be missed. UNDETERMINED is "
                        "the honest no-codeable-content class."),
        "labels": LABELS,
        "thresholds": {"macro_f1_min": 0.8},
        "baseline_must_improve": True,
        "split_seed": BASE_SEED,
        "model": {
            "base_model_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "revision": "ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
            "adapter_sha256": "0" * 64,  # placeholder; finalize_spec.py injects the trained adapter hash
            "adapter_name": "exception-codes-lora",
        },
        "dataset_path": "data.jsonl",
        "notes": "Dataset generated deterministically by generate_dataset.py (BASE_SEED=20261110).",
    }
    (out / "spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    n_critical = sum(1 for r in records if r["critical"])
    print(json.dumps({"records": len(records), "families": 12, "archetypes": 12,
                      "critical": n_critical, "out_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
