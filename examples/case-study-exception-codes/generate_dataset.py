#!/usr/bin/env python3
"""Generate the synthetic exception-resolution-code dataset for the Foundry case study.

Supports three deterministic synthetic dataset profiles:
- ``standard`` (default): 1152 records = 12 writing-style families x 12 scenario
  archetypes x 8 slot variants. Exact match to the frozen case-study dataset.
- ``adversarial-noisy``: 1152 records featuring operator typos, abbreviations
  (ccy/amt/dup/reconcil), negation flips, multi-sentence conflicts, and near-tie
  ambiguous ``UNDETERMINED`` cases.
- ``imbalanced``: 1200 records with rare critical escalations (4.0% critical across
  all families and splits) to stress recall under class skew.

Everything is synthetic; no real or customer data. Deterministic for a fixed seed.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

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
PROFILES = ("standard", "adversarial-noisy", "imbalanced")

PARTIES = [
    "Northwind Logistics",
    "Kestrel Payments",
    "Bluepeak Energy",
    "Cobalt Freight",
    "Alderpoint Bank",
    "Silverline Telecom",
    "Harborview Foods",
    "Ironvale Mining",
]
AMOUNTS = [
    "USD 1,204.55",
    "EUR 870.10",
    "USD 42.99",
    "GBP 15,300.00",
    "USD 0.02",
    "CAD 998.14",
    "EUR 12,450.00",
    "USD 617.33",
    "JPY 84,200",
    "AUD 2,055.80",
    "USD 133.07",
    "CHF 4,910.25",
]
DATES = [
    "Mar 3",
    "Mar 11",
    "Apr 2",
    "Apr 19",
    "May 6",
    "May 21",
    "Jun 1",
    "Jun 17",
    "Jul 8",
    "Jul 23",
    "Aug 5",
    "Aug 30",
]
REFS = [
    "INV-8841",
    "INV-2079",
    "PAY-5510",
    "PAY-6634",
    "BATCH-118",
    "BATCH-907",
    "TXN-4402",
    "TXN-7713",
    "WIRE-221",
    "WIRE-860",
]

# (gold, template, baseline_first_match_is_gold). Templates are worded so the first
# regex rule that fires either IS the gold rule (True) or a trap earlier in the rule
# order than the gold rule (False).
ARCHETYPES: list[tuple[str, str, bool]] = [
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

NOISY_ARCHETYPES: list[tuple[str, str, bool]] = [
    (D, "dup inv {ref} impoted twice on {date} w/ {party}; reconcil confirms same entry", True),
    (T, "wire frm {party} landed past cutoff {date}; settlment window explains {amount} amt gap", False),
    (A, "proc fee amt {amount} frm {party} on {date} explains varience; accepted", True),
    (C, "fx reval on {party} leg moved ccy rate; ccy delta amt {amount} accepted", False),
    (M, "not a dup and not a cutoff; {party} leg {ref} amt {amount} never posted, missng counterpart confirmed", False),
    (F, "Initial alert flagged {amount} gap on {ref} as critical. Reconcil update: expected feed pause for {party}, benign fp.", False),
    (R, "do not close {ref} as benign; {party} variance still fails reconcil on {date} - reopen for investgation", False),
    (U, "ambiguous {ref} ({party}, {amount}): logs show possible dup vs timing cutoff, cannot determine code; hold pend review", False),
    (R, "First pass guessed ccy fee drift on {ref}. Second pass: {party} batch short {amount} on {date}; escalate and re-open.", False),
    (M, "Suspected rounding fee on {ref}, but not an amt variance - {party} leg {amount} was never sent; missing counterpart.", False),
    (D, "not a ccy issue on {ref}; same {party} txn submitted twice on {date}, dup confirmed", True),
    (U, "desk A notes possible fx variance on {ref}, desk B says missing leg from {party} on {date}; conflicting evidence, undetermined", False),
]

NOISY_TAGS = [
    "[reconcil-v2]",
    "[ccy-chk]",
    "[pend-ops]",
    "(amt-note)",
    "(shift-b)",
    "[dup-scan]",
]

IMBALANCED_ALLOCATION: list[tuple[str, int]] = [
    (D, 24),
    (T, 20),
    (A, 18),
    (C, 14),
    (F, 12),
    (U, 8),
    (R, 2),
    (M, 2),
]

OPENERS = [
    "Note:",
    "Triage:",
    "Ops log:",
    "Reviewer:",
    "Desk memo:",
    "Ticket update:",
    "Shift handover:",
    "Queue comment:",
    "Audit trail:",
    "Ops:",
    "Analyst:",
    "Case file:",
]
PREAMBLES = [
    ["", "", ""],
    ["After review, ", "Following the queue item, ", "Per the runbook, "],
    ["Quick one - ", "FYI ", "Heads up: "],
    ["", "Checking in: ", "As discussed, "],
]
SUFFIXES = [
    ["", "", ""],
    [" (desk {k})", " (desk {k})", " [auto]"],
    [" - ref {ref2}", " - ref {ref2}", ""],
    ["", " [verified]", " [verified]"],
]


def compose(family: int, archetype: tuple[str, str, bool], variant_seed: int) -> str:
    """Compose a deterministic standard synthetic prompt for a family and archetype."""
    rng = random.Random(variant_seed)  # nosec B311 - deterministic synthetic-data styling, not security randomness
    opener = OPENERS[family]
    preamble = rng.choice(PREAMBLES[family % len(PREAMBLES)])
    suffix = rng.choice(SUFFIXES[family % len(SUFFIXES)])
    template = archetype[1]
    core = template.format(
        party=rng.choice(PARTIES),
        amount=rng.choice(AMOUNTS),
        date=rng.choice(DATES),
        ref=rng.choice(REFS),
    )
    if family % 3 == 1:  # terse style drops the opener
        opener = ""
    head = f"{opener} " if opener else ""
    text = f"{head}{preamble}{core}{suffix.format(k=family + 1, ref2=rng.choice(REFS))}"
    return " ".join(text.split())


def compose_noisy(family: int, archetype: tuple[str, str, bool], variant_seed: int) -> str:
    """Compose a deterministic noisy/adversarial synthetic prompt with abbreviations and tags."""
    rng = random.Random(variant_seed)  # nosec B311 - deterministic synthetic-data styling, not security randomness
    base = compose(family, archetype, variant_seed)
    tag = rng.choice(NOISY_TAGS)
    extra_ref = rng.choice(REFS)
    if variant_seed % 3 == 0:
        return f"{base} {tag} alt={extra_ref}"
    if variant_seed % 3 == 1:
        return f"{tag} {base} [seq:{variant_seed % 997}]"
    return f"{base} (reconcil batch {extra_ref})"


def generate_records(
    profile: str = "standard",
    seed: int = BASE_SEED,
    records_per_archetype: int = 8,
) -> list[dict[str, Any]]:
    """Generate synthetic exception-code records for the requested profile and seed."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")
    if records_per_archetype < 1:
        raise ValueError("records_per_archetype must be >= 1")

    records: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    idx = 0

    if profile == "imbalanced":
        scale = max(1, round(records_per_archetype / 8))
        archetypes_by_label: dict[str, list[tuple[str, str, bool]]] = {lbl: [] for lbl in LABELS}
        for arch in ARCHETYPES:
            archetypes_by_label[arch[0]].append(arch)
        for family in range(12):
            for label_pos, (label, base_count) in enumerate(IMBALANCED_ALLOCATION):
                count = base_count * scale
                pool = archetypes_by_label[label]
                base_seed = seed + family * 10000 + label_pos * 500
                made, attempt = 0, 0
                while made < count:
                    archetype = pool[(attempt + family) % len(pool)]
                    raw = compose(family, archetype, base_seed + attempt)
                    prompt = raw if raw not in seen_prompts else f"{raw} [lot {attempt + 1}]"
                    attempt += 1
                    if prompt in seen_prompts:
                        continue
                    seen_prompts.add(prompt)
                    records.append({
                        "id": f"synthetic-exc-{idx:04d}",
                        "prompt": prompt,
                        "gold_label": label,
                        "critical": label in CRITICAL_GOLD,
                        "family": f"style-{family:02d}",
                    })
                    idx += 1
                    made += 1
        return records

    table = ARCHETYPES if profile == "standard" else NOISY_ARCHETYPES
    composer = compose if profile == "standard" else compose_noisy
    for family in range(12):
        for s_pos in range(12):
            archetype = table[(s_pos + family) % len(table)]
            base_seed = seed + family * 1000 + s_pos * 100
            made, attempt = 0, 0
            while made < records_per_archetype:
                prompt = composer(family, archetype, base_seed + attempt)
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
    return records


def build_spec(profile: str = "standard", seed: int = BASE_SEED) -> dict[str, Any]:
    """Build the Foundry task spec dict for the given dataset profile and seed."""
    notes = (
        f"Dataset generated deterministically by generate_dataset.py (BASE_SEED={BASE_SEED})."
        if profile == "standard" and seed == BASE_SEED
        else f"Dataset generated deterministically by generate_dataset.py (profile={profile}, seed={seed})."
    )
    return {
        "schema_version": "1.0",
        "task_id": "settler-exception-resolution-codes",
        "description": (
            "Classify operator adjudication notes into Settler resolution codes. "
            "Synthetic, family-disjoint. Critical codes REOPENED_FOR_INVESTIGATION "
            "and MISSING_COUNTERPART_CONFIRMED must never be missed. UNDETERMINED is "
            "the honest no-codeable-content class."
        ),
        "labels": LABELS,
        "thresholds": {"macro_f1_min": 0.8},
        "baseline_must_improve": True,
        "split_seed": seed,
        "model": {
            "base_model_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "revision": "ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
            "adapter_sha256": "0" * 64,  # placeholder; finalize_spec.py injects the trained adapter hash
            "adapter_name": "exception-codes-lora",
        },
        "dataset_path": "data.jsonl",
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic exception-code task datasets.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--records-per-archetype", type=int, default=8)
    parser.add_argument("--profile", choices=PROFILES, default="standard")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = generate_records(
        profile=args.profile,
        seed=args.seed,
        records_per_archetype=args.records_per_archetype,
    )
    with (out / "data.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for row in records:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    spec = build_spec(profile=args.profile, seed=args.seed)
    (out / "spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    n_critical = sum(1 for r in records if r["critical"])
    print(
        json.dumps(
            {
                "profile": args.profile,
                "seed": args.seed,
                "records": len(records),
                "families": 12,
                "archetypes": 12,
                "critical": n_critical,
                "critical_ratio": round(n_critical / len(records), 4),
                "out_dir": str(out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
