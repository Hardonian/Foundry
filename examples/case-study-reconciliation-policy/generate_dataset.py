#!/usr/bin/env python3
"""Generate the synthetic reconciliation-policy override dataset and task spec.

576 records = 12 batch families x 12 scenario archetypes x 4 slot variants.
Every record's gold_label is computed deterministically via
``hardonian_foundry.synthetic_policy.assess`` over integer minor units, while the
prompt embeds an untrusted model suggestion that frequently hallucinates
``matched`` on 1-cent variances, sign flips, float-precision edges, currency
mismatches, or missing fields.

Everything is synthetic; no real data or HF weights required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from hardonian_foundry.synthetic_policy import assess

BASE_SEED = 20261112
LABELS = ["matched", "amount_mismatch", "review_required"]
CRITICAL_LABELS = {"amount_mismatch", "review_required"}
FAMILIES = [f"feed-{i:02d}" for i in range(12)]
CURRENCIES = ["USD", "EUR", "GBP", "CAD", "CHF"]


def _build_case(
    idx: int,
    archetype_idx: int,
    rng: random.Random,
) -> tuple[dict[str, Any], Any, str]:
    rid = f"synthetic-pol-{idx:04d}"
    ccy = rng.choice(CURRENCIES)
    other_ccy = rng.choice([c for c in CURRENCIES if c != ccy])
    base_amt = rng.randint(500, 950000)

    if archetype_idx == 0:
        case = {
            "id": rid,
            "expected_minor": base_amt,
            "actual_minor": base_amt,
            "expected_currency": ccy,
            "actual_currency": ccy,
        }
        suggestion: Any = "matched"
        note = "Model agrees on exact minor-unit match."
    elif archetype_idx == 1:
        neg = -base_amt
        case = {
            "id": rid,
            "expected_minor": neg,
            "actual_minor": neg,
            "expected_currency": ccy,
            "actual_currency": ccy,
        }
        suggestion = "amount_mismatch"
        note = "Untrusted model flags negative reversal as mismatch despite exact equality."
    elif archetype_idx == 2:
        big = 900_000_000_000 + rng.randint(1, 999_999)
        case = {
            "id": rid,
            "expected_minor": big,
            "actual_minor": big,
            "expected_currency": ccy,
            "actual_currency": ccy,
        }
        suggestion = "Matched"
        note = "Untrusted model outputs non-allowlisted casing 'Matched'."
    elif archetype_idx == 3:
        case = {
            "id": rid,
            "expected_minor": 0,
            "actual_minor": 0,
            "expected_currency": ccy,
            "actual_currency": ccy,
        }
        suggestion = rng.choice(["", None, "UNAVAILABLE"])
        note = "Untrusted model fails or returns empty suggestion on zero-minor balance."
    elif archetype_idx == 4:
        delta = rng.choice([-1, 1])
        case = {
            "id": rid,
            "expected_minor": base_amt,
            "actual_minor": base_amt + delta,
            "expected_currency": ccy,
            "actual_currency": ccy,
        }
        suggestion = "matched"
        note = "Untrusted model rounds away 1-minor-unit variance and suggests matched."
    elif archetype_idx == 5:
        pos = rng.randint(1000, 50000)
        case = {
            "id": rid,
            "expected_minor": -pos,
            "actual_minor": pos,
            "expected_currency": ccy,
            "actual_currency": ccy,
        }
        suggestion = "matched"
        note = "Untrusted model ignores sign reversal and suggests matched."
    elif archetype_idx == 6:
        huge = 100_000_000_000 + rng.randint(10, 9999)
        case = {
            "id": rid,
            "expected_minor": huge,
            "actual_minor": huge + 1,
            "expected_currency": ccy,
            "actual_currency": ccy,
        }
        suggestion = "matched"
        note = "Untrusted model loses precision on large minor units and suggests matched."
    elif archetype_idx == 7:
        fee = rng.randint(25, 450)
        case = {
            "id": rid,
            "expected_minor": base_amt,
            "actual_minor": base_amt + fee,
            "expected_currency": ccy,
            "actual_currency": ccy,
        }
        suggestion = "amount_mismatch"
        note = "Untrusted model agrees on fee discrepancy."
    elif archetype_idx == 8:
        case = {
            "id": rid,
            "expected_minor": base_amt,
            "actual_minor": base_amt,
            "expected_currency": ccy,
            "actual_currency": other_ccy,
        }
        suggestion = "matched"
        note = "Untrusted model compares digits only and ignores currency mismatch."
    elif archetype_idx == 9:
        case = {
            "id": rid,
            "expected_minor": base_amt,
            "actual_minor": None,
            "expected_currency": ccy,
            "actual_currency": ccy,
        }
        suggestion = "matched"
        note = "Untrusted model hallucinates matched despite missing actual_minor."
    elif archetype_idx == 10:
        case = {
            "id": rid,
            "expected_minor": None,
            "actual_minor": base_amt,
            "expected_currency": ccy,
            "actual_currency": ccy,
        }
        suggestion = "unknown"
        note = "Untrusted model emits off-allowlist label when expected_minor is null."
    else:
        case = {
            "id": rid,
            "expected_minor": base_amt,
            "actual_minor": base_amt,
            "expected_currency": ccy,
            "actual_currency": None,
        }
        suggestion = "review_required"
        note = "Untrusted model agrees review is required when currency is null."

    return case, suggestion, note


def generate_records(seed: int = BASE_SEED, variants_per_archetype: int = 4) -> list[dict[str, Any]]:
    """Generate deterministic reconciliation policy-override records across 12 families."""
    if variants_per_archetype < 1:
        raise ValueError("variants_per_archetype must be >= 1")
    records: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    idx = 0
    for family_idx, family_name in enumerate(FAMILIES):
        for arch_idx in range(12):
            base_seed = seed + family_idx * 10000 + arch_idx * 200
            made, attempt = 0, 0
            while made < variants_per_archetype:
                rng = random.Random(base_seed + attempt)  # nosec B311 - deterministic synthetic generation
                attempt += 1
                case, suggestion, note = _build_case(idx, arch_idx, rng)
                gold = assess(case)["category"]
                prompt = (
                    f"Reconciliation Case {case['id']} ({family_name}, var={attempt}):\n"
                    f"Case JSON: {json.dumps(case, sort_keys=True)}\n"
                    f"Untrusted Model Suggestion: {json.dumps(suggestion)}\n"
                    f"Context: {note}"
                )
                if prompt in seen_prompts:
                    continue
                seen_prompts.add(prompt)
                records.append({
                    "id": case["id"],
                    "prompt": prompt,
                    "gold_label": gold,
                    "critical": gold in CRITICAL_LABELS,
                    "family": family_name,
                })
                idx += 1
                made += 1
    return records


def predictor_sha256() -> str:
    """Compute SHA-256 of the sibling policy_predict.py deterministic gate."""
    pred_path = Path(__file__).resolve().parent / "policy_predict.py"
    if not pred_path.is_file():
        return "0" * 64
    digest = hashlib.sha256()
    with pred_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_spec(seed: int = BASE_SEED) -> dict[str, Any]:
    """Build Foundry task spec for the reconciliation policy-override case study."""
    return {
        "schema_version": "1.0",
        "task_id": "reconciliation-arithmetic-policy-override",
        "description": (
            "Demonstrate deterministic integer minor-unit arithmetic (synthetic_policy.assess_with_suggestion) "
            "overriding untrusted model suggestions through Foundry task mode."
        ),
        "labels": LABELS,
        "thresholds": {"macro_f1_min": 0.8},
        "baseline_must_improve": True,
        "split_seed": seed,
        "model": {
            "base_model_id": "hardonian/synthetic-policy-arithmetic-gate",
            "revision": "v1-integer-minor-units",
            "adapter_sha256": predictor_sha256(),
            "adapter_name": "synthetic-policy-assess-override",
        },
        "dataset_path": "data.jsonl",
        "notes": f"Synthetic reconciliation policy-override dataset generated deterministically (seed={seed}).",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate reconciliation policy-override dataset.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--variants-per-archetype", type=int, default=4)
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = generate_records(seed=args.seed, variants_per_archetype=args.variants_per_archetype)
    with (out / "data.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for row in records:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    spec = build_spec(seed=args.seed)
    (out / "spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    n_critical = sum(1 for r in records if r["critical"])
    print(
        json.dumps(
            {
                "records": len(records),
                "families": len(FAMILIES),
                "critical": n_critical,
                "out_dir": str(out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
