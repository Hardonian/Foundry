# Case Study #3: Deterministic Reconciliation Arithmetic Overriding Untrusted Model Suggestions

This directory contains a complete, reproducible, synthetic-only Foundry task-mode case study demonstrating how `hardonian_foundry.synthetic_policy.assess` and `assess_with_suggestion` enforce **deterministic integer minor-unit arithmetic** so an untrusted model suggestion can be logged and audited without ever altering the reconciliation verdict or bypassing human review.

No Hugging Face weights, GPU, or network access are required.

## What this proves

When an upstream model emits reconciliation labels (`matched`, `amount_mismatch`, `review_required`), trusting its output directly creates silent failures across five recurring traps:

1. **Sub-unit / 1-minor-unit rounding hallucinations:** `expected_minor: 10000`, `actual_minor: 10001` where the model outputs `matched`
2. **Sign reversals:** `expected_minor: -12000`, `actual_minor: 12000` where the model compares magnitude digits and outputs `matched`
3. **Large-integer float-precision truncation:** `100000000000` vs `100000000001`
4. **Currency mismatches:** equal minor units (`50000 == 50000`) across `USD` vs `EUR` — must become `review_required`
5. **Missing / null fields and malformed outputs:** `actual_minor: null`, `actual_currency: null`, or off-allowlist strings (`"Matched"`, `""`, `"unknown"`)

Calling `assess_with_suggestion(case, suggestion)` computes `assess(case)` deterministically over exact integer minor units and records `model_agrees` and `model_status` (`REVIEW_REQUIRED` or `UNAVAILABLE`) separately — guaranteeing `review_required=True` and `financial_write=False` on every record.

On the 192-record family-disjoint test split (576 total records across 12 synthetic feed families, `BASE_SEED=20261112`):

| Predictor | Accuracy | Macro-F1 | Critical False-Negative Rate |
|---|---:|---:|---:|
| Untrusted-suggestion baseline (`baseline_predict.py`) | 0.3333 | 0.3444 | **0.6250** (80/128 critical misses) |
| Deterministic policy gate (`policy_predict.py` via `assess_with_suggestion`) | 1.0000 | 1.0000 | **0.0000** (0/128 critical misses) |

In the test split evaluation of `policy_predict.py`, `48` untrusted suggestions agreed, `144` were overridden (including `48` marked `model_status: "UNAVAILABLE"`), `all_review_required` remained `true`, and `any_financial_write` remained `false`.

## Files

- `generate_dataset.py` — deterministic generator producing 576 synthetic records (`12` feed families × `12` archetypes × `4` slot variants) with gold labels computed directly by `synthetic_policy.assess`, plus `spec.json` pinning `policy_predict.py` SHA-256
- `baseline_predict.py` — naive baseline that trusts allowlisted untrusted model suggestions
- `policy_predict.py` — integrates `synthetic_policy.assess_with_suggestion` to override untrusted model suggestions deterministically (and verifies its own SHA-256 against `spec.json`)
- `run_case_study.sh` — full end-to-end Foundry task-mode run (`generate_dataset` → predict → `validate-dataset` → `run` → `inspect` → `promote` blocked exit 2 → `keygen` → `sign` → `promote --keyring`)
- `task/` — generated `spec.json`, `data.jsonl`, `baseline.jsonl`, `model.jsonl`, and `manifest.json`

## Run locally

```sh
PYTHONPATH=src python3 examples/case-study-reconciliation-policy/generate_dataset.py --out-dir examples/case-study-reconciliation-policy/task
PYTHONPATH=src python3 examples/case-study-reconciliation-policy/baseline_predict.py --task-dir examples/case-study-reconciliation-policy/task --out examples/case-study-reconciliation-policy/task/baseline.jsonl
PYTHONPATH=src python3 examples/case-study-reconciliation-policy/policy_predict.py --task-dir examples/case-study-reconciliation-policy/task --out examples/case-study-reconciliation-policy/task/model.jsonl
PYTHONPATH=src python3 -m hardonian_foundry.cli refinery validate-dataset --task-spec examples/case-study-reconciliation-policy/task/spec.json
```

Or run the complete signed promotion flow (requires `hardonian-foundry[signing]`):

```sh
bash examples/case-study-reconciliation-policy/run_case_study.sh
```

## Limitations

- Uses synthetic reconciliation records with `synthetic-` prefix IDs; real financial integration must preserve tenant authorization, proofpacks, and human review (`review_required: true`, `financial_write: false`).
