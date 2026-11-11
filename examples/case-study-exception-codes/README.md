# Case Study: Can a 0.5B LoRA replace a keyword regex for exception coding — and can you *prove* it?

This directory contains a complete, reproducible, synthetic-only run of the Foundry
promotion pipeline on a real failure mode from
[Settler](https://github.com/Hardonian/Settler) (same owner): classifying operator
adjudication notes into reconciliation resolution codes.

**The claim under test:** a small fine-tuned model can beat Settler's production
first-match-wins keyword regex, with **zero critical false negatives**, and the
promotion decision is backed by a tamper-evident, cryptographically signed
evidence ledger — not a screenshot of a notebook.

## The honest problem

Settler's `exception-intelligence.ts` codes notes with seven ordered regexes,
first match wins. That ordering is structurally unsafe: a note like

> "earlier notes suspected a duplicate on INV-8841, but the Kestrel Payments
> figures still do not tie out; reopen for investigation"

matches `duplicate` first and gets silently coded `DUPLICATE_RECORD_CONFIRMED`
— a benign closure — when the operator asked to **reopen for investigation**.
On the frozen adversarial test split of this synthetic dataset, the ported
baseline scores:

| Predictor | top-1 | macro-F1 | critical false-negative rate |
|---|---|---|---|
| Keyword regex (Settler port, first-match-wins) | `__BASE_TOP1__` | `__BASE_MACRO_F1__` | `__BASE_CFN__` (`__CRIT_MISSED__`/`__CRIT_TOTAL__` critical notes miscoded) |
| Qwen2.5-Coder-0.5B-Instruct + LoRA r=8 | `__MODEL_TOP1__` | `__MODEL_MACRO_F1__` | `__MODEL_CFN__` |

Critical codes: `REOPENED_FOR_INVESTIGATION`, `MISSING_COUNTERPART_CONFIRMED` —
the two outcomes that must never be silently downgraded to a benign closure.

## Why this is not theater

- **Synthetic, deterministic, seeded.** 900 notes = 12 writing-style families x
  5 scenarios x 15 slot variants (`generate_dataset.py`, `BASE_SEED=20261110`).
  No real or customer data anywhere.
- **Family-disjoint frozen split.** Train/val/test never share a style family
  (7/1/4 families). The test split was scored exactly once, after training.
- **The baseline is a faithful port** of the production regexes
  (`baseline_predict.py`), not a strawman — it gets the easy notes right.
- **The gate is fail-closed.** The unsigned promotion attempt is captured in
  `promote_blocked.json` (exit 2, exact `missing_evidence`). Only after the
  licensing manifest is bound into the ledger and the ledger head is signed
  with Ed25519 does `promote --keyring` return `ALLOWED_REVIEW_REQUIRED`
  (`promote_final.json`) — and a human still has to act on it.
- **Provenance is pinned.** Spec freezes `base_model_id`, revision
  `ea3f2471...`, and the adapter's sha256 *before* scoring
  (`finalize_spec.py` refuses to rewrite a finalized spec; the predictor
  refuses to run if dataset or adapter hashes drift).

## Reproduce

Requires: Linux, ~2 GB disk, the pinned HF snapshot cached under `$HF_HOME`,
and the Foundry training stack (Python 3.12, torch 2.6.0+cpu,
transformers 4.49.0, peft 0.14.0 — see `docs/training.md`).

```sh
python3 generate_dataset.py --out-dir task          # frozen dataset + spec
python3 baseline_predict.py --task-dir task --out task/baseline.jsonl
bash run_case_study.sh                              # train -> predict -> signed gate
```

`run_case_study.sh` is the whole story in one log (`case-study.log`): train the
LoRA (CPU, ~`__TRAIN_MIN__` min), bind adapter provenance into the spec, score
the frozen test split, then walk the Foundry pipeline: `validate-dataset` ->
`run` -> `inspect` -> `promote` (BLOCKED, exit 2) -> `attach-license` ->
`keygen` -> `sign` -> `promote --keyring` (ALLOWED_REVIEW_REQUIRED, exit 0).

## Evidence files (committed)

| File | What it proves |
|---|---|
| `task/spec.json` | Frozen task contract incl. adapter sha256 + model revision |
| `task/data.jsonl` | The 900-record synthetic dataset |
| `task/baseline.jsonl` / `task/model.jsonl` | Raw predictions, both predictors, frozen test split |
| `task/validate.json`, `task/run.json`, `task/inspect.json` | Foundry validation, run provenance, ledger verification |
| `task/promote_blocked.json` | Fail-closed gate before licensing/signing (exit 2) |
| `task/attach_license.json`, `task/sign.json`, `task/keyring.json` | License decision bound into ledger; Ed25519 attestation; trusted key (public only) |
| `task/promote_final.json` | `ALLOWED_REVIEW_REQUIRED` with zero missing evidence |

Adapter weights, run ledgers, and the private signing key stay local
(`.gitignore`d) — the keyring holds only the public key.

## Verdict

`__VERDICT__`

## Limitations (read before citing numbers)

- In-style synthetic notes only. Family-disjoint splits test style
  generalization, not the messiness of real operator prose.
- A single base model at one revision, one seed, CPU greedy decoding. No
  confidence intervals; treat deltas as one honest measurement, not a
  distribution.
- The observed test split is no longer pristine after scoring; re-running the
  study on the same split would be fitting to the test set.
- `ALLOWED_REVIEW_REQUIRED` is not deployment. Foundry gates *evidence*, not
  suitability; a human decides.
