# Case Study: Can a 0.5B LoRA replace a keyword regex for exception coding — and can you *prove* it?

This directory contains a complete, reproducible, synthetic-only run of the Foundry
promotion pipeline on a real failure mode from
[Settler](https://github.com/Hardonian/Settler) (same owner): classifying operator
adjudication notes into reconciliation resolution codes.

**The claim under test:** a small fine-tuned model can beat Settler's production
first-match-wins keyword regex, with **zero critical false negatives**, and the
promotion decision is backed by a tamper-evident, cryptographically signed
evidence ledger — not a screenshot of a notebook.

**Result (run `run-20261111T031107Z-6a0895d1`):** the claim cleared every gate.
`ALLOWED_REVIEW_REQUIRED`, exit 0, zero missing evidence, signature verified at
the ledger head. Full numbers below — read the Limitations section before
citing them.

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
| Keyword regex (Settler port, first-match-wins) | 0.750 | 0.7875 | **0.600** (96/160 critical notes miscoded) |
| Qwen2.5-Coder-0.5B-Instruct + LoRA r=8 | 1.000 | 1.000 | **0.000** (0/160) |

The per-class view is where the story lives. Baseline recall on the two codes
that must never be downgraded: `REOPENED_FOR_INVESTIGATION` **0.333**,
`MISSING_COUNTERPART_CONFIRMED` **0.500** — and its precision on benign closures
gets poisoned by the adversarial traps (0.50 on AMOUNT/CURRENCY, 0.67 on
DUPLICATE): the regex confidently codes notes whose *first* keyword appears
before the operator's actual decision. Critical codes:
`REOPENED_FOR_INVESTIGATION`, `MISSING_COUNTERPART_CONFIRMED`.

## Why this is not theater

- **Synthetic, deterministic, seeded.** 1152 notes = 12 writing-style families ×
  12 scenario archetypes × 8 slot variants (`generate_dataset.py`,
  `BASE_SEED=20261110`). No real or customer data anywhere.
- **Family-disjoint frozen split.** Train/validation/test = 7/1/4 style families
  (672/96/384 records); no split shares a family. Every family carries all eight
  gold labels (including `UNDETERMINED`), so any test split satisfies Foundry's
  label-coverage gate. The test split was scored exactly once, after training.
- **The baseline is a faithful port** of the production rule order and regexes
  (`baseline_predict.py`), not a strawman — it gets 0.75 top-1, including one
  adversarial trap that accidentally spells the trigger word for its own gold
  class ("…submitted the same payment **twice**" → rule 1 fires correctly).
- **The gate is fail-closed.** The unsigned promotion attempt is captured in
  `task/promote_blocked.json` (exit 2, missing `cryptographic_promotion_signature`
  and `model_licensing_manifest`). Only after the licensing manifest was bound
  into the ledger (seq 2) and the ledger head signed with Ed25519 (seq 3) did
  `promote --keyring` return `ALLOWED_REVIEW_REQUIRED` (`task/promote_final.json`)
  — and a human still has to act on it.
- **Provenance is pinned.** The spec freezes base model
  `Qwen/Qwen2.5-Coder-0.5B-Instruct` @ revision `ea3f2471cf1b…`, adapter
  `sha256 7a867d837114…`, dataset `sha256 fbe0a1b143fa…`, baseline predictions
  `sha256 3100a51cf1fb…`, model predictions `sha256 dbb764bf3421…` *before*
  scoring (`finalize_spec.py` refuses to rewrite a finalized spec; the predictor
  refuses to run if dataset or adapter hashes drift).

## Evidence chain (all committed)

```
seq 1  run record           3f3d071c2d19…   (frozen split, dual scoring, gates PASS at metric level)
seq 2  model_licensing_manifest  93844bb27be0…   (Apache-2.0, decision ALLOWED, manifest sha d0f6a185…)
seq 3  promotion_signature  afec55be6283…   Ed25519 over the run record + ledger head 93844bb27be0…
       public key           435c4ca03849dbba69b4db08bc429ea87d352a756b59a86ca377a13774c3d426
```

Verify without trusting this README: `foundry refinery inspect <run-id>
--output-root task/runs` replays the hash chain in `task/runs/evidence.jsonl`
(exit 0 = chain intact), and `promote --keyring task/keyring.json` re-verifies
the signature against the pinned public key. The private key never left
`task/keys/` (gitignored); the committed keyring holds the public key only.

## Reproduce

Requires: Linux, ~2 GB disk, the pinned HF snapshot cached under `$HF_HOME`,
and the Foundry training stack (Python 3.12, torch 2.6.0+cpu, transformers
4.49.0, peft 0.14.0, `hardonian-foundry[signing]` — see `docs/training.md`).

```sh
~/venvs/foundry-train/bin/pip install -e ../../[signing]   # one package, one split impl
python3 generate_dataset.py --out-dir task                  # frozen dataset + spec (--profile standard)
python3 baseline_predict.py --task-dir task --out task/baseline.jsonl
bash run_case_study.sh                                      # train -> predict -> signed gate
```

### Portable Replay Verification (No Weights Required)

On any freshly cloned checkout (Linux, macOS, or Windows), you can replay and verify the committed signed evidence chain without training or modifying `task/run.json` / `task/runs/`:

```sh
PYTHONPATH=../../src python3 -m hardonian_foundry.cli refinery inspect run-20261111T031107Z-6a0895d1 --output-root task/runs
PYTHONPATH=../../src python3 -m hardonian_foundry.cli refinery promote run-20261111T031107Z-6a0895d1 --output-root task/runs --keyring task/keyring.json
```

### Dataset Stress Profiles (`--profile`)

`generate_dataset.py` preserves exact byte/record parity under `--profile standard --seed 20261110` (1,152 records) while offering two additional deterministic stress profiles for baseline and adapter evaluation:

| Profile | Records | Critical Share | Baseline Macro-F1 | Baseline Critical FN Rate | What It Tests |
|---|---:|---:|---:|---:|---|
| `standard` (default) | 1,152 | 41.67% | 0.7875 | 0.6000 | First-match keyword ordering traps across 12 style families |
| `adversarial-noisy` | 1,152 | 33.33% | 0.4792 | 1.0000 | Typos, abbreviations (`ccy`/`amt`/`dup`/`reconcil`), negation flips, multi-sentence conflicts, near-tie `UNDETERMINED` notes |
| `imbalanced` | 1,200 | 4.00% | 0.8940 | 0.5625 | Rare critical escalations masked by high aggregate macro-F1 |

To run weight-free offline case studies covering agent tool-call safety and deterministic reconciliation arithmetic override, see [case-study-agent-tool-gate](../case-study-agent-tool-gate/README.md) and [case-study-reconciliation-policy](../case-study-reconciliation-policy/README.md).

`run_case_study.sh` is the whole story in one log: train the LoRA on CPU
(32.4 min, 12 threads, loss converged by epoch 1), bind adapter provenance into
the spec, greedily score the frozen 384-record test split (7.0 min; all 384
outputs matched the allowlist, zero mapped to `UNDETERMINED`), then walk the
pipeline: `validate-dataset` → `run` → `inspect` → `promote` (BLOCKED, exit 2)
→ `attach-license` → `keygen` → `sign` → `promote --keyring`
(`ALLOWED_REVIEW_REQUIRED`, exit 0).

## Limitations (read before citing numbers)

- **The model's 1.000 is a ceiling artifact of synthetic templates, not real-world
  accuracy.** A 0.5B LoRA memorizes closed-vocabulary slot geometry quickly; the
  point of this study is the *decision process and provenance chain*, not a claim
  that the adapter is production-accurate on real operator prose. The comparison
  that matters is structural: the regex fails exactly the notes that mix a stale
  keyword with a live escalation, and the trained model doesn't — on this
  distribution.
- In-style synthetic notes only. Family-disjoint splits test style
  generalization, not messiness. Real notes have typos, multilingual fragments,
  and tenant-specific jargon; a pilot on real (accessed-with-consent) adjudication
  history is the next honest step.
- A single base model at one revision, one seed, CPU greedy decoding. No
  confidence intervals; one honest measurement, not a distribution.
- The observed test split is no longer pristine after scoring; re-running on the
  same split would be fitting to the test set.
- `ALLOWED_REVIEW_REQUIRED` is not deployment. Foundry gates *evidence*, not
  suitability; a human decides.

## Verdict

Every gate passed on real run output: metrics (`macro-F1 1.000 ≥ 0.800`
threshold; strict improvement over baseline `0.7875`), safety
(`critical_fn 0.000` vs baseline `0.600`), licensing (`ALLOWED`, SPDX
Apache-2.0), provenance (`PROVENANCE_VERIFIED`), and cryptography (signature
verifies at ledger head). This is Foundry's first end-to-end signed promotion
attestation — and note what it does **not** say: the ledger proves *this run's
evidence was complete and untampered at signing time*. It cannot prove the
model is right about real notes. That separation is the product.
