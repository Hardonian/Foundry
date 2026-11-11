# Developer Refinery Workflow

The Refinery workflow connects dataset validation, training preflight, reproducible evaluation, tamper-evident run tracking, and model promotion gating into a coherent developer experience.

## Guiding Principles

- **Synthetic Only:** All inputs, fixtures, and evaluations use synthetic data. Never load, process, or commit real or customer financial records.
- **Fail-Closed Security:** Incomplete evidence, broken hash chains, untrusted licenses, or missing cryptographic signatures block promotion with exit code 2.
- **Review Required:** All evaluation decisions, heuristic classifications, and model suggestions require human review (`review_rate = 1.0`, `coverage_automatic_decision = 0.0`). Zero autonomous financial or tool writes exist.
- **Tamper Evidence:** Provenance is anchored into an append-only, hash-chained evidence ledger with external head verification.

---

## Workflow Components

```
                    ┌─────────────────────────┐
                    │    Synthetic Dataset    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   validate-dataset      │
                    │ (schema, leakage, split)│
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │        preflight        │
                    │(env, license, disk, dir)│
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     run (evaluate)      │
                    │  (metrics, holdout set) │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    evidence.jsonl       │
                    │  (hash-chained ledger)  │
                    └────────────┬────────────┘
                                 │
                     ┌───────────┴───────────┐
                     ▼                       ▼
          ┌─────────────────────┐ ┌─────────────────────┐
          │   inspect <run_id>  │ │   promote <run_id>  │
          │(verify provenance)  │ │ (evaluate evidence) │
          └─────────────────────┘ └──────────┬──────────┘
                                             │
                                             ▼
                                  [BLOCKED: Fail-Closed]
```

### 1. Dataset Validation (`validate-dataset`)
Validates integer minor units, synthetic ID prefixes (`synthetic-`), maximum ID lengths (<= 100), absence of duplicate IDs, and absence of identical record contents across differing IDs (leakage hazard). Verifies that the dataset can be partitioned into train, validation, and test splits with zero cross-split overlap.

Command:
```sh
PYTHONPATH=src python3 -m hardonian_foundry.cli refinery validate-dataset [--data <path>]
```

### 2. Training Preflight (`preflight`)
Executes diagnostic checks prior to executing a run or training task:
- Validates the target dataset.
- Verifies Python runtime (>= 3.11) and offline CPU mode.
- Validates the output directory and verifies storage availability.
- Evaluates the model licensing manifest against the SPDX allowlist (`Apache-2.0`, `MIT`, `BSD-3-Clause`, `CC0-1.0`, `OFL-1.1`), synthetic provenance, and synthetic data origin.
- Asserts execution safety boundaries (no autonomous tools or remote code).

Command:
```sh
PYTHONPATH=src python3 -m hardonian_foundry.cli refinery preflight [--output-root <dir>] [--manifest <path>]
```

### 3. End-to-End Run Execution (`run`)
Performs preflight checks, executes reproducible held-out evaluation against the deterministic baseline, creates a mode-0700 run directory (`run-<timestamp>-<hash>`), appends an entry to the tamper-evident evidence ledger (`evidence.jsonl`), and saves `run.json` with full provenance.

Command:
```sh
PYTHONPATH=src python3 -m hardonian_foundry.cli refinery run --output-root .foundry/runs
```

### 4. Run Inspection & Provenance Verification (`inspect`)
Loads the run record and cryptographically audits the evidence ledger. Confirms that:
1. The evidence ledger hash chain is intact from genesis.
2. The sequence number and entry hash match the recorded provenance.
3. The chain head is verified.

Command:
```sh
PYTHONPATH=src python3 -m hardonian_foundry.cli refinery inspect <run_id> --output-root .foundry/runs
```

### 5. Promotion Gate (`promote`)
Evaluates whether a candidate model run possesses all required evidence for promotion:
- **Run Metadata:** `run.json` exists and parses cleanly.
- **Provenance Proof:** Evidence ledger verifies with `PROVENANCE_VERIFIED`.
- **Preflight:** Training preflight passed cleanly.
- **Benchmarks:** Evaluation metrics present, `critical_false_negative_rate == 0.0`, `review_rate == 1.0`, `f1 >= 0.8`. For task-mode runs, each contract label must have positive test support; macro-F1 averages all contract labels, must clear `thresholds.macro_f1_min` and strictly beat the recorded baseline when required.
- **License Allowlist:** The SPDX-approved manifest must be in the signed run record or attached to the ledger before signing.
- **Signed Promotion:** A version 2 Ed25519 signature binds the complete canonical run record and ledger head, verified against a trusted keyring supplied with `--keyring`.

When any required evidence is missing or unverified, the gate reports the exact missing items in `missing_evidence` and exits with code 2. When everything verifies, the decision is `ALLOWED_REVIEW_REQUIRED` (exit 0) — a human still signs off; the gate never auto-promotes.

Command:
```sh
PYTHONPATH=src python3 -m hardonian_foundry.cli refinery promote <run_id> --output-root .foundry/runs [--keyring keyring.json]
```

### 6. Task-Mode Evaluation (bring-your-own predictors)
Task mode scores *externally produced* predictions against a frozen task contract — Foundry never calls the model itself, so any predictor (heuristic, LoRA adapter, hosted model) can be evaluated under identical, auditable rules.

- **Task spec** (`spec.json`): `task_id`, `labels`, `thresholds` (`macro_f1_min` >= 0.8 floor), `baseline_must_improve`, `split_seed`, and a `model` block pinning `base_model_id`, `revision`, and `adapter_sha256` into run provenance.
- **Dataset** (JSONL): `{"id", "prompt", "gold_label", "critical", "family"}`. Validation rejects duplicate ids/prompts, unknown labels, and mixed family presence. Splits are deterministic and **group-disjoint** (~60/15/25 by family), so no template family ever crosses a split boundary.
- **Predictions** (JSONL): `{"id", "label"}` for the test split, produced by the baseline predictor and the candidate. Coverage must be exact — missing or extra ids fail closed.
- **Metrics:** per-class precision/recall/F1, macro-F1, accuracy, and `critical_false_negative_rate` over records flagged `critical` (e.g. reopen/escalate classes that must never be silently coded benign). Runs with zero critical test cases report `null` and block promotion — coverage cannot be assumed.

Commands:
```sh
PYTHONPATH=src python3 -m hardonian_foundry.cli refinery validate-dataset --task-spec spec.json
PYTHONPATH=src python3 -m hardonian_foundry.cli refinery run --task-spec spec.json \
  --baseline-predictions baseline.jsonl --model-predictions model.jsonl --output-root .foundry/runs
```

### 7. Signing & Keyring (`keygen`, `attach-license`, `sign`)
Promotion attestations are Ed25519 signatures over the run digest and ledger head (`signing.py`):
- `keygen` creates a new private key file (refuses to overwrite; file mode forced to 0600).
- `attach-license <run_id> --manifest m.json` binds the SPDX allowlist decision into the ledger (once per run).
- Sign only after an allowed license is present in the run record or attached to its ledger. A prior legacy or premature signature cannot be upgraded or reused, and attaching a license after signing does not authorize promotion. Recovery: recreate a new run from the original inputs, supply or attach an allowed license, then sign the new run and promote with a trusted keyring. Existing runs and signatures remain unchanged for audit.
- `sign <run_id> --key-file k.pem` appends a `promotion_signature` entry covering every ledger entry up to that head (once per run).
- `promote --keyring keyring.json` verifies the signature against trusted public keys (64-char lowercase hex). Later unrelated ledger appends are permitted; license evidence used for promotion must precede the signature.

Legacy signatures do not bind the complete run and are blocked, not grandfathered. Signing is currently once per run: if an old signature exists, or a license was attached after signing, create and sign a new run with the license already present. The ledger alone does not prevent rollback without an external checkpoint. Signing attests the supplied evidence, not the actual model weights, predictor execution, or truth of a license claim.

Security posture: the optional `cryptography` backend is **fail-closed** — without the `signing` extra, `sign` errors out and `promote` reports `cryptographic_promotion_signature` as missing evidence. Private key material is file-based, mode 0600, and never logged or embedded into run records.

```sh
PYTHONPATH=src python3 -m hardonian_foundry.cli refinery keygen --key-file sign.pem
PYTHONPATH=src python3 -m hardonian_foundry.cli refinery attach-license <run_id> --manifest manifest.json --output-root .foundry/runs
PYTHONPATH=src python3 -m hardonian_foundry.cli refinery sign <run_id> --key-file sign.pem --output-root .foundry/runs
```

---

## Remaining Gaps and Future Work

- **Trained Model Registry:** Durable, encrypted artifact store for validated model checkpoints and LoRA adapters.
- **Live Inference Gateway:** Sandboxed, authenticated model router with timeout and circuit-breaking controls.
- **Adversarial Evaluation:** Expanded challenge sets beyond template-generated synthetic cases.

Completed in 0.2.0: cryptographic signing (Ed25519 over ledger heads + trusted keyring), licensing-manifest and benchmark promotion gates.

