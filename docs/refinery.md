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
- **Benchmarks:** Evaluation metrics present, `critical_false_negative_rate == 0.0`, `review_rate == 1.0`, `f1 >= 0.8`.
- **License Allowlist:** Model manifest present and SPDX approved.
- **Signed Promotion:** Cryptographic signing authority token.

When required evidence is missing, the gate reports the exact missing items in `missing_evidence` and exits with code 2. Because cryptographic promotion signatures are not yet implemented in Foundry, promotion always fails closed.

Command:
```sh
PYTHONPATH=src python3 -m hardonian_foundry.cli refinery promote <run_id> --output-root .foundry/runs
```

---

## Remaining Gaps and Future Work

- **Cryptographic Signing:** Digital signatures (e.g. Sigstore / Ed25519) for evidence chain heads and promotion attestations.
- **Trained Model Registry:** Durable, encrypted artifact store for validated model checkpoints and LoRA adapters.
- **Live Inference Gateway:** Sandboxed, authenticated model router with timeout and circuit-breaking controls.
- **Adversarial Evaluation:** Expanded challenge sets beyond template-generated synthetic cases.
