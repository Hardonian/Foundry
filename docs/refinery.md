# Developer Refinery Workflow

The Refinery workflow connects dataset validation, training preflight, reproducible evaluation, tamper-evident run tracking, external head anchoring, and signed model promotion gating into a unified developer workflow.

## Guiding Principles

- **Synthetic by Default:** Built-in fixtures and case studies use synthetic data only (`data_origin: "synthetic"`). Optional `consented_pilot` / `redacted_eval` origins require explicit `governance_attestation` and remain disallowed by default.
- **Fail-Closed Security:** Incomplete evidence, broken hash chains, untrusted or revoked licenses/keys, missed M-of-N quorums, anchor mismatches, or missing cryptographic signatures block promotion with exit code `2`.
- **Review Required:** All evaluation decisions, heuristic classifications, and model suggestions require human review (`review_rate = 1.0`, `coverage_automatic_decision = 0.0`, `financial_write = false`, `executed = false`).
- **Tamper Evidence & Rollback Detection:** Provenance is anchored into an append-only hash-chained ledger (`evidence.jsonl` or SQLite WAL store) and verified against an external head checkpoint (`refinery anchor` / `--expected-head`).

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
                    │    evidence.jsonl / DB  │
                    │  (hash-chained ledger)  │
                    └────────────┬────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
 ┌─────────────────────┐ ┌──────────────────┐ ┌─────────────────────┐
 │   inspect <run_id>  │ │  anchor <run_id> │ │   promote <run_id>  │
 │(verify provenance)  │ │(checkpoint head) │ │ (evaluate evidence) │
 └─────────────────────┘ └──────────────────┘ └──────────┬──────────┘
                                                         │
                                     ┌───────────────────┴───────────────────┐
                                     ▼                                       ▼
                        [BLOCKED: exit 2]                       [ALLOWED_REVIEW_REQUIRED: exit 0]
```

### 1. Dataset Validation (`validate-dataset`)
Validates integer minor units, synthetic ID prefixes (`synthetic-`), maximum ID lengths (`<= 100`), absence of duplicate IDs, and absence of identical content across differing IDs (leakage hazard). In `--task-spec` mode, validates the frozen spec contract, traversal-free relative `dataset_path` (POSIX and Windows paths), label coverage, and deterministic group-disjoint splits.

```sh
foundry refinery validate-dataset [--data <path>]
foundry refinery validate-dataset --task-spec <spec.json>
```

### 2. Training Preflight (`preflight`)
Executes diagnostic checks prior to executing a run or training task:
- Validates the target built-in/custom dataset or `--task-spec` task dataset.
- Verifies Python runtime (`>= 3.11`) and offline CPU mode.
- Validates the output directory (mode `0700` on POSIX, owner/ACL check on Windows) and storage headroom.
- Evaluates the model licensing manifest against the SPDX allowlist (`Apache-2.0`, `MIT`, `BSD-3-Clause`, `CC0-1.0`, `OFL-1.1`), provenance, and `data_origin` tier.
- Asserts execution safety boundaries (no autonomous financial/tool execution or remote code).

```sh
foundry refinery preflight [--data <path>] [--task-spec <spec.json>] [--output-root <dir>] [--manifest <path>]
```

### 3. End-to-End Run Execution (`run`)
Performs preflight checks, executes reproducible held-out evaluation (or scores external task-mode predictions), creates a private run directory (`run-<timestamp>-<random_hex>`), stores a portable relative `"evidence.jsonl"` ledger path in `run.json`, records `run_record_sha256` into the tamper-evident evidence ledger, and writes `run.json`.

```sh
foundry refinery run [--output-root .foundry/runs] [--data <path>] [--manifest <path>] [--seed 42] [--tenant-id <id>] [--request-id <id>]
foundry refinery run --task-spec <spec.json> --baseline-predictions <baseline.jsonl> --model-predictions <model.jsonl> --output-root .foundry/runs [--ood-predictions <ood.jsonl>] [--tenant-id <id>] [--request-id <id>]
```

### 4. Run Inspection, Listing & Diffing (`inspect`, `list`, `diff`)
- **`inspect`:** Replays the ledger hash chain from `GENESIS`, safely resolves `root_dir / "evidence.jsonl"` when inspecting cloned runs (rejecting traversal or paths outside `root_dir`), and optionally verifies external rollback pins (`--expected-head`, `--expected-count`) or tenant context (`--tenant-id`, `--request-id`).
- **`list`:** Lists recorded runs under `--output-root`, with optional status/task filtering and inline chain verification.
- **`diff`:** Compares metrics, thresholds, dataset hashes, and provenance between two run IDs.

```sh
foundry refinery inspect <run_id> --output-root .foundry/runs [--expected-head <hex>] [--expected-count <n>] [--tenant-id <id>] [--request-id <id>]
foundry refinery list --output-root .foundry/runs [--status <status>] [--task-id <id>] [--verify]
foundry refinery diff <run_a> <run_b> --output-root .foundry/runs
```

### 5. External Anchoring & Bundle Export (`anchor`, `export-bundle`)
Because a local append-only ledger cannot by itself detect an attacker rolling back the entire directory to an earlier signed snapshot, Foundry provides external head anchoring and portable bundle export:

```sh
foundry refinery anchor [run_id] --output-root .foundry/runs [--out anchor.json]
foundry refinery export-bundle <run_id> --output-root .foundry/runs --out bundle.tar.gz [--format dir|zip|tar|tar.gz|tgz]
```

Pass the recorded anchor digest to `inspect`, `promote`, or `registry promote` via `--expected-head <hex>` (and `--expected-count <n>`) or `--anchor <anchor.json>` to fail closed on rollback.

### 6. Content-Addressed Artifacts (`artifact`)
Manages local content-addressed artifacts with SHA-256 verification:

```sh
foundry refinery artifact add <file> [--store-dir <dir>] [--name <label>] [--kind <type>] [--run-id <run_id>]
foundry refinery artifact verify <target> [--expected-sha256 <hex>] [--store-dir <dir>]
foundry refinery artifact list [--store-dir <dir>] [--run-id <run_id>]
```

### 7. Promotion Gate (`promote`)
Evaluates whether a candidate model run possesses all required evidence for promotion:
- **Run Metadata:** `run.json` exists, parses cleanly, and `run_id` passes strict traversal checks.
- **Provenance Proof:** Evidence ledger verifies with `PROVENANCE_VERIFIED` (and matches `--expected-head` / `--expected-count` / `--anchor` when supplied).
- **Preflight:** Training preflight passed cleanly.
- **Benchmarks:** Evaluation metrics present, `critical_false_negative_rate == 0.0`, `review_rate == 1.0`, `f1 >= 0.8`. For task-mode runs, each contract label must have positive test support; `macro_f1` must clear `thresholds.macro_f1_min` (`>= 0.8`) and strictly beat the recorded baseline when `baseline_must_improve` is true (plus any optional thresholds configured on the spec).
- **License Allowlist:** The SPDX-approved manifest must be in the signed run record or attached to the ledger **before** signing.
- **Signed Promotion & Keyring Quorum:** A version 2 Ed25519 signature binds the complete canonical run record and ledger head, verified against a trusted keyring supplied with `--keyring` (enforcing non-revoked keys and optional `required_signatures` M-of-N quorum / `required_roles`).

When any required evidence is missing or unverified, the gate reports `missing_evidence` and exits with code `2`. When everything verifies, the decision is `ALLOWED_REVIEW_REQUIRED` (exit `0`) — a human still signs off; the gate never auto-promotes.

```sh
foundry refinery promote <run_id> --output-root .foundry/runs --keyring keyring.json [--expected-head <hex>] [--expected-count <n>] [--anchor <anchor.json>] [--tenant-id <id>] [--request-id <id>]
```

### 8. Signing & Keyring (`keygen`, `attach-license`, `sign`)
Promotion attestations are Ed25519 signatures over the canonical run digest and ledger head (`signing.py`):
- `keygen --out <path>` (also accepts `--key-file <path>` alias) creates a new Ed25519 private key file (refuses to overwrite; mode forced to `0600` on POSIX, protected owner-only DACL on Windows).
- `attach-license <run_id> --manifest m.json` binds the SPDX allowlist decision into the ledger (once per run).
- `sign <run_id> --key-file k.pem [--cosign]` verifies `run.json` matches the seq-1 ledger `run_record_sha256` before appending a `promotion_signature` entry covering the ledger head.
- Sign only after an allowed license is present in the run record or attached to its ledger. Legacy `v1` or premature signatures cannot be upgraded; recreate a new run with the license attached before signing.

```sh
foundry refinery keygen --out sign.pem
foundry refinery attach-license <run_id> --manifest manifest.json --output-root .foundry/runs
foundry refinery sign <run_id> --key-file sign.pem --output-root .foundry/runs
```

### 9. Task-Mode Evaluation (`task_eval.py`)
Task mode scores externally produced predictions against a frozen task contract (`spec.json`) without executing models inside Foundry:
- **Spec:** `task_id`, `labels`, `thresholds` (`macro_f1_min >= 0.8` plus optional `critical_fn_rate_max`, `per_class_recall_min`, `invalid_output_rate_max`, `latency_p95_ms_max`), `baseline_must_improve`, `split_seed`, `model` (`base_model_id`, `revision`, `adapter_sha256`, `adapter_name`), and relative `dataset_path`.
- **Dataset (JSONL):** `{"id", "prompt", "gold_label", "critical", "family"?}` with deterministic group-disjoint ~60/15/25 splits.
- **Predictions (JSONL):** `{"id", "label"}` (with optional `"confidence"`, `"latency_ms"`, `"abstained"`, `"raw_invalid"`) with exact test-ID coverage.
- **Metrics:** Single-pass `O(|T|)` scoring emitting per-class P/R/F1, `macro_f1`, `accuracy`, `confusion_matrix`, `critical_false_negative_rate` (`null` when `critical_total == 0`, blocking promotion), and optional calibration/abstention metrics.

### 10. Opt-in Provider Adapter (`provider`, `provider predict`)
- **`foundry provider --origin <url> --model <id>`:** Single-record loopback probe (`probe`) with redirects/proxies disabled, response byte cap, circuit breaker, and strict total deadline.
- **`foundry provider predict ...`:** Batch prediction generator writing `{id, label}` JSONL with fail-closed `--fallback-label UNDETERMINED` / `UNAVAILABLE` handling (never substituting heuristics on provider errors). Supports `--task-spec`, `--data`, `--labels`, `--split`, `--out`, `--failure-threshold`, plus opt-in `--allow-tls`, `--allow-remote`, `--ca-bundle`, and `--auth-token-env`.

### 11. Registry Gate (`registry list`, `registry promote`)
Manages registered model candidates backed by verified refinery evidence:

```sh
foundry registry list --output-root .foundry/runs
foundry registry promote [model_id] --output-root .foundry/runs --keyring keyring.json [--expected-head <hex>] [--expected-count <n>] [--tenant-id <id>] [--request-id <id>]
```

---

## Exit Codes & Global Flags

| Command / Outcome | Exit Code | Meaning |
|---|---:|---|
| Success (`doctor`, `demo`, `run`, `inspect` verified, `promote` `ALLOWED_REVIEW_REQUIRED`, `validate-dataset` `VALID`, `preflight` `PASSED`, `attach-license` `ALLOWED`, `sign`, `keygen`, `anchor`, `export-bundle`) | `0` | Completed and verified |
| Validation / Preflight / Inspection / Provider Unavailable failure | `1` | Invalid dataset, failed preflight, tampered ledger, or provider `UNAVAILABLE` |
| Fail-closed Gate / Configuration / CLI Input Error (`promote` `BLOCKED`, `registry promote`, invalid args, traversal/missing files, failed anchor/quorum) | `2` | Promotion blocked or fail-closed safety/input error |

Global CLI utilities:
- `foundry --version` — print package version
- `foundry --json-errors ...` — emit structured JSON error objects on stderr
- `foundry doctor [--verbose]` — offline runtime and optional-extra diagnostics (`signing`, `train` availability)

---

## Cross-Platform Notes (Windows & POSIX)

- **File Locking:** `evidence.py` uses `fcntl.flock` on Linux/macOS and bounded retry/backoff over `msvcrt.LK_NBLCK` on Windows.
- **Key & Directory Permissions:** On POSIX (`os.name == "posix"`), private keys require strict `0600` mode (`st_mode & 0o077 == 0`) and run directories use `0700`. On Windows (`os.name == "nt"`), `generate_keypair()` applies a protected owner-only DACL (owner + `SYSTEM` + `Administrators` via `SetNamedSecurityInfoW`), and `GetNamedSecurityInfoW` rejects NULL DACLs or inherited broad SIDs (`Everyone`, `BUILTIN\Users`) without breaking temporary-directory workflows.
- **Portable Ledgers:** Cloned runs such as `examples/case-study-exception-codes/task/runs/run-20261111T031107Z-6a0895d1` verify cleanly across Windows, macOS, and Linux because `inspect` resolves `root_dir / "evidence.jsonl"` safely without modifying committed signed evidence.
- **Path Traversal Prevention:** `task_eval.validate_spec` and `refinery` reject `/`, `\`, `..`, and Windows drive letters (`C:`) in `run_id` and `dataset_path`.

---

## Remaining Gaps and Future Work

Completed in `0.3.0`: external head anchoring (`anchor`, `--expected-head`), M-of-N keyring quorum & revocation, SQLite WAL evidence store, content-addressed local artifact store (`artifact.py`), `list`/`diff`/`export-bundle` commands, cross-platform Windows/macOS/Linux support, and three end-to-end synthetic case studies.

Remaining production gaps:
- **Remote KMS / HSM & Immutable Audit Service:** File-based PEM keys and local anchors rely on operator custody; production requires KMS/HSM signing and write-once remote anchoring.
- **Live Inference Serving:** Foundry gates evaluation evidence; production routing remains separate (`model-tools`).
- **Consented Pilot Calibration:** Real-world pilot accuracy requires multi-rater human calibration on consented, redacted records per [docs/dataset-card.md](dataset-card.md).
