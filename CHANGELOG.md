# Changelog

All notable changes to Foundry are documented here. The format is loosely
based on Keep a Changelog; this project uses Semantic Versioning.

## [0.3.0] - 2026-11-12

Cross-platform Windows/macOS/Linux hardening, modular Refinery architecture,
external rollback anchoring, M-of-N keyring quorum, batched offline training
utilities, and two new zero-weight runnable case studies.

### Added
- **Core Ledger & Canonical Serialization:**
  - `canonical.py`: single-pass canonical JSON serializer (`canonical`) and SHA-256 digest (`digest`) rejecting non-string keys, circular refs, and `NaN`/`Infinity`.
  - `evidence.py`: `EvidenceStore` protocol, `JsonlEvidenceStore`, WAL-backed `SQLiteEvidenceStore`, `checkpoint()` helper, and `allowed_origins` policy gating (defaulting strictly to `("synthetic",)`).
  - `signing.py`: `parse_keyring` / `load_keyring_policy` with optional `key_id`, `role`, `revoked_at`, `required_signatures` M-of-N quorum, and `required_roles`.
  - `licensing.py`: bounded `notes` validation (`<= 1024` chars), local `verify_weights_path` SHA-256 verification, and `consented_pilot` / `redacted_eval` governance tiers gated on explicit `governance_attestation`.
  - `tenancy.py`: fail-closed `authorize_run` and `scope_ledger_query` helpers.
- **Refinery, Evaluation, Artifacts & CLI:**
  - Modularized `refinery.py` facade backed by `refinery_validation.py`, `refinery_preflight.py`, `refinery_runs.py`, `refinery_gates.py`, `refinery_attestation.py`, `types.py`, and `artifact.py` (content-addressed local store).
  - `refinery` CLI commands & flags: `list` (`--status`, `--task-id`, `--verify`), `diff`, `anchor`, `export-bundle` (`--format`), `artifact {add,verify,list}` (`--store-dir`), `preflight --task-spec`, `--expected-head`, `--expected-count`, `--anchor`, `--tenant-id`, `--request-id`, `--ood-predictions`, `--version`, `doctor --verbose`, `--json-errors`, `provider predict`, and `registry promote --keyring`.
  - `task_eval.py`: single-pass `O(|T|)` scoring, confusion matrix, per-family breakdown, optional `confidence` calibration (`ece`, `brier`, abstention), `latency_ms` / `latency_p95_ms`, optional thresholds, bootstrap CI, and McNemar significance helpers.
- **Training & Case Studies:**
  - `training_utils.py`: shared offline snapshot/Git-blob verification, `auto_map` rejection, cross-platform `verify_private_dir` & `available_ram_bytes`, cooperative locking, JSON model registry config, padded micro-batching, and `sft`/`qlora`/`dpo`/`rlvr` adapter hooks.
  - `examples/case-study-exception-codes/generate_dataset.py`: `--profile adversarial-noisy`, `--profile imbalanced` (4% critical), and `--seed` while preserving exact `standard` parity.
  - `examples/case-study-agent-tool-gate/`: offline case study #2 (`ALLOW_PROPOSAL`, `DENY`, `ESCALATE_REVIEW`) with prompt injection, path traversal, and argument smuggling.
  - `examples/case-study-reconciliation-policy/`: offline case study #3 integrating `synthetic_policy.assess` / `assess_with_suggestion` to override untrusted model suggestions deterministically.
  - Documentation: [docs/dataset-card.md](docs/dataset-card.md), expanded [docs/architecture.md](docs/architecture.md) (Mermaid diagrams & threat model), updated [docs/refinery.md](docs/refinery.md) and [docs/training.md](docs/training.md).
- **Packaging & CI:**
  - `pyproject.toml` extras (`dev`, `train`, `all`) plus `[tool.ruff]`, `[tool.pytest.ini_options]`, `[tool.mypy]`, and `[tool.coverage]` configuration.
  - Multi-OS × Python (`ubuntu-latest`, `windows-latest`, `macos-latest` × `3.11`, `3.12`, `3.13`) CI matrix, `mypy src`, `pytest --cov`, `no-signing-extra` fail-closed job, wheel install smoke, cloned case-study replay smoke, and expanded `bandit`.

### Changed
- **Performance & Portability:** Replaced `O(N^2)` ledger rewrite on append with crash-safe tail append; replaced per-call `python -I` subprocess spawns in `provider.py` with deadline-enforced socket/HTTP connection handling, circuit breaker, and structured `UNAVAILABLE` codes; stored portable relative `"evidence.jsonl"` on new runs while resolving `root_dir / "evidence.jsonl"` safely for cloned legacy runs.
- **Training Scripts:** Fixed `grad_accum` trailing remainder step and silent truncation in `train_lora.py`, fixed `to_label()` prefix collisions and added batched inference in `model_predict.py`, and fixed split range leaks plus added validation monitoring in `experiment_balanced_lora.py`.

### Fixed
- **Windows Compatibility:** Fixed `signing.py` POSIX-only `0o077` permission checks, `evidence.py` `msvcrt.LK_LOCK` retry limits, `training_utils.py` `os.getuid()` / `os.sysconf` crashes, and `task_eval.validate_spec` Windows traversal (`\`, drive letters).
- **Security Hardening:** Verified `run.json` against seq-1 `run_record_sha256` before `sign_run`; strictly rejected traversal in `run_id`; fixed `docs/refinery.md` `keygen --out` flag reference; replaced `time.monotonic()` in `run_id` generation with `secrets.token_hex(6)`.

### Known Gaps / Boundaries
- Synthetic datasets remain the default; `consented_pilot` / `redacted_eval` origins require explicit `governance_attestation` and human review (`ALLOWED_REVIEW_REQUIRED` never auto-deploys).
- Remote KMS/HSM signing, write-once remote anchoring, and production serving gateways remain external to Foundry.

## [0.2.1] - 2026-11-10

### Security
- Fix promotion evidence validation: version 2 signatures bind the complete
  canonical run record and the ledger head at signing, including prior license
  evidence. A trusted keyring is required for signature verification.
- **The v0.2.0 release remains vulnerable:** its promotion gate can accept
  inadequately bound evidence. Do not rely on its promotion decisions; upgrade
  and reassess them under v0.2.1.
- Legacy signatures are blocked, not upgraded. Signing is once per run: create
  a new run, supply or attach an allowed license before signing, and sign the
  new run before promoting it.
- Signatures bind supplied evidence, not actual model weights, predictor
  execution, or the truth of license claims. The ledger alone cannot prevent
  rollback; detection requires a trusted external head/checkpoint.

## [0.2.0] - 2026-11-10

**Security warning:** v0.2.0 promotion is vulnerable; see v0.2.1 above. The
following describes the historical release, not a security guarantee.

Task-mode evaluation and cryptographic promotion signing. The promotion gate
can now say ALLOWED — but only when every piece of evidence verifies, and the
decision is still `ALLOWED_REVIEW_REQUIRED` (a human signs off; nothing
auto-promotes).

### Added
- Task-mode evaluation (`task_eval.py`): frozen task-spec contracts, JSONL
  datasets/predictions, deterministic group-disjoint 60/15/25 splits,
  per-class P/R/F1 + macro-F1 + critical-false-negative scoring
- `refinery validate-dataset --task-spec` and
  `refinery run --task-spec --baseline-predictions --model-predictions`
- Ed25519 promotion signing (`signing.py`): `refinery keygen`,
  `refinery sign`, `refinery promote --keyring`; the original signature
  binding was incomplete, as corrected in v0.2.1
- `refinery attach-license` binds the SPDX allowlist decision into the ledger
- Optional `signing` extra (`pip install hardonian-foundry[signing]`);
  without it, signing fails closed and promotion reports the missing evidence
- Task-mode benchmark gate: macro-F1 floor (>= 0.8 enforced minimum) plus
  strict baseline improvement when `baseline_must_improve` is set
- 23 new tests (109 total), including tamper-after-sign, untrusted-key,
  no-improvement, denied-license and world-readable-key rejection paths

### Changed
- Promotion gate: `ALLOWED_REVIEW_REQUIRED` (exit 0) replaces always-BLOCKED
  when all evidence verifies; any missing/unverified item still exits 2 with
  exact `missing_evidence` reasons
- CI installs `cryptography==50.0.2`; ruff coverage extended to the new modules

## [0.1.0] - 2026-11-10

First public baseline release. Everything in this version is offline,
CPU-only, and synthetic; see README "Honest status".

### Added
- `refinery` workflow: `validate-dataset`, `preflight`, `run`, `inspect`, `promote`
- Hash-chained, tamper-evident run ledger (`evidence.jsonl`) with per-run provenance
- Fail-closed promotion gate with explicit missing-evidence reasons (always BLOCKED at 0.1.0, by design)
- Deterministic synthetic baseline (`demo`), environment health check (`doctor`)
- Opt-in synthetic-only OpenAI-compatible loopback provider adapter (`provider`)
- Offline CPU LoRA training smoke with pinned, bounded preflight (docs/training.md)
- Cross-platform `fcntl` guards for evidence and training scripts
- `foundry` console script entry point
- CI (`offline-baseline`): 86 unit tests, synthetic smoke, refinery smoke, compileall,
  ruff, bandit, pip-audit, gitleaks — all actions pinned to full commit SHAs

### Known gaps (intentionally loud, not hidden)
- No signed provenance, licensing-manifest or benchmark promotion gates yet
  → `refinery promote` fails closed with the exact missing artifacts listed
- No production API, tenant isolation, inference server or model registry
