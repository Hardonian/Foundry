# Changelog

All notable changes to Foundry are documented here. The format is loosely
based on Keep a Changelog; this project uses Semantic Versioning.

## [0.2.0] - 2026-11-10

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
  `refinery sign`, `refinery promote --keyring`; signatures bind to the exact
  ledger head, so any post-signing mutation blocks promotion
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
