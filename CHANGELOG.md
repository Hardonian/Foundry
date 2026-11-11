# Changelog

All notable changes to Foundry are documented here. The format is loosely
based on Keep a Changelog; this project uses Semantic Versioning.

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
