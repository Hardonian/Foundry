# Foundry

[![CI](https://github.com/Hardonian/Foundry/actions/workflows/ci.yml/badge.svg)](https://github.com/Hardonian/Foundry/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-green)](pyproject.toml)
[![Status: research baseline](https://img.shields.io/badge/status-research%20baseline-orange)](#honest-status)

**Evidence-first model lifecycle for AI agents.** Every evaluation run becomes a tamper-evident record, and model promotion **fails closed** until the evidence actually justifies it. No model gets promoted on vibes.

![Refinery: validate, preflight, run, inspect, promote](docs/assets/refinery-demo.gif)

## The problem

AI pipelines treat model promotion as a vibe check: the loss curve looked fine, the demo worked, so the artifact shipped. When a model makes decisions — exception triage, tool calls, money-adjacent judgments — nobody can show *what evidence* justified deploying it.

Foundry makes that evidence the gate. Dataset validation, preflight safety/licensing/resource checks, hash-chained run records, and a promotion verdict that refuses to say yes without required evidence artifacts.

Promotion is the interesting output. Missing or invalid evidence blocks it;
eligible task-mode runs can produce `ALLOWED_REVIEW_REQUIRED`, not automatic
promotion. See [the refinery documentation](docs/refinery.md) for the gates.

**Security:** v0.2.0's promotion gate is vulnerable; upgrade to v0.2.1 and
reassess its promotion decisions. Legacy signatures are blocked: create a new
run, supply or attach an allowed license before signing, then sign and promote
the new run. A signature binds supplied evidence, not actual model weights,
predictor execution, or the truth of a license claim. Detecting ledger rollback
requires a trusted external head/checkpoint. See the [changelog](CHANGELOG.md).

## What works today

Every command below is verified by CI (`offline-baseline`) on every push:

| Command | What it does |
|---|---|
| `foundry doctor` | Environment and self-health check |
| `foundry demo` | Deterministic synthetic baseline run |
| `foundry refinery validate-dataset` | Integrity, uniqueness and splitability checks on the synthetic dataset |
| `foundry refinery preflight` | Safety, resource and licensing checks before any run |
| `foundry refinery run` | End-to-end evaluation with a hash-chained, tamper-evident run record |
| `foundry refinery inspect <run_id>` | Re-verify a run against the evidence ledger |
| `foundry refinery promote <run_id>` | Fail-closed promotion verdict with explicit missing-evidence reasons |

The baseline runs offline on CPU without network calls.

## Quick start

```sh
git clone https://github.com/Hardonian/Foundry && cd Foundry
python3 -m venv .venv && .venv/bin/pip install .
.venv/bin/foundry doctor
.venv/bin/foundry refinery run --output-root .foundry/runs
.venv/bin/foundry refinery promote <run_id> --output-root .foundry/runs   # exits 2, by design
```

Source-tree mode without install: `PYTHONPATH=src python3 -m hardonian_foundry.cli <command>`.
The package name `hardonian-foundry` is reserved for PyPI publication. Check the
[release version](https://github.com/Hardonian/Foundry/releases) before installing
a wheel; v0.2.0 remains vulnerable and may still be the latest published release.

## Documentation

| Document | Covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Component boundaries and lifecycle contracts |
| [docs/refinery.md](docs/refinery.md) | The refinery workflow: inputs, outputs, evidence format, gate semantics |
| [docs/training.md](docs/training.md) | Bounded offline CPU LoRA smoke: preflight, restrictions, limitations |

## Honest status

Read this before building on Foundry:

- This is an **early, offline, CPU-only research baseline**. The deterministic heuristics are **not a trained model**, are **not commercially validated**, and process **synthetic inputs only**.
- **Do not deploy** for real financial data or autonomous tool execution. No inference server, production training system, authenticated tenant isolation, production API, production-grade evidence store, or usable model registry is implemented yet.
- `registry promote` **always fails closed**. Outputs are suggestions only and always require human review.
- The opt-in OpenAI-compatible provider adapter must point at an **operator-approved loopback server** and sends synthetic samples only. Errors return `UNAVAILABLE` with nonzero exit and **no heuristic substitution**; a successful response is always `review-required`. It is not verified against a live model-tools deployment (the inspected `app.main` returns HTTP 501 for `/v1/chat/completions`). No real provider call is made by tests.
- Evidence hashes are **metadata**, not encrypted or tamper-proof storage.
- Synthetic self-consistency is **not pilot accuracy**. The synthetic dataset is constructed programmatically; its split is seed-stable within the documented Python runtime.
- No model weights or customer data are included in this project.
- The first worked case study — a real fine-tuned open-weight model promoted through the refinery with a public evidence bundle — is the next milestone. See the roadmap in issues #1–#3.

## Why trust the boring parts

Every claim in this README is a command you can run in 60 seconds, and CI runs the full refinery pipeline offline on every push. Gates fail closed; errors are loud; nothing silently falls back.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Start with the [`good first issue` labels](https://github.com/Hardonian/Foundry/labels/good%20first%20issue). Security issues: see [SECURITY.md](SECURITY.md) — no GitHub issues for vulnerabilities.

## License

Apache-2.0. See [LICENSE](LICENSE).
