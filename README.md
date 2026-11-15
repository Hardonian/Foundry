# Foundry

[![CI](https://github.com/Hardonian/Foundry/actions/workflows/ci.yml/badge.svg)](https://github.com/Hardonian/Foundry/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-green)](pyproject.toml)
[![Status: research baseline](https://img.shields.io/badge/status-research%20baseline-orange)](#honest-status)

**Evidence-first model lifecycle for AI agents.** Every evaluation run becomes a tamper-evident record, and model promotion **fails closed** until the evidence actually justifies it. No model gets promoted on vibes.

![Refinery: validate, preflight, run, inspect, promote](docs/assets/refinery-demo.gif)

## The problem

AI pipelines treat model promotion as a vibe check: the loss curve looked fine, the demo worked, so the artifact shipped. When a model makes decisions — exception triage, tool calls, money-adjacent judgments — nobody can show *what evidence* justified deploying it.

Foundry makes that evidence the gate. Dataset validation, preflight safety/licensing/resource checks, hash-chained run records, external head anchoring, M-of-N Ed25519 keyring verification, and a promotion verdict that refuses to say yes without complete, untampered evidence.

Promotion is the interesting output. Missing or invalid evidence blocks it (`BLOCKED`, exit `2`); eligible runs with complete signed evidence produce `ALLOWED_REVIEW_REQUIRED` (exit `0`), where a human still signs off. See [the refinery documentation](docs/refinery.md) for the full gate specification.

**Security:** v0.2.0's promotion gate is vulnerable; upgrade to v0.2.1+ / v0.3.0 and reassess its promotion decisions. Legacy `v1` signatures are blocked: create a new run, supply or attach an allowed license before signing, then sign, anchor, and promote the new run. A signature binds supplied evidence, not actual model weights, predictor execution, or the truth of a license claim. Detecting ledger rollback requires a trusted external head/checkpoint (`--expected-head` / `refinery anchor`). See [CHANGELOG.md](CHANGELOG.md).

## What works today

Every command below is cross-platform (`ubuntu-latest`, `windows-latest`, `macos-latest`) and verified offline on every push:

| Command | What it does |
|---|---|
| `foundry --version` / `doctor [--verbose]` | Package version and offline environment/extra diagnostics |
| `foundry demo` / `eval` / `policy-demo` | Deterministic synthetic reconciliation & tool-policy demonstrations |
| `foundry refinery validate-dataset` | Integrity, uniqueness, leakage, and splitability checks (`--data` or `--task-spec`) |
| `foundry refinery preflight` | Safety, runtime, storage, and licensing checks (`--task-spec`, `--manifest`) |
| `foundry refinery run` | End-to-end evaluation with hash-chained ledger and portable `run.json` |
| `foundry refinery inspect <run_id>` | Re-verify run provenance from genesis (plus `--expected-head` / `--expected-count`) |
| `foundry refinery promote <run_id>` | Fail-closed promotion gate with `--keyring` (M-of-N quorum/revocation) & anchor checks |
| `foundry refinery attach-license` | Bind SPDX allowlist and data-origin decision into the ledger before signing |
| `foundry refinery keygen --out <pem>` | Generate an Ed25519 private key (mode `0600` on POSIX, owner-restricted on Windows) |
| `foundry refinery sign <run_id>` | Verify `run.json` against seq-1 digest and append `v2` Ed25519 signature |
| `foundry refinery list` / `diff` | List tracked runs and diff metrics/provenance between runs |
| `foundry refinery anchor` / `export-bundle` / `verify-bundle` | Emit external/WORM head checkpoints and export/verify SHA-256 evidence bundles |
| `foundry refinery report` / `compare` | Render JSON/Markdown compliance audit reports and McNemar + bootstrap model comparisons |
| `foundry refinery proofpack` / `verify-proofpack` | Build and verify SHA-256-sealed read-only `Proofpack v2` contracts for downstream consumers |
| `foundry refinery artifact {add,verify,list,attest,verify-attestation}` | Local content-addressed artifact store and SLSA/in-toto build attestations |
| `foundry provider` / `provider predict` | Opt-in loopback provider probe & batch JSONL predictions (fails closed to `UNAVAILABLE`) |
| `foundry registry {list,promote,register,get,verify}` | Fail-closed SHA-256 model registry & promotion gate requiring `--keyring` and verified evidence |

## Quick start

```sh
git clone https://github.com/Hardonian/Foundry && cd Foundry
python3 -m venv .venv && .venv/bin/pip install ".[signing]"
.venv/bin/foundry doctor --verbose
.venv/bin/foundry refinery run --output-root .foundry/runs
.venv/bin/foundry refinery promote <run_id> --output-root .foundry/runs   # exits 2, by design
```

Source-tree mode without install: `PYTHONPATH=src python3 -m hardonian_foundry <command>` (or `$env:PYTHONPATH="src"` on Windows PowerShell).

Python SDK (`import hardonian_foundry`):

```python
import hardonian_foundry as hf

res = hf.evaluate_promotion("<run_id>", output_root=".foundry/runs", keyring="keyring.json")
report = hf.generate_audit_report("<run_id>", output_root=".foundry/runs", fmt="markdown")
```

Optional extras in [pyproject.toml](pyproject.toml):
- `.[signing]` — Ed25519 signing/verification (`cryptography`)
- `.[dev]` — signing + `pytest`, `pytest-cov`, `ruff`, `mypy`, `bandit`, `build`, `pillow`, `pyyaml`
- `.[train]` — offline LoRA training stack (`torch`, `transformers`, `peft`, `accelerate`, `safetensors`)
- `.[all]` — all optional dependencies

## Worked Case Studies

| Case Study | Offline Weights Needed? | What It Demonstrates |
|---|---|---|
| [examples/case-study-exception-codes](examples/case-study-exception-codes/README.md) | Optional (committed signed evidence replays offline) | Qwen2.5-Coder-0.5B LoRA vs Settler keyword regex, plus `standard`, `adversarial-noisy`, and `imbalanced` (4% critical) dataset profiles |
| [examples/case-study-agent-tool-gate](examples/case-study-agent-tool-gate/README.md) | **None** (runs in seconds) | Agent tool-call safety (`ALLOW_PROPOSAL`, `DENY`, `ESCALATE_REVIEW`) against prompt injection, path traversal, and argument smuggling |
| [examples/case-study-reconciliation-policy](examples/case-study-reconciliation-policy/README.md) | **None** (runs in seconds) | `synthetic_policy.assess_with_suggestion` deterministic minor-unit arithmetic overriding untrusted model suggestions |

Replay the committed signed exception-codes evidence chain on any OS without modifying committed run files:

```sh
PYTHONPATH=src python3 -m hardonian_foundry.cli refinery inspect run-20261111T031107Z-6a0895d1 --output-root examples/case-study-exception-codes/task/runs
PYTHONPATH=src python3 -m hardonian_foundry.cli refinery promote run-20261111T031107Z-6a0895d1 --output-root examples/case-study-exception-codes/task/runs --keyring examples/case-study-exception-codes/task/keyring.json
```

## Documentation

| Document | Covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Mermaid architecture/sequence diagrams, module boundaries, threat model, `task_eval` / `signing` / `training_utils` contracts |
| [docs/refinery.md](docs/refinery.md) | Full refinery CLI reference, gate semantics, exit codes, anchoring, Windows/POSIX behavior |
| [docs/training.md](docs/training.md) | Cross-platform preflight, batched CPU LoRA training/eval, model registry config, DPO/RLVR/GPU adapter contracts |
| [docs/dataset-card.md](docs/dataset-card.md) | Dataset taxonomies, spurious-shortcut warnings, and consented pilot migration checklist |
| [docs/roadmap.md](docs/roadmap.md) | Long-term product roadmap (`v1.0.0` shipped through `v2.0` KMS/WORM/agentic/federated proofpacks) |

## Honest Status & Long-Term Roadmap (v1.0.0)

Read this before building on Foundry:

- **Synthetic-first baseline:** Built-in fixtures and case studies use **synthetic data only**. Optional `consented_pilot` / `redacted_eval` tiers in `licensing.py` and `evidence.py` require explicit `governance_attestation` and remain disallowed by default.
- **Human review is mandatory:** `ALLOWED_REVIEW_REQUIRED` certifies evidence completeness at a verified ledger head—never production accuracy or unattended deployment readiness (`review_rate = 1.0`, `financial_write = false`, `executed = false`).
- **No production serving or managed cloud KMS/TSA daemon in core package:** Foundry provides pluggable `SignerProvider` / `AnchorSink` (`WormDirectoryAnchorSink`) SPIs, SLSA/in-toto artifact attestations, agentic trajectory & `scan_dataset_pii` gates, `cohens_kappa` / `fleiss_kappa`, local SHA-256 `registry.json`, and frozen read-only `Proofpack v2` contracts. Live production inference gateways, IAM/SSO services, and vendor-specific cloud KMS/TSA adapters remain external SPI plugins tracked in [docs/roadmap.md](docs/roadmap.md).
- **Loopback provider guard:** The opt-in `provider` adapter defaults to operator-approved loopback origins (optional TLS/bearer env auth requires explicit opt-in), disables redirects/proxies, enforces timeouts/circuit breakers, and never substitutes heuristics on failure (`UNAVAILABLE`).
- **Synthetic self-consistency is not pilot accuracy:** High synthetic scores (such as `1.000` on closed-vocabulary templates) demonstrate structural gate behavior, not real-world generalization. See [docs/dataset-card.md](docs/dataset-card.md).

## Contributing & Security

See [CONTRIBUTING.md](CONTRIBUTING.md). Security vulnerabilities: see [SECURITY.md](SECURITY.md) — never open public GitHub issues for vulnerabilities or submit credentials/customer data.

## License

Apache-2.0. See [LICENSE](LICENSE).
