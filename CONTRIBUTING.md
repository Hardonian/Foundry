# Contributing to Foundry

Thank you for contributing to Hardonian Foundry. Please follow these guidelines to keep the evaluation pipeline deterministic, cross-platform, and fail-closed.

## Core Rules

1. **Synthetic fixtures only by default.** Do not commit real customer records, PII, credentials, model weights, or private keys (`*.pem`, `keys/`).
2. **Preserve committed signed evidence.** Never alter `examples/case-study-exception-codes/task/run.json` or any committed signed ledger/run artifacts in `examples/case-study-exception-codes/task/runs/`.
3. **Fail closed on security-critical paths.** Missing evidence, invalid inputs, broken hash chains, untrusted keys, or missing optional security dependencies (`cryptography`) must fail explicitly—never silently degrade to `ALLOW`.
4. **Human review remains mandatory.** Outputs and promotion decisions (`ALLOWED_REVIEW_REQUIRED`) are evidence attestations; never introduce autonomous financial writes or unreviewed tool execution.
5. **Cross-platform support.** Avoid POSIX-only assumptions (`os.getuid`, `os.sysconf`, POSIX-only mode bitmasks) without Windows (`os.name == "nt"`) guards.

## Local Development Setup

Install Foundry in editable mode with development and signing dependencies:

```sh
python -m pip install -e ".[dev]"
```

If working without installing the package, set `PYTHONPATH=src` (POSIX) or `$env:PYTHONPATH="src"` (PowerShell).

## Required Checks

Run the full verification suite before opening a pull request:

```sh
# Unit tests (stdlib unittest or pytest)
python -m unittest discover -s tests -v
pytest --cov=hardonian_foundry --cov-fail-under=85

# Lint and static type checks
ruff check .
mypy src

# Security scan
bandit -q -r src scripts examples --severity-level medium
```

For changes touching `examples/` dataset generators or case studies, also run:

```sh
python -m unittest tests/test_content_datasets.py -v
```

## Pull Request Expectations

- Add or update unit tests for every new gate, CLI flag, or dataset generator.
- Keep functions type-annotated (`Python >= 3.11`) and docstrings accurate.
- Update `docs/` and `CHANGELOG.md` when adding user-facing CLI flags, schemas, or case studies.
