# Security Policy

## Scope & Supported Posture

Hardonian Foundry is an offline-first, evidence-gated model evaluation and lifecycle framework.

- **Default Posture:** All built-in datasets, demos, and case studies use **synthetic data only** and execute offline on CPU with human review required (`review_rate = 1.0`, `financial_write = false`, `executed = false`).
- **Pilot / Redacted Governance Tiers:** Optional `consented_pilot` and `redacted_eval` origins in `licensing.py` / `evidence.py` require explicit `governance_attestation` and remain subject to operator authorization, external anchoring, and manual review. Raw unconsented customer records must never be processed or committed.
- **v0.2.0 Advisory:** Releases prior to `0.2.1` had incomplete promotion signature binding (`v1` signatures). Legacy signatures are rejected; re-run and re-sign with `v2` signatures (`0.2.1+` / `0.3.0`).

## Security Controls

- **Fail-Closed Promotion Gate:** Incomplete evidence, missing/denied license manifests, untrusted or revoked signing keys, failed quorum, or head mismatches block promotion with exit code `2`.
- **Optional Signing Extra:** Without `cryptography`, `refinery sign` errors out and `refinery promote` blocks reporting `cryptographic_promotion_signature` as missing evidence.
- **Cryptographic Key Handling:** Private keys (`Ed25519` PEM) are generated with `0600` permissions on POSIX and restricted permissions on Windows, never overwrite existing files, and are never logged or embedded in run records.
- **External Rollback Detection:** Local append-only ledgers (`evidence.jsonl` / SQLite WAL store) provide tamper evidence, not rollback prevention on their own; verify runs against a trusted external anchor (`--expected-head` / `refinery anchor`).
- **Offline Model Integrity:** Snapshot loaders enforce `local_files_only=True`, `trust_remote_code=False`, `token=False`, reject `auto_map` custom code, and verify pinned `safetensors` / adapter SHA-256 digests.
- **Loopback Provider Guard:** The opt-in provider adapter restricts calls to operator-approved loopback origins (unless TLS/auth is explicitly opted in), disables HTTP redirects and ambient proxies, enforces strict response-size and wall-clock limits, and never substitutes heuristic results on failure (`UNAVAILABLE`).

## Reporting a Vulnerability

Please report security vulnerabilities **privately** to the repository maintainers via GitHub Private Vulnerability Reporting (or direct maintainer contact) rather than opening public GitHub issues.

- **Do not** submit private keys, tokens, credentials, or real customer records in any issue or pull request.
- Include affected version/commit, reproduction steps, and potential security impact.
