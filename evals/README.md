# Foundry licensing-gate red-team (promptfoo)

Deterministic, CI-resident security eval for `hardonian_foundry.licensing.evaluate`
— the fail-closed SPDX allowlist / provenance gate that guards model promotion.
No LLM, no network, no GPU: the provider calls the real gate on adversarial
manifests and promptfoo asserts the decision.

## Invariants asserted on EVERY case
- `promotion_allowed` is always `false` (promotion requires signed Refinery gates,
  never this review-only validator).
- The decision matches the expected ALLOWED / DENIED for the manifest.

## Coverage
- Allowed SPDX ids (Apache-2.0, MIT, BSD-3-Clause, CC0-1.0, OFL-1.1) -> ALLOWED.
- Rejected licenses: Proprietary, AGPL/GPL, SSPL, BUSL, Elastic-2.0, OpenRAIL-M,
  compound expressions, case/whitespace/newline tricks, empty/non-string ids.
- Data origin: only `synthetic` by default; governance tiers need explicit
  `allowed_data_origins` + a valid `governance_attestation` (redacted_eval must be
  redacted=true).
- source_url: https or numeric-loopback http only (rejects remote http, javascript:,
  credentials, non-numeric localhost, ftp).
- weights_sha256: 64-char lowercase hex.
- Structure: unknown fields, empty/padded name, non-dict manifest, missing fields.

## Run locally
    npx -y promptfoo@0.125.1 eval -c evals/promptfoo.licensing.yaml --no-cache
Requires Node >= 22.22.0 (CI uses Node 24) and Python 3.11+ on PATH. The provider imports the gate from
../src, so no package install is needed.

## CI
The `promptfoo-redteam` job in .github/workflows/ci.yml runs this suite on every
push/PR. It is additive and does not modify the coverage/mypy/bandit/pip-audit gates.

## Add a case
Append a test to promptfoo.licensing.yaml: `vars.payload` is a JSON envelope
{"manifest": {...}, "opts": {...}}; assert the expected `decision` and
`promotion_allowed: false`.
