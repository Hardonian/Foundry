# Hardonian Foundry

Early, offline CPU-only research baseline. Apache-2.0. The deterministic heuristics are NOT a trained model, are NOT commercially validated, and process SYNTHETIC inputs only. No inference server, production training system, authenticated tenant isolation, production API, production-grade evidence store or usable model registry is implemented yet. Do not deploy for real financial data or autonomous tool execution.

```sh
PYTHONPATH=src python3 -m hardonian_foundry.cli doctor
PYTHONPATH=src python3 -m hardonian_foundry.cli demo
PYTHONPATH=src python3 -m hardonian_foundry.cli eval
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

### Developer Refinery Workflow

The developer-facing Refinery workflow integrates dataset validation, preflight safety checks, reproducible evaluation, tamper-evident run tracking, and fail-closed promotion gating:

```sh
# Validate synthetic dataset integrity, id uniqueness, and splitability
PYTHONPATH=src python3 -m hardonian_foundry.cli refinery validate-dataset

# Run preflight safety, resource, and licensing checks
PYTHONPATH=src python3 -m hardonian_foundry.cli refinery preflight

# Execute end-to-end refinery run with tamper-evident evidence tracking
PYTHONPATH=src python3 -m hardonian_foundry.cli refinery run --output-root .foundry/runs

# Inspect run record and verify tamper-evident evidence chain provenance
PYTHONPATH=src python3 -m hardonian_foundry.cli refinery inspect <run_id> --output-root .foundry/runs

# Evaluate model promotion gate (fails closed with explicit missing evidence reasons)
PYTHONPATH=src python3 -m hardonian_foundry.cli refinery promote <run_id> --output-root .foundry/runs
```

An offline CPU LoRA training smoke is available using a separately installed, pinned compatible environment and an already cached open-weight model. It trains only synthetic examples; see docs/training.md for preflight, security restrictions, and limitations. This is not a production training platform.

An opt-in, synthetic-only OpenAI-compatible loopback provider adapter is available as `python3 -m hardonian_foundry.cli provider --origin http://127.0.0.1:12345 --model example`. The address and model must identify an operator-approved local server; this example is not a configured service. Provider errors return UNAVAILABLE with nonzero exit and no heuristic substitution. A successful response is always review-required. This adapter is not verified against a live model-tools deployment; the inspected model-tools `app.main` implementation returns HTTP 501 for `/v1/chat/completions`. No real provider call is made by tests.

`registry promote` always fails closed. Outputs are suggestions only and always require human review. The synthetic dataset is constructed programmatically; its split is seed-stable within the documented Python runtime. Evidence hashes are metadata, not encrypted or tamper-proof storage. Synthetic self-consistency is not pilot accuracy. No financial writes or tool execution exist.

See docs/architecture.md and docs/refinery.md for boundaries, lifecycle contracts, and outstanding work. Project does not include model weights or customer data.
