# Hardonian Foundry

Early, offline CPU-only research baseline. Apache-2.0. The deterministic heuristics are NOT a trained model, are NOT commercially validated, and process SYNTHETIC inputs only. No inference server, training system, tenant isolation, production API, persistent evidence store or usable model registry is implemented yet. Do not deploy for real financial data or autonomous tool execution.

```sh
PYTHONPATH=src python3 -m hardonian_foundry.cli doctor
PYTHONPATH=src python3 -m hardonian_foundry.cli demo
PYTHONPATH=src python3 -m hardonian_foundry.cli eval
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

`registry promote` always fails closed. Outputs are suggestions only and always require human review. The synthetic dataset is constructed programmatically; its split is seed-stable within the documented Python runtime. Evidence hashes are metadata, not encrypted or tamper-proof storage. Synthetic self-consistency is not pilot accuracy. No financial writes or tool execution exist.

See docs/architecture.md for boundaries and outstanding work. Project does not include model weights or customer data.
