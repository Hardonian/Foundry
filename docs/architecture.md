# Architecture and limitations

The baseline validates integer minor units, computes deterministic mismatch suggestions, and produces hashes and synthetic held-out evaluation. An independent allowlist classifies agent tool proposals without invoking tools. All outcomes require human review.

The public `model-tools` project has an inference gateway and router; Foundry does not replace them. An opt-in integration requires verified current gateway contract, authentication, timeout and explicit unavailable results. Settler integration is not implemented: preserving its deterministic reconciliation, tenant authorization, proofpacks and manual review is mandatory. SAT-Frontier integration is not implemented.

No production API, authenticated tenant context, encrypted artifact store, immutable evidence service, model licensing validator, signed promotion, SFT, LoRA or QLoRA is present. Training and promotion must remain blocked pending these components and independent tests. Do not store raw private data in this repository.
