# Offline synthetic CPU LoRA smoke

This is an early demonstrator, not a production training platform. It uses only built-in, generated synthetic reconciliation labels. Do not use real or customer data. No model or adapter is promoted, deployed, published, or proven suitable for any production task.

The existing local cache must contain `Qwen/Qwen2.5-Coder-0.5B-Instruct` at revision `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`. At this revision, the upstream model card identifies Apache-2.0. Confirm the current license, model card, attribution/redistribution obligations, and any derivative rights before each use or distribution: https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/tree/ea3f2471cf1b1f0db85067f1ef93848e38e88c25 . The runner verifies the pinned safetensors SHA-256 and pinned Git blob IDs for required config/tokenizer files, refuses custom remote code, turns off Hub networking and implicit credentials, and does not download weights.

Use an existing isolated CPU environment with Python 3.12, torch 2.6.0+cpu, transformers 4.49.0, peft 0.14.0, accelerate 1.4.0, and safetensors 0.8.0. The smoke was exercised with these installed versions. The repository does not yet provide a locked, clean-room installation of this training stack; do not silently upgrade packages in an existing environment.

Set `HF_HOME` to a verified cache and `FOUNDRY_RUN_ROOT` to an existing, owner-only directory on a filesystem with at least 10 GiB free and less than 80% utilization. For genuine private IP or sensitive data, use separately provisioned encrypted, access-controlled storage; owner-only permissions alone do not encrypt data. No raw sensitive data is supported by this demonstration.

Run with the Python interpreter from the isolated CPU environment:

    python scripts/train_synthetic_lora.py --hf-home "$HF_HOME" --output-root "$FOUNDRY_RUN_ROOT" --steps 4 --threads 4

The process limits CPU threads and steps, checks RAM and disk, uses a nonblocking cooperative lock, creates a fresh mode-0700 run directory, trains attention LoRA on CPU, compares base and adapter teacher-forced losses on four separate synthetic examples, saves a safetensors adapter, reloads it offline, and verifies loss agreement. The run result includes model/data/script/adapter hashes, dependency versions, exact revision, losses, seed, duration and limitations. A lock protects only participating jobs; it does not reserve GPU resources. Failures can leave a partial directory without `result.json`; inspect it and preserve or clean up deliberately.

Warnings observed during the reference run: Transformers warns that sliding-window attention is not implemented for the selected eager path; the cached configuration has `use_sliding_window: false`, and the smoke's sequences are at most 256 tokens, but this does not validate long-context behavior. PEFT warns that it cannot find an online base-model config while saving and assumes the vocabulary has not changed; this runner never resizes the vocabulary. Investigate both warnings for any serious training. Lower toy held-out loss is not a measured accuracy improvement.

Not implemented: checkpoint/resume, dataset imports, QLoRA, GPU training, resource reservation, a full adversarial evaluation, production serving and promotion. DPO, distillation and RLVR are not implemented. Do not bypass these limitations by calling the adapter a trained product.
