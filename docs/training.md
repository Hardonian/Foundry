# Offline Synthetic Training & Evaluation Stack

Foundry includes an offline training and evaluation stack (`src/hardonian_foundry/training_utils.py`, `scripts/`, and `examples/case-study-exception-codes/`) designed for reproducible synthetic experiments and Foundry task-mode provenance binding.

> **Scope Warning:** Built-in training workflows target synthetic datasets (`data_origin: "synthetic"`). No trained adapter is automatically promoted or proven suitable for production financial data without independent human review and consented pilot validation.

---

## 1. Offline Integrity & Snapshot Verification

All training and prediction runners share `hardonian_foundry.training_utils`:

- **Network & Credential Isolation:** `set_offline_env()` sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, and enforces `local_files_only=True`, `trust_remote_code=False`, and `token=False` on both tokenizer and model loaders.
- **Snapshot Pinning (`verify_snapshot`, `verify_pinned_blobs`, `load_verified_snapshot`):** Verifies `model.safetensors` SHA-256 (`sha_file`) and pinned Git blob SHA-1 digests (`git_blob_sha1`) for configuration/tokenizer files, and rejects any `config.json` containing `auto_map` custom code entries.
- **Model Registry Config (`model_registry`, `resolve_model_spec`):** JSON registry configuration (`--model-registry`) allows registering additional base models, pinned revisions, safetensors/blob digests, and LoRA `target_modules` beyond the default `Qwen/Qwen2.5-Coder-0.5B-Instruct` (`ea3f2471cf1b1f0db85067f1ef93848e38e88c25`, Apache-2.0 upstream model card).

---

## 2. Cross-Platform Preflight & Cooperative Locking

`training_utils.py` removes POSIX-only assumptions so preflight checks run cleanly on **Linux, macOS, and Windows**:

- **Private Output Directory (`verify_private_dir`, `mkdir_private`, `chmod_private`):**
  - On POSIX: verifies directory existence, `st_uid == os.getuid()`, and strict `0700` permissions (`st_mode & 0o077 == 0`).
  - On Windows (`os.name == "nt"`): checks directory ownership/ACL restrictions via Win32 `advapi32`/`kernel32` without crashing on missing `os.getuid()`.
- **Memory & Disk Checks (`available_ram_bytes`, `check_disk_space`):**
  - On POSIX: reads `os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")` (with `vm_stat` fallback on macOS).
  - On Windows: queries `GlobalMemoryStatusEx` via `ctypes`.
  - Enforces free disk headroom (>= 10 GiB and < 80% utilization where configured) before loading weights or writing checkpoints.
- **Cooperative Locking (`acquire_training_lock`, `TrainingLockError`):** Uses `fcntl.flock` on POSIX and bounded non-blocking `msvcrt.locking` (`LK_NBLCK`) on Windows. Cooperative locks protect participating jobs against concurrent write collisions; they do not reserve hardware compute.

---

## 3. Case-Study LoRA Training & Batched Inference

### 3.1 Training (`examples/case-study-exception-codes/train_lora.py`)

```sh
python examples/case-study-exception-codes/train_lora.py \
  --hf-home "$HF_HOME" \
  --task-dir examples/case-study-exception-codes/task \
  --output examples/case-study-exception-codes/task/adapter \
  --epochs 3 --batch-size 4 --grad-accum 8 --threads 8
```

Key behaviors and flags:
- **Fail-Closed Truncation Guard:** Rejects training runs if `--max-seq` truncates any encoded training example (rather than silently dropping label tokens).
- **Exact Gradient Accumulation Remainder (`grad_accum_window_size`, `should_step_grad_accum`):** Rescales trailing micro-batches cleanly when `len(train_rows)` is not evenly divisible by `--grad-accum`.
- **Padded Micro-Batching & LR Schedule:** Dynamic micro-batching (`--batch-size`), scheduler options (`--scheduler`, `--warmup-ratio`, `--max-grad-norm`), validation monitoring (`--eval-val` / `--no-eval-val`), and checkpointing (`--save-checkpoints`, `--resume`).
- **Device, Method & Registry Selection:** `--device` (`cpu` default; optional `auto`, `cuda`, `mps`, `dml`), `--method` (`lora`, plus structured contracts for `qlora`, `dpo` with `--preference-data`, and `rlvr`), and `--model-registry`.

### 3.2 Provenance Pinning (`finalize_spec.py`)

```sh
python examples/case-study-exception-codes/finalize_spec.py \
  --task-dir examples/case-study-exception-codes/task \
  --adapter-dir examples/case-study-exception-codes/task/adapter \
  [--verify] [--json]
```

Injects `adapter_sha256`, `revision`, and `base_model_id` into `spec.json`, refuses to overwrite an already-finalized spec, and supports `--verify` and `--json` to validate an existing binding cleanly.

### 3.3 Batched Evaluation (`model_predict.py`)

```sh
python examples/case-study-exception-codes/model_predict.py \
  --hf-home "$HF_HOME" \
  --task-dir examples/case-study-exception-codes/task \
  --adapter-dir examples/case-study-exception-codes/task/adapter \
  --out examples/case-study-exception-codes/task/model.jsonl \
  --batch-size 8 --threads 8 [--constrained] [--base-only]
```

Key behaviors:
- **Pre-Inference Integrity Verification:** Verifies `dataset_sha256`, `spec_sha256`, `base_weights_sha256`, and `adapter_sha256` against `train_manifest.json` and `spec.json` before decoding (supports `--base-only` for zero-shot baseline comparison).
- **Left-Padded Batched Decoding (`torch.inference_mode()`, `batched_greedy_generate`):** Generates predictions in batches (`--batch-size`) with optional constrained decoding (`--constrained` via `LabelPrefixTrie` / `PrefixTrieLogitsProcessor`), recording per-case `latency_ms`, `confidence`, `raw_invalid`, and `latency_p95_ms` in `predict_manifest.json`.
- **Boundary-Safe Allowlist Normalization (`to_label`):** Distinguishes `exact_allowlist_outputs`, `normalized_outputs`, and `mapped_to_undetermined`, preventing prefix collisions (e.g., `FOO` vs `FOO_BAR`). Unparseable generations fail closed to `UNDETERMINED`.

Case-study helpers (`baseline_predict.py` and `show_pubkey.py`) also support `--json` for structured machine-readable output.

---

## 4. Synthetic Smoke, Validation Monitoring & OOD Challenge Scripts

| Script | Purpose |
|---|---|
| `scripts/train_synthetic_lora.py` | Bounded offline CPU smoke (`--batch-size`, `--device`, `--model-registry`) comparing teacher-forced base vs. adapter loss and reload agreement |
| `scripts/evaluate_synthetic_lora.py` | Greedy generation on smoke cases (`--batch-size`, `--device`, `--model-registry`) checking model, dataset, adapter-weight, and adapter-config hashes |
| `scripts/experiment_balanced_lora.py` | Group-disjoint balanced experiment (train / validation / held-out test) with per-epoch validation tracking, early stopping, `--emit-task-dir`, `--run-refinery`, and `--refinery-output-root` |
| `scripts/challenge_lora.py` | Expanded 48-case (16/class) out-of-template numeric/currency/null OOD challenge set with `--emit-predictions` Foundry JSONL export |

> **Holdout Hygiene:** Once a held-out test split or challenge file is scored, it is no longer pristine—never tune hyperparameters or select checkpoints on the test split.

---

## 5. Device & Post-Training Adapter Contracts (`qlora`, `dpo`, `rlvr`)

`training_utils.py` (`get_training_hook`, `SFTAdapterHook`, `QLoRAAdapterHook`, `DPOAdapterHook`, `RLVRAdapterHook`, `compute_dpo_loss`, `RLVRVerifier`, `ExactMatchVerifier`) defines modular adapter interfaces across:
- **Devices (`select_device`):** `cpu` (default, verified in offline CI), `cuda`, `mps`, `dml`, and `auto`.
- **Adapter Methods:** `lora` / `sft` (active offline CPU default), plus structured contracts for `qlora`, `dpo`, and `rlvr`. Check `train_lora.py --help` and runner behavior before invoking non-default post-training modes; verify optional quantization/GPU packages (`bitsandbytes`, CUDA `torch`, etc.) in an isolated environment as they are not bundled or exercised in CPU-only CI runners.
