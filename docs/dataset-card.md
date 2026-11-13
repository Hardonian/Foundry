# Dataset Card & Synthetic-to-Pilot Migration Guide

This document describes the synthetic datasets shipped with **Foundry**, their label taxonomies, known spurious-shortcut hazards, and the mandatory governance checklist before evaluating any consented pilot data.

> **Scope Warning:** Every dataset in this repository is **100% synthetic** and program-generated. High scores on these datasets measure structural behavior on controlled templates—they do **not** establish production accuracy on real customer or financial data.

---

## 1. Dataset Inventory & Taxonomies

### 1.1 Built-in Baseline Dataset (`baseline.py`)
- **Size:** 60 synthetic records (`synthetic-000` .. `synthetic-059`).
- **Schema:** `{"id", "expected_minor", "actual_minor", "critical"}` using bounded integer minor units (`abs(x) <= 10^12`).
- **Deterministic split:** 60% train (36) / 20% validation (12) / 20% test (12), seed `42`.
- **Purpose:** Offline smoke testing, integer minor-unit invariants, and CI self-consistency (`foundry demo`, `foundry eval`, `foundry refinery run`).

### 1.2 Case Study #1: Exception Resolution Codes (`examples/case-study-exception-codes/`)
- **Task ID:** `settler-exception-resolution-codes`
- **Gold labels (8):**
  | Label | Critical | Meaning |
  |---|---|---|
  | `DUPLICATE_RECORD_CONFIRMED` | No | Same transaction/invoice imported more than once |
  | `TIMING_DIFFERENCE_ACCEPTED` | No | Settlement window or cutoff timing explains variance |
  | `AMOUNT_VARIANCE_ACCEPTED` | No | Known fee, tax, or rounding delta accepted |
  | `CURRENCY_VARIANCE_ACCEPTED` | No | FX revaluation / rate delta accepted |
  | `MISSING_COUNTERPART_CONFIRMED` | **Yes** | Counterpart leg never posted (`critical: true`) |
  | `FALSE_POSITIVE_CONFIRMED` | No | Expected gap / benign alert |
  | `REOPENED_FOR_INVESTIGATION` | **Yes** | Unresolved discrepancy requiring escalation (`critical: true`) |
  | `UNDETERMINED` | No | Insufficient/conflicting information; honest fallback class |
- **Profiles (`generate_dataset.py --profile`):**
  - `standard` (default): **1,152 records** (`12` style families × `12` archetypes × `8` slot variants; `41.67%` critical), matching committed `task/data.jsonl`. Four archetypes embed adversarial first-match regex traps.
  - `adversarial-noisy`: **1,152 records** (`12` style families × `12` archetypes × `8` variants; `33.33%` critical) injecting operator typos (`impoted`, `settlment`, `investgation`, `missng`, `varience`), abbreviations (`ccy`, `amt`, `dup`, `reconcil`, `fp`), negation flips, multi-sentence conflicts, and near-tie ambiguous `UNDETERMINED` notes.
  - `imbalanced`: **1,200 records** (`12` style families × `100` records/family; **4.0% critical** = `24` `REOPENED_FOR_INVESTIGATION` + `24` `MISSING_COUNTERPART_CONFIRMED`) demonstrating how high macro-F1 (`0.894` for the keyword baseline) can conceal a `0.5625` critical false-negative rate.

### 1.3 Case Study #2: Agent Tool-Call Safety Gate (`examples/case-study-agent-tool-gate/`)
- **Task ID:** `agent-tool-call-safety-gate`
- **Size:** **720 records** (`12` workflow families × `15` scenario archetypes × `4` slot variants; `BASE_SEED=20261111`).
- **Gold labels (3):**
  - `ALLOW_PROPOSAL` (`critical: false`, 240 records) — schema-exact allowlisted read/compute calls with safe arguments
  - `DENY` (`critical: true`, 240 records) — forbidden tools, path traversal (`../`, `..\`, `%2e%2e%2f`, drive letters), argument smuggling (`bypass_gate`, `__proto__`, `exec_cmd`, `sql_override`), shell/SQL metacharacters, and prompt-injection overrides
  - `ESCALATE_REVIEW` (`critical: true`, 240 records) — oversized batch limits (`limit > 100`), `export_audit_bundle`, `quarantine-` records, `external_contractor`/`unverified_agent` roles, or missing/TBD justification
- **Zero-weight offline runnable:** Uses `gate_predict.py` vs `baseline_predict.py` without requiring HF weights.

### 1.4 Case Study #3: Reconciliation Policy Override (`examples/case-study-reconciliation-policy/`)
- **Task ID:** `reconciliation-arithmetic-policy-override`
- **Size:** **576 records** (`12` synthetic feed families × `12` archetypes × `4` variants; `BASE_SEED=20261112`).
- **Gold labels (3):** Computed deterministically via `hardonian_foundry.synthetic_policy.assess`:
  - `matched` (`critical: false`, 192 records)
  - `amount_mismatch` (`critical: true`, 192 records)
  - `review_required` (`critical: true`, 192 records)
- **Purpose:** Demonstrates `synthetic_policy.assess_with_suggestion` keeping `review_required=True` and `financial_write=False` while auditing and overriding untrusted model suggestions on 1-minor-unit discrepancies, sign flips, large-integer precision edges, currency mismatches, and null/off-allowlist responses.

---

## 2. Spurious-Shortcut Warnings

When building or reviewing evaluations on these datasets, guard against these known traps:

1. **Template Memorization Ceiling:** Small models (e.g., 0.5B LoRA) can achieve `1.000` macro-F1 on closed-vocabulary synthetic templates. Treat `1.000` as a baseline hygiene check—not evidence of generalization to free-form operator prose.
2. **Aggregate Metric Illusion under Class Skew:** As demonstrated by `--profile imbalanced`, a classifier can score `> 0.89` macro-F1 while missing over half of rare critical escalations. Always gate on `critical_false_negative_rate == 0.0` alongside per-class recall.
3. **Style / Family Leakage:** Splitting randomly by record leaks writing-style openers, slot distributions, and phrasing across train and test. Always include a `family` field on every record so `task_eval.split_task_dataset` performs group-disjoint splitting (~60/15/25 by family).
4. **Regex & Keyword Shortcuts:** First-match-wins rules latch onto stale historical keywords ("earlier notes suspected a duplicate... reopen for investigation") or benign tool wrappers ("routine read_record audit... execute_sql"). Test with `--profile adversarial-noisy` and out-of-template challenge sets.
5. **Decoder Label-Prefix Collisions:** Off-allowlist decoding must match exact tokens or token boundaries before fallback to `UNDETERMINED`, avoiding prefix bugs (e.g., `FOO` matching `FOO_BAR`) or case-insensitive leakage (`Matched` vs `matched`).

---

## 3. Pilot Migration Checklist (Before Consented Data)

Do **not** point Foundry at real customer or production financial data unless all governance, isolation, and verification gates below are completed:

- [ ] **Explicit Data Origin & Governance Attestation:** Configure `data_origin` strictly to `consented_pilot` or `redacted_eval` with an explicit `governance_attestation` inside the licensing manifest and verify policy approval—never bypass the default `("synthetic",)` fail-closed origin guard without authorization.
- [ ] **PII & Tenant Secret Redaction:** Strip customer names, account numbers, IBANs, tax IDs, raw tokens, and proprietary notes before dataset creation; assign opaque tenant-scoped IDs and verify `TenantContext` isolation.
- [ ] **Duplicate & Near-Duplicate Leakage Audit:** Run `foundry refinery validate-dataset --task-spec <spec>` and inspect near-duplicate clusters across tenant/date/batch families so no incident thread spans train and test splits.
- [ ] **Multi-Rater Adjudication on Ambiguous / Critical Classes:** Independently double-label `UNDETERMINED`, `ESCALATE_REVIEW`, `REOPENED_FOR_INVESTIGATION`, and `MISSING_COUNTERPART_CONFIRMED` examples before freezing the pilot test split.
- [ ] **Deterministic Guardrail Preservation:** Keep `synthetic_policy.assess` / `assess_with_suggestion` and explicit tool allowlists authoritative over model outputs (`review_required: true`, `financial_write: false`, `executed: false`).
- [ ] **Artifact Integrity & Multi-Signature Promotion:** Pin weights/adapter SHA-256 digests, verify local artifact hashes, anchor the ledger head externally (`refinery anchor`), and enforce M-of-N keyring quorum (`refinery promote --keyring ... --expected-head ...`) with explicit human sign-off.
