"""Developer-facing Refinery workflow: backward-compatible facade over modular Refinery subsystems.

Synthetic-only research baseline. Never deploy for real financial data or autonomous tool execution.
"""
from __future__ import annotations

try:
    from .canonical import canonical, digest  # type: ignore[import-not-found,import-untyped]
except ImportError:  # pragma: no cover - pre-merge fallback
    from .baseline import canonical, digest

from . import signing
from .baseline import (
    dataset as default_dataset,
)
from .baseline import (
    reconciliation,
    split,
)
from .evidence import _append_with_builder as evidence_append_with_builder
from .evidence import append as evidence_append
from .evidence import verify_entries as evidence_verify_entries
from .licensing import evaluate as evaluate_license
from .refinery_attestation import (
    _find_ledger_entry,
    _iter_ledger_entries,
    anchor_run,
    attach_license,
    extract_verified_license_attachment,
    load_keyring,
    load_keyring_policy,
    parse_keyring,
    sign_run,
)
from .refinery_gates import evaluate_promotion
from .refinery_preflight import run_preflight
from .refinery_runs import (
    create_run,
    create_task_run,
    diff_runs,
    evaluate_dataset,
    export_bundle,
    inspect_run,
    list_runs,
    verify_bundle,
)
from .refinery_validation import (
    DEFAULT_RUNS_DIR,
    _code_digest,
    resolve_ledger_path,
    resolve_run_dir,
    validate_and_split_dataset,
    validate_dataset,
    validate_run_id,
)
from .task_eval import (
    score_predictions,
    split_task_dataset,
    validate_spec,
    validate_task_dataset,
)

__all__ = [
    "DEFAULT_RUNS_DIR",
    "_code_digest",
    "_find_ledger_entry",
    "_iter_ledger_entries",
    "anchor_run",
    "attach_license",
    "canonical",
    "create_run",
    "create_task_run",
    "default_dataset",
    "diff_runs",
    "digest",
    "evaluate_dataset",
    "evaluate_license",
    "evaluate_promotion",
    "evidence_append",
    "evidence_append_with_builder",
    "evidence_verify_entries",
    "export_bundle",
    "extract_verified_license_attachment",
    "inspect_run",
    "list_runs",
    "load_keyring",
    "load_keyring_policy",
    "parse_keyring",
    "reconciliation",
    "resolve_ledger_path",
    "resolve_run_dir",
    "run_preflight",
    "score_predictions",
    "sign_run",
    "signing",
    "split",
    "split_task_dataset",
    "validate_and_split_dataset",
    "validate_dataset",
    "validate_run_id",
    "validate_spec",
    "validate_task_dataset",
    "verify_bundle",
]
