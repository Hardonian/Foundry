"""Developer-facing Refinery workflow: backward-compatible facade over modular Refinery subsystems.

Synthetic-only research baseline. Never deploy for real financial data or autonomous tool execution.
"""
from __future__ import annotations

from . import signing
from .artifact import create_build_attestation, verify_build_attestation
from .baseline import (
    dataset as default_dataset,
)
from .baseline import (
    reconciliation,
    split,
)
from .canonical import canonical, digest
from .compliance import (
    evaluate_compliance_profile,
    evaluate_policy_bundle,
    render_audit_html,
)
from .evidence import _append_with_builder as evidence_append_with_builder
from .evidence import append as evidence_append
from .evidence import verify_entries as evidence_verify_entries
from .licensing import evaluate as evaluate_license
from .proofpack import build_proofpack, verify_proofpack
from .refinery_attestation import (
    AnchorSink,
    WormDirectoryAnchorSink,
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
    generate_audit_report,
    generate_dataset_card,
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
from .registry import (
    get_model,
    list_models,
    register_model,
    verify_registry,
)
from .signing import (
    CallbackSignerProvider,
    LocalPemSignerProvider,
    SignerProvider,
)
from .task_eval import (
    cohens_kappa,
    compare_task_predictions,
    evaluate_trajectories,
    fleiss_kappa,
    scan_dataset_pii,
    score_predictions,
    split_task_dataset,
    validate_spec,
    validate_task_dataset,
)

__all__ = [
    "AnchorSink",
    "CallbackSignerProvider",
    "DEFAULT_RUNS_DIR",
    "LocalPemSignerProvider",
    "SignerProvider",
    "WormDirectoryAnchorSink",
    "_code_digest",
    "_find_ledger_entry",
    "_iter_ledger_entries",
    "anchor_run",
    "attach_license",
    "build_proofpack",
    "canonical",
    "cohens_kappa",
    "compare_task_predictions",
    "create_build_attestation",
    "create_run",
    "create_task_run",
    "default_dataset",
    "diff_runs",
    "digest",
    "evaluate_compliance_profile",
    "evaluate_dataset",
    "evaluate_license",
    "evaluate_policy_bundle",
    "evaluate_promotion",
    "evaluate_trajectories",
    "evidence_append",
    "evidence_append_with_builder",
    "evidence_verify_entries",
    "export_bundle",
    "extract_verified_license_attachment",
    "fleiss_kappa",
    "generate_audit_report",
    "generate_dataset_card",
    "get_model",
    "inspect_run",
    "list_models",
    "list_runs",
    "load_keyring",
    "load_keyring_policy",
    "parse_keyring",
    "reconciliation",
    "register_model",
    "render_audit_html",
    "resolve_ledger_path",
    "resolve_run_dir",
    "run_preflight",
    "scan_dataset_pii",
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
    "verify_build_attestation",
    "verify_bundle",
    "verify_proofpack",
    "verify_registry",
]
