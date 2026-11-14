"""Hardonian Foundry: offline-first, evidence-gated model evaluation and lifecycle suite."""
from __future__ import annotations

from .artifact import (
    add_artifact,
    get_artifact_path,
    list_artifacts,
    verify_artifact,
)
from .canonical import canonical, digest
from .evidence import JsonlEvidenceStore, SQLiteEvidenceStore
from .provider import (
    CircuitBreaker,
    ProviderError,
    predict_batch,
    suggest,
)
from .refinery import (
    anchor_run,
    attach_license,
    create_run,
    create_task_run,
    diff_runs,
    evaluate_promotion,
    export_bundle,
    generate_audit_report,
    inspect_run,
    list_runs,
    load_keyring,
    load_keyring_policy,
    run_preflight,
    sign_run,
    validate_dataset,
    verify_bundle,
)
from .registry import (
    get_model,
    list_models,
    register_model,
    verify_registry,
)
from .task_eval import (
    bootstrap_ci,
    compare_task_predictions,
    load_jsonl,
    load_predictor_dataset,
    mcnemar_test,
    score_predictions,
    split_task_dataset,
    validate_spec,
    validate_task_dataset,
)
from .tenancy import TenantContext

__version__ = "1.0.0"

__all__ = [
    "CircuitBreaker",
    "JsonlEvidenceStore",
    "ProviderError",
    "SQLiteEvidenceStore",
    "TenantContext",
    "__version__",
    "add_artifact",
    "anchor_run",
    "attach_license",
    "bootstrap_ci",
    "canonical",
    "compare_task_predictions",
    "create_run",
    "create_task_run",
    "diff_runs",
    "digest",
    "evaluate_promotion",
    "export_bundle",
    "generate_audit_report",
    "get_artifact_path",
    "get_model",
    "inspect_run",
    "list_artifacts",
    "list_models",
    "list_runs",
    "load_jsonl",
    "load_keyring",
    "load_keyring_policy",
    "load_predictor_dataset",
    "mcnemar_test",
    "predict_batch",
    "register_model",
    "run_preflight",
    "score_predictions",
    "sign_run",
    "split_task_dataset",
    "suggest",
    "validate_dataset",
    "validate_spec",
    "validate_task_dataset",
    "verify_artifact",
    "verify_bundle",
    "verify_registry",
]
