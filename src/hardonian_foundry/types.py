"""Shared Python 3.11+ type definitions and exit codes for Hardonian Foundry."""
from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

EXIT_OK: int = 0
EXIT_CHECK_FAILED: int = 1
EXIT_BLOCKED_OR_ERROR: int = 2

DataOrigin = Literal["synthetic", "consented_pilot", "redacted_eval"]
ValidationStatus = Literal["VALID", "INVALID"]
PreflightStatus = Literal["PASSED", "FAILED"]
PromotionStatus = Literal["ALLOWED_REVIEW_REQUIRED", "BLOCKED"]
ProviderStatus = Literal["REVIEW_REQUIRED", "UNAVAILABLE"]
BundleFormat = Literal["dir", "zip", "tar", "tar.gz", "tgz"]


class ValidationResult(TypedDict):
    status: ValidationStatus
    record_count: int
    dataset_sha256: str | None
    split_counts: dict[str, int] | None
    errors: list[str]
    family_disjoint: NotRequired[bool]
    family_count: NotRequired[int]


class TaskDatasetValidationResult(TypedDict):
    status: ValidationStatus
    record_count: int
    dataset_sha256: str | None
    errors: list[str]
    family_disjoint: NotRequired[bool]
    family_count: NotRequired[int]


class PreflightCheck(TypedDict):
    name: str
    status: PreflightStatus
    details: str


class PreflightResult(TypedDict):
    status: PreflightStatus
    checks: list[PreflightCheck]
    errors: list[str]


class PerClassMetrics(TypedDict):
    precision: float
    recall: float
    f1: float
    support: int


class FamilyMetrics(TypedDict):
    count: int
    correct: int
    accuracy: float
    critical_total: int
    critical_misses: int
    critical_false_negative_rate: float | None


class CalibrationMetrics(TypedDict):
    ece: float
    brier_score: float
    mean_confidence: float
    bins: int


class TaskScoreMetrics(TypedDict):
    test_count: int
    accuracy: float | None
    macro_f1: float | None
    per_class: dict[str, PerClassMetrics]
    confusion_matrix: dict[str, dict[str, int]]
    per_family: dict[str, FamilyMetrics]
    critical_total: int
    critical_misses: int
    critical_false_negative_rate: float | None
    labels_scored: list[str]
    predictions_sha256: str
    invalid_output_count: int
    invalid_output_rate: float | None
    abstention_count: int
    abstention_rate: float | None
    selective_accuracy: float | None
    calibration: CalibrationMetrics | None
    latency_p95_ms: float | None


class ProvenanceInfo(TypedDict):
    code_digest: str
    runtime: str
    evidence_ledger_path: str
    evidence_entry_hash: str
    evidence_seq: int
    evidence_key: str
    request_id: NotRequired[str]


class RunRecord(TypedDict):
    schema_version: str
    run_id: str
    created_at: str
    status: str
    dataset_sha256: str
    test_sha256: str
    split_seed: int
    preflight: PreflightResult
    evaluation: dict[str, Any]
    model_manifest: dict[str, Any] | None
    provenance: ProvenanceInfo
    tenant_id: NotRequired[str]


class InspectResult(TypedDict):
    run_id: str
    status: str | None
    created_at: str | None
    provenance_status: str
    dataset_sha256: str | None
    evaluation: dict[str, Any] | None
    preflight_status: str | None
    provenance: dict[str, Any] | None
    ledger_report: dict[str, Any] | None
    matched_entry: dict[str, Any] | None
    model_manifest: dict[str, Any] | None
    resolved_ledger_path: NotRequired[str | None]
    anchor_verified: NotRequired[bool]
    anchor_error: NotRequired[str | None]
    run_record: NotRequired[dict[str, Any]]
    ledger_entries: NotRequired[list[dict[str, Any]]]


class PromotionResult(TypedDict):
    promoted: bool
    status: PromotionStatus
    run_id: str
    reasons: list[str]
    missing_evidence: list[str]
    evidence_summary: dict[str, Any]
    review_required: bool
    note: str


class AnchorRecord(TypedDict):
    schema_version: str
    data_origin: str
    run_id: str | None
    head: str
    count: int
    anchored_at: str
    ledger_sha256: str
    anchor_path: NotRequired[str | None]


class RunSummary(TypedDict):
    run_id: str
    created_at: str | None
    status: str | None
    schema_version: str | None
    mode: str | None
    task_id: str | None
    dataset_sha256: str | None
    test_sha256: str | None
    f1: float | None
    critical_false_negative_rate: float | None
    tenant_id: NotRequired[str | None]
    provenance_status: NotRequired[str | None]


class RunDiff(TypedDict):
    run_id_a: str
    run_id_b: str
    provenance_a: str
    provenance_b: str
    same_dataset: bool
    same_test_split: bool
    same_code_digest: bool
    same_model_identity: bool
    metrics_delta: dict[str, float | None]
    per_class_f1_delta: dict[str, float | None]
    thresholds_changed: bool


class ProviderResult(TypedDict):
    status: ProviderStatus
    mode: str
    data_origin: str
    review_required: bool
    audit_sha256: str
    latency_ms: float
    category: NotRequired[str]
    arithmetic_agrees: NotRequired[bool]
    reason_code: NotRequired[str]
    reason: NotRequired[str]
    http_status: NotRequired[int]


class ArtifactEntry(TypedDict):
    schema_version: str
    artifact_id: str
    sha256: str
    size_bytes: int
    name: str
    kind: str
    run_id: str | None
    created_at: str
    blob_relpath: str
    blob_path: str


class ArtifactVerifyResult(TypedDict):
    status: Literal["VERIFIED", "MISMATCH", "MISSING"]
    verified: bool
    sha256: str | None
    expected_sha256: str | None
    size_bytes: int | None
    blob_path: str | None
    reason: str | None


class BundleManifest(TypedDict):
    schema_version: str
    run_id: str
    created_at: str
    format: str
    head: str | None
    count: int | None
    files: list[dict[str, Any]]
    bundle_sha256: str
