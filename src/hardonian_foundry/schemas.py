"""Draft 2020-12 JSON Schema definitions for Foundry contracts (`foundry schema`)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_HEX64 = "^[0-9a-f]{64}$"

_SCHEMAS: dict[str, dict[str, Any]] = {
    "task-spec": {
        "$schema": _DRAFT_2020_12,
        "$id": "https://hardonian.dev/schemas/foundry/task-spec.v1.json",
        "title": "Foundry Task Specification (v1.0)",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "task_id", "labels", "thresholds"],
        "properties": {
            "schema_version": {"type": "string", "const": "1.0"},
            "task_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "description": {"type": "string", "maxLength": 1024},
            "labels": {"type": "array", "minItems": 2, "maxItems": 64, "items": {"type": "string"}},
            "thresholds": {
                "type": "object",
                "required": ["macro_f1_min"],
                "properties": {
                    "macro_f1_min": {"type": "number", "minimum": 0.8, "maximum": 1.0},
                    "critical_fn_rate_max": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "invalid_output_rate_max": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "latency_p95_ms_max": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 600000.0},
                },
            },
            "baseline_must_improve": {"type": "boolean"},
            "split_seed": {"type": "integer"},
            "dataset_path": {"type": "string"},
            "model": {"type": "object"},
            "notes": {"type": "string", "maxLength": 1024},
        },
    },
    "keyring": {
        "$schema": _DRAFT_2020_12,
        "$id": "https://hardonian.dev/schemas/foundry/keyring.v1.json",
        "title": "Foundry Trusted Keyring Policy (v1.0)",
        "type": "object",
        "additionalProperties": False,
        "required": ["trusted_keys"],
        "properties": {
            "schema_version": {"type": "string"},
            "version": {"type": "string"},
            "description": {"type": "string"},
            "notes": {"type": "string"},
            "required_signatures": {"type": "integer", "minimum": 1},
            "required_roles": {"type": "array", "items": {"type": "string"}},
            "trusted_keys": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {"type": "string", "pattern": _HEX64},
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["public_key_hex"],
                            "properties": {
                                "public_key_hex": {"type": "string", "pattern": _HEX64},
                                "key_id": {"type": "string", "minLength": 1, "maxLength": 128},
                                "role": {"type": "string", "minLength": 1, "maxLength": 64},
                                "revoked_at": {"type": "string", "minLength": 1, "maxLength": 64},
                                "label": {"type": "string", "maxLength": 128},
                            },
                        },
                    ]
                },
            },
        },
    },
    "license-manifest": {
        "$schema": _DRAFT_2020_12,
        "$id": "https://hardonian.dev/schemas/foundry/license-manifest.v1.json",
        "title": "Foundry Model Licensing & Origin Manifest (v1.0)",
        "type": "object",
        "required": ["name", "license_id", "provenance", "data_origin"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "license_id": {"type": "string", "enum": ["Apache-2.0", "MIT", "BSD-3-Clause", "CC0-1.0", "OFL-1.1"]},
            "provenance": {"type": "string", "minLength": 1},
            "data_origin": {"type": "string", "enum": ["synthetic", "consented_pilot", "redacted_eval"]},
            "source_url": {"type": "string", "format": "uri"},
            "weights_sha256": {"type": "string", "pattern": _HEX64},
            "governance_attestation": {"type": "object"},
        },
    },
    "proofpack-v2": {
        "$schema": _DRAFT_2020_12,
        "$id": "https://hardonian.dev/schemas/foundry/proofpack.v2.json",
        "title": "Foundry Read-Only Proofpack v2",
        "type": "object",
        "required": [
            "schema_version",
            "proofpack_type",
            "run_id",
            "status",
            "promoted",
            "review_required",
            "financial_write",
            "executed",
            "coverage_automatic_decision",
            "downstream_contract",
            "proofpack_sha256",
        ],
        "properties": {
            "schema_version": {"const": "2.0"},
            "proofpack_type": {"const": "hardonian.foundry.proofpack.v2"},
            "run_id": {"type": "string"},
            "status": {"type": "string", "enum": ["ALLOWED_REVIEW_REQUIRED", "BLOCKED"]},
            "promoted": {"type": "boolean"},
            "review_required": {"const": True},
            "financial_write": {"const": False},
            "executed": {"const": False},
            "coverage_automatic_decision": {"const": 0.0},
            "proofpack_sha256": {"type": "string", "pattern": _HEX64},
            "cryptographic_receipt": {"type": "object"},
            "downstream_contract": {
                "type": "object",
                "required": ["consumers", "mode", "autonomous_execution_allowed", "requires_human_review"],
                "properties": {
                    "consumers": {"type": "array", "items": {"type": "string"}},
                    "mode": {"const": "read_only_advisory"},
                    "autonomous_execution_allowed": {"const": False},
                    "requires_human_review": {"const": True},
                },
            },
        },
    },
    "registry": {
        "$schema": _DRAFT_2020_12,
        "$id": "https://hardonian.dev/schemas/foundry/registry.v1.json",
        "title": "Foundry SHA-256 Model Registry (v1.0)",
        "type": "object",
        "required": ["schema_version", "models", "registry_sha256"],
        "properties": {
            "schema_version": {"const": "1.0"},
            "registry_sha256": {"type": "string", "pattern": _HEX64},
            "models": {"type": "array", "items": {"type": "object"}},
        },
    },
    "slsa-attestation": {
        "$schema": _DRAFT_2020_12,
        "$id": "https://hardonian.dev/schemas/foundry/slsa-attestation.v1.json",
        "title": "Foundry in-toto / SLSA v1 Build Attestation",
        "type": "object",
        "required": ["statement_type", "predicate_type", "subject", "predicate", "attestation_sha256"],
        "properties": {
            "statement_type": {"const": "https://in-toto.io/Statement/v1"},
            "predicate_type": {"const": "https://slsa.dev/provenance/v1"},
            "attestation_sha256": {"type": "string", "pattern": _HEX64},
            "subject": {"type": "array"},
            "predicate": {"type": "object"},
        },
    },
    "policy-bundle": {
        "$schema": _DRAFT_2020_12,
        "$id": "https://hardonian.dev/schemas/foundry/policy-bundle.v1.json",
        "title": "Foundry Promotion Policy Bundle (v1.0)",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "string"},
            "policy_id": {"type": "string"},
            "min_macro_f1": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "max_critical_fn_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "max_latency_p95_ms": {"type": "number", "exclusiveMinimum": 0.0},
            "min_slice_recall": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "require_zero_false_tool_exec": {"type": "boolean"},
            "allowed_data_origins": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
        },
    },
}


def list_schemas() -> list[str]:
    """Return sorted schema names supported by Foundry."""
    return sorted(_SCHEMAS)


def get_schema(name: str) -> dict[str, Any]:
    """Return a Draft 2020-12 JSON Schema dictionary for a named Foundry contract."""
    if not isinstance(name, str) or name not in _SCHEMAS:
        supported = ", ".join(sorted(_SCHEMAS))
        raise ValueError(f"unknown schema {name!r}; expected one of: {supported}")
    return json.loads(json.dumps(_SCHEMAS[name]))


def export_schemas(out_dir: str | Path) -> dict[str, Any]:
    """Export all Draft 2020-12 JSON Schemas into out_dir cleanly."""
    raw = str(out_dir)
    if not raw or any(ord(ch) < 32 for ch in raw):
        raise ValueError("out_dir must be a clean non-empty path")
    if any(part == ".." for part in re.split(r"[/\\]", raw)):
        raise ValueError("out_dir must not contain path traversal")
    dest = Path(out_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []
    for name in sorted(_SCHEMAS):
        target = dest / f"{name}.schema.json"
        target.write_text(json.dumps(_SCHEMAS[name], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        exported.append(str(target))
    return {"status": "EXPORTED", "out_dir": str(dest), "schemas": sorted(_SCHEMAS), "files": exported}
