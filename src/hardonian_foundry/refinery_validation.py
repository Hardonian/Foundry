"""Refinery dataset validation, code digesting, and path/identifier sanitization."""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

from .baseline import reconciliation, split
from .canonical import digest
from .types import ValidationResult

DEFAULT_RUNS_DIR = os.path.join(".foundry", "runs")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def facade_attr(name: str, fallback: Any) -> Any:
    """Resolve an attribute via hardonian_foundry.refinery when loaded, preserving test patches."""
    mod = sys.modules.get("hardonian_foundry.refinery")
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    return fallback


def _code_digest() -> str:
    """Compute a deterministic SHA-256 digest across all package Python modules."""
    pkg_dir = Path(__file__).resolve().parent
    components: list[str] = []
    for path in sorted(pkg_dir.glob("*.py"), key=lambda p: p.name):
        if path.is_file():
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            components.append(f"{path.name}:{file_hash}")
    return hashlib.sha256(":".join(components).encode("utf-8")).hexdigest()


def validate_run_id(run_id: Any) -> str:
    """Strictly validate run_id against path traversal, drive letters, and separators."""
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if ".." in run_id or "/" in run_id or "\\" in run_id or ":" in run_id or "\x00" in run_id:
        raise ValueError(f"invalid run_id (path traversal or separator rejected): {run_id!r}")
    if run_id.endswith(".") or not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"invalid run_id format: {run_id!r}")
    return run_id


def resolve_run_dir(root_dir: Path, run_id: str) -> Path:
    """Resolve root_dir / run_id safely, rejecting escapes and symlinks outside root_dir."""
    clean_id = validate_run_id(run_id)
    root_resolved = root_dir.resolve()
    run_dir = (root_resolved / clean_id).resolve()
    if not run_dir.is_relative_to(root_resolved) or run_dir == root_resolved:
        raise ValueError(f"run_id escapes output_root: {run_id!r}")
    return run_dir


def resolve_ledger_path(
    root_dir: Path,
    raw_ledger_path: Any,
    *,
    allow_legacy_outside_fallback: bool = False,
) -> Path:
    """Resolve evidence ledger path strictly confined within root_dir.

    New runs store relative "evidence.jsonl". Legacy committed runs (e.g. the
    signed case-study run) stored an absolute path ending in "evidence.jsonl";
    when allow_legacy_outside_fallback=True and the stored path is absent or an
    absolute legacy path ending in "evidence.jsonl" outside root_dir (without
    traversal), resolve root_dir / "evidence.jsonl" instead of reading outside.
    """
    root_resolved = root_dir.resolve()
    default_ledger = root_resolved / "evidence.jsonl"

    if raw_ledger_path is None or raw_ledger_path == "":
        if allow_legacy_outside_fallback:
            candidate = default_ledger.resolve()
            if not candidate.is_relative_to(root_resolved):
                raise ValueError("evidence ledger symlink escapes output_root")
            return candidate
        raise ValueError("evidence_ledger_path missing from run provenance")

    if not isinstance(raw_ledger_path, str) or any(ord(ch) < 32 for ch in raw_ledger_path):
        raise ValueError("evidence_ledger_path must be a clean string")

    parts = re.split(r"[/\\]", raw_ledger_path)
    if any(part == ".." for part in parts):
        raise ValueError("evidence_ledger_path contains path traversal")

    is_abs = (
        raw_ledger_path.startswith(("/", "\\"))
        or bool(PureWindowsPath(raw_ledger_path).drive)
        or Path(raw_ledger_path).is_absolute()
    )

    if not is_abs:
        if "\\" in raw_ledger_path or ":" in raw_ledger_path:
            raise ValueError("evidence_ledger_path must be a portable relative path")
        candidate = (root_resolved / raw_ledger_path).resolve()
        if not candidate.is_relative_to(root_resolved):
            raise ValueError("evidence_ledger_path escapes output_root")
        return candidate

    # Absolute path handling
    posix_name = PurePosixPath(raw_ledger_path.replace("\\", "/")).name
    candidate_abs = Path(raw_ledger_path).resolve()
    if candidate_abs.is_relative_to(root_resolved):
        return candidate_abs

    if allow_legacy_outside_fallback and posix_name == "evidence.jsonl":
        candidate = default_ledger.resolve()
        if not candidate.is_relative_to(root_resolved):
            raise ValueError("evidence ledger symlink escapes output_root")
        return candidate

    raise ValueError("evidence_ledger_path outside output_root is rejected")


def validate_and_split_dataset(
    records: Any,
    *,
    seed: int = 42,
) -> tuple[ValidationResult, dict[str, list[dict[str, Any]]] | None]:
    """Validate synthetic records and compute deterministic splits in a single validation pass."""
    errors: list[str] = []
    if not isinstance(records, list):
        return (
            {
                "status": "INVALID",
                "record_count": 0,
                "dataset_sha256": None,
                "split_counts": None,
                "errors": ["dataset must be a list of records"],
            },
            None,
        )
    if len(records) < 3:
        errors.append("dataset must contain at least three records for train/validation/test partitions")

    seen_ids: set[str] = set()
    seen_contents: set[str] = set()

    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            errors.append(f"record at index {idx} is not a dictionary")
            continue
        try:
            reconciliation(row)
        except ValueError as exc:
            errors.append(f"record at index {idx} ({row.get('id', 'unknown')}) failed validation: {exc}")
            continue

        row_id = row["id"]
        if row_id in seen_ids:
            errors.append(f"duplicate record id: {row_id}")
        else:
            seen_ids.add(row_id)

        content_digest = digest({k: v for k, v in row.items() if k != "id"})
        if content_digest in seen_contents:
            errors.append(f"duplicate record content detected (leakage hazard) for id {row_id}")
        else:
            seen_contents.add(content_digest)

    splits: dict[str, list[dict[str, Any]]] | None = None
    if not errors and len(records) >= 3:
        try:
            splits = split(records, seed=seed)
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"failed to partition dataset: {exc}")

    status: Literal["VALID", "INVALID"] = "VALID" if not errors else "INVALID"
    result: ValidationResult = {
        "status": status,
        "record_count": len(records),
        "dataset_sha256": digest(records) if not errors else None,
        "split_counts": {k: len(v) for k, v in splits.items()} if splits else None,
        "errors": sorted(errors),
    }
    return result, splits


def validate_dataset(records: Any, *, seed: int = 42) -> ValidationResult:
    """Validate a dataset against Foundry synthetic schema and split requirements.

    Enforces synthetic data origins, schema fields, bounded integer minor units,
    unique IDs, unique content (leakage detection), and deterministic splitability.
    """
    result, _ = validate_and_split_dataset(records, seed=seed)
    return result
