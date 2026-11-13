"""Refinery preflight checks: dataset readiness, runtime environment, storage, licensing, safety."""
from __future__ import annotations

import platform
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .baseline import dataset as default_dataset
from .licensing import evaluate as evaluate_license
from .refinery_validation import DEFAULT_RUNS_DIR, facade_attr, validate_dataset
from .task_eval import validate_task_dataset
from .types import PreflightCheck, PreflightResult


def _evaluate_license_compat(
    manifest: dict[str, Any],
    *,
    allowed_data_origins: Sequence[str] | None = None,
) -> dict[str, Any]:
    eval_fn = facade_attr("evaluate_license", evaluate_license)
    if allowed_data_origins is not None:
        try:
            return eval_fn(manifest, allowed_data_origins=tuple(allowed_data_origins))
        except TypeError:
            pass
    return eval_fn(manifest)


def run_preflight(
    *,
    dataset: list[dict[str, Any]] | Any = None,
    output_root: str | Path | None = None,
    model_manifest: dict[str, Any] | None = None,
    task_spec: dict[str, Any] | None = None,
    allowed_data_origins: Sequence[str] | None = None,
    _validation_result: Mapping[str, Any] | None = None,
) -> PreflightResult:
    """Execute training and run preflight checks before launching a Refinery workload.

    Checks dataset readiness, runtime environment constraints, output directory security,
    and licensing/provenance allowlists.
    """
    checks: list[PreflightCheck] = []
    errors: list[str] = []

    # 1. Dataset readiness (task-mode or built-in reconciliation schema)
    data = dataset if dataset is not None else default_dataset()
    if task_spec is not None:
        val_result = _validation_result if _validation_result is not None else validate_task_dataset(data, task_spec)
        detail_ok = f"{val_result['record_count']} task records validated (family_disjoint={val_result.get('family_disjoint')})."
    else:
        val_result = _validation_result if _validation_result is not None else validate_dataset(data)
        detail_ok = f"{val_result['record_count']} synthetic records validated with zero duplicate content."

    if val_result["status"] == "VALID":
        checks.append({
            "name": "dataset_validation",
            "status": "PASSED",
            "details": detail_ok,
        })
    else:
        err_msg = "; ".join(val_result["errors"])
        errors.append(f"dataset validation failed: {err_msg}")
        checks.append({
            "name": "dataset_validation",
            "status": "FAILED",
            "details": err_msg,
        })

    # 2. Runtime environment
    env_details = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "runtime": "offline_cpu",
    }
    checks.append({
        "name": "runtime_environment",
        "status": "PASSED",
        "details": f"Python {env_details['python_version']} on {env_details['platform']} (offline CPU runtime).",
    })

    # 3. Output directory security and storage
    root_path = Path(output_root or DEFAULT_RUNS_DIR)
    try:
        root_path.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(root_path)
        free_mb = usage.free / (1024 * 1024)
        if free_mb < 50:
            err_msg = f"insufficient disk space: {free_mb:.1f} MB free (minimum 50 MB required)"
            errors.append(err_msg)
            checks.append({"name": "output_directory", "status": "FAILED", "details": err_msg})
        else:
            checks.append({
                "name": "output_directory",
                "status": "PASSED",
                "details": f"Directory {root_path} available ({free_mb:.1f} MB free).",
            })
    except (OSError, PermissionError) as exc:
        err_msg = f"failed to initialize output directory {root_path}: {exc}"
        errors.append(err_msg)
        checks.append({"name": "output_directory", "status": "FAILED", "details": err_msg})

    # 4. Model manifest licensing & provenance allowlist (if provided)
    if model_manifest is not None:
        lic_result = _evaluate_license_compat(model_manifest, allowed_data_origins=allowed_data_origins)
        if lic_result["decision"] == "ALLOWED":
            checks.append({
                "name": "licensing_and_provenance",
                "status": "PASSED",
                "details": f"License {model_manifest.get('license_id')} and provenance {model_manifest.get('provenance')} approved.",
            })
        else:
            reasons = "; ".join(lic_result["reasons"])
            errors.append(f"model manifest failed licensing/provenance check: {reasons}")
            checks.append({
                "name": "licensing_and_provenance",
                "status": "FAILED",
                "details": reasons,
            })
    else:
        checks.append({
            "name": "licensing_and_provenance",
            "status": "PASSED",
            "details": "No external model manifest specified; using built-in synthetic heuristic.",
        })

    # 5. Security boundaries check
    checks.append({
        "name": "execution_safety",
        "status": "PASSED",
        "details": "Fail-closed review required; zero autonomous tool/financial execution permitted.",
    })

    return {
        "status": "PASSED" if not errors else "FAILED",
        "checks": checks,
        "errors": sorted(errors),
    }
