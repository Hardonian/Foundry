"""Task-mode evaluation: spec-driven classification datasets scored from frozen predictions.

Pure stdlib, offline, synthetic-only. Predictions are produced EXTERNALLY (by a
node harness, a torch eval script, etc.) and handed to Foundry as JSONL files;
Foundry validates, splits family-disjointly, scores deterministically, and
binds everything into the evidence ledger. Foundry never executes a model.

Record schema:   {"id", "prompt", "gold_label", "critical", "family"?}
Prediction:      {"id", "label", "confidence"?, "latency_ms"?, "raw_invalid"?, "abstained"?}
Spec:            see validate_spec()
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

try:
    from .canonical import digest  # type: ignore[import-not-found,import-untyped]
except ImportError:  # pragma: no cover - pre-merge fallback
    from .baseline import digest

from .types import (
    CalibrationMetrics,
    FamilyMetrics,
    PerClassMetrics,
    TaskDatasetValidationResult,
    TaskScoreMetrics,
)

MAX_ID = 128
MAX_PROMPT = 4000
MAX_LABEL = 64
MAX_LABELS = 64
MAX_NOTES = 1024
HEX64_LEN = 64

ALLOWED_SPEC_FIELDS = frozenset({
    "schema_version",
    "task_id",
    "description",
    "labels",
    "thresholds",
    "baseline_must_improve",
    "split_seed",
    "model",
    "dataset_path",
    "notes",
})

ALLOWED_THRESHOLD_FIELDS = frozenset({
    "macro_f1_min",
    "critical_fn_rate_max",
    "per_class_recall_min",
    "invalid_output_rate_max",
    "latency_p95_ms_max",
    "ood_challenge_predictions",
})

ALLOWED_PREDICTION_FIELDS = frozenset({
    "id",
    "label",
    "confidence",
    "latency_ms",
    "raw_invalid",
    "abstained",
})


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_clean_relative_path(path_value: Any, *, max_len: int = 1024) -> bool:
    """Reject empty, oversized, absolute, Windows-drive, backslash, or traversal paths."""
    if not isinstance(path_value, str) or not path_value or len(path_value) > max_len:
        return False
    if any(ord(ch) < 32 for ch in path_value):
        return False
    if path_value.startswith(("/", "\\")) or "\\" in path_value or ":" in path_value:
        return False
    if PurePosixPath(path_value).is_absolute() or PureWindowsPath(path_value).is_absolute():
        return False
    if PureWindowsPath(path_value).drive:
        return False
    parts = re.split(r"[/\\]", path_value)
    return not any(part == ".." or part == "" for part in parts)


def _validate_thresholds(thresholds: Any, labels: list[str] | Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(thresholds, dict) or "macro_f1_min" not in thresholds:
        return ["thresholds must be an object containing macro_f1_min"]
    unknown = set(thresholds) - ALLOWED_THRESHOLD_FIELDS
    if unknown:
        errors.append(f"unknown threshold fields: {sorted(unknown)}")

    m = thresholds["macro_f1_min"]
    if not _is_finite_number(m) or not (0.8 <= float(m) <= 1.0):
        errors.append("macro_f1_min must be a number in [0.8, 1.0]")

    for rate_key in ("critical_fn_rate_max", "invalid_output_rate_max"):
        if rate_key in thresholds:
            val = thresholds[rate_key]
            if not _is_finite_number(val) or not (0.0 <= float(val) <= 1.0):
                errors.append(f"thresholds.{rate_key} must be a number in [0.0, 1.0]")

    if "latency_p95_ms_max" in thresholds:
        lat = thresholds["latency_p95_ms_max"]
        if not _is_finite_number(lat) or not (0.0 < float(lat) <= 600_000.0):
            errors.append("thresholds.latency_p95_ms_max must be a positive number <= 600000")

    if "per_class_recall_min" in thresholds:
        pcr = thresholds["per_class_recall_min"]
        if _is_finite_number(pcr):
            if not (0.0 <= float(pcr) <= 1.0):
                errors.append("thresholds.per_class_recall_min must be in [0.0, 1.0]")
        elif isinstance(pcr, dict) and pcr:
            label_set = set(labels) if isinstance(labels, list) else set()
            for k, v in pcr.items():
                if not isinstance(k, str) or (label_set and k not in label_set):
                    errors.append(f"thresholds.per_class_recall_min contains invalid label {k!r}")
                if not _is_finite_number(v) or not (0.0 <= float(v) <= 1.0):
                    errors.append(f"thresholds.per_class_recall_min[{k!r}] must be a number in [0.0, 1.0]")
        else:
            errors.append("thresholds.per_class_recall_min must be a number in [0.0, 1.0] or a label map")

    if "ood_challenge_predictions" in thresholds:
        ood = thresholds["ood_challenge_predictions"]
        if isinstance(ood, bool) or isinstance(ood, int) and ood >= 1:
            pass
        elif isinstance(ood, str):
            if not _is_clean_relative_path(ood):
                errors.append("thresholds.ood_challenge_predictions path must be relative and traversal-free")
        elif isinstance(ood, dict):
            allowed_ood = {"min_count", "min_accuracy", "macro_f1_min", "dataset_path", "predictions_path"}
            if set(ood) - allowed_ood:
                errors.append(f"unknown thresholds.ood_challenge_predictions fields: {sorted(set(ood) - allowed_ood)}")
            if "min_count" in ood and (isinstance(ood["min_count"], bool) or not isinstance(ood["min_count"], int) or ood["min_count"] < 1):
                errors.append("thresholds.ood_challenge_predictions.min_count must be an integer >= 1")
            for rfield in ("macro_f1_min", "min_accuracy"):
                if rfield in ood and (not _is_finite_number(ood[rfield]) or not (0.0 <= float(ood[rfield]) <= 1.0)):
                    errors.append(f"thresholds.ood_challenge_predictions.{rfield} must be in [0.0, 1.0]")
            for pfield in ("dataset_path", "predictions_path"):
                if pfield in ood and not _is_clean_relative_path(ood[pfield]):
                    errors.append(f"thresholds.ood_challenge_predictions.{pfield} must be relative and traversal-free")
        else:
            errors.append("thresholds.ood_challenge_predictions must be bool, positive int, relative path, or object")

    return errors


def validate_spec(spec: Any) -> list[str]:
    """Return sorted list of spec errors; empty list means valid. Fail-closed."""
    errors: list[str] = []
    if not isinstance(spec, dict):
        return ["task spec must be a JSON object"]
    for key in spec:
        if key not in ALLOWED_SPEC_FIELDS:
            errors.append(f"unknown spec field: {key}")
    if spec.get("schema_version") != "1.0":
        errors.append("spec schema_version must be '1.0'")
    tid = spec.get("task_id")
    if not isinstance(tid, str) or not (1 <= len(tid) <= 64) or any(c.isspace() for c in tid):
        errors.append("task_id must be a 1-64 char non-space string")
    desc = spec.get("description")
    if not isinstance(desc, str) or not desc or len(desc) > 512:
        errors.append("description must be a nonempty string <= 512 chars")
    labels = spec.get("labels")
    if not isinstance(labels, list) or not (2 <= len(labels) <= MAX_LABELS):
        errors.append("labels must be a list of 2..64 entries")
    else:
        if any(not isinstance(lbl, str) or not lbl or len(lbl) > MAX_LABEL or lbl != lbl.strip() for lbl in labels):
            errors.append("each label must be a clean nonempty string <= 64 chars")
        if len(set(labels)) != len(labels):
            errors.append("labels must be unique")
    errors.extend(_validate_thresholds(spec.get("thresholds"), labels))
    if not isinstance(spec.get("baseline_must_improve", True), bool):
        errors.append("baseline_must_improve must be a boolean when present")
    seed = spec.get("split_seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        errors.append("split_seed must be an integer")
    model = spec.get("model")
    if not isinstance(model, dict) or set(model) != {"base_model_id", "revision", "adapter_sha256", "adapter_name"}:
        errors.append("model must contain exactly base_model_id, revision, adapter_sha256, adapter_name")
    else:
        for key, limit in (("base_model_id", 256), ("revision", 128), ("adapter_name", 128)):
            val = model[key]
            if not isinstance(val, str) or not val or len(val) != len(val.strip()) or len(val) > limit or any(ord(c) < 32 for c in val):
                errors.append(f"model.{key} must be a clean nonempty string <= {limit} chars")
        sha = model["adapter_sha256"]
        if not isinstance(sha, str) or len(sha) != HEX64_LEN or any(c not in "0123456789abcdef" for c in sha):
            errors.append("model.adapter_sha256 must be 64-char lowercase hex")
    if "dataset_path" in spec:
        dp = spec["dataset_path"]
        if not _is_clean_relative_path(dp):
            errors.append("dataset_path must be a relative, traversal-free path <= 1024 chars")
    if "notes" in spec:
        notes = spec["notes"]
        if not isinstance(notes, str) or len(notes) > MAX_NOTES or any(ord(c) < 32 and c not in "\n\r\t" for c in notes):
            errors.append(f"notes must be a clean string <= {MAX_NOTES} chars")
    return sorted(set(errors))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts; fail closed on any bad line."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {lineno} of {path}: {exc}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"line {lineno} of {path} is not a JSON object")
            records.append(value)
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def validate_task_dataset(records: Any, spec: dict[str, Any]) -> TaskDatasetValidationResult:
    """Validate task records against the spec. Returns dict with status + errors."""
    errors: list[str] = []
    if not isinstance(records, list):
        return {
            "status": "INVALID",
            "record_count": 0,
            "dataset_sha256": None,
            "errors": ["dataset must be a list"],
        }
    if len(records) < 9:
        errors.append("task dataset must contain at least 9 records")
    label_set = set(spec.get("labels", []))
    seen_ids: set[str] = set()
    seen_prompts: dict[str, str] = {}
    families: set[str] = set()
    have_family: bool | None = None
    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            errors.append(f"record at index {idx} is not an object")
            continue
        present = set(row)
        if not {"id", "prompt", "gold_label", "critical"} <= present:
            errors.append(
                f"record {row.get('id', idx)}: missing required fields {sorted({'id', 'prompt', 'gold_label', 'critical'} - present)}"
            )
            continue
        extra = present - {"id", "prompt", "gold_label", "critical", "family"}
        if extra:
            errors.append(f"record {row.get('id')}: unknown fields {sorted(extra)}")
        rid = row["id"]
        if not isinstance(rid, str) or not rid or len(rid) > MAX_ID:
            errors.append(f"record at index {idx}: id must be a 1-{MAX_ID} char string")
        elif rid in seen_ids:
            errors.append(f"duplicate record id: {rid}")
        else:
            seen_ids.add(rid)
        prompt = row["prompt"]
        if not isinstance(prompt, str) or not (1 <= len(prompt) <= MAX_PROMPT):
            errors.append(f"record {rid}: prompt must be a 1-{MAX_PROMPT} char string")
        else:
            fingerprint = hashlib.sha256(prompt.encode()).hexdigest()
            if fingerprint in seen_prompts and prompt == seen_prompts[fingerprint]:
                errors.append(f"duplicate prompt content (leakage hazard) for id {rid}")
            else:
                seen_prompts[fingerprint] = prompt
        gold = row["gold_label"]
        if not isinstance(gold, str) or gold not in label_set:
            errors.append(f"record {rid}: gold_label {gold!r} not in spec labels")
        if not isinstance(row["critical"], bool):
            errors.append(f"record {rid}: critical must be boolean")
        fam = row.get("family")
        if fam is not None:
            if not isinstance(fam, str) or not fam or len(fam) > 64:
                errors.append(f"record {rid}: family must be a 1-64 char string")
            else:
                families.add(fam)
        if have_family is None:
            have_family = fam is not None
        elif have_family != (fam is not None):
            errors.append(f"record {rid}: 'family' must be present on all records or none")
    status: Literal["VALID", "INVALID"] = "VALID" if not errors else "INVALID"
    return {
        "status": status,
        "record_count": len(records),
        "dataset_sha256": digest(records) if not errors else None,
        "family_disjoint": bool(have_family),
        "family_count": len(families),
        "errors": sorted(errors),
    }


def _digest_order(keys: Sequence[str], seed: int) -> list[str]:
    return sorted(keys, key=lambda k: (hashlib.sha256(f"{k}|{seed}".encode()).hexdigest(), k))


def split_task_dataset(records: Sequence[dict[str, Any]], spec: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Deterministic group-disjoint split (~60/15/25 by GROUP count, so every
    partition is nonempty and no family (or singleton record) ever crosses splits)."""
    seed = spec.get("split_seed", 42)
    use_family = all("family" in r for r in records)
    groups: dict[str, list[dict[str, Any]]] = {}
    if use_family:
        for row in records:
            groups.setdefault(row["family"], []).append(row)
    else:
        for row in records:
            groups[row["id"]] = [row]
    if len(groups) < 3:
        raise ValueError("need at least three families (or records) for a three-way disjoint split")
    for members in groups.values():
        members.sort(key=lambda r: r["id"])
    ordered_keys = _digest_order(list(groups), seed)
    n_groups = len(ordered_keys)
    train_count = min(max(1, int(n_groups * 0.6)), n_groups - 2)
    validation_count = max(1, int(n_groups * 0.15))
    validation_count = min(validation_count, n_groups - train_count - 1)
    split_at = (train_count, train_count + validation_count)
    buckets = {
        "train": ordered_keys[:split_at[0]],
        "validation": ordered_keys[split_at[0]:split_at[1]],
        "test": ordered_keys[split_at[1]:],
    }
    return {
        name: sorted([row for key in keys for row in groups[key]], key=lambda r: r["id"])
        for name, keys in buckets.items()
    }


def _nearest_rank_percentile(values: Sequence[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(pct * len(ordered)) - 1))
    return float(ordered[idx])


def _compute_calibration(confidences: Sequence[float], correctness: Sequence[bool], *, bins: int = 10) -> CalibrationMetrics:
    n = len(confidences)
    bin_counts = [0] * bins
    bin_conf_sums = [0.0] * bins
    bin_acc_sums = [0.0] * bins
    brier_sum = 0.0
    conf_sum = 0.0
    for conf, ok in zip(confidences, correctness, strict=True):
        target = 1.0 if ok else 0.0
        brier_sum += (conf - target) ** 2
        conf_sum += conf
        b_idx = min(bins - 1, int(conf * bins))
        bin_counts[b_idx] += 1
        bin_conf_sums[b_idx] += conf
        bin_acc_sums[b_idx] += target
    ece = 0.0
    for count, c_sum, a_sum in zip(bin_counts, bin_conf_sums, bin_acc_sums, strict=True):
        if count > 0:
            ece += (count / n) * abs((a_sum / count) - (c_sum / count))
    return {
        "ece": ece,
        "brier_score": brier_sum / n,
        "mean_confidence": conf_sum / n,
        "bins": bins,
    }


def _validate_predictions_map(
    test_rows: Sequence[dict[str, Any]],
    predictions: Any,
    label_set: set[str],
) -> tuple[dict[str, dict[str, Any]], bool, bool]:
    if not isinstance(predictions, (list, tuple)):
        raise TypeError("predictions must be a list of prediction objects")
    gold_ids = {row["id"] for row in test_rows}
    preds: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    saw_conf = 0
    saw_lat = 0

    for idx, row in enumerate(predictions):
        if not isinstance(row, dict):
            errors.append(f"prediction at index {idx} must be a JSON object")
            continue
        present = set(row)
        if not {"id", "label"} <= present:
            errors.append(f"prediction at index {idx} missing required id/label fields: {sorted(present)}")
            continue
        extra = present - ALLOWED_PREDICTION_FIELDS
        if extra:
            errors.append(f"prediction fields must be id,label (optional confidence,latency_ms,raw_invalid,abstained): unknown {sorted(extra)}")
            continue
        pid, plabel = row["id"], row["label"]
        if not isinstance(pid, str) or not pid:
            errors.append(f"prediction at index {idx}: id must be a nonempty string")
            continue
        if pid in preds:
            errors.append(f"duplicate prediction id: {pid}")
            continue
        if not isinstance(plabel, str) or plabel not in label_set:
            errors.append(f"prediction {pid}: label {plabel!r} not in spec labels")
            continue
        if "confidence" in row:
            conf = row["confidence"]
            if not _is_finite_number(conf) or not (0.0 <= float(conf) <= 1.0):
                errors.append(f"prediction {pid}: confidence must be a number in [0.0, 1.0]")
                continue
            saw_conf += 1
        if "latency_ms" in row:
            lat = row["latency_ms"]
            if not _is_finite_number(lat) or float(lat) < 0.0:
                errors.append(f"prediction {pid}: latency_ms must be a non-negative finite number")
                continue
            saw_lat += 1
        if "raw_invalid" in row and not isinstance(row["raw_invalid"], bool):
            errors.append(f"prediction {pid}: raw_invalid must be boolean")
            continue
        if "abstained" in row and not isinstance(row["abstained"], bool):
            errors.append(f"prediction {pid}: abstained must be boolean")
            continue
        preds[pid] = row

    missing = sorted(gold_ids - set(preds))
    extra_ids = sorted(set(preds) - gold_ids)
    if missing:
        errors.append(f"predictions missing {len(missing)} test ids (first 5: {missing[:5]})")
    if extra_ids:
        errors.append(f"predictions cover {len(extra_ids)} unknown ids (first 5: {extra_ids[:5]})")
    if not errors and preds:
        if 0 < saw_conf < len(preds):
            errors.append("confidence must be present on all predictions or none")
        if 0 < saw_lat < len(preds):
            errors.append("latency_ms must be present on all predictions or none")
    if errors:
        raise ValueError("; ".join(sorted(errors)))
    return preds, saw_conf == len(preds) and len(preds) > 0, saw_lat == len(preds) and len(preds) > 0


def _score_from_maps(
    test_rows: Sequence[dict[str, Any]],
    preds: dict[str, dict[str, Any]],
    sorted_labels: list[str],
    *,
    has_confidence: bool,
    has_latency: bool,
) -> TaskScoreMetrics:
    tp_counts: dict[str, int] = dict.fromkeys(sorted_labels, 0)
    fp_counts: dict[str, int] = dict.fromkeys(sorted_labels, 0)
    fn_counts: dict[str, int] = dict.fromkeys(sorted_labels, 0)
    confusion_matrix: dict[str, dict[str, int]] = {
        gold_lbl: dict.fromkeys(sorted_labels, 0) for gold_lbl in sorted_labels
    }
    family_acc: dict[str, dict[str, int]] = {}
    labels_present_set: set[str] = set()

    correct = 0
    critical_total = 0
    critical_misses = 0
    invalid_count = 0
    abstention_count = 0
    non_abstained_total = 0
    non_abstained_correct = 0
    confidences: list[float] = []
    correctness: list[bool] = []
    latencies: list[float] = []

    for row in test_rows:
        rid = row["id"]
        gold_lbl = row["gold_label"]
        pred_row = preds[rid]
        pred_lbl = pred_row["label"]
        is_correct = pred_lbl == gold_lbl

        labels_present_set.add(gold_lbl)
        confusion_matrix[gold_lbl][pred_lbl] += 1
        if is_correct:
            tp_counts[gold_lbl] += 1
            correct += 1
        else:
            fn_counts[gold_lbl] += 1
            fp_counts[pred_lbl] += 1

        if row["critical"]:
            critical_total += 1
            if not is_correct:
                critical_misses += 1

        fam = row.get("family")
        if fam is not None:
            fstat = family_acc.get(fam)
            if fstat is None:
                fstat = {"count": 0, "correct": 0, "critical_total": 0, "critical_misses": 0}
                family_acc[fam] = fstat
            fstat["count"] += 1
            if is_correct:
                fstat["correct"] += 1
            if row["critical"]:
                fstat["critical_total"] += 1
                if not is_correct:
                    fstat["critical_misses"] += 1

        if pred_row.get("raw_invalid", False):
            invalid_count += 1
        abstained = bool(pred_row.get("abstained", False)) or pred_lbl == "UNDETERMINED"
        if abstained:
            abstention_count += 1
        else:
            non_abstained_total += 1
            if is_correct:
                non_abstained_correct += 1

        if has_confidence:
            confidences.append(float(pred_row["confidence"]))
            correctness.append(is_correct)
        if has_latency:
            latencies.append(float(pred_row["latency_ms"]))

    per_class: dict[str, PerClassMetrics] = {}
    macro_sum = 0.0
    for label in sorted_labels:
        tp = tp_counts[label]
        fp = fp_counts[label]
        fn = fn_counts[label]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }
        macro_sum += f1

    per_family: dict[str, FamilyMetrics] = {}
    for fam in sorted(family_acc):
        fstat = family_acc[fam]
        ct = fstat["critical_total"]
        cm = fstat["critical_misses"]
        per_family[fam] = {
            "count": fstat["count"],
            "correct": fstat["correct"],
            "accuracy": fstat["correct"] / fstat["count"],
            "critical_total": ct,
            "critical_misses": cm,
            "critical_false_negative_rate": (cm / ct) if ct else None,
        }

    n_test = len(test_rows)
    macro_f1 = macro_sum / len(sorted_labels) if sorted_labels else None
    critical_fn_rate = (critical_misses / critical_total) if critical_total else None
    calibration = _compute_calibration(confidences, correctness) if has_confidence else None
    latency_p95_ms = _nearest_rank_percentile(latencies, 0.95) if has_latency else None

    canonical_preds = []
    for pid in sorted(preds):
        row_obj: dict[str, Any] = {"id": pid, "label": preds[pid]["label"]}
        for opt_key in ("confidence", "latency_ms", "raw_invalid", "abstained"):
            if opt_key in preds[pid]:
                row_obj[opt_key] = preds[pid][opt_key]
        canonical_preds.append(row_obj)

    return {
        "test_count": n_test,
        "accuracy": (correct / n_test) if n_test else None,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
        "per_family": per_family,
        "critical_total": critical_total,
        "critical_misses": critical_misses,
        "critical_false_negative_rate": critical_fn_rate,
        "labels_scored": sorted(labels_present_set),
        "predictions_sha256": digest(canonical_preds),
        "invalid_output_count": invalid_count,
        "invalid_output_rate": (invalid_count / n_test) if n_test else None,
        "abstention_count": abstention_count,
        "abstention_rate": (abstention_count / n_test) if n_test else None,
        "selective_accuracy": (non_abstained_correct / non_abstained_total) if non_abstained_total else None,
        "calibration": calibration,
        "latency_p95_ms": latency_p95_ms,
    }


def score_predictions(
    test_rows: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    spec: dict[str, Any],
) -> TaskScoreMetrics:
    """Score prediction records against the frozen test split in a single O(|T|) pass."""
    if not isinstance(test_rows, (list, tuple)):
        raise TypeError("test_rows must be a sequence")
    sorted_labels = sorted(set(spec["labels"]))
    preds, has_confidence, has_latency = _validate_predictions_map(test_rows, predictions, set(sorted_labels))
    return _score_from_maps(
        test_rows,
        preds,
        sorted_labels,
        has_confidence=has_confidence,
        has_latency=has_latency,
    )


def bootstrap_ci(
    test_rows: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    spec: dict[str, Any],
    *,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    seeds: Sequence[int] = (42,),
) -> dict[str, Any]:
    """Compute deterministic multi-seed bootstrap confidence intervals for macro_f1 and accuracy."""
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 10:
        raise ValueError("n_bootstrap must be an integer >= 10")
    if not _is_finite_number(confidence) or not (0.5 <= float(confidence) < 1.0):
        raise ValueError("confidence must be in [0.5, 1.0)")
    if not seeds or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds):
        raise ValueError("seeds must be a non-empty sequence of integers")
    if not test_rows:
        raise ValueError("test_rows cannot be empty")

    sorted_labels = sorted(set(spec["labels"]))
    preds, has_conf, has_lat = _validate_predictions_map(test_rows, predictions, set(sorted_labels))
    point = _score_from_maps(test_rows, preds, sorted_labels, has_confidence=has_conf, has_latency=has_lat)

    n = len(test_rows)
    alpha = (1.0 - float(confidence)) / 2.0
    all_macro_f1: list[float] = []
    all_acc: list[float] = []
    per_seed: dict[str, dict[str, Any]] = {}

    def _summarize(samples: list[float], point_val: float | None) -> dict[str, float | None]:
        ordered = sorted(samples)
        lo_idx = max(0, math.floor(alpha * len(ordered)))
        hi_idx = min(len(ordered) - 1, math.ceil((1.0 - alpha) * len(ordered)) - 1)
        mean_val = sum(ordered) / len(ordered)
        var = sum((x - mean_val) ** 2 for x in ordered) / len(ordered)
        return {
            "point": point_val,
            "mean": mean_val,
            "std": math.sqrt(var),
            "ci_low": ordered[lo_idx],
            "ci_high": ordered[hi_idx],
        }

    for seed in seeds:
        rng = random.Random(seed)
        seed_f1: list[float] = []
        seed_acc: list[float] = []
        for _ in range(n_bootstrap):
            sample = [test_rows[rng.randrange(n)] for _ in range(n)]
            res = _score_from_maps(sample, preds, sorted_labels, has_confidence=False, has_latency=False)
            seed_f1.append(float(res["macro_f1"] or 0.0))
            seed_acc.append(float(res["accuracy"] or 0.0))
        all_macro_f1.extend(seed_f1)
        all_acc.extend(seed_acc)
        per_seed[str(seed)] = {
            "macro_f1": _summarize(seed_f1, point["macro_f1"]),
            "accuracy": _summarize(seed_acc, point["accuracy"]),
        }

    return {
        "n_test": n,
        "n_bootstrap": n_bootstrap,
        "confidence": float(confidence),
        "seeds": list(seeds),
        "macro_f1": _summarize(all_macro_f1, point["macro_f1"]),
        "accuracy": _summarize(all_acc, point["accuracy"]),
        "per_seed": per_seed,
    }


def mcnemar_test(
    test_rows: Sequence[dict[str, Any]],
    baseline_predictions: Sequence[dict[str, Any]],
    model_predictions: Sequence[dict[str, Any]],
    *,
    exact_threshold: int = 25,
) -> dict[str, Any]:
    """Compare two predictors on paired test_rows using McNemar's test (exact binomial or chi2)."""
    if not isinstance(test_rows, (list, tuple)) or not test_rows:
        raise ValueError("test_rows must be a non-empty sequence")
    if not isinstance(baseline_predictions, (list, tuple)) or not isinstance(model_predictions, (list, tuple)):
        raise TypeError("predictions must be sequences of {id, label} objects")

    def _to_label_map(rows: Sequence[dict[str, Any]], name: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for idx, row in enumerate(rows):
            if not isinstance(row, dict) or "id" not in row or "label" not in row:
                raise ValueError(f"{name} prediction at index {idx} must have id and label")
            pid = row["id"]
            if pid in out:
                raise ValueError(f"duplicate {name} prediction id: {pid}")
            out[pid] = row["label"]
        return out

    base_map = _to_label_map(baseline_predictions, "baseline")
    mod_map = _to_label_map(model_predictions, "model")
    gold_ids = {row["id"] for row in test_rows}
    if set(base_map) != gold_ids or set(mod_map) != gold_ids:
        raise ValueError("baseline and model predictions must match test_rows ids exactly")

    n00 = n01 = n10 = n11 = 0
    for row in test_rows:
        rid = row["id"]
        gold = row["gold_label"]
        b_ok = base_map[rid] == gold
        m_ok = mod_map[rid] == gold
        if b_ok and m_ok:
            n11 += 1
        elif b_ok and not m_ok:
            n10 += 1
        elif not b_ok and m_ok:
            n01 += 1
        else:
            n00 += 1

    discordant = n10 + n01
    if discordant == 0:
        statistic = 0.0
        p_value = 1.0
        exact = True
    else:
        diff = abs(n10 - n01)
        statistic = ((diff - 1) ** 2) / discordant if diff >= 1 else 0.0
        if discordant <= exact_threshold:
            k_min = min(n10, n01)
            tail = sum(math.comb(discordant, k) for k in range(k_min + 1)) * (0.5 ** discordant)
            p_value = min(1.0, 2.0 * tail)
            exact = True
        else:
            p_value = math.erfc(math.sqrt(statistic / 2.0))
            exact = False

    return {
        "n": len(test_rows),
        "both_correct": n11,
        "both_wrong": n00,
        "baseline_only_correct": n10,
        "model_only_correct": n01,
        "discordant": discordant,
        "statistic": statistic,
        "p_value": p_value,
        "exact": exact,
        "significant_at_0_05": p_value < 0.05,
    }


def load_predictor_dataset(spec: dict[str, Any], base_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    """Resolve spec.dataset_path relative to the spec file directory; reject escapes."""
    rel = spec.get("dataset_path")
    if not rel or not _is_clean_relative_path(rel):
        raise ValueError("spec.dataset_path must be a relative, traversal-free path")
    base_resolved = base_dir.resolve()
    resolved = (base_resolved / rel).resolve()
    if not resolved.is_relative_to(base_resolved):
        raise ValueError("dataset_path escapes the spec directory")
    return load_jsonl(str(resolved)), resolved
