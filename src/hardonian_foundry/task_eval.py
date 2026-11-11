"""Task-mode evaluation: spec-driven classification datasets scored from frozen predictions.

Pure stdlib, offline, synthetic-only. Predictions are produced EXTERNALLY (by a
node harness, a torch eval script, etc.) and handed to Foundry as JSONL files;
Foundry validates, splits family-disjointly, scores deterministically, and
binds everything into the evidence ledger. Foundry never executes a model.

Record schema:   {"id", "prompt", "gold_label", "critical", "family"?}
Prediction:      {"id", "label"}
Spec:            see validate_spec()
"""
import hashlib
import json

from .baseline import digest

MAX_ID = 128
MAX_PROMPT = 4000
MAX_LABEL = 64
MAX_LABELS = 64
HEX64_LEN = 64


def validate_spec(spec):
    """Return sorted list of spec errors; empty list means valid. Fail-closed."""
    errors = []
    if not isinstance(spec, dict):
        return ["task spec must be a JSON object"]
    allowed = {"schema_version", "task_id", "description", "labels", "thresholds",
               "baseline_must_improve", "split_seed", "model", "dataset_path", "notes"}
    for key in spec:
        if key not in allowed:
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
        if any(not isinstance(l, str) or not l or len(l) > MAX_LABEL or l != l.strip() for l in labels):
            errors.append("each label must be a clean nonempty string <= 64 chars")
        if len(set(labels)) != len(labels):
            errors.append("labels must be unique")
    thresholds = spec.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != {"macro_f1_min"}:
        errors.append("thresholds must be an object with exactly macro_f1_min")
    else:
        m = thresholds["macro_f1_min"]
        if isinstance(m, bool) or not isinstance(m, (int, float)) or not (0.8 <= m <= 1.0):
            errors.append("macro_f1_min must be a number in [0.8, 1.0]")
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
    dp = spec.get("dataset_path")
    if dp is not None and (not isinstance(dp, str) or not dp or len(dp) > 1024 or dp.startswith("/") or ".." in dp.split("/")):
        errors.append("dataset_path must be a relative, traversal-free path <= 1024 chars")
    return sorted(errors)


def load_jsonl(path):
    """Load a JSONL file into a list of dicts; fail closed on any bad line."""
    records = []
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


def validate_task_dataset(records, spec):
    """Validate task records against the spec. Returns dict with status + errors."""
    errors = []
    if not isinstance(records, list):
        return {"status": "INVALID", "record_count": 0, "dataset_sha256": None, "errors": ["dataset must be a list"]}
    if len(records) < 9:
        errors.append("task dataset must contain at least 9 records")
    label_set = set(spec.get("labels", []))
    seen_ids, seen_prompts, families = set(), {}, set()
    have_family = None
    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            errors.append(f"record at index {idx} is not an object")
            continue
        present = set(row)
        if not {"id", "prompt", "gold_label", "critical"} <= present:
            errors.append(f"record {row.get('id', idx)}: missing required fields {sorted({'id','prompt','gold_label','critical'} - present)}")
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
        if gold not in label_set:
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
    status = "VALID" if not errors else "INVALID"
    return {
        "status": status,
        "record_count": len(records),
        "dataset_sha256": digest(records) if not errors else None,
        "family_disjoint": bool(have_family),
        "family_count": len(families),
        "errors": sorted(errors),
    }


def _digest_order(keys, seed):
    return sorted(keys, key=lambda k: (hashlib.sha256(f"{k}|{seed}".encode()).hexdigest(), k))


def split_task_dataset(records, spec):
    """Deterministic group-disjoint split (~60/15/25 by GROUP count, so every
    partition is nonempty and no family (or singleton record) ever crosses splits)."""
    seed = spec.get("split_seed", 42)
    use_family = all("family" in r for r in records)
    groups = {}
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
    train_count = max(1, int(n_groups * 0.6))
    train_count = min(train_count, n_groups - 2)
    validation_count = max(1, n_groups - train_count - 1)
    split_at = (train_count, train_count + validation_count)
    buckets = {"train": ordered_keys[:split_at[0]],
               "validation": ordered_keys[split_at[0]:split_at[1]],
               "test": ordered_keys[split_at[1]:]}
    return {name: sorted([row for key in keys for row in groups[key]], key=lambda r: r["id"])
            for name, keys in buckets.items()}


def score_predictions(test_rows, predictions, spec):
    """Score prediction records against the frozen test split. Fail-closed on coverage."""
    label_set = set(spec["labels"])
    gold = {row["id"]: row for row in test_rows}
    preds = {}
    errors = []
    for row in predictions:
        if set(row) != {"id", "label"}:
            errors.append(f"prediction fields must be exactly id,label: {sorted(row)}")
            continue
        pid, plabel = row["id"], row["label"]
        if pid in preds:
            errors.append(f"duplicate prediction id: {pid}")
            continue
        if plabel not in label_set:
            errors.append(f"prediction {pid}: label {plabel!r} not in spec labels")
            continue
        preds[pid] = plabel
    missing = sorted(set(gold) - set(preds))
    extra = sorted(set(preds) - set(gold))
    if missing:
        errors.append(f"predictions missing {len(missing)} test ids (first 5: {missing[:5]})")
    if extra:
        errors.append(f"predictions cover {len(extra)} unknown ids (first 5: {extra[:5]})")
    if errors:
        raise ValueError("; ".join(sorted(errors)))

    labels_present = sorted({row["gold_label"] for row in test_rows})
    per_class = {}
    macro_sum = 0.0
    for label in labels_present:
        tp = sum(1 for row in test_rows if row["gold_label"] == label and preds[row["id"]] == label)
        fp = sum(1 for row in test_rows if row["gold_label"] != label and preds[row["id"]] == label)
        fn = sum(1 for row in test_rows if row["gold_label"] == label and preds[row["id"]] != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": tp + fn}
        macro_sum += f1
    macro_f1 = macro_sum / len(labels_present) if labels_present else None
    correct = sum(1 for row in test_rows if preds[row["id"]] == row["gold_label"])
    critical_rows = [row for row in test_rows if row["critical"]]
    critical_misses = sum(1 for row in critical_rows if preds[row["id"]] != row["gold_label"])
    critical_fn_rate = (critical_misses / len(critical_rows)) if critical_rows else None
    return {
        "test_count": len(test_rows),
        "accuracy": correct / len(test_rows) if test_rows else None,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "critical_total": len(critical_rows),
        "critical_misses": critical_misses,
        "critical_false_negative_rate": critical_fn_rate,
        "labels_scored": labels_present,
        "predictions_sha256": digest(sorted(({"id": pid, "label": lab} for pid, lab in preds.items()),
                                           key=lambda item: item["id"])),
    }


def load_predictor_dataset(spec, base_dir):
    """Resolve spec.dataset_path relative to the spec file directory; reject escapes."""
    rel = spec.get("dataset_path")
    if not rel:
        raise ValueError("spec.dataset_path is required for task runs")
    resolved = (base_dir / rel).resolve()
    if not resolved.is_relative_to(base_dir.resolve()):
        raise ValueError("dataset_path escapes the spec directory")
    return load_jsonl(str(resolved)), resolved
