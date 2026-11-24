"""External evaluation importers converting Promptfoo, Inspect AI, lm-eval, and MLflow logs into Foundry task specs."""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .canonical import digest
from .task_eval import split_task_dataset

IMPORT_FORMATS: tuple[str, ...] = (
    "promptfoo",
    "inspect-ai",
    "lm-eval",
    "mlflow",
)

_SAFE_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _clean_path(path_value: str | Path, label: str) -> Path:
    raw = str(path_value)
    if not raw or any(ord(ch) < 32 for ch in raw):
        raise ValueError(f"{label} must be a clean non-empty path")
    if any(part == ".." for part in re.split(r"[/\\]", raw)):
        raise ValueError(f"{label} must not contain path traversal")
    return Path(path_value).resolve()


def _normalize_labels(labels: Sequence[str] | None, inferred: list[str]) -> list[str]:
    if labels is not None:
        cleaned = [str(x).strip() for x in labels if isinstance(x, str) and str(x).strip()]
        deduped = list(dict.fromkeys(cleaned))
        if len(deduped) < 2:
            raise ValueError("labels must contain at least 2 unique label strings")
        return deduped
    deduped = list(dict.fromkeys(inferred))
    if len(deduped) == 1:
        deduped.append("UNDETERMINED" if deduped[0] != "UNDETERMINED" else "FAIL")
    return deduped or ["PASS", "FAIL"]


def _emit_foundry_task(
    *,
    items: list[dict[str, Any]],
    out_dir: str | Path,
    task_id: str,
    description: str,
    labels: list[str],
    source_format: str,
    input_sha256: str,
) -> dict[str, Any]:
    if not items:
        raise ValueError(f"no evaluation records found in {source_format} input")
    if not _SAFE_TASK_ID_RE.fullmatch(task_id):
        raise ValueError("task_id must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

    dest = _clean_path(out_dir, "out_dir")
    dest.mkdir(parents=True, exist_ok=True)

    data_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []

    seen_prompts: set[str] = set()
    for idx, item in enumerate(items):
        rid = str(item.get("id") or f"imp-{source_format}-{idx:04d}")[:128]
        raw_prompt = str(item.get("prompt") or f"Imported record {rid}")[:3800]
        prompt = raw_prompt if raw_prompt not in seen_prompts else f"[{rid}] {raw_prompt}"
        if prompt in seen_prompts:
            prompt = f"[{idx}:{rid}] {raw_prompt}"
        prompt = prompt[:4000]
        seen_prompts.add(prompt)

        gold = str(item["gold_label"])
        model_lbl = str(item["model_label"])
        base_lbl = str(item["baseline_label"])
        fam = str(item.get("family") or f"imported-family-{idx % 4:02d}")[:64]
        critical = bool(item.get("critical", gold == "FAIL"))

        data_rows.append({
            "id": rid,
            "prompt": prompt,
            "gold_label": gold,
            "critical": critical,
            "family": fam,
        })
        base_entry: dict[str, Any] = {"id": rid, "label": base_lbl}
        mod_entry: dict[str, Any] = {"id": rid, "label": model_lbl}
        if item.get("latency_ms") is not None:
            mod_entry["latency_ms"] = float(item["latency_ms"])
        base_rows.append(base_entry)
        model_rows.append(mod_entry)

    spec_obj: dict[str, Any] = {
        "schema_version": "1.0",
        "task_id": task_id,
        "description": description,
        "labels": labels,
        "thresholds": {
            "macro_f1_min": 0.8,
            "critical_fn_rate_max": 0.0,
        },
        "baseline_must_improve": True,
        "split_seed": 42,
        "model": {
            "base_model_id": f"imported/{source_format}",
            "revision": "v1.0.0",
            "adapter_sha256": input_sha256,
            "adapter_name": f"{task_id}-imported",
        },
        "dataset_path": "data.jsonl",
    }

    if len({r["family"] for r in data_rows}) >= 3:
        test_ids = {r["id"] for r in split_task_dataset(data_rows, spec_obj)["test"]}
        base_out = [r for r in base_rows if r["id"] in test_ids]
        mod_out = [r for r in model_rows if r["id"] in test_ids]
    else:
        base_out = base_rows
        mod_out = model_rows

    spec_path = dest / "spec.json"
    data_path = dest / "data.jsonl"
    base_path = dest / "baseline.jsonl"
    mod_path = dest / "model.jsonl"

    spec_path.write_text(json.dumps(spec_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    data_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in data_rows), encoding="utf-8")
    base_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in base_out), encoding="utf-8")
    mod_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in mod_out), encoding="utf-8")

    return {
        "status": "IMPORTED",
        "source_format": source_format,
        "data_origin": "synthetic",
        "task_id": task_id,
        "record_count": len(data_rows),
        "validation_ready": len(data_rows) >= 9,
        "labels": labels,
        "input_sha256": input_sha256,
        "dataset_sha256": digest(data_rows),
        "out_dir": str(dest),
        "spec_path": str(spec_path),
        "data_path": str(data_path),
        "baseline_path": str(base_path),
        "model_path": str(mod_path),
    }


def _read_json_input(input_path: str | Path) -> tuple[Any, str]:
    src = _clean_path(input_path, "input_path")
    if not src.is_file():
        raise FileNotFoundError(f"import input file not found: {input_path}")
    raw = src.read_text(encoding="utf-8")
    return json.loads(raw), digest(raw)


def import_promptfoo(
    input_path: str | Path,
    *,
    out_dir: str | Path,
    task_id: str = "imported-promptfoo",
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Import Promptfoo JSON evaluation output into a Foundry task directory."""
    raw, in_sha = _read_json_input(input_path)
    if not isinstance(raw, dict):
        raise ValueError("promptfoo input must be a JSON object")
    results_block = raw.get("results")
    if isinstance(results_block, dict) and isinstance(results_block.get("results"), list):
        entries = results_block["results"]
    elif isinstance(results_block, list):
        entries = results_block
    elif isinstance(raw.get("outputs"), list):
        entries = raw["outputs"]
    else:
        raise ValueError("promptfoo input missing 'results' list")

    label_list = _normalize_labels(labels, ["PASS", "FAIL"])
    items: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        prompt_obj = entry.get("prompt")
        prompt_text = (
            prompt_obj.get("raw")
            if isinstance(prompt_obj, dict) and isinstance(prompt_obj.get("raw"), str)
            else json.dumps(entry.get("vars", entry), sort_keys=True)
        )
        expected = entry.get("expected_label") or entry.get("gold_label") or (label_list[0] if entry.get("expected_pass", True) else label_list[-1])
        passed = bool(entry.get("success", False))
        model_lbl = str(entry.get("model_label") or (expected if passed else ("FAIL" if expected != "FAIL" else "PASS")))
        base_lbl = str(entry.get("baseline_label") or label_list[-1])
        items.append({
            "id": entry.get("id") or f"pf-{idx:04d}",
            "prompt": prompt_text,
            "gold_label": expected,
            "model_label": model_lbl,
            "baseline_label": base_lbl,
            "critical": bool(entry.get("critical", expected == "FAIL")),
            "latency_ms": entry.get("latencyMs"),
            "family": entry.get("family"),
        })

    return _emit_foundry_task(
        items=items,
        out_dir=out_dir,
        task_id=task_id,
        description="Imported from Promptfoo evaluation log",
        labels=label_list,
        source_format="promptfoo",
        input_sha256=in_sha,
    )


def import_inspect_ai(
    input_path: str | Path,
    *,
    out_dir: str | Path,
    task_id: str = "imported-inspect-ai",
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Import Inspect AI evaluation log JSON into a Foundry task directory."""
    raw, in_sha = _read_json_input(input_path)
    if not isinstance(raw, dict) or not isinstance(raw.get("samples"), list):
        raise ValueError("inspect-ai input must be a JSON object with a 'samples' list")

    label_list = _normalize_labels(labels, ["PASS", "FAIL"])
    items: list[dict[str, Any]] = []
    for idx, sample in enumerate(raw["samples"]):
        if not isinstance(sample, dict):
            continue
        inp = sample.get("input", "")
        prompt_text = inp if isinstance(inp, str) else json.dumps(inp, sort_keys=True)
        scores = sample.get("scores")
        score_ok = False
        if isinstance(scores, dict) and scores:
            first_score = next(iter(scores.values()))
            if isinstance(first_score, dict):
                val = first_score.get("value")
                score_ok = val in ("C", "PASS", True, 1, 1.0)
        gold = str(sample.get("gold_label") or (label_list[0] if idx % 2 == 0 else label_list[-1]))
        model_lbl = str(sample.get("model_label") or (gold if score_ok else (label_list[-1] if gold != label_list[-1] else label_list[0])))
        base_lbl = str(sample.get("baseline_label") or label_list[-1])
        total_sec = sample.get("total_time")
        latency_ms = float(total_sec) * 1000.0 if isinstance(total_sec, (int, float)) and not isinstance(total_sec, bool) else None
        items.append({
            "id": str(sample.get("id") or f"inspect-{idx:04d}"),
            "prompt": prompt_text,
            "gold_label": gold,
            "model_label": model_lbl,
            "baseline_label": base_lbl,
            "critical": bool(sample.get("critical", gold == "FAIL")),
            "latency_ms": latency_ms,
            "family": (sample.get("metadata") or {}).get("family") if isinstance(sample.get("metadata"), dict) else None,
        })

    return _emit_foundry_task(
        items=items,
        out_dir=out_dir,
        task_id=task_id,
        description="Imported from Inspect AI sample log",
        labels=label_list,
        source_format="inspect-ai",
        input_sha256=in_sha,
    )


def import_lm_eval(
    input_path: str | Path,
    *,
    out_dir: str | Path,
    task_id: str = "imported-lm-eval",
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Import EleutherAI lm-evaluation-harness sample log JSON into a Foundry task directory."""
    raw, in_sha = _read_json_input(input_path)
    if not isinstance(raw, dict) or "samples" not in raw:
        raise ValueError("lm-eval input must be a JSON object containing 'samples'")

    raw_samples = raw["samples"]
    flat_samples: list[dict[str, Any]] = []
    if isinstance(raw_samples, list):
        flat_samples = [s for s in raw_samples if isinstance(s, dict)]
    elif isinstance(raw_samples, dict):
        for task_name in sorted(raw_samples):
            task_rows = raw_samples[task_name]
            if isinstance(task_rows, list):
                for s in task_rows:
                    if isinstance(s, dict):
                        flat_samples.append({"_task": task_name, **s})
    else:
        raise ValueError("lm-eval 'samples' must be a list or dict")

    label_list = _normalize_labels(labels, ["PASS", "FAIL"])
    items: list[dict[str, Any]] = []
    for idx, sample in enumerate(flat_samples):
        doc = sample.get("doc", sample.get("arguments", ""))
        prompt_text = doc if isinstance(doc, str) else json.dumps(doc, sort_keys=True)
        metric_val = sample.get("exact_match", sample.get("acc", 0))
        ok = bool(metric_val == 1 or metric_val == 1.0 or metric_val is True)
        gold = str(sample.get("gold_label") or (label_list[0] if idx % 2 == 0 else label_list[-1]))
        model_lbl = str(sample.get("model_label") or (gold if ok else label_list[-1]))
        base_lbl = str(sample.get("baseline_label") or label_list[-1])
        items.append({
            "id": f"lmeval-{sample.get('_task', 'task')}-{sample.get('doc_id', idx)}",
            "prompt": prompt_text,
            "gold_label": gold,
            "model_label": model_lbl,
            "baseline_label": base_lbl,
            "critical": bool(sample.get("critical", gold == "FAIL")),
            "family": sample.get("family"),
        })

    return _emit_foundry_task(
        items=items,
        out_dir=out_dir,
        task_id=task_id,
        description="Imported from lm-evaluation-harness log",
        labels=label_list,
        source_format="lm-eval",
        input_sha256=in_sha,
    )


def import_mlflow(
    input_path: str | Path,
    *,
    out_dir: str | Path,
    task_id: str = "imported-mlflow",
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Import MLflow evaluate JSON artifact table (`rows` or `data`) into a Foundry task directory."""
    raw, in_sha = _read_json_input(input_path)
    if not isinstance(raw, dict):
        raise ValueError("mlflow input must be a JSON object")
    rows = raw.get("rows") if isinstance(raw.get("rows"), list) else raw.get("data")
    if not isinstance(rows, list):
        raise ValueError("mlflow input must contain a 'rows' or 'data' list")

    inferred: list[str] = []
    parsed_items: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        gold = str(row.get("ground_truth", row.get("target", row.get("gold_label", "PASS"))))
        pred = str(row.get("prediction", row.get("outputs", row.get("model_label", gold))))
        base = str(row.get("baseline_prediction", row.get("baseline_label", pred)))
        inferred.extend([gold, pred, base])
        inp = row.get("inputs", row.get("prompt", f"mlflow-row-{idx}"))
        prompt_text = inp if isinstance(inp, str) else json.dumps(inp, sort_keys=True)
        parsed_items.append({
            "id": str(row.get("id") or f"mlflow-{idx:04d}"),
            "prompt": prompt_text,
            "gold_label": gold,
            "model_label": pred,
            "baseline_label": base,
            "critical": bool(row.get("critical", gold in ("FAIL", "DENY", "ESCALATE_REVIEW", "amount_mismatch"))),
            "latency_ms": row.get("latency_ms"),
            "family": row.get("family"),
        })

    label_list = _normalize_labels(labels, inferred)
    return _emit_foundry_task(
        items=parsed_items,
        out_dir=out_dir,
        task_id=task_id,
        description="Imported from MLflow evaluate output",
        labels=label_list,
        source_format="mlflow",
        input_sha256=in_sha,
    )


def import_eval_results(
    source_format: str,
    input_path: str | Path,
    *,
    out_dir: str | Path,
    task_id: str | None = None,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Dispatch external evaluation import cleanly by source_format."""
    fmt = source_format.strip().lower() if isinstance(source_format, str) else ""
    if fmt == "promptfoo":
        return import_promptfoo(input_path, out_dir=out_dir, task_id=task_id or "imported-promptfoo", labels=labels)
    if fmt == "inspect-ai":
        return import_inspect_ai(input_path, out_dir=out_dir, task_id=task_id or "imported-inspect-ai", labels=labels)
    if fmt == "lm-eval":
        return import_lm_eval(input_path, out_dir=out_dir, task_id=task_id or "imported-lm-eval", labels=labels)
    if fmt == "mlflow":
        return import_mlflow(input_path, out_dir=out_dir, task_id=task_id or "imported-mlflow", labels=labels)
    supported = ", ".join(IMPORT_FORMATS)
    raise ValueError(f"unsupported import format {source_format!r}; expected one of: {supported}")
