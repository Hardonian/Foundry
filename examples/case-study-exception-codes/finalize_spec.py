#!/usr/bin/env python3
"""Inject or verify the trained adapter's provenance in the frozen task spec.

Run after train_lora.py and BEFORE any Foundry run: the spec's model block
(`adapter_sha256`) must describe the exact adapter whose predictions are scored.
Idempotent when re-run against the same adapter manifest; refuses to overwrite
a spec carrying a different non-placeholder adapter hash.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from hardonian_foundry.training_utils import sha_file
except ImportError:
    sys.modules.pop("hardonian_foundry", None)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from hardonian_foundry.training_utils import sha_file

_HEX64 = frozenset("0123456789abcdef")


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in _HEX64 for ch in value)


def finalize_spec(
    task_dir: Path,
    adapter_dir: Path,
    *,
    verify_only: bool = False,
) -> dict[str, Any]:
    """Finalize or verify `task_dir/spec.json` against `adapter_dir/train_manifest.json`."""
    spec_path = Path(task_dir) / "spec.json"
    manifest_path = Path(adapter_dir) / "train_manifest.json"
    if not spec_path.is_file():
        raise FileNotFoundError(f"spec.json not found: {spec_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"train_manifest.json not found: {manifest_path}")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not isinstance(spec.get("model"), dict):
        raise TypeError("spec.json must contain a 'model' object")
    if not isinstance(manifest, dict):
        raise TypeError("train_manifest.json must be a JSON object")

    adapter_sha = manifest.get("adapter_sha256")
    revision = manifest.get("revision")
    base_model_id = manifest.get("base_model_id")
    if not _is_hex64(adapter_sha) or adapter_sha == "0" * 64:
        raise ValueError("train_manifest.json missing valid non-zero adapter_sha256")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("train_manifest.json missing revision")
    if not isinstance(base_model_id, str) or not base_model_id.strip():
        raise ValueError("train_manifest.json missing base_model_id")

    weights_path = Path(adapter_dir) / "adapter_model.safetensors"
    if weights_path.is_file() and sha_file(weights_path) != adapter_sha:
        raise ValueError("adapter_model.safetensors SHA-256 does not match train_manifest.json")

    current = spec["model"].get("adapter_sha256")
    matches_manifest = (
        current == adapter_sha
        and spec["model"].get("revision") == revision
        and spec["model"].get("base_model_id") == base_model_id
    )

    if verify_only:
        if current == "0" * 64:
            raise ValueError("spec.json is not finalized (adapter_sha256 is still placeholder)")
        if not matches_manifest:
            raise ValueError(
                f"spec.json model provenance mismatch (spec={str(current)[:12]}..., manifest={adapter_sha[:12]}...)"
            )
        return {
            "status": "verified",
            "spec": str(spec_path),
            "adapter_sha256": current,
            "revision": revision,
            "base_model_id": base_model_id,
        }

    if current == "0" * 64:
        spec["model"]["adapter_sha256"] = adapter_sha
        spec["model"]["revision"] = revision
        spec["model"]["base_model_id"] = base_model_id
        spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "status": "finalized",
            "finalized": str(spec_path),
            "adapter_sha256": adapter_sha,
            "revision": revision,
            "base_model_id": base_model_id,
        }

    if matches_manifest:
        return {
            "status": "already_finalized",
            "finalized": str(spec_path),
            "adapter_sha256": current,
            "revision": revision,
            "base_model_id": base_model_id,
        }

    raise ValueError(
        f"spec already finalized with different provenance (adapter_sha256={str(current)[:12]}...); refusing to rewrite"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--verify", action="store_true", help="verify spec matches adapter manifest without writing")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON output")
    args = parser.parse_args(argv)

    try:
        result = finalize_spec(args.task_dir, args.adapter_dir, verify_only=args.verify)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        if args.json:
            print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        else:
            print(f"finalize_spec error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2 if not args.json else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
