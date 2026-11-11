#!/usr/bin/env python3
"""Inject the trained adapter's provenance into the frozen task spec.

Run after train_lora.py and BEFORE any Foundry run: the spec's model block
(adapter_sha256) must describe the exact adapter whose predictions are scored.
Refuses to overwrite a spec that already carries a non-placeholder hash.
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    args = parser.parse_args()

    spec_path = Path(args.task_dir) / "spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    current = spec["model"]["adapter_sha256"]
    if current != "0" * 64:
        raise SystemExit(f"spec already finalized (adapter_sha256={current[:12]}...); refusing to rewrite")

    manifest = json.loads((Path(args.adapter_dir) / "train_manifest.json").read_text(encoding="utf-8"))
    spec["model"]["adapter_sha256"] = manifest["adapter_sha256"]
    spec["model"]["revision"] = manifest["revision"]
    spec["model"]["base_model_id"] = manifest["base_model_id"]
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"finalized": str(spec_path),
                      "adapter_sha256": spec["model"]["adapter_sha256"],
                      "revision": spec["model"]["revision"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
