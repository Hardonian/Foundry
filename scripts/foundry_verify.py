"""Air-gapped standalone Foundry verifier for Proofpack v2, bundles, and registry.json.

Uses Python standard library plus optional PyCA `cryptography` (or `hardonian_foundry`
when installed) so air-gapped compliance enclaves can audit artifacts offline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256_canonical(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def verify_proofpack_standalone(
    path: str | Path,
    *,
    keyring_path: str | Path | None = None,
    expected_head: str | None = None,
    require_promoted: bool = False,
) -> dict[str, Any]:
    """Verify Proofpack v2 offline using hardonian_foundry when available or pure stdlib fallback."""
    try:
        from hardonian_foundry.proofpack import verify_proofpack

        return verify_proofpack(
            path,
            require_promoted=require_promoted,
            keyring=keyring_path,
            expected_head=expected_head,
        )
    except ImportError:
        pass

    pack = json.loads(Path(path).read_text(encoding="utf-8"))
    errors: list[str] = []
    if not isinstance(pack, dict):
        return {"status": "INVALID", "verified": False, "errors": ["proofpack must be a JSON object"]}
    if pack.get("schema_version") != "2.0" or pack.get("proofpack_type") != "hardonian.foundry.proofpack.v2":
        errors.append("invalid proofpack schema_version or proofpack_type")
    if pack.get("review_required") is not True or pack.get("financial_write") is not False or pack.get("executed") is not False:
        errors.append("safety invariant violation")
    core = {k: v for k, v in pack.items() if k not in ("proofpack_sha256", "saved_path")}
    if pack.get("proofpack_sha256") != _sha256_canonical(core):
        errors.append("proofpack_sha256 mismatch")
    if expected_head is not None and pack.get("ledger_head") != expected_head:
        errors.append("expected_head mismatch")
    if require_promoted and (pack.get("promoted") is not True or pack.get("status") != "ALLOWED_REVIEW_REQUIRED"):
        errors.append("proofpack not promoted")
    if keyring_path is not None:
        errors.append("install cryptography / hardonian-foundry[signing] to verify Ed25519 keyring signatures")
    ok = not errors
    return {
        "status": "VERIFIED" if ok else "INVALID",
        "verified": ok,
        "run_id": pack.get("run_id"),
        "proofpack_sha256": pack.get("proofpack_sha256"),
        "errors": sorted(errors),
        "review_required": True,
    }


def verify_registry_standalone(path: str | Path) -> dict[str, Any]:
    """Verify whole-file SHA-256 integrity of a Foundry registry.json offline."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        return {"status": "TAMPERED", "verified": False, "errors": ["invalid registry structure"]}
    core = {"schema_version": data.get("schema_version"), "models": data.get("models")}
    expected = _sha256_canonical(core)
    actual = data.get("registry_sha256")
    ok = isinstance(actual, str) and actual == expected
    return {
        "status": "VERIFIED" if ok else "TAMPERED",
        "verified": ok,
        "model_count": len(data["models"]),
        "registry_sha256": actual,
        "errors": [] if ok else ["registry_sha256 mismatch"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="foundry-verify", description="Air-gapped offline verifier for Foundry artifacts")
    sub = parser.add_subparsers(dest="kind", required=True)

    pp = sub.add_parser("proofpack", help="Verify a Proofpack v2 JSON file offline")
    pp.add_argument("path")
    pp.add_argument("--keyring", default=None)
    pp.add_argument("--expected-head", default=None)
    pp.add_argument("--require-promoted", action="store_true")

    reg = sub.add_parser("registry", help="Verify a registry.json digest offline")
    reg.add_argument("path")

    bnd = sub.add_parser("bundle", help="Verify an exported verification bundle (dir/zip/tar.gz)")
    bnd.add_argument("path")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.kind == "proofpack":
        res = verify_proofpack_standalone(
            args.path,
            keyring_path=args.keyring,
            expected_head=args.expected_head,
            require_promoted=args.require_promoted,
        )
    elif args.kind == "registry":
        res = verify_registry_standalone(args.path)
    else:
        from hardonian_foundry.refinery import verify_bundle

        res = verify_bundle(args.path)

    print(json.dumps(res, indent=2, sort_keys=True))
    return 0 if res.get("verified") else 2


if __name__ == "__main__":
    sys.exit(main())
