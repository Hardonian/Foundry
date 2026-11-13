#!/usr/bin/env python3
"""Print the public key hex for an existing Ed25519 private key (no secrets leave the machine)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from hardonian_foundry import signing
except ImportError:
    sys.modules.pop("hardonian_foundry", None)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from hardonian_foundry import signing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key_path", type=Path, help="path to Ed25519 private key PEM")
    parser.add_argument("--json", action="store_true", help="emit JSON metadata with SHA-256 fingerprint")
    args = parser.parse_args(argv)

    try:
        if not args.key_path.exists():
            raise FileNotFoundError(f"private key not found: {args.key_path}")
        pub_hex = signing.public_key_hex(str(args.key_path))
    except (OSError, ValueError, TypeError, signing.SigningUnavailable) as exc:
        if args.json:
            print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        else:
            print(f"show_pubkey error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "public_key_hex": pub_hex,
                    "fingerprint_sha256": signing.fingerprint(pub_hex),
                    "key_path": str(args.key_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(pub_hex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
