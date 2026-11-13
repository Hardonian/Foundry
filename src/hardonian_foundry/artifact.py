"""Content-addressed SHA-256 artifact store with fail-closed verification."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .refinery_validation import DEFAULT_RUNS_DIR, validate_run_id
from .types import ArtifactEntry, ArtifactVerifyResult

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_NAME = 128
MAX_KIND = 64
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def sha256_file(path: str | Path) -> tuple[str, int]:
    """Compute SHA-256 hex digest and byte size of a regular file."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"artifact file not found: {path}")
    if target.is_symlink():
        target = target.resolve()
    if not target.is_file():
        raise ValueError(f"artifact path is not a regular file: {path}")
    hasher = hashlib.sha256()
    size = 0
    with open(target, "rb") as handle:
        while chunk := handle.read(1 << 16):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def _resolve_store_dir(
    *,
    store_dir: str | Path | None = None,
    output_root: str | Path | None = None,
) -> Path:
    if store_dir is not None:
        return Path(store_dir).resolve()
    root_dir = Path(output_root or DEFAULT_RUNS_DIR).resolve()
    return (root_dir / "artifacts").resolve()


def _atomic_write_bytes(dest: Path, data: bytes, *, mode: int = 0o600) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".foundry-art-", dir=str(dest.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "posix":
            os.chmod(tmp_name, mode)
        os.replace(tmp_name, dest)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _atomic_copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".foundry-blob-", dir=str(dest.parent))
    try:
        with open(src, "rb") as reader, os.fdopen(fd, "wb") as writer:
            while chunk := reader.read(1 << 16):
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if os.name == "posix":
            os.chmod(tmp_name, 0o444)
        os.replace(tmp_name, dest)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _validate_label(value: str, label: str, max_len: int) -> str:
    if not isinstance(value, str) or not value or len(value) > max_len or value.strip() != value:
        raise ValueError(f"{label} must be a clean nonempty string <= {max_len} chars")
    if "/" in value or "\\" in value or ".." in value or "\x00" in value:
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def get_artifact_path(
    sha256_hex: str,
    *,
    store_dir: str | Path | None = None,
    output_root: str | Path | None = None,
) -> Path:
    """Return the verified blob path for sha256_hex, failing closed on missing or corrupt blobs."""
    if not isinstance(sha256_hex, str) or not HEX64_RE.fullmatch(sha256_hex):
        raise ValueError("sha256 must be a 64-char lowercase hex string")
    root = _resolve_store_dir(store_dir=store_dir, output_root=output_root)
    blobs_root = (root / "blobs" / "sha256").resolve()
    blob_path = (blobs_root / sha256_hex).resolve()
    if not blob_path.is_relative_to(blobs_root):
        raise ValueError("artifact path escapes store directory")
    if not blob_path.is_file():
        raise FileNotFoundError(f"artifact blob not found: {sha256_hex}")
    actual, _ = sha256_file(blob_path)
    if actual != sha256_hex:
        raise ValueError(f"artifact digest mismatch for {sha256_hex}: got {actual}")
    return blob_path


def add_artifact(
    source_path: str | Path,
    *,
    store_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    name: str | None = None,
    kind: str = "model_artifact",
    run_id: str | None = None,
) -> ArtifactEntry:
    """Ingest a file into the content-addressed store and write its SHA-256 metadata."""
    src = Path(source_path).resolve()
    digest_hex, size_bytes = sha256_file(src)
    clean_name = _validate_label(name or src.name, "name", MAX_NAME)
    if not _SAFE_NAME_RE.fullmatch(clean_name):
        raise ValueError(f"invalid artifact name: {clean_name!r}")
    clean_kind = _validate_label(kind, "kind", MAX_KIND)
    clean_run_id = validate_run_id(run_id) if run_id is not None else None

    root = _resolve_store_dir(store_dir=store_dir, output_root=output_root)
    blobs_dir = root / "blobs" / "sha256"
    meta_dir = root / "metadata"
    blobs_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    blob_path = blobs_dir / digest_hex
    if blob_path.exists():
        existing_sha, _ = sha256_file(blob_path)
        if existing_sha != digest_hex:
            raise ValueError(f"existing corrupt blob in store for {digest_hex}: found {existing_sha}")
    else:
        _atomic_copy_file(src, blob_path)

    created_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    blob_rel = f"blobs/sha256/{digest_hex}"
    entry: ArtifactEntry = {
        "schema_version": "1.0",
        "artifact_id": f"sha256:{digest_hex}",
        "sha256": digest_hex,
        "size_bytes": size_bytes,
        "name": clean_name,
        "kind": clean_kind,
        "run_id": clean_run_id,
        "created_at": created_at,
        "blob_relpath": blob_rel,
        "blob_path": str(blob_path),
    }
    meta_path = meta_dir / f"{digest_hex}.json"
    _atomic_write_bytes(meta_path, (json.dumps(entry, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return entry


def verify_artifact(
    target: str | Path,
    *,
    expected_sha256: str | None = None,
    store_dir: str | Path | None = None,
    output_root: str | Path | None = None,
) -> ArtifactVerifyResult:
    """Verify an artifact by SHA-256 digest (in store) or file path against expected/stored hash."""
    if expected_sha256 is not None and (not isinstance(expected_sha256, str) or not HEX64_RE.fullmatch(expected_sha256)):
        raise ValueError("expected_sha256 must be a 64-char lowercase hex string")

    root = _resolve_store_dir(store_dir=store_dir, output_root=output_root)
    target_str = str(target)

    if HEX64_RE.fullmatch(target_str):
        expected = expected_sha256 or target_str
        blobs_root = (root / "blobs" / "sha256").resolve()
        blob_path = (blobs_root / target_str).resolve()
        if not blob_path.is_relative_to(blobs_root) or not blob_path.is_file():
            return {
                "status": "MISSING",
                "verified": False,
                "sha256": None,
                "expected_sha256": expected,
                "size_bytes": None,
                "blob_path": str(blob_path),
                "reason": "artifact blob not found in store",
            }
        actual, size_bytes = sha256_file(blob_path)
        ok = actual == expected
        return {
            "status": "VERIFIED" if ok else "MISMATCH",
            "verified": ok,
            "sha256": actual,
            "expected_sha256": expected,
            "size_bytes": size_bytes,
            "blob_path": str(blob_path),
            "reason": None if ok else f"digest mismatch: expected {expected}, got {actual}",
        }

    file_path = Path(target_str)
    if not file_path.is_file():
        return {
            "status": "MISSING",
            "verified": False,
            "sha256": None,
            "expected_sha256": expected_sha256,
            "size_bytes": None,
            "blob_path": None,
            "reason": f"artifact file not found: {target_str}",
        }
    actual, size_bytes = sha256_file(file_path)
    if expected_sha256 is not None:
        ok = actual == expected_sha256
        return {
            "status": "VERIFIED" if ok else "MISMATCH",
            "verified": ok,
            "sha256": actual,
            "expected_sha256": expected_sha256,
            "size_bytes": size_bytes,
            "blob_path": str(file_path.resolve()),
            "reason": None if ok else f"digest mismatch: expected {expected_sha256}, got {actual}",
        }
    # No expected_sha256 given: verify the matching blob exists in the content-addressed store.
    blob_path = (root / "blobs" / "sha256" / actual).resolve()
    if not blob_path.is_file():
        return {
            "status": "MISSING",
            "verified": False,
            "sha256": actual,
            "expected_sha256": actual,
            "size_bytes": size_bytes,
            "blob_path": str(blob_path),
            "reason": "matching digest not registered in artifact store",
        }
    blob_sha, _ = sha256_file(blob_path)
    ok = blob_sha == actual
    return {
        "status": "VERIFIED" if ok else "MISMATCH",
        "verified": ok,
        "sha256": actual,
        "expected_sha256": blob_sha,
        "size_bytes": size_bytes,
        "blob_path": str(blob_path),
        "reason": None if ok else f"store blob corrupt: expected {actual}, got {blob_sha}",
    }


def list_artifacts(
    *,
    store_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """List registered artifact metadata entries sorted by sha256."""
    clean_run_id = validate_run_id(run_id) if run_id is not None else None
    root = _resolve_store_dir(store_dir=store_dir, output_root=output_root)
    meta_dir = root / "metadata"
    if not meta_dir.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for path in sorted(meta_dir.glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(item, dict):
            continue
        if clean_run_id is not None and item.get("run_id") != clean_run_id:
            continue
        entries.append(item)
    return entries
