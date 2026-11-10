"""Tamper-evident, hash-chained, append-only evidence ledger (synthetic records only).

Each entry binds: sequence number, previous entry hash, canonical record hash, and an
optional idempotency key. Verification detects modification, reordering, and duplicate
keys. Deleting the tail is NOT detectable by the chain alone: callers must anchor an
external checkpoint with head() and re-check it later via verify(..., expected_head=...).

This is metadata-level tamper evidence, not encrypted or tamper-proof storage. It is not
an immutable production evidence service; no authentication, retention guarantees, or
signed anchors exist yet. Outcomes always require human review.
"""
import hashlib
import json
import os

GENESIS = "0" * 64
MAX_KEY = 128


def _canonical(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _entry_hash(seq, prev_hash, record):
    payload = f"{seq}\n{prev_hash}\n{_canonical(record)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_lines(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line for line in handle.read().splitlines() if line.strip()]


def append(path, record, *, key=None):
    """Append one review-only record; refuse to append onto a corrupted chain."""
    if not isinstance(record, dict) or not record:
        raise ValueError("record must be a nonempty dict")
    if key is not None and (not isinstance(key, str) or not key.strip() or len(key) > MAX_KEY):
        raise ValueError("key must be a nonempty string of at most 128 characters")
    if "data_origin" not in record:
        raise ValueError("record must declare data_origin")
    _canonical(record)  # rejects non-JSON values and NaN before any write
    entries = []
    try:
        lines = _load_lines(path)
    except FileNotFoundError:
        lines = []
    for line in lines:
        entry = json.loads(line)  # raises on corruption: fail closed, do not append
        entries.append(entry)
    report = verify_entries(entries)
    if report["status"] != "OK":
        raise ValueError(f"refusing to append onto broken chain: {report['status']}")
    if key is not None and any(entry.get("key") == key for entry in entries):
        raise ValueError("duplicate idempotency key")
    prev = entries[-1]["entry_hash"] if entries else GENESIS
    seq = len(entries) + 1
    entry = {
        "seq": seq,
        "key": key,
        "prev_hash": prev,
        "record": record,
        "entry_hash": _entry_hash(seq, prev, record),
    }
    new_lines = lines + [json.dumps(entry, sort_keys=True, separators=(",", ":"))]
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write("\n".join(new_lines) + "\n")
    os.replace(tmp, path)
    return entry


def verify(path, *, expected_head=None, expected_count=None):
    """Verify the on-disk chain; optional anchor checks catch truncation and rollback."""
    entries = [json.loads(line) for line in _load_lines(path)]
    report = verify_entries(entries)
    if report["status"] == "OK" and expected_head is not None and report["head"] != expected_head:
        report = {"status": "TAMPERED", "reason": "head does not match anchored checkpoint",
                  "count": report["count"], "head": report["head"]}
    if report["status"] == "OK" and expected_count is not None and report["count"] != expected_count:
        report = {"status": "TAMPERED", "reason": "count does not match anchored checkpoint",
                  "count": report["count"], "head": report["head"]}
    return report


def verify_entries(entries):
    prev = GENESIS
    seen_keys = set()
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            return {"status": "TAMPERED", "reason": "non-object entry", "count": index - 1, "head": None}
        if entry.get("seq") != index or entry.get("prev_hash") != prev:
            return {"status": "TAMPERED", "reason": "sequence or chain link broken",
                    "count": index - 1, "head": None}
        key = entry.get("key")
        if key is not None:
            if key in seen_keys:
                return {"status": "TAMPERED", "reason": "duplicate idempotency key",
                        "count": index - 1, "head": None}
            seen_keys.add(key)
        try:
            expected = _entry_hash(index, prev, entry["record"])
        except (KeyError, TypeError, ValueError):
            return {"status": "TAMPERED", "reason": "record unreadable", "count": index - 1, "head": None}
        if entry.get("entry_hash") != expected:
            return {"status": "TAMPERED", "reason": "record hash mismatch",
                    "count": index - 1, "head": None}
        prev = expected
    return {"status": "OK", "count": len(entries), "head": prev}


def head(path):
    """Return the current chain head hash for external checkpointing."""
    return verify(path)["head"]
