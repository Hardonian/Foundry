"""Tamper-evident, hash-chained, append-only evidence ledger.

Each entry binds: sequence number, previous entry hash, canonical record hash, and an
optional idempotency key. Verification detects modification, reordering, and duplicate
keys. Deleting the tail is NOT detectable by the chain alone: callers must anchor an
external checkpoint with head()/checkpoint() and re-check it later via
verify(..., expected_head=..., expected_count=...).

Default origin policy strictly allows only ``synthetic`` records unless callers
explicitly opt into approved governance tiers (``consented_pilot``, ``redacted_eval``).
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from contextlib import closing
from pathlib import Path
from typing import IO, Any, Protocol, runtime_checkable

from .canonical import canonical

# Arm selection by platform literal rather than try/except ImportError: identical on
# real platforms (fcntl on POSIX, msvcrt on Windows) but lets mypy statically skip the
# unreachable arm on each runner (typeshed gates these modules by sys.platform).
if sys.platform != "win32":
    import fcntl

    def _lock(file_obj: IO[Any], *, timeout: float = 30.0) -> None:
        del timeout
        fcntl.flock(file_obj, fcntl.LOCK_EX)

    def _unlock(file_obj: IO[Any]) -> None:
        fcntl.flock(file_obj, fcntl.LOCK_UN)

    _O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
else:
    import msvcrt

    def _lock(file_obj: IO[Any], *, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        delay = 0.001
        while True:
            file_obj.seek(0)
            try:
                msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out acquiring evidence lock after {timeout}s") from exc
                time.sleep(delay)
                delay = min(delay * 1.5, 0.05)

    def _unlock(file_obj: IO[Any]) -> None:
        try:
            file_obj.seek(0)
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

    _O_NOFOLLOW = 0

GENESIS = "0" * 64
MAX_KEY = 128
VALID_DATA_ORIGINS = frozenset({"synthetic", "consented_pilot", "redacted_eval"})
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ENTRY_FIELDS = frozenset({"seq", "key", "prev_hash", "record", "entry_hash"})
_CHECKPOINT_INTERVAL = 32


def _canonical(record: Any) -> str:
    return canonical(record)


def _entry_hash(seq: int, prev_hash: str, record: Any, key: str | None) -> str:
    payload = canonical({"seq": seq, "prev_hash": prev_hash, "record": record, "key": key})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_allowed_origins(allowed_origins: Iterable[str]) -> frozenset[str]:
    if isinstance(allowed_origins, (str, bytes)) or not isinstance(allowed_origins, Iterable):
        raise ValueError("allowed_origins must be a nonempty collection of supported origin tiers")  # noqa: TRY004
    normalized: set[str] = set()
    for item in allowed_origins:
        if not isinstance(item, str) or item not in VALID_DATA_ORIGINS:
            raise ValueError(f"unsupported data_origin tier in allowed_origins: {item!r}")
        normalized.add(item)
    if not normalized:
        raise ValueError("allowed_origins must not be empty")
    return frozenset(normalized)


def _validate_key(key: Any) -> None:
    if key is not None and (not isinstance(key, str) or not key.strip() or len(key) > MAX_KEY):
        raise ValueError("key must be a nonempty string of at most 128 characters")


def _validate_record(
    record: Any,
    key: Any,
    *,
    allowed_origins: Iterable[str] = ("synthetic",),
) -> frozenset[str]:
    origins = _normalize_allowed_origins(allowed_origins)
    if not isinstance(record, dict) or not record:
        raise ValueError("record must be a nonempty dict")
    _validate_key(key)
    origin = record.get("data_origin")
    if origins == frozenset({"synthetic"}):
        if origin != "synthetic":
            raise ValueError("record must declare synthetic data_origin")
    elif not isinstance(origin, str) or origin not in origins:
        raise ValueError(f"record data_origin {origin!r} is not permitted")
    canonical(record)  # single-pass validation (non-string keys, circular refs, NaN/Inf, surrogates)
    return origins


def _load_lines(path: str | os.PathLike[str]) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line for line in handle.read().splitlines() if line.strip()]


def _verify_single_entry(
    entry: Any,
    index: int,
    prev: str,
    seen_keys: set[str],
) -> tuple[str | None, str]:
    if not isinstance(entry, dict):
        return None, "non-object entry"
    if set(entry) != _ENTRY_FIELDS or type(entry.get("seq")) is not int:
        return None, "invalid entry fields"
    if entry.get("seq") != index or entry.get("prev_hash") != prev:
        return None, "sequence or chain link broken"
    key = entry.get("key")
    if key is not None:
        if not isinstance(key, str) or not key.strip() or len(key) > MAX_KEY:
            return None, "invalid idempotency key"
        if key in seen_keys:
            return None, "duplicate idempotency key"
        seen_keys.add(key)
    try:
        expected = _entry_hash(index, prev, entry["record"], key)
    except (KeyError, TypeError, ValueError):
        return None, "record unreadable"
    if entry.get("entry_hash") != expected:
        return None, "record hash mismatch"
    return expected, ""


def _write_atomic_text(path: str, content: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".foundry-evidence-", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _load_valid_checkpoint(ckpt_path: str, file_size: int) -> dict[str, Any] | None:
    if not os.path.isfile(ckpt_path):
        return None
    try:
        with open(ckpt_path, "r", encoding="utf-8") as handle:
            ckpt = json.load(handle)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(ckpt, dict) or ckpt.get("version") != 1:
        return None
    offset = ckpt.get("offset")
    count = ckpt.get("count")
    head_hash = ckpt.get("head")
    file_sha = ckpt.get("file_sha256")
    keys = ckpt.get("keys")
    if (
        type(offset) is not int
        or not (0 < offset <= file_size)
        or type(count) is not int
        or count < 1
        or not isinstance(head_hash, str)
        or _HEX64.fullmatch(head_hash) is None
        or not isinstance(file_sha, str)
        or _HEX64.fullmatch(file_sha) is None
        or not isinstance(keys, list)
        or any(not isinstance(k, str) or not k.strip() or len(k) > MAX_KEY for k in keys)
    ):
        return None
    return ckpt


def append(
    path: str | os.PathLike[str],
    record: dict[str, Any],
    *,
    key: str | None = None,
    allowed_origins: Iterable[str] = ("synthetic",),
) -> dict[str, Any]:
    """Append one review-only record; refuse to append onto a corrupted chain."""
    _validate_record(record, key, allowed_origins=allowed_origins)
    return _append_with_builder(
        path,
        lambda _entries, _head: record,
        key=key,
        allowed_origins=allowed_origins,
        _needs_entries=False,
    )


def _append_with_builder(
    path: str | os.PathLike[str],
    builder: Callable[[list[dict[str, Any]], str], dict[str, Any]],
    *,
    key: str | None = None,
    allowed_origins: Iterable[str] = ("synthetic",),
    _needs_entries: bool = True,
) -> dict[str, Any]:
    """Build and append against one verified head under the shared sidecar lock."""
    origins = _normalize_allowed_origins(allowed_origins)
    _validate_key(key)
    path_str = os.fspath(path)
    lock_fd = os.open(path_str + ".lock", os.O_CREAT | os.O_RDWR | _O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, "r+") as lock:
        _lock(lock)
        try:
            return _append_locked(path_str, builder, key, allowed_origins=origins, needs_entries=_needs_entries)
        finally:
            _unlock(lock)


def _read_existing_state(
    path_str: str,
    ckpt_path: str,
    *,
    needs_entries: bool,
) -> tuple[list[dict[str, Any]], str, int, set[str], int, int, bool, hashlib._Hash]:
    with open(path_str, "rb") as handle:
        file_size = os.fstat(handle.fileno()).st_size
        ckpt = _load_valid_checkpoint(ckpt_path, file_size)
        if ckpt is not None:
            prefix = handle.read(ckpt["offset"])
            hasher = hashlib.sha256(prefix)
            if hasher.hexdigest() == ckpt["file_sha256"]:
                tail_bytes = handle.read()
                hasher.update(tail_bytes)
                try:
                    tail_text = tail_bytes.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError("corrupt UTF-8 in ledger tail") from exc
                tail_lines = [line for line in tail_text.splitlines() if line.strip()]
                prev = ckpt["head"]
                seq = ckpt["count"] + 1
                seen_keys = set(ckpt["keys"])
                tail_entries: list[dict[str, Any]] = []
                for line in tail_lines:
                    entry = json.loads(line)
                    next_hash, reason = _verify_single_entry(entry, seq, prev, seen_keys)
                    if next_hash is None:
                        raise ValueError(f"refusing to append onto broken chain: TAMPERED ({reason})")
                    prev = next_hash
                    seq += 1
                    tail_entries.append(entry)
                entries: list[dict[str, Any]] = []
                if needs_entries:
                    prefix_text = prefix.decode("utf-8")
                    entries = [json.loads(line) for line in prefix_text.splitlines() if line.strip()]
                    entries.extend(tail_entries)
                ends_with_nl = (tail_bytes[-1:] == b"\n") if tail_bytes else (prefix[-1:] == b"\n")
                return entries, prev, seq, seen_keys, file_size, len(tail_lines), ends_with_nl, hasher

        handle.seek(0)
        raw_bytes = handle.read()

    hasher = hashlib.sha256(raw_bytes)
    ends_with_nl = (raw_bytes[-1:] == b"\n") if raw_bytes else True
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("corrupt UTF-8 in ledger") from exc
    lines = [line for line in text.splitlines() if line.strip()]
    entries = [json.loads(line) for line in lines]
    report = verify_entries(entries)
    if report["status"] != "OK":
        raise ValueError(f"refusing to append onto broken chain: {report['status']}")
    prev = report["head"]
    seq = report["count"] + 1
    seen_keys = {entry["key"] for entry in entries if entry.get("key") is not None}
    return entries, prev, seq, seen_keys, file_size, len(entries), ends_with_nl, hasher


def _append_locked(
    path_str: str,
    builder: Callable[[list[dict[str, Any]], str], dict[str, Any]],
    key: str | None,
    *,
    allowed_origins: frozenset[str],
    needs_entries: bool,
) -> dict[str, Any]:
    ckpt_path = path_str + ".ckpt"
    if not os.path.exists(path_str):
        entries: list[dict[str, Any]] = []
        prev = GENESIS
        seq = 1
        seen_keys: set[str] = set()
        record = builder(entries, prev)
        _validate_record(record, key, allowed_origins=allowed_origins)
        entry = {
            "seq": seq,
            "key": key,
            "prev_hash": prev,
            "record": record,
            "entry_hash": _entry_hash(seq, prev, record, key),
        }
        _write_atomic_text(path_str, canonical(entry) + "\n")
        return entry

    entries, prev, seq, seen_keys, file_size, tail_count, ends_with_nl, hasher = _read_existing_state(
        path_str,
        ckpt_path,
        needs_entries=needs_entries,
    )
    if key is not None and key in seen_keys:
        raise ValueError("duplicate idempotency key")

    record = builder(entries, prev)
    _validate_record(record, key, allowed_origins=allowed_origins)
    entry_hash = _entry_hash(seq, prev, record, key)
    entry = {
        "seq": seq,
        "key": key,
        "prev_hash": prev,
        "record": record,
        "entry_hash": entry_hash,
    }
    line_str = canonical(entry) + "\n"
    prefix_nl = b"\n" if (file_size > 0 and not ends_with_nl) else b""
    write_bytes = prefix_nl + line_str.encode("utf-8")

    with open(path_str, "ab") as handle:
        handle.write(write_bytes)
        handle.flush()
        os.fsync(handle.fileno())

    if tail_count + 1 >= _CHECKPOINT_INTERVAL:
        hasher.update(write_bytes)
        if key is not None:
            seen_keys.add(key)
        ckpt_payload = {
            "version": 1,
            "offset": file_size + len(write_bytes),
            "file_sha256": hasher.hexdigest(),
            "count": seq,
            "head": entry_hash,
            "keys": sorted(seen_keys),
        }
        _write_atomic_text(ckpt_path, canonical(ckpt_payload))

    return entry


def _apply_anchors(
    report: dict[str, Any],
    *,
    expected_head: str | None = None,
    expected_count: int | None = None,
) -> dict[str, Any]:
    if report["status"] == "OK" and expected_head is not None and report["head"] != expected_head:
        return {
            "status": "TAMPERED",
            "reason": "head does not match anchored checkpoint",
            "count": report["count"],
            "head": report["head"],
        }
    if report["status"] == "OK" and expected_count is not None and (
        type(expected_count) is not int or report["count"] != expected_count
    ):
        return {
            "status": "TAMPERED",
            "reason": "count does not match anchored checkpoint",
            "count": report["count"],
            "head": report["head"],
        }
    return report


def verify(
    path: str | os.PathLike[str],
    *,
    expected_head: str | None = None,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Verify the on-disk chain; optional anchor checks catch truncation and rollback."""
    entries = [json.loads(line) for line in _load_lines(path)]
    report = verify_entries(entries)
    return _apply_anchors(report, expected_head=expected_head, expected_count=expected_count)


def verify_entries(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Verify an in-memory sequence of ledger entry dicts from genesis."""
    prev = GENESIS
    seen_keys: set[str] = set()
    count = 0
    for index, entry in enumerate(entries, start=1):
        count = index
        next_hash, reason = _verify_single_entry(entry, index, prev, seen_keys)
        if next_hash is None:
            return {"status": "TAMPERED", "reason": reason, "count": index - 1, "head": None}
        prev = next_hash
    return {"status": "OK", "count": count, "head": prev}


def head(path: str | os.PathLike[str]) -> str | None:
    """Return the current chain head hash for external checkpointing."""
    return verify(path)["head"]


def checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Verify the ledger at path and return an external checkpoint {head, count}."""
    report = verify(path)
    if report["status"] != "OK":
        raise ValueError(f"cannot checkpoint broken ledger: {report['reason']}")
    return {"head": report["head"], "count": report["count"]}


@runtime_checkable
class EvidenceStore(Protocol):
    """Protocol implemented by Foundry tamper-evident ledger stores."""

    def append(
        self,
        record: dict[str, Any],
        *,
        key: str | None = None,
        allowed_origins: Iterable[str] | None = None,
    ) -> dict[str, Any]: ...

    def append_with_builder(
        self,
        builder: Callable[[list[dict[str, Any]], str], dict[str, Any]],
        *,
        key: str | None = None,
        allowed_origins: Iterable[str] | None = None,
    ) -> dict[str, Any]: ...

    def verify(
        self,
        *,
        expected_head: str | None = None,
        expected_count: int | None = None,
    ) -> dict[str, Any]: ...

    def head(self) -> str | None: ...

    def checkpoint(self) -> dict[str, Any]: ...

    def entries(self) -> list[dict[str, Any]]: ...


class JsonlEvidenceStore:
    """JSONL-backed tamper-evident evidence ledger implementing EvidenceStore."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        allowed_origins: Iterable[str] = ("synthetic",),
    ) -> None:
        self.path = os.fspath(path)
        self.allowed_origins = _normalize_allowed_origins(allowed_origins)

    def append(
        self,
        record: dict[str, Any],
        *,
        key: str | None = None,
        allowed_origins: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        origins = self.allowed_origins if allowed_origins is None else _normalize_allowed_origins(allowed_origins)
        return append(self.path, record, key=key, allowed_origins=origins)

    def append_with_builder(
        self,
        builder: Callable[[list[dict[str, Any]], str], dict[str, Any]],
        *,
        key: str | None = None,
        allowed_origins: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        origins = self.allowed_origins if allowed_origins is None else _normalize_allowed_origins(allowed_origins)
        return _append_with_builder(self.path, builder, key=key, allowed_origins=origins)

    def verify(
        self,
        *,
        expected_head: str | None = None,
        expected_count: int | None = None,
    ) -> dict[str, Any]:
        return verify(self.path, expected_head=expected_head, expected_count=expected_count)

    def head(self) -> str | None:
        return head(self.path)

    def checkpoint(self) -> dict[str, Any]:
        return checkpoint(self.path)

    def entries(self) -> list[dict[str, Any]]:
        loaded = [json.loads(line) for line in _load_lines(self.path)]
        report = verify_entries(loaded)
        if report["status"] != "OK":
            raise ValueError(f"broken ledger chain: {report['reason']}")
        return loaded


class SQLiteEvidenceStore:
    """WAL-backed SQLite append-only hash-chained store implementing EvidenceStore."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        allowed_origins: Iterable[str] = ("synthetic",),
    ) -> None:
        self.path = os.fspath(path)
        self.allowed_origins = _normalize_allowed_origins(allowed_origins)
        parent = Path(self.path).parent
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS evidence_entries (
                    seq INTEGER PRIMARY KEY CHECK (seq >= 1),
                    key TEXT UNIQUE,
                    prev_hash TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    entry_hash TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS evidence_no_update
                BEFORE UPDATE ON evidence_entries
                BEGIN
                    SELECT RAISE(ABORT, 'evidence_entries is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS evidence_no_delete
                BEFORE DELETE ON evidence_entries
                BEGIN
                    SELECT RAISE(ABORT, 'evidence_entries is append-only');
                END;
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _load_rows(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT seq, key, prev_hash, record_json, entry_hash FROM evidence_entries ORDER BY seq ASC"
        ).fetchall()
        entries: list[dict[str, Any]] = []
        for seq, key, prev_hash, record_json, entry_hash in rows:
            record = json.loads(record_json)
            entries.append({
                "seq": seq,
                "key": key,
                "prev_hash": prev_hash,
                "record": record,
                "entry_hash": entry_hash,
            })
        return entries

    def append(
        self,
        record: dict[str, Any],
        *,
        key: str | None = None,
        allowed_origins: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        origins = self.allowed_origins if allowed_origins is None else _normalize_allowed_origins(allowed_origins)
        _validate_record(record, key, allowed_origins=origins)
        return self.append_with_builder(
            lambda _entries, _head: record,
            key=key,
            allowed_origins=origins,
        )

    def append_with_builder(
        self,
        builder: Callable[[list[dict[str, Any]], str], dict[str, Any]],
        *,
        key: str | None = None,
        allowed_origins: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        origins = self.allowed_origins if allowed_origins is None else _normalize_allowed_origins(allowed_origins)
        _validate_key(key)
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                entries = self._load_rows(conn)
                report = verify_entries(entries)
                if report["status"] != "OK":
                    raise ValueError(f"refusing to append onto broken chain: {report['status']}")
                if key is not None and any(entry.get("key") == key for entry in entries):
                    raise ValueError("duplicate idempotency key")
                prev = report["head"]
                seq = report["count"] + 1
                record = builder(entries, prev)
                _validate_record(record, key, allowed_origins=origins)
                entry_hash = _entry_hash(seq, prev, record, key)
                conn.execute(
                    "INSERT INTO evidence_entries (seq, key, prev_hash, record_json, entry_hash) VALUES (?, ?, ?, ?, ?)",
                    (seq, key, prev, canonical(record), entry_hash),
                )
                conn.execute("COMMIT")
                return {
                    "seq": seq,
                    "key": key,
                    "prev_hash": prev,
                    "record": record,
                    "entry_hash": entry_hash,
                }
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise

    def verify(
        self,
        *,
        expected_head: str | None = None,
        expected_count: int | None = None,
    ) -> dict[str, Any]:
        try:
            with closing(self._connect()) as conn:
                conn.execute("BEGIN")
                try:
                    entries = self._load_rows(conn)
                finally:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
        except (sqlite3.DatabaseError, json.JSONDecodeError, ValueError, TypeError) as exc:
            return {"status": "TAMPERED", "reason": f"corrupt sqlite ledger: {exc}", "count": 0, "head": None}
        report = verify_entries(entries)
        return _apply_anchors(report, expected_head=expected_head, expected_count=expected_count)

    def head(self) -> str | None:
        return self.verify()["head"]

    def checkpoint(self) -> dict[str, Any]:
        report = self.verify()
        if report["status"] != "OK":
            raise ValueError(f"cannot checkpoint broken ledger: {report['reason']}")
        return {"head": report["head"], "count": report["count"]}

    def entries(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            conn.execute("BEGIN")
            try:
                loaded = self._load_rows(conn)
            finally:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
        report = verify_entries(loaded)
        if report["status"] != "OK":
            raise ValueError(f"broken ledger chain: {report['reason']}")
        return loaded
