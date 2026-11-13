"""Offline tests for the tamper-evident evidence chain and stores."""

from __future__ import annotations

import json
import multiprocessing
import os
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from hardonian_foundry import evidence
from hardonian_foundry.evidence import (
    EvidenceStore,
    JsonlEvidenceStore,
    SQLiteEvidenceStore,
    checkpoint,
)


def record(n: int, *, origin: str = "synthetic") -> dict[str, object]:
    return {"data_origin": origin, "value_minor": n, "note": f"entry-{n}"}


def _mp_jsonl_worker(args: tuple[str, int]) -> dict[str, object]:
    path, idx = args
    return evidence.append(path, {"data_origin": "synthetic", "worker": idx}, key=f"mp-key-{idx}")


def _mp_sqlite_worker(args: tuple[str, int]) -> dict[str, object]:
    path, idx = args
    store = SQLiteEvidenceStore(path)
    return store.append({"data_origin": "synthetic", "worker": idx}, key=f"sq-mp-{idx}")


class EvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "ledger.jsonl")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_append_and_verify_chain(self) -> None:
        first = evidence.append(self.path, record(1), key="alpha")
        second = evidence.append(self.path, record(2), key="beta")
        self.assertEqual(first["seq"], 1)
        self.assertEqual(first["prev_hash"], evidence.GENESIS)
        self.assertEqual(second["prev_hash"], first["entry_hash"])
        report = evidence.verify(self.path)
        self.assertEqual(report["status"], "OK")
        self.assertEqual(report["count"], 2)

    def test_tampered_record_detected(self) -> None:
        evidence.append(self.path, record(1))
        evidence.append(self.path, record(2))
        with open(self.path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        entry = json.loads(lines[0])
        entry["record"]["value_minor"] = 999
        lines[0] = json.dumps(entry, sort_keys=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        report = evidence.verify(self.path)
        self.assertEqual(report["status"], "TAMPERED")
        self.assertEqual(report["count"], 0)

    def test_reordered_entries_detected(self) -> None:
        for index in (1, 2, 3):
            evidence.append(self.path, record(index))
        with open(self.path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        self.assertEqual(evidence.verify(self.path)["status"], "TAMPERED")

    def test_concurrent_cooperating_writers_preserve_entries(self) -> None:
        with ThreadPoolExecutor(max_workers=8) as pool:
            entries = list(pool.map(lambda n: evidence.append(self.path, record(n), key=f"key-{n}"), range(20)))
        self.assertEqual(len(entries), 20)
        report = evidence.verify(self.path)
        self.assertEqual(report["status"], "OK")
        self.assertEqual(report["count"], 20)

    def test_multiprocessing_writers_preserve_entries(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=4) as pool:
            entries = pool.map(_mp_jsonl_worker, [(self.path, idx) for idx in range(16)])
        self.assertEqual(len(entries), 16)
        report = evidence.verify(self.path)
        self.assertEqual(report["status"], "OK")
        self.assertEqual(report["count"], 16)
        self.assertEqual(sorted(e["seq"] for e in entries), list(range(1, 17)))

    def test_tampered_idempotency_key_detected(self) -> None:
        evidence.append(self.path, record(1), key="original")
        anchor = evidence.head(self.path)
        with open(self.path, encoding="utf-8") as handle:
            entry = json.loads(handle.read())
        entry["key"] = "altered"
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        self.assertEqual(evidence.verify(self.path, expected_head=anchor)["status"], "TAMPERED")
        with self.assertRaises(ValueError):
            evidence.append(self.path, record(2), key="original")

    def test_duplicate_key_rejected(self) -> None:
        evidence.append(self.path, record(1), key="alpha")
        with self.assertRaises(ValueError):
            evidence.append(self.path, record(2), key="alpha")

    def test_truncation_detected_only_with_anchor(self) -> None:
        evidence.append(self.path, record(1))
        evidence.append(self.path, record(2))
        anchor = evidence.head(self.path)
        with open(self.path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()[:1]
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        self.assertEqual(evidence.verify(self.path)["status"], "OK")  # chain alone sees nothing
        report = evidence.verify(self.path, expected_head=anchor)
        self.assertEqual(report["status"], "TAMPERED")
        self.assertEqual(evidence.verify(self.path, expected_count=True)["status"], "TAMPERED")

    def test_append_refuses_broken_chain(self) -> None:
        evidence.append(self.path, record(1))
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write("{not json}\n")
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            evidence.append(self.path, record(2))

    def test_invalid_records_rejected_before_write(self) -> None:
        bad_records: tuple[object, ...] = (
            [],
            "string",
            {"no_origin": 1},
            {"data_origin": "customer"},
            {"data_origin": "synthetic", "x": float("nan")},
        )
        for bad in bad_records:
            with self.assertRaises(ValueError):
                evidence.append(self.path, bad)  # type: ignore[arg-type]
        self.assertFalse(os.path.exists(self.path))

    def test_nested_non_string_keys_rejected_before_write(self) -> None:
        for nested in ({2: 1, 10: 2}, {"inner": [{1: "bad"}]}, ({2: "bad"},)):
            with self.subTest(nested=nested), self.assertRaises(ValueError):
                evidence.append(self.path, {"data_origin": "synthetic", "counts": nested})
        self.assertFalse(os.path.exists(self.path))

    def test_bad_keys_rejected(self) -> None:
        for bad in ("", "  ", "x" * 129):
            with self.assertRaises(ValueError):
                evidence.append(self.path, record(1), key=bad)

    def test_no_tmp_file_left_behind(self) -> None:
        evidence.append(self.path, record(1))
        self.assertFalse(os.path.exists(self.path + ".tmp"))
        leftovers = [name for name in os.listdir(self.tmp.name) if name.startswith(".foundry-evidence-")]
        self.assertEqual(leftovers, [])

    def test_incremental_checkpoint_tamper_and_crash_recovery(self) -> None:
        for idx in range(70):
            evidence.append(self.path, record(idx), key=f"k-{idx:03d}")
        ckpt_file = self.path + ".ckpt"
        self.assertTrue(os.path.isfile(ckpt_file))
        snap = checkpoint(self.path)
        self.assertEqual(snap["count"], 70)
        self.assertEqual(snap["head"], evidence.head(self.path))

        # Corrupting sidecar checkpoint falls back to full verification and recovers on next append
        Path(ckpt_file).write_text("{corrupt", encoding="utf-8")
        evidence.append(self.path, record(70), key="k-070")
        self.assertEqual(evidence.verify(self.path)["count"], 71)

        # Same-length byte tampering in covered prefix is still caught before append
        raw = Path(self.path).read_bytes()
        needle = b'"entry-0"'
        pos = raw.find(needle)
        self.assertNotEqual(pos, -1)
        tampered = raw[:pos] + b'"entry-X"' + raw[pos + len(needle):]
        self.assertEqual(len(tampered), len(raw))
        Path(self.path).write_bytes(tampered)
        with self.assertRaises(ValueError):
            evidence.append(self.path, record(71), key="k-071")

    def test_crash_partial_write_and_leftover_temp_files(self) -> None:
        evidence.append(self.path, record(1), key="one")
        # Leftover temp file from an earlier crash must not interfere
        leftover = Path(self.tmp.name) / ".foundry-evidence-stale.tmp"
        leftover.write_text('{"partial": ', encoding="utf-8")
        self.assertEqual(evidence.verify(self.path)["status"], "OK")

        # File lacking trailing newline recovers cleanly on next valid append
        content = Path(self.path).read_text(encoding="utf-8").rstrip("\n")
        Path(self.path).write_text(content, encoding="utf-8")
        evidence.append(self.path, record(2), key="two")
        self.assertEqual(evidence.verify(self.path)["count"], 2)

        # Simulated mid-line partial write causes fail-closed verify, checkpoint, and append
        before = Path(self.path).read_text(encoding="utf-8")
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write('{"entry_hash":"abc","seq":3')
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            evidence.verify(self.path)
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            checkpoint(self.path)
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            evidence.append(self.path, record(3), key="three")
        self.assertIn('{"entry_hash":"abc","seq":3', Path(self.path).read_text(encoding="utf-8"))
        self.assertTrue(Path(self.path).read_text(encoding="utf-8").startswith(before))

    def test_allowed_origins_policy_tiers(self) -> None:
        with self.assertRaises(ValueError):
            evidence.append(self.path, record(1, origin="consented_pilot"))

        entry = evidence.append(
            self.path,
            record(1, origin="consented_pilot"),
            allowed_origins=("synthetic", "consented_pilot"),
        )
        self.assertEqual(entry["record"]["data_origin"], "consented_pilot")

        with self.assertRaises(ValueError):
            evidence.append(
                self.path,
                record(2, origin="redacted_eval"),
                allowed_origins=("synthetic", "consented_pilot"),
            )
        for bad_origins in ("synthetic", (), ("customer",)):
            with self.assertRaises(ValueError):
                evidence.append(self.path, record(3), allowed_origins=bad_origins)  # type: ignore[arg-type]

    def test_jsonl_store_protocol_and_checkpoint(self) -> None:
        store = JsonlEvidenceStore(self.path)
        self.assertIsInstance(store, EvidenceStore)
        e1 = store.append(record(1), key="s1")
        e2 = store.append_with_builder(lambda entries, prev: record(len(entries) + 10), key="s2")
        self.assertEqual(e2["prev_hash"], e1["entry_hash"])
        snap = store.checkpoint()
        self.assertEqual(snap, {"head": e2["entry_hash"], "count": 2})
        self.assertEqual(store.head(), e2["entry_hash"])
        self.assertEqual(len(store.entries()), 2)

        # Tamper and verify checkpoint/entries fail closed
        Path(self.path).write_text(
            Path(self.path).read_text(encoding="utf-8").replace('"value_minor":1', '"value_minor":9'),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            store.checkpoint()
        with self.assertRaises(ValueError):
            store.entries()

    def test_sqlite_evidence_store_wal_and_tamper_detection(self) -> None:
        db_path = os.path.join(self.tmp.name, "ledger.db")
        store = SQLiteEvidenceStore(db_path)
        self.assertIsInstance(store, EvidenceStore)
        self.assertEqual(store.checkpoint(), {"head": evidence.GENESIS, "count": 0})

        with closing(sqlite3.connect(db_path)) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            self.assertEqual(mode.lower(), "wal")

        e1 = store.append(record(1), key="a")
        e2 = store.append(record(2), key="b")
        self.assertEqual(e2["prev_hash"], e1["entry_hash"])
        self.assertEqual(store.checkpoint(), {"head": e2["entry_hash"], "count": 2})
        self.assertEqual(store.verify(expected_head=e2["entry_hash"], expected_count=2)["status"], "OK")
        self.assertEqual(store.verify(expected_head="f" * 64)["status"], "TAMPERED")
        self.assertEqual(store.verify(expected_count=1)["status"], "TAMPERED")

        with self.assertRaises(ValueError):
            store.append(record(3), key="a")
        with self.assertRaises(ValueError):
            store.append(record(4, origin="consented_pilot"))

        # Append-only SQL triggers block UPDATE and DELETE
        with closing(sqlite3.connect(db_path)) as conn:
            with self.assertRaises(sqlite3.DatabaseError):
                conn.execute("UPDATE evidence_entries SET record_json = '{}' WHERE seq = 1")
            with self.assertRaises(sqlite3.DatabaseError):
                conn.execute("DELETE FROM evidence_entries WHERE seq = 2")

        # Concurrent threads + multiprocessing preserve full chain in SQLiteEvidenceStore
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda n: store.append(record(n), key=f"thr-{n}"), range(10)))
        self.assertEqual(store.verify()["count"], 12)

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=3) as pool:
            pool.map(_mp_sqlite_worker, [(db_path, idx) for idx in range(9)])
        self.assertEqual(store.verify()["count"], 21)

        # Simulate bypass tamper (dropping trigger + altering payload) -> fail closed
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("DROP TRIGGER evidence_no_update")
            conn.execute("UPDATE evidence_entries SET record_json = '{\"data_origin\":\"synthetic\",\"value_minor\":999}' WHERE seq = 1")
            conn.commit()
        self.assertEqual(store.verify()["status"], "TAMPERED")
        with self.assertRaises(ValueError):
            store.checkpoint()
        with self.assertRaises(ValueError):
            store.append(record(99), key="after-tamper")


if __name__ == "__main__":
    unittest.main()
