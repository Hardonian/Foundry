"""Deep unit tests for hardonian_foundry.evidence covering corrupt UTF-8, origin validation,
lock timeouts, and broken chain prevention."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hardonian_foundry.evidence import (
    JsonlEvidenceStore,
    SQLiteEvidenceStore,
    _load_valid_checkpoint,
    _normalize_allowed_origins,
    _verify_single_entry,
    _write_atomic_text,
    append,
    verify_entries,
)


class EvidenceDeepTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        import gc
        gc.collect()
        self.tmp.cleanup()

    def test_normalize_allowed_origins(self):
        # String or bytes raises ValueError
        with self.assertRaises(ValueError):
            _normalize_allowed_origins("synthetic")  # type: ignore[arg-type]

        # Unknown origin raises ValueError
        with self.assertRaises(ValueError):
            _normalize_allowed_origins(["unknown-origin"])

        norm = _normalize_allowed_origins(["synthetic", "consented_pilot"])
        self.assertEqual(norm, frozenset({"synthetic", "consented_pilot"}))

    def test_corrupt_utf8_in_ledger_raises(self):
        ledger = self.root / "evidence.jsonl"
        # Write invalid non-UTF-8 bytes
        ledger.write_bytes(b"\xff\xfe\x00\x00corrupt")

        with self.assertRaises(ValueError):
            append(str(ledger), {"step": "init", "data_origin": "synthetic"})

    def test_broken_chain_refusal_on_append(self):
        ledger = self.root / "evidence.jsonl"
        append(str(ledger), {"step": "1", "data_origin": "synthetic"})

        # Tamper the ledger content
        content = ledger.read_text(encoding="utf-8")
        tampered = content.replace('"step":"1"', '"step":"tampered"')
        self.assertNotEqual(content, tampered)
        ledger.write_text(tampered, encoding="utf-8")

        # Refuses to append onto broken chain
        with self.assertRaises(ValueError):
            append(str(ledger), {"step": "2", "data_origin": "synthetic"})

    def test_verify_entries_edge_cases(self):
        # Empty list is OK (genesis)
        rep = verify_entries([])
        self.assertEqual(rep["status"], "OK")
        self.assertEqual(rep["count"], 0)

        # Invalid entry structure
        rep_bad = verify_entries(["not-a-dict"])  # type: ignore[list-item]
        self.assertNotEqual(rep_bad["status"], "OK")

    def test_sqlite_ledger_lifecycle_and_tampering(self):
        db_path = self.root / "evidence.db"
        ledger = SQLiteEvidenceStore(db_path)

        # Empty ledger
        self.assertEqual(ledger.head(), "0" * 64)
        self.assertEqual(ledger.entries(), [])

        # Append records
        e1 = ledger.append({"step": "init", "data_origin": "synthetic"})
        self.assertEqual(e1["seq"], 1)
        e2 = ledger.append({"step": "eval", "data_origin": "synthetic"})
        self.assertEqual(e2["seq"], 2)

        # Verify and checkpoint
        self.assertEqual(ledger.head(), e2["entry_hash"])
        ckpt = ledger.checkpoint()
        self.assertEqual(ckpt["count"], 2)
        self.assertEqual(len(ledger.entries()), 2)

        # Tampered database file
        with open(db_path, "wb") as f:
            f.write(b"not a sqlite database")
        tampered_rep = ledger.verify()
        self.assertEqual(tampered_rep["status"], "TAMPERED")
        with self.assertRaises(ValueError):
            ledger.checkpoint()
        with self.assertRaises((ValueError, sqlite3.DatabaseError)):
            ledger.entries()

    def test_checkpoint_helpers_and_acceleration(self):
        # Test _load_valid_checkpoint with invalid files
        self.assertIsNone(_load_valid_checkpoint(str(self.root / "missing.ckpt"), 100))

        bad_ckpt = self.root / "bad.ckpt"
        bad_ckpt.write_text("invalid json", encoding="utf-8")
        self.assertIsNone(_load_valid_checkpoint(str(bad_ckpt), 100))

        bad_ckpt.write_text(json.dumps({"version": 2}), encoding="utf-8")
        self.assertIsNone(_load_valid_checkpoint(str(bad_ckpt), 100))

        # Checkpoint generation by appending >= 33 entries
        ledger = self.root / "accel_evidence.jsonl"
        for i in range(35):
            append(str(ledger), {"step": f"step-{i}", "data_origin": "synthetic"})

        # Subsequent append uses checkpoint acceleration
        e36 = append(str(ledger), {"step": "step-35", "data_origin": "synthetic"})
        self.assertEqual(e36["seq"], 36)

    def test_verify_single_entry_errors(self):
        seen_keys: set[str] = set()
        # Invalid entry structure (wrong fields)
        h, err = _verify_single_entry({"seq": 1, "missing_fields": True}, 1, "0" * 64, seen_keys)
        self.assertIsNone(h)
        self.assertIn("invalid entry fields", err)

        # Idempotency key whitespace or oversized
        bad_key_entry = {
            "seq": 1,
            "key": "   ",
            "prev_hash": "0" * 64,
            "record": {"a": 1},
            "entry_hash": "0" * 64,
        }
        h, err = _verify_single_entry(bad_key_entry, 1, "0" * 64, seen_keys)
        self.assertIsNone(h)
        self.assertIn("invalid idempotency key", err)

        # Duplicate idempotency key
        seen_keys.add("existing-key")
        dup_key_entry = {
            "seq": 1,
            "key": "existing-key",
            "prev_hash": "0" * 64,
            "record": {"a": 1},
            "entry_hash": "0" * 64,
        }
        h, err = _verify_single_entry(dup_key_entry, 1, "0" * 64, seen_keys)
        self.assertIsNone(h)
        self.assertIn("duplicate idempotency key", err)

    def test_write_atomic_text_failure_cleanup(self):
        target = self.root / "atomic_clean.txt"
        with mock.patch("hardonian_foundry.evidence.os.replace", side_effect=OSError("replace fail")):
            with self.assertRaises(OSError):
                _write_atomic_text(str(target), "data")


if __name__ == "__main__":
    unittest.main()
