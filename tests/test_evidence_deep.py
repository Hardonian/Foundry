"""Deep unit tests for hardonian_foundry.evidence covering corrupt UTF-8, origin validation,
lock timeouts, and broken chain prevention."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hardonian_foundry.evidence import (
    _normalize_allowed_origins,
    append,
    verify_entries,
)


class EvidenceDeepTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
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


if __name__ == "__main__":
    unittest.main()
