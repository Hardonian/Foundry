"""Deep unit tests for hardonian_foundry.refinery_validation covering path sanitization and dataset validation."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hardonian_foundry.refinery_validation import (
    resolve_ledger_path,
    validate_and_split_dataset,
    validate_run_id,
)


class RefineryValidationDeepTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def test_validate_run_id(self):
        self.assertEqual(validate_run_id("run-123_test.0"), "run-123_test.0")
        with self.assertRaises(ValueError):
            validate_run_id("bad/slash")
        with self.assertRaises(ValueError):
            validate_run_id("..")
        with self.assertRaises(ValueError):
            validate_run_id("")
        with self.assertRaises(ValueError):
            validate_run_id(123)  # type: ignore[arg-type]

    def test_resolve_ledger_path_edge_cases(self):
        # Empty string with allow_legacy_outside_fallback
        default_ledger = self.root / "evidence.jsonl"
        res = resolve_ledger_path(self.root, "", allow_legacy_outside_fallback=True)
        self.assertEqual(res, default_ledger)

        # Empty string without fallback raises
        with self.assertRaises(ValueError):
            resolve_ledger_path(self.root, "", allow_legacy_outside_fallback=False)

        # Unprintable characters
        with self.assertRaises(ValueError):
            resolve_ledger_path(self.root, "bad\x01path.jsonl")

        # Backslash in relative path on POSIX or colon
        with self.assertRaises(ValueError):
            resolve_ledger_path(self.root, "nested:rel.jsonl")

        # Path traversal
        with self.assertRaises(ValueError):
            resolve_ledger_path(self.root, "../escape.jsonl")

        # Absolute path inside root
        inside_abs = str(self.root / "evidence.jsonl")
        self.assertEqual(resolve_ledger_path(self.root, inside_abs), Path(inside_abs))

    def test_validate_dataset_non_dict_row(self):
        dataset = [
            {"id": "r1", "expected_minor": 100, "actual_minor": 90, "critical": False},
            "not-a-dict",
            {"id": "r2", "expected_minor": 200, "actual_minor": 200, "critical": False},
        ]
        res, splits = validate_and_split_dataset(dataset)
        self.assertEqual(res["status"], "INVALID")
        self.assertIsNone(splits)
        self.assertTrue(any("not a dictionary" in e for e in res["errors"]))


if __name__ == "__main__":
    unittest.main()
