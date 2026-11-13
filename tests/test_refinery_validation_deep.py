"""Deep unit tests for hardonian_foundry.refinery_validation covering path sanitization and dataset validation."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hardonian_foundry.refinery_validation import (
    _code_digest,
    facade_attr,
    resolve_ledger_path,
    resolve_run_dir,
    validate_and_split_dataset,
    validate_dataset,
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

        # Legacy absolute evidence.jsonl fallback returns default ledger (line 115)
        outside_path = "D:\\outside\\evidence.jsonl" if str(self.root).startswith("C:") else "/outside/evidence.jsonl"
        res_leg = resolve_ledger_path(self.root, outside_path, allow_legacy_outside_fallback=True)
        self.assertEqual(res_leg, default_ledger)

    def test_facade_attr_fallback(self):
        # Attribute nonexistent in refinery
        res = facade_attr("totally_nonexistent_xyz", 999)
        self.assertEqual(res, 999)

    def test_code_digest_directory_handling(self):
        # Normal code digest returns 64-char hex
        d = _code_digest()
        self.assertEqual(len(d), 64)

        # Mock non-file entry in glob (line 32 false branch)
        mock_file = mock.MagicMock(spec=Path)
        mock_file.name = "sub.py"
        mock_file.is_file.return_value = False
        with mock.patch.object(Path, "glob", return_value=[mock_file]):
            d_empty = _code_digest()
            self.assertEqual(len(d_empty), 64)

    def test_validate_dataset_non_dict_row(self):
        res = validate_dataset([
            {"id": "synthetic-001", "expected_minor": 101, "actual_minor": 101, "critical": False},
            "not-a-dict",
            {"id": "synthetic-002", "expected_minor": 102, "actual_minor": 102, "critical": False},
        ])
        self.assertEqual(res["status"], "INVALID")
        self.assertTrue(any("not a dictionary" in e for e in res["errors"]))

    def test_resolve_run_dir_escape(self):
        # Mock is_relative_to returning False
        with mock.patch.object(Path, "is_relative_to", return_value=False):
            with self.assertRaises(ValueError) as ctx:
                resolve_run_dir(self.root, "run-valid")
            self.assertIn("escapes output_root", str(ctx.exception))

    def test_resolve_ledger_path_symlink_escape_errors(self):
        # Line 80: default ledger symlink escape on empty raw path
        with mock.patch.object(Path, "is_relative_to", return_value=False):
            with self.assertRaises(ValueError) as ctx:
                resolve_ledger_path(self.root, "", allow_legacy_outside_fallback=True)
            self.assertIn("symlink escapes output_root", str(ctx.exception))

        # Line 102: relative path escape
        with mock.patch.object(Path, "is_relative_to", return_value=False):
            with self.assertRaises(ValueError) as ctx:
                resolve_ledger_path(self.root, "ledger.jsonl")
            self.assertIn("escapes output_root", str(ctx.exception))

        # Line 114: legacy absolute evidence.jsonl fallback symlink escape
        with mock.patch.object(Path, "is_relative_to", side_effect=[False, False]):
            with self.assertRaises(ValueError) as ctx:
                resolve_ledger_path(self.root, "/legacy/path/evidence.jsonl", allow_legacy_outside_fallback=True)
            self.assertIn("symlink escapes output_root", str(ctx.exception))

    def test_validate_and_split_dataset_split_exception(self):
        # Valid records with unique contents, but mock split raising an error
        valid_records = [
            {"id": f"synthetic-{i:03d}", "expected_minor": 100 + i, "actual_minor": 100 + i, "critical": False}
            for i in range(5)
        ]
        with mock.patch("hardonian_foundry.refinery_validation.split", side_effect=ValueError("mock split failure")):
            res, splits = validate_and_split_dataset(valid_records)
            self.assertEqual(res["status"], "INVALID")
            self.assertIsNone(splits)
            self.assertTrue(any("failed to partition dataset: mock split failure" in e for e in res["errors"]))


if __name__ == "__main__":
    unittest.main()
