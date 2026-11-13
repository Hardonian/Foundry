"""Comprehensive unit tests for hardonian_foundry.artifact content-addressed store."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from hardonian_foundry.artifact import (
    _resolve_store_dir,
    _validate_label,
    add_artifact,
    get_artifact_path,
    list_artifacts,
    sha256_file,
    verify_artifact,
)


class ArtifactStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = self.root / "store"

    def tearDown(self):
        self.tmp.cleanup()

    def test_sha256_file(self):
        f = self.root / "sample.txt"
        f.write_bytes(b"hello hardonian foundry")
        expected = hashlib.sha256(b"hello hardonian foundry").hexdigest()
        digest, size = sha256_file(f)
        self.assertEqual(digest, expected)
        self.assertEqual(size, len(b"hello hardonian foundry"))

        # Nonexistent file
        with self.assertRaises(FileNotFoundError):
            sha256_file(self.root / "nonexistent.txt")

        # Directory instead of file
        with self.assertRaises(ValueError):
            sha256_file(self.root)

    def test_resolve_store_dir(self):
        explicit = self.root / "custom"
        self.assertEqual(_resolve_store_dir(store_dir=explicit), explicit.resolve())
        out_root = self.root / "runs"
        self.assertEqual(_resolve_store_dir(output_root=out_root), (out_root / "artifacts").resolve())
        default_dir = _resolve_store_dir()
        self.assertTrue(str(default_dir).endswith(str(Path(".foundry") / "runs" / "artifacts")))

    def test_validate_label(self):
        self.assertEqual(_validate_label("clean_name-1.0", "name", 32), "clean_name-1.0")
        with self.assertRaises(ValueError):
            _validate_label("", "name", 32)
        with self.assertRaises(ValueError):
            _validate_label(" leading_space", "name", 32)
        with self.assertRaises(ValueError):
            _validate_label("a" * 33, "name", 32)
        with self.assertRaises(ValueError):
            _validate_label("path/traversal", "name", 32)
        with self.assertRaises(ValueError):
            _validate_label("win\\path", "name", 32)
        with self.assertRaises(ValueError):
            _validate_label("dots..name", "name", 32)
        with self.assertRaises(ValueError):
            _validate_label("null\x00byte", "name", 32)

    def test_add_and_get_artifact(self):
        src = self.root / "weights.bin"
        src.write_bytes(b"model-weights-v1")
        expected_sha = hashlib.sha256(b"model-weights-v1").hexdigest()

        entry = add_artifact(src, store_dir=self.store, name="weights.bin", kind="adapter", run_id="run-123")
        self.assertEqual(entry["sha256"], expected_sha)
        self.assertEqual(entry["name"], "weights.bin")
        self.assertEqual(entry["kind"], "adapter")
        self.assertEqual(entry["run_id"], "run-123")

        # Get artifact path succeeds
        blob_path = get_artifact_path(expected_sha, store_dir=self.store)
        self.assertTrue(blob_path.is_file())
        self.assertEqual(blob_path.read_bytes(), b"model-weights-v1")

        # Adding same artifact again succeeds idempotently
        entry2 = add_artifact(src, store_dir=self.store, name="weights.bin")
        self.assertEqual(entry2["sha256"], expected_sha)

        # Corrupt existing blob triggers error on add
        blob_path.write_bytes(b"tampered-blob")
        with self.assertRaises(ValueError):
            add_artifact(src, store_dir=self.store)

    def test_get_artifact_path_errors(self):
        with self.assertRaises(ValueError):
            get_artifact_path("not-a-valid-hex", store_dir=self.store)
        with self.assertRaises(FileNotFoundError):
            get_artifact_path("0" * 64, store_dir=self.store)

        # If blob file exists but digest mismatches
        blobs_dir = self.store / "blobs" / "sha256"
        blobs_dir.mkdir(parents=True, exist_ok=True)
        fake_sha = "1" * 64
        (blobs_dir / fake_sha).write_bytes(b"different content")
        with self.assertRaises(ValueError):
            get_artifact_path(fake_sha, store_dir=self.store)

    def test_add_artifact_validation_errors(self):
        src = self.root / "bad.bin"
        src.write_bytes(b"data")
        with self.assertRaises(ValueError):
            add_artifact(src, store_dir=self.store, name=" bad_name ")
        with self.assertRaises(ValueError):
            add_artifact(src, store_dir=self.store, name="invalid/slash")
        with self.assertRaises(ValueError):
            add_artifact(src, store_dir=self.store, kind="k" * 100)
        with self.assertRaises(ValueError):
            add_artifact(src, store_dir=self.store, run_id="invalid space id")

    def test_verify_artifact(self):
        src = self.root / "data.bin"
        src.write_bytes(b"deterministic-payload")
        digest = hashlib.sha256(b"deterministic-payload").hexdigest()

        # Before adding to store: verify by hex gives MISSING
        res_missing = verify_artifact(digest, store_dir=self.store)
        self.assertEqual(res_missing["status"], "MISSING")
        self.assertFalse(res_missing["verified"])

        # Add to store
        add_artifact(src, store_dir=self.store)

        # Verify by hex
        res_hex = verify_artifact(digest, store_dir=self.store)
        self.assertEqual(res_hex["status"], "VERIFIED")
        self.assertTrue(res_hex["verified"])

        # Verify by hex with mismatching expected_sha256
        res_mismatch = verify_artifact(digest, expected_sha256="f" * 64, store_dir=self.store)
        self.assertEqual(res_mismatch["status"], "MISMATCH")
        self.assertFalse(res_mismatch["verified"])

        # Verify by file path with expected_sha256
        res_file_ok = verify_artifact(src, expected_sha256=digest)
        self.assertEqual(res_file_ok["status"], "VERIFIED")
        self.assertTrue(res_file_ok["verified"])

        res_file_bad = verify_artifact(src, expected_sha256="a" * 64)
        self.assertEqual(res_file_bad["status"], "MISMATCH")

        # Verify by file path against store (no expected_sha256 provided)
        res_store_check = verify_artifact(src, store_dir=self.store)
        self.assertEqual(res_store_check["status"], "VERIFIED")

        # Verify by file path not registered in store
        unregistered = self.root / "unreg.bin"
        unregistered.write_bytes(b"not in store")
        res_unreg = verify_artifact(unregistered, store_dir=self.store)
        self.assertEqual(res_unreg["status"], "MISSING")

        # Verify non-existent file path
        res_nonexist = verify_artifact(self.root / "missing.bin")
        self.assertEqual(res_nonexist["status"], "MISSING")

        # Invalid expected_sha256
        with self.assertRaises(ValueError):
            verify_artifact(src, expected_sha256="not-hex")

    def test_list_artifacts(self):
        # Empty store
        self.assertEqual(list_artifacts(store_dir=self.store), [])

        f1 = self.root / "f1.bin"
        f1.write_bytes(b"content 1")
        f2 = self.root / "f2.bin"
        f2.write_bytes(b"content 2")

        add_artifact(f1, store_dir=self.store, run_id="run-A")
        add_artifact(f2, store_dir=self.store, run_id="run-B")

        # Corrupt metadata file is skipped
        meta_dir = self.store / "metadata"
        (meta_dir / "corrupt.json").write_text("invalid json", encoding="utf-8")
        (meta_dir / "non_dict.json").write_text("[1, 2, 3]", encoding="utf-8")

        all_items = list_artifacts(store_dir=self.store)
        self.assertEqual(len(all_items), 2)

        run_a_items = list_artifacts(store_dir=self.store, run_id="run-A")
        self.assertEqual(len(run_a_items), 1)
        self.assertEqual(run_a_items[0]["run_id"], "run-A")

        run_b_items = list_artifacts(store_dir=self.store, run_id="run-B")
        self.assertEqual(len(run_b_items), 1)
        self.assertEqual(run_b_items[0]["run_id"], "run-B")


if __name__ == "__main__":
    unittest.main()
