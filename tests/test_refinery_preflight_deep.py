"""Unit tests covering edge cases and failure modes in refinery_preflight."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from hardonian_foundry.refinery_preflight import run_preflight


class RefineryPreflightDeepTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_preflight_default_success(self):
        res = run_preflight(output_root=self.root)
        self.assertEqual(res["status"], "PASSED")
        self.assertEqual(res["errors"], [])
        check_names = {c["name"] for c in res["checks"]}
        self.assertIn("dataset_validation", check_names)
        self.assertIn("runtime_environment", check_names)
        self.assertIn("output_directory", check_names)
        self.assertIn("licensing_and_provenance", check_names)
        self.assertIn("execution_safety", check_names)

    def test_preflight_dataset_failure(self):
        bad_dataset = [{"id": "bad-record"}]  # missing gold_label/prompt
        res = run_preflight(dataset=bad_dataset, output_root=self.root)
        self.assertEqual(res["status"], "FAILED")
        self.assertTrue(any("dataset validation failed" in e for e in res["errors"]))

    def test_preflight_low_disk_space(self):
        # Mock shutil.disk_usage to return < 50 MB
        low_disk_stat = MagicMock(free=10 * 1024 * 1024)  # 10 MB free
        with patch("shutil.disk_usage", return_value=low_disk_stat):
            res = run_preflight(output_root=self.root)
            self.assertEqual(res["status"], "FAILED")
            self.assertTrue(any("insufficient disk space" in e for e in res["errors"]))

    def test_preflight_output_root_oserror(self):
        with patch.object(Path, "mkdir", side_effect=PermissionError("read-only filesystem")):
            res = run_preflight(output_root=self.root / "protected")
            self.assertEqual(res["status"], "FAILED")
            self.assertTrue(any("failed to initialize output directory" in e for e in res["errors"]))

    def test_preflight_model_manifest_allowed_and_denied(self):
        # Allowed manifest
        allowed_manifest = {
            "name": "synthetic-reviewer",
            "license_id": "Apache-2.0",
            "provenance": "synthetic",
            "data_origin": "synthetic",
            "weights_sha256": "0" * 64,
        }
        res_ok = run_preflight(
            output_root=self.root,
            model_manifest=allowed_manifest,
            allowed_data_origins=["synthetic"],
        )
        self.assertEqual(res_ok["status"], "PASSED")

        # Denied manifest (e.g. unknown proprietary license or unapproved origin)
        denied_manifest = {
            "name": "test-model",
            "license_id": "PROPRIETARY-COMMERCIAL",
            "provenance": "customer-data",
            "data_origin": "customer-live",
            "weights_sha256": "0" * 64,
        }
        res_denied = run_preflight(
            output_root=self.root,
            model_manifest=denied_manifest,
            allowed_data_origins=["synthetic"],
        )
        self.assertEqual(res_denied["status"], "FAILED")
        self.assertTrue(any("licensing/provenance" in e for e in res_denied["errors"]))

    def test_preflight_task_spec(self):
        task_spec = {
            "schema_version": "1.0",
            "task_id": "preflight-task",
            "description": "Task spec for preflight.",
            "labels": ["PASS", "FAIL"],
        }
        task_data = [
            {"id": f"r{i}", "prompt": f"p{i}", "gold_label": "PASS" if i % 2 == 0 else "FAIL", "critical": False, "family": f"f{i}"}
            for i in range(10)
        ]
        res = run_preflight(dataset=task_data, task_spec=task_spec, output_root=self.root)
        self.assertEqual(res["status"], "PASSED")

    def test_preflight_license_compat_typeerror_fallback(self):
        # Test facade evaluate_license that does not accept allowed_data_origins keyword
        def legacy_eval(manifest):
            return {"decision": "ALLOWED", "reasons": []}

        with patch("hardonian_foundry.refinery_preflight.facade_attr", return_value=legacy_eval):
            res = run_preflight(
                output_root=self.root,
                model_manifest={"name": "test"},
                allowed_data_origins=["synthetic"],
            )
            self.assertEqual(res["status"], "PASSED")


if __name__ == "__main__":
    unittest.main()
