"""Deep unit tests for hardonian_foundry.refinery_runs covering bundle export (tar.gz/dir),
bundle verification failure modes, run diffing, and run listing."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hardonian_foundry.refinery_runs import (
    create_run,
    diff_runs,
    export_bundle,
    list_runs,
    verify_bundle,
)


class RefineryRunsDeepTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runs_dir = self.root / "runs"

    def tearDown(self):
        self.tmp.cleanup()

    def test_bundle_export_tar_and_dir_roundtrip(self):
        run = create_run(output_root=self.runs_dir)
        rid = run["run_id"]

        # Export as dir
        dir_bundle = self.root / "bundle_dir"
        res_dir = export_bundle(rid, dest_path=dir_bundle, fmt="dir", output_root=self.runs_dir)
        self.assertEqual(res_dir["format"], "dir")
        self.assertTrue((dir_bundle / "manifest.json").is_file())
        v_dir = verify_bundle(dir_bundle)
        self.assertEqual(v_dir["status"], "VERIFIED")

        # Export as tar.gz
        tar_bundle = self.root / "bundle.tar.gz"
        res_tar = export_bundle(rid, dest_path=tar_bundle, fmt="tar.gz", output_root=self.runs_dir)
        self.assertEqual(res_tar["format"], "tar.gz")
        self.assertTrue(tar_bundle.is_file())
        v_tar = verify_bundle(tar_bundle)
        self.assertEqual(v_tar["status"], "VERIFIED")

        # Unsupported format raises ValueError
        with self.assertRaises(ValueError):
            export_bundle(rid, dest_path=self.root / "err", fmt="unsupported-fmt", output_root=self.runs_dir)

    def test_verify_bundle_tamper_checks(self):
        run = create_run(output_root=self.runs_dir)
        rid = run["run_id"]
        dir_bundle = self.root / "bundle_tamper"
        export_bundle(rid, dest_path=dir_bundle, fmt="dir", output_root=self.runs_dir)

        # 1. Nonexistent bundle raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            verify_bundle(self.root / "nonexistent_bundle")

        # 2. Missing manifest.json raises ValueError
        (dir_bundle / "manifest.json").unlink()
        with self.assertRaises(ValueError):
            verify_bundle(dir_bundle)

        # Re-export clean bundle
        export_bundle(rid, dest_path=dir_bundle, fmt="dir", output_root=self.runs_dir)

        # 3. Tampered manifest hash raises ValueError
        manifest_path = dir_bundle / "manifest.json"
        m_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        m_data["bundle_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(m_data), encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_bundle(dir_bundle)

        # Re-export clean bundle
        export_bundle(rid, dest_path=dir_bundle, fmt="dir", output_root=self.runs_dir)

        # 4. Tampered run.json file content raises ValueError
        (dir_bundle / "run.json").write_text('{"tampered": true}', encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_bundle(dir_bundle)

    def test_diff_runs_errors(self):
        run1 = create_run(output_root=self.runs_dir)
        with self.assertRaises(FileNotFoundError):
            diff_runs(run1["run_id"], "nonexistent-run", output_root=self.runs_dir)
        with self.assertRaises(FileNotFoundError):
            diff_runs("nonexistent-run", run1["run_id"], output_root=self.runs_dir)

    def test_export_bundle_tenant_mismatch_raises(self):
        from hardonian_foundry.tenancy import TenantContext

        run = create_run(output_root=self.runs_dir)
        rid = run["run_id"]
        # The default run doesn't have tenant_id matching foreign-tenant
        with self.assertRaises(PermissionError):
            export_bundle(
                rid,
                dest_path=self.root / "b_foreign",
                fmt="dir",
                output_root=self.runs_dir,
                tenant_context=TenantContext("foreign-tenant", "req-1"),
            )

    def test_list_runs(self):
        # Empty
        self.assertEqual(list_runs(output_root=self.runs_dir), [])

        run1 = create_run(output_root=self.runs_dir)
        runs = list_runs(output_root=self.runs_dir, verify_provenance=True)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], run1["run_id"])
        self.assertEqual(runs[0]["provenance_status"], "PROVENANCE_VERIFIED")


if __name__ == "__main__":
    unittest.main()
