"""Deep unit tests for hardonian_foundry.refinery_runs covering bundle export (tar.gz/dir/zip),
bundle verification failure modes, run diffing with task metrics, run listing, task runs, and tamper checks."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hardonian_foundry.refinery_runs import (
    create_run,
    create_task_run,
    diff_runs,
    export_bundle,
    inspect_run,
    list_runs,
    verify_bundle,
)
from hardonian_foundry.task_eval import split_task_dataset
from hardonian_foundry.tenancy import TenantContext


class RefineryRunsDeepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runs_dir = self.root / "runs"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_bundle_export_tar_dir_zip_roundtrip(self) -> None:
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

        # Export as zip
        zip_bundle = self.root / "bundle.zip"
        res_zip = export_bundle(rid, dest_path=zip_bundle, fmt="zip", output_root=self.runs_dir)
        self.assertEqual(res_zip["format"], "zip")
        self.assertTrue(zip_bundle.is_file())
        v_zip = verify_bundle(zip_bundle)
        self.assertEqual(v_zip["status"], "VERIFIED")

        # Unsupported format raises ValueError
        with self.assertRaises(ValueError):
            export_bundle(rid, dest_path=self.root / "err", fmt="unsupported-fmt", output_root=self.runs_dir)

    def test_verify_bundle_tamper_checks(self) -> None:
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

    def test_diff_runs_errors(self) -> None:
        run1 = create_run(output_root=self.runs_dir)
        with self.assertRaises(FileNotFoundError):
            diff_runs(run1["run_id"], "nonexistent-run", output_root=self.runs_dir)
        with self.assertRaises(FileNotFoundError):
            diff_runs("nonexistent-run", run1["run_id"], output_root=self.runs_dir)

    def test_export_bundle_unverified_and_tenant_mismatch(self) -> None:
        run = create_run(output_root=self.runs_dir)
        rid = run["run_id"]

        # The default run doesn't have tenant_id matching foreign-tenant
        with self.assertRaises(PermissionError):
            export_bundle(
                rid,
                dest_path=self.root / "b.tar.gz",
                output_root=self.runs_dir,
                tenant_context=TenantContext(tenant_id="foreign-tenant", request_id="req-1"),
            )

        # Tampered unverified run export refusal
        run_file = self.runs_dir / rid / "run.json"
        raw = json.loads(run_file.read_text(encoding="utf-8"))
        raw["dataset_sha256"] = "0" * 64
        run_file.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            export_bundle(rid, dest_path=self.root / "tampered.tar.gz", output_root=self.runs_dir)
        self.assertIn("refusing to export unverified run", str(ctx.exception))

    def test_diff_runs_success_and_list_runs_filtering(self) -> None:
        run1 = create_run(output_root=self.runs_dir)
        run2 = create_run(output_root=self.runs_dir)

        # Successful diff
        diff = diff_runs(run1["run_id"], run2["run_id"], output_root=self.runs_dir)
        self.assertEqual(diff["run_id_a"], run1["run_id"])
        self.assertEqual(diff["run_id_b"], run2["run_id"])

        # list_runs
        all_runs = list_runs(output_root=self.runs_dir)
        self.assertGreaterEqual(len(all_runs), 2)

        # Corrupt run directory in output_root is ignored safely
        corrupt_dir = self.runs_dir / "corrupt-run"
        corrupt_dir.mkdir(parents=True)
        (corrupt_dir / "run.json").write_text("invalid json", encoding="utf-8")
        runs_still_ok = list_runs(output_root=self.runs_dir)
        self.assertGreaterEqual(len(runs_still_ok), 2)

    def test_list_runs_and_tenant_filtering(self) -> None:
        # Empty
        self.assertEqual(list_runs(output_root=self.runs_dir), [])

        tc1 = TenantContext(tenant_id="tenant-alpha", request_id="req-a")
        run1 = create_run(output_root=self.runs_dir, tenant_context=tc1)
        runs = list_runs(output_root=self.runs_dir, verify_provenance=True, tenant_context=tc1)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], run1["run_id"])
        self.assertEqual(runs[0]["provenance_status"], "PROVENANCE_VERIFIED")

        # Foreign tenant cannot see run1
        tc_foreign = TenantContext(tenant_id="tenant-beta", request_id="req-b")
        runs_foreign = list_runs(output_root=self.runs_dir, tenant_context=tc_foreign)
        self.assertEqual(len(runs_foreign), 0)

    def test_inspect_run_metadata_and_ledger_mismatches(self) -> None:
        run = create_run(output_root=self.runs_dir)
        rid = run["run_id"]

        # Anchor count mismatch
        ins_anchor = inspect_run(rid, output_root=self.runs_dir, expected_count=999)
        self.assertEqual(ins_anchor["provenance_status"], "TAMPERED_ANCHOR_MISMATCH")
        self.assertFalse(ins_anchor["anchor_verified"])

        # Missing provenance raises TypeError
        run_file = self.runs_dir / rid / "run.json"
        run_file.write_text('{"schema_version": "1.0"}', encoding="utf-8")
        with self.assertRaises(TypeError):
            inspect_run(rid, output_root=self.runs_dir)

        # Ledger not found
        (self.runs_dir / "evidence.jsonl").unlink()
        run_file.write_text(json.dumps(run), encoding="utf-8")
        ins_no_ledger = inspect_run(rid, output_root=self.runs_dir)
        self.assertEqual(ins_no_ledger["provenance_status"], "LEDGER_NOT_FOUND")

    def test_create_task_run_full(self) -> None:
        task_spec = {
            "schema_version": "1.0",
            "task_id": "eval-task",
            "description": "Evaluation task description.",
            "labels": ["PASS", "FAIL"],
            "thresholds": {"macro_f1_min": 0.85},
            "model": {
                "base_model_id": "test/model",
                "revision": "main",
                "adapter_sha256": "a" * 64,
                "adapter_name": "adapter-v1",
            },
            "split_seed": 42,
        }
        dataset = [
            {
                "id": f"rec_{i}",
                "prompt": f"prompt content {i}",
                "gold_label": "PASS" if i % 2 == 0 else "FAIL",
                "critical": False,
                "family": f"fam_{i % 3}",
            }
            for i in range(12)
        ]
        splits = split_task_dataset(dataset, task_spec)
        test_rows = splits["test"]
        test_preds = [
            {"id": r["id"], "label": r["gold_label"], "confidence": 0.95, "latency_ms": 12.0}
            for r in test_rows
        ]
        ood_preds = [
            {"id": "ood_1", "label": "PASS", "gold_label": "PASS", "raw_invalid": False},
            {"id": "ood_2", "label": "FAIL", "gold_label": "PASS", "raw_invalid": True},
        ]

        # 1. Success execution with tenant context and OOD challenge
        tc = TenantContext(tenant_id="tenant-task", request_id="req-task-1")
        record1 = create_task_run(
            task_spec=task_spec,
            dataset_records=dataset,
            baseline_predictions=test_preds,
            model_predictions=test_preds,
            output_root=self.runs_dir,
            ood_challenge_predictions=ood_preds,
            tenant_context=tc,
        )
        self.assertTrue(record1["run_id"].startswith("run-"))
        self.assertEqual(record1["tenant_id"], "tenant-task")
        self.assertEqual(record1["evaluation"]["task"]["ood_challenge"]["count"], 2)
        self.assertEqual(record1["provenance"]["runtime"], "offline_cpu")

        # Inspect task run
        ins = inspect_run(record1["run_id"], output_root=self.runs_dir, tenant_context=tc)
        self.assertEqual(ins["provenance_status"], "PROVENANCE_VERIFIED")

        # Run 2 for task run diffing (testing model_metrics diffing in diff_runs)
        record2 = create_task_run(
            task_spec=task_spec,
            dataset_records=dataset,
            baseline_predictions=test_preds,
            model_predictions=test_preds,
            output_root=self.runs_dir,
            tenant_context=tc,
        )
        diff_task = diff_runs(record1["run_id"], record2["run_id"], output_root=self.runs_dir, tenant_context=tc)
        self.assertIn("accuracy_delta", diff_task["metrics_delta"])
        self.assertIn("macro_f1_delta", diff_task["metrics_delta"])

        # 2. Invalid tenant context type
        with self.assertRaises(ValueError):
            create_task_run(
                task_spec=task_spec,
                dataset_records=dataset,
                baseline_predictions=test_preds,
                model_predictions=test_preds,
                tenant_context="bad-type",  # type: ignore[arg-type]
            )

        # 3. Invalid task spec
        with self.assertRaises(ValueError):
            create_task_run(
                task_spec={},
                dataset_records=dataset,
                baseline_predictions=test_preds,
                model_predictions=test_preds,
            )

        # 4. Invalid dataset
        with self.assertRaises(ValueError):
            create_task_run(
                task_spec=task_spec,
                dataset_records=[{"id": "r1"}],
                baseline_predictions=test_preds,
                model_predictions=test_preds,
            )

        # 5. Preflight aborted due to model manifest adapter mismatch
        mismatched_manifest = {
            "adapter_sha256": "0" * 64,
            "base_model_id": "test/model",
            "revision": "main",
            "adapter_name": "adapter-v1",
        }
        with self.assertRaises(ValueError) as ctx_preflight:
            create_task_run(
                task_spec=task_spec,
                dataset_records=dataset,
                baseline_predictions=test_preds,
                model_predictions=test_preds,
                model_manifest=mismatched_manifest,
            )
        self.assertIn("preflight failed", str(ctx_preflight.exception))

        # 6. Invalid OOD predictions
        with self.assertRaises(ValueError):
            create_task_run(
                task_spec=task_spec,
                dataset_records=dataset,
                baseline_predictions=test_preds,
                model_predictions=test_preds,
                ood_challenge_predictions=[],
            )
        with self.assertRaises(ValueError):
            create_task_run(
                task_spec=task_spec,
                dataset_records=dataset,
                baseline_predictions=test_preds,
                model_predictions=test_preds,
                ood_challenge_predictions=[{"id": 123}],  # type: ignore[list-item]
            )


if __name__ == "__main__":
    unittest.main()
