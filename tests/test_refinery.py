"""Offline tests for the developer-facing Refinery workflow."""
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hardonian_foundry.artifact import add_artifact, list_artifacts, verify_artifact
from hardonian_foundry.baseline import dataset as default_dataset
from hardonian_foundry.refinery import (
    _code_digest,
    anchor_run,
    attach_license,
    create_run,
    diff_runs,
    evaluate_dataset,
    evaluate_promotion,
    export_bundle,
    inspect_run,
    list_runs,
    run_preflight,
    sign_run,
    validate_dataset,
    validate_run_id,
    verify_bundle,
)
from hardonian_foundry.tenancy import TenantContext


class RefineryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output_root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_validate_dataset_valid_default(self):
        data = default_dataset()
        res = validate_dataset(data)
        self.assertEqual(res["status"], "VALID")
        self.assertEqual(res["record_count"], 60)
        self.assertEqual(res["split_counts"], {"train": 36, "validation": 12, "test": 12})
        self.assertTrue(isinstance(res["dataset_sha256"], str))
        self.assertEqual(len(res["dataset_sha256"]), 64)
        self.assertEqual(res["errors"], [])

    def test_validate_dataset_rejects_non_list_or_too_small(self):
        self.assertEqual(validate_dataset("not-a-list")["status"], "INVALID")
        self.assertEqual(validate_dataset([])["status"], "INVALID")
        self.assertEqual(validate_dataset(default_dataset()[:2])["status"], "INVALID")

    def test_validate_dataset_rejects_duplicate_id(self):
        data = default_dataset()[:10]
        data.append({**data[0], "expected_minor": 99999})
        res = validate_dataset(data)
        self.assertEqual(res["status"], "INVALID")
        self.assertTrue(any("duplicate record id" in err for err in res["errors"]))

    def test_validate_dataset_rejects_duplicate_content_leakage(self):
        data = default_dataset()[:10]
        data.append({**data[0], "id": "synthetic-different-id"})
        res = validate_dataset(data)
        self.assertEqual(res["status"], "INVALID")
        self.assertTrue(any("duplicate record content" in err for err in res["errors"]))

    def test_validate_dataset_rejects_non_synthetic_id_or_invalid_fields(self):
        bad_id = [{"id": "real-123", "expected_minor": 100, "actual_minor": 100, "critical": False}]
        self.assertEqual(validate_dataset(bad_id)["status"], "INVALID")

        bad_amount = [{"id": "synthetic-1", "expected_minor": "100", "actual_minor": 100, "critical": False}]
        self.assertEqual(validate_dataset(bad_amount)["status"], "INVALID")

    def test_run_preflight_success_default(self):
        report = run_preflight(output_root=self.output_root)
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(report["errors"], [])
        check_names = {c["name"] for c in report["checks"]}
        self.assertIn("dataset_validation", check_names)
        self.assertIn("runtime_environment", check_names)
        self.assertIn("output_directory", check_names)
        self.assertIn("licensing_and_provenance", check_names)
        self.assertIn("execution_safety", check_names)

    def test_run_preflight_with_valid_model_manifest(self):
        manifest = {
            "name": "synthetic-qwen-adapter",
            "license_id": "Apache-2.0",
            "provenance": "synthetic",
            "data_origin": "synthetic",
        }
        report = run_preflight(output_root=self.output_root, model_manifest=manifest)
        self.assertEqual(report["status"], "PASSED")

    def test_run_preflight_fails_with_unapproved_license_or_data_origin(self):
        unapproved = {
            "name": "proprietary-model",
            "license_id": "Proprietary",
            "provenance": "synthetic",
            "data_origin": "synthetic",
        }
        report = run_preflight(output_root=self.output_root, model_manifest=unapproved)
        self.assertEqual(report["status"], "FAILED")
        self.assertTrue(any("licensing/provenance" in err for err in report["errors"]))

    def test_evaluate_dataset_reproducible(self):
        eval1 = evaluate_dataset()
        eval2 = evaluate_dataset()
        self.assertEqual(eval1, eval2)
        self.assertEqual(eval1["data_origin"], "synthetic")
        self.assertEqual(eval1["review_rate"], 1.0)
        self.assertEqual(eval1["coverage_automatic_decision"], 0.0)

    def test_evaluate_dataset_fails_on_invalid_data(self):
        with self.assertRaises(ValueError):
            evaluate_dataset([{"id": "synthetic-1"}])

    def test_create_and_inspect_run_provenance(self):
        manifest = {
            "name": "synthetic-test-model",
            "license_id": "MIT",
            "provenance": "synthetic",
            "data_origin": "synthetic",
        }
        run = create_run(output_root=self.output_root, model_manifest=manifest)
        run_id = run["run_id"]
        self.assertTrue(run_id.startswith("run-"))
        self.assertEqual(run["status"], "COMPLETED")

        # Inspect run
        inspection = inspect_run(run_id, output_root=self.output_root)
        self.assertEqual(inspection["provenance_status"], "PROVENANCE_VERIFIED")
        self.assertEqual(inspection["preflight_status"], "PASSED")
        self.assertEqual(inspection["evaluation"]["test_count"], 12)
        self.assertIsNotNone(inspection["matched_entry"])
        self.assertEqual(inspection["matched_entry"]["key"], run_id)

    def test_inspect_run_detects_tampered_evidence(self):
        run = create_run(output_root=self.output_root)
        run_id = run["run_id"]

        ledger_path = os.path.join(self.output_root, "evidence.jsonl")
        with open(ledger_path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()

        entry = json.loads(lines[0])
        entry["record"]["f1"] = 0.0  # tamper record without updating hash
        lines[0] = json.dumps(entry, sort_keys=True)
        with open(ledger_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

        inspection = inspect_run(run_id, output_root=self.output_root)
        self.assertNotEqual(inspection["provenance_status"], "PROVENANCE_VERIFIED")

    def test_inspect_run_detects_tampered_run_json_metadata(self):
        run = create_run(output_root=self.output_root)
        run_id = run["run_id"]

        # Tamper run.json while leaving evidence ledger intact
        run_file = os.path.join(self.output_root, run_id, "run.json")
        with open(run_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        data["evaluation"]["f1"] = 0.5
        with open(run_file, "w", encoding="utf-8") as handle:
            json.dump(data, handle)

        inspection = inspect_run(run_id, output_root=self.output_root)
        self.assertEqual(inspection["provenance_status"], "TAMPERED_METADATA_MISMATCH")

    def test_evaluate_promotion_blocks_degenerate_uncovered_benchmarks(self):
        # Distinct matching cases with no mismatches (tp+fp=0 -> f1 is None) and no critical (cfn is None)
        matching_data = [
            {"id": f"synthetic-{i:03d}", "expected_minor": 100 + i * 10, "actual_minor": 100 + i * 10, "critical": False}
            for i in range(10)
        ]
        manifest = {
            "name": "synthetic-model",
            "license_id": "Apache-2.0",
            "provenance": "synthetic",
            "data_origin": "synthetic",
        }
        run = create_run(output_root=self.output_root, dataset=matching_data, model_manifest=manifest)
        run_id = run["run_id"]

        promo = evaluate_promotion(run_id, output_root=self.output_root)
        self.assertFalse(promo["promoted"])
        self.assertEqual(promo["status"], "BLOCKED")
        self.assertIn("critical_case_coverage", promo["missing_evidence"])
        self.assertIn("positive_case_coverage", promo["missing_evidence"])

    def test_evaluate_promotion_blocked_missing_evidence(self):
        # Create run without model manifest
        run = create_run(output_root=self.output_root)
        run_id = run["run_id"]

        promo = evaluate_promotion(run_id, output_root=self.output_root)
        self.assertFalse(promo["promoted"])
        self.assertEqual(promo["status"], "BLOCKED")
        self.assertIn("model_licensing_manifest", promo["missing_evidence"])
        self.assertIn("cryptographic_promotion_signature", promo["missing_evidence"])

    def test_evaluate_promotion_blocked_with_valid_manifest_still_fails_closed(self):
        manifest = {
            "name": "synthetic-model",
            "license_id": "Apache-2.0",
            "provenance": "synthetic",
            "data_origin": "synthetic",
        }
        run = create_run(output_root=self.output_root, model_manifest=manifest)
        run_id = run["run_id"]

        promo = evaluate_promotion(run_id, output_root=self.output_root)
        self.assertFalse(promo["promoted"])
        self.assertEqual(promo["status"], "BLOCKED")
        # Manifest is present, but signed promotion mechanism is still missing
        self.assertNotIn("model_licensing_manifest", promo["missing_evidence"])
        self.assertIn("cryptographic_promotion_signature", promo["missing_evidence"])

    def test_evaluate_promotion_nonexistent_run(self):
        promo = evaluate_promotion("run-nonexistent", output_root=self.output_root)
        self.assertFalse(promo["promoted"])
        self.assertEqual(promo["status"], "BLOCKED")
        self.assertIn("run_metadata", promo["missing_evidence"])

    def test_cli_validate_dataset_command(self):
        import subprocess
        import sys
        env = {**os.environ, "PYTHONPATH": "src"}
        res = subprocess.run(
            [sys.executable, "-m", "hardonian_foundry.cli", "refinery", "validate-dataset"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        self.assertEqual(res.returncode, 0)
        out = json.loads(res.stdout)
        self.assertEqual(out["status"], "VALID")
        self.assertEqual(out["record_count"], 60)

    def test_cli_preflight_command(self):
        import subprocess
        import sys
        env = {**os.environ, "PYTHONPATH": "src"}
        res = subprocess.run(
            [sys.executable, "-m", "hardonian_foundry.cli", "refinery", "preflight", "--output-root", self.output_root],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        self.assertEqual(res.returncode, 0)
        out = json.loads(res.stdout)
        self.assertEqual(out["status"], "PASSED")

    def test_cli_run_inspect_promote_flow(self):
        import subprocess
        import sys
        env = {**os.environ, "PYTHONPATH": "src"}

        # 1. Run vertical slice
        res_run = subprocess.run(
            [sys.executable, "-m", "hardonian_foundry.cli", "refinery", "run", "--output-root", self.output_root],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        self.assertEqual(res_run.returncode, 0)
        run_data = json.loads(res_run.stdout)
        run_id = run_data["run_id"]
        self.assertTrue(run_id.startswith("run-"))

        # 2. Inspect run & verify provenance
        res_inspect = subprocess.run(
            [sys.executable, "-m", "hardonian_foundry.cli", "refinery", "inspect", run_id, "--output-root", self.output_root],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        self.assertEqual(res_inspect.returncode, 0)
        inspect_data = json.loads(res_inspect.stdout)
        self.assertEqual(inspect_data["provenance_status"], "PROVENANCE_VERIFIED")

        # 3. Promote (must fail closed with code 2)
        res_promote = subprocess.run(
            [sys.executable, "-m", "hardonian_foundry.cli", "refinery", "promote", run_id, "--output-root", self.output_root],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        self.assertEqual(res_promote.returncode, 2)
        promo_data = json.loads(res_promote.stdout)
        self.assertFalse(promo_data["promoted"])
        self.assertEqual(promo_data["status"], "BLOCKED")
        self.assertIn("Promotion blocked", res_promote.stderr)

    def test_cli_registry_promote_blocked(self):
        import subprocess
        import sys
        env = {**os.environ, "PYTHONPATH": "src"}
        res = subprocess.run(
            [sys.executable, "-m", "hardonian_foundry.cli", "registry", "promote"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        self.assertEqual(res.returncode, 2)
        self.assertIn("Promotion blocked", res.stderr)

    def test_run_id_path_traversal_rejected(self):
        bad_ids = [
            "",
            "../other",
            "..\\other",
            "a/b",
            "a\\b",
            "C:run-1",
            "run-1.",
            "..",
        ]
        for bad in bad_ids:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_run_id(bad)
                with self.assertRaises(ValueError):
                    inspect_run(bad, output_root=self.output_root)
                with self.assertRaises(ValueError):
                    evaluate_promotion(bad, output_root=self.output_root)
                with self.assertRaises(ValueError):
                    attach_license(bad, manifest={}, output_root=self.output_root)
                with self.assertRaises(ValueError):
                    sign_run(bad, private_key_path="missing.pem", output_root=self.output_root)

    def test_portable_ledger_path_and_outside_escape_rejected(self):
        manifest = {
            "name": "synthetic-reconciler",
            "license_id": "Apache-2.0",
            "provenance": "synthetic",
            "data_origin": "synthetic",
        }
        run = create_run(output_root=self.output_root, model_manifest=manifest)
        self.assertEqual(run["provenance"]["evidence_ledger_path"], "evidence.jsonl")
        run_id = run["run_id"]
        run_json = Path(self.output_root) / run_id / "run.json"

        for escaped in ("../evidence.jsonl", "..\\evidence.jsonl", "/tmp/other-ledger.jsonl"):
            data = json.loads(run_json.read_text(encoding="utf-8"))
            data["provenance"]["evidence_ledger_path"] = escaped
            run_json.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                inspect_run(run_id, output_root=self.output_root)

        # Even tampering non-hash provenance field invalidates seq-1 run_record_sha256
        data = json.loads(run_json.read_text(encoding="utf-8"))
        data["provenance"]["evidence_ledger_path"] = "evidence.jsonl"
        data["split_seed"] = 999
        run_json.write_text(json.dumps(data), encoding="utf-8")
        insp = inspect_run(run_id, output_root=self.output_root)
        self.assertEqual(insp["provenance_status"], "TAMPERED_METADATA_MISMATCH")

    def test_concurrent_create_run_unique_ids(self):
        data = default_dataset()
        with ThreadPoolExecutor(max_workers=4) as pool:
            runs = list(pool.map(lambda _: create_run(output_root=self.output_root, dataset=data), range(6)))
        run_ids = [r["run_id"] for r in runs]
        self.assertEqual(len(set(run_ids)), 6)
        for rid in run_ids:
            insp_r = inspect_run(rid, output_root=self.output_root)
            self.assertEqual(insp_r["provenance_status"], "PROVENANCE_VERIFIED")

    def test_evaluate_promotion_consistent_early_return_schema(self):
        missing = evaluate_promotion("nonexistent-run", output_root=self.output_root)
        expected_keys = {
            "promoted",
            "status",
            "run_id",
            "reasons",
            "missing_evidence",
            "evidence_summary",
            "review_required",
            "note",
        }
        self.assertEqual(set(missing.keys()), expected_keys)
        self.assertFalse(missing["promoted"])
        self.assertTrue(missing["review_required"])
        self.assertTrue(missing["note"])

        run = create_run(output_root=self.output_root)
        ledger = Path(self.output_root) / "evidence.jsonl"
        entry = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
        entry["record"]["f1"] = 0.0
        ledger.write_text(json.dumps(entry) + "\n", encoding="utf-8")
        tampered = evaluate_promotion(run["run_id"], output_root=self.output_root)
        self.assertEqual(set(tampered.keys()), expected_keys)
        self.assertTrue(tampered["review_required"])

    def test_code_digest_covers_package_modules(self):
        digest_a = _code_digest()
        digest_b = _code_digest()
        self.assertEqual(digest_a, digest_b)
        self.assertEqual(len(digest_a), 64)

    def test_anchor_run_detects_truncation_and_rollback(self):
        run1 = create_run(output_root=self.output_root)
        anchor1 = anchor_run(run1["run_id"], output_root=self.output_root)
        run2 = create_run(output_root=self.output_root)
        anchor_file = Path(self.output_root) / "anchor.json"
        anchor2 = anchor_run(run2["run_id"], output_root=self.output_root, anchor_path=anchor_file)
        self.assertEqual(anchor2["count"], 2)
        self.assertTrue(anchor_file.is_file())

        insp_bad = inspect_run(
            run2["run_id"],
            output_root=self.output_root,
            expected_head=anchor1["head"],
            expected_count=anchor2["count"],
        )
        self.assertEqual(insp_bad["provenance_status"], "TAMPERED_ANCHOR_MISMATCH")

        promo_bad = evaluate_promotion(
            run2["run_id"],
            output_root=self.output_root,
            expected_head=anchor1["head"],
            expected_count=anchor1["count"],
        )
        self.assertFalse(promo_bad["promoted"])
        self.assertIn("anchor_checkpoint", promo_bad["missing_evidence"])

    def test_list_and_diff_runs(self):
        r1 = create_run(output_root=self.output_root, seed=11)
        r2 = create_run(output_root=self.output_root, seed=22)
        listed = list_runs(output_root=self.output_root, verify_provenance=True)
        self.assertEqual({r["run_id"] for r in listed}, {r1["run_id"], r2["run_id"]})
        self.assertTrue(all(r["provenance_status"] == "PROVENANCE_VERIFIED" for r in listed))
        diff = diff_runs(r1["run_id"], r2["run_id"], output_root=self.output_root)
        self.assertTrue(diff["same_dataset"])
        self.assertFalse(diff["same_test_split"])
        self.assertEqual(diff["provenance_a"], "PROVENANCE_VERIFIED")
        self.assertEqual(diff["provenance_b"], "PROVENANCE_VERIFIED")

    def test_export_and_verify_bundle_formats(self):
        run = create_run(output_root=self.output_root)
        rid = run["run_id"]
        art_src = Path(self.output_root) / "weights.bin"
        art_src.write_bytes(b"synthetic-adapter-weights-v1")
        add_artifact(art_src, output_root=self.output_root, name="weights.bin", run_id=rid)

        for fmt, suffix in (("dir", "bundle-dir"), ("zip", "bundle.zip"), ("tar.gz", "bundle.tar.gz")):
            dest = Path(self.output_root) / suffix
            exported = export_bundle(rid, output_root=self.output_root, dest_path=dest, fmt=fmt)
            self.assertEqual(exported["status"], "EXPORTED")
            verified = verify_bundle(dest)
            self.assertEqual(verified["status"], "VERIFIED")
            self.assertEqual(verified["run_id"], rid)

        # Corrupting exported bundle dir fails verification closed
        corrupt_run = Path(self.output_root) / "bundle-dir" / "run.json"
        corrupt_run.write_text(corrupt_run.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_bundle(Path(self.output_root) / "bundle-dir")

    def test_artifact_store_add_verify_and_tamper_detection(self):
        src = Path(self.output_root) / "model.safetensors"
        src.write_bytes(b"mock-safetensors-payload")
        entry = add_artifact(src, output_root=self.output_root, name="model.safetensors", kind="adapter")
        ok_by_hash = verify_artifact(entry["sha256"], output_root=self.output_root)
        self.assertTrue(ok_by_hash["verified"])
        ok_by_file = verify_artifact(src, expected_sha256=entry["sha256"], output_root=self.output_root)
        self.assertTrue(ok_by_file["verified"])
        self.assertEqual(len(list_artifacts(output_root=self.output_root)), 1)

        # Corrupt stored blob. It is sealed read-only (0o444); chmod writable
        # first so the tamper simulation works for non-root (CI runs as root).
        blob = Path(entry["blob_path"])
        blob.chmod(0o644)
        blob.write_bytes(b"corrupted-payload")
        bad = verify_artifact(entry["sha256"], output_root=self.output_root)
        self.assertFalse(bad["verified"])

    def test_tenant_context_isolation(self):
        ctx_a = TenantContext(tenant_id="tenant-a", request_id="req-a-1")
        ctx_a2 = TenantContext(tenant_id="tenant-a", request_id="req-a-2")
        ctx_b = TenantContext(tenant_id="tenant-b", request_id="req-b-1")
        run_a = create_run(output_root=self.output_root, tenant_context=ctx_a)
        rid = run_a["run_id"]

        insp_a = inspect_run(rid, output_root=self.output_root, tenant_context=ctx_a2)
        self.assertEqual(insp_a["provenance_status"], "PROVENANCE_VERIFIED")
        self.assertEqual(len(list_runs(output_root=self.output_root, tenant_context=ctx_a2)), 1)
        self.assertEqual(len(list_runs(output_root=self.output_root, tenant_context=ctx_b)), 0)
        with self.assertRaises(PermissionError):
            inspect_run(rid, output_root=self.output_root, tenant_context=ctx_b)
        promo_b = evaluate_promotion(rid, output_root=self.output_root, tenant_context=ctx_b)
        self.assertFalse(promo_b["promoted"])
        self.assertIn("tenant_authorization", promo_b["missing_evidence"])


if __name__ == "__main__":
    unittest.main()
