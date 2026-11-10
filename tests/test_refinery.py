"""Offline tests for the developer-facing Refinery workflow."""
import json
import os
import tempfile
import unittest

from hardonian_foundry.baseline import dataset as default_dataset
from hardonian_foundry.refinery import (
    create_run,
    evaluate_dataset,
    evaluate_promotion,
    inspect_run,
    run_preflight,
    validate_dataset,
)


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


if __name__ == "__main__":
    unittest.main()
