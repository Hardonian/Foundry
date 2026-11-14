"""End-to-end tests for Hardonian Foundry v1.0.0 productization deliverables."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import hardonian_foundry as hf
from hardonian_foundry import __main__ as hf_main
from hardonian_foundry.cli import main as cli_main
from hardonian_foundry.refinery import (
    anchor_run,
    attach_license,
    create_run,
    export_bundle,
    generate_audit_report,
    sign_run,
)
from hardonian_foundry.registry import (
    get_model,
    list_models,
    register_model,
    resolve_registry_path,
    verify_registry,
)
from hardonian_foundry.signing import generate_keypair
from hardonian_foundry.task_eval import compare_task_predictions, split_task_dataset
from hardonian_foundry.tenancy import DENY_REASON, TenantContext

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"

_VALID_MANIFEST: dict[str, Any] = {
    "name": "synthetic-exception-classifier",
    "license_id": "Apache-2.0",
    "provenance": "synthetic",
    "data_origin": "synthetic",
}


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_SRC_DIR}{os.pathsep}{existing}" if existing else str(_SRC_DIR)
    return env


def _build_task_fixture(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path, Path, Path]:
    records: list[dict[str, Any]] = []
    for fam_idx in range(4):
        fam = f"fam-{fam_idx}"
        for item_idx in range(4):
            label = "EXC_MATCH" if item_idx % 2 == 0 else "EXC_MISMATCH"
            rec_id = f"synthetic-cmp-{fam_idx}-{item_idx}"
            records.append(
                {
                    "id": rec_id,
                    "prompt": f"Exception case {fam_idx}-{item_idx} amount 100{fam_idx}{item_idx} expected {label}",
                    "gold_label": label,
                    "critical": item_idx == 1,
                    "family": fam,
                }
            )

    spec: dict[str, Any] = {
        "schema_version": "1.0",
        "task_id": "exception-compare-test",
        "description": "Synthetic family-disjoint classification comparison fixture",
        "labels": ["EXC_MATCH", "EXC_MISMATCH"],
        "thresholds": {
            "macro_f1_min": 0.8,
            "critical_fn_rate_max": 0.0,
        },
        "baseline_must_improve": True,
        "split_seed": 42,
        "model": {
            "base_model_id": "synthetic-base",
            "revision": "rev-1",
            "adapter_sha256": "a" * 64,
            "adapter_name": "synthetic-adapter",
        },
        "dataset_path": "dataset.jsonl",
    }

    splits = split_task_dataset(records, spec)
    test_rows = splits["test"]

    baseline_preds: list[dict[str, Any]] = []
    model_preds: list[dict[str, Any]] = []
    flipped = False
    for row in test_rows:
        gold = str(row["gold_label"])
        other = "EXC_MISMATCH" if gold == "EXC_MATCH" else "EXC_MATCH"
        model_preds.append({"id": row["id"], "label": gold, "confidence": 0.95})
        if not flipped:
            baseline_preds.append({"id": row["id"], "label": other, "confidence": 0.55})
            flipped = True
        else:
            baseline_preds.append({"id": row["id"], "label": gold, "confidence": 0.75})

    spec_path = root / "spec.json"
    dataset_path = root / "dataset.jsonl"
    baseline_path = root / "baseline.jsonl"
    model_path = root / "model.jsonl"

    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    dataset_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    baseline_path.write_text("\n".join(json.dumps(p) for p in baseline_preds) + "\n", encoding="utf-8")
    model_path.write_text("\n".join(json.dumps(p) for p in model_preds) + "\n", encoding="utf-8")
    return spec, records, spec_path, baseline_path, model_path


class TestSdkPackagingAndEntrypoint(unittest.TestCase):
    def test_top_level_exports_and_py_typed(self) -> None:
        self.assertEqual(hf.__version__, "1.0.0")
        pkg_dir = Path(hf.__file__).resolve().parent
        self.assertTrue((pkg_dir / "py.typed").is_file())
        for symbol in hf.__all__:
            self.assertTrue(hasattr(hf, symbol), f"missing exported symbol {symbol}")
        for expected in (
            "register_model",
            "list_models",
            "get_model",
            "verify_registry",
            "compare_task_predictions",
            "generate_audit_report",
            "evaluate_promotion",
            "__version__",
        ):
            self.assertIn(expected, hf.__all__)

    def test_top_level_import_does_not_load_torch(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, hardonian_foundry; assert 'torch' not in sys.modules; print(hardonian_foundry.__version__)",
            ],
            env=_subprocess_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "1.0.0")

    def test_python_module_entrypoint(self) -> None:
        with self.assertRaises(SystemExit) as sys_exc:
            hf_main.run(["--version"])
        self.assertEqual(sys_exc.exception.code, 0)
        self.assertEqual(hf_main.run(["doctor"]), 0)
        proc = subprocess.run(
            [sys.executable, "-m", "hardonian_foundry", "--version"],
            env=_subprocess_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("1.0.0", proc.stdout)


class TestProductizationRegistryReportBundleCompare(unittest.TestCase):
    def _create_signed_run(
        self,
        tmp_path: Path,
        *,
        tenant_context: TenantContext | None = None,
    ) -> tuple[Path, str, list[str], Path]:
        runs_dir = tmp_path / "runs"
        key_path = tmp_path / "signing.pem"
        kp = generate_keypair(key_path)
        keyring = [kp["public_key_hex"]]
        keyring_path = tmp_path / "keyring.json"
        keyring_path.write_text(json.dumps({"trusted_keys": keyring}), encoding="utf-8")

        rec = create_run(output_root=runs_dir, tenant_context=tenant_context)
        run_id = rec["run_id"]
        attach_license(run_id, manifest=dict(_VALID_MANIFEST), output_root=runs_dir)
        sign_run(run_id, private_key_path=key_path, output_root=runs_dir)
        return runs_dir, run_id, keyring, keyring_path

    def test_registry_lifecycle_tamper_detection_and_tenant_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tenant_a = TenantContext(tenant_id="tenant-alpha", request_id="req-1")
            tenant_b = TenantContext(tenant_id="tenant-beta", request_id="req-2")

            with self.assertRaises(ValueError):
                resolve_registry_path(registry_path="../escape/registry.json")
            with self.assertRaises(ValueError):
                resolve_registry_path(output_root="bad\x00path")
            with self.assertRaises(ValueError):
                resolve_registry_path(output_root="../bad-root")

            runs_dir, run_id, keyring, keyring_path = self._create_signed_run(tmp_path, tenant_context=tenant_a)

            empty = list_models(output_root=runs_dir, tenant_context=tenant_a)
            self.assertEqual(empty["registry_status"], "EMPTY")
            self.assertEqual(empty["count"], 0)

            blocked = register_model(run_id, output_root=runs_dir, tenant_context=tenant_a)
            self.assertFalse(blocked["promoted"])
            self.assertFalse(blocked["registered"])
            self.assertEqual(blocked["status"], "BLOCKED")

            reg = register_model(
                run_id,
                model_id="exc-classifier-v1",
                output_root=runs_dir,
                keyring=keyring,
                tenant_context=tenant_a,
                notes="Verified release candidate",
            )
            self.assertTrue(reg["registered"])
            self.assertEqual(reg["registry_entry"]["status"], "REGISTERED_REVIEW_REQUIRED")
            self.assertTrue(reg["registry_entry"]["review_required"])

            listed = list_models(output_root=runs_dir, tenant_context=tenant_a)
            self.assertEqual(listed["registry_status"], "VERIFIED")
            self.assertEqual(listed["count"], 1)
            self.assertEqual(list_models(output_root=runs_dir, tenant_context=tenant_b)["count"], 0)

            by_model = get_model("exc-classifier-v1", output_root=runs_dir, tenant_context=tenant_a)
            by_run = get_model(run_id, output_root=runs_dir, tenant_context=tenant_a)
            self.assertEqual(by_model["run_id"], by_run["run_id"])

            with self.assertRaises(KeyError):
                get_model("missing-model", output_root=runs_dir, tenant_context=tenant_a)
            with self.assertRaises(PermissionError) as perm_exc:
                get_model("exc-classifier-v1", output_root=runs_dir, tenant_context=tenant_b)
            self.assertIn(DENY_REASON, str(perm_exc.exception))

            ver = verify_registry(output_root=runs_dir, keyring=keyring, tenant_context=tenant_a)
            self.assertEqual(ver["status"], "VERIFIED")
            self.assertTrue(ver["verified"])
            self.assertEqual(ver["model_count"], 1)

            # Tamper registry.json directly and ensure fail-closed detection
            reg_file = Path(reg["registry_path"])
            tampered = json.loads(reg_file.read_text(encoding="utf-8"))
            tampered["models"][0]["notes"] = "tampered-notes"
            reg_file.write_text(json.dumps(tampered), encoding="utf-8")
            tampered_ver = verify_registry(output_root=runs_dir, keyring=keyring, tenant_context=tenant_a)
            self.assertEqual(tampered_ver["status"], "TAMPERED")
            self.assertFalse(tampered_ver["verified"])

            # Restore registry via re-registration, then tamper backing run.json
            reg_file.unlink()
            register_model(
                run_id,
                model_id="exc-classifier-v1",
                output_root=runs_dir,
                keyring=keyring,
                tenant_context=tenant_a,
            )
            run_json = runs_dir / run_id / "run.json"
            run_data = json.loads(run_json.read_text(encoding="utf-8"))
            run_data["dataset_sha256"] = "0" * 64
            run_json.write_text(json.dumps(run_data), encoding="utf-8")
            run_tampered = verify_registry(output_root=runs_dir, keyring=keyring, tenant_context=tenant_a)
            self.assertEqual(run_tampered["status"], "TAMPERED")
            self.assertFalse(run_tampered["verified"])

            # Verify CLI registry commands end-to-end cleanly on a fresh run
            runs2, run2_id, _kr2, kr2_path = self._create_signed_run(tmp_path / "cli_case")
            custom_reg = tmp_path / "cli_case" / "custom-registry.json"
            rc_reg = cli_main(
                [
                    "registry",
                    "register",
                    run2_id,
                    "--output-root",
                    str(runs2),
                    "--registry-path",
                    str(custom_reg),
                    "--keyring",
                    str(kr2_path),
                    "--name",
                    "cli-model-v1",
                    "--notes",
                    "CLI registered",
                ]
            )
            self.assertEqual(rc_reg, 0)
            self.assertEqual(
                cli_main(
                    [
                        "registry",
                        "get",
                        "cli-model-v1",
                        "--output-root",
                        str(runs2),
                        "--registry-path",
                        str(custom_reg),
                    ]
                ),
                0,
            )
            self.assertEqual(
                cli_main(
                    [
                        "registry",
                        "verify",
                        "--output-root",
                        str(runs2),
                        "--registry-path",
                        str(custom_reg),
                        "--keyring",
                        str(kr2_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                cli_main(
                    [
                        "registry",
                        "promote",
                        run2_id,
                        "--output-root",
                        str(runs2),
                        "--registry-path",
                        str(custom_reg),
                        "--keyring",
                        str(kr2_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                cli_main(
                    [
                        "registry",
                        "get",
                        "nonexistent-model",
                        "--output-root",
                        str(runs2),
                        "--registry-path",
                        str(custom_reg),
                    ]
                ),
                2,
            )

    def test_audit_report_and_verify_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runs_dir, run_id, keyring, keyring_path = self._create_signed_run(tmp_path)
            anchor = anchor_run(run_id, output_root=runs_dir, anchor_path=tmp_path / "anchor.json")

            rep_blocked = generate_audit_report(run_id, output_root=runs_dir)
            self.assertFalse(rep_blocked["promoted"])
            self.assertTrue(rep_blocked["review_required"])
            self.assertIn("BLOCKED", rep_blocked["markdown"])

            md_out = tmp_path / "audit.md"
            rep_ok = generate_audit_report(
                run_id,
                output_root=runs_dir,
                keyring=keyring,
                expected_head=anchor["head"],
                expected_count=anchor["count"],
                fmt="markdown",
                out_path=md_out,
            )
            self.assertTrue(rep_ok["promoted"])
            self.assertTrue(rep_ok["anchor_verified"])
            self.assertTrue(md_out.is_file())
            self.assertIn("ALLOWED_REVIEW_REQUIRED", md_out.read_text(encoding="utf-8"))

            with self.assertRaises(ValueError):
                generate_audit_report(run_id, output_root=runs_dir, fmt="html")
            with self.assertRaises(ValueError):
                generate_audit_report(run_id, output_root=runs_dir, out_path="../escape.md")

            # CLI refinery report
            json_out = tmp_path / "audit.json"
            self.assertEqual(
                cli_main(
                    [
                        "refinery",
                        "report",
                        run_id,
                        "--output-root",
                        str(runs_dir),
                        "--keyring",
                        str(keyring_path),
                        "--anchor",
                        str(tmp_path / "anchor.json"),
                        "--format",
                        "json",
                        "--out",
                        str(json_out),
                    ]
                ),
                0,
            )
            self.assertTrue(json.loads(json_out.read_text(encoding="utf-8"))["promoted"])
            self.assertEqual(
                cli_main(
                    [
                        "refinery",
                        "report",
                        run_id,
                        "--output-root",
                        str(runs_dir),
                        "--keyring",
                        str(keyring_path),
                        "--format",
                        "markdown",
                    ]
                ),
                0,
            )

            # Export and verify directory + zip bundles, then test tamper exit code 2
            bundle_dir = tmp_path / "bundle_dir"
            bundle_zip = tmp_path / "bundle.zip"
            export_bundle(run_id, output_root=runs_dir, dest_path=bundle_dir, fmt="dir")
            export_bundle(run_id, output_root=runs_dir, dest_path=bundle_zip, fmt="zip")

            self.assertEqual(cli_main(["refinery", "verify-bundle", str(bundle_dir)]), 0)
            self.assertEqual(cli_main(["refinery", "verify-bundle", str(bundle_zip)]), 0)

            (bundle_dir / "run.json").write_text('{"tampered": true}\n', encoding="utf-8")
            self.assertEqual(cli_main(["refinery", "verify-bundle", str(bundle_dir)]), 2)

    def test_compare_task_predictions_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spec, records, spec_path, base_path, mod_path = _build_task_fixture(tmp_path)
            base_preds = [json.loads(ln) for ln in base_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            mod_preds = [json.loads(ln) for ln in mod_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

            res = compare_task_predictions(spec, records, base_preds, mod_preds, n_bootstrap=20)
            self.assertTrue(res["review_required"])
            self.assertTrue(res["improved_macro_f1"])
            self.assertGreater(res["macro_f1_delta"], 0.0)
            self.assertGreater(res["accuracy_delta"], 0.0)
            self.assertIn("p_value", res["mcnemar"])
            self.assertEqual(res["baseline_ci"]["n_bootstrap"], 20)
            self.assertEqual(res["model_ci"]["n_bootstrap"], 20)

            with self.assertRaises(ValueError):
                compare_task_predictions({"schema_version": "0.1"}, records, base_preds, mod_preds)

            rc = cli_main(
                [
                    "refinery",
                    "compare",
                    "--task-spec",
                    str(spec_path),
                    "--baseline-predictions",
                    str(base_path),
                    "--model-predictions",
                    str(mod_path),
                    "--n-bootstrap",
                    "20",
                ]
            )
            self.assertEqual(rc, 0)

            bad_spec_path = tmp_path / "bad_spec.json"
            bad_spec = dict(spec)
            del bad_spec["dataset_path"]
            bad_spec_path.write_text(json.dumps(bad_spec), encoding="utf-8")
            self.assertEqual(
                cli_main(
                    [
                        "refinery",
                        "compare",
                        "--task-spec",
                        str(bad_spec_path),
                        "--baseline-predictions",
                        str(base_path),
                        "--model-predictions",
                        str(mod_path),
                    ]
                ),
                2,
            )


if __name__ == "__main__":
    unittest.main()
