"""End-to-end Horizon Suite tests (Options A-D) and defensive branch coverage hardening."""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from typing import Any

import hardonian_foundry as hf
from hardonian_foundry import (
    AnchorSink,
    CallbackSignerProvider,
    LocalPemSignerProvider,
    SignerProvider,
    WormDirectoryAnchorSink,
    build_proofpack,
    cohens_kappa,
    create_build_attestation,
    evaluate_trajectories,
    fleiss_kappa,
    provider,
    scan_dataset_pii,
    verify_build_attestation,
    verify_proofpack,
)
from hardonian_foundry.artifact import sha256_file
from hardonian_foundry.baseline import dataset
from hardonian_foundry.canonical import digest
from hardonian_foundry.cli import main as cli_main
from hardonian_foundry.refinery import (
    anchor_run,
    create_run,
    evaluate_promotion,
    export_bundle,
    generate_audit_report,
    sign_run,
    verify_bundle,
)
from hardonian_foundry.registry import (
    get_model,
    list_models,
    register_model,
    resolve_registry_path,
    verify_registry,
)
from hardonian_foundry.signing import generate_keypair, parse_keyring, sign
from hardonian_foundry.tenancy import TenantContext

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"

_SYNTHETIC_MANIFEST: dict[str, Any] = {
    "name": "horizon-synthetic-model",
    "license_id": "Apache-2.0",
    "provenance": "synthetic",
    "data_origin": "synthetic",
}

_NEW_PUBLIC_EXPORTS: tuple[str, ...] = (
    "AnchorSink",
    "CallbackSignerProvider",
    "LocalPemSignerProvider",
    "SignerProvider",
    "WormDirectoryAnchorSink",
    "build_proofpack",
    "cohens_kappa",
    "create_build_attestation",
    "evaluate_trajectories",
    "fleiss_kappa",
    "scan_dataset_pii",
    "verify_build_attestation",
    "verify_proofpack",
)


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_SRC_DIR}{os.pathsep}{existing}" if existing else str(_SRC_DIR)
    return env


def _build_keyring(
    keys: list[tuple[str, str, str]],
    *,
    required_signatures: int = 1,
    required_roles: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "required_signatures": required_signatures,
        "trusted_keys": [
            {
                "label": label,
                "public_key_hex": pub,
                "role": role,
            }
            for label, pub, role in keys
        ],
    }
    if required_roles is not None:
        payload["required_roles"] = required_roles
    return payload


def _rewrite_json_preserve_ledger(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class TestTopLevelHorizonExports(unittest.TestCase):
    def test_top_level_exports_and_lazy_torch_isolation(self) -> None:
        for name in _NEW_PUBLIC_EXPORTS:
            self.assertIn(name, hf.__all__)
            self.assertIsNotNone(getattr(hf, name))

        code = (
            "import sys, hardonian_foundry as hf\n"
            "assert 'torch' not in sys.modules\n"
            "assert callable(hf.build_proofpack)\n"
            "assert 'torch' not in sys.modules\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=_subprocess_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


class TestOptionACustodyAndWormSink(unittest.TestCase):
    def test_local_and_callback_signer_providers_with_cosign(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"
            kp_a = generate_keypair(root / "operator.pem")
            kp_b = generate_keypair(root / "auditor.pem")
            priv_b = root / "auditor.pem"

            run = create_run(output_root=runs_dir, model_manifest=_SYNTHETIC_MANIFEST)
            run_id = run["run_id"]

            local_signer = LocalPemSignerProvider(root / "operator.pem")
            self.assertIsInstance(local_signer, SignerProvider)
            self.assertEqual(local_signer.backend_name(), "cryptography")
            self.assertEqual(local_signer.public_key_hex(), kp_a["public_key_hex"])

            callback_signer = CallbackSignerProvider(
                public_key_hex=kp_b["public_key_hex"],
                sign_fn=lambda msg: sign(msg, priv_b),
                provider_name="hsm_mock_kms",
            )
            self.assertIsInstance(callback_signer, SignerProvider)
            self.assertEqual(callback_signer.backend_name(), "hsm_mock_kms")

            sig1 = sign_run(run_id, signer=local_signer, output_root=runs_dir)
            sig2 = sign_run(run_id, signer=callback_signer, output_root=runs_dir, cosign=True)
            self.assertEqual(sig1["run_id"], run_id)
            self.assertEqual(sig1["public_key_hex"], kp_a["public_key_hex"])
            self.assertEqual(sig2["public_key_hex"], kp_b["public_key_hex"])
            self.assertEqual(sig2["entry_seq"], sig1["entry_seq"] + 1)

            keyring = _build_keyring(
                [
                    ("op", kp_a["public_key_hex"], "operator"),
                    ("aud", kp_b["public_key_hex"], "auditor"),
                ],
                required_signatures=2,
                required_roles=["operator", "auditor"],
            )
            promo = evaluate_promotion(run_id, output_root=runs_dir, keyring=keyring)
            self.assertTrue(promo["promoted"], promo["reasons"])

    def test_signer_provider_validation_and_fail_closed_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"
            kp = generate_keypair(root / "key.pem")
            run = create_run(output_root=runs_dir, model_manifest=_SYNTHETIC_MANIFEST)

            with self.assertRaises(ValueError):
                CallbackSignerProvider(public_key_hex="not-hex", sign_fn=lambda m: "0" * 128)
            with self.assertRaises(TypeError):
                CallbackSignerProvider(public_key_hex=kp["public_key_hex"], sign_fn="nope")  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                CallbackSignerProvider(public_key_hex=kp["public_key_hex"], sign_fn=lambda m: "0" * 128, provider_name="")

            bad_callback = CallbackSignerProvider(
                public_key_hex=kp["public_key_hex"],
                sign_fn=lambda _msg: "0" * 128,
                provider_name="bad_hsm",
            )
            with self.assertRaises(ValueError):
                bad_callback.sign(b"hello")

            with self.assertRaises(ValueError):
                sign_run(run["run_id"], output_root=runs_dir)

            with self.assertRaises(TypeError):
                sign_run(run["run_id"], signer=object(), output_root=runs_dir)  # type: ignore[arg-type]

    def test_worm_directory_anchor_sink_idempotency_and_immutability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"
            worm_dir = root / "worm_store"
            run = create_run(output_root=runs_dir, model_manifest=_SYNTHETIC_MANIFEST)
            anchor_path = root / "anchor.json"

            rec = anchor_run(run["run_id"], output_root=runs_dir, anchor_path=anchor_path, worm_dir=worm_dir)
            self.assertIn("sink_receipt", rec)
            self.assertEqual(rec["sink_receipt"]["sink_type"], "worm_directory")
            self.assertTrue(rec["sink_receipt"]["immutable"])

            sink = WormDirectoryAnchorSink(worm_dir)
            self.assertIsInstance(sink, AnchorSink)
            rec_replay = anchor_run(run["run_id"], output_root=runs_dir, anchor_path=anchor_path, sink=sink)
            self.assertEqual(rec_replay["sink_receipt"]["receipt_sha256"], rec["sink_receipt"]["receipt_sha256"])

            # Mutate the anchored payload for the same head and confirm WORM conflict fails closed.
            mutated = dict(rec)
            mutated["count"] = int(rec["count"]) + 99
            with self.assertRaises(ValueError) as ctx:
                sink.publish(mutated)  # type: ignore[arg-type]
            self.assertIn("immutability violation", str(ctx.exception))

            with self.assertRaises(ValueError):
                WormDirectoryAnchorSink("../escape_worm")
            with self.assertRaises(ValueError):
                WormDirectoryAnchorSink("bad\x00dir")
            with self.assertRaises(TypeError):
                anchor_run(run["run_id"], output_root=runs_dir, sink=object())  # type: ignore[arg-type]

            # CLI integration with --worm-dir.
            worm_cli = root / "worm_cli"
            cli_anchor = root / "cli_anchor.json"
            rc = cli_main(
                [
                    "refinery",
                    "anchor",
                    run["run_id"],
                    "--output-root",
                    str(runs_dir),
                    "--out",
                    str(cli_anchor),
                    "--worm-dir",
                    str(worm_cli),
                ]
            )
            self.assertEqual(rc, 0)
            target_worm_file = next(worm_cli.glob("anchor-*.json"))
            self.assertTrue(target_worm_file.is_file())


class TestOptionBBuildLineageAttestation(unittest.TestCase):
    def test_build_attestation_sdk_and_cli_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_file = root / "adapter.bin"
            model_file.write_bytes(b"deterministic-adapter-weights-v1")
            data_file = root / "train.jsonl"
            data_file.write_text('{"id":"1"}\n', encoding="utf-8")
            data_sha, _ = sha256_file(data_file)
            subj_sha, _ = sha256_file(model_file)

            att_path = root / "adapter.attestation.json"
            att = create_build_attestation(
                model_file,
                builder_id="hardonian-foundry/test-builder@v1",
                materials=[{"uri": "dataset:train.jsonl", "sha256": data_sha}],
                run_id="run-20261111T000000Z-abcdef12",
                out_path=att_path,
            )
            self.assertEqual(att["statement_type"], "https://in-toto.io/Statement/v1")
            self.assertEqual(att["predicate_type"], "https://slsa.dev/provenance/v1")
            self.assertTrue(att["predicate"]["reproducible"])
            self.assertTrue(att["predicate"]["review_required"])
            self.assertTrue(att_path.is_file())

            ver = verify_build_attestation(att, subject_path=model_file)
            self.assertEqual(ver["status"], "VERIFIED")
            self.assertTrue(ver["verified"])

            ver_path = verify_build_attestation(att_path, expected_subject_sha256=subj_sha)
            self.assertTrue(ver_path["verified"])

            # Tampering checks fail closed.
            bad_att = dict(att)
            bad_att["attestation_sha256"] = "0" * 64
            ver_bad = verify_build_attestation(bad_att)
            self.assertEqual(ver_bad["status"], "INVALID")
            self.assertFalse(ver_bad["verified"])

            wrong_subject = root / "other.bin"
            wrong_subject.write_bytes(b"different-bytes")
            ver_mismatch = verify_build_attestation(att_path, subject_path=wrong_subject)
            self.assertFalse(ver_mismatch["verified"])

            # Validation errors.
            with self.assertRaises(ValueError):
                create_build_attestation(model_file, builder_id="bad builder\x00")
            with self.assertRaises(ValueError):
                create_build_attestation(model_file, materials=[{"uri": "bad", "sha256": "short"}])

            # CLI workflow: attest + verify-attestation (pass & fail).
            cli_att_path = root / "cli.attestation.json"
            rc_attest = cli_main(
                [
                    "refinery",
                    "artifact",
                    "attest",
                    str(model_file),
                    "--material",
                    f"train.jsonl={data_sha}",
                    "--out",
                    str(cli_att_path),
                ]
            )
            self.assertEqual(rc_attest, 0)

            rc_verify_ok = cli_main(
                [
                    "refinery",
                    "artifact",
                    "verify-attestation",
                    str(cli_att_path),
                    "--subject",
                    str(model_file),
                ]
            )
            self.assertEqual(rc_verify_ok, 0)

            rc_verify_fail = cli_main(
                [
                    "refinery",
                    "artifact",
                    "verify-attestation",
                    str(cli_att_path),
                    "--subject",
                    str(wrong_subject),
                ]
            )
            self.assertEqual(rc_verify_fail, 1)

            rc_bad_mat = cli_main(
                [
                    "refinery",
                    "artifact",
                    "attest",
                    str(model_file),
                    "--material",
                    "missing-equals-sign",
                ]
            )
            self.assertEqual(rc_bad_mat, 2)


class TestOptionCAgreementTrajectoriesPiiAndGovernance(unittest.TestCase):
    def test_cohens_and_fleiss_kappa(self) -> None:
        res_perfect = cohens_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"], min_kappa=0.75)
        self.assertEqual(res_perfect["kappa"], 1.0)
        self.assertTrue(res_perfect["passes_threshold"])
        self.assertTrue(res_perfect["review_required"])

        res_low = cohens_kappa(["A", "B", "A", "B"], ["B", "A", "B", "A"], min_kappa=0.75)
        self.assertFalse(res_low["passes_threshold"])
        self.assertLess(res_low["kappa"], 0.0)

        with self.assertRaises(ValueError):
            cohens_kappa(["A"], ["A", "B"])
        with self.assertRaises(ValueError):
            cohens_kappa(["A"], ["A"], min_kappa=1.5)

        fl_high = fleiss_kappa([[3, 0, 0], [0, 3, 0], [0, 0, 3]], min_kappa=0.75)
        self.assertEqual(fl_high["kappa"], 1.0)
        self.assertTrue(fl_high["passes_threshold"])

        fl_low = fleiss_kappa([[1, 2], [2, 1], [1, 2], [2, 1]], min_kappa=0.75)
        self.assertFalse(fl_low["passes_threshold"])

        with self.assertRaises(ValueError):
            fleiss_kappa([[1, 2], [3, 1]])
        with self.assertRaises(ValueError):
            fleiss_kappa([[1], [1]])

    def test_evaluate_trajectories_fail_closed(self) -> None:
        safe = [
            {
                "id": "traj-safe-1",
                "steps": [
                    {
                        "tool": "lookup_policy",
                        "decision": "ALLOW_PROPOSAL",
                        "executed": False,
                        "args": {"doc": "policy.md"},
                    },
                    {"tool": "execute_wire", "decision": "DENY", "executed": False, "args": {}},
                ],
            }
        ]
        ok = evaluate_trajectories(safe, allowed_tools=["lookup_policy"], forbidden_tools=["execute_wire"])
        self.assertEqual(ok["status"], "PASSED")
        self.assertTrue(ok["passed"])
        self.assertEqual(ok["false_tool_execution_count"], 0)
        self.assertTrue(ok["review_required"])

        bad = [
            {
                "id": "traj-bad-1",
                "steps": [
                    {
                        "tool": "execute_wire",
                        "decision": "ALLOW_PROPOSAL",
                        "executed": True,
                        "args": {"path": "../secret"},
                    },
                ],
            }
        ]
        blocked = evaluate_trajectories(bad, allowed_tools=["lookup_policy"], forbidden_tools=["execute_wire"])
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertFalse(blocked["passed"])
        self.assertEqual(blocked["false_tool_execution_count"], 1)
        self.assertEqual(blocked["unauthorized_tool_count"], 1)
        self.assertEqual(blocked["unsafe_arg_count"], 1)

        with self.assertRaises(ValueError):
            evaluate_trajectories([], allowed_tools=["lookup_policy"])

    def test_scan_dataset_pii_and_consented_pilot_end_to_end(self) -> None:
        clean_records = [
            {"id": "rec-1", "prompt": "Compare invoice 1001 vs receipt 1001", "notes": "synthetic sample"},
            {"id": "rec-2", "nested": ["amount 450.00", {"currency": "USD"}]},
        ]
        clean_report = scan_dataset_pii(clean_records)
        self.assertEqual(clean_report["status"], "CLEAN")
        self.assertTrue(clean_report["clean"])
        self.assertEqual(clean_report["violation_count"], 0)

        secret_email = "alice.operator@example.com"
        secret_ssn = "123-45-6789"
        secret_iban = "DE89370400440532013000"
        secret_api = "sk-AbCdEfGhIjKlMnOpQrSt1234"
        dirty_records = [
            {
                "id": "dirty-1",
                "email": secret_email,
                "meta": {"ssn": secret_ssn, "iban": secret_iban, "key": secret_api},
            }
        ]
        dirty_report = scan_dataset_pii(dirty_records)
        self.assertEqual(dirty_report["status"], "VIOLATION")
        self.assertFalse(dirty_report["clean"])
        self.assertEqual(dirty_report["violation_count"], 4)
        serialized = json.dumps(dirty_report)
        for secret in (secret_email, secret_ssn, secret_iban, secret_api):
            self.assertNotIn(secret, serialized)

        # End-to-end consented_pilot + governance_attestation lifecycle.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"
            kp = generate_keypair(root / "pilot.pem")
            pilot_manifest = {
                "name": "consented-pilot-adapter",
                "license_id": "Apache-2.0",
                "provenance": "consented_pilot",
                "data_origin": "consented_pilot",
                "governance_attestation": {
                    "tier": "consented_pilot",
                    "attestation_id": "gov-att-2026-001",
                    "approved_by": "compliance-officer",
                    "approved_at": "2026-11-14T12:00:00Z",
                },
            }
            # Denied under default synthetic-only policy.
            with self.assertRaises(ValueError):
                create_run(output_root=runs_dir, model_manifest=pilot_manifest)

            # Allowed when consented_pilot is explicitly authorized alongside governance_attestation.
            allowed = ("synthetic", "consented_pilot")
            run = create_run(
                output_root=runs_dir,
                model_manifest=pilot_manifest,
                allowed_data_origins=allowed,
            )
            sign_run(
                run["run_id"],
                private_key_path=root / "pilot.pem",
                output_root=runs_dir,
                allowed_data_origins=allowed,
            )
            keyring = _build_keyring([("pilot-op", kp["public_key_hex"], "operator")])
            promo = evaluate_promotion(
                run["run_id"],
                output_root=runs_dir,
                keyring=keyring,
                allowed_data_origins=allowed,
            )
            self.assertTrue(promo["promoted"], promo["reasons"])


class TestOptionDProofpackV2(unittest.TestCase):
    def test_proofpack_sdk_and_cli_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"
            kp = generate_keypair(root / "release.pem")
            keyring = _build_keyring([("rel", kp["public_key_hex"], "release")])
            keyring_file = root / "keyring.json"
            keyring_file.write_text(json.dumps(keyring), encoding="utf-8")

            run = create_run(output_root=runs_dir, model_manifest=_SYNTHETIC_MANIFEST)
            run_id = run["run_id"]

            # Unpromoted proofpack verifies structurally, but fails --require-promoted.
            unpromoted_path = root / "unpromoted.proofpack.json"
            unpromoted = build_proofpack(run_id, output_root=runs_dir, out_path=unpromoted_path)
            self.assertEqual(unpromoted["schema_version"], "2.0")
            self.assertEqual(unpromoted["proofpack_type"], "hardonian.foundry.proofpack.v2")
            self.assertTrue(verify_proofpack(unpromoted)["verified"])
            self.assertFalse(verify_proofpack(unpromoted_path, require_promoted=True)["verified"])

            rc_cli_unpromoted = cli_main(["refinery", "verify-proofpack", str(unpromoted_path), "--require-promoted"])
            self.assertEqual(rc_cli_unpromoted, 2)

            # Sign & anchor, then build promoted Proofpack v2 via SDK & CLI.
            sign_run(run_id, private_key_path=root / "release.pem", output_root=runs_dir)
            anchor = anchor_run(run_id, output_root=runs_dir, worm_dir=root / "worm")

            promoted_path = root / "promoted.proofpack.json"
            rc_build = cli_main(
                [
                    "refinery",
                    "proofpack",
                    run_id,
                    "--output-root",
                    str(runs_dir),
                    "--keyring",
                    str(keyring_file),
                    "--expected-head",
                    anchor["head"],
                    "--expected-count",
                    str(anchor["count"]),
                    "--out",
                    str(promoted_path),
                ]
            )
            self.assertEqual(rc_build, 0)
            rc_ver = cli_main(["refinery", "verify-proofpack", str(promoted_path), "--require-promoted"])
            self.assertEqual(rc_ver, 0)

            loaded = json.loads(promoted_path.read_text(encoding="utf-8"))
            self.assertTrue(verify_proofpack(loaded, require_promoted=True)["verified"])

            # Fail-closed invariant checks on tampered proofpack fields.
            tampered_fin = json.loads(json.dumps(loaded))
            tampered_fin["financial_write"] = True
            self.assertFalse(verify_proofpack(tampered_fin)["verified"])

            tampered_rev = json.loads(json.dumps(loaded))
            tampered_rev["review_required"] = False
            self.assertFalse(verify_proofpack(tampered_rev)["verified"])

            tampered_digest = json.loads(json.dumps(loaded))
            tampered_digest["proofpack_sha256"] = "f" * 64
            self.assertFalse(verify_proofpack(tampered_digest)["verified"])


class TestRegistryAndBundleCoverageHardening(unittest.TestCase):
    def test_registry_defensive_and_error_branches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"
            kp = generate_keypair(root / "reg.pem")
            keyring = _build_keyring([("reg", kp["public_key_hex"], "release")])

            # Path and parameter validation errors.
            with self.assertRaises(ValueError):
                resolve_registry_path(registry_path="")
            with self.assertRaises(ValueError):
                resolve_registry_path(registry_path="../escape.json")
            with self.assertRaises(ValueError):
                list_models(output_root=runs_dir, tenant_context="not-a-tenant")  # type: ignore[arg-type]

            tenant_a = TenantContext(tenant_id="tenant-a", request_id="req-a-001")
            tenant_b = TenantContext(tenant_id="tenant-b", request_id="req-b-001")
            run_a = create_run(output_root=runs_dir, model_manifest=_SYNTHETIC_MANIFEST, tenant_context=tenant_a)
            sign_run(run_a["run_id"], private_key_path=root / "reg.pem", output_root=runs_dir)

            reg_path = runs_dir / "registry.json"
            with self.assertRaises(KeyError):
                get_model("missing-model", output_root=runs_dir, registry_path=reg_path)

            with self.assertRaises(ValueError):
                register_model(
                    run_a["run_id"],
                    output_root=runs_dir,
                    registry_path=reg_path,
                    keyring=keyring,
                    tenant_context=tenant_a,
                    notes="bad\x00note",
                )

            register_model(
                run_a["run_id"],
                model_id="shared-model",
                output_root=runs_dir,
                registry_path=reg_path,
                keyring=keyring,
                tenant_context=tenant_a,
                notes="clean release note",
            )

            # Tenant isolation on same model_id.
            run_b = create_run(output_root=runs_dir, model_manifest=_SYNTHETIC_MANIFEST, tenant_context=tenant_b)
            sign_run(run_b["run_id"], private_key_path=root / "reg.pem", output_root=runs_dir)
            with self.assertRaises(PermissionError):
                register_model(
                    run_b["run_id"],
                    model_id="shared-model",
                    output_root=runs_dir,
                    registry_path=reg_path,
                    keyring=keyring,
                    tenant_context=tenant_b,
                )

            with self.assertRaises(PermissionError):
                get_model("shared-model", output_root=runs_dir, registry_path=reg_path, tenant_context=tenant_b)

            # Corrupt registry file cases -> verify_registry TAMPERED.
            corrupt_non_dict = root / "corrupt_list.json"
            corrupt_non_dict.write_text("[]", encoding="utf-8")
            self.assertEqual(verify_registry(registry_path=corrupt_non_dict)["status"], "TAMPERED")

            corrupt_missing_keys = root / "corrupt_keys.json"
            corrupt_missing_keys.write_text('{"schema_version": 123}', encoding="utf-8")
            self.assertEqual(verify_registry(registry_path=corrupt_missing_keys)["status"], "TAMPERED")

            # Missing run_id / mismatched test_sha256 / deleted run dir in signed registry.
            reg_data = json.loads(reg_path.read_text(encoding="utf-8"))
            broken_models = [dict(reg_data["models"][0])]
            broken_models[0]["test_sha256"] = "0" * 64
            broken_models.append({"model_id": "no-run", "run_id": ""})
            broken_core = {"schema_version": reg_data["schema_version"], "models": broken_models}
            broken_reg = root / "broken_reg.json"
            _rewrite_json_preserve_ledger(
                broken_reg,
                {**broken_core, "registry_sha256": digest(broken_core)},
            )
            res_broken = verify_registry(output_root=runs_dir, registry_path=broken_reg, keyring=keyring)
            self.assertEqual(res_broken["status"], "TAMPERED")
            self.assertTrue(any("test_sha256 mismatch" in err for err in res_broken["errors"]))
            self.assertTrue(any("missing valid run_id" in err for err in res_broken["errors"]))

            # Deleted run directory handled cleanly as TAMPERED error.
            shutil.rmtree(runs_dir / run_a["run_id"])
            res_deleted = verify_registry(output_root=runs_dir, registry_path=reg_path)
            self.assertEqual(res_deleted["status"], "TAMPERED")

    def test_bundle_verification_and_report_error_branches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"
            run = create_run(output_root=runs_dir, model_manifest=_SYNTHETIC_MANIFEST)
            run_id = run["run_id"]

            # Remove declared non-core file (`inspect.json`) -> fail-closed ValueError.
            dir_bundle = root / "bundle_dir"
            export_bundle(run_id, output_root=runs_dir, dest_path=dir_bundle, fmt="dir")
            self.assertTrue(verify_bundle(dir_bundle)["verified"])
            (dir_bundle / "inspect.json").unlink()
            with self.assertRaises(ValueError):
                verify_bundle(dir_bundle)

            # Remove required core file (`run.json`) -> fail-closed ValueError.
            (dir_bundle / "run.json").unlink()
            with self.assertRaises(ValueError):
                verify_bundle(dir_bundle)

            # Tamper ledger inside exported directory bundle -> sha256 mismatch ValueError.
            tampered_bundle = root / "tampered_bundle"
            export_bundle(run_id, output_root=runs_dir, dest_path=tampered_bundle, fmt="dir")
            bad_ledger = json.dumps({"seq": 1, "prev_hash": "0" * 64, "record": {}, "entry_hash": "1" * 64}) + "\n"
            (tampered_bundle / "evidence.jsonl").write_text(bad_ledger, encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_bundle(tampered_bundle)

            # Update manifest digest for the corrupted ledger to exercise bundle ledger verification failure.
            mdata = json.loads((tampered_bundle / "manifest.json").read_text(encoding="utf-8"))
            for item in mdata["files"]:
                if item["path"] == "evidence.jsonl":
                    bad_bytes = bad_ledger.encode("utf-8")
                    item["sha256"] = hashlib.sha256(bad_bytes).hexdigest()
                    item["size_bytes"] = len(bad_bytes)
            mcore = {k: v for k, v in mdata.items() if k != "bundle_sha256"}
            _rewrite_json_preserve_ledger(
                tampered_bundle / "manifest.json",
                {**mcore, "bundle_sha256": digest(mcore)},
            )
            with self.assertRaises(ValueError):
                verify_bundle(tampered_bundle)

            # Unsafe zip member path (`../escape.txt`).
            unsafe_zip = root / "unsafe.zip"
            with zipfile.ZipFile(unsafe_zip, "w") as zf:
                zf.writestr("../escape.txt", "boom")
            with self.assertRaises(ValueError):
                verify_bundle(unsafe_zip)

            # Unsafe tar member path (`../escape.txt`).
            unsafe_tar = root / "unsafe.tar.gz"
            with tarfile.open(unsafe_tar, "w:gz") as tf:
                payload = root / "payload.txt"
                payload.write_text("boom", encoding="utf-8")
                tf.add(payload, arcname="../escape.txt")
            with self.assertRaises(ValueError):
                verify_bundle(unsafe_tar)

            # generate_audit_report validation branches.
            with self.assertRaises(ValueError):
                generate_audit_report(run_id, output_root=runs_dir, tenant_context="bad")  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                generate_audit_report(run_id, output_root=runs_dir, out_path="")


class TestHorizonValidationAndCoverage(unittest.TestCase):
    def test_proofpack_validation_and_invariant_error_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"
            run = create_run(output_root=runs_dir, model_manifest=_SYNTHETIC_MANIFEST)
            run_id = run["run_id"]
            pack = build_proofpack(run_id, output_root=runs_dir)

            with self.assertRaises(ValueError):
                build_proofpack(run_id, output_root=runs_dir, out_path="")
            with self.assertRaises(ValueError):
                build_proofpack(run_id, output_root=runs_dir, out_path="../escape.json")

            # Non-dict / nonexistent / corrupt JSON / wrong type.
            with self.assertRaises(TypeError):
                verify_proofpack(123)  # type: ignore[arg-type]
            with self.assertRaises(FileNotFoundError):
                verify_proofpack(root / "missing.proofpack.json")
            corrupt_pp = root / "corrupt.proofpack.json"
            corrupt_pp.write_text("{bad-json", encoding="utf-8")
            self.assertFalse(verify_proofpack(corrupt_pp)["verified"])
            list_file = root / "list.json"
            list_file.write_text("[]", encoding="utf-8")
            self.assertFalse(verify_proofpack(list_file)["verified"])

            for field, bad_val in (
                ("schema_version", "9.9"),
                ("proofpack_type", "wrong.type"),
                ("executed", True),
                ("coverage_automatic_decision", 0.5),
                ("run_id", "../bad"),
                ("dataset_sha256", "not-hex"),
                ("test_sha256", "not-hex"),
            ):
                mutated = json.loads(json.dumps(pack))
                mutated[field] = bad_val
                self.assertFalse(verify_proofpack(mutated)["verified"], f"expected failure for {field}")

            bad_prov = json.loads(json.dumps(pack))
            bad_prov["provenance_status"] = "TAMPERED"
            bad_prov["promoted"] = True
            bad_prov["status"] = "ALLOWED_REVIEW_REQUIRED"
            self.assertFalse(verify_proofpack(bad_prov, require_promoted=True)["verified"])

            # CLI error paths for verify-proofpack and proofpack.
            self.assertEqual(cli_main(["refinery", "verify-proofpack", str(root / "missing.json")]), 2)
            self.assertEqual(
                cli_main(["refinery", "proofpack", run_id, "--output-root", str(runs_dir), "--out", "../bad.json"]),
                2,
            )

    def test_build_attestation_and_worm_signer_edge_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_file = root / "model.bin"
            model_file.write_bytes(b"weights")
            att = create_build_attestation(model_file)

            with self.assertRaises((TypeError, ValueError)):
                create_build_attestation(model_file, materials="bad")  # type: ignore[arg-type]
            with self.assertRaises((TypeError, ValueError)):
                create_build_attestation(model_file, materials=["bad"])  # type: ignore[list-item]
            with self.assertRaises(ValueError):
                create_build_attestation(model_file, out_path="")

            with self.assertRaises(TypeError):
                verify_build_attestation(123)  # type: ignore[arg-type]
            with self.assertRaises(FileNotFoundError):
                verify_build_attestation(root / "nope.json")
            corrupt_att = root / "corrupt_att.json"
            corrupt_att.write_text("{bad-json", encoding="utf-8")
            self.assertFalse(verify_build_attestation(corrupt_att)["verified"])
            bad_list = root / "arr.json"
            bad_list.write_text("[]", encoding="utf-8")
            self.assertFalse(verify_build_attestation(bad_list)["verified"])

            for mutator in (
                lambda p: p.__setitem__("statement_type", "wrong"),
                lambda p: p.__setitem__("predicate_type", "wrong"),
                lambda p: p.__setitem__("subject", "not-dict"),
                lambda p: p["subject"].__setitem__("sha256", "short"),
                lambda p: p.__setitem__("predicate", "not-dict"),
                lambda p: p["predicate"].__setitem__("review_required", False),
            ):
                mut = json.loads(json.dumps(att))
                mutator(mut)
                self.assertFalse(verify_build_attestation(mut)["verified"])

            with self.assertRaises(ValueError):
                verify_build_attestation(att, expected_subject_sha256="bad-hex")
            self.assertFalse(verify_build_attestation(att, expected_subject_sha256="0" * 64)["verified"])

            att_path = root / "att.json"
            att_path.write_text(json.dumps(att), encoding="utf-8")
            rc = cli_main(
                [
                    "refinery",
                    "artifact",
                    "verify-attestation",
                    str(att_path),
                    "--expected-sha256",
                    "0" * 64,
                ]
            )
            self.assertEqual(rc, 1)

            # WormDirectoryAnchorSink validation branches.
            with self.assertRaises(ValueError):
                WormDirectoryAnchorSink("")
            worm_dir = root / "worm"
            sink = WormDirectoryAnchorSink(worm_dir)
            with self.assertRaises(ValueError):
                sink.publish({})  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                sink.publish({"head": "bad", "count": 1, "run_id": "run-1"})  # type: ignore[typeddict-item]

            # Corrupt pre-existing WORM anchor file for head.
            worm_dir.mkdir(parents=True, exist_ok=True)
            (worm_dir / f"anchor-{'b' * 64}.json").write_text("{corrupt", encoding="utf-8")
            with self.assertRaises(ValueError):
                sink.publish({"head": "b" * 64, "count": 1, "run_id": "run-1"})  # type: ignore[typeddict-item]

            # SignerProvider validation branches.
            kp = generate_keypair(root / "signer.pem")
            missing_signer = LocalPemSignerProvider(root / "nonexistent.pem")
            with self.assertRaises(FileNotFoundError):
                missing_signer.public_key_hex()
            with self.assertRaises(FileNotFoundError):
                missing_signer.sign(b"hello")
            with self.assertRaises(ValueError):
                CallbackSignerProvider(
                    public_key_hex=kp["public_key_hex"],
                    sign_fn=lambda m: sign(m, root / "signer.pem"),
                    provider_name="bad\x00name",
                )
            non_str_cb = CallbackSignerProvider(
                public_key_hex=kp["public_key_hex"],
                sign_fn=lambda _m: 123,  # type: ignore[arg-type, return-value]
                provider_name="non_str_cb",
            )
            with self.assertRaises((TypeError, ValueError)):
                non_str_cb.sign(b"hello")

    def test_task_eval_horizon_validation_branches(self) -> None:
        with self.assertRaises(ValueError):
            cohens_kappa("AB", ["A", "B"])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            cohens_kappa([" A "], ["A"])
        with self.assertRaises(ValueError):
            fleiss_kappa("bad")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fleiss_kappa(["bad"])  # type: ignore[list-item]
        with self.assertRaises(ValueError):
            fleiss_kappa([[1, 1], [1, 1, 0]])
        with self.assertRaises(ValueError):
            fleiss_kappa([[True, 2]])  # type: ignore[list-item]
        with self.assertRaises(ValueError):
            fleiss_kappa([[1, 0]])
        with self.assertRaises(ValueError):
            fleiss_kappa([[1, 1]], min_kappa=-0.1)

        with self.assertRaises(ValueError):
            evaluate_trajectories([{"id": "t1", "steps": [{"tool": "a", "decision": "DENY"}]}], allowed_tools=["a"], forbidden_tools="bad")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            evaluate_trajectories(["bad"], allowed_tools=["a"])  # type: ignore[list-item]
        with self.assertRaises(ValueError):
            evaluate_trajectories([{"id": "", "steps": []}], allowed_tools=["a"])
        with self.assertRaises(ValueError):
            evaluate_trajectories([{"id": "t1", "steps": []}], allowed_tools=["a"])
        with self.assertRaises(ValueError):
            evaluate_trajectories([{"id": "t1", "steps": ["bad"]}], allowed_tools=["a"])  # type: ignore[list-item]
        with self.assertRaises(ValueError):
            evaluate_trajectories([{"id": "t1", "steps": [{"tool": "", "decision": "DENY"}]}], allowed_tools=["a"])

        # Nested control chars in list args trigger unsafe_args.
        res_nested = evaluate_trajectories(
            [
                {
                    "id": "t-nested",
                    "steps": [{"tool": "a", "decision": "ALLOW_PROPOSAL", "args": ["ok", "bad\x01arg"]}],
                }
            ],
            allowed_tools=["a"],
        )
        self.assertEqual(res_nested["unsafe_arg_count"], 1)

        with self.assertRaises(ValueError):
            scan_dataset_pii("bad")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            scan_dataset_pii(["not-a-dict"])  # type: ignore[list-item]

    def test_provider_signing_keyring_and_bundle_edge_paths(self) -> None:
        # provider URL, SSL context, and bearer token validation branches.
        with self.assertRaises(ValueError):
            provider.validate_loopback_url("")
        with self.assertRaises(ValueError):
            provider._build_ssl_context(Path("does_not_exist_ca_bundle.pem"))
        with self.assertRaises(ValueError):
            provider._resolve_bearer_header("INVALID-ENV-NAME")
        os.environ.pop("HF_MISSING_TOKEN_XYZ", None)
        with self.assertRaises(provider.ProviderError):
            provider._resolve_bearer_header("HF_MISSING_TOKEN_XYZ")

        # signing.parse_keyring defensive branches.
        valid_pub = "a" * 64
        for bad_keyring in (
            "not-a-dict",
            {"unknown_field": 1, "trusted_keys": [{"label": "k", "public_key_hex": valid_pub}]},
            {"required_signatures": 0, "trusted_keys": [{"label": "k", "public_key_hex": valid_pub}]},
            {"required_signatures": True, "trusted_keys": [{"label": "k", "public_key_hex": valid_pub}]},
            {"required_roles": "operator", "trusted_keys": [{"label": "k", "public_key_hex": valid_pub}]},
            {"required_roles": [""], "trusted_keys": [{"label": "k", "public_key_hex": valid_pub}]},
            {"trusted_keys": "bad"},
            {"trusted_keys": ["bad"]},
            {
                "trusted_keys": [
                    {"label": "k1", "public_key_hex": valid_pub},
                    {"label": "k2", "public_key_hex": valid_pub},
                ]
            },
            {"trusted_keys": [{"label": "k1", "public_key_hex": valid_pub, "revoked_at": ""}]},
            {"trusted_keys": [{"label": "k1", "public_key_hex": valid_pub, "role": ""}]},
        ):
            with self.assertRaises((ValueError, TypeError)):
                parse_keyring(bad_keyring)

        revoked_parsed = parse_keyring(
            {"trusted_keys": [{"label": "k1", "public_key_hex": valid_pub, "revoked_at": "2026-01-01T00:00:00Z"}]}
        )
        self.assertEqual(revoked_parsed["revoked_keys"], [valid_pub])

        # Bundle & audit report format edge cases.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"
            run = create_run(output_root=runs_dir, model_manifest=_SYNTHETIC_MANIFEST)
            run_id = run["run_id"]

            zip_bundle = root / "b.zip"
            export_bundle(run_id, output_root=runs_dir, dest_path=zip_bundle, fmt="zip")
            self.assertTrue(verify_bundle(zip_bundle)["verified"])

            tar_bundle = root / "b.tar.gz"
            export_bundle(run_id, output_root=runs_dir, dest_path=tar_bundle, fmt="tar.gz")
            self.assertTrue(verify_bundle(tar_bundle)["verified"])

            with self.assertRaises(ValueError):
                export_bundle(run_id, output_root=runs_dir, dest_path=root / "b.bad", fmt="7z")
            with self.assertRaises(FileNotFoundError):
                verify_bundle(root / "missing_bundle")

            dir_bundle = root / "bad_manifest_dir"
            export_bundle(run_id, output_root=runs_dir, dest_path=dir_bundle, fmt="dir")
            manifest = json.loads((dir_bundle / "manifest.json").read_text(encoding="utf-8"))
            manifest["bundle_sha256"] = "0" * 64
            _rewrite_json_preserve_ledger(dir_bundle / "manifest.json", manifest)
            with self.assertRaises(ValueError):
                verify_bundle(dir_bundle)

            md_out = root / "report.md"
            rep_md = generate_audit_report(run_id, output_root=runs_dir, fmt="markdown", out_path=md_out)
            self.assertTrue(md_out.is_file())
            self.assertEqual(rep_md["saved_path"], str(md_out.resolve()))
            with self.assertRaises(ValueError):
                generate_audit_report(run_id, output_root=runs_dir, fmt="xml")

    def test_cli_and_provider_http_coverage_hardening(self) -> None:
        # CLI provider/registry error & unpromoted paths.
        self.assertEqual(cli_main(["provider", "probe"]), 2)
        self.assertEqual(cli_main(["provider", "probe", "--origin", "http://evil.com:8080", "--model", "m"]), 2)
        self.assertEqual(cli_main(["registry", "list"]), 0)
        self.assertEqual(cli_main(["registry", "get"]), 2)
        self.assertEqual(cli_main(["registry", "register"]), 2)

        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            run = create_run(output_root=runs_dir, model_manifest=_SYNTHETIC_MANIFEST)
            self.assertEqual(
                cli_main(["registry", "promote", run["run_id"], "--output-root", str(runs_dir), "--json-errors"]),
                2,
            )

        # Real loopback HTTP server exercising 200, 302 redirect rejection, and 500 HTTP error.
        status_mode = {"code": 200}

        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                _ = self.rfile.read(length)
                code = status_mode["code"]
                if code == 200:
                    body = json.dumps({"choices": [{"message": {"content": '{"category": "matched"}'}}]}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif code == 302:
                    self.send_response(302)
                    self.send_header("Location", "http://127.0.0.1:9999/")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                else:
                    self.send_response(500)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

        srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            origin = f"http://127.0.0.1:{srv.server_port}"
            breaker = provider.CircuitBreaker(failure_threshold=5)
            case = dataset()[0]
            ok = provider.suggest(case, origin=origin, model="synthetic", circuit_breaker=breaker)
            self.assertEqual(ok["status"], "REVIEW_REQUIRED")

            status_mode["code"] = 302
            redir = provider.suggest(case, origin=origin, model="synthetic", circuit_breaker=breaker)
            self.assertEqual(redir["status"], "UNAVAILABLE")
            self.assertEqual(redir["reason_code"], "REDIRECT_REJECTED")

            status_mode["code"] = 500
            err = provider.suggest(case, origin=origin, model="synthetic", circuit_breaker=breaker)
            self.assertEqual(err["status"], "UNAVAILABLE")
            self.assertEqual(err["reason_code"], "HTTP_ERROR")
        finally:
            srv.shutdown()
            srv.server_close()



