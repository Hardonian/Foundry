"""End-to-end tests for Foundry P0-P3 differentiation, adoption, moat, and expansion suite."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import yaml

import hardonian_foundry as hf
from hardonian_foundry import (
    COMPLIANCE_PROFILES,
    IMPORT_FORMATS,
    PROVIDER_PRESETS,
    TEMPLATES,
    FoundryMcpServer,
    ObjectLockAnchorSink,
    Rfc3161TsaAnchorSink,
    SignerProvider,
    SubprocessKmsSignerProvider,
    build_proofpack,
    create_consent_receipt,
    evaluate_compliance_profile,
    evaluate_policy_bundle,
    export_schemas,
    generate_audit_report,
    generate_dataset_card,
    generate_mcp_adversarial_suite,
    get_schema,
    import_eval_results,
    init_project,
    list_schemas,
    render_audit_html,
    resolve_provider_origin,
    verify_consent_receipt,
    verify_proofpack,
)
from hardonian_foundry.cli import main as cli_main
from hardonian_foundry.refinery import AnchorSink, anchor_run, create_run, create_task_run, register_model, sign_run
from hardonian_foundry.signing import generate_keypair
from hardonian_foundry.task_eval import load_jsonl, validate_spec, validate_task_dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SYNTHETIC_MANIFEST: dict[str, Any] = {
    "name": "expansion-synthetic-model",
    "license_id": "Apache-2.0",
    "provenance": "synthetic",
    "data_origin": "synthetic",
}


class _InMemoryObjectLockClient:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}

    def head_object(self, *, bucket: str, key: str) -> dict[str, Any] | None:
        return self.objects.get((bucket, key))

    def put_object(
        self,
        *,
        bucket: str,
        key: str,
        body: bytes,
        object_lock_mode: str,
        retain_until_date: str,
    ) -> dict[str, Any]:
        from hardonian_foundry.canonical import digest

        payload = json.loads(body.decode("utf-8"))
        self.objects[(bucket, key)] = {
            "receipt_sha256": digest(payload),
            "object_lock_mode": object_lock_mode,
            "retain_until_date": retain_until_date,
        }
        return {"etag": '"etag-test-001"'}


def _run_cli_json(argv: list[str]) -> tuple[int, dict[str, Any]]:
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = cli_main(argv)
    return rc, json.loads(buf.getvalue())


class TestScaffoldAndSchemas(unittest.TestCase):
    def test_init_all_templates_and_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for tpl in TEMPLATES:
                dest = root / f"proj-{tpl}"
                res = init_project(dest, template=tpl, name=f"demo-{tpl}")
                self.assertEqual(res["status"], "CREATED")
                self.assertEqual(res["record_count"], 24)
                spec = json.loads((dest / "spec.json").read_text(encoding="utf-8"))
                self.assertEqual(validate_spec(spec), [])
                records = load_jsonl(dest / "data.jsonl")
                val = validate_task_dataset(records, spec)
                self.assertEqual(val["status"], "VALID")
                self.assertTrue((dest / ".github" / "workflows" / "foundry-gate.yml").is_file())
                if tpl == "trajectory":
                    self.assertTrue((dest / "trajectories.jsonl").is_file())

            # Refuses overwrite without force=True
            with self.assertRaises(FileExistsError):
                init_project(root / "proj-classifier", template="classifier", name="demo-classifier")
            forced = init_project(root / "proj-classifier", template="classifier", name="demo-classifier", force=True)
            self.assertEqual(forced["status"], "CREATED")

            # Path traversal and invalid template/name fail closed
            with self.assertRaises(ValueError):
                init_project(root / "bad-tpl", template="unknown")
            with self.assertRaises(ValueError):
                init_project("../escape-init")
            with self.assertRaises(ValueError):
                init_project(root / "bad-name", name="bad name!")

            # CLI init
            rc, out = _run_cli_json(["init", str(root / "cli-proj"), "--template", "tool-gate", "--name", "cli-gate"])
            self.assertEqual(rc, 0)
            self.assertEqual(out["task_id"], "cli-gate")
            rc_dup = cli_main(["init", str(root / "cli-proj"), "--template", "tool-gate", "--name", "cli-gate"])
            self.assertEqual(rc_dup, 2)

    def test_schemas_api_and_cli(self) -> None:
        names = list_schemas()
        self.assertIn("proofpack-v2", names)
        self.assertIn("task-spec", names)
        for name in names:
            schema = get_schema(name)
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")

        with self.assertRaises(ValueError):
            get_schema("nonexistent-schema")
        with self.assertRaises(ValueError):
            export_schemas("../escape-schemas")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            res = export_schemas(root / "schemas")
            self.assertEqual(len(res["files"]), len(names))
            rc, list_out = _run_cli_json(["schema"])
            self.assertEqual(rc, 0)
            self.assertEqual(list_out["schemas"], names)
            rc_one, one_out = _run_cli_json(["schema", "proofpack-v2"])
            self.assertEqual(rc_one, 0)
            self.assertEqual(one_out["$schema"], "https://json-schema.org/draft/2020-12/schema")
            rc_all, all_out = _run_cli_json(["schema", "--all", "--out", str(root / "cli-schemas")])
            self.assertEqual(rc_all, 0)
            self.assertEqual(all_out["status"], "EXPORTED")


class TestSignedProofpackAndAirGapVerifier(unittest.TestCase):
    def test_proofpack_cryptographic_receipt_keyring_and_standalone_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"
            kp1 = generate_keypair(root / "k1.pem")
            kp2 = generate_keypair(root / "k2.pem")
            keyring = {
                "required_signatures": 1,
                "required_roles": ["release"],
                "trusted_keys": [
                    {"public_key_hex": kp1["public_key_hex"], "key_id": "k1", "role": "release"},
                    {"public_key_hex": kp2["public_key_hex"], "key_id": "k2", "role": "security", "revoked_at": "2026-01-01T00:00:00Z"},
                ],
            }
            keyring_file = root / "keyring.json"
            keyring_file.write_text(json.dumps(keyring), encoding="utf-8")

            run = create_run(output_root=runs_dir, model_manifest=_SYNTHETIC_MANIFEST)
            sign_run(run["run_id"], private_key_path=root / "k1.pem", output_root=runs_dir)
            anchor = anchor_run(run["run_id"], output_root=runs_dir)
            register_model(run["run_id"], output_root=runs_dir, keyring=keyring)

            pp_path = root / "pack.json"
            pack = build_proofpack(
                run["run_id"],
                output_root=runs_dir,
                keyring=keyring,
                expected_head=anchor["head"],
                expected_count=anchor["count"],
                out_path=pp_path,
                consumers=["CustomConsumer"],
            )
            self.assertEqual(pack["downstream_contract"]["consumers"], ["CustomConsumer"])
            self.assertIn("cryptographic_receipt", pack)
            self.assertEqual(pack["cryptographic_receipt"]["signature_count"], 1)

            ver = verify_proofpack(pp_path, require_promoted=True, keyring=keyring_file, expected_head=anchor["head"])
            self.assertTrue(ver["verified"])
            self.assertTrue(ver["signature_verified"])
            self.assertEqual(ver["verified_signers"], [kp1["public_key_hex"]])

            # Wrong expected head or untrusted/revoked key fails closed
            bad_head = verify_proofpack(pp_path, keyring=keyring_file, expected_head="0" * 64)
            self.assertFalse(bad_head["verified"])
            wrong_kr = verify_proofpack(pp_path, keyring=[kp2["public_key_hex"]])
            self.assertFalse(wrong_kr["verified"])

            # Revoked signer fails closed even if previously verified
            revoked_kr = {"trusted_keys": [{"public_key_hex": kp1["public_key_hex"], "revoked_at": "2026-11-24T00:00:00Z"}]}
            self.assertFalse(verify_proofpack(pp_path, keyring=revoked_kr)["verified"])

            # CLI verify-proofpack with --keyring and --expected-head
            rc_pp, pp_cli = _run_cli_json([
                "refinery",
                "verify-proofpack",
                str(pp_path),
                "--require-promoted",
                "--keyring",
                str(keyring_file),
                "--expected-head",
                anchor["head"],
            ])
            self.assertEqual(rc_pp, 0)
            self.assertTrue(pp_cli["signature_verified"])

            # Standalone air-gapped verifier script
            sys.path.insert(0, str(_REPO_ROOT / "scripts"))
            try:
                import foundry_verify  # type: ignore[import-not-found]

                out_buf = io.StringIO()
                with redirect_stdout(out_buf):
                    rc_v1 = foundry_verify.main(["proofpack", str(pp_path), "--keyring", str(keyring_file), "--expected-head", anchor["head"], "--require-promoted"])
                self.assertEqual(rc_v1, 0)
                with redirect_stdout(io.StringIO()):
                    rc_v2 = foundry_verify.main(["registry", str(runs_dir / "registry.json")])
                self.assertEqual(rc_v2, 0)
            finally:
                sys.path.pop(0)


class TestImportersComplianceAndMcp(unittest.TestCase):
    def test_all_four_eval_importers_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            pf_in = root / "promptfoo.json"
            pf_entries = [
                {
                    "id": f"pf-{i}",
                    "prompt": {"raw": f"Promptfoo case {i}"},
                    "expected_label": "PASS" if i % 2 == 0 else "FAIL",
                    "success": True,
                    "latencyMs": 14.0,
                    "family": f"fam-{i % 4}",
                }
                for i in range(12)
            ]
            pf_in.write_text(json.dumps({"results": {"results": pf_entries}}), encoding="utf-8")

            ins_in = root / "inspect.json"
            ins_in.write_text(json.dumps({
                "samples": [
                    {"id": f"ins-{i}", "input": f"Inspect sample {i}", "scores": {"acc": {"value": "C"}}, "total_time": 0.02}
                    for i in range(12)
                ]
            }), encoding="utf-8")

            lme_in = root / "lmeval.json"
            lme_in.write_text(json.dumps({
                "samples": {
                    "hellaswag_syn": [
                        {"doc_id": i, "doc": f"Doc {i}", "exact_match": 1}
                        for i in range(12)
                    ]
                }
            }), encoding="utf-8")

            mlf_in = root / "mlflow.json"
            mlf_in.write_text(json.dumps({
                "rows": [
                    {
                        "id": f"mlf-{i}",
                        "inputs": f"MLflow input {i}",
                        "ground_truth": "PASS" if i % 2 == 0 else "FAIL",
                        "prediction": "PASS" if i % 2 == 0 else "FAIL",
                        "baseline_prediction": "PASS",
                        "latency_ms": 9.5,
                    }
                    for i in range(12)
                ]
            }), encoding="utf-8")

            for fmt, src_file in (
                ("promptfoo", pf_in),
                ("inspect-ai", ins_in),
                ("lm-eval", lme_in),
                ("mlflow", mlf_in),
            ):
                out_dir = root / f"out-{fmt}"
                rc, res = _run_cli_json(["import", "--from", fmt, "--input", str(src_file), "--out-dir", str(out_dir)])
                self.assertEqual(rc, 0)
                self.assertEqual(res["status"], "IMPORTED")
                self.assertTrue(res["validation_ready"])
                spec = json.loads((out_dir / "spec.json").read_text(encoding="utf-8"))
                self.assertEqual(validate_task_dataset(load_jsonl(out_dir / "data.jsonl"), spec)["status"], "VALID")

            self.assertEqual(set(IMPORT_FORMATS), {"promptfoo", "inspect-ai", "lm-eval", "mlflow"})
            with self.assertRaises(ValueError):
                import_eval_results("unsupported", pf_in, out_dir=root / "bad")

    def test_compliance_profiles_html_dossier_and_policy_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"
            kp = generate_keypair(root / "comp.pem")
            keyring = [kp["public_key_hex"]]
            run = create_run(output_root=runs_dir, model_manifest=_SYNTHETIC_MANIFEST)
            sign_run(run["run_id"], private_key_path=root / "comp.pem", output_root=runs_dir)

            for prof in COMPLIANCE_PROFILES:
                rep = generate_audit_report(run["run_id"], output_root=runs_dir, keyring=keyring, compliance_profile=prof)
                self.assertTrue(rep["compliance"]["all_controls_passed"])
                self.assertTrue(rep["compliance"]["advisory_only"])

            html_out = root / "dossier.html"
            rc, rep_cli = _run_cli_json([
                "refinery",
                "report",
                run["run_id"],
                "--output-root",
                str(runs_dir),
                "--format",
                "html",
                "--compliance-profile",
                "eu-ai-act",
                "--out",
                str(html_out),
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(html_out.is_file())
            html_body = html_out.read_text(encoding="utf-8")
            self.assertIn("<!DOCTYPE html>", html_body)
            self.assertIn("eu-ai-act", html_body)

            # Policy-as-code check + HTML XSS escaping check
            xss_html = render_audit_html({"run_id": "<script>alert(1)</script>", "status": "BLOCKED", "promoted": False, "review_required": True})
            self.assertNotIn("<script>alert(1)</script>", xss_html)
            self.assertIn("&lt;script&gt;", xss_html)

            policy_pass = evaluate_policy_bundle(run, {"min_macro_f1": 0.85, "max_critical_fn_rate": 0.0, "allowed_data_origins": ["synthetic"]})
            self.assertTrue(policy_pass["passed"])
            policy_fail = evaluate_policy_bundle(run, {"min_macro_f1": 1.0, "require_zero_false_tool_exec": True})
            self.assertFalse(policy_fail["passed"])
            with self.assertRaises(ValueError):
                evaluate_compliance_profile(rep_cli, "unknown-profile")

    def test_mcp_adversarial_generator_and_read_only_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema_file = root / "tools.json"
            schema_file.write_text(json.dumps({
                "tools": [
                    {"name": "read_record", "description": "Read a synthetic record"},
                    {"name": "list_exceptions", "description": "List exception entries"},
                ]
            }), encoding="utf-8")

            suite_dir = root / "mcp_suite"
            sdk_gen = generate_mcp_adversarial_suite(schema_file, out_dir=root / "mcp_sdk_suite")
            self.assertEqual(sdk_gen["record_count"], 24)
            rc, gen = _run_cli_json([
                "mcp",
                "generate-suite",
                "--tools-schema",
                str(schema_file),
                "--out-dir",
                str(suite_dir),
            ])
            self.assertEqual(rc, 0)
            self.assertEqual(gen["record_count"], 24)
            spec = json.loads((suite_dir / "spec.json").read_text(encoding="utf-8"))
            self.assertEqual(validate_task_dataset(load_jsonl(suite_dir / "data.jsonl"), spec)["status"], "VALID")

            # Run a real Refinery task evaluation on the generated adversarial suite
            runs_dir = root / "runs"
            task_run = create_task_run(
                task_spec=spec,
                dataset_records=load_jsonl(suite_dir / "data.jsonl"),
                baseline_predictions=load_jsonl(suite_dir / "baseline.jsonl"),
                model_predictions=load_jsonl(suite_dir / "model.jsonl"),
                output_root=runs_dir,
                model_manifest=_SYNTHETIC_MANIFEST,
            )

            srv = FoundryMcpServer(default_output_root=runs_dir)
            init_resp = srv.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
            assert init_resp is not None
            self.assertEqual(init_resp["result"]["serverInfo"]["name"], "hardonian-foundry-governance")

            list_resp = srv.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            assert list_resp is not None
            tool_names = {t["name"] for t in list_resp["result"]["tools"]}
            self.assertIn("foundry_inspect_run", tool_names)
            self.assertNotIn("foundry_sign_run", tool_names)

            call_resp = srv.handle_request({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "foundry_inspect_run", "arguments": {"run_id": task_run["run_id"]}},
            })
            assert call_resp is not None
            self.assertFalse(call_resp["result"]["isError"])
            self.assertEqual(call_resp["result"]["structuredContent"]["provenance_status"], "PROVENANCE_VERIFIED")

            bad_tool = srv.handle_request({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "foundry_sign_run", "arguments": {}},
            })
            assert bad_tool is not None
            self.assertIn("error", bad_tool)

            # stdio loop test
            stdin = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 9, "method": "ping"}) + "\n")
            stdout = io.StringIO()
            self.assertEqual(srv.serve_stdio(stdin=stdin, stdout=stdout), 0)
            self.assertEqual(json.loads(stdout.getvalue())["id"], 9)


class TestTrustAdaptersCliExposureAndWorkflows(unittest.TestCase):
    def test_trust_adapters_and_signed_consent_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kp = generate_keypair(root / "kms.pem")
            pub = kp["public_key_hex"]

            # Subprocess KMS signer reading payload hex from stdin and signing with temp PEM
            helper_script = root / "kms_helper.py"
            helper_script.write_text(
                "\n".join([
                    "import sys",
                    f"sys.path.insert(0, {str(_REPO_ROOT / 'src')!r})",
                    "from hardonian_foundry.signing import sign",
                    "payload = bytes.fromhex(sys.stdin.read().strip())",
                    f"print(sign(payload, {str(root / 'kms.pem')!r}))",
                ]),
                encoding="utf-8",
            )
            kms = SubprocessKmsSignerProvider(public_key_hex=pub, command=[sys.executable, str(helper_script)])
            self.assertIsInstance(kms, SignerProvider)
            runs_dir = root / "runs"
            run = create_run(output_root=runs_dir, model_manifest=_SYNTHETIC_MANIFEST)
            signed = sign_run(run["run_id"], signer=kms, output_root=runs_dir)
            self.assertEqual(signed["public_key_hex"], pub)

            # RFC 3161 TSA sink + Object Lock sink
            tsa = Rfc3161TsaAnchorSink(
                "https://tsa.example.invalid/ts",
                transport=lambda url, body, timeout: {
                    "message_imprint": json.loads(body.decode("utf-8"))["message_imprint"],
                    "gen_time": "2026-11-24T19:00:00Z",
                    "serial_number": "tsa-001",
                },
            )
            self.assertIsInstance(tsa, AnchorSink)
            anc_tsa = anchor_run(run["run_id"], output_root=runs_dir, sink=tsa)
            self.assertEqual(anc_tsa["sink_receipt"]["sink_type"], "rfc3161_tsa")

            client = _InMemoryObjectLockClient()
            ol_sink = ObjectLockAnchorSink(bucket="foundry-worm", client=client)
            anc_ol = anchor_run(run["run_id"], output_root=runs_dir, sink=ol_sink)
            self.assertEqual(anc_ol["sink_receipt"]["object_lock_mode"], "COMPLIANCE")
            # Re-anchoring same head succeeds idempotently; conflicting head fails closed
            anchor_run(run["run_id"], output_root=runs_dir, sink=ol_sink)
            with self.assertRaises(ValueError):
                ol_sink.publish({**anc_ol, "count": 999})  # type: ignore[arg-type]

            # Signed consent receipt
            att = {"tier": "redacted_eval", "attestation_id": "att-001", "approved_by": "irb", "approved_at": "2026-11-24T00:00:00Z", "redacted": True}
            receipt = create_consent_receipt(att, private_key_path=root / "kms.pem")
            self.assertTrue(verify_consent_receipt(receipt, keyring=[pub])["verified"])
            tampered_receipt = {**receipt, "attestation": {**att, "redacted": False}}
            self.assertFalse(verify_consent_receipt(tampered_receipt, keyring=[pub])["verified"])

    def test_provider_presets_and_refinery_hidden_sdk_cli(self) -> None:
        self.assertEqual(resolve_provider_origin(preset="ollama"), PROVIDER_PRESETS["ollama"])
        with self.assertRaises(ValueError):
            resolve_provider_origin(preset="unknown-engine")
        with self.assertRaises(ValueError):
            resolve_provider_origin(origin="http://127.0.0.1:8000", preset="vllm")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = root / "proj"
            init_project(proj, template="trajectory", name="cli-sdk-check")

            # scan-pii CLI clean vs dirty
            rc_clean, res_clean = _run_cli_json(["refinery", "scan-pii", "--data", str(proj / "data.jsonl")])
            self.assertEqual(rc_clean, 0)
            self.assertTrue(res_clean["clean"])

            dirty_file = root / "dirty.json"
            dirty_file.write_text(json.dumps([{"id": "d1", "text": "contact alice@example.com"}]), encoding="utf-8")
            rc_dirty, res_dirty = _run_cli_json(["refinery", "scan-pii", "--data", str(dirty_file)])
            self.assertEqual(rc_dirty, 1)
            self.assertFalse(res_dirty["clean"])

            # eval-trajectories CLI
            rc_traj, res_traj = _run_cli_json([
                "refinery",
                "eval-trajectories",
                "--trajectories",
                str(proj / "trajectories.jsonl"),
                "--allowed-tools",
                "read_record,list_exceptions",
            ])
            self.assertEqual(rc_traj, 0)
            self.assertTrue(res_traj["passed"])

            # kappa CLI (cohen + fleiss)
            cohen_file = root / "cohen.json"
            cohen_file.write_text(json.dumps({"rater_a": ["A", "B", "A", "B"], "rater_b": ["A", "B", "A", "B"]}), encoding="utf-8")
            rc_k1, res_k1 = _run_cli_json(["refinery", "kappa", "--mode", "cohen", "--input", str(cohen_file)])
            self.assertEqual(rc_k1, 0)
            self.assertEqual(res_k1["kappa"], 1.0)

            fleiss_file = root / "fleiss.json"
            fleiss_file.write_text(json.dumps([[3, 0], [0, 3], [3, 0]]), encoding="utf-8")
            rc_k2, res_k2 = _run_cli_json(["refinery", "kappa", "--mode", "fleiss", "--input", str(fleiss_file)])
            self.assertEqual(rc_k2, 0)
            self.assertEqual(res_k2["kappa"], 1.0)

            # dataset-card CLI + SDK
            sdk_card = generate_dataset_card(proj / "spec.json")
            self.assertEqual(sdk_card["validation_status"], "VALID")
            card_md = root / "card.md"
            rc_dc, res_dc = _run_cli_json([
                "refinery",
                "dataset-card",
                "--task-spec",
                str(proj / "spec.json"),
                "--out",
                str(card_md),
            ])
            self.assertEqual(rc_dc, 0)
            self.assertTrue(res_dc["pii_clean"])
            self.assertTrue(card_md.is_file())
            self.assertIn("Foundry Dataset Card", card_md.read_text(encoding="utf-8"))

            # Verify workflows & action YAML parse cleanly
            action_cfg = yaml.safe_load((_REPO_ROOT / "action.yml").read_text(encoding="utf-8"))
            self.assertEqual(action_cfg["runs"]["using"], "composite")
            rel_cfg = yaml.safe_load((_REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8"))
            self.assertIn("pypi-publish", rel_cfg["jobs"])

            # Verify public exports
            for sym in (
                "COMPLIANCE_PROFILES",
                "IMPORT_FORMATS",
                "PROVIDER_PRESETS",
                "TEMPLATES",
                "FoundryMcpServer",
                "ObjectLockAnchorSink",
                "Rfc3161TsaAnchorSink",
                "SubprocessKmsSignerProvider",
                "create_consent_receipt",
                "evaluate_compliance_profile",
                "evaluate_policy_bundle",
                "export_schemas",
                "generate_dataset_card",
                "generate_mcp_adversarial_suite",
                "get_schema",
                "import_eval_results",
                "init_project",
                "list_schemas",
                "render_audit_html",
                "resolve_provider_origin",
                "verify_consent_receipt",
            ):
                self.assertIn(sym, hf.__all__)
                self.assertTrue(hasattr(hf, sym))
