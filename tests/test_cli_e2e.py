"""End-to-end CLI tests covering all subcommands, exit codes, task-mode lifecycle,
anchor checks, bundle export/verification, artifact store, provider predict, and
case-study replay."""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

from hardonian_foundry import __version__, cli
from hardonian_foundry.refinery import verify_bundle
from hardonian_foundry.task_eval import split_task_dataset


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    labels_by_prompt: ClassVar[dict[str, str]] = {}

    def log_message(self, *_args):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        req = json.loads(raw.decode("utf-8"))
        user_msg = req["messages"][-1]["content"]
        if "reconcil" in user_msg.lower() or "synthetic-provider" in user_msg or "expected_minor" in user_msg:
            content = json.dumps({"category": "amount_mismatch"})
        else:
            label = self.labels_by_prompt.get(user_msg, "GAMMA")
            content = json.dumps({"label": label})
        body = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _run_cli_inproc(argv: list[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


def _run_cli_subproc(argv: list[str]) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": "src"}
    return subprocess.run(
        [sys.executable, "-m", "hardonian_foundry.cli", *argv],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


class CliE2ETest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runs = self.root / "runs"

    def tearDown(self):
        self.tmp.cleanup()

    def _write_task_fixture(self) -> dict[str, Any]:
        task_dir = self.root / "task"
        task_dir.mkdir(parents=True, exist_ok=True)
        labels = ["ALPHA", "BETA", "GAMMA", "UNDETERMINED"]
        records = []
        for fam_idx in range(8):
            for item_idx in range(4):
                lbl = labels[(fam_idx + item_idx) % 4]
                records.append({
                    "id": f"rec-{fam_idx}-{item_idx}",
                    "prompt": f"Exception case {fam_idx}-{item_idx}",
                    "gold_label": lbl,
                    "critical": lbl == "ALPHA",
                    "family": f"fam_{fam_idx}",
                })
        dataset_file = task_dir / "dataset.jsonl"
        dataset_file.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

        spec = {
            "schema_version": "1.0",
            "task_id": "cli-e2e-task",
            "description": "End-to-end CLI verification task fixture.",
            "dataset_path": "dataset.jsonl",
            "labels": labels,
            "thresholds": {"macro_f1_min": 0.8},
            "baseline_must_improve": True,
            "split_seed": 42,
            "model": {
                "base_model_id": "Qwen/Qwen2.5-Coder-0.5B",
                "revision": "0123456789abcdef",
                "adapter_sha256": "b" * 64,
                "adapter_name": "cli-e2e-adapter",
            },
        }
        spec_file = task_dir / "spec.json"
        spec_file.write_text(json.dumps(spec, indent=2), encoding="utf-8")

        test_rows = split_task_dataset(records, spec)["test"]
        base_preds = []
        model_preds = []
        for idx, row in enumerate(test_rows):
            gold = row["gold_label"]
            model_preds.append({"id": row["id"], "label": gold})
            if idx % 2 == 0:
                wrong = labels[(labels.index(gold) + 1) % len(labels)]
                base_preds.append({"id": row["id"], "label": wrong})
            else:
                base_preds.append({"id": row["id"], "label": gold})

        base_file = task_dir / "base_preds.jsonl"
        model_file = task_dir / "model_preds.jsonl"
        base_file.write_text("".join(json.dumps(p) + "\n" for p in base_preds), encoding="utf-8")
        model_file.write_text("".join(json.dumps(p) + "\n" for p in model_preds), encoding="utf-8")

        manifest = {
            "name": "cli-e2e-adapter",
            "license_id": "Apache-2.0",
            "provenance": "synthetic",
            "data_origin": "synthetic",
        }
        manifest_file = task_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return {
            "spec": spec_file,
            "dataset": dataset_file,
            "base": base_file,
            "model": model_file,
            "manifest": manifest_file,
            "test_rows": test_rows,
        }

    def test_version_doctor_demo_eval_policy_demo(self):
        ver = _run_cli_subproc(["--version"])
        self.assertEqual(ver.returncode, 0)
        self.assertIn(__version__, ver.stdout)

        rc, out, _ = _run_cli_inproc(["doctor", "--verbose"])
        self.assertEqual(rc, 0)
        doc = json.loads(out)
        self.assertEqual(doc["version"], __version__)
        self.assertEqual(doc["runtime"], "offline_cpu")
        self.assertIn("signing_backend_available", doc)

        for cmd in ("demo", "eval", "policy-demo"):
            rc, out, _ = _run_cli_inproc([cmd])
            self.assertEqual(rc, 0)
            self.assertIsInstance(json.loads(out), dict)

    def test_json_errors_and_exit_codes(self):
        # Usage error with --json-errors emits structured stderr JSON and returns exit 2
        rc, _, err = _run_cli_inproc(["--json-errors", "refinery", "unknown-cmd"])
        self.assertEqual(rc, 2)
        payload = json.loads(err)
        self.assertEqual(payload["exit_code"], 2)
        self.assertIn("error", payload)

        # Missing run inspect with --json-errors
        rc2, _, err2 = _run_cli_inproc([
            "refinery", "inspect", "missing-run", "--output-root", str(self.runs), "--json-errors",
        ])
        self.assertEqual(rc2, 2)
        self.assertEqual(json.loads(err2)["exit_code"], 2)

        # Failed preflight returns exit code 1
        bad_manifest = self.root / "bad_manifest.json"
        bad_manifest.write_text(json.dumps({
            "name": "m",
            "license_id": "GPL-3.0",
            "provenance": "synthetic",
            "data_origin": "synthetic",
        }), encoding="utf-8")
        rc3, out3, _ = _run_cli_inproc([
            "refinery", "preflight", "--manifest", str(bad_manifest), "--output-root", str(self.runs),
        ])
        self.assertEqual(rc3, 1)
        self.assertEqual(json.loads(out3)["status"], "FAILED")

    def test_provider_probe_and_predict_e2e(self):
        fix = self._write_task_fixture()
        _MockOpenAIHandler.labels_by_prompt = {
            row["prompt"]: row["gold_label"] for row in fix["test_rows"]
        }
        server = ThreadingHTTPServer(("127.0.0.1", 0), _MockOpenAIHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        origin = f"http://127.0.0.1:{server.server_port}"
        try:
            rc_probe, out_probe, _ = _run_cli_inproc(["provider", "--origin", origin, "--model", "local"])
            self.assertEqual(rc_probe, 0)
            self.assertEqual(json.loads(out_probe)["status"], "REVIEW_REQUIRED")

            pred_out = self.root / "predicted.jsonl"
            rc_pred, out_pred, _ = _run_cli_inproc([
                "provider",
                "predict",
                "--origin",
                origin,
                "--model",
                "local",
                "--task-spec",
                str(fix["spec"]),
                "--out",
                str(pred_out),
            ])
            self.assertEqual(rc_pred, 0)
            summary = json.loads(out_pred)
            self.assertEqual(summary["status"], "COMPLETED")
            self.assertEqual(summary["fallback_count"], 0)
            self.assertTrue(pred_out.is_file())
        finally:
            server.shutdown()
            server.server_close()

    def test_task_mode_cli_anchor_list_diff_bundle_artifact_and_registry(self):
        fix = self._write_task_fixture()

        # validate-dataset and preflight with --task-spec
        rc_val, out_val, _ = _run_cli_inproc(["refinery", "validate-dataset", "--task-spec", str(fix["spec"])])
        self.assertEqual(rc_val, 0)
        self.assertEqual(json.loads(out_val)["status"], "VALID")

        rc_pre, out_pre, _ = _run_cli_inproc([
            "refinery",
            "preflight",
            "--task-spec",
            str(fix["spec"]),
            "--manifest",
            str(fix["manifest"]),
            "--output-root",
            str(self.runs),
        ])
        self.assertEqual(rc_pre, 0)
        self.assertEqual(json.loads(out_pre)["status"], "PASSED")

        # Create two runs cleanly via CLI
        rc_r1, out_r1, _ = _run_cli_inproc([
            "refinery",
            "run",
            "--task-spec",
            str(fix["spec"]),
            "--baseline-predictions",
            str(fix["base"]),
            "--model-predictions",
            str(fix["model"]),
            "--output-root",
            str(self.runs),
        ])
        self.assertEqual(rc_r1, 0)
        rid1 = json.loads(out_r1)["run_id"]

        rc_r2, out_r2, _ = _run_cli_inproc([
            "refinery",
            "run",
            "--task-spec",
            str(fix["spec"]),
            "--baseline-predictions",
            str(fix["base"]),
            "--model-predictions",
            str(fix["model"]),
            "--output-root",
            str(self.runs),
        ])
        self.assertEqual(rc_r2, 0)
        rid2 = json.loads(out_r2)["run_id"]

        # anchor + inspect --anchor
        anchor_file = self.root / "anchor.json"
        rc_anc, out_anc, _ = _run_cli_inproc([
            "refinery",
            "anchor",
            rid2,
            "--output-root",
            str(self.runs),
            "--out",
            str(anchor_file),
        ])
        self.assertEqual(rc_anc, 0)
        anchor = json.loads(out_anc)
        self.assertEqual(anchor["count"], 2)

        rc_insp, out_insp, _ = _run_cli_inproc([
            "refinery",
            "inspect",
            rid2,
            "--output-root",
            str(self.runs),
            "--anchor",
            str(anchor_file),
        ])
        self.assertEqual(rc_insp, 0)
        self.assertEqual(json.loads(out_insp)["provenance_status"], "PROVENANCE_VERIFIED")

        # Wrong expected count returns exit 1
        rc_bad_anc, out_bad_anc, _ = _run_cli_inproc([
            "refinery",
            "inspect",
            rid2,
            "--output-root",
            str(self.runs),
            "--expected-head",
            anchor["head"],
            "--expected-count",
            "999",
        ])
        self.assertEqual(rc_bad_anc, 1)
        self.assertEqual(json.loads(out_bad_anc)["provenance_status"], "TAMPERED_ANCHOR_MISMATCH")

        # list and diff
        rc_list, out_list, _ = _run_cli_inproc([
            "refinery", "list", "--output-root", str(self.runs), "--verify",
        ])
        self.assertEqual(rc_list, 0)
        self.assertEqual(json.loads(out_list)["count"], 2)

        rc_diff, out_diff, _ = _run_cli_inproc([
            "refinery", "diff", rid1, rid2, "--output-root", str(self.runs),
        ])
        self.assertEqual(rc_diff, 0)
        diff = json.loads(out_diff)
        self.assertTrue(diff["same_dataset"])
        self.assertEqual(diff["metrics_delta"]["macro_f1_delta"], 0.0)

        # artifact add / verify / list
        blob_src = self.root / "adapter.bin"
        blob_src.write_bytes(b"e2e-artifact-bytes")
        rc_aa, out_aa, _ = _run_cli_inproc([
            "refinery", "artifact", "add", str(blob_src), "--output-root", str(self.runs), "--run-id", rid2,
        ])
        self.assertEqual(rc_aa, 0)
        sha = json.loads(out_aa)["sha256"]

        rc_av, out_av, _ = _run_cli_inproc([
            "refinery", "artifact", "verify", sha, "--output-root", str(self.runs),
        ])
        self.assertEqual(rc_av, 0)
        self.assertTrue(json.loads(out_av)["verified"])

        rc_al, out_al, _ = _run_cli_inproc([
            "refinery", "artifact", "list", "--output-root", str(self.runs), "--run-id", rid2,
        ])
        self.assertEqual(rc_al, 0)
        self.assertEqual(json.loads(out_al)["count"], 1)

        # export-bundle zip & verify_bundle
        bundle_zip = self.root / "export.zip"
        rc_exp, out_exp, _ = _run_cli_inproc([
            "refinery",
            "export-bundle",
            rid2,
            "--output-root",
            str(self.runs),
            "--out",
            str(bundle_zip),
            "--format",
            "zip",
        ])
        self.assertEqual(rc_exp, 0)
        self.assertEqual(json.loads(out_exp)["status"], "EXPORTED")
        self.assertEqual(verify_bundle(bundle_zip)["status"], "VERIFIED")

        # registry list + attach-license
        rc_reg_list, out_rl, _ = _run_cli_inproc(["registry", "list", "--output-root", str(self.runs)])
        self.assertEqual(rc_reg_list, 0)
        self.assertEqual(len(json.loads(out_rl)["runs"]), 2)

        rc_att, out_att, _ = _run_cli_inproc([
            "refinery", "attach-license", rid2, "--manifest", str(fix["manifest"]), "--output-root", str(self.runs),
        ])
        self.assertEqual(rc_att, 0)
        self.assertEqual(json.loads(out_att)["decision"], "ALLOWED")

        # keygen supports both --out and --key-file aliases
        k1 = self.root / "k1.pem"
        k2 = self.root / "k2.pem"
        rc_kg1, out_kg1, _ = _run_cli_inproc(["refinery", "keygen", "--out", str(k1)])
        rc_kg2, out_kg2, _ = _run_cli_inproc(["refinery", "keygen", "--key-file", str(k2)])
        self.assertEqual(rc_kg1, 0)
        self.assertEqual(rc_kg2, 0)
        pub1 = json.loads(out_kg1)["public_key_hex"]
        pub2 = json.loads(out_kg2)["public_key_hex"]

        keyring_file = self.root / "keyring.json"
        keyring_file.write_text(json.dumps({
            "trusted_keys": [
                {"public_key_hex": pub1, "role": "maintainer"},
                {"public_key_hex": pub2, "role": "security"},
            ],
            "required_signatures": 2,
            "required_roles": ["maintainer", "security"],
        }), encoding="utf-8")

        # Sign with key 1 then cosign with key 2, then promote via both refinery promote and registry promote
        rc_s1, _, _ = _run_cli_inproc([
            "refinery", "sign", rid2, "--key-file", str(k1), "--output-root", str(self.runs),
        ])
        self.assertEqual(rc_s1, 0)
        rc_s2, _, _ = _run_cli_inproc([
            "refinery", "sign", rid2, "--key-file", str(k2), "--output-root", str(self.runs), "--cosign",
        ])
        self.assertEqual(rc_s2, 0)

        rc_prom, out_prom, _ = _run_cli_inproc([
            "refinery", "promote", rid2, "--output-root", str(self.runs), "--keyring", str(keyring_file),
        ])
        self.assertEqual(rc_prom, 0)
        self.assertTrue(json.loads(out_prom)["promoted"])

        rc_rprom, out_rprom, _ = _run_cli_inproc([
            "registry", "promote", rid2, "--output-root", str(self.runs), "--keyring", str(keyring_file),
        ])
        self.assertEqual(rc_rprom, 0)
        self.assertTrue(json.loads(out_rprom)["promoted"])

    def test_committed_case_study_signed_run_replay(self):
        case_runs = Path("examples/case-study-exception-codes/task/runs")
        case_keyring = Path("examples/case-study-exception-codes/task/keyring.json")
        run_id = "run-20261111T031107Z-6a0895d1"
        if not (case_runs / run_id / "run.json").is_file():
            self.skipTest("case-study fixture not present")

        rc_insp, out_insp, _ = _run_cli_inproc([
            "refinery", "inspect", run_id, "--output-root", str(case_runs),
        ])
        self.assertEqual(rc_insp, 0)
        self.assertEqual(json.loads(out_insp)["provenance_status"], "PROVENANCE_VERIFIED")

        rc_prom, out_prom, _ = _run_cli_inproc([
            "refinery", "promote", run_id, "--output-root", str(case_runs), "--keyring", str(case_keyring),
        ])
        self.assertEqual(rc_prom, 0)
        self.assertTrue(json.loads(out_prom)["promoted"])


if __name__ == "__main__":
    unittest.main()
