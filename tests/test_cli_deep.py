"""Deep unit tests for hardonian_foundry.cli covering error formatting, argument validation,
and uncovered CLI subcommands."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from hardonian_foundry import cli
from hardonian_foundry.cli import _emit_error, _load_json_file


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


class CliDeepTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_json_file_errors(self):
        with self.assertRaises(ValueError):
            _load_json_file(str(self.root / "missing.json"), "test file")

        corrupt = self.root / "corrupt.json"
        corrupt.write_text("invalid json content", encoding="utf-8")
        with self.assertRaises(ValueError):
            _load_json_file(str(corrupt), "test file")

    def test_emit_error_text_and_json(self):
        err_out = io.StringIO()
        with redirect_stderr(err_out):
            rc = _emit_error("plain error message", json_errors=False, exit_code=2)
        self.assertEqual(rc, 2)
        self.assertIn("plain error message", err_out.getvalue())

    def test_cli_tenant_context_without_tenant_id_fails(self):
        rc, _, err = _run_cli(["refinery", "list", "--request-id", "req-123", "--output-root", str(self.root)])
        self.assertEqual(rc, 2)
        self.assertIn("--tenant-id is required", err)

    def test_cli_provider_probe_missing_args(self):
        rc, _, err = _run_cli(["provider", "probe"])
        self.assertEqual(rc, 2)
        self.assertIn("required: --origin, --model", err)

    def test_cli_provider_predict_missing_args(self):
        rc, _, err = _run_cli(["provider", "predict", "--origin", "http://127.0.0.1:8000", "--model", "m"])
        self.assertEqual(rc, 2)
        self.assertIn("requires either --task-spec or both --data and --labels", err)

    def test_cli_refinery_validate_and_preflight_with_data_file(self):
        from hardonian_foundry.baseline import dataset

        data_file = self.root / "data.json"
        data_file.write_text(json.dumps(dataset()), encoding="utf-8")

        # Validate with --data
        rc_val, out_val, _ = _run_cli(["refinery", "validate-dataset", "--data", str(data_file)])
        self.assertEqual(rc_val, 0)
        self.assertEqual(json.loads(out_val)["status"], "VALID")

        # Preflight with --data
        rc_pre, out_pre, _ = _run_cli(["refinery", "preflight", "--data", str(data_file), "--output-root", str(self.root)])
        self.assertEqual(rc_pre, 0)
        self.assertEqual(json.loads(out_pre)["status"], "PASSED")

        # Run with --data
        rc_run, out_run, _ = _run_cli(["refinery", "run", "--data", str(data_file), "--output-root", str(self.root)])
        self.assertEqual(rc_run, 0)
        self.assertIn("run_id", json.loads(out_run))

    def test_cli_task_mode_run_missing_predictions_args(self):
        spec_file = self.root / "spec.json"
        spec_file.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")

        rc, _, err = _run_cli(["refinery", "run", "--task-spec", str(spec_file), "--output-root", str(self.root)])
        self.assertEqual(rc, 2)
        self.assertIn("task-mode run requires --baseline-predictions and --model-predictions", err)

    def test_cli_promote_blocked_errors(self):
        from hardonian_foundry.refinery_runs import create_run

        run = create_run(output_root=self.root)
        rid = run["run_id"]

        # Promote unsigned run without --json-errors
        rc1, _, err1 = _run_cli(["refinery", "promote", rid, "--output-root", str(self.root)])
        self.assertEqual(rc1, 2)
        self.assertIn("Promotion blocked", err1)

        # Promote unsigned run with --json-errors
        rc2, _, err2 = _run_cli(["--json-errors", "refinery", "promote", rid, "--output-root", str(self.root)])
        self.assertEqual(rc2, 2)
        self.assertIn("Promotion blocked", err2)
        self.assertIn('"exit_code": 2', err2)

    def test_doctor_verbose_and_anchor_args(self):
        # doctor --verbose
        rc, out, _ = _run_cli(["doctor", "--verbose"])
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertEqual(data["status"], "ok")
        self.assertIn("platform", data)
        self.assertIn("default_runs_dir", data)

        # anchor expectation resolution with --anchor
        anchor_file = self.root / "anchor.json"
        anchor_file.write_text(json.dumps({"head": "a" * 64, "count": 5}), encoding="utf-8")
        args = cli._build_parser().parse_args(["refinery", "inspect", "run-1", "--anchor", str(anchor_file)])
        head, count = cli._resolve_anchor_expectations(args)
        self.assertEqual(head, "a" * 64)
        self.assertEqual(count, 5)

        # anchor expectation invalid json
        bad_anchor = self.root / "bad_anchor.json"
        bad_anchor.write_text("[1, 2, 3]", encoding="utf-8")
        args_bad = cli._build_parser().parse_args(["refinery", "inspect", "run-1", "--anchor", str(bad_anchor)])
        with self.assertRaises(ValueError):
            cli._resolve_anchor_expectations(args_bad)

    def test_registry_commands(self):
        # registry list
        rc_list, out_list, _ = _run_cli(["registry", "list", "--output-root", str(self.root)])
        self.assertEqual(rc_list, 0)
        self.assertIn("models", json.loads(out_list))

        # registry promote without --model-id
        rc_missing, _, err_missing = _run_cli(["registry", "promote"])
        self.assertEqual(rc_missing, 2)
        self.assertIn("Promotion blocked", err_missing)

    def test_refinery_artifact_cli(self):
        sample = self.root / "blob.bin"
        sample.write_bytes(b"cli-artifact-content")

        # artifact add
        rc_add, out_add, _ = _run_cli(["refinery", "artifact", "add", str(sample), "--output-root", str(self.root)])
        self.assertEqual(rc_add, 0)
        art = json.loads(out_add)
        sha = art["sha256"]

        # artifact verify
        rc_v, out_v, _ = _run_cli(["refinery", "artifact", "verify", sha, "--output-root", str(self.root)])
        self.assertEqual(rc_v, 0)
        self.assertTrue(json.loads(out_v)["verified"])

        # artifact list
        rc_l, out_l, _ = _run_cli(["refinery", "artifact", "list", "--output-root", str(self.root)])
        self.assertEqual(rc_l, 0)
        self.assertEqual(json.loads(out_l)["count"], 1)


if __name__ == "__main__":
    unittest.main()
