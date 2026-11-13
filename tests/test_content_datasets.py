"""Tests for Sprint 1-4 synthetic dataset generators, case studies, and baselines."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from hardonian_foundry import task_eval
from hardonian_foundry.synthetic_policy import assess

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(rel_path: str, name: str) -> ModuleType:
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    return env


class ExceptionCodesDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gen = _load_module("examples/case-study-exception-codes/generate_dataset.py", "exc_gen")
        cls.baseline = _load_module("examples/case-study-exception-codes/baseline_predict.py", "exc_base")

    def test_standard_profile_matches_committed_dataset_and_is_deterministic(self) -> None:
        committed = task_eval.load_jsonl(REPO_ROOT / "examples/case-study-exception-codes/task/data.jsonl")
        first = self.gen.generate_records(profile="standard", seed=self.gen.BASE_SEED)
        second = self.gen.generate_records(profile="standard", seed=self.gen.BASE_SEED)
        self.assertEqual(first, second)
        self.assertEqual(first, committed)
        self.assertEqual(len(first), 1152)

        spec = self.gen.build_spec(profile="standard", seed=self.gen.BASE_SEED)
        self.assertEqual(task_eval.validate_spec(spec), [])
        val = task_eval.validate_task_dataset(first, spec)
        self.assertEqual(val["status"], "VALID")
        self.assertTrue(val["family_disjoint"])

    def test_adversarial_noisy_profile_valid_deterministic_and_challenges_baseline(self) -> None:
        first = self.gen.generate_records(profile="adversarial-noisy", seed=self.gen.BASE_SEED)
        second = self.gen.generate_records(profile="adversarial-noisy", seed=self.gen.BASE_SEED)
        other_seed = self.gen.generate_records(profile="adversarial-noisy", seed=self.gen.BASE_SEED + 1)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other_seed)
        self.assertEqual(len(first), 1152)

        spec = self.gen.build_spec(profile="adversarial-noisy", seed=self.gen.BASE_SEED)
        val = task_eval.validate_task_dataset(first, spec)
        self.assertEqual(val["status"], "VALID")

        prompts = [row["prompt"] for row in first]
        self.assertEqual(len(prompts), len(set(prompts)))
        joined = "\n".join(prompts)
        for token in ("ccy", "amt", "dup", "reconcil", "not a"):
            self.assertIn(token, joined)

        splits = task_eval.split_task_dataset(first, spec)
        for split_name, split_rows in splits.items():
            self.assertEqual({r["gold_label"] for r in split_rows}, set(spec["labels"]), split_name)
        train_families = {r["family"] for r in splits["train"]}
        val_families = {r["family"] for r in splits["validation"]}
        test_families = {r["family"] for r in splits["test"]}
        self.assertFalse(train_families & val_families)
        self.assertFalse(train_families & test_families)
        self.assertFalse(val_families & test_families)

        preds = [{"id": r["id"], "label": self.baseline.predict(r["prompt"])} for r in splits["test"]]
        scored = task_eval.score_predictions(splits["test"], preds, spec)
        self.assertGreater(scored["critical_false_negative_rate"], 0.0)

    def test_imbalanced_profile_has_rare_critical_rate_and_full_coverage(self) -> None:
        rows = self.gen.generate_records(profile="imbalanced", seed=self.gen.BASE_SEED)
        self.assertEqual(len(rows), 1200)
        crit_ratio = sum(1 for r in rows if r["critical"]) / len(rows)
        self.assertGreaterEqual(crit_ratio, 0.03)
        self.assertLessEqual(crit_ratio, 0.05)

        spec = self.gen.build_spec(profile="imbalanced", seed=self.gen.BASE_SEED)
        self.assertEqual(task_eval.validate_task_dataset(rows, spec)["status"], "VALID")
        splits = task_eval.split_task_dataset(rows, spec)
        for split_name, split_rows in splits.items():
            self.assertEqual({r["gold_label"] for r in split_rows}, set(spec["labels"]), split_name)
            self.assertTrue(any(r["critical"] for r in split_rows), split_name)

    def test_generator_cli_smoke(self) -> None:
        script = str(REPO_ROOT / "examples/case-study-exception-codes/generate_dataset.py")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "noisy"
            proc = subprocess.run(
                [sys.executable, script, "--out-dir", str(out), "--profile", "adversarial-noisy", "--seed", "1234"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue((out / "data.jsonl").is_file())
            self.assertTrue((out / "spec.json").is_file())

    def test_generator_rejects_invalid_profile_or_count(self) -> None:
        with self.assertRaises(ValueError):
            self.gen.generate_records(profile="unknown")
        with self.assertRaises(ValueError):
            self.gen.generate_records(records_per_archetype=0)


class AgentToolGateCaseStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gen = _load_module("examples/case-study-agent-tool-gate/generate_dataset.py", "tool_gen")
        cls.baseline = _load_module("examples/case-study-agent-tool-gate/baseline_predict.py", "tool_base")
        cls.gate = _load_module("examples/case-study-agent-tool-gate/gate_predict.py", "tool_gate")

    def test_determinism_zero_leakage_and_attack_coverage(self) -> None:
        first = self.gen.generate_records(seed=self.gen.BASE_SEED)
        second = self.gen.generate_records(seed=self.gen.BASE_SEED)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 720)

        spec = self.gen.build_spec(seed=self.gen.BASE_SEED)
        self.assertEqual(task_eval.validate_spec(spec), [])
        self.assertEqual(task_eval.validate_task_dataset(first, spec)["status"], "VALID")
        self.assertEqual(
            spec["model"]["adapter_sha256"],
            self.gate.sha256_file(REPO_ROOT / "examples/case-study-agent-tool-gate/gate_predict.py"),
        )

        prompts = "\n".join(r["prompt"] for r in first)
        for marker in ("../../etc/passwd", "bypass_gate", "Ignore previous safety policies", "; cat /etc/shadow"):
            self.assertIn(marker, prompts)

        splits = task_eval.split_task_dataset(first, spec)
        train_families = {r["family"] for r in splits["train"]}
        val_families = {r["family"] for r in splits["validation"]}
        test_families = {r["family"] for r in splits["test"]}
        self.assertFalse(train_families & val_families)
        self.assertFalse(train_families & test_families)
        self.assertFalse(val_families & test_families)

        test = splits["test"]
        self.assertEqual({r["gold_label"] for r in test}, set(spec["labels"]))

        base_preds = [{"id": r["id"], "label": self.baseline.predict(r["prompt"])} for r in test]
        gate_preds = [{"id": r["id"], "label": self.gate.predict(r["prompt"])} for r in test]
        base_score = task_eval.score_predictions(test, base_preds, spec)
        gate_score = task_eval.score_predictions(test, gate_preds, spec)

        self.assertGreater(base_score["critical_false_negative_rate"], 0.0)
        self.assertEqual(gate_score["critical_false_negative_rate"], 0.0)
        self.assertEqual(gate_score["macro_f1"], 1.0)
        self.assertGreater(gate_score["macro_f1"], base_score["macro_f1"])

    def test_gate_edge_cases_and_provenance_mismatch(self) -> None:
        self.assertEqual(self.gate.predict("unparseable prompt"), "DENY")
        self.assertEqual(self.gate.predict("Tool Proposal: {not-json}"), "DENY")
        with tempfile.TemporaryDirectory() as td:
            task_dir = Path(td)
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "examples/case-study-agent-tool-gate/generate_dataset.py"),
                    "--out-dir",
                    str(task_dir),
                ],
                env=_subprocess_env(),
                capture_output=True,
                check=True,
            )
            spec = json.loads((task_dir / "spec.json").read_text(encoding="utf-8"))
            spec["model"]["adapter_sha256"] = "f" * 64
            (task_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "examples/case-study-agent-tool-gate/gate_predict.py"),
                    "--task-dir",
                    str(task_dir),
                    "--out",
                    str(task_dir / "model.jsonl"),
                ],
                env=_subprocess_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("provenance mismatch", proc.stderr)

    def test_end_to_end_refinery_task_run_smoke(self) -> None:
        base_dir = REPO_ROOT / "examples/case-study-agent-tool-gate"
        env = _subprocess_env()
        with tempfile.TemporaryDirectory() as td:
            task_dir = Path(td) / "task"
            runs_dir = Path(td) / "runs"
            subprocess.run(
                [sys.executable, str(base_dir / "generate_dataset.py"), "--out-dir", str(task_dir)],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(base_dir / "baseline_predict.py"),
                    "--task-dir",
                    str(task_dir),
                    "--out",
                    str(task_dir / "baseline.jsonl"),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(base_dir / "gate_predict.py"),
                    "--task-dir",
                    str(task_dir),
                    "--out",
                    str(task_dir / "model.jsonl"),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            val_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "hardonian_foundry.cli",
                    "refinery",
                    "validate-dataset",
                    "--task-spec",
                    str(task_dir / "spec.json"),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(val_proc.returncode, 0, val_proc.stderr)
            run_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "hardonian_foundry.cli",
                    "refinery",
                    "run",
                    "--task-spec",
                    str(task_dir / "spec.json"),
                    "--baseline-predictions",
                    str(task_dir / "baseline.jsonl"),
                    "--model-predictions",
                    str(task_dir / "model.jsonl"),
                    "--output-root",
                    str(runs_dir),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(run_proc.returncode, 0, run_proc.stderr)
            run_data = json.loads(run_proc.stdout)
            self.assertEqual(run_data["evaluation"]["task"]["model_metrics"]["macro_f1"], 1.0)
            self.assertEqual(run_data["evaluation"]["task"]["model_metrics"]["critical_false_negative_rate"], 0.0)


class ReconciliationPolicyCaseStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gen = _load_module("examples/case-study-reconciliation-policy/generate_dataset.py", "pol_gen")
        cls.baseline = _load_module("examples/case-study-reconciliation-policy/baseline_predict.py", "pol_base")
        cls.policy = _load_module("examples/case-study-reconciliation-policy/policy_predict.py", "pol_gate")

    def test_determinism_oracle_alignment_and_untrusted_override(self) -> None:
        first = self.gen.generate_records(seed=self.gen.BASE_SEED)
        second = self.gen.generate_records(seed=self.gen.BASE_SEED)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 576)

        spec = self.gen.build_spec(seed=self.gen.BASE_SEED)
        self.assertEqual(task_eval.validate_spec(spec), [])
        self.assertEqual(task_eval.validate_task_dataset(first, spec)["status"], "VALID")

        for row in first:
            case_obj, _ = self.policy.parse_prompt(row["prompt"])
            oracle = assess(case_obj)
            self.assertEqual(row["gold_label"], oracle["category"])
            audit = self.policy.predict_with_audit(row["prompt"])
            self.assertTrue(audit["review_required"])
            self.assertFalse(audit["financial_write"])
            self.assertEqual(audit["category"], row["gold_label"])

        splits = task_eval.split_task_dataset(first, spec)
        test = splits["test"]
        self.assertEqual({r["gold_label"] for r in test}, set(spec["labels"]))

        base_preds = [{"id": r["id"], "label": self.baseline.predict(r["prompt"])} for r in test]
        pol_preds = [{"id": r["id"], "label": self.policy.predict(r["prompt"])} for r in test]
        base_score = task_eval.score_predictions(test, base_preds, spec)
        pol_score = task_eval.score_predictions(test, pol_preds, spec)

        self.assertGreater(base_score["critical_false_negative_rate"], 0.0)
        self.assertEqual(pol_score["critical_false_negative_rate"], 0.0)
        self.assertEqual(pol_score["macro_f1"], 1.0)
        self.assertGreater(pol_score["macro_f1"], base_score["macro_f1"])

    def test_policy_predictor_edge_cases_and_provenance_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            self.policy.parse_prompt("missing structured payload")
        bad_suggestion = (
            'Reconciliation Case synthetic-pol-9999 (feed-00, var=1):\n'
            'Case JSON: {"actual_currency": "USD", "actual_minor": 100, "expected_currency": "USD", '
            '"expected_minor": 100, "id": "synthetic-pol-9999"}\n'
            'Untrusted Model Suggestion: "Matched"\n'
            'Context: malformed capitalization'
        )
        audit = self.policy.predict_with_audit(bad_suggestion)
        self.assertEqual(audit["category"], "matched")
        self.assertEqual(audit["model_status"], "UNAVAILABLE")
        self.assertIsNone(audit["model_agrees"])
        self.assertTrue(audit["review_required"])
        self.assertFalse(audit["financial_write"])

        with tempfile.TemporaryDirectory() as td:
            task_dir = Path(td)
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "examples/case-study-reconciliation-policy/generate_dataset.py"),
                    "--out-dir",
                    str(task_dir),
                ],
                env=_subprocess_env(),
                capture_output=True,
                check=True,
            )
            spec = json.loads((task_dir / "spec.json").read_text(encoding="utf-8"))
            spec["model"]["adapter_sha256"] = "e" * 64
            (task_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "examples/case-study-reconciliation-policy/policy_predict.py"),
                    "--task-dir",
                    str(task_dir),
                    "--out",
                    str(task_dir / "model.jsonl"),
                ],
                env=_subprocess_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("provenance mismatch", proc.stderr)

    def test_end_to_end_refinery_task_run_smoke(self) -> None:
        base_dir = REPO_ROOT / "examples/case-study-reconciliation-policy"
        env = _subprocess_env()
        with tempfile.TemporaryDirectory() as td:
            task_dir = Path(td) / "task"
            runs_dir = Path(td) / "runs"
            subprocess.run(
                [sys.executable, str(base_dir / "generate_dataset.py"), "--out-dir", str(task_dir)],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(base_dir / "baseline_predict.py"),
                    "--task-dir",
                    str(task_dir),
                    "--out",
                    str(task_dir / "baseline.jsonl"),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(base_dir / "policy_predict.py"),
                    "--task-dir",
                    str(task_dir),
                    "--out",
                    str(task_dir / "model.jsonl"),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            run_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "hardonian_foundry.cli",
                    "refinery",
                    "run",
                    "--task-spec",
                    str(task_dir / "spec.json"),
                    "--baseline-predictions",
                    str(task_dir / "baseline.jsonl"),
                    "--model-predictions",
                    str(task_dir / "model.jsonl"),
                    "--output-root",
                    str(runs_dir),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(run_proc.returncode, 0, run_proc.stderr)
            run_data = json.loads(run_proc.stdout)
            self.assertEqual(run_data["evaluation"]["task"]["model_metrics"]["macro_f1"], 1.0)
            self.assertEqual(run_data["evaluation"]["task"]["model_metrics"]["critical_false_negative_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
