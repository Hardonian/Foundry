"""Task-mode tests: spec validation, family-disjoint splits, prediction scoring,
license attachment, Ed25519 signing, and the promotion gate end-to-end."""
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from hardonian_foundry import evidence, signing
from hardonian_foundry.baseline import canonical, digest
from hardonian_foundry.evidence import append as append_evidence
from hardonian_foundry.evidence import verify as verify_evidence
from hardonian_foundry.refinery import (
    attach_license,
    create_task_run,
    evaluate_promotion,
    inspect_run,
    load_keyring,
    sign_run,
)
from hardonian_foundry.task_eval import (
    score_predictions,
    split_task_dataset,
    validate_spec,
    validate_task_dataset,
)

LABELS = ["ALPHA", "BETA", "GAMMA"]
GOOD_SPEC = {
    "schema_version": "1.0",
    "task_id": "test-exception-codes",
    "description": "Synthetic exception-code classification contract test.",
    "labels": LABELS,
    "thresholds": {"macro_f1_min": 0.8},
    "baseline_must_improve": True,
    "split_seed": 7,
    "model": {
        "base_model_id": "Qwen/Qwen2.5-Coder-0.5B",
        "revision": "0123456789abcdef",
        "adapter_sha256": "a" * 64,
        "adapter_name": "test-adapter",
    },
    "dataset_path": "data.jsonl",
}
ALLOWED_MANIFEST = {
    "name": "qwen2.5-coder-0.5b-test-adapter",
    "license_id": "Apache-2.0",
    "provenance": "synthetic",
    "data_origin": "synthetic",
}


def make_records(count=30, families=6):
    rows = []
    for i in range(count):
        family = i % families
        block = i // families
        rows.append({
            "id": f"case-{i:03d}",
            "prompt": f"operator note variant {i}: reviewed the record and closed it accordingly",
            "gold_label": LABELS[block % len(LABELS)],
            # every family gets critical rows, so any test split exercises the
            # critical-false-negative gate regardless of family ordering.
            "critical": block % 2 == 0,
            "family": f"template-{family}",
        })
    return rows


def build_task_dir(root, records, spec=None):
    spec = spec or GOOD_SPEC
    (Path(root) / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    with (Path(root) / "data.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return spec


def predictions_for(test_rows, strategy="gold"):
    if strategy == "gold":
        return [{"id": row["id"], "label": row["gold_label"]} for row in test_rows]
    return [{"id": row["id"], "label": LABELS[0]} for row in test_rows]


class SpecValidationTests(unittest.TestCase):
    def test_good_spec_valid(self):
        self.assertEqual(validate_spec(GOOD_SPEC), [])

    def test_duplicate_labels_rejected(self):
        bad = json.loads(json.dumps(GOOD_SPEC))
        bad["labels"] = ["ALPHA", "ALPHA"]
        self.assertIn("labels must be unique", validate_spec(bad))

    def test_adapter_sha_must_be_hex64(self):
        bad = json.loads(json.dumps(GOOD_SPEC))
        bad["model"]["adapter_sha256"] = "ZZ"
        self.assertIn("model.adapter_sha256 must be 64-char lowercase hex", validate_spec(bad))

    def test_threshold_below_floor_rejected(self):
        bad = json.loads(json.dumps(GOOD_SPEC))
        bad["thresholds"] = {"macro_f1_min": 0.1}
        self.assertIn("macro_f1_min must be a number in [0.8, 1.0]", validate_spec(bad))

    def test_traversal_dataset_path_rejected(self):
        bad = json.loads(json.dumps(GOOD_SPEC))
        bad["dataset_path"] = "../outside.jsonl"
        self.assertIn("dataset_path must be a relative, traversal-free path <= 1024 chars", validate_spec(bad))


class DatasetValidationTests(unittest.TestCase):
    def test_valid_records(self):
        result = validate_task_dataset(make_records(), GOOD_SPEC)
        self.assertEqual(result["status"], "VALID")
        self.assertTrue(result["family_disjoint"])

    def test_duplicate_ids_detected(self):
        rows = make_records()
        rows[1]["id"] = rows[0]["id"]
        errors = validate_task_dataset(rows, GOOD_SPEC)["errors"]
        self.assertTrue(any("duplicate record id" in e for e in errors))

    def test_gold_label_outside_spec(self):
        rows = make_records()
        rows[3]["gold_label"] = "OMEGA"
        errors = validate_task_dataset(rows, GOOD_SPEC)["errors"]
        self.assertTrue(any("not in spec labels" in e for e in errors))

    def test_family_mixing_rejected(self):
        rows = make_records()
        del rows[2]["family"]
        errors = validate_task_dataset(rows, GOOD_SPEC)["errors"]
        self.assertTrue(any("'family' must be present on all records or none" in e for e in errors))

    def test_duplicate_prompt_leakage(self):
        rows = make_records()
        rows[5]["prompt"] = rows[4]["prompt"]
        errors = validate_task_dataset(rows, GOOD_SPEC)["errors"]
        self.assertTrue(any("leakage hazard" in e for e in errors))


class SplitTests(unittest.TestCase):
    def test_family_disjoint_and_deterministic(self):
        rows = make_records()
        first = split_task_dataset(rows, GOOD_SPEC)
        second = split_task_dataset(rows, GOOD_SPEC)
        for name in ("train", "validation", "test"):
            self.assertEqual([r["id"] for r in first[name]], [r["id"] for r in second[name]])
        families = {name: {r["family"] for r in first[name]} for name in first}
        self.assertFalse(families["train"] & families["validation"])
        self.assertFalse(families["train"] & families["test"])
        self.assertFalse(families["validation"] & families["test"])
        total = sum(len(first[name]) for name in first)
        self.assertEqual(total, len(rows))
        for name in first:
            self.assertGreater(len(first[name]), 0)


class ScoringTests(unittest.TestCase):
    def test_golden_macro_f1(self):
        rows = [
            {"id": "a", "prompt": "x", "gold_label": "ALPHA", "critical": False},
            {"id": "b", "prompt": "y", "gold_label": "ALPHA", "critical": False},
            {"id": "c", "prompt": "z", "gold_label": "BETA", "critical": True},
            {"id": "d", "prompt": "w", "gold_label": "BETA", "critical": False},
        ]
        spec = json.loads(json.dumps(GOOD_SPEC))
        spec["labels"] = ["ALPHA", "BETA"]
        preds = [{"id": "a", "label": "ALPHA"}, {"id": "b", "label": "ALPHA"},
                 {"id": "c", "label": "BETA"}, {"id": "d", "label": "ALPHA"}]
        metrics = score_predictions(rows, preds, spec)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        # ALPHA: tp=2 fp=1 fn=0 -> p=2/3 r=1 f1=0.8 ; BETA: tp=1 fp=0 fn=1 -> p=1 r=0.5 f1=2/3
        self.assertAlmostEqual(metrics["per_class"]["ALPHA"]["f1"], 0.8, places=12)
        self.assertAlmostEqual(metrics["per_class"]["BETA"]["f1"], 2 / 3, places=12)
        self.assertAlmostEqual(metrics["macro_f1"], (0.8 + 2 / 3) / 2, places=12)
        self.assertEqual(metrics["critical_false_negative_rate"], 0.0)
        self.assertEqual(metrics["critical_total"], 1)

    def test_critical_miss_rate(self):
        rows = [
            {"id": "a", "prompt": "x", "gold_label": "ALPHA", "critical": True},
            {"id": "b", "prompt": "y", "gold_label": "BETA", "critical": False},
        ]
        spec = json.loads(json.dumps(GOOD_SPEC))
        spec["labels"] = ["ALPHA", "BETA"]
        preds = [{"id": "a", "label": "BETA"}, {"id": "b", "label": "BETA"}]
        metrics = score_predictions(rows, preds, spec)
        self.assertEqual(metrics["critical_false_negative_rate"], 1.0)

    def test_prediction_coverage_fail_closed(self):
        rows = [{"id": "a", "prompt": "x", "gold_label": "ALPHA", "critical": False}]
        spec = json.loads(json.dumps(GOOD_SPEC))
        with self.assertRaises(ValueError):
            score_predictions(rows, [], spec)
        with self.assertRaises(ValueError):
            score_predictions(rows, [{"id": "a", "label": "OMEGA"}], spec)


@unittest.skipUnless(signing.backend_available(), "signing extra unavailable")
class TaskPromotionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.records = make_records()
        self.spec = build_task_dir(self.root, self.records)
        splits = split_task_dataset(self.records, self.spec)
        self.baseline = self.root / "baseline.jsonl"
        self.model = self.root / "model.jsonl"
        self._write(self.baseline, predictions_for(splits["test"], "constant"))
        self._write(self.model, predictions_for(splits["test"], "gold"))
        self.runs = str(self.root / "runs")

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def _write(path, preds):
        with path.open("w", encoding="utf-8") as handle:
            for row in preds:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _create_run(self):
        return create_task_run(
            task_spec=self.spec,
            dataset_records=self.records,
            baseline_predictions=[json.loads(line) for line in self.baseline.read_text().splitlines()],
            model_predictions=[json.loads(line) for line in self.model.read_text().splitlines()],
            output_root=self.runs,
        )

    def _sign(self, run_id):
        key_path = str(self.root / "sign.pem")
        public = signing.generate_keypair(key_path)["public_key_hex"]
        sign_run(run_id, private_key_path=key_path, output_root=self.runs)
        return [public]

    def _premature_signature(self, rid):
        key_path = str(self.root / "premature.pem")
        public = signing.generate_keypair(key_path)["public_key_hex"]
        ledger = str(Path(self.runs) / "evidence.jsonl")
        run = json.loads((Path(self.runs) / rid / "run.json").read_text())
        head = verify_evidence(ledger)["head"]
        timestamp = "20260101T000000Z"
        from hardonian_foundry.baseline import digest
        payload = {"version": "2", "run_id": rid, "signed_head": head,
                   "created_at": timestamp, "run_sha256": digest(run)}
        append_evidence(ledger, {"data_origin": "synthetic", "type": "promotion_signature",
                                 **payload, "public_key_hex": public,
                                 "signature_hex": signing.sign(canonical(payload).encode(), key_path)},
                        key=f"{rid}#sig")
        return [public]

    def test_signing_requires_allowed_license(self):
        rid = self._create_run()["run_id"]
        key = str(self.root / "no-license.pem")
        signing.generate_keypair(key)
        ledger = Path(self.runs) / "evidence.jsonl"
        before = ledger.read_text()
        with self.assertRaisesRegex(ValueError, "allowed license first"):
            sign_run(rid, private_key_path=key, output_root=self.runs)
        self.assertEqual(before, ledger.read_text())
        denied = dict(ALLOWED_MANIFEST, license_id="GPL-3.0")
        attach_license(rid, manifest=denied, output_root=self.runs)
        with self.assertRaisesRegex(ValueError, "allowed license first"):
            sign_run(rid, private_key_path=key, output_root=self.runs)

    def test_malformed_nested_metrics_fail_closed(self):
        rid = self._create_run()["run_id"]
        attach_license(rid, manifest=ALLOWED_MANIFEST, output_root=self.runs)
        keys = self._sign(rid)
        path = Path(self.runs) / rid / "run.json"
        original = json.loads(path.read_text())
        cases = [("task", [1]), ("baseline", [1]), ("model_metrics", [1]),
                 ("thresholds", [1]), ("macro_f1", "bad"), ("macro_f1", float("nan")),
                 ("baseline_must_improve", "false"), ("f1", "bad"),
                 ("evaluation", [1])]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                changed = json.loads(json.dumps(original))
                if field == "evaluation":
                    changed["evaluation"] = value
                elif field in ("f1", "task"):
                    changed["evaluation"][field] = value
                elif field == "macro_f1":
                    changed["evaluation"]["task"]["baseline"][field] = value
                else:
                    changed["evaluation"]["task"][field] = value
                path.write_text(json.dumps(changed))
                result = evaluate_promotion(rid, output_root=self.runs, keyring=keys)
                self.assertFalse(result["promoted"])
                self.assertEqual(result["status"], "BLOCKED")
                expected = ("run_inspection_failed" if field == "evaluation" or
                            (field == "macro_f1" and isinstance(value, float)) else
                            "positive_case_coverage" if field == "f1" else
                            "task_label_coverage" if field in ("task", "baseline", "model_metrics") else
                            "task_benchmark_metrics")
                self.assertIn(expected, result["missing_evidence"])

    def test_signed_run_metadata_cannot_be_changed(self):
        splits = split_task_dataset(self.records, self.spec)
        self._write(self.baseline, predictions_for(splits["test"], "gold"))
        run = self._create_run()
        rid = run["run_id"]
        attach_license(rid, manifest=ALLOWED_MANIFEST, output_root=self.runs)
        keys = self._sign(rid)
        path = Path(self.runs) / rid / "run.json"
        original = json.loads(path.read_text())
        self.assertFalse(evaluate_promotion(rid, output_root=self.runs, keyring=keys)["promoted"])
        original["evaluation"]["task"]["baseline_must_improve"] = False
        path.write_text(json.dumps(original))
        result = evaluate_promotion(rid, output_root=self.runs, keyring=keys)
        self.assertFalse(result["promoted"])
        self.assertIn("verified_promotion_signature", result["missing_evidence"])

    def test_signed_adapter_identity_cannot_be_changed(self):
        rid = self._create_run()["run_id"]
        attach_license(rid, manifest=ALLOWED_MANIFEST, output_root=self.runs)
        keys = self._sign(rid)
        self.assertTrue(evaluate_promotion(rid, output_root=self.runs, keyring=keys)["promoted"])
        path = Path(self.runs) / rid / "run.json"
        changed = json.loads(path.read_text())
        changed["evaluation"]["task"]["model_identity"]["adapter_sha256"] = "b" * 64
        path.write_text(json.dumps(changed))
        result = evaluate_promotion(rid, output_root=self.runs, keyring=keys)
        self.assertFalse(result["promoted"])
        self.assertIn("verified_promotion_signature", result["missing_evidence"])

    def test_license_appended_after_signature_does_not_authorize(self):
        rid = self._create_run()["run_id"]
        keys = self._premature_signature(rid)
        attach_license(rid, manifest=ALLOWED_MANIFEST, output_root=self.runs)
        with self.assertRaisesRegex(ValueError, "recreate the run"):
            sign_run(rid, private_key_path=str(self.root / "premature.pem"), output_root=self.runs)
        result = evaluate_promotion(rid, output_root=self.runs, keyring=keys)
        self.assertFalse(result["promoted"])
        self.assertIn("verified_promotion_signature", result["missing_evidence"])

    def test_evaluation_uses_verified_ledger_snapshot(self):
        from hardonian_foundry.evidence import verify_entries

        rid = self._create_run()["run_id"]
        keys = self._premature_signature(rid)

        def append_during_verification(entries):
            report = verify_entries(entries)
            attach_license(rid, manifest=ALLOWED_MANIFEST, output_root=self.runs)
            return report

        with patch("hardonian_foundry.refinery.evidence_verify_entries", side_effect=append_during_verification):
            result = evaluate_promotion(rid, output_root=self.runs, keyring=keys)
        self.assertFalse(result["promoted"])
        self.assertIn("model_licensing_manifest", result["missing_evidence"])

    def test_missing_contract_label_blocks_promotion(self):
        self.spec = json.loads(json.dumps(self.spec))
        self.spec["labels"] += ["DELTA", "EPSILON", "ZETA"]
        for row in self.records:
            row["gold_label"] = self.spec["labels"][int(row["id"].split("-")[1]) // 6]
        test_rows = split_task_dataset(self.records, self.spec)["test"]
        self._write(self.baseline, predictions_for(test_rows, "constant"))
        self._write(self.model, predictions_for(test_rows, "gold"))
        run = self._create_run()
        self.assertGreater(run["evaluation"]["f1"], 0.8)
        self.assertEqual(run["evaluation"]["task"]["model_metrics"]["per_class"]["ZETA"]["support"], 0)
        rid = run["run_id"]
        attach_license(rid, manifest=ALLOWED_MANIFEST, output_root=self.runs)
        keys = self._sign(rid)
        result = evaluate_promotion(rid, output_root=self.runs, keyring=keys)
        self.assertFalse(result["promoted"])
        self.assertIn("task_label_coverage", result["missing_evidence"])

    def test_malformed_contract_labels_block_promotion(self):
        rid = self._create_run()["run_id"]
        attach_license(rid, manifest=ALLOWED_MANIFEST, output_root=self.runs)
        keys = self._sign(rid)
        path = Path(self.runs) / rid / "run.json"
        original = json.loads(path.read_text())
        for labels in ([[]], [{}], [None], [1], [""], ["ALPHA", "ALPHA"]):
            with self.subTest(labels=labels):
                changed = json.loads(json.dumps(original))
                changed["evaluation"]["task"]["labels"] = labels
                path.write_text(json.dumps(changed))
                result = evaluate_promotion(rid, output_root=self.runs, keyring=keys)
                self.assertEqual(result["status"], "BLOCKED")
                self.assertIn("task_label_coverage", result["missing_evidence"])

    def test_concurrent_signers_only_one_attestation(self):
        rid = self._create_run()["run_id"]
        attach_license(rid, manifest=ALLOWED_MANIFEST, output_root=self.runs)
        key = str(self.root / "race.pem")
        public = signing.generate_keypair(key)["public_key_hex"]
        barrier = threading.Barrier(2)
        def attempt():
            barrier.wait(timeout=5)
            try:
                return sign_run(rid, private_key_path=key, output_root=self.runs)
            except ValueError as exc:
                return str(exc)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: attempt(), range(2)))
        self.assertEqual(sum(isinstance(result, dict) for result in results), 1)
        self.assertTrue(any(isinstance(result, str) and "recreate the run" in result for result in results))
        self.assertTrue(evaluate_promotion(rid, output_root=self.runs, keyring=[public])["promoted"])

    def test_signing_serializes_unrelated_append(self):
        rid = self._create_run()["run_id"]
        attach_license(rid, manifest=ALLOWED_MANIFEST, output_root=self.runs)
        key_path = str(self.root / "concurrent.pem")
        public = signing.generate_keypair(key_path)["public_key_hex"]
        ledger = str(Path(self.runs) / "evidence.jsonl")
        signing_started = threading.Event()
        release_signing = threading.Event()
        lock_attempted = threading.Event()
        real_sign = signing.sign
        real_lock = evidence._lock

        def paused_sign(message, key):
            signing_started.set()
            if not release_signing.wait(5):
                raise TimeoutError("signing was not released")
            return real_sign(message, key)

        def competing_append():
            return append_evidence(ledger, {"data_origin": "synthetic", "type": "unrelated"})

        def observed_lock(handle):
            if signing_started.is_set():
                lock_attempted.set()
            return real_lock(handle)

        with (ThreadPoolExecutor(max_workers=2) as pool,
              patch("hardonian_foundry.refinery.signing.sign", side_effect=paused_sign),
              patch("hardonian_foundry.evidence._lock", side_effect=observed_lock)):
            signer = pool.submit(sign_run, rid, private_key_path=key_path, output_root=self.runs)
            try:
                self.assertTrue(signing_started.wait(5))
                writer = pool.submit(competing_append)
                self.assertTrue(lock_attempted.wait(5))
                with self.assertRaises(TimeoutError):
                    writer.result(timeout=0.1)
            finally:
                release_signing.set()
            signed = signer.result(timeout=5)
            other = writer.result(timeout=5)
        entries = [json.loads(line) for line in Path(ledger).read_text().splitlines()]
        signature = next(e for e in entries if e["entry_hash"] == signed["entry_hash"])
        self.assertEqual(signature["prev_hash"], signature["record"]["signed_head"])
        self.assertGreater(other["seq"], signature["seq"])
        self.assertEqual(verify_evidence(ledger)["status"], "OK")
        self.assertTrue(evaluate_promotion(rid, output_root=self.runs, keyring=[public])["promoted"])

    def test_signing_rejects_corrupt_chain_without_appending(self):
        rid = self._create_run()["run_id"]
        key_path = str(self.root / "corrupt.pem")
        signing.generate_keypair(key_path)
        ledger = Path(self.runs) / "evidence.jsonl"
        entry = json.loads(ledger.read_text())
        entry["record"]["run_id"] = "modified"
        broken = json.dumps(entry) + "\n"
        ledger.write_text(broken)
        with self.assertRaises(ValueError):
            sign_run(rid, private_key_path=key_path, output_root=self.runs)
        self.assertEqual(ledger.read_text(), broken)

    def test_signing_rejects_existing_signature_with_other_key(self):
        rid = self._create_run()["run_id"]
        ledger = str(Path(self.runs) / "evidence.jsonl")
        append_evidence(ledger, {"data_origin": "synthetic", "type": "promotion_signature",
                                 "run_id": rid}, key="nonstandard-key")
        key_path = str(self.root / "duplicate.pem")
        signing.generate_keypair(key_path)
        before = Path(ledger).read_text()
        with self.assertRaisesRegex(ValueError, "already signed"):
            sign_run(rid, private_key_path=key_path, output_root=self.runs)
        self.assertEqual(Path(ledger).read_text(), before)

    def test_legacy_signature_is_blocked(self):
        rid = self._create_run()["run_id"]
        attach_license(rid, manifest=ALLOWED_MANIFEST, output_root=self.runs)
        key_path = str(self.root / "old.pem")
        public = signing.generate_keypair(key_path)["public_key_hex"]
        ledger = str(Path(self.runs) / "evidence.jsonl")
        signed_head = verify_evidence(ledger)["head"]
        timestamp = "20260101T000000Z"
        message = canonical({"run_id": rid, "signed_head": signed_head, "created_at": timestamp}).encode()
        append_evidence(ledger, {"data_origin": "synthetic", "type": "promotion_signature", "run_id": rid,
                                 "signed_head": signed_head, "created_at": timestamp,
                                 "public_key_hex": public, "signature_hex": signing.sign(message, key_path)},
                        key=f"{rid}#sig")
        result = evaluate_promotion(rid, output_root=self.runs, keyring=[public])
        self.assertFalse(result["promoted"])
        self.assertIn("verified_promotion_signature", result["missing_evidence"])
        with self.assertRaisesRegex(ValueError, "recreate the run"):
            sign_run(rid, private_key_path=key_path, output_root=self.runs)

    def test_signed_run_manifest_and_later_unrelated_ledger_entry(self):
        test_rows = split_task_dataset(self.records, self.spec)["test"]
        run = create_task_run(task_spec=self.spec, dataset_records=self.records,
                              baseline_predictions=predictions_for(test_rows, "constant"),
                              model_predictions=predictions_for(test_rows), output_root=self.runs,
                              model_manifest=ALLOWED_MANIFEST)
        rid = run["run_id"]
        keys = self._sign(rid)
        append_evidence(str(Path(self.runs) / "evidence.jsonl"),
                        {"data_origin": "synthetic", "type": "unrelated"})
        self.assertTrue(evaluate_promotion(rid, output_root=self.runs, keyring=keys)["promoted"])
        path = Path(self.runs) / rid / "run.json"
        changed = json.loads(path.read_text())
        changed["model_manifest"]["name"] = "other-model"
        path.write_text(json.dumps(changed))
        self.assertFalse(evaluate_promotion(rid, output_root=self.runs, keyring=keys)["promoted"])

    def test_full_promotion_path(self):
        run = self._create_run()
        run_id = run["run_id"]
        self.assertEqual(inspect_run(run_id, output_root=self.runs)["provenance_status"], "PROVENANCE_VERIFIED")
        self.assertEqual(run["schema_version"], "1.1")
        self.assertEqual(run["evaluation"]["mode"], "task_predictor_pair")
        self.assertEqual(run["evaluation"]["f1"], 1.0)
        self.assertEqual(run["evaluation"]["critical_false_negative_rate"], 0.0)

        blocked = evaluate_promotion(run_id, output_root=self.runs)
        self.assertFalse(blocked["promoted"])
        self.assertIn("model_licensing_manifest", blocked["missing_evidence"])
        self.assertIn("cryptographic_promotion_signature", blocked["missing_evidence"])

        attach = attach_license(run_id, manifest=ALLOWED_MANIFEST, output_root=self.runs)
        self.assertEqual(attach["decision"], "ALLOWED")

        key_path = str(self.root / "sign.pem")
        generated = signing.generate_keypair(key_path)
        keyring_path = self.root / "keyring.json"
        keyring_path.write_text(json.dumps({"trusted_keys": [
            {"label": "test", "public_key_hex": generated["public_key_hex"]},
        ]}), encoding="utf-8")

        signed = sign_run(run_id, private_key_path=key_path, output_root=self.runs)
        self.assertEqual(len(signed["signed_head"]), 64)

        promo = evaluate_promotion(run_id, output_root=self.runs, keyring=load_keyring(str(keyring_path)))
        self.assertTrue(promo["promoted"])
        self.assertEqual(promo["status"], "ALLOWED_REVIEW_REQUIRED")
        self.assertTrue(promo["review_required"])
        self.assertEqual(promo["missing_evidence"], [])

    def test_signed_but_no_keyring_blocks(self):
        run = self._create_run()
        run_id = run["run_id"]
        attach_license(run_id, manifest=ALLOWED_MANIFEST, output_root=self.runs)
        key_path = str(self.root / "sign.pem")
        signing.generate_keypair(key_path)
        sign_run(run_id, private_key_path=key_path, output_root=self.runs)
        promo = evaluate_promotion(run_id, output_root=self.runs)
        self.assertFalse(promo["promoted"])
        self.assertIn("trusted_promotion_keyring", promo["missing_evidence"])

    def test_untrusted_key_blocks(self):
        run = self._create_run()
        run_id = run["run_id"]
        attach_license(run_id, manifest=ALLOWED_MANIFEST, output_root=self.runs)
        key_path = str(self.root / "sign.pem")
        signing.generate_keypair(key_path)
        sign_run(run_id, private_key_path=key_path, output_root=self.runs)
        other = signing.generate_keypair(str(self.root / "other.pem"))
        promo = evaluate_promotion(run_id, output_root=self.runs, keyring=[other["public_key_hex"]])
        self.assertFalse(promo["promoted"])
        self.assertIn("verified_promotion_signature", promo["missing_evidence"])

    def test_no_improvement_over_baseline_blocks(self):
        splits = split_task_dataset(self.records, self.spec)
        self._write(self.model, predictions_for(splits["test"], "constant"))
        run = self._create_run()
        run_id = run["run_id"]
        attach_license(run_id, manifest=ALLOWED_MANIFEST, output_root=self.runs)
        key_path = str(self.root / "sign.pem")
        generated = signing.generate_keypair(key_path)
        sign_run(run_id, private_key_path=key_path, output_root=self.runs)
        promo = evaluate_promotion(run_id, output_root=self.runs,
                                   keyring=[generated["public_key_hex"]])
        self.assertFalse(promo["promoted"])
        self.assertIn("baseline_improvement", promo["missing_evidence"])
        self.assertIn("task_macro_f1_threshold", promo["missing_evidence"])

    def test_denied_license_blocks(self):
        run = self._create_run()
        run_id = run["run_id"]
        denied = dict(ALLOWED_MANIFEST, license_id="GPL-3.0-only")
        attach = attach_license(run_id, manifest=denied, output_root=self.runs)
        self.assertEqual(attach["decision"], "DENIED")
        promo = evaluate_promotion(run_id, output_root=self.runs)
        self.assertFalse(promo["promoted"])
        self.assertIn("approved_spdx_license", promo["missing_evidence"])

    def test_ledger_tampering_after_sign_blocks(self):
        run = self._create_run()
        run_id = run["run_id"]
        attach_license(run_id, manifest=ALLOWED_MANIFEST, output_root=self.runs)
        key_path = str(self.root / "sign.pem")
        generated = signing.generate_keypair(key_path)
        sign_run(run_id, private_key_path=key_path, output_root=self.runs)
        keyring = [generated["public_key_hex"]]

        ledger = Path(self.runs) / "evidence.jsonl"
        lines = ledger.read_text(encoding="utf-8").splitlines()
        mutated = []
        for line in lines:
            entry = json.loads(line)
            if entry.get("record", {}).get("type") == "model_licensing_manifest":
                entry["record"]["manifest_sha256"] = "f" * 64
                line = json.dumps(entry, sort_keys=True, separators=(",", ":"))
            mutated.append(line)
        ledger.write_text("\n".join(mutated) + "\n", encoding="utf-8")

        promo = evaluate_promotion(run_id, output_root=self.runs, keyring=keyring)
        self.assertFalse(promo["promoted"])
        self.assertIn("tamper_evident_proof", promo["missing_evidence"])

    def test_world_readable_key_rejected(self):
        run = self._create_run()
        key_path = str(self.root / "loose.pem")
        signing.generate_keypair(key_path)
        os.chmod(key_path, 0o644)
        with self.assertRaises(ValueError):
            sign_run(run["run_id"], private_key_path=key_path, output_root=self.runs)

    def test_double_sign_and_double_license_rejected(self):
        run = self._create_run()
        run_id = run["run_id"]
        attach_license(run_id, manifest=ALLOWED_MANIFEST, output_root=self.runs)
        with self.assertRaises(ValueError):
            attach_license(run_id, manifest=ALLOWED_MANIFEST, output_root=self.runs)
        key_path = str(self.root / "sign.pem")
        signing.generate_keypair(key_path)
        sign_run(run_id, private_key_path=key_path, output_root=self.runs)
        with self.assertRaises(ValueError):
            sign_run(run_id, private_key_path=key_path, output_root=self.runs)

    def test_run_and_attach_hashes_are_bound(self):
        run = self._create_run()
        run_id = run["run_id"]
        attach = attach_license(run_id, manifest=ALLOWED_MANIFEST, output_root=self.runs)
        self.assertEqual(attach["manifest_sha256"], digest(ALLOWED_MANIFEST))


if __name__ == "__main__":
    unittest.main()
