"""Deep unit tests for hardonian_foundry.refinery_attestation covering keyring validation,
M-of-N policies, license attachment, and ledger anchoring."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hardonian_foundry.refinery_attestation import (
    _evaluate_license_compat,
    _iter_ledger_entries,
    anchor_run,
    attach_license,
    extract_verified_license_attachment,
    load_keyring,
    load_keyring_policy,
    parse_keyring,
    sign_run,
)
from hardonian_foundry.signing import generate_keypair


class RefineryAttestationDeepTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_keyring_validations(self):
        k1 = generate_keypair(self.root / "k1.pem")
        k2 = generate_keypair(self.root / "k2.pem")
        pub1 = k1["public_key_hex"]
        pub2 = k2["public_key_hex"]

        # Valid policy dict
        policy = parse_keyring({
            "trusted_keys": [
                {"public_key_hex": pub1, "role": "maintainer"},
                {"public_key_hex": pub2, "role": "security"},
            ],
            "required_signatures": 2,
            "required_roles": ["maintainer", "security"],
        })
        self.assertEqual(len(policy["keys"]), 2)
        self.assertEqual(policy["required_signatures"], 2)
        self.assertEqual(policy["required_roles"], ["maintainer", "security"])

        # Type errors
        with self.assertRaises(TypeError):
            parse_keyring("not-a-list-or-dict")
        with self.assertRaises(TypeError):
            parse_keyring({"missing_trusted_keys": True})

        # Unknown top-level fields
        with self.assertRaises(ValueError):
            parse_keyring({"trusted_keys": [pub1], "extra_unknown_field": True})

        # Invalid required_signatures
        with self.assertRaises(ValueError):
            parse_keyring({"trusted_keys": [pub1], "required_signatures": 0})
        with self.assertRaises(ValueError):
            parse_keyring({"trusted_keys": [pub1], "required_signatures": -1})
        with self.assertRaises(ValueError):
            parse_keyring({"trusted_keys": [pub1], "required_signatures": "two"})

        # Invalid required_roles
        with self.assertRaises(ValueError):
            parse_keyring({"trusted_keys": [pub1], "required_roles": [""]})
        with self.assertRaises(ValueError):
            parse_keyring({"trusted_keys": [pub1], "required_roles": "not-a-list"})

        # Unknown entry fields in trusted_keys
        with self.assertRaises(ValueError):
            parse_keyring({"trusted_keys": [{"public_key_hex": pub1, "rogue_field": "x"}]})

        # Invalid role / key_id / revoked_at in entry
        with self.assertRaises(ValueError):
            parse_keyring({"trusted_keys": [{"public_key_hex": pub1, "role": "  "}]})
        with self.assertRaises(ValueError):
            parse_keyring({"trusted_keys": [{"public_key_hex": pub1, "key_id": ""}]})
        with self.assertRaises(ValueError):
            parse_keyring({"trusted_keys": [{"public_key_hex": pub1, "revoked_at": ""}]})
        with self.assertRaises(ValueError):
            parse_keyring({"trusted_keys": [{"public_key_hex": pub1, "label": 123}]})
        with self.assertRaises(TypeError):
            parse_keyring({"trusted_keys": [12345]})

        # Duplicate public key in policy raises ValueError from signing_mod
        with self.assertRaises(ValueError) as ctx:
            parse_keyring({
                "trusted_keys": [
                    {"public_key_hex": pub1, "role": "sec"},
                    {"public_key_hex": pub1, "role": "admin"},
                ]
            })
        self.assertIn("duplicate public_key_hex", str(ctx.exception))

        # Merging duplicate key records in refinery_attestation with mock signing_mod
        mock_sign = mock.MagicMock()
        mock_sign.parse_keyring.return_value = {}
        mock_sign.normalize_trusted_keys.return_value = [pub1]
        with mock.patch("hardonian_foundry.refinery_attestation.facade_attr", return_value=mock_sign):
            merged = parse_keyring({
                "trusted_keys": [
                    {"public_key_hex": pub1, "role": "sec", "key_id": "k1"},
                    {"public_key_hex": pub1, "revoked_at": "2026-11-12T00:00:00Z", "label": "test-label"},
                ]
            })
            self.assertEqual(len(merged["keys"]), 1)
            self.assertTrue(merged["keys"][0]["revoked"])
            self.assertEqual(merged["keys"][0]["role"], "sec")
            self.assertEqual(merged["keys"][0]["label"], "test-label")

    def test_iter_ledger_entries_and_license_compat(self):
        # Missing or non-file path returns []
        self.assertEqual(_iter_ledger_entries(self.root / "missing.jsonl", "run-1", "type1"), [])
        self.assertEqual(_iter_ledger_entries("", "run-1", "type1"), [])

        # Ledger with blank lines and corrupt JSON returns []
        bad_ledger = self.root / "bad_ledger.jsonl"
        bad_ledger.write_text("\n\n{bad json\n", encoding="utf-8")
        self.assertEqual(_iter_ledger_entries(bad_ledger, "run-1", "type1"), [])

        # _evaluate_license_compat fallback on TypeError
        def legacy_eval(manifest):
            return {"decision": "ALLOWED", "reasons": []}

        with mock.patch("hardonian_foundry.refinery_attestation.facade_attr", return_value=legacy_eval):
            res = _evaluate_license_compat({"name": "synth"}, allowed_data_origins=["synthetic"])
            self.assertEqual(res["decision"], "ALLOWED")

    def test_load_keyring_and_load_keyring_policy(self):
        k = generate_keypair(self.root / "k3.pem")
        pub = k["public_key_hex"]
        keyring_file = self.root / "keyring.json"

        # Simple list of keys in trusted_keys
        keyring_file.write_text(json.dumps({"trusted_keys": [pub]}), encoding="utf-8")
        loaded_simple = load_keyring(keyring_file)
        self.assertEqual(loaded_simple, [pub])

        # Policy dict
        keyring_file.write_text(
            json.dumps({
                "trusted_keys": [{"public_key_hex": pub, "role": "sec"}],
                "required_signatures": 1,
                "required_roles": ["sec"],
            }),
            encoding="utf-8",
        )
        loaded_policy = load_keyring(keyring_file)
        self.assertIsInstance(loaded_policy, dict)
        self.assertEqual(loaded_policy["required_roles"], ["sec"])

        policy_obj = load_keyring_policy(keyring_file)
        self.assertEqual(policy_obj["required_signatures"], 1)

        # Invalid file content
        keyring_file.write_text(json.dumps([pub]), encoding="utf-8")
        with self.assertRaises(TypeError):
            load_keyring(keyring_file)
        with self.assertRaises(TypeError):
            load_keyring_policy(keyring_file)

    def test_attach_license_and_sign_run_edge_cases(self):
        priv = generate_keypair(self.root / "sign_key.pem")
        manifest = {
            "name": "synth-model",
            "license_id": "Apache-2.0",
            "provenance": "synthetic",
            "data_origin": "synthetic",
        }

        # attach_license when run does not exist
        with self.assertRaises(FileNotFoundError):
            attach_license("run-nonexistent", manifest=manifest, output_root=self.root)

        # sign_run when run does not exist
        with self.assertRaises(FileNotFoundError):
            sign_run("run-nonexistent", private_key_path=priv["private_key_path"], output_root=self.root)

        # Create mock run directory
        run_dir = self.root / "run-test-tenant"
        run_dir.mkdir(parents=True)
        run_file = run_dir / "run.json"

        # Missing provenance
        run_file.write_text(json.dumps({"run_id": "run-test-tenant"}), encoding="utf-8")
        with self.assertRaises(TypeError):
            sign_run("run-test-tenant", private_key_path=priv["private_key_path"], output_root=self.root)

        # Ledger file missing
        run_file.write_text(
            json.dumps({"run_id": "run-test-tenant", "provenance": {"evidence_ledger_path": "missing_ledger.jsonl"}}),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            attach_license("run-test-tenant", manifest=manifest, output_root=self.root)
        with self.assertRaises(ValueError):
            sign_run("run-test-tenant", private_key_path=priv["private_key_path"], output_root=self.root)

    def test_anchor_run_failures(self):
        # Anchor unverified run_id
        with mock.patch("hardonian_foundry.refinery_attestation.inspect_run", return_value={"provenance_status": "TAMPERED"}):
            with self.assertRaises(ValueError) as ctx:
                anchor_run("run-unverified", output_root=self.root)
            self.assertIn("cannot anchor unverified run", str(ctx.exception))

        # Whole ledger missing
        missing_root = self.root / "missing_root"
        with self.assertRaises(FileNotFoundError):
            anchor_run(output_root=missing_root)

        # Whole ledger symlink escape
        with mock.patch.object(Path, "is_relative_to", return_value=False):
            with self.assertRaises(ValueError) as ctx:
                anchor_run(output_root=self.root)
            self.assertIn("escapes output_root", str(ctx.exception))

        # Whole ledger tampered
        tampered_ledger = self.root / "evidence.jsonl"
        tampered_ledger.write_text(json.dumps({"seq": 1, "invalid": True}) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            anchor_run(output_root=self.root)
        self.assertIn("cannot anchor tampered ledger", str(ctx.exception))

    def test_extract_verified_license_attachment(self):
        # Empty entries
        att, stored, ok = extract_verified_license_attachment([], "run-1")
        self.assertIsNone(att)
        self.assertIsNone(stored)
        self.assertFalse(ok)

        # Corrupt manifest hash
        bad_entry = {
            "record": {
                "type": "model_licensing_manifest",
                "run_id": "run-1",
                "manifest_sha256": "wrong-hash",
                "manifest": {"license": "MIT"},
            }
        }
        att_bad, stored_bad, ok_bad = extract_verified_license_attachment([bad_entry], "run-1")
        self.assertIsNotNone(att_bad)
        self.assertIsNone(stored_bad)
        self.assertFalse(ok_bad)

    def test_anchor_run_whole_ledger(self):
        from hardonian_foundry.evidence import append as evidence_append

        ledger_path = self.root / "evidence.jsonl"
        evidence_append(str(ledger_path), {"step": "init", "data_origin": "synthetic"})
        evidence_append(str(ledger_path), {"step": "eval", "data_origin": "synthetic"})

        out_anchor = self.root / "anchor.json"
        anchor = anchor_run(output_root=self.root, anchor_path=out_anchor)
        self.assertEqual(anchor["count"], 2)
        self.assertTrue(out_anchor.is_file())
        self.assertEqual(anchor["head"], json.loads(out_anchor.read_text(encoding="utf-8"))["head"])


if __name__ == "__main__":
    unittest.main()
