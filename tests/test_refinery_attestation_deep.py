"""Deep unit tests for hardonian_foundry.refinery_attestation covering keyring validation,
M-of-N policies, license attachment, and ledger anchoring."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hardonian_foundry.refinery_attestation import (
    anchor_run,
    extract_verified_license_attachment,
    load_keyring,
    load_keyring_policy,
    parse_keyring,
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
