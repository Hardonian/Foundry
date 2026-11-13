"""Deep unit tests for hardonian_foundry.signing covering key generation, signing,
verification error cases, key permissions, and public key formatting."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hardonian_foundry.signing import (
    SigningUnavailable,
    _check_private_key_permissions,
    backend_available,
    check_keyring_quorum,
    fingerprint,
    generate_keypair,
    load_keyring_policy,
    normalize_trusted_keys,
    parse_keyring,
    public_key_hex,
    sign,
    verify,
)


class SigningDeepTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_backend_available(self):
        self.assertTrue(backend_available())

    def test_keypair_and_sign_verify_roundtrip(self):
        key_path = self.root / "test_key.pem"
        pair = generate_keypair(key_path)
        self.assertTrue(key_path.is_file())
        pub_hex = pair["public_key_hex"]
        self.assertEqual(len(pub_hex), 64)

        # public_key_hex reads correctly from file
        self.assertEqual(public_key_hex(key_path), pub_hex)

        message = b"tamper-evident audit message"
        sig_hex = sign(message, key_path)
        self.assertEqual(len(sig_hex), 128)

        # Successful verification
        self.assertTrue(verify(message, sig_hex, pub_hex))

        # Tampered message
        self.assertFalse(verify(b"tampered message", sig_hex, pub_hex))

        # Corrupt signature (flipped bit)
        corrupt_sig = ("0" if sig_hex[0] != "0" else "1") + sig_hex[1:]
        self.assertFalse(verify(message, corrupt_sig, pub_hex))

        # Invalid non-hex signature or wrong length returns False
        self.assertFalse(verify(message, "invalid-hex", pub_hex))
        self.assertFalse(verify(message, "0" * 64, pub_hex))

        # Invalid non-hex public key or wrong length returns False
        self.assertFalse(verify(message, sig_hex, "not-a-hex-pub-key"))
        self.assertFalse(verify(message, sig_hex, "0" * 32))

    def test_sign_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            sign(b"msg", self.root / "missing.pem")

    def test_public_key_hex_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            public_key_hex(self.root / "missing.pem")

    def test_normalize_trusted_keys(self):
        k1 = generate_keypair(self.root / "k1.pem")
        pub1 = k1["public_key_hex"]

        # List of hex strings
        norm = normalize_trusted_keys([pub1])
        self.assertEqual(norm, [pub1])

        # List of dicts
        norm_dict = normalize_trusted_keys([{"public_key_hex": pub1}])
        self.assertEqual(norm_dict, [pub1.lower()])

        # Invalid entries
        with self.assertRaises(ValueError):
            normalize_trusted_keys(["not-hex"])
        with self.assertRaises(ValueError):
            normalize_trusted_keys([123])  # type: ignore[list-item]

    def test_normalize_trusted_keys_none_and_fingerprint(self):
        self.assertEqual(normalize_trusted_keys(None), [])
        k = generate_keypair(self.root / "k_fp.pem")
        fp = fingerprint(k["public_key_hex"])
        self.assertEqual(len(fp), 64)

    def test_backend_unavailable_simulation(self):
        key_path = self.root / "key_unavail.pem"
        with mock.patch("hardonian_foundry.signing.backend_available", return_value=False):
            with self.assertRaises(SigningUnavailable):
                generate_keypair(key_path)
            with self.assertRaises(SigningUnavailable):
                sign(b"test", key_path)
            with self.assertRaises(SigningUnavailable):
                public_key_hex(key_path)
            self.assertFalse(verify(b"test", "0" * 128, "0" * 64))

    def test_check_private_key_permissions_not_file(self):
        # Passing a directory instead of a regular file
        stat_mock = mock.MagicMock(st_mode=0o040755)  # Directory mode
        with self.assertRaises(ValueError) as ctx:
            _check_private_key_permissions(self.root, stat_mock)
        self.assertIn("not a regular file", str(ctx.exception))

    def test_signing_keyring_quorum_and_policy(self):
        k1 = generate_keypair(self.root / "k1_q.pem")
        k2 = generate_keypair(self.root / "k2_q.pem")
        pub1 = k1["public_key_hex"]
        pub2 = k2["public_key_hex"]

        # Parse raw list of keys
        policy_from_list = parse_keyring([pub1, pub2])
        self.assertEqual(policy_from_list["required_signatures"], 1)
        self.assertEqual(len(policy_from_list["active_keys"]), 2)

        # Quorum satisfied with active keys
        q_res = check_keyring_quorum(policy_from_list, [pub1])
        self.assertTrue(q_res["satisfied"])
        self.assertEqual(q_res["valid_count"], 1)

        # Quorum with revoked key
        policy_revoked = parse_keyring({
            "trusted_keys": [
                {"public_key_hex": pub1, "role": "admin", "revoked_at": "2026-11-12T00:00:00Z"},
                {"public_key_hex": pub2, "role": "admin"},
            ],
            "required_signatures": 1,
            "required_roles": ["admin"],
        })
        # Verifying with only revoked key fails quorum
        q_rev = check_keyring_quorum(policy_revoked, [pub1])
        self.assertFalse(q_rev["satisfied"])
        self.assertIn(pub1, q_rev["revoked_signers"])

        # Verifying with active key satisfies quorum
        q_ok = check_keyring_quorum(policy_revoked, [pub2])
        self.assertTrue(q_ok["satisfied"])
        self.assertEqual(q_ok["roles_satisfied"], ["admin"])

        # load_keyring_policy non-dict JSON error
        bad_json = self.root / "bad_policy.json"
        bad_json.write_text("[1, 2, 3]", encoding="utf-8")
        with self.assertRaises(TypeError):
            load_keyring_policy(bad_json)


if __name__ == "__main__":
    unittest.main()
