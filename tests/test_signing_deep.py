"""Deep unit tests for hardonian_foundry.signing covering key generation, signing,
verification error cases, key permissions, and public key formatting."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hardonian_foundry.signing import (
    backend_available,
    generate_keypair,
    normalize_trusted_keys,
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


if __name__ == "__main__":
    unittest.main()
