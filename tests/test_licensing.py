"""Offline tests for licensing, governance tiers, signing keyrings, and key permissions."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hardonian_foundry import licensing, signing

VALID = {
    "name": "synthetic-reviewer",
    "license_id": "Apache-2.0",
    "provenance": "synthetic",
    "data_origin": "synthetic",
    "weights_sha256": "0" * 64,
}


class LicensingTest(unittest.TestCase):
    def assert_denied(self, manifest: object, fragment: str, **kwargs: object) -> dict[str, object]:
        result = licensing.evaluate(manifest, **kwargs)  # type: ignore[arg-type]
        self.assertEqual(result["decision"], "DENIED")
        self.assertFalse(result["promotion_allowed"])
        self.assertTrue(result["review_required"])
        self.assertTrue(any(fragment in reason for reason in result["reasons"]), result["reasons"])
        return result

    def test_valid_manifest_allowed_but_never_promotable(self) -> None:
        result = licensing.evaluate(dict(VALID))
        self.assertEqual(result["decision"], "ALLOWED")
        self.assertEqual(result["reasons"], [])
        self.assertFalse(result["promotion_allowed"])
        self.assertTrue(result["review_required"])

    def test_unknown_license_denied(self) -> None:
        self.assert_denied({**VALID, "license_id": "LicenseRef-Custom"}, "allowlist")

    def test_missing_fields_denied(self) -> None:
        for field in ("name", "license_id", "provenance", "data_origin"):
            manifest = {k: v for k, v in VALID.items() if k != field}
            self.assert_denied(manifest, f"missing required field: {field}")

    def test_non_synthetic_data_origin_denied(self) -> None:
        self.assert_denied({**VALID, "data_origin": "customer"}, "synthetic data origins")

    def test_unapproved_provenance_denied(self) -> None:
        self.assert_denied({**VALID, "provenance": "trained"}, "provenance")

    def test_unhashable_provenance_denied(self) -> None:
        bad_values: tuple[object, ...] = ([], {}, ["synthetic"])
        for value in bad_values:
            self.assert_denied({**VALID, "provenance": value}, "provenance")

    def test_spoofed_loopback_url_denied(self) -> None:
        for url in (
            "http://127.0.0.1:80@external.example/weights",
            "http://127.0.0.1:80.example.com/x",
            "https://user:pass@example.com/x",
            "https://",
            "http://127.0.0.1:invalid/x",
        ):
            self.assert_denied({**VALID, "source_url": url}, "source_url")

    def test_unknown_field_denied(self) -> None:
        self.assert_denied({**VALID, "backdoor_url": "http://evil.example"}, "unknown field")

    def test_malformed_weights_hash_denied(self) -> None:
        for bad in ("XYZ", "0" * 63, "A" * 64, "0" * 64 + "\n", 12345):
            self.assert_denied({**VALID, "weights_sha256": bad}, "weights_sha256")

    def test_source_url_scheme_denied(self) -> None:
        for bad in ("http://example.com/m", "ftp://host/m", 123):
            self.assert_denied({**VALID, "source_url": bad}, "source_url")

    def test_padded_or_oversized_name_denied(self) -> None:
        self.assert_denied({**VALID, "name": " padded "}, "name")
        self.assert_denied({**VALID, "name": "n" * 129}, "name")

    def test_non_dict_manifest_denied(self) -> None:
        for bad in (None, "string", 42, ["list"]):
            result = licensing.evaluate(bad)
            self.assertEqual(result["decision"], "DENIED")
            self.assertFalse(result["promotion_allowed"])
            self.assertTrue(result["review_required"])

    def test_loopback_source_url_allowed(self) -> None:
        result = licensing.evaluate({**VALID, "source_url": "http://127.0.0.1:8080/weights"})
        self.assertEqual(result["decision"], "ALLOWED")

    def test_notes_validation(self) -> None:
        self.assertEqual(licensing.evaluate({**VALID, "notes": "Verified adapter.\nOk."})["decision"], "ALLOWED")
        self.assertEqual(licensing.evaluate({**VALID, "notes": "n" * 1024})["decision"], "ALLOWED")
        for bad_notes in ("n" * 1025, 123, ["a"], None, "bad\x00note"):
            self.assert_denied({**VALID, "notes": bad_notes}, "notes")

    def test_verify_weights_path_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "adapter.safetensors"
            payload = b"synthetic-weights-bytes-v1"
            weights.write_bytes(payload)
            digest_hex = hashlib.sha256(payload).hexdigest()

            ok = licensing.evaluate({**VALID, "weights_sha256": digest_hex}, verify_weights_path=weights)
            self.assertEqual(ok["decision"], "ALLOWED")
            self.assertTrue(ok["weights_verified"])

            mismatch = self.assert_denied(
                {**VALID, "weights_sha256": "1" * 64},
                "does not match weights_sha256",
                verify_weights_path=weights,
            )
            self.assertFalse(mismatch["weights_verified"])

            no_hash = {k: v for k, v in VALID.items() if k != "weights_sha256"}
            missing_hash = self.assert_denied(no_hash, "weights_sha256 is required", verify_weights_path=weights)
            self.assertFalse(missing_hash["weights_verified"])

            missing_file = self.assert_denied(
                {**VALID, "weights_sha256": digest_hex},
                "not found",
                verify_weights_path=Path(tmp) / "missing.bin",
            )
            self.assertFalse(missing_file["weights_verified"])

    def test_governance_tiers_and_allowed_data_origins(self) -> None:
        pilot_attestation = {
            "tier": "consented_pilot",
            "attestation_id": "att-2026-001",
            "approved_by": "security-board",
            "approved_at": "2026-11-12T00:00:00Z",
            "consent_scope": "pilot-synthetic-shadow",
        }
        pilot_manifest = {
            **VALID,
            "provenance": "public_research",
            "data_origin": "consented_pilot",
            "governance_attestation": pilot_attestation,
        }
        # Default evaluate() remains strict synthetic-only
        self.assert_denied(pilot_manifest, "synthetic data origins")

        allowed = licensing.evaluate(pilot_manifest, allowed_data_origins=("synthetic", "consented_pilot"))
        self.assertEqual(allowed["decision"], "ALLOWED")

        # Missing or mismatched governance_attestation fails closed
        no_att = {k: v for k, v in pilot_manifest.items() if k != "governance_attestation"}
        self.assert_denied(no_att, "governance_attestation", allowed_data_origins=("consented_pilot",))
        wrong_tier = {**pilot_manifest, "governance_attestation": {**pilot_attestation, "tier": "redacted_eval"}}
        self.assert_denied(wrong_tier, "tier must match", allowed_data_origins=("consented_pilot",))

        redacted_manifest = {
            **VALID,
            "provenance": "redacted_eval",
            "data_origin": "redacted_eval",
            "governance_attestation": {
                "tier": "redacted_eval",
                "attestation_id": "att-red-01",
                "approved_by": "privacy-lead",
                "approved_at": "2026-11-12T00:00:00Z",
                "redacted": True,
            },
        }
        self.assertEqual(
            licensing.evaluate(redacted_manifest, allowed_data_origins=("redacted_eval",))["decision"],
            "ALLOWED",
        )
        bad_redacted = {
            **redacted_manifest,
            "governance_attestation": {
                **dict(redacted_manifest["governance_attestation"]),  # type: ignore[arg-type]
                "redacted": False,
            },
        }
        self.assert_denied(bad_redacted, "redacted to be true", allowed_data_origins=("redacted_eval",))

        for bad_origins in ("synthetic", (), ("customer",)):
            self.assert_denied(dict(VALID), "allowed_data_origins", allowed_data_origins=bad_origins)


@unittest.skipUnless(signing.backend_available(), "signing extra unavailable")
class SigningKeyringAndPermissionTest(unittest.TestCase):
    def test_parse_keyring_and_load_policy_m_of_n_and_revocation(self) -> None:
        k1 = "a" * 64
        k2 = "b" * 64
        k3 = "c" * 64

        self.assertEqual(signing.normalize_trusted_keys([k1, {"public_key_hex": k1}, {"public_key_hex": k2}]), [k1, k2])
        with self.assertRaises(TypeError):
            signing.normalize_trusted_keys(123)  # type: ignore[arg-type]

        raw_policy = {
            "schema_version": "1.0",
            "required_signatures": 2,
            "required_roles": ["release", "security"],
            "trusted_keys": [
                {"public_key_hex": k1, "key_id": "rel-1", "role": "release"},
                {"public_key_hex": k2, "key_id": "sec-1", "role": "security"},
                {"public_key_hex": k3, "key_id": "sec-old", "role": "security", "revoked_at": "2026-11-01T00:00:00Z"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "keyring.json"
            policy_path.write_text(json.dumps(raw_policy), encoding="utf-8")
            policy = signing.load_keyring_policy(policy_path)

        self.assertEqual(policy["required_signatures"], 2)
        self.assertEqual(policy["required_roles"], ["release", "security"])
        self.assertEqual(policy["active_keys"], [k1, k2])
        self.assertEqual(policy["revoked_keys"], [k3])

        # Revoked key or duplicate single key cannot satisfy 2-of-2 + role quorum
        self.assertFalse(signing.check_keyring_quorum(policy, [k1, k3])["satisfied"])
        self.assertIn(k3, signing.check_keyring_quorum(policy, [k1, k3])["revoked_signers"])
        self.assertFalse(signing.check_keyring_quorum(policy, [k1, k1])["satisfied"])
        self.assertEqual(signing.check_keyring_quorum(policy, [k1])["missing_roles"], ["security"])

        ok = signing.check_keyring_quorum(policy, [k1, k2])
        self.assertTrue(ok["satisfied"])
        self.assertEqual(ok["valid_count"], 2)
        self.assertEqual(ok["roles_satisfied"], ["release", "security"])

    def test_parse_keyring_rejects_malformed_policies(self) -> None:
        k1 = "a" * 64
        for bad in (
            {"trusted_keys": [k1], "required_signatures": 0},
            {"trusted_keys": [k1], "required_signatures": True},
            {"trusted_keys": [k1], "required_roles": "release"},
            {"trusted_keys": [k1], "required_roles": [""]},
            {"trusted_keys": [k1, k1]},
            {"trusted_keys": [{"public_key_hex": k1, "revoked_at": ""}]},
            {"trusted_keys": [{"public_key_hex": k1, "revoked": True}]},
            {"trusted_keys": [{"public_key_hex": k1, "revoke_at": "2026-01-01T00:00:00Z"}]},
            {"trusted_keys": [k1], "unexpected_policy_flag": True},
        ):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                signing.parse_keyring(bad)

    def test_posix_mode_enforcement_unit_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            key_path = os.path.join(tmp, "k.pem")
            signing.generate_keypair(key_path)
            fake_stat = SimpleNamespace(st_mode=0o100644)
            with (
                patch("hardonian_foundry.signing.os.name", "posix"),
                patch("hardonian_foundry.signing.os.stat", return_value=fake_stat),
                self.assertRaisesRegex(ValueError, "0600"),
            ):
                signing.public_key_hex(key_path)

    @unittest.skipUnless(os.name == "posix", "POSIX chmod semantics only")
    def test_posix_chmod_644_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            key_path = os.path.join(tmp, "posix.pem")
            signing.generate_keypair(key_path)
            os.chmod(key_path, 0o600)
            self.assertEqual(len(signing.public_key_hex(key_path)), 64)
            os.chmod(key_path, 0o644)
            with self.assertRaises(ValueError):
                signing.sign(b"test", key_path)

    @unittest.skipUnless(os.name == "nt", "Windows NTFS ACL check only")
    def test_windows_acl_rejects_broad_grant_and_allows_owner_temp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            key_path = os.path.join(tmp, "win.pem")
            info = signing.generate_keypair(key_path)
            sig = signing.sign(b"payload", key_path)
            self.assertTrue(signing.verify(b"payload", sig, info["public_key_hex"]))

            # Granting Everyone read access must be rejected fail-closed
            subprocess.run(["icacls", key_path, "/grant", "*S-1-1-0:R"], check=True, stdout=subprocess.DEVNULL)
            with self.assertRaisesRegex(ValueError, "broad Windows ACL"):
                signing.sign(b"payload", key_path)

    @unittest.skipUnless(os.name == "nt", "Windows NTFS inherited ACL check only")
    def test_windows_generate_keypair_protects_against_inherited_broad_dir_acl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shared_parent = os.path.join(tmp, "shared_workspace")
            os.makedirs(shared_parent)
            # Simulate C:\ / CI workspace dir granting inherited read to BUILTIN\Users & Everyone
            subprocess.run(
                ["icacls", shared_parent, "/grant", "*S-1-5-32-545:(OI)(CI)R", "*S-1-1-0:(OI)(CI)R"],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            key_path = os.path.join(shared_parent, "ci-runner.pem")
            info = signing.generate_keypair(key_path)
            sig = signing.sign(b"ci-payload", key_path)
            self.assertTrue(signing.verify(b"ci-payload", sig, info["public_key_hex"]))


if __name__ == "__main__":
    unittest.main()
