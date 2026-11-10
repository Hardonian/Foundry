"""Offline tests for the fail-closed licensing validator."""
import unittest

from hardonian_foundry import licensing

VALID = {
    "name": "synthetic-reviewer",
    "license_id": "Apache-2.0",
    "provenance": "synthetic",
    "data_origin": "synthetic",
    "weights_sha256": "0" * 64,
}


class LicensingTest(unittest.TestCase):
    def assert_denied(self, manifest, fragment):
        result = licensing.evaluate(manifest)
        self.assertEqual(result["decision"], "DENIED")
        self.assertFalse(result["promotion_allowed"])
        self.assertTrue(result["review_required"])
        self.assertTrue(any(fragment in reason for reason in result["reasons"]), result["reasons"])

    def test_valid_manifest_allowed_but_never_promotable(self):
        result = licensing.evaluate(dict(VALID))
        self.assertEqual(result["decision"], "ALLOWED")
        self.assertEqual(result["reasons"], [])
        self.assertFalse(result["promotion_allowed"])
        self.assertTrue(result["review_required"])

    def test_unknown_license_denied(self):
        self.assert_denied({**VALID, "license_id": "LicenseRef-Custom"}, "allowlist")

    def test_missing_fields_denied(self):
        for field in ("name", "license_id", "provenance", "data_origin"):
            manifest = {k: v for k, v in VALID.items() if k != field}
            self.assert_denied(manifest, f"missing required field: {field}")

    def test_non_synthetic_data_origin_denied(self):
        self.assert_denied({**VALID, "data_origin": "customer"}, "synthetic data origins")

    def test_unapproved_provenance_denied(self):
        self.assert_denied({**VALID, "provenance": "trained"}, "provenance")

    def test_unhashable_provenance_denied(self):
        for value in ([], {}, ["synthetic"]):
            self.assert_denied({**VALID, "provenance": value}, "provenance")

    def test_spoofed_loopback_url_denied(self):
        for url in ("http://127.0.0.1:80@external.example/weights", "http://127.0.0.1:80.example.com/x",
                    "https://user:pass@example.com/x", "https://", "http://127.0.0.1:invalid/x"):
            self.assert_denied({**VALID, "source_url": url}, "source_url")

    def test_unknown_field_denied(self):
        self.assert_denied({**VALID, "backdoor_url": "http://evil.example"}, "unknown field")

    def test_malformed_weights_hash_denied(self):
        for bad in ("XYZ", "0" * 63, "A" * 64, "0" * 64 + "\n", 12345):
            self.assert_denied({**VALID, "weights_sha256": bad}, "weights_sha256")

    def test_source_url_scheme_denied(self):
        for bad in ("http://example.com/m", "ftp://host/m", 123):
            self.assert_denied({**VALID, "source_url": bad}, "source_url")

    def test_padded_or_oversized_name_denied(self):
        self.assert_denied({**VALID, "name": " padded "}, "name")
        self.assert_denied({**VALID, "name": "n" * 129}, "name")

    def test_non_dict_manifest_denied(self):
        for bad in (None, "string", 42, ["list"]):
            result = licensing.evaluate(bad)
            self.assertEqual(result["decision"], "DENIED")
            self.assertFalse(result["promotion_allowed"])

    def test_loopback_source_url_allowed(self):
        result = licensing.evaluate({**VALID, "source_url": "http://127.0.0.1:8080/weights"})
        self.assertEqual(result["decision"], "ALLOWED")


if __name__ == "__main__":
    unittest.main()
