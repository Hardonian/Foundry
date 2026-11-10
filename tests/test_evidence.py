"""Offline tests for the tamper-evident evidence chain (synthetic records only)."""
import json
import os
import tempfile
import unittest

from hardonian_foundry import evidence


def record(n):
    return {"data_origin": "synthetic", "value_minor": n, "note": f"entry-{n}"}


class EvidenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "ledger.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_append_and_verify_chain(self):
        first = evidence.append(self.path, record(1), key="alpha")
        second = evidence.append(self.path, record(2), key="beta")
        self.assertEqual(first["seq"], 1)
        self.assertEqual(first["prev_hash"], evidence.GENESIS)
        self.assertEqual(second["prev_hash"], first["entry_hash"])
        report = evidence.verify(self.path)
        self.assertEqual(report["status"], "OK")
        self.assertEqual(report["count"], 2)

    def test_tampered_record_detected(self):
        evidence.append(self.path, record(1))
        evidence.append(self.path, record(2))
        with open(self.path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        entry = json.loads(lines[0])
        entry["record"]["value_minor"] = 999
        lines[0] = json.dumps(entry, sort_keys=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        report = evidence.verify(self.path)
        self.assertEqual(report["status"], "TAMPERED")
        self.assertEqual(report["count"], 0)

    def test_reordered_entries_detected(self):
        for index in (1, 2, 3):
            evidence.append(self.path, record(index))
        with open(self.path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        self.assertEqual(evidence.verify(self.path)["status"], "TAMPERED")

    def test_duplicate_key_rejected(self):
        evidence.append(self.path, record(1), key="alpha")
        with self.assertRaises(ValueError):
            evidence.append(self.path, record(2), key="alpha")

    def test_truncation_detected_only_with_anchor(self):
        evidence.append(self.path, record(1))
        evidence.append(self.path, record(2))
        anchor = evidence.head(self.path)
        with open(self.path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()[:1]
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        self.assertEqual(evidence.verify(self.path)["status"], "OK")  # chain alone sees nothing
        report = evidence.verify(self.path, expected_head=anchor)
        self.assertEqual(report["status"], "TAMPERED")

    def test_append_refuses_broken_chain(self):
        evidence.append(self.path, record(1))
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write("{not json}\n")
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            evidence.append(self.path, record(2))

    def test_invalid_records_rejected_before_write(self):
        for bad in ([], "string", {"no_origin": 1}, {"data_origin": "synthetic", "x": float("nan")}):
            with self.assertRaises(ValueError):
                evidence.append(self.path, bad)
        self.assertFalse(os.path.exists(self.path))

    def test_bad_keys_rejected(self):
        for bad in ("", "  ", "x" * 129):
            with self.assertRaises(ValueError):
                evidence.append(self.path, record(1), key=bad)

    def test_no_tmp_file_left_behind(self):
        evidence.append(self.path, record(1))
        self.assertFalse(os.path.exists(self.path + ".tmp"))


if __name__ == "__main__":
    unittest.main()
