"""Offline tests for canonical JSON serialization and deterministic synthetic baselines."""

from __future__ import annotations

import json
import math
import random
import unittest

from hardonian_foundry.baseline import (
    dataset,
    evaluate,
    evaluate_agents,
    reconciliation,
    split,
    verify_trace,
)
from hardonian_foundry.canonical import canonical, digest


class BaselineTest(unittest.TestCase):
    def test_arithmetic_and_review(self) -> None:
        for a in range(-100, 101, 7):
            for b in range(-100, 101, 11):
                r = reconciliation({"id": "synthetic-case", "expected_minor": a, "actual_minor": b, "critical": False})
                self.assertEqual(r["difference_minor"], b - a)
                self.assertEqual(r["category"], "matched" if a == b else "amount_mismatch")
                self.assertTrue(r["review_required"])

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            reconciliation({"id": "synthetic-a", "expected_minor": True, "actual_minor": 1, "critical": False})
        with self.assertRaises(ValueError):
            reconciliation({"id": "synthetic-a", "expected_minor": 1, "actual_minor": 1, "critical": "not-bool"})
        with self.assertRaises(ValueError):
            reconciliation({"id": "non-synthetic-id", "expected_minor": 1, "actual_minor": 1, "critical": False})
        with self.assertRaises(ValueError):
            reconciliation({"id": "synthetic-" + "x" * 101, "expected_minor": 1, "actual_minor": 1, "critical": False})
        with self.assertRaises(ValueError):
            reconciliation({"id": 12345, "expected_minor": 1, "actual_minor": 1, "critical": False})  # type: ignore[dict-item]
        with self.assertRaises(ValueError):
            reconciliation({"id": "a", "expected_minor": 1, "actual_minor": 1, "critical": False, "extra": 1})
        with self.assertRaises(ValueError):
            verify_trace("not-a-dict")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            verify_trace({"id": "synthetic-case", "tool": "read_record", "arguments": "not-a-dict"})  # type: ignore[dict-item]

    def test_policy_does_not_execute_and_blocks_adversarial_inputs(self) -> None:
        for tool in ("shell", "delete", "read_record ", "write_record", "READ_RECORD", "read\u200brecord", "bash"):
            r = verify_trace({"id": "synthetic-case", "tool": tool, "arguments": {"record_id": "x"}})
            self.assertEqual(r["policy_result"], "DENY")
            self.assertFalse(r["executed"])
        for bad_record_id in (
            "",
            "   ",
            "../etc/passwd",
            "..\\secret",
            "a/b",
            "x\x00y",
            "x; rm -rf /",
            ".hidden",
            "x" * 101,
            123,
            None,
        ):
            r = verify_trace({"id": "synthetic-case", "tool": "read_record", "arguments": {"record_id": bad_record_id}})
            self.assertEqual(r["policy_result"], "DENY")
            self.assertFalse(r["executed"])
        self.assertEqual(
            verify_trace({"id": "synthetic-case", "tool": "read_record", "arguments": {"record_id": "x"}})[
                "policy_result"
            ],
            "ALLOW_PROPOSAL",
        )
        self.assertEqual(
            verify_trace({"id": "synthetic-case", "tool": "read_record", "arguments": {"record_id": "x", "extra": 1}})[
                "policy_result"
            ],
            "DENY",
        )

    def test_split_reproducibility_and_leakage(self) -> None:
        a = split(dataset())
        self.assertEqual(a, split(dataset()))
        self.assertEqual([len(a[x]) for x in ("train", "validation", "test")], [36, 12, 12])
        ids = [{r["id"] for r in rows} for rows in a.values()]
        self.assertTrue(all(not (ids[i] & ids[j]) for i in range(3) for j in range(i + 1, 3)))
        with self.assertRaises(ValueError):
            split(dataset() + [dataset()[0]])
        with self.assertRaises(ValueError):
            split(dataset() + [{**dataset()[0], "id": "synthetic-duplicate"}])
        with self.assertRaises(ValueError):
            split(dataset()[:2])
        with self.assertRaises(ValueError):
            split(dataset(), seed="not-an-int")  # type: ignore[arg-type]
        self.assertEqual(split(dataset()), split(list(reversed(dataset()))))
        for n in range(3, 60):
            parts = split(dataset()[:n])
            self.assertTrue(all(parts.values()))
            self.assertEqual(sum(map(len, parts.values())), n)

    def test_synthetic_metrics(self) -> None:
        e = evaluate()
        self.assertEqual(e["data_origin"], "synthetic")
        self.assertEqual(e["test_count"], 12)
        self.assertEqual(e["review_rate"], 1.0)
        self.assertEqual(e["dataset_sha256"], digest(dataset()))
        agent = evaluate_agents()
        self.assertEqual(agent["unsafe_count"], 20)
        self.assertEqual(agent["unsafe_action_false_negative_rate"], 0.0)
        self.assertTrue(agent["review_required"])

    def test_canonical_edge_cases_and_fuzz_properties(self) -> None:
        # -0.0 normalization
        self.assertEqual(canonical(-0.0), "0.0")
        self.assertEqual(canonical(0.0), "0.0")
        self.assertEqual(digest(-0.0), digest(0.0))
        self.assertEqual(digest({"z": -0.0, "a": [-0.0]}), digest({"a": [0.0], "z": 0.0}))

        # NaN / Inf / -Inf rejected everywhere
        for bad_float in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                canonical({"x": [bad_float]})

        # Non-string dict keys rejected (including int, bool, float, None, tuple)
        for bad_dict in ({1: "a"}, {True: "a"}, {1.5: "a"}, {None: "a"}, {(1, 2): "a"}, {"ok": [{2: "b"}]}):
            with self.assertRaises(ValueError):
                canonical(bad_dict)

        # Unserializable objects raise TypeError
        for bad_type in (set(), {1, 2}, object(), lambda: None):
            with self.assertRaises(TypeError):
                canonical(bad_type)

        # Lone UTF-16 surrogates rejected in keys and values without UnicodeEncodeError
        for bad_surrogate in ("\ud800", "\udfff", {"\udc00": 1}, {"k": ["\ud83d"]}):
            with self.assertRaises(ValueError):
                digest(bad_surrogate)

        # Circular references rejected, while shared DAG references succeed
        cyclic_list: list[object] = []
        cyclic_list.append(cyclic_list)
        cyclic_dict: dict[str, object] = {}
        cyclic_dict["self"] = cyclic_dict
        mixed_dict: dict[str, object] = {}
        mixed_dict["t"] = (mixed_dict,)
        for cyc in (cyclic_list, cyclic_dict, mixed_dict):
            with self.assertRaises(ValueError):
                canonical(cyc)

        shared = [1, {"k": "v"}]
        self.assertEqual(canonical({"a": shared, "b": shared}), '{"a":[1,{"k":"v"}],"b":[1,{"k":"v"}]}')

        # Deep nesting handled iteratively without RecursionError
        deep: dict[str, object] = {"leaf": True}
        for _ in range(1200):
            deep = {"n": deep}
        encoded_deep = canonical(deep)
        self.assertTrue(encoded_deep.startswith('{"n":' * 1200))

        # Property/fuzz test across 150 randomized nested structures
        rng = random.Random(20261112)

        def rand_val(depth: int = 0) -> object:
            kind = rng.randint(0, 6 if depth < 3 else 4)
            if kind == 0:
                return None
            if kind == 1:
                return rng.choice([True, False])
            if kind == 2:
                return rng.randint(-10**9, 10**9)
            if kind == 3:
                return rng.choice([-0.0, 0.0, 1.5, -3.25, math.pi])
            if kind == 4:
                return rng.choice(["abc", "café", "日本語", "line\n\t\"\\", "\U0001f512"])
            if kind == 5:
                return [rand_val(depth + 1) for _ in range(rng.randint(0, 4))]
            keys = [f"k{rng.randint(0, 20)}" for _ in range(rng.randint(0, 5))]
            return {k: rand_val(depth + 1) for k in keys}

        for _ in range(150):
            obj = rand_val()
            ser = canonical(obj)
            reparsed = json.loads(ser)
            self.assertEqual(canonical(reparsed), ser)
            self.assertEqual(digest(reparsed), digest(obj))
            if isinstance(obj, dict):
                shuffled_keys = list(obj.keys())
                rng.shuffle(shuffled_keys)
                shuffled = {k: obj[k] for k in shuffled_keys}
                self.assertEqual(canonical(shuffled), ser)
                self.assertEqual(digest(shuffled), digest(obj))


if __name__ == "__main__":
    unittest.main()
