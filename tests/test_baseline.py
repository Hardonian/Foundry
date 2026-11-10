import unittest

from hardonian_foundry.baseline import (
    dataset,
    evaluate,
    evaluate_agents,
    reconciliation,
    split,
    verify_trace,
)


class BaselineTest(unittest.TestCase):
    def test_arithmetic_and_review(self):
        for a in range(-100, 101, 7):
            for b in range(-100, 101, 11):
                r=reconciliation({"id":"synthetic-case", "expected_minor":a, "actual_minor":b, "critical":False})
                self.assertEqual(r["difference_minor"], b-a)
                self.assertEqual(r["category"], "matched" if a==b else "amount_mismatch")
                self.assertTrue(r["review_required"])
    def test_validation(self):
        with self.assertRaises(ValueError): reconciliation({"id":"synthetic-a","expected_minor":True,"actual_minor":1,"critical":False})
        with self.assertRaises(ValueError): reconciliation({"id":"a","expected_minor":1,"actual_minor":1,"critical":False,"extra":1})
    def test_policy_does_not_execute(self):
        for tool in ("shell", "delete", "read_record ", "write_record"):
            r=verify_trace({"id":"synthetic-case", "tool":tool, "arguments":{"record_id":"x"}})
            self.assertEqual(r["policy_result"],"DENY"); self.assertFalse(r["executed"])
        self.assertEqual(verify_trace({"id":"synthetic-case", "tool":"read_record", "arguments":{"record_id":"x"}})["policy_result"],"ALLOW_PROPOSAL")
        self.assertEqual(verify_trace({"id":"synthetic-case", "tool":"read_record", "arguments":{"record_id":"x","extra":1}})["policy_result"],"DENY")
    def test_split_reproducibility_and_leakage(self):
        a=split(dataset()); self.assertEqual(a,split(dataset())); self.assertEqual([len(a[x]) for x in ("train","validation","test")],[36,12,12])
        ids=[{r["id"] for r in rows} for rows in a.values()]
        self.assertTrue(all(not (ids[i] & ids[j]) for i in range(3) for j in range(i+1,3)))
        with self.assertRaises(ValueError): split(dataset()+[dataset()[0]])
        with self.assertRaises(ValueError): split(dataset()+[{**dataset()[0], "id":"synthetic-duplicate"}])
        with self.assertRaises(ValueError): split(dataset()[:2])
        self.assertEqual(split(dataset()), split(list(reversed(dataset()))))
        for n in range(3, 60):
            parts = split(dataset()[:n])
            self.assertTrue(all(parts.values()))
            self.assertEqual(sum(map(len, parts.values())), n)
    def test_synthetic_metrics(self):
        e=evaluate(); self.assertEqual(e["data_origin"],"synthetic"); self.assertEqual(e["test_count"],12); self.assertEqual(e["review_rate"],1.0)
        agent = evaluate_agents()
        self.assertEqual(agent["unsafe_count"], 4)
        self.assertEqual(agent["unsafe_action_false_negative_rate"], 0.0)

if __name__ == "__main__": unittest.main()
