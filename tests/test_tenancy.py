"""Offline tests for fail-closed tenant isolation on synthetic records and runs."""

from __future__ import annotations

import dataclasses
import unittest

from hardonian_foundry import tenancy
from hardonian_foundry.tenancy import TenantContext

CTX = TenantContext(tenant_id="t-a1", request_id="r-1")


class TenancyTest(unittest.TestCase):
    def test_exact_tenant_match_allowed(self) -> None:
        result = tenancy.read(CTX, {"tenant_id": "t-a1", "value_minor": 1})
        self.assertEqual(result["status"], "ALLOWED")
        self.assertTrue(result["review_required"])

    def test_cross_tenant_denied(self) -> None:
        result = tenancy.read(CTX, {"tenant_id": "t-b2", "value_minor": 1})
        self.assertEqual(result["status"], "DENIED")

    def test_malformed_or_missing_records_denied(self) -> None:
        for bad in (None, "string", {}, {"tenant_id": None}):
            self.assertEqual(tenancy.read(CTX, bad)["status"], "DENIED")

    def test_non_context_principal_denied(self) -> None:
        for bad in (None, "t-a1", 42, {"tenant_id": "t-a1"}):
            self.assertEqual(tenancy.read(bad, {"tenant_id": "t-a1"})["status"], "DENIED")

    def test_denial_reason_is_uniform(self) -> None:
        denials = [
            tenancy.read(CTX, {"tenant_id": "t-other"}),
            tenancy.read(CTX, {}),
            tenancy.read(None, {"tenant_id": "t-a1"}),
            tenancy.authorize_run(CTX, {"run_id": "run-1", "tenant_id": "t-other"}),
            tenancy.authorize_run(CTX, {"run_id": "run-1"}),
        ]
        reasons = {result["reason"] for result in denials}
        self.assertEqual(reasons, {tenancy.DENY_REASON})

    def test_invalid_identifiers_rejected_at_construction(self) -> None:
        for tenant in ("", "U", "-lead", "x" * 65, None, 5):
            with self.assertRaises(ValueError):
                TenantContext(tenant_id=tenant, request_id="r-1")  # type: ignore[arg-type]
        for request in ("", "R-1", None):
            with self.assertRaises(ValueError):
                TenantContext(tenant_id="t-a1", request_id=request)  # type: ignore[arg-type]

    def test_context_is_immutable(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            CTX.tenant_id = "t-b2"  # type: ignore[misc]

    def test_filter_records_keeps_only_own_tenant(self) -> None:
        rows = [
            {"tenant_id": "t-a1", "id": 1},
            {"tenant_id": "t-b2", "id": 2},
            {"id": 3},
            "junk",
        ]
        kept = tenancy.filter_records(CTX, rows)
        self.assertEqual(kept, [{"tenant_id": "t-a1", "id": 1}])

    def test_filter_records_rejects_non_list(self) -> None:
        self.assertEqual(tenancy.filter_records(CTX, "not-a-list"), [])
        self.assertEqual(tenancy.filter_records(CTX, None), [])

    def test_authorize_run_matches_and_fails_closed(self) -> None:
        self.assertEqual(
            tenancy.authorize_run(CTX, {"run_id": "r1", "tenant_id": "t-a1"})["status"],
            "ALLOWED",
        )
        self.assertEqual(
            tenancy.authorize_run(CTX, {"run_id": "r1", "provenance": {"tenant_id": "t-a1"}})["status"],
            "ALLOWED",
        )
        self.assertEqual(
            tenancy.authorize_run(
                CTX,
                {"run_id": "r1", "tenant_id": "t-a1", "provenance": {"tenant_id": "t-a1"}},
            )["status"],
            "ALLOWED",
        )
        # Split-brain mismatch, unscoped, cross-tenant, malformed inputs all fail closed
        for bad in (
            None,
            "not-a-dict",
            {"run_id": "r1"},
            {"run_id": "r1", "tenant_id": "t-b2"},
            {"run_id": "r1", "provenance": {"tenant_id": "t-b2"}},
            {"run_id": "r1", "tenant_id": "t-a1", "provenance": {"tenant_id": "t-b2"}},
            {"run_id": "r1", "tenant_id": "INVALID"},
            {"run_id": "r1", "provenance": "bad", "tenant_id": "t-a1"},
        ):
            res = tenancy.authorize_run(CTX, bad)
            self.assertEqual(res["status"], "DENIED")
            self.assertEqual(res["reason"], tenancy.DENY_REASON)
            self.assertTrue(res["review_required"])
        self.assertEqual(tenancy.authorize_run("bad-ctx", {"tenant_id": "t-a1"})["status"], "DENIED")

    def test_scope_ledger_query_filters_and_fails_closed(self) -> None:
        entries = [
            {"seq": 1, "record": {"tenant_id": "t-a1", "data_origin": "synthetic"}},
            {"seq": 2, "record": {"tenant_id": "t-b2", "data_origin": "synthetic"}},
            {"seq": 3, "record": {"data_origin": "synthetic"}},
            {"seq": 4, "tenant_id": "t-a1", "record": {"tenant_id": "t-b2"}},
            {"seq": 5, "tenant_id": "t-a1"},  # missing record dict
            "malformed",
        ]
        scoped = tenancy.scope_ledger_query(CTX, entries)
        self.assertEqual(scoped, [{"seq": 1, "record": {"tenant_id": "t-a1", "data_origin": "synthetic"}}])
        self.assertEqual(tenancy.scope_ledger_query(None, entries), [])
        self.assertEqual(tenancy.scope_ledger_query(CTX, "not-a-list"), [])


if __name__ == "__main__":
    unittest.main()
