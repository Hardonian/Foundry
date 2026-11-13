"""Deep unit tests for hardonian_foundry.provider covering circuit breakers, connection pooling,
auth resolution, URL validation, batching, transport errors, and persistence."""
from __future__ import annotations

import http.client
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from hardonian_foundry.provider import (
    CircuitBreaker,
    ProviderError,
    _ActiveRequest,
    _audit_hash,
    _build_ssl_context,
    _classify_exception,
    _ConnectionPool,
    _extract_chat_content,
    _request,
    _resolve_bearer_header,
    predict_batch,
    suggest,
    validate_loopback_url,
)


class ProviderDeepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_circuit_breaker_lifecycle(self) -> None:
        with self.assertRaises(ValueError):
            CircuitBreaker(failure_threshold=0)
        with self.assertRaises(ValueError):
            CircuitBreaker(recovery_timeout=-1.0)

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        self.assertEqual(cb.state, "CLOSED")
        self.assertTrue(cb.allow_request())

        # 1 failure: stays closed
        cb.record_failure()
        self.assertEqual(cb.state, "CLOSED")
        self.assertTrue(cb.allow_request())

        # 2 failures: opens
        cb.record_failure()
        self.assertEqual(cb.state, "OPEN")
        self.assertFalse(cb.allow_request())

        # Wait recovery timeout: transitions to HALF_OPEN
        time.sleep(0.12)
        self.assertEqual(cb.state, "HALF_OPEN")
        # In OPEN state, allow_request() transitions to HALF_OPEN when elapsed >= recovery_timeout
        self.assertTrue(cb.allow_request())

        # Failure in half-open immediately returns to OPEN
        cb.record_failure()
        self.assertEqual(cb.state, "OPEN")

        # Reset puts it back in CLOSED
        cb.reset()
        self.assertEqual(cb.state, "CLOSED")
        self.assertTrue(cb.allow_request())

        # Success resets failure count
        cb.record_failure()
        cb.record_success()
        self.assertEqual(cb.state, "CLOSED")

    def test_connection_pool(self) -> None:
        pool = _ConnectionPool(max_per_origin=2)
        # Acquire HTTP connection
        conn1, reused1 = pool.acquire("http", "127.0.0.1", 8080, 5.0, None, None)
        self.assertFalse(reused1)
        self.assertIsInstance(conn1, http.client.HTTPConnection)

        # Acquire HTTPS connection
        conn_https, _ = pool.acquire("https", "127.0.0.1", 8443, 5.0, None, None)
        self.assertIsInstance(conn_https, http.client.HTTPSConnection)

        # Release without socket does nothing
        conn1.sock = None
        pool.release("http", "127.0.0.1", 8080, None, conn1)

        # Mock an active socket on conn1 and release
        conn1.sock = MagicMock()
        pool.release("http", "127.0.0.1", 8080, None, conn1)

        # Acquire again gets reused conn
        conn_reused, reused2 = pool.acquire("http", "127.0.0.1", 8080, 5.0, None, None)
        self.assertTrue(reused2)
        self.assertIs(conn_reused, conn1)

        # Release more than max_per_origin
        conn_a = http.client.HTTPConnection("127.0.0.1", 8080)
        conn_a.sock = MagicMock()
        conn_b = http.client.HTTPConnection("127.0.0.1", 8080)
        conn_b.sock = MagicMock()
        conn_c = http.client.HTTPConnection("127.0.0.1", 8080)
        conn_c.sock = MagicMock()
        pool.release("http", "127.0.0.1", 8080, None, conn_a)
        pool.release("http", "127.0.0.1", 8080, None, conn_b)
        pool.release("http", "127.0.0.1", 8080, None, conn_c)  # overflows max=2, closed

        pool.close_all()

    def test_active_request_timeout(self) -> None:
        mock_conn = MagicMock()
        mock_sock = MagicMock()
        mock_conn.sock = mock_sock

        req = _ActiveRequest(mock_conn)
        self.assertFalse(req.done)
        self.assertFalse(req.timed_out)

        req.abort_timeout()
        self.assertTrue(req.done)
        self.assertTrue(req.timed_out)
        mock_sock.close.assert_called_once()

        # Calling abort_timeout again after done is a no-op
        req.abort_timeout()

    def test_validate_loopback_url(self) -> None:
        # Valid numeric loopback URLs (origins without path/query)
        validate_loopback_url("http://127.0.0.1:8000")
        validate_loopback_url("http://[::1]:9000")

        # Localhost name rejected without numeric loopback (anti-DNS-rebinding)
        with self.assertRaises(ValueError):
            validate_loopback_url("http://localhost:9000")

        # URLs with path/query
        with self.assertRaises(ValueError):
            validate_loopback_url("http://127.0.0.1:8000/v1/chat")

        # Invalid schemes
        with self.assertRaises(ValueError):
            validate_loopback_url("ftp://127.0.0.1:8000")
        with self.assertRaises(ValueError):
            validate_loopback_url("file:///etc/passwd")

        # Remote non-loopback addresses
        with self.assertRaises(ValueError):
            validate_loopback_url("http://192.168.1.1:8000")
        with self.assertRaises(ValueError):
            validate_loopback_url("http://8.8.8.8:8000")
        with self.assertRaises(ValueError):
            validate_loopback_url("http://google.com:8000")

    def test_resolve_bearer_header(self) -> None:
        # None returns empty dict
        self.assertEqual(_resolve_bearer_header(None), {})

        # Invalid env var name
        with self.assertRaises(ValueError):
            _resolve_bearer_header("invalid-name-with-hyphen")

        # Unset env var raises ProviderError
        with self.assertRaises(ProviderError):
            _resolve_bearer_header("_FOUNDRY_UNSET_VAR_12345")

        # Set env var
        os.environ["_FOUNDRY_TEST_TOKEN"] = "secret-token-xyz"
        try:
            headers = _resolve_bearer_header("_FOUNDRY_TEST_TOKEN")
            self.assertEqual(headers, {"Authorization": "Bearer secret-token-xyz"})
        finally:
            os.environ.pop("_FOUNDRY_TEST_TOKEN", None)

    def test_request_url_validation_errors(self) -> None:
        # Invalid host or port
        with self.assertRaises(ValueError) as ctx:
            _request("http://", b"{}", 1.0)
        self.assertIn("invalid provider request URL", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            _request("http://no-port", b"{}", 1.0)
        self.assertIn("invalid provider request URL", str(ctx2.exception))

    def test_predict_batch_validation_and_circuit_open(self) -> None:
        # Empty batch raises ValueError
        with self.assertRaises(ValueError):
            predict_batch([], labels=["A", "B"], origin="http://127.0.0.1:8000", model="test")

        # Bad labels
        with self.assertRaises(ValueError):
            predict_batch([{"id": "c1", "prompt": "p"}], labels=[], origin="http://127.0.0.1:8000", model="test")

        # Bad fallback_label
        with self.assertRaises(ValueError):
            predict_batch(
                [{"id": "c1", "prompt": "p"}],
                labels=["A", "B"],
                fallback_label="UNKNOWN_FALLBACK",
                origin="http://127.0.0.1:8000",
                model="test",
            )

        # Bad model
        with self.assertRaises(ValueError):
            predict_batch(
                [{"id": "c1", "prompt": "p"}],
                labels=["A", "B"],
                origin="http://127.0.0.1:8000",
                model="",
            )

        # Bad timeout
        with self.assertRaises(ValueError):
            predict_batch(
                [{"id": "c1", "prompt": "p"}],
                labels=["A", "B"],
                origin="http://127.0.0.1:8000",
                model="test",
                timeout=0,
            )

        # Bad record
        with self.assertRaises(ValueError):
            predict_batch(
                ["not-a-dict"],  # type: ignore[list-item]
                labels=["A", "B"],
                origin="http://127.0.0.1:8000",
                model="test",
            )

        # Circuit open falls back safely to UNDETERMINED for all records
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        self.assertEqual(cb.state, "OPEN")

        records = [{"id": "c1", "prompt": "test exception case", "critical": False}]
        res = predict_batch(
            records,
            labels=["A", "B", "UNDETERMINED"],
            origin="http://127.0.0.1:8000",
            model="test-model",
            circuit_breaker=cb,
            fallback_label="UNDETERMINED",
        )
        self.assertEqual(res["total"], 1)
        self.assertEqual(res["fallback_count"], 1)
        self.assertEqual(res["predictions"][0]["label"], "UNDETERMINED")

    def test_suggest_and_batch_httperror_fallback(self) -> None:
        import io
        import urllib.error

        def failing_transport(*args, **kwargs):
            fp = io.BytesIO(b"internal server error")
            raise urllib.error.HTTPError("http://127.0.0.1:8000", 500, "Internal Server Error", {}, fp)

        cb = CircuitBreaker(failure_threshold=2)
        res = suggest(
            {"id": "synthetic-case-500", "expected_minor": 100, "actual_minor": 90, "critical": False},
            origin="http://127.0.0.1:8000",
            model="test-model",
            transport=failing_transport,
            circuit_breaker=cb,
        )
        self.assertEqual(res["status"], "UNAVAILABLE")
        self.assertEqual(res["http_status"], 500)
        self.assertEqual(cb.state, "CLOSED")

        # predict_batch with failing transport and saving output to file
        preds_file = self.root / "preds.jsonl"
        res_batch = predict_batch(
            [{"id": "c1", "prompt": "case 1", "critical": False}],
            labels=["PASS", "FAIL", "UNDETERMINED"],
            origin="http://127.0.0.1:8000",
            model="test-model",
            transport=failing_transport,
            fallback_label="UNDETERMINED",
            out_path=preds_file,
        )
        self.assertEqual(res_batch["fallback_count"], 1)
        self.assertEqual(res_batch["predictions"][0]["label"], "UNDETERMINED")
        self.assertTrue(preds_file.is_file())

    def test_suggest_and_batch_success_paths(self) -> None:
        def success_suggest_transport(*args, **kwargs):
            return {"choices": [{"message": {"content": '{"category": "matched"}'}}]}

        cb = CircuitBreaker(failure_threshold=2)
        res = suggest(
            {"id": "synthetic-case-ok", "expected_minor": 100, "actual_minor": 100, "critical": False},
            origin="http://127.0.0.1:8000",
            model="test-model",
            transport=success_suggest_transport,
            circuit_breaker=cb,
        )
        self.assertEqual(res["status"], "REVIEW_REQUIRED")
        self.assertEqual(res["category"], "matched")
        self.assertTrue(res["arithmetic_agrees"])
        self.assertTrue(res["review_required"])

        # Success predict_batch with schema and allowlist validation
        def success_batch_transport(*args, **kwargs):
            return {"choices": [{"message": {"content": '{"label": "PASS", "confidence": 0.99}'}}]}

        res_batch = predict_batch(
            [{"id": "c1", "prompt": "case 1", "critical": False}],
            labels=["PASS", "FAIL", "UNDETERMINED"],
            origin="http://127.0.0.1:8000",
            model="test-model",
            transport=success_batch_transport,
            circuit_breaker=cb,
        )
        self.assertEqual(res_batch["status"], "COMPLETED")
        self.assertEqual(res_batch["fallback_count"], 0)
        self.assertEqual(res_batch["predictions"][0]["label"], "PASS")

        # Invalid prediction JSON schema fallback
        def invalid_schema_transport(*args, **kwargs):
            return {"choices": [{"message": {"content": '{"label": "PASS", "unknown_key": 1}'}}]}

        res_invalid = predict_batch(
            [{"id": "c1", "prompt": "case 1", "critical": False}],
            labels=["PASS", "FAIL", "UNDETERMINED"],
            origin="http://127.0.0.1:8000",
            model="test-model",
            transport=invalid_schema_transport,
            circuit_breaker=cb,
        )
        self.assertEqual(res_invalid["status"], "DEGRADED_FALLBACK")
        self.assertEqual(res_invalid["fallback_count"], 1)

    def test_extract_chat_content_formats(self) -> None:
        # OpenAI style
        openai_resp = {"choices": [{"message": {"content": "prediction A"}}]}
        self.assertEqual(_extract_chat_content(openai_resp), "prediction A")

        # Non-dict envelope raises TypeError
        with self.assertRaises(TypeError):
            _extract_chat_content("not-a-dict")

        # Malformed choices raises ValueError
        for bad_choices in (
            {},
            {"choices": []},
            {"choices": [{"message": "not-a-dict"}]},
            {"choices": [{"message": {"content": 123}}]},
        ):
            with self.assertRaises(ValueError):
                _extract_chat_content(bad_choices)

    def test_build_ssl_context_and_classify_exception(self) -> None:
        import ssl

        # Default SSL context
        ctx = _build_ssl_context(None)
        self.assertIsInstance(ctx, ssl.SSLContext)

        # Classification
        status, reason, code = _classify_exception(TimeoutError("timed out"))
        self.assertEqual(status, "TIMEOUT")
        self.assertIsNone(code)

        status_ref, _, _ = _classify_exception(ConnectionRefusedError("connection refused"))
        self.assertEqual(status_ref, "CONNECTION_ERROR")

        # ProviderError classification
        p_err = ProviderError("CUSTOM_REASON", "custom message", http_status=418)
        c_status, c_reason, c_code = _classify_exception(p_err)
        self.assertEqual(c_status, "CUSTOM_REASON")
        self.assertEqual(c_code, 418)

        # Unknown exception fallback
        u_status, _, _ = _classify_exception(KeyboardInterrupt())
        self.assertEqual(u_status, "UNAVAILABLE")

        # Audit hash helper
        ah = _audit_hash(origin="http://127.0.0.1:8000", model="test-model", case_id="c1", status="OK")
        self.assertEqual(len(ah), 64)


if __name__ == "__main__":
    unittest.main()
