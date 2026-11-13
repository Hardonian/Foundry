import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from hardonian_foundry import provider
from hardonian_foundry.provider import (
    CircuitBreaker,
    predict_batch,
    suggest,
    validate_loopback_url,
)

CASE = {"id": "synthetic-provider", "expected_minor": 100, "actual_minor": 99, "critical": False}


class Handler(BaseHTTPRequestHandler):
    mode = "good"
    last_auth_header: str | None = None

    def log_message(self, *_args):
        pass

    def do_POST(self):
        Handler.last_auth_header = self.headers.get("Authorization")
        self.rfile.read(int(self.headers["Content-Length"]))
        if self.mode == "redirect":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/forbidden")
            self.end_headers()
            return
        if self.mode == "truncated_chunk":
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(b"20\r\npartial")
            self.wfile.flush()
            self.close_connection = True
            return
        if self.mode == "error":
            self.send_error(501)
            return
        if self.mode == "nested":
            response = {"choices": [{"message": {"content": "[" * 1100 + "]" * 1100}}]}
        else:
            response = {"choices": [{"message": {"content": json.dumps({"category": "amount_mismatch"})}}]}
        body = json.dumps(response).encode()
        if self.mode == "trickle":
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                for byte in body:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ProviderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.origin = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def tearDown(self):
        Handler.mode = "good"

    def test_real_loopback_transport(self):
        result = suggest(CASE, origin=self.origin, model="synthetic-model")
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertTrue(result["arithmetic_agrees"])
        self.assertTrue(result["review_required"])

    def test_fail_closed_redirect_and_501(self):
        for mode in ("redirect", "error", "truncated_chunk", "nested"):
            with self.subTest(mode=mode):
                Handler.mode = mode
                result = suggest(CASE, origin=self.origin, model="synthetic-model")
                self.assertEqual(result["status"], "UNAVAILABLE")
                self.assertNotIn("category", result)

    def test_trickle_response_has_total_deadline(self):
        Handler.mode = "trickle"
        started = time.monotonic()
        result = suggest(CASE, origin=self.origin, model="synthetic-model", timeout=0.2)
        self.assertLess(time.monotonic() - started, 2.5)
        self.assertEqual(result["status"], "UNAVAILABLE")

    def test_decoder_recursion_fails_closed(self):
        response = {"choices": [{"message": {"content": "[]"}}]}
        with patch.object(provider.json, "loads", side_effect=RecursionError("deep JSON")):
            result = suggest(CASE, origin=self.origin, model="synthetic-model", transport=lambda *_: response)
        self.assertEqual(result["status"], "UNAVAILABLE")

    def test_bad_schema_and_timeout(self):
        def invalid(*_):
            return {"choices": [{"message": {"content": '{"category":"matched", "extra":1}'}}]}
        def timeout(*_):
            raise TimeoutError()
        for transport in (invalid, timeout):
            result = suggest(CASE, origin=self.origin, model="synthetic-model", transport=transport)
            self.assertEqual(result["status"], "UNAVAILABLE")

    def test_no_remote_url(self):
        for url in ("https://127.0.0.1:1", "http://localhost:1", "http://8.8.8.8:1234", "http://127.0.0.1:1/path", "http://user:pass@127.0.0.1:1", "http://127.0.0.1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_loopback_url(url)

    def test_invalid_inputs_do_not_reach_transport(self):
        def unexpected(*_):
            self.fail("transport should not be called")
        with self.assertRaises(ValueError):
            suggest({**CASE, "actual_minor": True}, origin=self.origin, model="x", transport=unexpected)
        with self.assertRaises(ValueError):
            suggest(CASE, origin=self.origin, model="x", timeout=99, transport=unexpected)

    def test_disagreement_is_review_not_approval(self):
        def wrong(*_):
            return {"choices": [{"message": {"content": '{"category":"matched"}'}}]}
        result = suggest(CASE, origin=self.origin, model="x", transport=wrong)
        self.assertFalse(result["arithmetic_agrees"])
        self.assertTrue(result["review_required"])

    def test_fast_pool_transport_audit_hash_and_latency(self):
        Handler.mode = "good"
        t0 = time.monotonic()
        for _ in range(10):
            res = suggest(CASE, origin=self.origin, model="local-model", timeout=1.5)
            self.assertEqual(res["status"], "REVIEW_REQUIRED")
            self.assertEqual(len(res["audit_sha256"]), 64)
            self.assertGreaterEqual(res["latency_ms"], 0.0)
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 2.0)

    def test_structured_unavailable_reasons_and_circuit_breaker(self):
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=5.0)
        Handler.mode = "error"
        r1 = suggest(CASE, origin=self.origin, model="x", circuit_breaker=breaker)
        r2 = suggest(CASE, origin=self.origin, model="x", circuit_breaker=breaker)
        self.assertEqual(r1["status"], "UNAVAILABLE")
        self.assertEqual(r1["reason_code"], "HTTP_ERROR")
        self.assertEqual(r2["reason_code"], "HTTP_ERROR")

        # 3rd call trips open circuit without touching server
        Handler.mode = "good"
        r3 = suggest(CASE, origin=self.origin, model="x", circuit_breaker=breaker)
        self.assertEqual(r3["status"], "UNAVAILABLE")
        self.assertEqual(r3["reason_code"], "CIRCUIT_OPEN")
        self.assertTrue(r3["review_required"])

    def test_opt_in_tls_remote_and_env_token(self):
        with self.assertRaises(ValueError):
            validate_loopback_url("https://api.example.com:8443")
        with self.assertRaises(ValueError):
            validate_loopback_url("https://127.0.0.1:8443")
        with self.assertRaises(ValueError):
            validate_loopback_url("https://api.example.com:8443", allow_tls=True)
        self.assertEqual(
            validate_loopback_url("https://api.example.com:8443", allow_tls=True, allow_remote=True),
            "https://api.example.com:8443",
        )

        # Auth token is read from env, sent over the wire, and never echoed into result payload
        Handler.mode = "good"
        Handler.last_auth_header = None
        with patch.dict(os.environ, {"FOUNDRY_TEST_TOKEN": "super-secret-value-123"}):
            res = suggest(
                CASE,
                origin=self.origin,
                model="local",
                auth_token_env="FOUNDRY_TEST_TOKEN",
            )
        self.assertEqual(res["status"], "REVIEW_REQUIRED")
        self.assertEqual(Handler.last_auth_header, "Bearer super-secret-value-123")
        self.assertNotIn("super-secret-value-123", json.dumps(res))

    def test_predict_batch_jsonl_and_fail_closed_undetermined(self):
        labels = ["ALPHA", "BETA", "UNDETERMINED"]
        records = [
            {"id": "t1", "prompt": "case 1", "gold_label": "ALPHA", "critical": False},
            {"id": "t2", "prompt": "case 2", "gold_label": "BETA", "critical": True},
            {"id": "t3", "prompt": "case 3", "gold_label": "ALPHA", "critical": False},
        ]
        replies = iter([
            {"choices": [{"message": {"content": '{"label": "ALPHA", "confidence": 0.9}'}}]},
            {"choices": [{"message": {"content": '{"label": "INVENTED_LABEL"}'}}]},
            OSError("connection dropped"),
        ])

        def batch_transport(*_args, **_kwargs):
            nxt = next(replies)
            if isinstance(nxt, BaseException):
                raise nxt
            return nxt

        with tempfile.TemporaryDirectory() as tmp:
            out_file = Path(tmp) / "preds.jsonl"
            summary = predict_batch(
                records,
                labels=labels,
                origin=self.origin,
                model="local-batch",
                transport=batch_transport,
                out_path=out_file,
            )
            self.assertEqual(summary["status"], "DEGRADED_FALLBACK")
            self.assertEqual(summary["total"], 3)
            self.assertEqual(summary["predicted_count"], 1)
            self.assertEqual(summary["fallback_count"], 2)
            self.assertEqual(len(summary["audit_sha256"]), 64)
            lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([r["id"] for r in lines], ["t1", "t2", "t3"])
            self.assertEqual([r["label"] for r in lines], ["ALPHA", "UNDETERMINED", "UNDETERMINED"])


if __name__ == "__main__":
    unittest.main()
