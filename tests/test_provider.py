import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from hardonian_foundry import provider
from hardonian_foundry.provider import suggest, validate_loopback_url

CASE = {"id": "synthetic-provider", "expected_minor": 100, "actual_minor": 99, "critical": False}


class Handler(BaseHTTPRequestHandler):
    mode = "good"

    def log_message(self, *_args):
        pass

    def do_POST(self):
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


if __name__ == "__main__":
    unittest.main()
