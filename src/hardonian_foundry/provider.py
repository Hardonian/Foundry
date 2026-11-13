"""Opt-in local OpenAI-compatible provider, returning review-only outcomes.

This adapter is not enabled by default. It never falls back to heuristic output
for reconciliation cases, and falls back to UNDETERMINED on batch classification
failures so invalid model outputs never masquerade as valid predictions.
"""
from __future__ import annotations

import collections
import heapq
import http.client
import ipaddress
import json
import os
import re
import socket
import ssl
import tempfile
import threading
import time
import urllib.error
import urllib.parse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .canonical import digest
from .types import ProviderResult

MAX_RESPONSE = 32768
_ENV_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ProviderError(OSError):
    """Structured provider transport/protocol error with stable reason_code."""

    def __init__(self, reason_code: str, message: str, *, http_status: int | None = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.http_status = http_status


class CircuitBreaker:
    """Thread-safe circuit breaker for provider inference endpoints."""

    def __init__(self, *, failure_threshold: int = 5, recovery_timeout: float = 30.0) -> None:
        if isinstance(failure_threshold, bool) or not isinstance(failure_threshold, int) or failure_threshold < 1:
            raise ValueError("failure_threshold must be an integer >= 1")
        if isinstance(recovery_timeout, bool) or not isinstance(recovery_timeout, (int, float)) or recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be > 0")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = float(recovery_timeout)
        self._lock = threading.Lock()
        self._failures = 0
        self._state = "CLOSED"
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        with self._lock:
            if (
                self._state == "OPEN"
                and self._opened_at is not None
                and (time.monotonic() - self._opened_at) >= self.recovery_timeout
            ):
                self._state = "HALF_OPEN"
            return self._state

    def allow_request(self) -> bool:
        with self._lock:
            if self._state == "CLOSED":
                return True
            if self._state == "OPEN":
                if self._opened_at is not None and (time.monotonic() - self._opened_at) >= self.recovery_timeout:
                    self._state = "HALF_OPEN"
                    return True
                return False
            # HALF_OPEN: allow single probe; keep closed to others until probe resolves
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "CLOSED"
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == "HALF_OPEN" or self._failures >= self.failure_threshold:
                self._state = "OPEN"
                self._opened_at = time.monotonic()

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "CLOSED"
            self._opened_at = None


class _DeadlineGuard:
    """Shared daemon timer thread closing sockets strictly at total request deadlines."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._heap: list[tuple[float, int, _ActiveRequest]] = []
        self._counter = 0
        self._started = False

    def _ensure_started(self) -> None:
        if not self._started:
            thread = threading.Thread(target=self._run, name="foundry-provider-deadline", daemon=True)
            thread.start()
            self._started = True

    def register(self, deadline: float, req: _ActiveRequest) -> None:
        with self._cond:
            self._ensure_started()
            self._counter += 1
            heapq.heappush(self._heap, (deadline, self._counter, req))
            self._cond.notify_all()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._heap:
                    self._cond.wait()
                while self._heap:
                    deadline, _, req = self._heap[0]
                    if req.done:
                        heapq.heappop(self._heap)
                        continue
                    now = time.monotonic()
                    if now < deadline:
                        self._cond.wait(timeout=deadline - now)
                        continue
                    heapq.heappop(self._heap)
                    req.abort_timeout()


class _ActiveRequest:
    def __init__(self, conn: http.client.HTTPConnection) -> None:
        self._lock = threading.Lock()
        self.conn = conn
        self.done = False
        self.timed_out = False

    def finish(self) -> None:
        with self._lock:
            self.done = True

    def abort_timeout(self) -> None:
        with self._lock:
            if self.done:
                return
            self.timed_out = True
            self.done = True
            sock = self.conn.sock
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass


class _ConnectionPool:
    """Thread-safe pool of persistent HTTP/HTTPS connections (proxies disabled)."""

    def __init__(self, max_per_origin: int = 8) -> None:
        self._lock = threading.Lock()
        self._max_per_origin = max_per_origin
        self._pools: dict[tuple[str, str, int, str | None], collections.deque[http.client.HTTPConnection]] = {}

    def acquire(
        self,
        scheme: str,
        host: str,
        port: int,
        timeout: float,
        ssl_context: ssl.SSLContext | None,
        pool_tag: str | None,
    ) -> tuple[http.client.HTTPConnection, bool]:
        key = (scheme, host, port, pool_tag)
        with self._lock:
            dq = self._pools.get(key)
            while dq:
                conn = dq.popleft()
                if conn.sock is not None:
                    return conn, True
                try:
                    conn.close()
                except OSError:
                    pass
        if scheme == "https":
            conn = http.client.HTTPSConnection(host, port=port, timeout=timeout, context=ssl_context)
        else:
            conn = http.client.HTTPConnection(host, port=port, timeout=timeout)
        return conn, False

    def release(
        self,
        scheme: str,
        host: str,
        port: int,
        pool_tag: str | None,
        conn: http.client.HTTPConnection,
    ) -> None:
        if conn.sock is None:
            return
        key = (scheme, host, port, pool_tag)
        with self._lock:
            dq = self._pools.setdefault(key, collections.deque())
            if len(dq) < self._max_per_origin:
                dq.append(conn)
                return
        try:
            conn.close()
        except OSError:
            pass

    def close_all(self) -> None:
        with self._lock:
            for dq in self._pools.values():
                while dq:
                    conn = dq.popleft()
                    try:
                        conn.close()
                    except OSError:
                        pass
            self._pools.clear()


_DEADLINE_GUARD = _DeadlineGuard()
_CONNECTION_POOL = _ConnectionPool()


def validate_loopback_url(
    url: str,
    *,
    allow_tls: bool = False,
    allow_remote: bool = False,
) -> str:
    """Validate provider origin URL.

    Defaults preserve strict numeric loopback HTTP only; TLS and non-loopback HTTPS
    require explicit opt-in flags. Plaintext HTTP to non-loopback hosts is never allowed.
    """
    if not isinstance(url, str) or not url:
        raise ValueError("provider URL must be a non-empty string")
    parsed = urllib.parse.urlsplit(url)
    allowed_schemes = ("http", "https") if allow_tls else ("http",)
    if parsed.scheme not in allowed_schemes or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("provider must be a loopback HTTP URL without credentials")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("provider URL must be an origin without path, query, or fragment")
    port = parsed.port
    if not port:
        raise ValueError("provider URL must specify an explicit port")

    is_loopback = False
    try:
        is_loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        if not (allow_tls and allow_remote and parsed.scheme == "https"):
            raise ValueError("provider must use a numeric loopback address and valid port") from None

    if not is_loopback and not (allow_tls and allow_remote and parsed.scheme == "https"):
        raise ValueError("provider URL must be a numeric loopback origin with explicit port")

    return url.rstrip("/")


def _build_ssl_context(ca_bundle: str | Path | None) -> ssl.SSLContext:
    cafile = str(ca_bundle) if ca_bundle is not None else None
    if cafile is not None and not Path(cafile).is_file():
        raise ValueError(f"CA bundle file not found: {cafile}")
    ctx = ssl.create_default_context(cafile=cafile)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


def _resolve_bearer_header(auth_token_env: str | None) -> dict[str, str]:
    if auth_token_env is None:
        return {}
    if not isinstance(auth_token_env, str) or not _ENV_VAR_RE.fullmatch(auth_token_env):
        raise ValueError("auth_token_env must be a valid environment variable name")
    token = os.environ.get(auth_token_env)
    if not token or not token.strip() or any(ord(c) < 32 for c in token):
        raise ProviderError("AUTH_MISSING", f"bearer token environment variable {auth_token_env} is unset or invalid")
    return {"Authorization": f"Bearer {token.strip()}"}


def _single_http_attempt(
    scheme: str,
    host: str,
    port: int,
    path: str,
    body: bytes,
    timeout: float,
    headers: dict[str, str],
    ssl_context: ssl.SSLContext | None,
    pool_tag: str | None,
    *,
    allow_retry_stale: bool,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    conn, from_pool = _CONNECTION_POOL.acquire(scheme, host, port, timeout, ssl_context, pool_tag)
    guard_req = _ActiveRequest(conn)
    _DEADLINE_GUARD.register(deadline, guard_req)
    reusable = False
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("provider deadline exceeded")
        conn.timeout = remaining
        if conn.sock is not None:
            conn.sock.settimeout(remaining)
        conn.request("POST", path, body=body, headers=headers)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("provider deadline exceeded")
        if conn.sock is not None:
            conn.sock.settimeout(remaining)
        resp = conn.getresponse()

        if 300 <= resp.status < 400:
            raise ProviderError("REDIRECT_REJECTED", f"provider redirects disabled (HTTP {resp.status})", http_status=resp.status)
        if resp.status != 200:
            raise ProviderError("HTTP_ERROR", f"provider returned HTTP {resp.status}", http_status=resp.status)

        chunks: list[bytes] = []
        total = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or guard_req.timed_out:
                raise TimeoutError("provider deadline exceeded")
            if conn.sock is not None:
                conn.sock.settimeout(remaining)
            chunk = resp.read1(min(4096, MAX_RESPONSE + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_RESPONSE:
                raise ProviderError("RESPONSE_TOO_LARGE", "provider response exceeds maximum byte limit")
            chunks.append(chunk)

        if guard_req.timed_out or time.monotonic() > deadline:
            raise TimeoutError("provider deadline exceeded")

        conn_hdr = (resp.getheader("Connection") or "").lower()
        reusable = (conn_hdr != "close") and not resp.will_close
        raw_bytes = b"".join(chunks)
        return json.loads(raw_bytes)
    except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError) as exc:
        if from_pool and allow_retry_stale and not guard_req.timed_out and time.monotonic() < deadline:
            guard_req.finish()
            try:
                conn.close()
            except OSError:
                pass
            remaining = deadline - time.monotonic()
            return _single_http_attempt(
                scheme,
                host,
                port,
                path,
                body,
                remaining,
                headers,
                ssl_context,
                pool_tag,
                allow_retry_stale=False,
            )
        if guard_req.timed_out:
            raise TimeoutError("provider deadline exceeded") from exc
        raise ProviderError("CONNECTION_ERROR", f"provider connection failed: {exc}") from exc
    except TimeoutError as exc:
        raise TimeoutError("provider deadline exceeded") from exc
    except ValueError as exc:
        if guard_req.timed_out:
            raise TimeoutError("provider deadline exceeded") from exc
        raise
    except OSError as exc:
        if guard_req.timed_out:
            raise TimeoutError("provider deadline exceeded") from exc
        if isinstance(exc, ProviderError):
            raise
        raise ProviderError("CONNECTION_ERROR", f"provider connection failed: {exc}") from exc
    except http.client.HTTPException as exc:
        if guard_req.timed_out:
            raise TimeoutError("provider deadline exceeded") from exc
        raise ProviderError("CONNECTION_ERROR", f"provider protocol error: {exc}") from exc
    finally:
        guard_req.finish()
        if reusable and not guard_req.timed_out:
            _CONNECTION_POOL.release(scheme, host, port, pool_tag, conn)
        else:
            try:
                conn.close()
            except OSError:
                pass


def _request(
    url: str,
    body: bytes,
    timeout: float,
    *,
    extra_headers: dict[str, str] | None = None,
    ssl_context: ssl.SSLContext | None = None,
    ca_bundle_key: str | None = None,
) -> dict[str, Any]:
    """Fast persistent socket-deadline HTTP transport (no redirects, no proxies)."""
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname
    port = parsed.port
    if not host or not port:
        raise ValueError("invalid provider request URL")
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
    if extra_headers:
        headers.update(extra_headers)
    return _single_http_attempt(
        parsed.scheme,
        host,
        port,
        path,
        body,
        float(timeout),
        headers,
        ssl_context,
        ca_bundle_key,
        allow_retry_stale=True,
    )


def _classify_exception(exc: BaseException) -> tuple[str, str, int | None]:
    if isinstance(exc, ProviderError):
        return exc.reason_code, str(exc), exc.http_status
    if isinstance(exc, TimeoutError):
        return "TIMEOUT", str(exc) or "provider deadline exceeded", None
    if isinstance(exc, urllib.error.HTTPError):
        return "HTTP_ERROR", f"provider returned HTTP {exc.code}", exc.code
    if isinstance(exc, (http.client.HTTPException, OSError)):
        return "CONNECTION_ERROR", str(exc) or "provider connection error", None
    if isinstance(exc, (ValueError, KeyError, TypeError, RecursionError, json.JSONDecodeError)):
        return "INVALID_RESPONSE_SCHEMA", str(exc) or "invalid provider response", None
    return "UNAVAILABLE", str(exc) or "provider unavailable", None


def _audit_hash(
    *,
    origin: str,
    model: str,
    case_id: str | None,
    status: str,
    category_or_label: str | None = None,
    reason_code: str | None = None,
) -> str:
    return digest({
        "origin": origin,
        "model": model,
        "case_id": case_id,
        "status": status,
        "output": category_or_label,
        "reason_code": reason_code,
    })


def _extract_chat_content(response: Any) -> str:
    if not isinstance(response, dict):
        raise TypeError("invalid provider envelope")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("invalid choices")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or len(content) > MAX_RESPONSE:
        raise ValueError("invalid content")
    return content


def suggest(
    case: dict[str, Any],
    *,
    origin: str,
    model: str,
    timeout: float = 3.0,
    transport: Callable[..., dict[str, Any]] | None = None,
    allow_tls: bool = False,
    allow_remote: bool = False,
    auth_token_env: str | None = None,
    ca_bundle: str | Path | None = None,
    circuit_breaker: CircuitBreaker | None = None,
) -> ProviderResult:
    """Return unavailable on transport/schema failure, never fabricate a prediction.

    Only synthetic, bounded integer records are accepted by the underlying baseline validator.
    """
    from .baseline import reconciliation

    reconciliation(case)  # validates the entire input before any network request
    clean_origin = validate_loopback_url(origin, allow_tls=allow_tls, allow_remote=allow_remote)
    if not isinstance(model, str) or not model.strip() or len(model) > 128:
        raise ValueError("model must be a nonempty string of at most 128 characters")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= 10:
        raise ValueError("timeout must be greater than zero and at most ten seconds")

    ssl_ctx = _build_ssl_context(ca_bundle) if clean_origin.startswith("https://") else None
    ca_key = str(Path(ca_bundle).resolve()) if ca_bundle is not None else None
    started = time.monotonic()

    if circuit_breaker is not None and not circuit_breaker.allow_request():
        latency_ms = round((time.monotonic() - started) * 1000.0, 3)
        return {
            "status": "UNAVAILABLE",
            "mode": "local_provider",
            "data_origin": "synthetic",
            "review_required": True,
            "reason_code": "CIRCUIT_OPEN",
            "reason": "provider circuit breaker is open",
            "audit_sha256": _audit_hash(
                origin=clean_origin,
                model=model,
                case_id=case.get("id"),
                status="UNAVAILABLE",
                reason_code="CIRCUIT_OPEN",
            ),
            "latency_ms": latency_ms,
        }

    payload = json.dumps({
        "model": model,
        "stream": False,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "Return ONLY JSON with category: matched or amount_mismatch. This is synthetic data.",
            },
            {"role": "user", "content": json.dumps(case, sort_keys=True)},
        ],
    }).encode("utf-8")

    try:
        auth_headers = _resolve_bearer_header(auth_token_env)
        endpoint = clean_origin + "/v1/chat/completions"
        if transport is not None:
            response = transport(endpoint, payload, timeout)
        else:
            response = _request(
                endpoint,
                payload,
                timeout,
                extra_headers=auth_headers,
                ssl_context=ssl_ctx,
                ca_bundle_key=ca_key,
            )

        content = _extract_chat_content(response)
        result = json.loads(content)
        if (
            not isinstance(result, dict)
            or set(result) != {"category"}
            or result["category"] not in ("matched", "amount_mismatch")
        ):
            raise ValueError("invalid prediction schema")
        if circuit_breaker is not None:
            circuit_breaker.record_success()
        expected = "matched" if case["actual_minor"] == case["expected_minor"] else "amount_mismatch"
        latency_ms = round((time.monotonic() - started) * 1000.0, 3)
        return {
            "status": "REVIEW_REQUIRED",
            "mode": "local_provider",
            "data_origin": "synthetic",
            "category": result["category"],
            "arithmetic_agrees": result["category"] == expected,
            "review_required": True,
            "audit_sha256": _audit_hash(
                origin=clean_origin,
                model=model,
                case_id=case.get("id"),
                status="REVIEW_REQUIRED",
                category_or_label=result["category"],
            ),
            "latency_ms": latency_ms,
        }
    except urllib.error.HTTPError as exc:
        exc.close()
        if circuit_breaker is not None:
            circuit_breaker.record_failure()
        reason_code, reason, http_status = _classify_exception(exc)
        latency_ms = round((time.monotonic() - started) * 1000.0, 3)
        out: ProviderResult = {
            "status": "UNAVAILABLE",
            "mode": "local_provider",
            "data_origin": "synthetic",
            "review_required": True,
            "reason_code": reason_code,
            "reason": reason,
            "audit_sha256": _audit_hash(
                origin=clean_origin,
                model=model,
                case_id=case.get("id"),
                status="UNAVAILABLE",
                reason_code=reason_code,
            ),
            "latency_ms": latency_ms,
        }
        if http_status is not None:
            out["http_status"] = http_status
        return out
    except (
        OSError,
        TimeoutError,
        ValueError,
        KeyError,
        TypeError,
        RecursionError,
        http.client.HTTPException,
        json.JSONDecodeError,
    ) as exc:
        if circuit_breaker is not None:
            circuit_breaker.record_failure()
        reason_code, reason, http_status = _classify_exception(exc)
        latency_ms = round((time.monotonic() - started) * 1000.0, 3)
        out = {
            "status": "UNAVAILABLE",
            "mode": "local_provider",
            "data_origin": "synthetic",
            "review_required": True,
            "reason_code": reason_code,
            "reason": reason,
            "audit_sha256": _audit_hash(
                origin=clean_origin,
                model=model,
                case_id=case.get("id"),
                status="UNAVAILABLE",
                reason_code=reason_code,
            ),
            "latency_ms": latency_ms,
        }
        if http_status is not None:
            out["http_status"] = http_status
        return out


def predict_batch(
    records: Sequence[dict[str, Any]],
    *,
    labels: Sequence[str],
    origin: str,
    model: str,
    timeout: float = 3.0,
    transport: Callable[..., dict[str, Any]] | None = None,
    allow_tls: bool = False,
    allow_remote: bool = False,
    auth_token_env: str | None = None,
    ca_bundle: str | Path | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    fallback_label: str = "UNDETERMINED",
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate task-mode {id, label} JSONL predictions with fail-closed fallback.

    Any transport error, timeout, circuit-open state, malformed response, or
    off-allowlist label falls back safely to fallback_label (default UNDETERMINED).
    """
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("records must be a non-empty sequence of task objects")
    if not isinstance(labels, (list, tuple)) or not labels or any(not isinstance(lbl, str) or not lbl.strip() for lbl in labels):
        raise ValueError("labels must be a non-empty sequence of label strings")
    label_set = set(labels)
    if fallback_label not in label_set:
        raise ValueError(f"fallback_label {fallback_label!r} must be present in spec labels")

    clean_origin = validate_loopback_url(origin, allow_tls=allow_tls, allow_remote=allow_remote)
    if not isinstance(model, str) or not model.strip() or len(model) > 128:
        raise ValueError("model must be a nonempty string of at most 128 characters")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= 10:
        raise ValueError("timeout must be greater than zero and at most ten seconds")

    for idx, row in enumerate(records):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or not isinstance(row.get("prompt"), str):
            raise ValueError(f"record at index {idx} must include string id and prompt")

    ssl_ctx = _build_ssl_context(ca_bundle) if clean_origin.startswith("https://") else None
    ca_key = str(Path(ca_bundle).resolve()) if ca_bundle is not None else None
    breaker = circuit_breaker if circuit_breaker is not None else CircuitBreaker()
    auth_headers = _resolve_bearer_header(auth_token_env) if auth_token_env is not None else {}
    endpoint = clean_origin + "/v1/chat/completions"
    label_menu = ", ".join(labels)

    predictions: list[dict[str, str]] = []
    reason_counts: dict[str, int] = {}
    fallback_count = 0

    for row in records:
        rid = row["id"]
        if not breaker.allow_request():
            fallback_count += 1
            reason_counts["CIRCUIT_OPEN"] = reason_counts.get("CIRCUIT_OPEN", 0) + 1
            predictions.append({"id": rid, "label": fallback_label})
            continue

        payload = json.dumps({
            "model": model,
            "stream": False,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": f"Return ONLY JSON {{\"label\": \"<CODE>\"}} using one of: {label_menu}.",
                },
                {"role": "user", "content": row["prompt"]},
            ],
        }).encode("utf-8")

        try:
            if transport is not None:
                response = transport(endpoint, payload, timeout)
            else:
                response = _request(
                    endpoint,
                    payload,
                    timeout,
                    extra_headers=auth_headers,
                    ssl_context=ssl_ctx,
                    ca_bundle_key=ca_key,
                )
            content = _extract_chat_content(response)
            parsed = json.loads(content)
            if not isinstance(parsed, dict) or "label" not in parsed or not set(parsed) <= {"label", "confidence"}:
                raise ValueError("invalid prediction JSON schema")
            pred_label = parsed["label"]
            if not isinstance(pred_label, str) or pred_label not in label_set:
                raise ValueError(f"predicted label {pred_label!r} not in allowlist")
            breaker.record_success()
            predictions.append({"id": rid, "label": pred_label})
        except urllib.error.HTTPError as exc:
            exc.close()
            breaker.record_failure()
            reason_code, _, _ = _classify_exception(exc)
            fallback_count += 1
            reason_counts[reason_code] = reason_counts.get(reason_code, 0) + 1
            predictions.append({"id": rid, "label": fallback_label})
        except (
            OSError,
            TimeoutError,
            ValueError,
            KeyError,
            TypeError,
            RecursionError,
            http.client.HTTPException,
            json.JSONDecodeError,
        ) as exc:
            breaker.record_failure()
            reason_code, _, _ = _classify_exception(exc)
            fallback_count += 1
            reason_counts[reason_code] = reason_counts.get(reason_code, 0) + 1
            predictions.append({"id": rid, "label": fallback_label})

    saved_path: str | None = None
    if out_path is not None:
        dest = Path(out_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".foundry-preds-", dir=str(dest.parent), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for item in predictions:
                    handle.write(json.dumps(item, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, dest)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        saved_path = str(dest)

    return {
        "status": "COMPLETED" if fallback_count == 0 else "DEGRADED_FALLBACK",
        "mode": "local_provider",
        "data_origin": "synthetic",
        "review_required": True,
        "total": len(records),
        "predicted_count": len(records) - fallback_count,
        "fallback_count": fallback_count,
        "fallback_label": fallback_label,
        "unavailable_reasons": reason_counts,
        "predictions_sha256": digest(predictions),
        "audit_sha256": digest({
            "origin": clean_origin,
            "model": model,
            "total": len(records),
            "fallback_count": fallback_count,
            "predictions_sha256": digest(predictions),
        }),
        "predictions": predictions,
        "out_path": saved_path,
    }
