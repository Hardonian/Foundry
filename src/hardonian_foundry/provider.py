"""Opt-in local OpenAI-compatible provider, returning review-only outcomes.

This adapter is not enabled by default. It never falls back to heuristic output.
"""
import http.client
import ipaddress
import json
import urllib.error
import urllib.parse
import urllib.request

MAX_RESPONSE = 32768


def validate_loopback_url(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("provider must be a loopback HTTP URL without credentials")
    try:
        loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        port = parsed.port
    except ValueError as exc:
        raise ValueError("provider must use a numeric loopback address and valid port") from exc
    if not loopback or not port or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("provider URL must be a numeric loopback origin with explicit port")
    return url.rstrip("/")


def _request(url, body, timeout):
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    # Do not follow redirects: a loopback service must not redirect private data elsewhere.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(req, timeout=timeout) as response:
        data = response.read(MAX_RESPONSE + 1)
        if len(data) > MAX_RESPONSE:
            raise ValueError("provider response too large")
        return json.loads(data)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def suggest(case, *, origin, model, timeout=3.0, transport=_request):
    """Return unavailable on transport/schema failure, never fabricate a prediction.

    Only synthetic, bounded integer records are accepted by the underlying baseline validator.
    """
    from .baseline import reconciliation
    reconciliation(case)  # validates the entire input before any network request
    origin = validate_loopback_url(origin)
    if not isinstance(model, str) or not model.strip() or len(model) > 128:
        raise ValueError("model must be a nonempty string of at most 128 characters")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= 10:
        raise ValueError("timeout must be greater than zero and at most ten seconds")
    payload = json.dumps({"model": model, "stream": False, "temperature": 0,
                          "messages": [{"role": "system", "content": "Return ONLY JSON with category: matched or amount_mismatch. This is synthetic data."},
                                       {"role": "user", "content": json.dumps(case, sort_keys=True)}]}).encode()
    try:
        response = transport(origin + "/v1/chat/completions", payload, timeout)
        if not isinstance(response, dict):
            raise TypeError("invalid provider envelope")
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("invalid choices")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or len(content) > MAX_RESPONSE:
            raise ValueError("invalid content")
        result = json.loads(content)
        if not isinstance(result, dict) or set(result) != {"category"} or result["category"] not in ("matched", "amount_mismatch"):
            raise ValueError("invalid prediction schema")
        # Deterministic arithmetic remains authoritative: disagreement requires review.
        expected = "matched" if case["actual_minor"] == case["expected_minor"] else "amount_mismatch"
        return {"status": "REVIEW_REQUIRED", "mode": "local_provider", "data_origin": "synthetic",
                "category": result["category"], "arithmetic_agrees": result["category"] == expected,
                "review_required": True}
    except urllib.error.HTTPError as exc:
        exc.close()
        return {"status": "UNAVAILABLE", "mode": "local_provider", "data_origin": "synthetic", "review_required": True}
    except (OSError, TimeoutError, ValueError, KeyError, TypeError, http.client.HTTPException):
        return {"status": "UNAVAILABLE", "mode": "local_provider", "data_origin": "synthetic", "review_required": True}
