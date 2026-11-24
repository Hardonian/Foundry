"""Reference enterprise trust SPI adapters (KMS subprocess signer, RFC 3161 TSA, Object Lock, signed consent)."""
from __future__ import annotations

import json
import os
import re
import subprocess  # nosec B404 - shell=False with validated argv only
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from . import signing
from .canonical import canonical, digest
from .refinery_attestation import parse_keyring
from .types import AnchorRecord

_SHELL_META = re.compile(r"[;&|`$><\r\n\x00]")


class SubprocessKmsSignerProvider:
    """Fail-closed SignerProvider invoking an external KMS/HSM CLI helper via shell=False."""

    def __init__(
        self,
        *,
        public_key_hex: str,
        command: Sequence[str],
        provider_name: str = "subprocess_kms",
        timeout: float = 10.0,
    ) -> None:
        if not isinstance(public_key_hex, str) or not signing.HEX64.fullmatch(public_key_hex):
            raise ValueError("public_key_hex must be a 64-char lowercase ed25519 hex string")
        if isinstance(command, (str, bytes)) or not isinstance(command, Sequence) or not command:
            raise ValueError("command must be a non-empty sequence of argv strings")
        argv: list[str] = []
        for idx, token in enumerate(command):
            if not isinstance(token, str) or not token or _SHELL_META.search(token):
                raise ValueError(f"unsafe command token at index {idx}")
            argv.append(token)
        if not isinstance(provider_name, str) or not provider_name.strip() or len(provider_name) > 64:
            raise ValueError("provider_name must be a clean string <= 64 chars")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be > 0")

        self._public_key_hex = public_key_hex
        self._command = tuple(argv)
        self._provider_name = provider_name.strip()
        self._timeout = float(timeout)

    def public_key_hex(self) -> str:
        return self._public_key_hex

    def backend_name(self) -> str:
        return self._provider_name

    def sign(self, message_bytes: bytes) -> str:
        if not signing.backend_available():
            raise signing.SigningUnavailable("signing backend unavailable")
        payload_hex = message_bytes.hex()
        proc = subprocess.run(  # nosec B603 - validated non-shell argv sequence
            list(self._command),
            input=payload_hex,
            capture_output=True,
            text=True,
            timeout=self._timeout,
            check=False,
        )
        if proc.returncode != 0:
            raise ValueError(f"external KMS signer command failed (exit {proc.returncode}): {proc.stderr.strip()}")
        sig_hex = proc.stdout.strip().splitlines()[-1].strip() if proc.stdout.strip() else ""
        if not signing.verify(message_bytes, sig_hex, [self._public_key_hex]):
            raise ValueError("subprocess KMS signer returned invalid or unverifiable Ed25519 signature")
        return sig_hex


class Rfc3161TsaAnchorSink:
    """AnchorSink submitting canonical head imprints to an RFC 3161 TSA (or injectable transport)."""

    def __init__(
        self,
        tsa_url: str,
        *,
        transport: Callable[[str, bytes, float], dict[str, Any]] | None = None,
        timeout: float = 5.0,
    ) -> None:
        if not isinstance(tsa_url, str) or not tsa_url.startswith("https://"):
            raise ValueError("tsa_url must be an https:// URL")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be > 0")
        self._tsa_url = tsa_url
        self._transport = transport
        self._timeout = float(timeout)

    def _default_transport(self, url: str, payload: bytes, timeout: float) -> dict[str, Any]:
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/timestamp-query", "Accept": "application/json"},
        )
        # HTTPS enforced by constructor; timeout bounded.
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            raw = resp.read(65536).decode("utf-8")
        return dict(json.loads(raw))

    def publish(self, anchor: AnchorRecord) -> dict[str, Any]:
        head = anchor.get("head")
        if not isinstance(head, str) or not signing.HEX64.fullmatch(head):
            raise ValueError("anchor head must be a 64-char lowercase hex string")
        request_envelope = {
            "protocol": "rfc3161-sha256-json",
            "hash_algorithm": "sha256",
            "message_imprint": head,
            "run_id": anchor.get("run_id"),
            "count": anchor.get("count"),
        }
        body = canonical(request_envelope).encode("utf-8")
        fn = self._transport if self._transport is not None else self._default_transport
        reply = fn(self._tsa_url, body, self._timeout)
        if not isinstance(reply, dict) or reply.get("message_imprint") != head:
            raise ValueError("RFC 3161 TSA receipt message_imprint mismatch")
        token_sha256 = reply.get("token_sha256") or digest(reply)
        if not isinstance(token_sha256, str) or not signing.HEX64.fullmatch(token_sha256):
            raise ValueError("RFC 3161 TSA receipt token_sha256 must be 64-char hex")
        return {
            "sink_type": "rfc3161_tsa",
            "immutable": True,
            "tsa_url": self._tsa_url,
            "head": head,
            "count": anchor.get("count"),
            "gen_time": reply.get("gen_time"),
            "serial_number": reply.get("serial_number"),
            "token_sha256": token_sha256,
            "receipt_sha256": digest(dict(anchor)),
        }


@runtime_checkable
class ObjectLockClient(Protocol):
    """Minimal protocol for S3/GCS Object Lock WORM storage clients."""

    def head_object(self, *, bucket: str, key: str) -> dict[str, Any] | None:
        ...

    def put_object(
        self,
        *,
        bucket: str,
        key: str,
        body: bytes,
        object_lock_mode: str,
        retain_until_date: str,
    ) -> dict[str, Any]:
        ...


class ObjectLockAnchorSink:
    """AnchorSink writing COMPLIANCE-locked anchor objects via an ObjectLockClient."""

    def __init__(
        self,
        *,
        bucket: str,
        client: ObjectLockClient,
        prefix: str = "foundry-anchors",
        retention_days: int = 365,
    ) -> None:
        if not isinstance(bucket, str) or not bucket.strip():
            raise ValueError("bucket must be a non-empty string")
        if not isinstance(client, ObjectLockClient):
            raise TypeError("client must satisfy ObjectLockClient protocol")
        if isinstance(retention_days, bool) or not isinstance(retention_days, int) or retention_days < 1:
            raise ValueError("retention_days must be an integer >= 1")
        self._bucket = bucket.strip()
        self._prefix = prefix.strip("/")
        self._client = client
        self._retention_days = retention_days

    def publish(self, anchor: AnchorRecord) -> dict[str, Any]:
        head = anchor.get("head")
        if not isinstance(head, str) or not signing.HEX64.fullmatch(head):
            raise ValueError("anchor head must be a 64-char lowercase hex string")
        key = f"{self._prefix}/anchor-{head}.json" if self._prefix else f"anchor-{head}.json"
        payload = dict(anchor)
        receipt_sha256 = digest(payload)

        existing = self._client.head_object(bucket=self._bucket, key=key)
        if existing is not None:
            existing_sha = existing.get("receipt_sha256")
            if existing_sha != receipt_sha256:
                raise ValueError("ObjectLock WORM immutability violation: conflicting anchor exists for head")
            return {
                "sink_type": "object_lock",
                "immutable": True,
                "bucket": self._bucket,
                "key": key,
                "object_lock_mode": "COMPLIANCE",
                "head": head,
                "receipt_sha256": receipt_sha256,
            }

        retain_until = datetime.fromtimestamp(
            datetime.now(UTC).timestamp() + (self._retention_days * 86400),
            tz=UTC,
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        put_res = self._client.put_object(
            bucket=self._bucket,
            key=key,
            body=canonical(payload).encode("utf-8"),
            object_lock_mode="COMPLIANCE",
            retain_until_date=retain_until,
        )
        return {
            "sink_type": "object_lock",
            "immutable": True,
            "bucket": self._bucket,
            "key": key,
            "object_lock_mode": "COMPLIANCE",
            "retain_until_date": retain_until,
            "etag": put_res.get("etag"),
            "head": head,
            "receipt_sha256": receipt_sha256,
        }


def create_consent_receipt(
    attestation: dict[str, Any],
    *,
    private_key_path: str | os.PathLike[str] | None = None,
    signer: signing.SignerProvider | None = None,
) -> dict[str, Any]:
    """Sign a governance_attestation payload with Ed25519 for consented_pilot / redacted_eval runs."""
    if not isinstance(attestation, dict):
        raise TypeError("attestation must be a dict")
    if signer is not None and not isinstance(signer, signing.SignerProvider):
        raise TypeError("signer must satisfy SignerProvider protocol")
    if signer is None and private_key_path is None:
        raise ValueError("either private_key_path or signer must be provided")
    if not signing.backend_available():
        raise signing.SigningUnavailable("signing backend unavailable")

    signed_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    core = {
        "schema_version": "1.0",
        "receipt_type": "hardonian.foundry.consent_receipt.v1",
        "signed_at": signed_at,
        "attestation": attestation,
    }
    msg = canonical(core).encode("utf-8")
    if signer is not None:
        pub = signer.public_key_hex()
        sig_hex = signer.sign(msg)
        backend = signer.backend_name()
    else:
        assert private_key_path is not None
        pub = signing.public_key_hex(private_key_path)
        sig_hex = signing.sign(msg, private_key_path)
        backend = signing.SIGNING_BACKEND or "cryptography"

    return {
        **core,
        "public_key_hex": pub,
        "signature_hex": sig_hex,
        "signing_backend": backend,
        "receipt_sha256": digest(core),
    }


def verify_consent_receipt(receipt: dict[str, Any], *, keyring: Any) -> dict[str, Any]:
    """Verify an Ed25519-signed governance_attestation consent receipt against a trusted keyring."""
    if not isinstance(receipt, dict):
        raise TypeError("receipt must be a dict")
    if not signing.backend_available():
        return {"verified": False, "status": "UNAVAILABLE", "errors": ["signing backend unavailable"]}

    if isinstance(keyring, (str, Path)):
        raw_kr = json.loads(Path(keyring).read_text(encoding="utf-8"))
        policy = parse_keyring(raw_kr)
    else:
        policy = parse_keyring(keyring)

    errors: list[str] = []
    if receipt.get("receipt_type") != "hardonian.foundry.consent_receipt.v1":
        errors.append("invalid receipt_type")
    if not isinstance(receipt.get("attestation"), dict):
        errors.append("missing attestation object")

    core = {
        "schema_version": receipt.get("schema_version"),
        "receipt_type": receipt.get("receipt_type"),
        "signed_at": receipt.get("signed_at"),
        "attestation": receipt.get("attestation"),
    }
    if receipt.get("receipt_sha256") != digest(core):
        errors.append("receipt_sha256 mismatch")

    pub = receipt.get("public_key_hex")
    key_by_hex = {k["public_key_hex"]: k for k in policy["keys"]}
    if not isinstance(pub, str) or pub not in key_by_hex:
        errors.append("signer public_key_hex not in trusted keyring")
    elif key_by_hex[pub].get("revoked"):
        errors.append("signer key has been revoked")
    else:
        msg = canonical(core).encode("utf-8")
        if not signing.verify(msg, receipt.get("signature_hex"), [pub]):
            errors.append("Ed25519 signature verification failed")

    ok = not errors
    return {
        "verified": ok,
        "status": "VERIFIED" if ok else "INVALID",
        "signer_public_key_hex": pub if isinstance(pub, str) else None,
        "receipt_sha256": receipt.get("receipt_sha256"),
        "errors": sorted(errors),
    }
