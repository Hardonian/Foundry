"""Ed25519 signing and verification for promotion attestations.

Fail-closed by design: if the optional `signing` extra (cryptography) is not
installed, nothing verifies and callers must treat that as a blocking missing
evidence item. Private key material must be file-based, mode 0600, and is never
logged or embedded into run records. Public keys are stored as 64-char lowercase
hex (ed25519 raw 32-byte encoding).
"""
import hashlib
import os
import re
import stat

HEX64 = re.compile(r"^[0-9a-f]{64}$")

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    SIGNING_BACKEND = "cryptography"
except ImportError:  # pragma: no cover - exercised only without the extra
    SIGNING_BACKEND = None


class SigningUnavailable(RuntimeError):
    """Raised when a signing operation is requested but the backend is absent."""


def backend_available():
    return SIGNING_BACKEND is not None


def generate_keypair(private_key_path):
    """Create a new Ed25519 private key PEM at path (refuses if it exists).

    Returns {"public_key_hex", "private_key_path"}. File is mode 0600 and the
    parent directory must already exist.
    """
    if not backend_available():
        raise SigningUnavailable("signing backend unavailable")
    if os.path.exists(private_key_path):
        raise ValueError(f"refusing to overwrite existing key file: {private_key_path}")
    private_key = Ed25519PrivateKey.generate()
    payload = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    directory = os.path.dirname(os.path.abspath(private_key_path))
    os.makedirs(directory, exist_ok=True)
    descriptor = os.open(private_key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    public_hex = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    ).hex()
    return {"public_key_hex": public_hex, "private_key_path": private_key_path}


def _load_private_key(private_key_path):
    if not backend_available():
        raise SigningUnavailable("signing backend unavailable")
    info = os.stat(private_key_path)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("private key path is not a regular file")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("private key must not be accessible by group or others (need mode 0600)")
    with open(private_key_path, "rb") as handle:
        key = serialization.load_pem_private_key(handle.read(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("private key must be an Ed25519 key")
    return key


def public_key_hex(private_key_path):
    if not backend_available():
        raise SigningUnavailable("signing backend unavailable")
    key = _load_private_key(private_key_path)
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    ).hex()


def sign(message_bytes, private_key_path):
    """Sign exact bytes with the file-based private key; returns 128-char hex."""
    if not backend_available():
        raise SigningUnavailable("signing backend unavailable")
    key = _load_private_key(private_key_path)
    return key.sign(message_bytes).hex()


def fingerprint(public_hex):
    return hashlib.sha256(bytes.fromhex(public_hex)).hexdigest()


def normalize_trusted_keys(entries):
    """Accept list of hex strings or dicts with public_key_hex; return unique hex list."""
    out = []
    for entry in entries or []:
        value = entry.get("public_key_hex") if isinstance(entry, dict) else entry
        if not isinstance(value, str) or not HEX64.fullmatch(value):
            raise ValueError("trusted key must be 64-char lowercase ed25519 hex")
        if value not in out:
            out.append(value)
    return out


def verify(message_bytes, signature_hex, public_keys):
    """True only if a supplied key validates the signature. Any failure -> False."""
    if not backend_available():
        return False
    if not isinstance(signature_hex, str) or len(signature_hex) != 128:
        return False
    if isinstance(public_keys, str):
        keys = [public_keys]
    elif isinstance(public_keys, list):
        keys = public_keys
    else:
        return False
    try:
        raw = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    for candidate in keys:
        if not isinstance(candidate, str) or not HEX64.fullmatch(candidate):
            continue
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(candidate)).verify(raw, message_bytes)
            return True
        except (InvalidSignature, ValueError, OSError):
            continue
    return False
