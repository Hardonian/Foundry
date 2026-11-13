"""Ed25519 signing and verification for promotion attestations.

Fail-closed by design: if the optional `signing` extra (cryptography) is not
installed, nothing verifies and callers must treat that as a blocking missing
evidence item. Private key material must be file-based, owner-only (mode 0600 on
POSIX, restricted DACL without broad group/world access on Windows), and is never
logged or embedded into run records. Public keys are stored as 64-char lowercase
hex (ed25519 raw 32-byte encoding).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Iterable
from pathlib import Path
from typing import Any

HEX64 = re.compile(r"^[0-9a-f]{64}$")

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    SIGNING_BACKEND: str | None = "cryptography"
except ImportError:  # pragma: no cover - exercised only without the extra
    SIGNING_BACKEND = None

_KEYRING_TOP_FIELDS = frozenset({
    "schema_version",
    "version",
    "trusted_keys",
    "required_signatures",
    "required_roles",
    "description",
    "notes",
})

# Broad Windows SIDs that must never hold access to private key files.
_WIN_BROAD_SIDS = frozenset({
    "S-1-1-0",       # Everyone
    "S-1-5-7",       # Anonymous Logon
    "S-1-5-11",      # Authenticated Users
    "S-1-5-32-545",  # BUILTIN\Users
    "S-1-5-32-546",  # BUILTIN\Guests
})
# Privileged/self SIDs expected on owner-created NTFS files.
_WIN_SAFE_EXPLICIT_SIDS = frozenset({
    "S-1-5-18",      # NT AUTHORITY\SYSTEM
    "S-1-5-32-544",  # BUILTIN\Administrators
    "S-1-3-4",       # OWNER RIGHTS
})


class SigningUnavailable(RuntimeError):
    """Raised when a signing operation is requested but the backend is absent."""


def backend_available() -> bool:
    """Return True when the cryptography Ed25519 backend is installed."""
    return SIGNING_BACKEND is not None


def _check_windows_key_acl(private_key_path: str | os.PathLike[str]) -> None:
    """Reject Windows private key files whose NTFS DACL grants broad or unexpected access."""
    import ctypes
    import ctypes.wintypes as wt

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    se_file_object = 1
    owner_sec_info = 0x00000001
    dacl_sec_info = 0x00000004
    acl_size_info_class = 2
    access_allowed_ace_type = 0
    inherit_only_ace = 0x08
    inherited_ace = 0x10

    class _AclSizeInfo(ctypes.Structure):
        _fields_ = [("AceCount", wt.DWORD), ("AclBytesInUse", wt.DWORD), ("AclBytesFree", wt.DWORD)]

    class _AceHeader(ctypes.Structure):
        _fields_ = [("AceType", ctypes.c_ubyte), ("AceFlags", ctypes.c_ubyte), ("AceSize", wt.WORD)]

    class _AccessAllowedAce(ctypes.Structure):
        _fields_ = [("Header", _AceHeader), ("Mask", wt.DWORD), ("SidStart", wt.DWORD)]

    advapi32.GetNamedSecurityInfoW.argtypes = [
        wt.LPCWSTR, wt.DWORD, wt.DWORD,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.GetNamedSecurityInfoW.restype = wt.DWORD
    advapi32.GetAclInformation.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wt.DWORD, wt.DWORD]
    advapi32.GetAclInformation.restype = wt.BOOL
    advapi32.GetAce.argtypes = [ctypes.c_void_p, wt.DWORD, ctypes.POINTER(ctypes.c_void_p)]
    advapi32.GetAce.restype = wt.BOOL
    advapi32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wt.LPWSTR)]
    advapi32.ConvertSidToStringSidW.restype = wt.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p

    p_owner = ctypes.c_void_p()
    p_dacl = ctypes.c_void_p()
    p_sd = ctypes.c_void_p()
    err = advapi32.GetNamedSecurityInfoW(
        os.fspath(private_key_path),
        se_file_object,
        owner_sec_info | dacl_sec_info,
        ctypes.byref(p_owner),
        None,
        ctypes.byref(p_dacl),
        None,
        ctypes.byref(p_sd),
    )
    if err != 0:
        raise ValueError(f"cannot verify private key Windows ACL (error {err})")

    def _sid_to_str(sid_ptr: ctypes.c_void_p | int) -> str | None:
        sid_str = wt.LPWSTR()
        if not advapi32.ConvertSidToStringSidW(sid_ptr, ctypes.byref(sid_str)):
            return None
        try:
            return sid_str.value
        finally:
            kernel32.LocalFree(sid_str)

    try:
        if not p_dacl.value:
            raise ValueError("private key must not have an unrestricted NULL DACL")
        owner_sid = _sid_to_str(p_owner) if p_owner.value else None
        size_info = _AclSizeInfo()
        if not advapi32.GetAclInformation(p_dacl, ctypes.byref(size_info), ctypes.sizeof(size_info), acl_size_info_class):
            raise ValueError("cannot inspect private key DACL entries")

        safe_sids = set(_WIN_SAFE_EXPLICIT_SIDS)
        if owner_sid:
            safe_sids.add(owner_sid)

        for idx in range(size_info.AceCount):
            p_ace = ctypes.c_void_p()
            if not advapi32.GetAce(p_dacl, idx, ctypes.byref(p_ace)) or not p_ace.value:
                raise ValueError("cannot read private key ACE")
            ace_addr = p_ace.value
            hdr = _AceHeader.from_address(ace_addr)
            if hdr.AceType != access_allowed_ace_type or (hdr.AceFlags & inherit_only_ace):
                continue
            sid_ptr = ctypes.c_void_p(ace_addr + _AccessAllowedAce.SidStart.offset)
            sid_str = _sid_to_str(sid_ptr)
            if sid_str is None:
                raise ValueError("cannot decode private key ACE principal")
            if sid_str in _WIN_BROAD_SIDS:
                raise ValueError("private key must not be accessible by group or others (broad Windows ACL)")
            if not (hdr.AceFlags & inherited_ace) and sid_str not in safe_sids:
                raise ValueError("private key must not grant explicit access to other principals")
    finally:
        if p_sd.value:
            kernel32.LocalFree(p_sd)


def _check_private_key_permissions(private_key_path: str | os.PathLike[str], info: os.stat_result) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("private key path is not a regular file")
    if os.name == "posix":
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("private key must not be accessible by group or others (need mode 0600)")
    elif os.name == "nt":
        _check_windows_key_acl(private_key_path)


def generate_keypair(private_key_path: str | os.PathLike[str]) -> dict[str, str]:
    """Create a new Ed25519 private key PEM at path (refuses if it exists).

    Returns {"public_key_hex", "private_key_path"}. File is mode 0600 on POSIX
    and checked against broad Windows ACLs on Windows.
    """
    if not backend_available():
        raise SigningUnavailable("signing backend unavailable")
    path_str = os.fspath(private_key_path)
    if os.path.exists(path_str):
        raise ValueError(f"refusing to overwrite existing key file: {path_str}")
    private_key = Ed25519PrivateKey.generate()
    payload = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    directory = os.path.dirname(os.path.abspath(path_str))
    os.makedirs(directory, exist_ok=True)
    descriptor = os.open(
        path_str,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    public_hex = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    ).hex()
    return {"public_key_hex": public_hex, "private_key_path": path_str}


def _load_private_key(private_key_path: str | os.PathLike[str]) -> Any:
    if not backend_available():
        raise SigningUnavailable("signing backend unavailable")
    info = os.stat(private_key_path)
    _check_private_key_permissions(private_key_path, info)
    with open(private_key_path, "rb") as handle:
        key = serialization.load_pem_private_key(handle.read(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("private key must be an Ed25519 key")
    return key


def public_key_hex(private_key_path: str | os.PathLike[str]) -> str:
    """Return the 64-char lowercase raw Ed25519 public key hex for private_key_path."""
    if not backend_available():
        raise SigningUnavailable("signing backend unavailable")
    key = _load_private_key(private_key_path)
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    ).hex()


def sign(message_bytes: bytes, private_key_path: str | os.PathLike[str]) -> str:
    """Sign exact bytes with the file-based private key; returns 128-char hex."""
    if not backend_available():
        raise SigningUnavailable("signing backend unavailable")
    key = _load_private_key(private_key_path)
    return key.sign(message_bytes).hex()


def fingerprint(public_hex: str) -> str:
    """Return SHA-256 hex fingerprint of raw Ed25519 public key bytes."""
    return hashlib.sha256(bytes.fromhex(public_hex)).hexdigest()


def normalize_trusted_keys(entries: Iterable[str | dict[str, Any]] | None) -> list[str]:
    """Accept list of hex strings or dicts with public_key_hex; return unique hex list."""
    if entries is None:
        return []
    if isinstance(entries, (str, bytes)) or not isinstance(entries, (list, tuple, set)):
        raise TypeError("trusted keys must be a list of hex strings or key dicts")
    out: list[str] = []
    for entry in entries:
        value = entry.get("public_key_hex") if isinstance(entry, dict) else entry
        if not isinstance(value, str) or not HEX64.fullmatch(value):
            raise ValueError("trusted key must be 64-char lowercase ed25519 hex")
        if value not in out:
            out.append(value)
    return out


def _clean_optional_str(value: Any, *, field: str, limit: int = 128) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or len(value) > limit
        or any(ord(ch) < 32 for ch in value)
    ):
        raise ValueError(f"{field} must be a clean nonempty string <= {limit} chars when present")
    return value


def parse_keyring(data: Any) -> dict[str, Any]:
    """Parse and validate a keyring policy dict or key list.

    Supports optional key_id, role, revoked_at on key entries, plus policy fields
    required_signatures (default 1) and required_roles (default []).
    """
    if isinstance(data, list):
        raw_keys = data
        required_signatures: Any = 1
        required_roles_raw: Any = []
    elif isinstance(data, dict):
        unknown = sorted(set(data) - _KEYRING_TOP_FIELDS)
        if unknown:
            raise ValueError(f"unknown keyring policy fields: {', '.join(unknown)}")
        if not isinstance(data.get("trusted_keys"), list):
            raise TypeError("keyring must be an object with a trusted_keys list")
        raw_keys = data["trusted_keys"]
        required_signatures = data.get("required_signatures", 1)
        required_roles_raw = data.get("required_roles", [])
    else:
        raise TypeError("keyring must be a trusted_keys list or keyring policy object")

    if type(required_signatures) is not int or required_signatures < 1:
        raise ValueError("required_signatures must be an integer >= 1")

    if isinstance(required_roles_raw, (str, bytes)) or not isinstance(required_roles_raw, (list, tuple)):
        raise TypeError("required_roles must be a list of role strings")
    required_roles: list[str] = []
    for role_val in required_roles_raw:
        clean_role = _clean_optional_str(role_val, field="required_roles[]", limit=64)
        if clean_role is None:
            raise ValueError("required_roles must not contain null entries")
        if clean_role not in required_roles:
            required_roles.append(clean_role)

    keys: list[dict[str, Any]] = []
    active_keys: list[str] = []
    revoked_keys: list[str] = []
    keys_by_hex: dict[str, dict[str, Any]] = {}

    for entry in raw_keys:
        if isinstance(entry, str):
            raw_pub: Any = entry
            key_id: str | None = None
            role: str | None = None
            revoked_at: str | None = None
            label: str | None = None
        elif isinstance(entry, dict):
            raw_pub = entry.get("public_key_hex")
            key_id = _clean_optional_str(entry.get("key_id"), field="key_id", limit=128)
            role = _clean_optional_str(entry.get("role"), field="role", limit=64)
            revoked_at = _clean_optional_str(entry.get("revoked_at"), field="revoked_at", limit=64)
            label = _clean_optional_str(entry.get("label"), field="label", limit=128)
        else:
            raise TypeError("trusted key entry must be a 64-char hex string or dict")

        if not isinstance(raw_pub, str) or not HEX64.fullmatch(raw_pub):
            raise ValueError("trusted key must be 64-char lowercase ed25519 hex")
        pub = raw_pub
        if pub in keys_by_hex:
            raise ValueError(f"duplicate public_key_hex in keyring: {pub}")

        revoked = revoked_at is not None
        parsed_entry: dict[str, Any] = {
            "public_key_hex": pub,
            "key_id": key_id,
            "role": role,
            "revoked_at": revoked_at,
            "revoked": revoked,
        }
        if label is not None:
            parsed_entry["label"] = label
        keys.append(parsed_entry)
        keys_by_hex[pub] = parsed_entry
        if revoked:
            revoked_keys.append(pub)
        else:
            active_keys.append(pub)

    return {
        "keys": keys,
        "active_keys": active_keys,
        "revoked_keys": revoked_keys,
        "required_signatures": required_signatures,
        "required_roles": required_roles,
        "keys_by_hex": keys_by_hex,
    }


def load_keyring_policy(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate a keyring policy JSON file from disk."""
    with open(Path(path), "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError("keyring file must contain a JSON object")
    return parse_keyring(data)


def check_keyring_quorum(policy: dict[str, Any], verified_public_keys: Iterable[str]) -> dict[str, Any]:
    """Check whether distinct verified active signers satisfy M-of-N and role quorum."""
    keys_by_hex: dict[str, dict[str, Any]] = policy["keys_by_hex"]
    active_set = set(policy["active_keys"])
    valid_signers: list[str] = []
    revoked_signers: list[str] = []
    roles_satisfied: set[str] = set()

    for pub in verified_public_keys:
        entry = keys_by_hex.get(pub)
        if entry is None:
            continue
        if pub not in active_set or entry.get("revoked"):
            if pub not in revoked_signers:
                revoked_signers.append(pub)
            continue
        if pub not in valid_signers:
            valid_signers.append(pub)
            if entry.get("role"):
                roles_satisfied.add(entry["role"])

    required_signatures = int(policy["required_signatures"])
    required_roles = list(policy["required_roles"])
    missing_roles = [role for role in required_roles if role not in roles_satisfied]
    satisfied = len(valid_signers) >= required_signatures and not missing_roles
    return {
        "satisfied": satisfied,
        "valid_signers": valid_signers,
        "revoked_signers": revoked_signers,
        "valid_count": len(valid_signers),
        "required_signatures": required_signatures,
        "roles_satisfied": sorted(roles_satisfied),
        "missing_roles": missing_roles,
    }


def verify(message_bytes: bytes, signature_hex: Any, public_keys: Any) -> bool:
    """True only if a supplied key validates the signature. Any failure -> False."""
    if not backend_available():
        return False
    if not isinstance(signature_hex, str) or len(signature_hex) != 128:
        return False
    if isinstance(public_keys, str):
        keys: list[Any] = [public_keys]
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
