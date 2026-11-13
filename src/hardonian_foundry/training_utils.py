"""Shared offline training, evaluation, and preflight utilities.

Synthetic-only, offline-by-default helper layer shared across Foundry training
scripts and case studies. Enforces pinned local snapshot verification,
cross-platform private directory checks, non-blocking cooperative locks,
memory/disk preflight gates, padded micro-batching, batched greedy decoding,
and typed hooks for device selection and post-training methods.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, TextIO, TypeVar

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback uses msvcrt
    fcntl = None  # type: ignore[assignment]

if os.name == "nt":
    import msvcrt
else:  # pragma: no cover - POSIX uses fcntl
    msvcrt = None  # type: ignore[assignment]

T = TypeVar("T")

DEFAULT_MODEL_ID: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEFAULT_REVISION: str = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
DEFAULT_MODEL_SHA256: str = "f9523886352217ded3aeeef552b381af79d568c6d49a4b9e423288cea56b0a44"
DEFAULT_BLOBS: dict[str, str] = {
    "config.json": "e2aa18293b0dd66539341467fa454878a5adb2be",
    "generation_config.json": "c28f9c697cbc09f047434efec57556505af31111",
    "merges.txt": "20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0",
    "tokenizer.json": "443909a61d429dff23010e5bddd28ff530edda00",
    "tokenizer_config.json": "acee076f49bf3c0298e15de0909d1da7b392f0c3",
    "vocab.json": "4783fe10ac3adce15ac8f358ef5462739852c569",
}
DEFAULT_LORA_TARGETS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

MIN_FREE_DISK_BYTES: int = 10 * 1024**3
MAX_DISK_UTILIZATION: float = 0.80
MIN_AVAILABLE_RAM_BYTES: int = 8 * 1024**3

SUPPORTED_DEVICES: tuple[str, ...] = ("cpu", "cuda", "mps", "dml", "auto")
SUPPORTED_METHODS: tuple[str, ...] = ("sft", "lora", "qlora", "dpo", "rlvr")

_HEX40_CHARS = frozenset("0123456789abcdef")
_HEX64_CHARS = frozenset("0123456789abcdef")

# Broad non-owner Windows SIDs that must never have access to private output roots.
_WIN_BROAD_SIDS: frozenset[str] = frozenset({
    "S-1-1-0",       # Everyone
    "S-1-5-11",      # Authenticated Users
    "S-1-5-32-545",  # BUILTIN\Users
    "S-1-5-32-546",  # BUILTIN\Guests
})
_WIN_ADMIN_SID = "S-1-5-32-544"


class TrainingLockError(BlockingIOError):
    """Raised when another participating Foundry job holds the training lock."""


@dataclass(frozen=True)
class ModelSpec:
    """Pinned model specification loaded from defaults or a model registry JSON."""

    model_id: str
    revision: str
    model_sha256: str | None = None
    blobs: dict[str, str] = field(default_factory=dict)
    weight_files: dict[str, str] = field(default_factory=dict)
    lora_target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_LORA_TARGETS))
    default_lora_r: int = 8
    default_lora_alpha: int = 16
    license_id: str = "Apache-2.0"
    max_seq: int = 256

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeviceSpec:
    """Resolved execution device metadata with offline-CPU default."""

    device: str
    backend: str
    torch_dtype_name: str
    allow_gpu: bool
    note: str = ""


@dataclass(frozen=True)
class GenerationResult:
    """One decoded prediction from batched greedy generation."""

    text: str
    token_ids: list[int]
    max_token_limit_hit: bool
    confidence: float | None
    latency_ms: float


def sha_bytes(value: bytes) -> str:
    """Return lowercase SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(value).hexdigest()


def sha_file(path: str | Path) -> str:
    """Return lowercase SHA-256 hex digest of a file read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(payload: bytes) -> str:
    """Compute Git blob SHA-1 object ID for Hugging Face Hub metadata verification."""
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def snapshot_path(
    hf_home: str | Path,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
) -> Path:
    """Resolve the Hugging Face Hub local snapshot path for a pinned model/revision."""
    repo_dir = f"models--{model_id.replace('/', '--')}"
    return Path(hf_home).expanduser() / "hub" / repo_dir / "snapshots" / revision


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(ch in _HEX64_CHARS for ch in value)
    )


def verify_pinned_blobs(
    snap: str | Path,
    blobs: Mapping[str, str] | None = None,
) -> None:
    """Verify Git blob SHA-1 digests for pinned metadata/tokenizer files in a snapshot."""
    snap_dir = Path(snap)
    expected_blobs = DEFAULT_BLOBS if blobs is None else blobs
    for filename, expected_sha1 in expected_blobs.items():
        target = snap_dir / filename
        if not target.is_file():
            raise ValueError(f"required pinned model files are missing: {filename}; no download attempted")
        if not _is_hex(expected_sha1, 40):
            raise ValueError(f"invalid expected Git blob SHA-1 for {filename}")
        actual = git_blob_sha1(target.read_bytes())
        if actual != expected_sha1:
            raise ValueError(f"cached model metadata mismatch: {filename}")


def verify_snapshot(
    snap: str | Path,
    *,
    model_sha256: str | None = DEFAULT_MODEL_SHA256,
    blobs: Mapping[str, str] | None = DEFAULT_BLOBS,
    weight_files: Mapping[str, str] | None = None,
    require_config: bool = True,
) -> dict[str, Any]:
    """Fail-closed verification of a local model snapshot without network access.

    Checks required safetensors weights (single or sharded), Git blob IDs for
    tokenizer/config artifacts, and rejects any `auto_map` remote custom code.
    """
    snap_dir = Path(snap)
    if not snap_dir.is_dir():
        raise ValueError("required pinned model files are missing; no download attempted")

    weights_hashes: dict[str, str] = {}
    if weight_files:
        for rel_name, expected_hash in weight_files.items():
            weight_path = snap_dir / rel_name
            if not weight_path.is_file():
                raise ValueError(
                    f"required pinned model files are missing ({rel_name}); no download attempted"
                )
            actual_hash = sha_file(weight_path)
            if expected_hash and actual_hash != expected_hash:
                raise ValueError(f"cached model SHA-256 mismatch: {rel_name}")
            weights_hashes[rel_name] = actual_hash
    else:
        single_weight = snap_dir / "model.safetensors"
        if not single_weight.is_file():
            raise ValueError("required pinned model files are missing; no download attempted")
        actual_hash = sha_file(single_weight)
        if model_sha256 is not None and actual_hash != model_sha256:
            raise ValueError("cached model SHA-256 mismatch")
        weights_hashes["model.safetensors"] = actual_hash

    config_path = snap_dir / "config.json"
    if require_config and not config_path.is_file():
        raise ValueError("required pinned model files are missing; no download attempted")

    if blobs:
        missing = [name for name in blobs if not (snap_dir / name).is_file()]
        if missing:
            raise ValueError(
                f"required pinned model files are missing ({', '.join(missing)}); no download attempted"
            )
        verify_pinned_blobs(snap_dir, blobs)

    config: dict[str, Any] = {}
    if config_path.is_file():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("model config.json must be a JSON object")
        config = loaded
        if config.get("auto_map"):
            raise ValueError("remote custom model code (auto_map) is not permitted")

    tok_cfg_path = snap_dir / "tokenizer_config.json"
    if tok_cfg_path.is_file():
        tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
        if isinstance(tok_cfg, dict) and tok_cfg.get("auto_map"):
            raise ValueError("remote custom tokenizer code (auto_map) is not permitted")

    primary_sha = weights_hashes.get("model.safetensors") or next(iter(weights_hashes.values()), None)
    return {
        "snapshot": str(snap_dir),
        "config": config,
        "weights_sha256": primary_sha,
        "weight_files": weights_hashes,
    }


def load_verified_snapshot(
    snap: str | Path,
    tokenizer_cls: Any,
    model_cls: Any,
    torch_module: Any,
    *,
    device: str = "cpu",
    dtype: Any = None,
    quantization_config: Any = None,
) -> tuple[Any, Any]:
    """Load only a preflight-verified local snapshot, forbidding Hub downloads and remote code."""
    snap_str = str(snap)
    tokenizer = tokenizer_cls.from_pretrained(
        snap_str,
        local_files_only=True,
        trust_remote_code=False,
        token=False,
    )
    resolved_dtype = dtype if dtype is not None else torch_module.float32
    load_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "token": False,
        "torch_dtype": resolved_dtype,
        "attn_implementation": "eager",
    }
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    model = model_cls.from_pretrained(snap_str, **load_kwargs)
    if quantization_config is None:
        model = model.to(device)
    return tokenizer, model


def set_offline_env(
    hf_home: str | Path,
    *,
    threads: int | None = None,
    allow_gpu: bool = False,
) -> dict[str, str]:
    """Configure environment variables for strict offline Hugging Face / Transformers execution."""
    updates: dict[str, str] = {
        "HF_HOME": str(Path(hf_home)),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",  # nosec B105 - opt-out flag disabling token discovery
        "TOKENIZERS_PARALLELISM": "false",
    }
    if not allow_gpu:
        updates["CUDA_VISIBLE_DEVICES"] = ""
    if threads is not None:
        updates["OMP_NUM_THREADS"] = str(threads)
        updates["MKL_NUM_THREADS"] = str(threads)
    os.environ.update(updates)
    return updates


def _verify_windows_private_dir(resolved: Path, label: str) -> None:
    """Inspect Windows owner and NTFS DACL via advapi32/kernel32 ctypes."""
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _AclSizeInfo(ctypes.Structure):
        _fields_ = [
            ("AceCount", wintypes.DWORD),
            ("AclBytesInUse", wintypes.DWORD),
            ("AclBytesFree", wintypes.DWORD),
        ]

    class _AceHeader(ctypes.Structure):
        _fields_ = [
            ("AceType", ctypes.c_ubyte),
            ("AceFlags", ctypes.c_ubyte),
            ("AceSize", wintypes.WORD),
        ]

    class _SidAndAttrs(ctypes.Structure):
        _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]

    class _TokenUser(ctypes.Structure):
        _fields_ = [("User", _SidAndAttrs)]

    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p

    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.ConvertSidToStringSidW.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_wchar_p),
    ]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    advapi32.ConvertStringSidToSidW.argtypes = [
        wintypes.LPCWSTR,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.ConvertStringSidToSidW.restype = wintypes.BOOL
    advapi32.CheckTokenMembership.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.BOOL),
    ]
    advapi32.CheckTokenMembership.restype = wintypes.BOOL
    advapi32.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPCWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi32.GetAclInformation.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_int,
    ]
    advapi32.GetAclInformation.restype = wintypes.BOOL
    advapi32.GetAce.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.GetAce.restype = wintypes.BOOL

    def sid_to_str(sid_ptr: ctypes.c_void_p) -> str | None:
        out_str = ctypes.c_wchar_p()
        if not advapi32.ConvertSidToStringSidW(sid_ptr, ctypes.byref(out_str)):
            return None
        try:
            return out_str.value
        finally:
            kernel32.LocalFree(out_str)

    # Current process user SID
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
        raise OSError(ctypes.get_last_error(), "OpenProcessToken failed")
    user_sid_str: str | None = None
    is_admin = False
    try:
        needed = wintypes.DWORD(0)
        advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))
        buf = ctypes.create_string_buffer(needed.value)
        if not advapi32.GetTokenInformation(token, 1, buf, needed, ctypes.byref(needed)):
            raise OSError(ctypes.get_last_error(), "GetTokenInformation failed")
        tu = _TokenUser.from_buffer(buf)
        user_sid_str = sid_to_str(ctypes.c_void_p(tu.User.Sid))

        admin_sid = ctypes.c_void_p()
        if advapi32.ConvertStringSidToSidW(_WIN_ADMIN_SID, ctypes.byref(admin_sid)):
            try:
                member = wintypes.BOOL(False)
                if advapi32.CheckTokenMembership(None, admin_sid, ctypes.byref(member)):
                    is_admin = bool(member.value)
            finally:
                kernel32.LocalFree(admin_sid)
    finally:
        kernel32.CloseHandle(token)

    p_owner = ctypes.c_void_p()
    p_dacl = ctypes.c_void_p()
    p_sd = ctypes.c_void_p()
    # SE_FILE_OBJECT=1, OWNER_SECURITY_INFORMATION(1) | DACL_SECURITY_INFORMATION(4) = 5
    rc = advapi32.GetNamedSecurityInfoW(
        str(resolved),
        1,
        5,
        ctypes.byref(p_owner),
        None,
        ctypes.byref(p_dacl),
        None,
        ctypes.byref(p_sd),
    )
    if rc != 0:
        raise OSError(rc, f"GetNamedSecurityInfoW failed for {resolved}")

    try:
        owner_sid = sid_to_str(p_owner)
        if owner_sid != user_sid_str and not (is_admin and owner_sid == _WIN_ADMIN_SID):
            raise ValueError(f"{label} must be an existing owner-only directory: {resolved}")
        if not p_dacl.value:
            # A NULL DACL on Windows grants Everyone full control.
            raise ValueError(f"{label} must be an existing owner-only directory: {resolved}")

        acl_info = _AclSizeInfo()
        if not advapi32.GetAclInformation(
            p_dacl, ctypes.byref(acl_info), ctypes.sizeof(acl_info), 2
        ):
            raise OSError(ctypes.get_last_error(), "GetAclInformation failed")

        for idx in range(acl_info.AceCount):
            p_ace = ctypes.c_void_p()
            if not advapi32.GetAce(p_dacl, idx, ctypes.byref(p_ace)) or not p_ace.value:
                continue
            hdr = _AceHeader.from_address(p_ace.value)
            # Skip INHERIT_ONLY_ACE (0x08), which does not apply to the directory itself.
            if hdr.AceFlags & 0x08:
                continue
            sid_offset: int | None = None
            if hdr.AceType == 0:  # ACCESS_ALLOWED_ACE_TYPE
                sid_offset = 8
            elif hdr.AceType == 5:  # ACCESS_ALLOWED_OBJECT_ACE_TYPE
                obj_flags = ctypes.c_uint32.from_address(p_ace.value + 8).value
                offset = 12
                if obj_flags & 0x1:
                    offset += 16
                if obj_flags & 0x2:
                    offset += 16
                sid_offset = offset
            if sid_offset is None or sid_offset >= hdr.AceSize:
                continue
            ace_sid = sid_to_str(ctypes.c_void_p(p_ace.value + sid_offset))
            if ace_sid in _WIN_BROAD_SIDS:
                raise ValueError(f"{label} must be an existing owner-only directory: {resolved}")
    finally:
        if p_sd.value:
            kernel32.LocalFree(p_sd)


def verify_private_dir(path: str | Path, *, label: str = "directory") -> Path:
    """Verify that `path` is an existing, non-symlink, owner-only directory.

    Enforces strict POSIX `st_uid == getuid()` and `(st_mode & 0o077) == 0` on
    POSIX platforms, and verifies NTFS owner + absence of broad principals
    (Everyone, Authenticated Users, Users, Guests, NULL DACL) on Windows.
    """
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ValueError(f"{label} must be an existing owner-only directory (symlinks not allowed): {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} must be an existing owner-only directory: {raw}") from exc
    if not resolved.is_dir():
        raise ValueError(f"{label} must be an existing owner-only directory: {resolved}")

    if os.name == "nt":
        _verify_windows_private_dir(resolved, label)
        return resolved

    st = resolved.stat()
    getuid = getattr(os, "getuid", None)
    if getuid is not None and st.st_uid != getuid():
        raise ValueError(f"{label} must be an existing owner-only directory: {resolved}")
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise ValueError(f"{label} must be an existing owner-only directory: {resolved}")
    return resolved


def mkdir_private(path: str | Path, *, parents: bool = False, exist_ok: bool = False) -> Path:
    """Create a private directory (mode 0700 on POSIX)."""
    target = Path(path)
    target.mkdir(mode=0o700, parents=parents, exist_ok=exist_ok)
    if os.name != "nt":
        os.chmod(target, 0o700)
    return target


def chmod_private(path: str | Path) -> None:
    """Restrict a private file to mode 0600 on POSIX."""
    if os.name != "nt":
        os.chmod(path, 0o600)


def available_ram_bytes() -> int:
    """Return available physical RAM in bytes across Linux, macOS, and Windows."""
    if os.name == "nt":
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
        kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
        return int(status.ullAvailPhys)

    sysconf = getattr(os, "sysconf", None)
    sysconf_names = getattr(os, "sysconf_names", {})
    if sysconf is not None and "SC_AVPHYS_PAGES" in sysconf_names and "SC_PAGE_SIZE" in sysconf_names:
        return int(sysconf("SC_AVPHYS_PAGES")) * int(sysconf("SC_PAGE_SIZE"))

    if sys.platform == "darwin":
        try:
            vm_stat = subprocess.check_output(["vm_stat"], text=True, timeout=5)  # nosec B603 B607
            page_size = 4096
            pages_free = 0
            for line in vm_stat.splitlines():
                if "page size of" in line:
                    digits = "".join(ch for ch in line if ch.isdigit())
                    if digits:
                        page_size = int(digits)
                elif line.startswith(("Pages free:", "Pages inactive:", "Pages speculative:")):
                    val = line.split(":", 1)[1].strip().rstrip(".")
                    if val.isdigit():
                        pages_free += int(val)
            if pages_free > 0:
                return pages_free * page_size
        except (OSError, subprocess.SubprocessError, ValueError):
            pass

    if sysconf is not None and "SC_PHYS_PAGES" in sysconf_names and "SC_PAGE_SIZE" in sysconf_names:
        return int(sysconf("SC_PHYS_PAGES")) * int(sysconf("SC_PAGE_SIZE"))
    raise OSError("unable to determine available physical RAM on this platform")


def check_disk_space(
    path: str | Path,
    *,
    min_free_bytes: int = MIN_FREE_DISK_BYTES,
    max_used_ratio: float = MAX_DISK_UTILIZATION,
) -> tuple[int, int, int]:
    """Verify minimum free disk bytes and maximum utilization watermark."""
    usage = shutil.disk_usage(path)
    used_ratio = (usage.used / usage.total) if usage.total > 0 else 1.0
    if usage.free < min_free_bytes or used_ratio >= max_used_ratio:
        raise ValueError(
            f"output filesystem below {min_free_bytes / (1024**3):.1f} GiB free "
            f"or at least {max_used_ratio * 100:.0f}% utilized"
        )
    return (int(usage.total), int(usage.used), int(usage.free))


@contextmanager
def acquire_training_lock(lock_path: str | Path) -> Iterator[TextIO]:
    """Acquire an exclusive non-blocking cooperative lock across POSIX (fcntl) and Windows (msvcrt)."""
    path = Path(lock_path)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "r+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise TrainingLockError("another participating Foundry training job holds the lock") from exc
            try:
                yield lock_file
            finally:
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                except OSError:
                    pass
        elif msvcrt is not None:
            lock_file.seek(0)
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise TrainingLockError("another participating Foundry training job holds the lock") from exc
            try:
                yield lock_file
            finally:
                try:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        else:  # pragma: no cover
            yield lock_file


def _default_model_entries() -> dict[str, ModelSpec]:
    return {
        DEFAULT_MODEL_ID: ModelSpec(
            model_id=DEFAULT_MODEL_ID,
            revision=DEFAULT_REVISION,
            model_sha256=DEFAULT_MODEL_SHA256,
            blobs=dict(DEFAULT_BLOBS),
            weight_files={},
            lora_target_modules=list(DEFAULT_LORA_TARGETS),
            default_lora_r=8,
            default_lora_alpha=16,
            license_id="Apache-2.0",
            max_seq=256,
        )
    }


def _parse_model_entry(model_id_hint: str | None, raw: Mapping[str, Any]) -> ModelSpec:
    if not isinstance(raw, Mapping):
        raise TypeError("model registry entry must be a JSON object")
    model_id = raw.get("model_id", model_id_hint)
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model registry entry missing nonempty model_id")
    revision = raw.get("revision")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError(f"model {model_id}: revision must be a nonempty string")

    model_sha256 = raw.get("model_sha256")
    if model_sha256 is not None and not _is_hex(model_sha256, 64):
        raise ValueError(f"model {model_id}: model_sha256 must be 64-char lowercase hex")

    raw_blobs = raw.get("blobs", {})
    if not isinstance(raw_blobs, Mapping):
        raise TypeError(f"model {model_id}: blobs must be an object")
    blobs: dict[str, str] = {}
    for fname, sha1 in raw_blobs.items():
        if not isinstance(fname, str) or not fname or not _is_hex(sha1, 40):
            raise ValueError(f"model {model_id}: invalid blob digest for {fname!r}")
        blobs[fname] = sha1

    raw_weights = raw.get("weight_files", {})
    if not isinstance(raw_weights, Mapping):
        raise TypeError(f"model {model_id}: weight_files must be an object")
    weight_files: dict[str, str] = {}
    for fname, sha256_hex in raw_weights.items():
        if not isinstance(fname, str) or not fname or not _is_hex(sha256_hex, 64):
            raise ValueError(f"model {model_id}: invalid weight_files digest for {fname!r}")
        weight_files[fname] = sha256_hex

    raw_targets = raw.get("lora_target_modules", list(DEFAULT_LORA_TARGETS))
    if (
        not isinstance(raw_targets, list)
        or not raw_targets
        or any(not isinstance(t, str) or not t.strip() for t in raw_targets)
    ):
        raise ValueError(f"model {model_id}: lora_target_modules must be a nonempty list of strings")

    r_val = raw.get("default_lora_r", 8)
    alpha_val = raw.get("default_lora_alpha", 16)
    max_seq = raw.get("max_seq", 256)
    for field_name, val in (("default_lora_r", r_val), ("default_lora_alpha", alpha_val), ("max_seq", max_seq)):
        if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
            raise ValueError(f"model {model_id}: {field_name} must be a positive integer")

    license_id = raw.get("license_id", "Apache-2.0")
    if not isinstance(license_id, str) or not license_id.strip():
        raise ValueError(f"model {model_id}: license_id must be a nonempty string")

    return ModelSpec(
        model_id=model_id,
        revision=revision,
        model_sha256=model_sha256,
        blobs=blobs,
        weight_files=weight_files,
        lora_target_modules=list(raw_targets),
        default_lora_r=r_val,
        default_lora_alpha=alpha_val,
        license_id=license_id,
        max_seq=max_seq,
    )


def model_registry(
    path: str | Path | None = None,
    *,
    include_defaults: bool = True,
) -> dict[str, ModelSpec]:
    """Load configurable model/LoRA specifications from a JSON registry file.

    Accepts JSON formatted as `{"models": {<id>: {...}}}`, `{<id>: {...}}`, or
    `[{"model_id": ..., ...}]`. Merges onto built-in pinned defaults when
    `include_defaults=True`.
    """
    entries: dict[str, ModelSpec] = _default_model_entries() if include_defaults else {}
    if path is None:
        return entries

    reg_path = Path(path)
    raw = json.loads(reg_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        container = raw.get("models") if "models" in raw and isinstance(raw["models"], (dict, list)) else raw
        if isinstance(container, dict):
            for key, item in container.items():
                spec = _parse_model_entry(key, item)
                entries[spec.model_id] = spec
        elif isinstance(container, list):
            for item in container:
                spec = _parse_model_entry(None, item)
                entries[spec.model_id] = spec
        else:  # pragma: no cover
            raise TypeError("model registry JSON must map model IDs to objects")
    elif isinstance(raw, list):
        for item in raw:
            spec = _parse_model_entry(None, item)
            entries[spec.model_id] = spec
    else:
        raise TypeError("model registry JSON must be an object or list")
    return entries


def resolve_model_spec(
    model_id: str | None = None,
    *,
    revision: str | None = None,
    registry_path: str | Path | None = None,
) -> ModelSpec:
    """Resolve a ModelSpec from defaults or a user-supplied model registry JSON."""
    registry = model_registry(registry_path, include_defaults=True)
    target_id = model_id or DEFAULT_MODEL_ID
    if target_id not in registry:
        raise ValueError(f"model {target_id!r} not found in model registry")
    spec = registry[target_id]
    if revision is not None and revision != spec.revision:
        raise ValueError(f"revision mismatch for {target_id}: expected {spec.revision}, got {revision}")
    return spec


def select_device(requested: str = "cpu", torch_module: Any | None = None) -> DeviceSpec:
    """Select execution device with strict offline-CPU default and fail-closed explicit backends."""
    if requested not in SUPPORTED_DEVICES:
        raise ValueError(f"unsupported device {requested!r}; expected one of {SUPPORTED_DEVICES}")
    if requested == "cpu":
        return DeviceSpec(
            device="cpu",
            backend="cpu",
            torch_dtype_name="float32",
            allow_gpu=False,
            note="offline CPU default",
        )

    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]

    cuda_avail = bool(getattr(torch_module, "cuda", None) and torch_module.cuda.is_available())
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    mps_avail = bool(mps_backend is not None and mps_backend.is_available())
    try:
        import torch_directml  # type: ignore[import-not-found]
        dml_avail = bool(torch_directml.device())
    except ImportError:
        torch_directml = None
        dml_avail = False

    def _cuda_dtype() -> str:
        is_bf16 = getattr(torch_module.cuda, "is_bf16_supported", None)
        return "bfloat16" if (callable(is_bf16) and is_bf16()) else "float16"

    if requested == "cuda":
        if not cuda_avail:
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
        return DeviceSpec(device="cuda", backend="cuda", torch_dtype_name=_cuda_dtype(), allow_gpu=True)
    if requested == "mps":
        if not mps_avail:
            raise RuntimeError("MPS device requested but torch.backends.mps.is_available() is False")
        return DeviceSpec(device="mps", backend="mps", torch_dtype_name="float16", allow_gpu=True)
    if requested == "dml":
        if not dml_avail or torch_directml is None:
            raise RuntimeError("DirectML device requested but torch_directml is unavailable")
        return DeviceSpec(
            device=str(torch_directml.device()),
            backend="dml",
            torch_dtype_name="float32",
            allow_gpu=True,
        )

    # requested == "auto"
    if cuda_avail:
        return DeviceSpec(device="cuda", backend="cuda", torch_dtype_name=_cuda_dtype(), allow_gpu=True, note="auto-detected CUDA")
    if mps_avail:
        return DeviceSpec(device="mps", backend="mps", torch_dtype_name="float16", allow_gpu=True, note="auto-detected MPS")
    if dml_avail and torch_directml is not None:
        return DeviceSpec(
            device=str(torch_directml.device()),
            backend="dml",
            torch_dtype_name="float32",
            allow_gpu=True,
            note="auto-detected DirectML",
        )
    return DeviceSpec(device="cpu", backend="cpu", torch_dtype_name="float32", allow_gpu=False, note="auto-fallback CPU")


class RLVRVerifier(Protocol):
    """Protocol for deterministic verifiable-reward scorers in RLVR hooks."""

    def score(self, prompt: str, completion: str, gold: str) -> float:
        ...


@dataclass(frozen=True)
class ExactMatchVerifier:
    """Deterministic verifiable reward scorer checking exact (or stripped) label match."""

    reward_match: float = 1.0
    reward_miss: float = 0.0

    def score(self, prompt: str, completion: str, gold: str) -> float:
        del prompt
        cleaned = completion.strip().splitlines()[0].strip(".\"'` ") if completion.strip() else ""
        return self.reward_match if cleaned == gold else self.reward_miss


@dataclass
class SFTAdapterHook:
    """Standard LoRA supervised fine-tuning adapter hook."""

    method: str = "sft"
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGETS

    def lora_config_kwargs(self) -> dict[str, Any]:
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": list(self.target_modules),
        }

    def compute_loss(self, model: Any, batch: Mapping[str, Any]) -> Any:
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch["labels"],
        ).loss


@dataclass
class QLoRAAdapterHook(SFTAdapterHook):
    """4-bit QLoRA adapter hook (requires bitsandbytes + CUDA)."""

    method: str = "qlora"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    def ensure_supported(self, device_spec: DeviceSpec) -> None:
        if device_spec.backend != "cuda":
            raise RuntimeError(
                f"qlora requires a CUDA device (got backend={device_spec.backend!r}); "
                "refusing silent CPU fallback"
            )
        try:
            import bitsandbytes  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("qlora requires bitsandbytes installed alongside CUDA PyTorch") from exc

    def build_quantization_config(self, torch_module: Any) -> Any:
        from transformers import BitsAndBytesConfig

        compute_dtype = getattr(torch_module, "bfloat16", torch_module.float16)
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )


def compute_dpo_loss(
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    ref_chosen_logps: Any,
    ref_rejected_logps: Any,
    *,
    beta: float = 0.1,
    torch_module: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Compute Direct Preference Optimization (DPO) loss and chosen/rejected implicit rewards."""
    if beta <= 0:
        raise ValueError("DPO beta must be positive")
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    loss = -torch_module.nn.functional.logsigmoid(logits).mean()
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return loss, chosen_rewards, rejected_rewards


@dataclass
class DPOAdapterHook(SFTAdapterHook):
    """DPO preference-alignment training adapter hook."""

    method: str = "dpo"
    beta: float = 0.1

    def validate_preferences(self, records: Sequence[Mapping[str, Any]]) -> None:
        if not records:
            raise ValueError("dpo requires nonempty preference pair records")
        for idx, row in enumerate(records):
            if not all(isinstance(row.get(k), str) and row[k] for k in ("prompt", "chosen", "rejected")):
                raise ValueError(
                    f"dpo preference record at index {idx} must include nonempty 'prompt', 'chosen', and 'rejected' strings"
                )

    def loss(
        self,
        policy_chosen_logps: Any,
        policy_rejected_logps: Any,
        ref_chosen_logps: Any,
        ref_rejected_logps: Any,
    ) -> tuple[Any, Any, Any]:
        return compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=self.beta,
        )


@dataclass
class RLVRAdapterHook(SFTAdapterHook):
    """Reinforcement Learning with Verifiable Rewards (RLVR) adapter hook."""

    method: str = "rlvr"
    verifier: RLVRVerifier = field(default_factory=ExactMatchVerifier)
    kl_coef: float = 0.02

    def score_completions(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        golds: Sequence[str],
    ) -> list[float]:
        if not (len(prompts) == len(completions) == len(golds)):
            raise ValueError("prompts, completions, and golds must have equal length")
        return [float(self.verifier.score(p, c, g)) for p, c, g in zip(prompts, completions, golds, strict=True)]

    def compute_policy_gradient_loss(
        self,
        logps: Any,
        rewards: Any,
        ref_logps: Any | None = None,
        *,
        torch_module: Any | None = None,
    ) -> Any:
        if torch_module is None:
            import torch as torch_module  # type: ignore[no-redef]
        baseline = rewards.mean() if rewards.numel() > 1 else 0.0
        advantages = rewards - baseline
        pg_loss = -(advantages.detach() * logps).mean()
        if ref_logps is not None and self.kl_coef > 0.0:
            kl_penalty = (logps - ref_logps).mean()
            return pg_loss + self.kl_coef * kl_penalty
        return pg_loss


def get_training_hook(
    method: str = "sft",
    **kwargs: Any,
) -> SFTAdapterHook | QLoRAAdapterHook | DPOAdapterHook | RLVRAdapterHook:
    """Instantiate a typed training adapter hook for `sft`/`lora`, `qlora`, `dpo`, or `rlvr`."""
    normalized = method.lower()
    if normalized not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported training method {method!r}; expected one of {SUPPORTED_METHODS}")
    if normalized in ("sft", "lora"):
        return SFTAdapterHook(method=normalized, **kwargs)
    if normalized == "qlora":
        return QLoRAAdapterHook(**kwargs)
    if normalized == "dpo":
        return DPOAdapterHook(**kwargs)
    return RLVRAdapterHook(**kwargs)


def grad_accum_window_size(batch_idx: int, total_batches: int, grad_accum: int) -> int:
    """Return exact micro-batch count in the accumulation window containing `batch_idx`.

    Rescales trailing remainder windows (`total_batches % grad_accum != 0`) so
    every optimizer update averages over the actual micro-batches accumulated.
    """
    if grad_accum <= 0:
        raise ValueError("grad_accum must be positive")
    if total_batches <= 0 or batch_idx < 0 or batch_idx >= total_batches:
        raise ValueError("batch_idx out of bounds for total_batches")
    win_start = (batch_idx // grad_accum) * grad_accum
    return min(grad_accum, total_batches - win_start)


def should_step_grad_accum(batch_idx: int, total_batches: int, grad_accum: int) -> bool:
    """Return True strictly on completed `grad_accum` boundaries or the final batch."""
    if grad_accum <= 0:
        raise ValueError("grad_accum must be positive")
    if total_batches <= 0 or batch_idx < 0 or batch_idx >= total_batches:
        raise ValueError("batch_idx out of bounds for total_batches")
    return (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == total_batches


def iter_micro_batches(items: Sequence[T], batch_size: int) -> Iterator[list[T]]:
    """Yield contiguous micro-batches of up to `batch_size` items."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def collate_causal_lm_batch(
    examples: Sequence[tuple[Sequence[int], Sequence[int]] | Mapping[str, Sequence[int]]],
    *,
    pad_token_id: int,
    torch_module: Any | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Right-pad variable-length causal LM examples (`input_ids`, `labels`, `attention_mask`)."""
    if not examples:
        raise ValueError("cannot collate empty batch")
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]

    parsed_ids: list[list[int]] = []
    parsed_labels: list[list[int]] = []
    for item in examples:
        if isinstance(item, Mapping):
            ids = list(item["input_ids"])
            lbls = list(item["labels"])
        else:
            ids = list(item[0])
            lbls = list(item[1])
        if len(ids) != len(lbls):
            raise ValueError("input_ids and labels lengths must match")
        parsed_ids.append(ids)
        parsed_labels.append(lbls)

    max_len = max(len(x) for x in parsed_ids)
    padded_ids: list[list[int]] = []
    padded_labels: list[list[int]] = []
    attention_masks: list[list[int]] = []
    for ids, lbls in zip(parsed_ids, parsed_labels, strict=True):
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [pad_token_id] * pad_len)
        padded_labels.append(lbls + [-100] * pad_len)
        attention_masks.append([1] * len(ids) + [0] * pad_len)

    kwargs: dict[str, Any] = {"dtype": torch_module.long}
    if device is not None:
        kwargs["device"] = device
    return {
        "input_ids": torch_module.tensor(padded_ids, **kwargs),
        "labels": torch_module.tensor(padded_labels, **kwargs),
        "attention_mask": torch_module.tensor(attention_masks, **kwargs),
    }


def collate_left_padded_prompts(
    prompts: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    torch_module: Any | None = None,
    device: str | None = None,
) -> tuple[Any, Any]:
    """Left-pad prompt token sequences for causal LM batched greedy generation."""
    if not prompts:
        raise ValueError("cannot collate empty prompt batch")
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]

    max_len = max(len(p) for p in prompts)
    padded: list[list[int]] = []
    masks: list[list[int]] = []
    for p in prompts:
        seq = list(p)
        pad_len = max_len - len(seq)
        padded.append([pad_token_id] * pad_len + seq)
        masks.append([0] * pad_len + [1] * len(seq))

    kwargs: dict[str, Any] = {"dtype": torch_module.long}
    if device is not None:
        kwargs["device"] = device
    return (
        torch_module.tensor(padded, **kwargs),
        torch_module.tensor(masks, **kwargs),
    )


class LabelPrefixTrie:
    """Token-level prefix trie restricting causal generation to an exact label allowlist."""

    def __init__(
        self,
        labels: Sequence[str],
        tokenizer: Any,
        *,
        eos_token_id: int | None = None,
    ) -> None:
        if not labels:
            raise ValueError("labels allowlist must be nonempty")
        resolved_eos = eos_token_id if eos_token_id is not None else getattr(tokenizer, "eos_token_id", None)
        if resolved_eos is None:
            raise ValueError("eos_token_id is required to terminate trie paths")
        self.eos_token_id: int = int(resolved_eos)
        self.transitions: dict[tuple[int, ...], set[int]] = {}
        self.sequences: dict[str, list[int]] = {}
        for label in labels:
            token_ids = [int(t) for t in tokenizer.encode(label, add_special_tokens=False)]
            if not token_ids:
                raise ValueError(f"label {label!r} tokenized to empty sequence")
            self.sequences[label] = token_ids
            full_seq = token_ids + [self.eos_token_id]
            for pos in range(len(full_seq)):
                prefix = tuple(full_seq[:pos])
                self.transitions.setdefault(prefix, set()).add(full_seq[pos])

    def allowed_next_tokens(self, generated_prefix: Sequence[int]) -> list[int]:
        """Return valid next token IDs given generated tokens so far."""
        if self.eos_token_id in generated_prefix:
            return [self.eos_token_id]
        allowed = self.transitions.get(tuple(int(t) for t in generated_prefix))
        if not allowed:
            return [self.eos_token_id]
        return sorted(allowed)


class PrefixTrieLogitsProcessor:
    """Logits processor compatible with Hugging Face `generate(..., logits_processor=...)`."""

    def __init__(self, trie: LabelPrefixTrie, prompt_length: int) -> None:
        self.trie = trie
        self.prompt_length = prompt_length

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        masked = torch.full_like(scores, float("-inf"))
        batch_size = int(input_ids.shape[0])
        for row_idx in range(batch_size):
            suffix = input_ids[row_idx, self.prompt_length :].tolist()
            allowed = self.trie.allowed_next_tokens(suffix)
            idx_tensor = torch.tensor(allowed, dtype=torch.long, device=scores.device)
            masked[row_idx, idx_tensor] = scores[row_idx, idx_tensor]
        return masked


def build_prefix_trie_logits_processor(
    labels: Sequence[str],
    tokenizer: Any,
    prompt_length: int,
    *,
    eos_token_id: int | None = None,
) -> Any:
    """Build a Hugging Face `LogitsProcessorList` wrapping `PrefixTrieLogitsProcessor`."""
    trie = LabelPrefixTrie(labels, tokenizer, eos_token_id=eos_token_id)
    processor = PrefixTrieLogitsProcessor(trie, prompt_length=prompt_length)
    try:
        from transformers import LogitsProcessorList

        return LogitsProcessorList([processor])
    except ImportError:
        return [processor]


def percentile(values: Sequence[float], pct: float) -> float | None:
    """Compute linear-interpolated percentile in `[0, 100]` (None when empty)."""
    if not values:
        return None
    if not (0.0 <= pct <= 100.0):
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _extract_sequence_confidence(
    scores: Sequence[Any] | None,
    row_idx: int,
    active_tokens: Sequence[int],
    torch_module: Any,
) -> float | None:
    if not scores or not active_tokens:
        return None
    log_prob_sum = 0.0
    steps = min(len(scores), len(active_tokens))
    if steps == 0:
        return None
    for step_idx in range(steps):
        step_logits = scores[step_idx][row_idx]
        log_probs = torch_module.nn.functional.log_softmax(step_logits.float(), dim=-1)
        tok = int(active_tokens[step_idx])
        log_prob_sum += float(log_probs[tok].item())
    prob = math.exp(log_prob_sum)
    if not math.isfinite(prob):
        return None
    return max(0.0, min(1.0, prob))


def batched_greedy_generate(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Sequence[int]],
    *,
    max_new_tokens: int = 12,
    batch_size: int = 8,
    pad_token_id: int | None = None,
    eos_token_id: int | None = None,
    label_trie: LabelPrefixTrie | None = None,
    return_confidence: bool = True,
    torch_module: Any | None = None,
    device: str | None = None,
) -> list[GenerationResult]:
    """Run left-padded batched greedy decoding under `torch.inference_mode()`."""
    if not prompts:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]

    resolved_eos = eos_token_id if eos_token_id is not None else getattr(tokenizer, "eos_token_id", None)
    resolved_pad = pad_token_id
    if resolved_pad is None:
        tok_pad = getattr(tokenizer, "pad_token_id", None)
        resolved_pad = tok_pad if tok_pad is not None else resolved_eos
    if resolved_pad is None:
        resolved_pad = 0

    if device is None:
        try:
            first_param = next(model.parameters())
            device = str(first_param.device)
        except (AttributeError, StopIteration, TypeError):
            device = "cpu"

    if hasattr(model, "eval"):
        model.eval()
    results: list[GenerationResult] = []
    with torch_module.inference_mode():
        for batch_prompts in iter_micro_batches(list(prompts), batch_size):
            t_start = time.perf_counter()
            input_ids, attention_mask = collate_left_padded_prompts(
                batch_prompts,
                pad_token_id=int(resolved_pad),
                torch_module=torch_module,
                device=device,
            )
            prompt_len = int(input_ids.shape[1])
            gen_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "num_beams": 1,
                "pad_token_id": int(resolved_pad),
            }
            if resolved_eos is not None:
                gen_kwargs["eos_token_id"] = int(resolved_eos)
            if label_trie is not None:
                processor = PrefixTrieLogitsProcessor(label_trie, prompt_length=prompt_len)
                try:
                    from transformers import LogitsProcessorList

                    gen_kwargs["logits_processor"] = LogitsProcessorList([processor])
                except ImportError:
                    gen_kwargs["logits_processor"] = [processor]
            if return_confidence:
                gen_kwargs["output_scores"] = True
                gen_kwargs["return_dict_in_generate"] = True

            raw_out = model.generate(**gen_kwargs)
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            per_item_ms = elapsed_ms / len(batch_prompts)

            if hasattr(raw_out, "sequences"):
                sequences = raw_out.sequences
                scores = getattr(raw_out, "scores", None)
            else:
                sequences = raw_out
                scores = None

            for row_idx in range(len(batch_prompts)):
                gen_slice = [int(t) for t in sequences[row_idx, prompt_len:].tolist()]
                if resolved_eos is not None and int(resolved_eos) in gen_slice:
                    eos_pos = gen_slice.index(int(resolved_eos))
                    active_tokens = gen_slice[:eos_pos]
                    limit_hit = False
                else:
                    active_tokens = gen_slice
                    limit_hit = len(gen_slice) >= max_new_tokens
                decoded = str(tokenizer.decode(active_tokens, skip_special_tokens=True)).strip()
                conf = _extract_sequence_confidence(scores, row_idx, active_tokens, torch_module)
                results.append(
                    GenerationResult(
                        text=decoded,
                        token_ids=active_tokens,
                        max_token_limit_hit=limit_hit,
                        confidence=conf,
                        latency_ms=round(per_item_ms, 3),
                    )
                )
    return results
