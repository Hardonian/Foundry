"""Single-pass canonical JSON serializer and SHA-256 digest helpers."""

from __future__ import annotations

import hashlib
import json.encoder
import math
from typing import Any

_encode_basestring_ascii = json.encoder.encode_basestring_ascii

_OP_VALUE = 0
_OP_LITERAL = 1
_OP_EXIT = 2


def _validate_and_encode_str(text: str, *, context: str = "string") -> str:
    raw = text if type(text) is str else str(text)
    try:
        raw.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{context} must not contain lone UTF-16 surrogates") from exc
    return _encode_basestring_ascii(raw)


def canonical(value: Any) -> str:
    """Serialize value as deterministic single-pass canonical JSON.

    Rejects non-string dict keys, circular references, NaN/Inf floats,
    lone UTF-16 surrogates, and non-JSON-serializable types. Normalizes -0.0 to 0.0.
    """
    out: list[str] = []
    active: set[int] = set()
    stack: list[tuple[int, Any]] = [(_OP_VALUE, value)]

    while stack:
        op, item = stack.pop()
        if op == _OP_LITERAL:
            out.append(item)
            continue
        if op == _OP_EXIT:
            active.remove(item)
            continue

        if item is None:
            out.append("null")
        elif isinstance(item, bool):
            out.append("true" if item else "false")
        elif isinstance(item, int):
            out.append(str(int(item)))
        elif isinstance(item, float):
            number = float(item)
            if not math.isfinite(number):
                raise ValueError("Out of range float values are not JSON compliant")
            if number == 0.0:
                number = 0.0
            out.append(repr(number))
        elif isinstance(item, str):
            out.append(_validate_and_encode_str(item))
        elif isinstance(item, (list, tuple)):
            if not item:
                out.append("[]")
                continue
            obj_id = id(item)
            if obj_id in active:
                raise ValueError("Circular reference detected")
            active.add(obj_id)
            out.append("[")
            stack.append((_OP_EXIT, obj_id))
            stack.append((_OP_LITERAL, "]"))
            for idx in range(len(item) - 1, -1, -1):
                stack.append((_OP_VALUE, item[idx]))
                if idx > 0:
                    stack.append((_OP_LITERAL, ","))
        elif isinstance(item, dict):
            if not item:
                out.append("{}")
                continue
            obj_id = id(item)
            if obj_id in active:
                raise ValueError("Circular reference detected")
            encoded_items: list[tuple[str, str, Any]] = []
            for key, val in item.items():
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings")  # noqa: TRY004
                key_raw = key if type(key) is str else str(key)
                encoded_key = _validate_and_encode_str(key_raw, context="dict key")
                encoded_items.append((key_raw, encoded_key, val))
            encoded_items.sort(key=lambda entry: entry[0])
            active.add(obj_id)
            out.append("{")
            stack.append((_OP_EXIT, obj_id))
            stack.append((_OP_LITERAL, "}"))
            for idx in range(len(encoded_items) - 1, -1, -1):
                _, encoded_key, val = encoded_items[idx]
                stack.append((_OP_VALUE, val))
                stack.append((_OP_LITERAL, f"{encoded_key}:"))
                if idx > 0:
                    stack.append((_OP_LITERAL, ","))
        else:
            raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    return "".join(out)


def canonical_bytes(value: Any) -> bytes:
    """Return UTF-8 encoded canonical JSON bytes."""
    return canonical(value).encode("utf-8")


def digest(value: Any) -> str:
    """Return lowercase SHA-256 hex digest of canonical JSON representation."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()
