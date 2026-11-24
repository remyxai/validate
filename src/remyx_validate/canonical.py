"""Canonical JSON and the identities derived from it.

Canonical form: UTF-8 JSON, keys sorted, no insignificant whitespace, floats
that hold an integral value written as integers, NaN/Infinity rejected. Two
documents that differ only in key order, YAML style or `1` vs `1.0` hash the
same. Keys starting with `x-` are extension fields: preserved in files,
ignored by policy, and left out of every hash.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import time
from typing import Any


def normalize(value: Any, drop_extensions: bool = True) -> Any:
    if isinstance(value, dict):
        out = {}
        for k in sorted(value, key=str):
            if drop_extensions and isinstance(k, str) and k.startswith("x-"):
                continue
            out[str(k)] = normalize(value[k], drop_extensions)
        return out
    if isinstance(value, (list, tuple)):
        return [normalize(v, drop_extensions) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("NaN and Infinity have no canonical form")
        if value.is_integer() and abs(value) < 2 ** 53:
            return int(value)
        return value
    raise TypeError(f"cannot canonicalize {type(value).__name__}")


def canonical_json(value: Any, drop_extensions: bool = True) -> str:
    return json.dumps(normalize(value, drop_extensions), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(root: str) -> dict:
    """Digest of a directory: every regular file's relative path and sha256.

    Symlinks are recorded by target string, never followed, so a link that
    points outside the tree cannot pull outside content into the digest.
    """
    files = {}
    root = os.path.abspath(root)
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".git"))
        for name in sorted(names):
            full = os.path.join(base, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if name.endswith(".pyc"):
                continue
            if os.path.islink(full):
                files[rel] = "symlink:" + os.readlink(full)
            elif os.path.isfile(full):
                files[rel] = sha256_file(full)
    return {"files": files, "digest": digest(files)}


def new_run_id() -> str:
    return "run_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + secrets.token_hex(3)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
