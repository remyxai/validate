"""Runner-owned cache isolation for measured attempts.

Before every measured attempt the runner creates a new, empty directory
inside that attempt's folder and points the mutable caches it knows about at
it. After the attempt the directory is deleted. Nothing outside a directory
the runner created for the attempt is ever removed: no home directory, no
shared model cache, no system cache, no sudo.

What this controls: the listed cache locations, for processes that honour
the variables. What it does not control, and never claims to: the OS page
cache, CPU and GPU thermal and frequency state, GPU driver and device
caches, remote services and provider-side caches, and other load on the
machine. Those are listed in every report as not controlled.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess

# Variable -> subdirectory. Each is a mutable cache or scratch location whose
# contents can make a later attempt faster than an earlier one.
LAYERS = {
    "XDG_CACHE_HOME": "xdg",
    "TMPDIR": "tmp",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TRITON_CACHE_DIR": "triton",
    "NUMBA_CACHE_DIR": "numba",
    "CUDA_CACHE_PATH": "cuda",
    "MPLCONFIGDIR": "matplotlib",
}
RESERVED_SUBDIRS = set(LAYERS.values())

NOT_CONTROLLED = [
    "OS page cache and filesystem buffers",
    "CPU and GPU thermal, frequency and power state",
    "GPU driver, device and kernel-level caches",
    "remote services and provider-side caches",
    "other load on the same machine",
]


class IsolationError(Exception):
    pass


def layers(measurement: dict) -> dict:
    """Variable -> subdirectory for this declaration, including the
    application cache variables the declaration registers."""
    out = dict(LAYERS)
    for name in measurement["cache"]["isolate"]:
        out.setdefault(name, "app-" + name.lower())
    return out


def _inside(path: str, owner: str) -> bool:
    real, root = os.path.realpath(path), os.path.realpath(owner)
    return real != root and os.path.commonpath([real, root]) == root


def prepare(root: str, owner: str, measurement: dict) -> dict:
    """Create the attempt's cache directory, new and empty. Returns
    {variable: host path}. Refuses a path that already exists, is a link, or
    resolves outside the directory the runner owns."""
    parent = os.path.dirname(root)
    os.makedirs(parent, exist_ok=True)
    if os.path.lexists(root):
        raise IsolationError(f"cache directory already exists: {os.path.basename(root)}")
    if not _inside(parent, owner) and os.path.realpath(parent) != os.path.realpath(owner):
        raise IsolationError("cache directory would be outside the run directory")
    os.mkdir(root, 0o700)
    out = {}
    for name, sub in sorted(layers(measurement).items()):
        path = os.path.join(root, sub)
        os.mkdir(path, 0o700)
        out[name] = path
    return out


def is_clean(root: str, measurement: dict) -> bool:
    """True when the directory holds exactly the empty layer directories."""
    want = sorted(layers(measurement).values())
    try:
        if os.path.islink(root) or sorted(os.listdir(root)) != want:
            return False
        return all(not os.path.islink(os.path.join(root, sub)) and
                   not os.listdir(os.path.join(root, sub)) for sub in want)
    except OSError:
        return False


def remove(root: str, owner: str) -> str:
    """Delete an attempt's cache directory. Returns "removed", "absent" or a
    reason it was left alone. Links are never followed: a link at the root is
    refused, and links inside are unlinked, not traversed."""
    if not os.path.lexists(root):
        return "absent"
    if os.path.islink(root):
        return "refused: the cache directory is a link"
    if not _inside(root, owner):
        return "refused: outside the run directory"

    def writable(func, path, _exc):
        try:
            os.chmod(os.path.dirname(path), 0o700)
            if not os.path.islink(path):
                os.chmod(path, 0o700)
            func(path)
        except OSError:
            pass

    try:
        shutil.rmtree(root, onexc=writable)
    except TypeError:   # Python before 3.12
        shutil.rmtree(root, onerror=writable)
    return "removed" if not os.path.lexists(root) else "failed: some files could not be deleted"


def shared_assets(environ=None) -> dict:
    """Where immutable, content-addressed downloads already live on this
    host. Moving XDG_CACHE_HOME would otherwise relocate them and force a
    download inside the timed attempt."""
    env = os.environ if environ is None else environ
    cache = env.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return {"HF_HOME": env.get("HF_HOME") or os.path.join(cache, "huggingface"),
            "TORCH_HOME": env.get("TORCH_HOME") or os.path.join(cache, "torch")}


# ── what the machine looked like ──────────────────────────────────────────

def _nvidia(query: str) -> list:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        proc = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return [[cell.strip() for cell in line.split(",")] for line in proc.stdout.splitlines() if line.strip()]


def host_fingerprint() -> dict:
    """The machine the attempts ran on, as far as it can be read without
    privileges. No hostname, user name or address. Self-reported like the
    rest of the bundle."""
    model = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        model = platform.processor()
    try:
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        memory = None
    return {"system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "cpu_model": model, "cpu_count": os.cpu_count(),
            "memory_bytes": memory, "python": platform.python_version(),
            "gpus": [{"name": row[0], "driver": row[1]} for row in _nvidia("name,driver_version")
                     if len(row) == 2]}


def conditions() -> dict:
    """A reading taken just before and just after a measured attempt. Load
    is everything on the machine, including other tenants. Unreadable values
    are left out, never guessed."""
    out = {}
    try:
        out["load_1m"] = round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):
        pass
    gpus = []
    for row in _nvidia("temperature.gpu,utilization.gpu"):
        try:
            gpus.append({"temperature_c": float(row[0]), "utilization_pct": float(row[1])})
        except (ValueError, IndexError):
            continue
    if gpus:
        out["gpus"] = gpus
    return out
