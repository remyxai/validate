"""Resolve a declaration into an immutable plan, and gate execution on an
approval of that plan's digest. Anything that changes the plan (a moved
branch, an edited harness file, a new threshold) changes the digest and the
earlier approval no longer applies."""
from __future__ import annotations

import getpass
import json
import os
import shutil
import subprocess

from . import SPEC_VERSION, __version__, declaration as decl, gitops
from .backends.local import BASE_ENV as LOCAL_BASE_ENV
from .canonical import digest, now_iso, sha256_file, tree_digest


class PlanError(Exception):
    pass


class NotApproved(PlanError):
    def __init__(self, message, plan, drift=None):
        super().__init__(message)
        self.plan = plan
        self.drift = drift or []


def state_dir(project: str) -> str:
    return os.path.join(project, ".validate")


def harness_dir(d: dict, project: str):
    h = d.get("harness")
    if not h:
        return None
    declared = os.path.normpath(os.path.join(project, h["path"]))
    path, root = os.path.realpath(declared), os.path.realpath(project)
    if os.path.islink(declared) or os.path.commonpath([path, root]) != root or path == root:
        raise PlanError(f"harness path must be a real directory inside the project: {h['path']}")
    if not os.path.isdir(path):
        raise PlanError(f"harness directory not found: {h['path']}")
    for base, dirs, names in os.walk(path, followlinks=False):
        for name in dirs + names:
            full = os.path.join(base, name)
            if os.path.islink(full):
                raise PlanError("the harness contains a symbolic link "
                                f"({os.path.relpath(full, path)}); links are not allowed because "
                                "the approved digest would not cover what they point to")
            if not (os.path.isdir(full) or os.path.isfile(full)):
                raise PlanError(f"the harness contains a non-regular file: {os.path.relpath(full, path)}")
    return path


def image_id(image: str):
    """The local content id of a docker image, or None if it is not present.
    A tag can move; the id is what ran."""
    if not shutil.which("docker"):
        return None
    proc = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                          capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else None


def build(decl_path: str) -> dict:
    project = decl.project_dir(decl_path)
    d = decl.load(decl_path)
    mirror = gitops.ensure_mirror(d["change"]["repo"], project)
    shas = {arm: gitops.resolve_ref(mirror, d["change"][arm]["ref"])
            for arm in ("baseline", "candidate")}
    hdir = harness_dir(d, project)
    harness = None
    if hdir:
        tree = tree_digest(hdir)
        harness = {"path": d["harness"]["path"], "digest": tree["digest"],
                   "file_count": len(tree["files"])}
    identity = gitops.repo_identity(d["change"]["repo"], project)
    ex = d["execution"]
    docker = ex["backend"] == "docker"
    img_id = image_id(ex["image"]) if docker else None
    unpinned = ["Packages installed by `setup` are resolved at build time; the tool does not "
                "lock them. Pin versions in the setup commands or a lockfile for a repeatable "
                "environment."] if d["setup"] else []
    if docker and not img_id:
        unpinned.append(f"Image `{ex['image']}` is not present locally, so its content id is "
                        "unknown until it is pulled. A tag can point to different content later.")
    warnings = []
    if docker:
        warnings.append("The docker backend is experimental. It is container isolation, not a "
                        "security boundary for hostile code.")
    if ex["trust"] == "untrusted" and ex["network"] != "none":
        warnings.append("Untrusted code has network access by explicit acknowledgement.")
    if ex["pass_env"]:
        warnings.append("Host variables are passed to the workload: " + ", ".join(ex["pass_env"]))
    if "max_cost_usd" in d["budget"]:
        warnings.append("`max_cost_usd` is checked after each attempt against the cost the "
                        "harness reports. It is not a hard spending limit.")
    seeds = d["method"]["seeds"]
    tries = 1 + d["run"]["retries"]
    plan = {
        "spec_version": SPEC_VERSION,
        "kind": "plan",
        "runner_version": __version__,
        "declaration_file": os.path.basename(decl_path),
        "declaration_hash": decl.declaration_hash(d),
        "spec_hash": decl.spec_hash(d, shas, harness and harness["digest"]),
        "change": {
            "change_id": decl.change_id(identity, shas, d["arms"]),
            "repo": identity,
            "baseline": {"ref": d["change"]["baseline"]["ref"], "sha": shas["baseline"]},
            "candidate": {"ref": d["change"]["candidate"]["ref"], "sha": shas["candidate"]},
        },
        "env_keys": {arm: decl.env_key(d, arm, shas[arm], img_id) for arm in shas},
        "image_id": img_id,
        "access": _access(d, harness is not None),
        "warnings": warnings,
        "reproducibility": unpinned,
        "harness": harness,
        "declaration": d,
        "scope": {
            "attempts_min": 2 * len(seeds),
            "attempts_max": 2 * len(seeds) * tries,
            "per_attempt_timeout_s": d["run"]["timeout_s"],
            "wall_cap_s": d["budget"]["max_wall_s"],
            "cost_cap_usd": d["budget"].get("max_cost_usd"),
            "cost_estimate_usd": None,
            "cost_note": "unknown: the runner cannot price your compute or provider "
                         "calls. A cost cap is a threshold on what the harness reports "
                         "after each attempt, not a spending limit.",
            "wall_note": "the wall cap covers setup, preflight and attempts. Fetching the "
                         "repository while planning is outside it.",
            "network": _network_note(d),
            "secrets": ("none passed" if not d["execution"]["pass_env"]
                        else "host variables passed to the workload: "
                             + ", ".join(d["execution"]["pass_env"])),
            "outputs": ".validate/runs/<run_id>/ under the project directory; nothing is uploaded",
        },
    }
    plan["plan_digest"] = digest(plan)
    return plan


def _access(d: dict, has_harness: bool) -> dict:
    """What the workload can reach, by name. Values of variables never appear."""
    ex = d["execution"]
    names = sorted({k for arm in d["arms"].values() for k in arm["env"]})
    if ex["backend"] == "local":
        return {"filesystem": "everything your user can read and write",
                "network": {"setup": "unrestricted", "run": "unrestricted"},
                "env_names": names, "host_env_names": list(LOCAL_BASE_ENV) + list(ex["pass_env"])}
    net = ex["network"]
    return {
        "filesystem": {
            "checkout": "read-write in setup, read-only in run",
            "environment dir": "read-write in setup, read-only in run",
            "harness": "read-only" if has_harness else "none",
            "output dir": "read-write",
            "home, host env, docker daemon": "not mounted",
        },
        "network": {"setup": "on" if net in ("setup-only", "full") else "off",
                    "run": "on" if net == "full" else "off"},
        "image": ex["image"], "limits": ex["limits"],
        "env_names": names, "host_env_names": list(ex["pass_env"]),
    }


def _network_note(d: dict) -> str:
    ex = d["execution"]
    if ex["backend"] == "local":
        return "local subprocess: unrestricted (trusted code only)"
    return {"none": "docker: no network in setup or run",
            "setup-only": "docker: network during setup, none during the run",
            "full": "docker: network in setup and run"}[ex["network"]]


def save(project: str, plan: dict) -> str:
    path = os.path.join(state_dir(project), "plans", plan["plan_digest"] + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2, sort_keys=True)
    return path


def load_saved(project: str, plan_digest: str) -> dict:
    with open(os.path.join(state_dir(project), "plans", plan_digest + ".json"), encoding="utf-8") as fh:
        return json.load(fh)


def find_saved(project: str, prefix: str) -> str:
    folder = os.path.join(state_dir(project), "plans")
    names = [n[:-5] for n in os.listdir(folder)] if os.path.isdir(folder) else []
    hits = [n for n in names if n.startswith(prefix)]
    if len(prefix) < 8:
        raise PlanError("give at least 8 characters of the plan digest")
    if not hits:
        raise PlanError(f"no saved plan matches {prefix!r}; run `plan` first")
    if len(hits) > 1:
        raise PlanError(f"{prefix!r} matches {len(hits)} plans; give more characters")
    return hits[0]


def approve(project: str, plan_digest: str, approver: str = "") -> dict:
    plan = load_saved(project, plan_digest)
    if digest({k: v for k, v in plan.items() if k != "plan_digest"}) != plan_digest:
        raise PlanError("saved plan does not match its digest; re-run `plan`")
    record = {
        "plan_digest": plan_digest,
        "declaration_hash": plan["declaration_hash"],
        "change_id": plan["change"]["change_id"],
        "approved_by": approver or default_author(),
        "approved_at": now_iso(),
    }
    path = os.path.join(state_dir(project), "approvals", plan_digest + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
    return record


def approval(project: str, plan_digest: str):
    path = os.path.join(state_dir(project), "approvals", plan_digest + ".json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def latest_approved_plan(project: str, declaration_file: str = ""):
    folder = os.path.join(state_dir(project), "approvals")
    if not os.path.isdir(folder):
        return None
    records = []
    for name in os.listdir(folder):
        with open(os.path.join(folder, name), encoding="utf-8") as fh:
            records.append(json.load(fh))
    for rec in sorted(records, key=lambda r: r["approved_at"], reverse=True):
        try:
            plan = load_saved(project, rec["plan_digest"])
        except OSError:
            continue
        if not declaration_file or plan.get("declaration_file") == declaration_file:
            return plan
    return None


def require_approved(project: str, plan: dict) -> dict:
    """Fail closed unless this exact plan digest was approved."""
    rec = approval(project, plan["plan_digest"])
    if rec:
        return rec
    prior = latest_approved_plan(project, plan["declaration_file"])
    drift = diff(prior, plan) if prior else []
    if prior:
        msg = ("the plan changed after approval; the approval for "
               f"{prior['plan_digest'][:12]} does not cover {plan['plan_digest'][:12]}")
    else:
        msg = f"plan {plan['plan_digest'][:12]} has not been approved"
    raise NotApproved(msg, plan, drift)


def default_author() -> str:
    name = gitops.git(["config", "--get", "user.name"], check=False)
    email = gitops.git(["config", "--get", "user.email"], check=False)
    if name and email:
        return f"{name} <{email}>"
    return name or getpass.getuser()


def _flatten(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(value, list) and value and all(isinstance(v, dict) and "id" in v for v in value):
        for v in value:
            out.update(_flatten(v, f"{prefix}[{v['id']}]"))
    else:
        out[prefix] = value
    return out


def diff(old: dict, new: dict) -> list:
    """Field-level changes between two plans, ignoring derived digests."""
    skip = ("plan_digest",)
    a, b = _flatten(old or {}), _flatten(new or {})
    rows = []
    for key in sorted(set(a) | set(b)):
        if key in skip:
            continue
        if a.get(key) != b.get(key):
            rows.append({"field": key, "approved": a.get(key), "current": b.get(key)})
    return rows
