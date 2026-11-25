"""Load, validate and normalize a declaration (impact.yaml), and derive the
identities that depend on it.

Three hashes with three different jobs:

* ``declaration_hash``: what is being asked. Objective, criteria, method,
  validity checks and data scope. Editing a threshold changes it.
* ``env_key`` (per arm): what has to be built. Commit, setup commands,
  backend and image. Editing a threshold does not change it, so the built
  environment is reused.
* ``spec_hash``: what gets executed. Resolved commits, harness digest,
  command, arm environments, seeds and execution settings.
"""
from __future__ import annotations

import copy
import json
import os
import re
from importlib import resources
from typing import Any

import jsonschema
import yaml

from . import SPEC_VERSION, schedule
from .canonical import digest

PLACEHOLDER = "TODO"

DEFAULTS = {
    "method": {"unit": "example", "seeds": [0], "confidence": 0.95,
               "uncertainty": "interval", "max_missing_rate": 0.0},
    "run": {"timeout_s": 1800, "retries": 0},
    "execution": {"backend": "local", "trust": "trusted", "image": "python:3.12-slim",
                  "network": None, "pass_env": [], "limits": {}},
    "budget": {"max_wall_s": 14400},
    "validity": {"activation": [], "held_constant": [], "baseline_expect": []},
}

# Applied only when a declaration has a `measurement` block. A declaration
# without one keeps exactly the document, and the hashes, it had before the
# protocol existed.
MEASUREMENT_DEFAULTS = {"design": "crossover", "replicates": 1, "schedule_seed": 0,
                        "reset_hooks": []}
CACHE_DEFAULTS = {"profile": "cold", "assets": "shared", "isolate": [], "uncontrolled": []}

# A criterion that does not say whether it is a timing is read by its metric
# name and unit. Timing criteria get the measurement protocol by default, so
# the safe behaviour is the one you get without asking. `performance: false`
# on the criterion opts out, in writing.
TIMING_NAME = re.compile(
    r"(latenc|throughput|duration|elapsed|runtime|wall|ttft|tpot|qps|rps|fps|speed"
    r"|(^|_)time($|_)|_(ms|us|ns|s|sec|secs|seconds)$|per_(s|sec|second|min|minute)$)", re.I)
TIMING_UNIT = re.compile(r"^(ms|us|µs|ns|s|sec|secs|seconds|min|minutes|h|hours)$|/(s|sec|second|min)$", re.I)
INFERRED = "x-performance-inferred"


def looks_like_timing(criterion: dict) -> bool:
    return bool(TIMING_NAME.search(criterion.get("metric", ""))
                or TIMING_UNIT.search(criterion.get("unit", "") or ""))


class DeclarationError(Exception):
    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def load_schema(name: str) -> dict:
    text = resources.files("remyx_validate").joinpath("schemas", f"{name}.schema.json").read_text("utf-8")
    return json.loads(text)


def schema_problems(doc: Any, schema_name: str) -> list:
    validator = jsonschema.Draft202012Validator(load_schema(schema_name))
    out = []
    for err in sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path)):
        where = "/".join(str(p) for p in err.absolute_path) or "(root)"
        out.append(f"{where}: {err.message}")
    return out


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML silently keeps the last of two identical keys. A declaration with
    two `max:` lines is ambiguous about what was approved, so reject it."""


def _unique_mapping(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.YAMLError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
        seen.add(key)
    return loader.construct_mapping(node, deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)

CREDENTIAL_SHAPES = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"), re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
]
URL_WITH_CREDENTIALS = re.compile(r"^[a-z][a-z0-9+.-]*://[^/@\s]+:[^/@\s]+@", re.I)
ALLOWED_REPO_SCHEMES = ("https://", "ssh://", "git://", "file://", "http://")


def read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        try:
            data = yaml.load(fh, Loader=_UniqueKeyLoader)
        except yaml.YAMLError as exc:
            raise DeclarationError([f"{path}: not valid YAML: {exc}"])
    if not isinstance(data, dict):
        raise DeclarationError([f"{path}: expected a mapping at the top level"])
    return data


def normalize(raw: dict) -> dict:
    """Apply defaults so equivalent declarations become identical documents."""
    d = copy.deepcopy(raw)
    for section, defaults in DEFAULTS.items():
        block = d.setdefault(section, {})
        for key, value in defaults.items():
            block.setdefault(key, copy.deepcopy(value))
    d.setdefault("setup", [])
    arms = d.setdefault("arms", {})
    for arm in ("baseline", "candidate"):
        env = arms.setdefault(arm, {}).setdefault("env", {})
        arms[arm]["env"] = {k: _env_str(v) for k, v in env.items()}
    d["method"]["seeds"] = sorted(d["method"]["seeds"])
    ex = d["execution"]
    if ex["network"] is None:
        if ex["backend"] != "docker":
            ex["network"] = "full"
        else:
            ex["network"] = "none" if ex["trust"] == "untrusted" else "setup-only"
    ex.setdefault("acknowledge_untrusted_network", False)
    for c in d.get("criteria", []):
        if "performance" not in c and looks_like_timing(c):
            c["performance"] = True
            c[INFERRED] = True
    if "measurement" not in d and any(c.get("performance") for c in d.get("criteria", [])):
        d["measurement"] = {}
    if "measurement" in d:
        m = d["measurement"] = dict(d["measurement"] or {})
        for key, value in MEASUREMENT_DEFAULTS.items():
            m.setdefault(key, copy.deepcopy(value))
        cache = m["cache"] = dict(m.get("cache") or {})
        for key, value in CACHE_DEFAULTS.items():
            cache.setdefault(key, copy.deepcopy(value))
        if cache["profile"] == "warm":
            cache.setdefault("warmup_runs", 1)
        cache["isolate"] = sorted(cache["isolate"])
        for hook in m["reset_hooks"]:
            hook.setdefault("timeout_s", 60)
    for c in d.get("criteria", []):
        c.setdefault("aggregate", "mean")
        c.setdefault("relative", False)
        if c["role"] == "target":
            c.setdefault("min_improvement", 0)
    return d


def _env_str(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


def semantic_problems(d: dict) -> list:
    """Rules a JSON Schema cannot state."""
    out = []
    ids = [c["id"] for c in d["criteria"]]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        out.append(f"criteria: duplicate id {dup!r}")
    if not any(c["role"] in ("target", "guardrail") for c in d["criteria"]):
        out.append("criteria: declare at least one target or guardrail; "
                   "info criteria never produce a verdict")
    for c in d["criteria"]:
        cid = c["id"]
        if PLACEHOLDER in c["metric"] or PLACEHOLDER in cid:
            out.append(f"criteria/{cid}: still a template placeholder; name the "
                       "metric your harness reports")
        if c["role"] == "guardrail" and not any(k in c for k in ("max", "min", "max_regression")):
            out.append(f"criteria/{cid}: a guardrail needs `max`, `min` or `max_regression`")
        if c["role"] != "guardrail":
            for key in ("max", "min", "max_regression"):
                if key in c:
                    out.append(f"criteria/{cid}: `{key}` only applies to a guardrail; on a "
                               f"{c['role']} it would be ignored")
        if c["role"] != "target" and "min_improvement" in c:
            out.append(f"criteria/{cid}: `min_improvement` belongs on a target")
        if "max" in c and "min" in c and c["min"] > c["max"]:
            out.append(f"criteria/{cid}: `min` {c['min']} is above `max` {c['max']}; no value can satisfy both")
    if PLACEHOLDER in d["run"]["command"]:
        out.append("run/command: still a template placeholder")
    out += _measurement_problems(d)
    ex = d["execution"]
    if ex["trust"] == "review-required":
        out.append("execution/trust: you have not chosen how to run this code. Read the change "
                   "and the setup commands first. Then set `trust: trusted` with "
                   "`backend: local` for code you have reviewed, or `trust: untrusted` with "
                   "`backend: docker` (experimental, and not a security boundary for hostile code)")
    if ex["trust"] == "untrusted" and ex["backend"] != "docker":
        out.append("execution: untrusted code requires `backend: docker`; the "
                   "local backend runs with your user's file access")
    if ex["trust"] == "untrusted":
        if ex["pass_env"]:
            out.append("execution/pass_env: host variables cannot be passed to untrusted code")
        if ex["network"] != "none" and not ex.get("acknowledge_untrusted_network"):
            out.append(f"execution/network: `{ex['network']}` gives untrusted code network access "
                       "(setup commands are candidate-controlled too). Use `network: none`, or set "
                       "`acknowledge_untrusted_network: true` after reviewing the risk")
    repo = d["change"]["repo"]
    if URL_WITH_CREDENTIALS.match(repo):
        out.append("change/repo: the URL embeds credentials; use a credential helper instead")
    if "::" in repo.split("/")[0] or repo.startswith("-"):
        out.append("change/repo: unsupported git transport")
    if "://" in repo and not repo.lower().startswith(ALLOWED_REPO_SCHEMES):
        out.append("change/repo: unsupported URL scheme")
    for arm in ("baseline", "candidate"):
        if d["change"][arm]["ref"].startswith("-"):
            out.append(f"change/{arm}/ref: a ref cannot start with `-`")
    if ex["backend"] == "local" and ex["network"] != "full":
        out.append("execution: the local backend cannot restrict network; use "
                   "`backend: docker` for `network: none` or `setup-only`")
    reserved = {"VALIDATE_ARM", "VALIDATE_SEED", "VALIDATE_OUTPUT_DIR",
                "VALIDATE_HARNESS_DIR", "VALIDATE_ENV_DIR", "VALIDATE_REPO_DIR",
                "VALIDATE_PHASE", "VALIDATE_REPLICATE", "VALIDATE_SEQUENCE",
                "VALIDATE_PERIOD", "VALIDATE_CACHE_DIR"}
    if d.get("measurement"):
        from . import isolation
        reserved |= set(isolation.layers(d["measurement"]))
    for arm, cfg in d["arms"].items():
        clash = reserved & set(cfg["env"])
        if clash:
            out.append(f"arms/{arm}/env: {sorted(clash)} are set by the runner")
        for key, value in cfg["env"].items():
            if any(p.search(value) for p in CREDENTIAL_SHAPES):
                out.append(f"arms/{arm}/env/{key}: looks like a credential. Declarations are "
                           "hashed, stored and exported; pass secrets with `execution.pass_env`")
    return out


def _measurement_problems(d: dict) -> list:
    out = []
    m = d.get("measurement")
    if m is None:
        for c in d["criteria"]:
            if "order_tolerance" in c and not c.get("performance"):
                out.append(f"criteria/{c['id']}: `order_tolerance` only applies with "
                           "`performance: true`")
        return out
    from . import isolation
    cache = m["cache"]
    for c in d["criteria"]:
        cid = c["id"]
        if "order_tolerance" in c and not c.get("performance"):
            out.append(f"criteria/{cid}: `order_tolerance` only applies with `performance: true`")
        if (c.get("performance") and c["role"] != "info" and m["design"] == "crossover"
                and "order_tolerance" not in c):
            why = (f"`{c['metric']}` looks like a timing metric, so it is compared in both "
                   "run orders. D" if c.get(INFERRED) else "d")
            out.append(f"criteria/{cid}: {why}eclare `order_tolerance`, the largest difference "
                       "between the two run orders you would call unimportant. It is fixed "
                       "before the run so it cannot be chosen after seeing the result"
                       + (". If this metric is not a timing, set `performance: false`"
                          if c.get(INFERRED) else ""))
    if cache["profile"] == "cold" and "warmup_runs" in cache:
        out.append("measurement/cache: `warmup_runs` only applies to `profile: warm`")
    for name in cache["isolate"]:
        if name in isolation.LAYERS:
            out.append(f"measurement/cache/isolate: {name} is already isolated by the runner")
        if name.startswith("VALIDATE_") or name in ("HOME", "PATH"):
            out.append(f"measurement/cache/isolate: {name} cannot be redirected")
    subdirs = ["app-" + n.lower() for n in cache["isolate"]]
    if len(set(subdirs)) != len(subdirs):
        out.append("measurement/cache/isolate: two names differ only by case")
    ids = [h["id"] for h in m["reset_hooks"]]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        out.append(f"measurement/reset_hooks: duplicate id {dup!r}")
    for hook in m["reset_hooks"]:
        if PLACEHOLDER in hook["command"]:
            out.append(f"measurement/reset_hooks/{hook['id']}: still a template placeholder")
    if d["execution"]["trust"] == "untrusted" and m["reset_hooks"]:
        out.append("measurement/reset_hooks: reset hooks are trusted commands and cannot be "
                   "combined with `trust: untrusted`")
    return out


def load(path: str) -> dict:
    """Read, validate and normalize. Raises DeclarationError with every problem."""
    raw = read_yaml(path)
    problems = schema_problems(raw, "declaration")
    if problems:
        raise DeclarationError(problems)
    d = normalize(raw)
    problems = semantic_problems(d)
    if problems:
        raise DeclarationError(problems)
    return d


# ── identities ────────────────────────────────────────────────────────────

def declaration_hash(d: dict) -> str:
    body = {
        "spec_version": SPEC_VERSION,
        "objective": d.get("objective", ""),
        "criteria": sorted(d["criteria"], key=lambda c: c["id"]),
        "method": d["method"],
        "validity": d["validity"],
        "data": d.get("data", {}),
    }
    if d.get("measurement"):
        body["measurement"] = d["measurement"]
    return digest(body)


def env_key(d: dict, arm: str, arm_sha: str, image_id=None) -> str:
    """Identity of one arm's built environment. It includes the arm's own
    environment variables, so two arms at the same commit never share a
    directory. Criteria are not part of it."""
    ex = d["execution"]
    docker = ex["backend"] == "docker"
    return digest({
        "arm": arm,
        "sha": arm_sha,
        "setup": d["setup"],
        "arm_env": d["arms"][arm]["env"],
        "backend": ex["backend"],
        "image": ex["image"] if docker else None,
        "image_id": image_id if docker else None,
    })


def spec_hash(d: dict, shas: dict, harness_digest) -> str:
    body = {
        "shas": shas,
        "harness": harness_digest,
        "setup": d["setup"],
        "run": d["run"],
        "preflight": d.get("preflight"),
        "arms": d["arms"],
        "seeds": d["method"]["seeds"],
        "execution": d["execution"],
        "budget": d["budget"],
    }
    if d.get("measurement"):
        body["measurement"] = d["measurement"]
        body["schedule"] = planned_schedule(d)
    return digest(body)


def planned_schedule(d: dict):
    """The frozen run schedule, or None for a declaration with no protocol."""
    m = d.get("measurement")
    return schedule.build(d["method"]["seeds"], m) if m else None


def change_id(repo_identity: str, shas: dict, arms: dict) -> str:
    return "chg_" + digest({"repo": repo_identity, "shas": shas, "arms": arms})[:16]


def project_dir(decl_path: str) -> str:
    return os.path.dirname(os.path.abspath(decl_path))
