"""The supervisor: runs an approved plan and collects what the arms report.

It owns everything the workload must not: the criteria, the plan digest, the
attempt ledger, the checksums and the verdict. The workload only ever gets a
checkout, an environment directory, the harness and an output directory.
"""
from __future__ import annotations

import fcntl
import json
import os
import shlex
import shutil
import signal
import stat
import subprocess
import time

from . import backends, bundle, collect, compare, declaration as decl, gitops, plan as planmod
from .backends import Job
from .canonical import new_run_id, now_iso, tree_digest

ARMS = ("baseline", "candidate")
SUITE_SUFFIXES = (".py", ".sh", ".ipynb")


class Runner:
    def __init__(self, decl_path: str, log=None):
        self.decl_path = os.path.abspath(decl_path)
        self.project = decl.project_dir(self.decl_path)
        self.log = log or (lambda msg: None)
        self.cancelled = False
        self.current_job = None
        self.attempts = []
        self.observations = []
        self.cost_usd = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    def run(self) -> dict:
        plan = planmod.build(self.decl_path)
        approval = planmod.require_approved(self.project, plan)
        self.plan, self.d = plan, plan["declaration"]
        self.backend = backends.get(self.d["execution"])
        self.run_id = new_run_id()
        self.run_dir = os.path.join(planmod.state_dir(self.project), "runs", self.run_id)
        os.makedirs(self.run_dir)
        self.started = time.time()
        self.started_at = now_iso()
        self.hdir = self._snapshot_harness()
        self._status("RUNNING", current="starting")
        old = {s: signal.signal(s, self._on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
        execution_status, detail = "FAILED", ""
        try:
            execution_status, detail = self._execute()
        except Exception as exc:  # the ledger must still be written
            execution_status, detail = "FAILED", f"runner error: {exc}"
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
        if self.cancelled:
            execution_status, detail = "CANCELLED", "cancelled by the user"
        integrity = self._integrity() if execution_status == "COMPLETED" else []
        comparison = compare.compare(self.d, self.observations, execution_status, integrity)
        evidence = bundle.write(
            run_dir=self.run_dir, plan=plan, run_id=self.run_id, approval=approval,
            execution={"status": execution_status, "detail": detail,
                       "started_at": self.started_at, "ended_at": now_iso(),
                       "wall_s": round(time.time() - self.started, 3),
                       "backend": self.backend.plan(), "attempts": self.attempts},
            observations=self.observations, comparison=comparison,
            cost={"reported_usd": self.cost_usd,
                  "cap_usd": self.d["budget"].get("max_cost_usd"),
                  "attempts_without_reported_cost": sum(
                      1 for a in self.attempts if "reported_cost_usd" not in a),
                  "note": ("from the `cost_usd` metric of attempts that reported one. Attempts "
                           "that failed or reported nothing have unknown cost. The cap is a "
                           "check after each attempt, not a spending limit"
                           if self.cost_usd is not None else
                           "unknown: the harness reported no `cost_usd` metric")})
        self._status("DONE", execution_status=execution_status,
                     verdict=comparison["verdict"], bundle_digest=evidence["bundle_digest"])
        return {"run_id": self.run_id, "run_dir": self.run_dir,
                "execution_status": execution_status, "detail": detail,
                "validity": comparison["validity"]["status"],
                "verdict": comparison["verdict"], "reasons": comparison["reasons"],
                "bundle_digest": evidence["bundle_digest"]}

    def _on_signal(self, signum, frame):
        self.cancelled = True
        if self.current_job is not None:
            self.backend.cancel(self.current_job)

    def _status(self, state: str, **extra):
        doc = {"run_id": self.run_id, "state": state, "pid": os.getpid(),
               "pid_started": process_start(os.getpid()),
               "started_at": self.started_at, "updated_at": now_iso(),
               "plan_digest": self.plan["plan_digest"],
               "change_id": self.plan["change"]["change_id"],
               "declaration_hash": self.plan["declaration_hash"],
               "attempts_done": len(self.attempts), **extra}
        tmp = os.path.join(self.run_dir, "status.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
        os.replace(tmp, os.path.join(self.run_dir, "status.json"))

    def _snapshot_harness(self):
        """Copy the approved harness into the run directory and execute against
        the copy. The bytes that run are the bytes whose digest was approved,
        and the user's own files are never handed to the workload."""
        source = planmod.harness_dir(self.d, self.project)
        if not source:
            return None
        dest = os.path.join(self.run_dir, "harness")
        shutil.copytree(source, dest, symlinks=False,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
        if tree_digest(dest)["digest"] != self.plan["harness"]["digest"]:
            raise planmod.PlanError("the harness changed while the run was starting")
        for base, _, names in os.walk(dest):
            for name in names:
                full = os.path.join(base, name)
                os.chmod(full, stat.S_IMODE(os.stat(full).st_mode) & ~0o222)
        return dest

    def _remaining(self) -> float:
        return self.d["budget"]["max_wall_s"] - (time.time() - self.started)

    # ── execution ─────────────────────────────────────────────────────────

    def _execute(self):
        problems = self.backend.preflight()
        if problems:
            return "PREFLIGHT_FAILED", "; ".join(problems)
        self.envs = {}
        for arm in ARMS:
            ok, detail = self._prepare_env(arm)
            if not ok:
                return "PREFLIGHT_FAILED", detail
            if self.cancelled:
                return "CANCELLED", ""
        missing = self._missing_suite_files()
        if missing:
            return "PREFLIGHT_FAILED", "; ".join(missing)
        pre = self.d.get("preflight")
        if pre:
            for arm in ARMS:
                if self._remaining() <= 0:
                    return "BUDGET_EXCEEDED", "wall-clock cap reached before preflight"
                res, _ = self._exec(arm, self.d["method"]["seeds"][0], pre["command"], "preflight",
                                    os.path.join(self.run_dir, "preflight", arm),
                                    min(pre.get("timeout_s", 300), self._remaining()),
                                    {"VALIDATE_SMOKE": "1"})
                if res.status != "ok":
                    return "PREFLIGHT_FAILED", (f"preflight command {res.status} on {arm} "
                                                f"(exit {res.exit_code}); see preflight/{arm}/log.txt")
        retries = self.d["run"]["retries"]
        for i, seed in enumerate(self.d["method"]["seeds"]):
            order = ARMS if i % 2 == 0 else ARMS[::-1]
            for arm in order:
                done = False
                for attempt_no in range(1, retries + 2):
                    if self.cancelled:
                        return "CANCELLED", ""
                    if self._remaining() <= 0:
                        return "BUDGET_EXCEEDED", (f"wall-clock cap of "
                                                   f"{self.d['budget']['max_wall_s']}s reached")
                    status = self._attempt(arm, seed, attempt_no)
                    cap = self.d["budget"].get("max_cost_usd")
                    if cap is not None and self.cost_usd is not None and self.cost_usd > cap:
                        return "BUDGET_EXCEEDED", (f"reported spend ${self.cost_usd:.4f} is over "
                                                   f"the ${cap} cap")
                    if cap is not None and status == "ok" and "reported_cost_usd" not in self.attempts[-1]:
                        return "BUDGET_EXCEEDED", ("a cost cap is declared but the harness reported "
                                                   "no `cost_usd`, so spend is unknown; stopping")
                    if status == "ok":
                        done = True
                        break
                    if status == "cancelled":
                        return "CANCELLED", ""
                if not done:
                    last = self.attempts[-1]
                    kind = "TIMEOUT" if last["status"] == "timeout" else "FAILED"
                    return kind, (f"{arm} seed {seed}: {last['status']} after "
                                  f"{retries + 1} attempt(s), retry limit reached")
        return "COMPLETED", ""

    def _prepare_env(self, arm: str):
        sha = self.plan["change"][arm]["sha"]
        key = self.plan["env_keys"][arm]
        root = os.path.join(planmod.state_dir(self.project), "cache", "envs", key[:24])
        repo, env, marker = os.path.join(root, "repo"), os.path.join(root, "env"), os.path.join(root, "ready.json")
        self.envs[arm] = {"repo": repo, "env": env, "root": root}
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)   # one builder per environment
            if os.path.isfile(marker) and gitops.head_sha(repo) == sha and not gitops.tracked_changes(repo):
                self.log(f"[{arm}] environment reused ({key[:12]})")
                return True, ""
            self.log(f"[{arm}] building environment at {sha[:12]}")
            self._status("RUNNING", current=f"setup {arm}")
            if os.path.isfile(marker):
                os.unlink(marker)
            if os.path.isdir(env):
                shutil.rmtree(env)
            mirror = gitops.ensure_mirror(self.d["change"]["repo"], self.project)
            gitops.checkout(mirror, sha, repo)
            os.makedirs(env, exist_ok=True)
            for i, command in enumerate(self.d["setup"]):
                if self._remaining() <= 0:
                    return False, f"wall-clock cap reached during setup on {arm}"
                out = os.path.join(self.run_dir, "setup", arm, f"step{i}")
                res, _ = self._exec(arm, self.d["method"]["seeds"][0], command, "setup", out,
                                    self._remaining(), {})
                if res.status != "ok":
                    rel = os.path.relpath(os.path.join(out, "log.txt"), self.run_dir)
                    return False, f"setup step {i} {res.status} on {arm} (exit {res.exit_code}); see {rel}"
            changed = gitops.tracked_changes(repo)
            if changed:
                return False, (f"setup modified tracked files on {arm} ({changed[0]}"
                               f"{' and others' if len(changed) > 1 else ''}); the checkout no "
                               "longer matches the approved commit")
            with open(marker, "w", encoding="utf-8") as fh:
                json.dump({"sha": sha, "env_key": key, "built_at": now_iso()}, fh)
            return True, ""

    def _host_paths(self, arm: str) -> dict:
        return {"repo": self.envs[arm]["repo"], "env": self.envs[arm]["env"],
                "harness": self.hdir or "", "output": ""}

    def _missing_suite_files(self) -> list:
        """A suite file absent from one arm would run nothing on that arm and
        leave a blank column. Refuse before spending compute."""
        problems = []
        for arm in ARMS:
            host = self._host_paths(arm)
            try:
                tokens = shlex.split(self.backend.render(self.d["run"]["command"], host))
            except ValueError:
                continue
            for tok in tokens:
                if tok.startswith("-") or not tok.endswith(SUITE_SUFFIXES):
                    continue
                path = tok if os.path.isabs(tok) else os.path.join(host["repo"], tok)
                if not os.path.isfile(path):
                    shown = tok
                    if self.hdir and path.startswith(self.hdir + os.sep):
                        shown = "harness/" + os.path.relpath(path, self.hdir)
                    problems.append(f"suite file `{shown}` does not exist for the {arm} arm")
        return problems

    def _exec(self, arm, seed, command, phase, out_root, timeout_s, extra_env):
        output_dir = os.path.join(out_root, "output")
        os.makedirs(output_dir, exist_ok=True)
        env = dict(self.d["arms"][arm]["env"])
        env.update(extra_env)
        env.update({"VALIDATE_ARM": arm, "VALIDATE_SEED": str(seed)})
        job = Job(command=command, phase=phase, repo_dir=self.envs[arm]["repo"],
                  env_dir=self.envs[arm]["env"], output_dir=output_dir, harness_dir=self.hdir,
                  env=env, timeout_s=timeout_s, log_path=os.path.join(out_root, "log.txt"),
                  name=f"{phase}-{arm}-{seed}")
        self.current_job = job
        try:
            self.backend.submit(job)
            res = self.backend.collect(job)
        finally:
            self.current_job = None
        return res, output_dir

    def _attempt(self, arm: str, seed: int, attempt_no: int) -> str:
        attempt_id = f"att_{arm}_s{seed}_t{attempt_no}"
        self.log(f"[{arm}] seed {seed} attempt {attempt_no}")
        self._status("RUNNING", current=f"{arm} seed {seed} attempt {attempt_no}")
        out_root = os.path.join(self.run_dir, "attempts", attempt_id)
        timeout = min(self.d["run"]["timeout_s"], max(1.0, self._remaining()))
        res, output_dir = self._exec(arm, seed, self.d["run"]["command"], "run", out_root, timeout, {})
        record = {"attempt_id": attempt_id, "arm": arm, "seed": seed, "try": attempt_no,
                  "status": res.status, "exit_code": res.exit_code,
                  "duration_s": round(res.duration_s, 3),
                  "log": f"attempts/{attempt_id}/log.txt"}
        if res.status == "ok":
            try:
                parsed = collect.read(output_dir, res.stdout)
            except collect.ResultsError as exc:
                parsed = None
                record["status"], record["error"] = "invalid_results", str(exc)
            else:
                if parsed is None:
                    record["status"] = "no_results"
                    record["error"] = ("the command exited 0 but wrote no results.json and "
                                       "printed no METRICS_JSON line")
            if parsed is not None:
                rows = collect.flatten(parsed, arm, seed, attempt_id)
                self.observations += rows
                record["results_source"] = parsed["source"]
                record["n_observations"] = len(rows)
                if parsed["ignored"]:
                    record["ignored_non_numeric"] = parsed["ignored"]
                # A run-level cost_usd is the attempt's total. Otherwise per-unit
                # costs are summed. Never both, which would count spend twice.
                total = [r["value"] for r in rows if r["metric"] == "cost_usd" and r["unit"] is None]
                spend = total or [r["value"] for r in rows if r["metric"] == "cost_usd"]
                if any(v < 0 for v in spend):
                    record["status"], record["error"] = "invalid_results", "negative `cost_usd`"
                    del self.observations[-len(rows):]
                elif spend:
                    record["reported_cost_usd"] = sum(spend)
                    self.cost_usd = (self.cost_usd or 0.0) + sum(spend)
        self.attempts.append(record)
        return record["status"]

    # ── integrity ─────────────────────────────────────────────────────────

    def _integrity(self) -> list:
        findings = []
        if self.hdir:
            now = tree_digest(self.hdir)["digest"]
            ok = now == self.plan["harness"]["digest"]
            findings.append({"check": "harness_unchanged", "passed": ok, "code": "harness_modified",
                             "detail": "harness files match the approved plan" if ok else
                             "harness files changed during the run; the measurement cannot be trusted"})
        try:
            ok = decl.declaration_hash(decl.load(self.decl_path)) == self.plan["declaration_hash"]
        except Exception:
            ok = False
        findings.append({"check": "declaration_unchanged", "passed": ok, "code": "declaration_modified",
                         "detail": "criteria match the approved plan" if ok else
                         "the declaration file changed during the run; the approved criteria were "
                         "used, but the run is marked invalid"})
        for arm in ARMS:
            repo = self.envs[arm]["repo"]
            ok = gitops.head_sha(repo) == self.plan["change"][arm]["sha"]
            findings.append({"check": f"checkout_pinned_{arm}", "passed": ok, "code": "checkout_moved",
                             "detail": f"{arm} checkout is at the approved commit" if ok else
                             f"{arm} checkout no longer points at the approved commit"})
            changed = gitops.tracked_changes(repo)
            findings.append({"check": f"checkout_clean_{arm}", "passed": not changed,
                             "code": "checkout_modified",
                             "detail": f"{arm} tracked files match the approved commit" if not changed
                             else f"{arm} tracked files were modified during the run ({changed[0]}"
                                  f"{' and others' if len(changed) > 1 else ''})"})
        return findings


# ── helpers used by the CLI ───────────────────────────────────────────────

def process_start(pid: int) -> str:
    """When a pid started, as text. Lets `cancel` tell the runner from an
    unrelated process that later reused its pid."""
    try:
        proc = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True)
    except OSError:
        return ""
    return " ".join(proc.stdout.split())


def runs_dir(project: str) -> str:
    return os.path.join(planmod.state_dir(project), "runs")


def list_runs(project: str) -> list:
    folder = runs_dir(project)
    out = []
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder), reverse=True):
            path = os.path.join(folder, name, "status.json")
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as fh:
                    out.append(json.load(fh))
    return out


def find_run(project: str, run_id: str = "") -> str:
    folder = runs_dir(project)
    names = sorted(n for n in os.listdir(folder)
                   if os.path.isfile(os.path.join(folder, n, "status.json"))) if os.path.isdir(folder) else []
    if not names:
        raise FileNotFoundError("no runs yet in this project")
    if not run_id or run_id == "latest":
        return os.path.join(folder, names[-1])
    hits = [n for n in names if n == run_id or n.startswith(run_id) or n.endswith(run_id)]
    if len(hits) != 1:
        raise FileNotFoundError(f"run {run_id!r} matches {len(hits)} runs")
    return os.path.join(folder, hits[0])


def cancel(project: str, run_id: str = "") -> dict:
    run_dir = find_run(project, run_id)
    with open(os.path.join(run_dir, "status.json"), encoding="utf-8") as fh:
        status = json.load(fh)
    if status["state"] != "RUNNING":
        return {"run_id": status["run_id"], "cancelled": False, "reason": "run is not running"}
    if process_start(status["pid"]) != status.get("pid_started"):
        return {"run_id": status["run_id"], "cancelled": False,
                "reason": "the runner process is gone (its pid now belongs to something else)"}
    try:
        os.kill(status["pid"], signal.SIGTERM)
    except ProcessLookupError:
        return {"run_id": status["run_id"], "cancelled": False, "reason": "runner process is gone"}
    return {"run_id": status["run_id"], "cancelled": True}
