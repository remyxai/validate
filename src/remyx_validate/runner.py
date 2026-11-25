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

from . import (backends, bundle, collect, compare, declaration as decl, gitops, isolation,
               plan as planmod, schedule)
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
        self.protocol = None
        self.owned_caches = set()

    # ── lifecycle ─────────────────────────────────────────────────────────

    def run(self) -> dict:
        plan = planmod.build(self.decl_path)
        approval = planmod.require_approved(self.project, plan)
        self.plan, self.d = plan, plan["declaration"]
        self.backend = backends.get(self.d["execution"])
        self.m = self.d.get("measurement")
        self.protocol = self._protocol()
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
            for root in sorted(self.owned_caches):   # after a crash or a cancel
                isolation.remove(root, self.run_dir)
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
                       "backend": self.backend.plan(), "attempts": self.attempts,
                       "protocol": self.protocol,
                       **({"schedule": plan["schedule"]} if self.m else {})},
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

    def _protocol(self) -> dict:
        """What was and was not controlled about run order and caches. Written
        into the evidence so a reader does not have to infer it."""
        m = self.m
        if not m:
            return {"design": "alternating", "order_controlled": False,
                    "cache_isolation": "none",
                    "note": "No measurement protocol was declared. Each seed ran in one order "
                            "only, alternating by the seed's position in the list, and caches "
                            "were not reset between attempts. Timing differences in this "
                            "evidence are not separated from run order or cache warmth."}
        cache = m["cache"]
        shared = {}
        if cache["assets"] == "shared" and self.backend.name == "local":
            shared = isolation.shared_assets()
        self.shared_env = shared
        return {"design": m["design"], "order_controlled": m["design"] == "crossover",
                "replicates": m["replicates"], "schedule_seed": m["schedule_seed"],
                "attempts_per_seed": (4 if m["design"] == "crossover" else 2) * m["replicates"],
                "retry_policy": "a failed attempt restarts its whole sequence (both arms) "
                                "from a fresh cache; superseded attempts are kept in the "
                                "ledger and excluded from the comparison",
                "cache_isolation": "per-attempt", "cache_profile": cache["profile"],
                "warmup_runs": cache.get("warmup_runs", 0),
                "isolated_layers": sorted(isolation.layers(m)),
                "shared_immutable_assets": shared, "assets": cache["assets"],
                "reset_hooks": [h["id"] for h in m["reset_hooks"]],
                "declared_uncontrolled": list(cache["uncontrolled"]),
                "not_controlled": list(isolation.NOT_CONTROLLED),
                "fresh_process_per_attempt": True,
                "host": isolation.host_fingerprint(),
                "timing": "metrics are timed by the harness; `seconds` in the attempt table is "
                          "the runner's wall time for the whole process and is not a metric",
                "concurrency": "one measured attempt at a time; arms never run side by side"}

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
        if self.m:
            return self._execute_schedule()
        retries = self.d["run"]["retries"]
        for i, seed in enumerate(self.d["method"]["seeds"]):
            order = ARMS if i % 2 == 0 else ARMS[::-1]
            for arm in order:
                done = False
                for attempt_no in range(1, retries + 2):
                    status, stop = self._guarded_attempt(arm, seed, attempt_no)
                    if stop:
                        return stop
                    if status == "ok":
                        done = True
                        break
                if not done:
                    last = self.attempts[-1]
                    kind = "TIMEOUT" if last["status"] == "timeout" else "FAILED"
                    return kind, (f"{arm} seed {seed}: {last['status']} after "
                                  f"{retries + 1} attempt(s), retry limit reached")
        return "COMPLETED", ""

    def _guarded_attempt(self, arm, seed, attempt_no, slot=None):
        """One attempt with the cancel, wall and cost checks around it. Returns
        (attempt status, None) or (None, (execution status, detail)) to stop."""
        if self.cancelled:
            return None, ("CANCELLED", "")
        if self._remaining() <= 0:
            return None, ("BUDGET_EXCEEDED",
                          f"wall-clock cap of {self.d['budget']['max_wall_s']}s reached")
        status = self._attempt(arm, seed, attempt_no, slot)
        cap = self.d["budget"].get("max_cost_usd")
        if cap is not None and self.cost_usd is not None and self.cost_usd > cap:
            return None, ("BUDGET_EXCEEDED", f"reported spend ${self.cost_usd:.4f} is over "
                                             f"the ${cap} cap")
        if cap is not None and status == "ok" and "reported_cost_usd" not in self.attempts[-1]:
            return None, ("BUDGET_EXCEEDED", "a cost cap is declared but the harness reported "
                                             "no `cost_usd`, so spend is unknown; stopping")
        if status == "cancelled":
            return None, ("CANCELLED", "")
        return status, None

    def _execute_schedule(self):
        """Run the approved schedule. The unit that is retried is the whole
        sequence: if either arm of an AB or BA pair fails, both are run again
        from fresh caches, so a kept pair always has its two arms measured
        back to back in the planned order. Attempts from an abandoned try stay
        in the ledger, marked discarded, and their observations are dropped."""
        retries = self.d["run"]["retries"]
        for group in schedule.groups(self.plan["schedule"]):
            done = False
            for attempt_no in range(1, retries + 2):
                first = len(self.attempts)
                statuses = []
                for slot in group:
                    status, stop = self._guarded_attempt(slot["arm"], slot["seed"], attempt_no, slot)
                    if stop:
                        return stop
                    statuses.append(status)
                    if status != "ok":
                        break
                if statuses == ["ok"] * len(group):
                    done = True
                    break
                self._discard(self.attempts[first:])
            if not done:
                last = self.attempts[-1]
                kind = "TIMEOUT" if last["status"] == "timeout" else "FAILED"
                slot = group[0]
                return kind, (f"seed {slot['seed']} replicate {slot['replicate']} sequence "
                              f"{slot['sequence']}: {last['arm']} {last['status']} after "
                              f"{retries + 1} attempt(s) at the sequence, retry limit reached")
        return "COMPLETED", ""

    def _discard(self, records):
        gone = set()
        for record in records:
            record["discarded"] = True
            gone.add(record["attempt_id"])
        self.observations = [o for o in self.observations if o["attempt_id"] not in gone]

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

    def _exec(self, arm, seed, command, phase, out_root, timeout_s, extra_env, cache_dir=None):
        output_dir = os.path.join(out_root, "output")
        os.makedirs(output_dir, exist_ok=True)
        env = dict(self.d["arms"][arm]["env"])
        env.update(extra_env)
        env.update({"VALIDATE_ARM": arm, "VALIDATE_SEED": str(seed)})
        job = Job(command=command, phase=phase, repo_dir=self.envs[arm]["repo"],
                  env_dir=self.envs[arm]["env"], output_dir=output_dir, harness_dir=self.hdir,
                  env=env, timeout_s=timeout_s, log_path=os.path.join(out_root, "log.txt"),
                  name=f"{phase}-{arm}-{seed}")
        own_cache = False
        if self.m and phase != "setup":
            env["VALIDATE_PHASE"] = phase
            if cache_dir is None:
                # Preflight gets a throwaway cache of its own, so it cannot
                # warm anything a measured attempt will read.
                cache_dir, own_cache = os.path.join(out_root, "cache"), True
                self._new_cache(cache_dir)
            job.cache_dir = cache_dir
            job.cache_layers = isolation.layers(self.m)
            job.host_env = dict(self.shared_env)
        self.current_job = job
        try:
            self.backend.submit(job)
            res = self.backend.collect(job)
        finally:
            self.current_job = None
            if own_cache:
                self._drop_cache(cache_dir)
        return res, output_dir

    def _new_cache(self, root: str) -> None:
        self.owned_caches.add(root)
        isolation.prepare(root, self.run_dir, self.m)

    def _drop_cache(self, root: str) -> str:
        outcome = isolation.remove(root, self.run_dir)
        if outcome in ("removed", "absent"):
            self.owned_caches.discard(root)
        return outcome

    def _conditioned(self, arm, seed, out_root, slot, deadline, record):
        """Establish the declared cache state for one measured attempt: a new
        empty cache, then the reset hooks, then (warm profile) the declared
        number of untimed warm-up runs for this arm alone. Returns the cache
        directory, or None with record["status"] set when a step failed."""
        cache_dir = os.path.join(out_root, "cache")
        iso = record["isolation"] = {"profile": self.m["cache"]["profile"], "fresh": False,
                                     "reset_hooks": [], "warmup": []}
        try:
            self._new_cache(cache_dir)
        except (isolation.IsolationError, OSError) as exc:
            record["status"], record["error"] = "reset_failed", f"cache isolation: {exc}"
            return None
        iso["fresh"] = isolation.is_clean(cache_dir, self.m)
        if not iso["fresh"]:
            record["status"], record["error"] = "reset_failed", "the new cache directory is not empty"
            return cache_dir
        extra = {"VALIDATE_REPLICATE": str(slot["replicate"]), "VALIDATE_SEQUENCE": slot["sequence"],
                 "VALIDATE_PERIOD": str(slot["period"])}
        for hook in self.m["reset_hooks"]:
            res, _ = self._exec(arm, seed, hook["command"], "reset",
                                os.path.join(out_root, "reset", hook["id"]),
                                max(1.0, min(hook["timeout_s"], deadline())), extra, cache_dir)
            iso["reset_hooks"].append({"id": hook["id"], "status": res.status,
                                       "duration_s": round(res.duration_s, 3)})
            if res.status != "ok":
                record["status"] = "cancelled" if res.status == "cancelled" else "reset_failed"
                record["error"] = f"reset hook `{hook['id']}` {res.status} (exit {res.exit_code})"
                return cache_dir
        for i in range(self.m["cache"].get("warmup_runs", 0)):
            res, _ = self._exec(arm, seed, self.d["run"]["command"], "warmup",
                                os.path.join(out_root, "warmup", str(i + 1)),
                                max(1.0, min(self.d["run"]["timeout_s"], deadline())), extra, cache_dir)
            iso["warmup"].append({"run": i + 1, "status": res.status,
                                  "duration_s": round(res.duration_s, 3)})
            if res.status != "ok":
                record["status"] = "cancelled" if res.status == "cancelled" else "warmup_failed"
                record["error"] = f"warm-up run {i + 1} {res.status} (exit {res.exit_code})"
                return cache_dir
        return cache_dir

    def _attempt(self, arm: str, seed: int, attempt_no: int, slot=None) -> str:
        if slot is None:
            attempt_id, where, tags = f"att_{arm}_s{seed}_t{attempt_no}", f"seed {seed}", {}
        else:
            attempt_id = (f"att_{arm}_s{seed}_r{slot['replicate']}_{slot['sequence']}"
                          f"_t{attempt_no}")
            where = (f"seed {seed} replicate {slot['replicate']} {slot['sequence']} "
                     f"position {slot['period']}")
            tags = {"replicate": slot["replicate"], "sequence": slot["sequence"],
                    "period": slot["period"]}
        self.log(f"[{arm}] {where} attempt {attempt_no}")
        self._status("RUNNING", current=f"{arm} {where} attempt {attempt_no}")
        out_root = os.path.join(self.run_dir, "attempts", attempt_id)
        record = {"attempt_id": attempt_id, "arm": arm, "seed": seed, "try": attempt_no,
                  "status": "ok", "exit_code": None, "duration_s": 0.0,
                  "log": f"attempts/{attempt_id}/log.txt"}
        cache_dir, extra = None, {}
        if slot is not None:
            record.update(tags, block=slot["block"], planned_ordinal=slot["ordinal"],
                          ordinal=len(self.attempts) + 1)
            extra = {"VALIDATE_REPLICATE": str(slot["replicate"]),
                     "VALIDATE_SEQUENCE": slot["sequence"], "VALIDATE_PERIOD": str(slot["period"])}
            cache_dir = self._conditioned(arm, seed, out_root, slot,
                                          lambda: max(1.0, self._remaining()), record)
        res = None
        if record["status"] == "ok":
            timeout = min(self.d["run"]["timeout_s"], max(1.0, self._remaining()))
            if slot is not None:
                record["conditions"] = {"started_s": round(time.time() - self.started, 3),
                                        "before": isolation.conditions()}
            res, output_dir = self._exec(arm, seed, self.d["run"]["command"], "run", out_root,
                                         timeout, extra, cache_dir)
            record.update(status=res.status, exit_code=res.exit_code,
                          duration_s=round(res.duration_s, 3))
            if slot is not None:
                record["conditions"]["after"] = isolation.conditions()
        if cache_dir is not None:
            record["isolation"]["cleanup"] = self._drop_cache(cache_dir)
        if res is not None and res.status == "ok":
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
                rows = [{**row, **tags} for row in collect.flatten(parsed, arm, seed, attempt_id)]
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
