"""Docker backend, for pull-request code and anything else you have not read.

What the workload gets: its own checkout, a per-arm environment directory,
the harness mounted read-only, and an output directory. What it does not
get: your home directory, your shell environment, or (by default during the
run) a network. The checkout and the environment directory are read-only during the run, and
HOME is a fresh in-memory directory for every attempt, so one seed cannot
leave state for the next.

This backend is experimental.

Limits worth knowing: this is container isolation, not a VM. A kernel or
Docker escape defeats it, and anyone in the `docker` group is root-equivalent
on the host. The harness and the candidate share the container, so candidate
code can still lie about its own results.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid

from .base import Backend, Job

WORK = "/work"


class DockerBackend(Backend):
    name = "docker"

    def plan(self) -> dict:
        ex = self.execution
        return {"backend": "docker", "image": ex["image"], "network": ex["network"],
                "limits": ex["limits"], "isolation": "experimental. container; harness read-only; "
                "checkout and environment read-only during run; no host env; cap-drop ALL",
                "inherits_env": list(ex["pass_env"]), "fresh_process_per_attempt": True}

    def preflight(self) -> list:
        if not shutil.which("docker"):
            return ["docker is not installed or not on PATH"]
        proc = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            return ["cannot reach the docker daemon: " + proc.stderr.strip()[:200]]
        return []

    def placeholders(self, job: Job) -> dict:
        return {"repo": f"{WORK}/repo", "env": f"{WORK}/env", "output": f"{WORK}/output",
                "harness": f"{WORK}/harness" if job.harness_dir else ""}

    def argv(self, job: Job) -> list:
        ex = self.execution
        ph = self.placeholders(job)
        networked = ex["network"] == "full" or (ex["network"] == "setup-only" and job.phase == "setup")
        setup = job.phase == "setup"
        repo_mode = "rw" if setup else "ro"
        job.container = "validate-" + uuid.uuid4().hex[:12]
        args = ["docker", "run", "--rm", "--name", job.container,
                "--network", "bridge" if networked else "none",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", "1024", "--read-only", "--tmpfs", "/tmp:exec,mode=1777",
                "-v", f"{job.repo_dir}:{ph['repo']}:{repo_mode}",
                "-v", f"{job.env_dir}:{ph['env']}:{repo_mode}",
                "-v", f"{job.output_dir}:{ph['output']}:rw",
                "-w", ph["repo"]]
        if job.harness_dir:
            args += ["-v", f"{job.harness_dir}:{ph['harness']}:ro"]
        if job.cache_dir:
            args += ["-v", f"{job.cache_dir}:{WORK}/cache:rw"]
        limits = ex.get("limits") or {}
        if limits.get("cpus"):
            args += ["--cpus", str(limits["cpus"])]
        if limits.get("memory"):
            args += ["--memory", str(limits["memory"])]
        if limits.get("gpus"):
            args += ["--gpus", str(limits["gpus"])]
        env = dict(job.env)
        for key in ex["pass_env"]:
            if key in os.environ:
                env[key] = os.environ[key]
        env.update({"HOME": f"{ph['env']}/home" if setup else "/tmp/home", "VALIDATE_REPO_DIR": ph["repo"],
                    "VALIDATE_ENV_DIR": ph["env"], "VALIDATE_OUTPUT_DIR": ph["output"],
                    "VALIDATE_HARNESS_DIR": ph["harness"],
                    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"})
        if job.cache_dir:
            # Host paths in job.host_env mean nothing in the container and are
            # not passed. The cache layers are remapped onto the mount.
            env["VALIDATE_CACHE_DIR"] = f"{WORK}/cache"
            for name, sub in job.cache_layers.items():
                env[name] = f"{WORK}/cache/{sub}"
        for key, value in sorted(env.items()):
            args += ["-e", f"{key}={value}"]
        command = self.render(job.command, ph)
        if not setup:
            command = "mkdir -p /tmp/home && " + command
        args += [ex["image"], "sh", "-c", command]
        return args

    def submit(self, job: Job) -> Job:
        os.makedirs(os.path.join(job.env_dir, "home"), exist_ok=True)
        os.makedirs(os.path.dirname(job.log_path), exist_ok=True)
        log = open(job.log_path, "w", encoding="utf-8")
        job.started = time.time()
        job.proc = subprocess.Popen(self.argv(job), stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        log.close()
        return job

    def _kill(self, job: Job) -> None:
        if job.container:
            subprocess.run(["docker", "kill", job.container], capture_output=True)
        super()._kill(job)
