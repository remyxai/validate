"""Local subprocess backend, for code you already trust.

The workload runs as your user. Its environment is rebuilt from a short
allowlist, so tokens exported in your shell are not inherited, but it can
still read any file you can. Use the docker backend for code you have not
read.
"""
from __future__ import annotations

import os
import subprocess
import time

from .base import Backend, Job

BASE_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER", "SHELL")


class LocalBackend(Backend):
    name = "local"

    def plan(self) -> dict:
        return {"backend": "local", "isolation": "none (trusted code)",
                "inherits_env": list(BASE_ENV) + list(self.execution["pass_env"])}

    def placeholders(self, job: Job) -> dict:
        return {"repo": job.repo_dir, "env": job.env_dir, "output": job.output_dir,
                "harness": job.harness_dir or ""}

    def build_env(self, job: Job) -> dict:
        env = {k: os.environ[k] for k in BASE_ENV if k in os.environ}
        for key in self.execution["pass_env"]:
            if key in os.environ:
                env[key] = os.environ[key]
        ph = self.placeholders(job)
        env.update(job.env)
        env.update({"VALIDATE_REPO_DIR": ph["repo"], "VALIDATE_ENV_DIR": ph["env"],
                    "VALIDATE_OUTPUT_DIR": ph["output"], "VALIDATE_HARNESS_DIR": ph["harness"],
                    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"})
        return env

    def submit(self, job: Job) -> Job:
        command = self.render(job.command, self.placeholders(job))
        os.makedirs(os.path.dirname(job.log_path), exist_ok=True)
        log = open(job.log_path, "w", encoding="utf-8")
        job.started = time.time()
        job.proc = subprocess.Popen(["bash", "-c", command], cwd=job.repo_dir,
                                    env=self.build_env(job), stdout=log,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
        log.close()
        return job
