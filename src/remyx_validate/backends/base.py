from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional


MAX_LOG_READ = 8 * 1024 * 1024


@dataclass
class Job:
    """One command to run for one arm.

    ``phase`` is "setup", "preflight", "reset", "warmup" or "run". Paths are
    host paths; a backend maps them to wherever the workload sees them.

    ``cache_dir`` is the attempt's own cache directory and ``cache_layers``
    maps environment variables to subdirectories of it. ``host_env`` holds
    variables whose values are host paths, which only a backend that shares
    the host filesystem may pass on.
    """
    command: str
    phase: str
    repo_dir: str
    env_dir: str
    output_dir: str
    harness_dir: Optional[str]
    env: dict
    timeout_s: float
    log_path: str
    name: str = "job"
    cache_dir: Optional[str] = None
    cache_layers: dict = field(default_factory=dict)
    host_env: dict = field(default_factory=dict)
    proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    started: float = 0.0
    container: str = ""


@dataclass
class Result:
    status: str            # ok | failed | timeout | cancelled
    exit_code: Optional[int]
    duration_s: float
    stdout: str


class Backend:
    name = "base"

    def __init__(self, execution: dict):
        self.execution = execution
        self._cancelled = False

    def plan(self) -> dict:
        """What this backend will do, for the plan a person approves."""
        raise NotImplementedError

    def preflight(self) -> list:
        """Problems that would stop any job from starting."""
        return []

    def placeholders(self, job: Job) -> dict:
        raise NotImplementedError

    def submit(self, job: Job) -> Job:
        raise NotImplementedError

    def status(self, job: Job) -> str:
        if job.proc is None:
            return "pending"
        return "running" if job.proc.poll() is None else "exited"

    def cancel(self, job: Job) -> None:
        self._cancelled = True
        self._kill(job)

    def _kill(self, job: Job) -> None:
        if job.proc and job.proc.poll() is None:
            try:
                os.killpg(job.proc.pid, signal.SIGTERM)
                try:
                    job.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(job.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def collect(self, job: Job) -> Result:
        """Wait for the job, enforcing its timeout. Output is read from the log."""
        timed_out = False
        try:
            job.proc.wait(timeout=job.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill(job)
            job.proc.wait()
        duration = time.time() - job.started
        try:
            # Only the tail is parsed, so a workload cannot exhaust memory by logging.
            with open(job.log_path, "rb") as fh:
                size = os.fstat(fh.fileno()).st_size
                fh.seek(max(0, size - MAX_LOG_READ))
                stdout = fh.read().decode("utf-8", errors="replace")
        except OSError:
            stdout = ""
        if self._cancelled:
            status = "cancelled"
        elif timed_out:
            status = "timeout"
        elif job.proc.returncode == 0:
            status = "ok"
        else:
            status = "failed"
        return Result(status, job.proc.returncode, duration, stdout)

    @staticmethod
    def render(command: str, mapping: dict) -> str:
        """Substitute {repo} {env} {output} {harness}, shell-quoted so a path
        with spaces stays one word."""
        for key, value in mapping.items():
            command = command.replace("{" + key + "}", shlex.quote(value) if value else "")
        return command
