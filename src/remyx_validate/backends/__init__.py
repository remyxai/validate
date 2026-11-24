"""Execution backends. One interface: plan / submit / status / cancel / collect."""
from .base import Backend, Job, Result
from .local import LocalBackend
from .docker import DockerBackend


def get(execution: dict) -> Backend:
    if execution["backend"] == "docker":
        return DockerBackend(execution)
    return LocalBackend(execution)


__all__ = ["Backend", "Job", "Result", "LocalBackend", "DockerBackend", "get"]
