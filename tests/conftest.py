import json
import os
import subprocess
import sys

import pytest

from remyx_validate import demo, plan as planmod, runner as runmod


@pytest.fixture
def run_scenario(tmp_path):
    """Build a demo scenario, approve its plan, run it. Returns (summary, project dir)."""

    def _run(name, edit=None):
        path = demo.create(str(tmp_path / name), name)
        project = os.path.dirname(path)
        if edit:
            with open(path, encoding="utf-8") as fh:
                text = edit(fh.read())
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        plan = planmod.build(path)
        planmod.save(project, plan)
        planmod.approve(project, plan["plan_digest"], "test")
        return runmod.Runner(path).run(), project

    return _run


@pytest.fixture
def cli():
    def _cli(*args, cwd=None, env=None):
        proc = subprocess.run([sys.executable, "-m", "remyx_validate.cli", *args],
                              cwd=cwd, capture_output=True, text=True, env=env)
        return proc

    return _cli


def load_evidence(summary):
    with open(os.path.join(summary["run_dir"], "evidence.json"), encoding="utf-8") as fh:
        return json.load(fh)


def has_docker():
    try:
        return subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    except OSError:
        return False
