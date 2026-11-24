"""Hostile fixtures and the local-only guarantees.

Each test is one way a workload or a later editor could try to bend the
result, and the behaviour that stops it.
"""
import io
import json
import os
import re
import tarfile

import pytest

from remyx_validate import bundle, collect, demo, plan as planmod, runner as runmod
from remyx_validate.backends import DockerBackend, Job, LocalBackend
from remyx_validate.declaration import DEFAULTS
from conftest import has_docker, load_evidence

SRC = os.path.join(os.path.dirname(__file__), "..", "src", "remyx_validate")


# ── forged or altered evidence ────────────────────────────────────────────

def test_edited_evidence_is_detected(run_scenario):
    summary, _ = run_scenario("guardrail")
    path = os.path.join(summary["run_dir"], "evidence.json")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text.replace('"verdict": "FAIL"', '"verdict": "PASS"'))
    res = bundle.verify(summary["run_dir"])
    assert not res["ok"] and "modified file: evidence.json" in res["problems"]


def test_forged_verdict_with_regenerated_checksums_is_detected(run_scenario):
    """An editor who also rewrites checksums.json still cannot make the
    recorded verdict disagree with the observations."""
    summary, _ = run_scenario("guardrail")
    path = os.path.join(summary["run_dir"], "evidence.json")
    with open(path, encoding="utf-8") as fh:
        ev = json.load(fh)
    ev["comparison"]["verdict"] = "PASS"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ev, fh)
    bundle.write_checksums(summary["run_dir"])
    res = bundle.verify(summary["run_dir"])
    assert not res["ok"]
    assert any("recorded verdict PASS" in p for p in res["problems"])


def test_edited_observations_are_detected(run_scenario):
    summary, _ = run_scenario("guardrail")
    path = os.path.join(summary["run_dir"], "observations.jsonl")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text.replace('"metric":"latency_ms","seed":0,"slice":"long"',
                              '"metric":"latency_ms","seed":0,"slice":"short"', 1))
    bundle.write_checksums(summary["run_dir"])
    res = bundle.verify(summary["run_dir"])
    assert not res["ok"] and any("observations.jsonl" in p for p in res["problems"])


def test_clean_bundle_verifies(run_scenario):
    summary, _ = run_scenario("positive")
    res = bundle.verify(summary["run_dir"])
    assert res["ok"], res["problems"]
    assert res["recomputed_verdict"] == "PASS"


# ── criteria and plan cannot move after approval ──────────────────────────

def test_candidate_cannot_rewrite_the_criteria(run_scenario, tmp_path):
    """The workload appends a looser guardrail to impact.yaml mid-run. The
    approved criteria are still the ones judged, and the run is marked invalid."""
    target = tmp_path / "guardrail" / "impact.yaml"

    def hostile(text):
        return text.replace("command: python3 {harness}/eval.py",
                            f"command: \"python3 {{harness}}/eval.py && echo 'criteria: []' >> '{target}'\"")
    summary, _ = run_scenario("guardrail", edit=hostile)
    ev = load_evidence(summary)
    judged = {c["id"]: c for c in ev["declaration"]["resolved"]["criteria"]}
    assert judged["p95_latency"]["max"] == 800
    assert summary["verdict"] == "INCOMPLETE" and summary["validity"] == "INVALID"
    assert "integrity:declaration_modified" in summary["reasons"]


def test_harness_edit_after_approval_fails_closed(tmp_path):
    path = demo.create(str(tmp_path / "p"), "positive")
    project = os.path.dirname(path)
    plan = planmod.build(path)
    planmod.save(project, plan)
    planmod.approve(project, plan["plan_digest"], "test")
    with open(os.path.join(project, "harness", "eval.py"), "a", encoding="utf-8") as fh:
        fh.write("\n# quietly changed\n")
    with pytest.raises(planmod.NotApproved) as exc:
        runmod.Runner(path).run()
    assert any(row["field"] == "harness.digest" for row in exc.value.drift)
    assert runmod.list_runs(project) == []


def test_moved_branch_after_approval_fails_closed(tmp_path):
    path = demo.create(str(tmp_path / "p"), "positive")
    project = os.path.dirname(path)
    plan = planmod.build(path)
    planmod.save(project, plan)
    planmod.approve(project, plan["plan_digest"], "test")
    repo = os.path.join(project, "app-repo")
    demo._git(["checkout", "--quiet", "candidate"], repo)
    with open(os.path.join(repo, "app", "extra.py"), "w") as fh:
        fh.write("x = 1\n")
    demo._git(["add", "-A"], repo)
    demo._git(["commit", "--quiet", "-m", "moved"], repo)
    with pytest.raises(planmod.NotApproved) as exc:
        runmod.Runner(path).run()
    assert any(row["field"] == "change.candidate.sha" for row in exc.value.drift)


def test_edited_saved_plan_cannot_be_approved(tmp_path):
    path = demo.create(str(tmp_path / "p"), "positive")
    project = os.path.dirname(path)
    plan = planmod.build(path)
    saved = planmod.save(project, plan)
    plan["declaration"]["criteria"][1]["max"] = 99999
    with open(saved, "w", encoding="utf-8") as fh:
        json.dump(plan, fh)
    with pytest.raises(planmod.PlanError):
        planmod.approve(project, plan["plan_digest"], "test")


def test_unapproved_plan_does_not_run(tmp_path):
    path = demo.create(str(tmp_path / "p"), "positive")
    with pytest.raises(planmod.NotApproved):
        runmod.Runner(path).run()


# ── path escape ───────────────────────────────────────────────────────────

def test_harness_path_cannot_leave_the_project(tmp_path):
    path = demo.create(str(tmp_path / "p"), "positive")
    with open(path, encoding="utf-8") as fh:
        text = fh.read().replace("path: ./harness", "path: ../../outside")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    with pytest.raises(planmod.PlanError):
        planmod.build(path)


def test_results_symlink_is_rejected(tmp_path):
    secret = tmp_path / "secret.json"
    secret.write_text('{"resolved": 1}')
    out = tmp_path / "out"
    out.mkdir()
    os.symlink(secret, out / "results.json")
    with pytest.raises(collect.ResultsError):
        collect.read(str(out), "")


def test_harness_may_not_be_or_contain_a_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("x")
    for kind in ("file", "dir", "self"):
        path = demo.create(str(tmp_path / kind), "positive")
        harness = os.path.join(os.path.dirname(path), "harness")
        if kind == "file":
            os.symlink(outside / "secret.txt", os.path.join(harness, "link.txt"))
        elif kind == "dir":
            os.symlink(outside, os.path.join(harness, "linked"))
        else:
            os.rename(harness, harness + "-real")
            os.symlink(harness + "-real", harness)
        with pytest.raises(planmod.PlanError):
            planmod.build(path)


def test_workload_runs_against_a_snapshot_not_the_users_harness(run_scenario):
    """The tamper fixture writes to the harness it was given. That is the
    per-run copy: the run is invalidated and the user's files are untouched."""
    summary, project = run_scenario("tamper")
    assert "integrity:harness_modified" in summary["reasons"]
    from remyx_validate.canonical import tree_digest
    plan = json.load(open(os.path.join(summary["run_dir"], "manifest.json")))["plan"]
    assert tree_digest(os.path.join(project, "harness"))["digest"] == plan["harness"]["digest"]


def _bad_archive(tmp_path, name, build):
    path = tmp_path / name
    with tarfile.open(path, "w:gz") as tar:
        build(tar)
    return str(path)


def _member(tar, name, data=b"x", **attrs):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    for key, value in attrs.items():
        setattr(info, key, value)
    tar.addfile(info, io.BytesIO(data) if info.isreg() else None)


def test_malformed_archives_fail_without_writing_or_raising(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    escape = _bad_archive(work, "escape.tar.gz", lambda t: _member(t, "../escaped.txt"))
    link = _bad_archive(work, "link.tar.gz",
                        lambda t: _member(t, "b/evil", b"", type=tarfile.SYMTYPE, linkname="/etc/passwd"))
    twice = _bad_archive(work, "twice.tar.gz", lambda t: (_member(t, "b/a.txt"), _member(t, "b/a.txt")))
    junk = work / "junk.tar.gz"
    junk.write_bytes(b"this is not an archive")
    for path in (escape, link, twice, str(junk), str(work / "absent.tar.gz")):
        res = bundle.verify(path)
        assert res["ok"] is False and res["problems"], path
    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.parametrize("field", ["interval", "n_pairs", "reason", "plan", "approval", "report",
                                   "change_id", "attempt", "limitation"])
def test_edits_survive_no_rechecksum(run_scenario, field):
    """Every derived or cross-referenced field is rechecked, so rewriting
    checksums.json after an edit does not make the bundle consistent again."""
    summary, _ = run_scenario("positive")
    run_dir = summary["run_dir"]

    def rewrite(name, mutate):
        path = os.path.join(run_dir, name)
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        mutate(doc)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
            fh.write("\n")

    if field == "interval":
        rewrite("evidence.json", lambda e: e["comparison"]["criteria"][0]["delta_interval"].__setitem__(0, 0.9))
    elif field == "n_pairs":
        rewrite("evidence.json", lambda e: e["comparison"]["criteria"][0].update(n_pairs=100000))
    elif field == "reason":
        rewrite("evidence.json", lambda e: e["comparison"]["reasons"].append("guardrail_clear:invented"))
    elif field == "limitation":
        rewrite("evidence.json", lambda e: e["comparison"]["limitations"].pop(0))
    elif field == "plan":
        rewrite("manifest.json", lambda m: m["plan"]["declaration"]["run"].update(command="true"))
    elif field == "approval":
        rewrite("manifest.json", lambda m: m["approval"].update(plan_digest="0" * 64))
    elif field == "change_id":
        rewrite("evidence.json", lambda e: e["change"].update(change_id="chg_" + "0" * 16))
    elif field == "attempt":
        rewrite("evidence.json", lambda e: e["execution"]["attempts"][0].update(status="failed"))
    elif field == "report":
        with open(os.path.join(run_dir, "report.md"), "a", encoding="utf-8") as fh:
            fh.write("\nAll criteria passed with a wide margin.\n")
    bundle.write_checksums(run_dir)
    res = bundle.verify(run_dir)
    assert res["ok"] is False and res["problems"]


def test_verify_states_its_own_limits(run_scenario):
    summary, _ = run_scenario("positive")
    res = bundle.verify(summary["run_dir"])
    assert res["ok"] and "does not authenticate" in res["scope"]


def test_non_finite_numbers_in_a_bundle_are_rejected(run_scenario):
    summary, _ = run_scenario("positive")
    path = os.path.join(summary["run_dir"], "observations.jsonl")
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    lines[0] = lines[0].rsplit('"value":', 1)[0] + '"value":NaN}'
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    bundle.write_checksums(summary["run_dir"])
    assert bundle.verify(summary["run_dir"])["ok"] is False


# ── secrets and network ───────────────────────────────────────────────────

def _job(tmp_path, phase="run"):
    return Job(command="true", phase=phase, repo_dir=str(tmp_path / "repo"),
               env_dir=str(tmp_path / "env"), output_dir=str(tmp_path / "out"),
               harness_dir=str(tmp_path / "harness"), env={"OPTIMIZER": "adamw"},
               timeout_s=5, log_path=str(tmp_path / "log.txt"))


def _execution(**over):
    ex = dict(DEFAULTS["execution"], network="none", pass_env=[], limits={})
    ex.update(over)
    return ex


def test_local_backend_does_not_inherit_shell_secrets(tmp_path, monkeypatch):
    for name in ("GH_TOKEN", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "REMYXAI_API_KEY"):
        monkeypatch.setenv(name, "s3cret-value")
    env = LocalBackend(_execution(network="full")).build_env(_job(tmp_path))
    assert "s3cret-value" not in env.values()
    assert env["OPTIMIZER"] == "adamw"
    allowed = LocalBackend(_execution(network="full", pass_env=["OPENAI_API_KEY"])).build_env(_job(tmp_path))
    assert allowed["OPENAI_API_KEY"] == "s3cret-value" and "GH_TOKEN" not in allowed


def test_docker_run_phase_is_offline_readonly_and_secretless(tmp_path, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "s3cret-value")
    argv = DockerBackend(_execution(backend="docker", network="setup-only")).argv(_job(tmp_path, "run"))
    joined = " ".join(argv)
    assert argv[argv.index("--network") + 1] == "none"
    assert f"{tmp_path / 'harness'}:/work/harness:ro" in argv
    assert f"{tmp_path / 'repo'}:/work/repo:ro" in argv
    assert f"{tmp_path / 'env'}:/work/env:ro" in argv            # no state carried between seeds
    assert "HOME=/tmp/home" in argv
    assert "--cap-drop" in argv and "no-new-privileges" in argv and "--read-only" in argv
    assert "s3cret-value" not in joined and os.path.expanduser("~") + ":" not in joined


def test_docker_setup_phase_network_follows_policy(tmp_path):
    def net(policy, phase):
        argv = DockerBackend(_execution(backend="docker", network=policy)).argv(_job(tmp_path, phase))
        return argv[argv.index("--network") + 1]
    assert net("setup-only", "setup") == "bridge" and net("setup-only", "run") == "none"
    assert net("none", "setup") == "none" and net("full", "run") == "bridge"


@pytest.mark.skipif(not has_docker(), reason="docker not available")
def test_docker_candidate_cannot_modify_the_harness(run_scenario):
    """Under docker the harness is mounted read-only, so the tamper fixture's
    write fails and the measurement stands."""
    def dockerize(text):
        return text.replace("backend: local", "backend: docker").replace("trust: trusted", "trust: untrusted")
    summary, _ = run_scenario("tamper", edit=dockerize)
    assert summary["execution_status"] == "COMPLETED"
    assert summary["verdict"] == "PASS" and summary["validity"] == "VALID"


def test_package_makes_no_network_calls_except_the_documented_one():
    """No telemetry, no upload, no Remyx host. The single outbound HTTP call
    in the package is `init --pr` asking api.github.com for a PR's commits."""
    network = re.compile(r"\b(urllib\.request|http\.client|requests|httpx|aiohttp|socket)\b")
    offenders, hosts = [], []
    for name in sorted(os.listdir(SRC)) + ["backends/" + n for n in sorted(os.listdir(os.path.join(SRC, "backends")))]:
        if not name.endswith(".py"):
            continue
        with open(os.path.join(SRC, name), encoding="utf-8") as fh:
            text = fh.read()
        if network.search(text):
            offenders.append(name)
        hosts += re.findall(r"https?://[^\s\"')]+", text)
    assert offenders == ["gitops.py"]
    assert all("remyx" not in h for h in hosts), hosts


# ── untrusted output ──────────────────────────────────────────────────────

@pytest.mark.parametrize("payload", [
    '{"metrics": {"acc": NaN}}',
    '{"metrics": {"acc": Infinity}}',
    '{"observations": [{"metrics": {"acc": 1}}]}',
    '{"observations": [{"unit": "a", "metrics": {"x": 1}}, {"unit": "a", "metrics": {"x": 2}}]}',
    '{"observations": "all good"}',
    '[1, 2, 3]',
    'not json',
])
def test_malformed_results_are_rejected(tmp_path, payload):
    (tmp_path / "results.json").write_text(payload)
    with pytest.raises(collect.ResultsError):
        collect.read(str(tmp_path), "")


def test_non_numeric_values_are_dropped_and_listed(tmp_path):
    (tmp_path / "results.json").write_text('{"metrics": {"acc": 0.5, "model": "x", "ok": true, "nested": {"a": 1}}}')
    parsed = collect.read(str(tmp_path), "")
    assert parsed["metrics"] == {"acc": 0.5, "ok": 1.0}
    assert sorted(parsed["ignored"]) == ["metrics.model", "metrics.nested"]


def test_metrics_json_marker_and_precedence(tmp_path):
    out = 'noise\nMETRICS_JSON={"acc": 0.1}\nMETRICS_JSON={"acc": 0.4776}\ntrailing\n'
    assert collect.read(str(tmp_path), out)["metrics"] == {"acc": 0.4776}
    (tmp_path / "results.json").write_text('{"acc": 0.9}')
    parsed = collect.read(str(tmp_path), out)
    assert parsed["metrics"] == {"acc": 0.9} and parsed["source"] == "results.json"
    assert collect.read(str(tmp_path / "missing"), "nothing here") is None


# ── a run that did not measure never passes ───────────────────────────────

def test_exit_zero_without_results_is_incomplete(run_scenario):
    summary, _ = run_scenario("positive", edit=lambda t: t.replace(
        "command: python3 {harness}/eval.py", "command: echo all good"))
    assert summary["verdict"] == "INCOMPLETE"
    assert load_evidence(summary)["execution"]["attempts"][0]["status"] == "no_results"


def test_missing_suite_file_never_executes(run_scenario):
    summary, _ = run_scenario("positive", edit=lambda t: t.replace(
        "command: python3 {harness}/eval.py", "command: python3 {harness}/does_not_exist.py"))
    ev = load_evidence(summary)
    assert summary["execution_status"] == "PREFLIGHT_FAILED" and summary["verdict"] == "INCOMPLETE"
    assert ev["execution"]["attempts"] == []
    assert "does_not_exist.py" in summary["detail"]


def test_retries_stop_at_the_declared_limit(run_scenario):
    summary, _ = run_scenario("positive", edit=lambda t: t.replace(
        "command: python3 {harness}/eval.py", "command: exit 7").replace(
        "  timeout_s: 120", "  timeout_s: 120\n  retries: 2"))
    attempts = load_evidence(summary)["execution"]["attempts"]
    assert [a["try"] for a in attempts] == [1, 2, 3]
    assert all(a["status"] == "failed" and a["exit_code"] == 7 for a in attempts)
    assert summary["execution_status"] == "FAILED" and summary["verdict"] == "INCOMPLETE"


def test_wall_budget_stops_the_run(run_scenario):
    summary, _ = run_scenario("positive", edit=lambda t: t.replace(
        "command: python3 {harness}/eval.py", "command: sleep 2 && python3 {harness}/eval.py").replace(
        "max_wall_s: 300", "max_wall_s: 1"))
    assert summary["verdict"] == "INCOMPLETE"
    assert summary["execution_status"] in ("BUDGET_EXCEEDED", "TIMEOUT")


def test_cost_cap_stops_the_run(run_scenario):
    summary, _ = run_scenario("positive", edit=lambda t: t.replace("max_cost_usd: 5", "max_cost_usd: 0.1"))
    assert summary["execution_status"] == "BUDGET_EXCEEDED" and summary["verdict"] == "INCOMPLETE"
    assert len(load_evidence(summary)["execution"]["attempts"]) == 1


def test_cost_cap_with_no_reported_cost_stops_as_unknown(run_scenario):
    summary, _ = run_scenario("positive", edit=lambda t: t.replace(
        "command: python3 {harness}/eval.py",
        "command: \"echo 'METRICS_JSON={\\\"resolved\\\": 1}'\""))
    ev = load_evidence(summary)
    assert summary["execution_status"] == "BUDGET_EXCEEDED" and summary["verdict"] == "INCOMPLETE"
    assert "unknown" in summary["detail"] and ev["cost"]["reported_usd"] is None


def test_setup_respects_the_wall_cap(run_scenario):
    import time
    start = time.time()
    summary, _ = run_scenario("positive", edit=lambda t: t.replace(
        "run:\n", "setup:\n  - sleep 30\nrun:\n").replace("max_wall_s: 300", "max_wall_s: 2"))
    assert summary["verdict"] == "INCOMPLETE" and summary["execution_status"] == "PREFLIGHT_FAILED"
    assert time.time() - start < 20


def test_tracked_file_edit_with_unchanged_head_cannot_pass(run_scenario):
    """The workload rewrites a tracked source file. HEAD still points at the
    approved commit, but the bytes that ran are no longer that commit."""
    summary, _ = run_scenario("positive", edit=lambda t: t.replace(
        "command: python3 {harness}/eval.py",
        "command: \"python3 {harness}/eval.py && echo '# edited' >> app/ranker.py\""))
    assert summary["verdict"] == "INCOMPLETE" and summary["validity"] == "INVALID"
    assert "integrity:checkout_modified" in summary["reasons"]


def test_paths_with_spaces_work(tmp_path):
    path = demo.create(str(tmp_path / "my project dir"), "positive")
    project = os.path.dirname(path)
    plan = planmod.build(path)
    planmod.save(project, plan)
    planmod.approve(project, plan["plan_digest"], "test")
    assert runmod.Runner(path).run()["verdict"] == "PASS"


def test_cancel_does_not_signal_a_recycled_pid(run_scenario):
    summary, project = run_scenario("positive")
    status_path = os.path.join(summary["run_dir"], "status.json")
    with open(status_path, encoding="utf-8") as fh:
        status = json.load(fh)
    status.update(state="RUNNING", pid=os.getpid(), pid_started="Thu Jan  1 00:00:00 1970")
    with open(status_path, "w", encoding="utf-8") as fh:
        json.dump(status, fh)
    res = runmod.cancel(project, summary["run_id"])
    assert res["cancelled"] is False and "gone" in res["reason"]
