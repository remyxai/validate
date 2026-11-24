"""Plan approval, bundles, decisions and the CLI."""
import json
import os
import tarfile

import pytest

from remyx_validate import bundle, decisions, declaration as decl, demo, plan as planmod
from conftest import load_evidence


def test_bundle_contents_and_schemas(run_scenario):
    summary, _ = run_scenario("guardrail")
    run_dir = summary["run_dir"]
    for name in ("evidence.json", "report.md", "manifest.json", "declaration.json",
                 "observations.jsonl", "checksums.json", "status.json"):
        assert os.path.isfile(os.path.join(run_dir, name)), name
    ev = load_evidence(summary)
    assert decl.schema_problems(ev, "evidence") == []
    for row in bundle.load_observations(run_dir)[:50]:
        assert decl.schema_problems(row, "observation") == []
    assert ev["provenance"]["trust_level"] == "self_reported_local"
    assert {a["attempt_id"] for a in ev["execution"]["attempts"]} == {"att_baseline_s0_t1", "att_candidate_s0_t1"}
    report = open(os.path.join(run_dir, "report.md"), encoding="utf-8").read()
    assert "FAIL" in report and "guardrail veto" in report and "Limitations" in report


def test_failed_and_null_runs_still_leave_a_bundle(run_scenario):
    for name in ("timeout", "null", "inactive"):
        summary, _ = run_scenario(name)
        assert bundle.verify(summary["run_dir"])["ok"]


def test_decision_is_separate_and_supersedable(run_scenario):
    summary, project = run_scenario("guardrail")
    first = decisions.record(project, summary["run_dir"], "hold", "waiting on latency numbers", "a@example.invalid")
    second = decisions.record(project, summary["run_dir"], "revise", "p95 over budget", "a@example.invalid",
                              supersedes=first["decision_id"])
    assert decl.schema_problems(second, "decision") == []
    assert [d["decision_id"] for d in decisions.current(project)] == [second["decision_id"]]
    assert len(decisions.load(project)) == 2
    assert second["evidence"][0]["verdict"] == "FAIL"
    assert bundle.verify(summary["run_dir"])["ok"]          # recording does not touch the bundle
    with pytest.raises(ValueError):
        decisions.record(project, summary["run_dir"], "ship", "  ", "a@example.invalid")
    with pytest.raises(ValueError):
        decisions.record(project, summary["run_dir"], "ship", "x", "a@example.invalid", supersedes="dec_missing")


def test_public_export_redacts_and_still_verifies(run_scenario, tmp_path):
    summary, project = run_scenario("guardrail")
    log = os.path.join(summary["run_dir"], "attempts", "att_candidate_s0_t1", "log.txt")
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"token=ghp_{'a' * 36}\nwrote {project}/tmp\nmailto someone@example.com\n")
    bundle.write_checksums(summary["run_dir"])
    dest = str(tmp_path / "public.tar.gz")
    res = bundle.export(summary["run_dir"], dest, project, public=True, include_logs=True)
    assert res["redacted"]
    with tarfile.open(dest) as tar:
        blob = b"".join(tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()).decode()
    assert "ghp_" not in blob and project not in blob and os.path.expanduser("~") not in blob
    assert "<REDACTED>" in blob and "<PROJECT>" in blob
    assert "someone@example.com" not in blob and "<EMAIL>" in blob
    assert bundle.verify(dest)["ok"]
    no_logs = str(tmp_path / "nologs.tar.gz")
    bundle.export(summary["run_dir"], no_logs, project, public=True)
    with tarfile.open(no_logs) as tar:
        assert not any("attempts/" in m.name for m in tar.getmembers())


def test_public_export_refuses_to_alter_hashed_content(run_scenario, tmp_path):
    """An email in the objective is part of the declaration hash. Redacting it
    would produce evidence that was never approved or run, so export refuses."""
    summary, project = run_scenario("positive", edit=lambda t: t.replace(
        "objective: Resolve", "objective: Ask owner@example.com. Resolve"))
    with pytest.raises(bundle.ExportRefused) as exc:
        bundle.export(summary["run_dir"], str(tmp_path / "x.tar.gz"), project, public=True)
    assert "declaration.json" in str(exc.value)
    assert not (tmp_path / "x.tar.gz").exists()
    assert bundle.verify(summary["run_dir"])["ok"]            # the source bundle is untouched


def test_public_export_is_an_allowlist(run_scenario, tmp_path):
    summary, project = run_scenario("positive")
    stray = os.path.join(summary["run_dir"], "attempts", "att_candidate_s0_t1", "output", "weights.bin")
    with open(stray, "wb") as fh:
        fh.write(bytes(range(256)))
    bundle.write_checksums(summary["run_dir"])
    dest = str(tmp_path / "public.tar.gz")
    res = bundle.export(summary["run_dir"], dest, project, public=True, include_logs=True)
    with tarfile.open(dest) as tar:
        names = {m.name.split("/", 1)[1] for m in tar.getmembers() if m.isfile()}
    assert {"evidence.json", "manifest.json", "declaration.json", "observations.jsonl",
            "report.md", "checksums.json"} <= names
    assert not any(n.endswith(("weights.bin", "results.json")) or n.startswith("harness/") for n in names)
    assert res["dropped"] > 0 and "Read every file" in res["review"]


def test_private_export_round_trips_with_decisions(run_scenario, tmp_path):
    summary, project = run_scenario("guardrail")
    decisions.record(project, summary["run_dir"], "revise", "p95 over budget", "a@example.invalid")
    dest = str(tmp_path / "private.tar.gz")
    bundle.export(summary["run_dir"], dest, project)
    res = bundle.verify(dest)
    assert res["ok"], res["problems"]
    with tarfile.open(dest) as tar:
        assert any(m.name.endswith("decisions.jsonl") for m in tar.getmembers())


def test_init_for_a_pull_request_requires_a_trust_choice(tmp_path):
    """The PR scaffold cannot be planned until a person chooses how to run it."""
    from remyx_validate import cli
    text = cli.INIT_TEMPLATE.format(
        repo="https://github.com/example/repo", baseline="a" * 40, candidate="b" * 40,
        pr_line="", baseline_note="", candidate_note="", execution=cli.INIT_EXECUTION_PR,
        command="python3 {harness}/eval.py").replace("TODO_primary", "primary").replace(
        "TODO_metric_name", "accuracy")
    path = tmp_path / "impact.yaml"
    path.write_text(text)
    with pytest.raises(decl.DeclarationError) as exc:
        decl.load(str(path))
    assert any("not chosen how to run" in p for p in exc.value.problems)
    path.write_text(text.replace("trust: review-required", "trust: trusted"))
    assert decl.load(str(path))["execution"]["trust"] == "trusted"


def test_plan_shows_access_by_name_and_never_values(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_PROVIDER_KEY", "s3cret-value")
    path = demo.create(str(tmp_path / "p"), "positive")
    with open(path, encoding="utf-8") as fh:
        text = fh.read().replace("  trust: trusted", "  trust: trusted\n  pass_env: [MY_PROVIDER_KEY]")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    plan = planmod.build(path)
    blob = json.dumps(plan)
    assert "MY_PROVIDER_KEY" in plan["access"]["host_env_names"] and "s3cret-value" not in blob
    assert any("MY_PROVIDER_KEY" in w for w in plan["warnings"])
    assert any("not a hard spending limit" in w for w in plan["warnings"])


def test_plan_diff_names_the_changed_field(tmp_path):
    path = demo.create(str(tmp_path / "p"), "guardrail")
    before = planmod.build(path)
    with open(path, encoding="utf-8") as fh:
        text = fh.read().replace("max: 800 ", "max: 950 ")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    after = planmod.build(path)
    fields = {row["field"] for row in planmod.diff(before, after)}
    assert fields == {"declaration.criteria[p95_latency].max", "declaration_hash"}
    assert before["env_keys"] == after["env_keys"] and before["spec_hash"] == after["spec_hash"]


def test_cli_flow_with_json_output(cli, tmp_path):
    project = str(tmp_path / "proj")
    demo.create(project, "positive")
    env = dict(os.environ)

    refused = cli("run", "--json", cwd=project, env=env)
    assert refused.returncode == 3 and "plan_digest" in json.loads(refused.stdout)

    plan = json.loads(cli("plan", "--json", cwd=project, env=env).stdout)
    assert plan["approved"] is False
    digest = plan["plan"]["plan_digest"]
    assert cli("approve", digest[:12], "--json", cwd=project, env=env).returncode == 0

    run = cli("run", "--json", "--strict", cwd=project, env=env)
    summary = json.loads(run.stdout)
    assert run.returncode == 0 and summary["verdict"] == "PASS"

    assert json.loads(cli("verify", "--json", cwd=project, env=env).stdout)["ok"] is True
    rec = cli("record", "--decision", "ship", "--reason", "meets both criteria", "--json", cwd=project, env=env)
    assert json.loads(rec.stdout)["decision"] == "ship"
    status = json.loads(cli("status", "--json", cwd=project, env=env).stdout)
    assert status["runs"][0]["verdict"] == "PASS"
    assert "target met" in cli("report", cwd=project, env=env).stdout


def test_cli_strict_exit_codes(cli, tmp_path):
    out = str(tmp_path / "demo")
    assert cli("demo", "--dir", out, "--scenario", "guardrail", "--json").returncode == 0
    assert cli("run", "--strict", "--json", cwd=out).returncode == 1          # FAIL
    out2 = str(tmp_path / "demo2")
    cli("demo", "--dir", out2, "--scenario", "inactive", "--json")
    assert cli("run", "--strict", "--json", cwd=out2).returncode == 4         # NO_SIGNAL


def test_cli_reports_every_declaration_problem(cli, tmp_path):
    project = str(tmp_path / "proj")
    assert cli("init", "--dir", project, "--repo", ".", "--json").returncode == 0
    res = cli("plan", "--json", cwd=project)
    problems = json.loads(res.stdout)["problems"]
    assert res.returncode == 2
    assert any("placeholder" in p for p in problems)


def test_init_from_local_refs(cli, tmp_path):
    project = str(tmp_path / "proj")
    res = cli("init", "--dir", project, "--repo", "../app", "--baseline", "main", "--candidate", "feature",
              "--command", "python3 {harness}/eval.py")
    assert res.returncode == 0
    text = open(os.path.join(project, "impact.yaml"), encoding="utf-8").read()
    assert "repo: ../app" in text and "ref: feature" in text and "python3 {harness}/eval.py" in text
    assert os.path.isfile(os.path.join(project, "harness", "README.md"))
