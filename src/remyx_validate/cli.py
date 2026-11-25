"""remyx-validate command line.

Everything runs on this machine. No command contacts a Remyx service. The
only network use is git fetching the repo you name and, for `init --pr`, one
call to api.github.com.

Exit codes: 0 ok · 2 usage or declaration error · 3 plan not approved ·
with `run --strict`: 1 FAIL, 4 INCOMPLETE or NO_SIGNAL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

from . import __version__, bundle, decisions, declaration as decl, demo as demomod, gitops
from . import plan as planmod, runner as runmod

DEFAULT_FILE = "impact.yaml"


class CliError(Exception):
    def __init__(self, message, code=2, payload=None):
        super().__init__(message)
        self.code = code
        self.payload = payload or {}


def emit(args, payload: dict, text: str) -> None:
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(text)


def _decl_path(args) -> str:
    path = os.path.abspath(args.file)
    if not os.path.isfile(path):
        raise CliError(f"{args.file} not found. Run `remyx-validate init` to create one, "
                       "or pass -f PATH.")
    return path


def _project(args) -> str:
    return decl.project_dir(os.path.abspath(args.file))


# ── rendering ─────────────────────────────────────────────────────────────

def render_plan(plan: dict, approved, drift=None) -> str:
    d, ch, sc = plan["declaration"], plan["change"], plan["scope"]
    lines = [
        f"Plan {plan['plan_digest'][:12]}   ({'APPROVED by ' + approved['approved_by'] if approved else 'not approved'})",
        "",
        f"  repo        {ch['repo']}",
        f"  baseline    {ch['baseline']['ref']}  ->  {ch['baseline']['sha'][:12]}",
        f"  candidate   {ch['candidate']['ref']}  ->  {ch['candidate']['sha'][:12]}",
    ]
    for arm in ("baseline", "candidate"):
        if d["arms"][arm]["env"]:
            lines.append(f"  {arm} env  " + " ".join(f"{k}={v}" for k, v in sorted(d["arms"][arm]["env"].items())))
    h = plan["harness"]
    lines += [
        f"  harness     {h['path']} ({h['file_count']} files, digest {h['digest'][:12]})" if h
        else "  harness     none declared (the command must exist in both commits)",
        f"  data        {d.get('data', {}).get('identity', 'not declared')}",
        "",
        "  setup       " + ("; ".join(d["setup"]) if d["setup"] else "none"),
        f"  command     {d['run']['command']}",
        f"  seeds       {d['method']['seeds']}   unit: {d['method']['unit']}   "
        f"confidence: {d['method']['confidence']}   rule: {d['method']['uncertainty']}",
        f"  attempts    {sc['attempts_min']} (up to {sc['attempts_max']} with retries), "
        f"{sc['per_attempt_timeout_s']}s each",
    ]
    lines += _protocol_lines(plan)
    lines += ["", "  criteria"]
    for c in d["criteria"]:
        bits = []
        if c["role"] == "target":
            bits.append(f"improve by >= {c['min_improvement']}" + (" (relative)" if c["relative"] else ""))
        for key, word in (("max", "<="), ("min", ">=")):
            if key in c:
                bits.append(f"candidate {word} {c[key]}")
        if "max_regression" in c:
            bits.append(f"regress by <= {c['max_regression']}" + (" (relative)" if c["relative"] else ""))
        if c.get("performance"):
            bits.append("timing" + (" (inferred)" if c.get("x-performance-inferred") else "")
                        + (f", order tolerance ±{c['order_tolerance']}"
                           if "order_tolerance" in c else ""))
        lines.append(f"    {c['role']:<9} {c['id']}: {c['aggregate']} of `{c['metric']}`, "
                     f"{c['direction']} is better" + (", " + ", ".join(bits) if bits else ""))
    v = d["validity"]
    lines += [
        "",
        f"  validity    activation: {[a['metric'] for a in v['activation']] or 'none'}   "
        f"held constant: {v['held_constant'] or 'none'}",
        f"  backend     {d['execution']['backend']} ({d['execution']['trust']} code)"
        + (f", image {d['execution']['image']}" if d["execution"]["backend"] == "docker" else ""),
        f"  network     setup: {plan['access']['network']['setup']}   run: {plan['access']['network']['run']}"
        "   (git fetch during planning is separate)",
        f"  files       {_files_line(plan['access']['filesystem'])}",
        f"  env names   workload: {plan['access']['env_names'] or 'none'}   "
        f"from host: {plan['access']['host_env_names'] or 'none'}",
        f"  budget      wall cap {sc['wall_cap_s']}s; cost cap "
        + (f"${sc['cost_cap_usd']}" if sc["cost_cap_usd"] is not None else "none")
        + " (reported-cost check, not a spending limit)" * (sc["cost_cap_usd"] is not None)
        + "; cost estimate unknown",
        f"  outputs     {sc['outputs']}",
        "",
        f"  change_id         {ch['change_id']}",
        f"  declaration_hash  {plan['declaration_hash'][:16]}",
        f"  spec_hash         {plan['spec_hash'][:16]}",
        f"  env keys          baseline {plan['env_keys']['baseline'][:12]}   "
        f"candidate {plan['env_keys']['candidate'][:12]}",
    ]
    for text in plan.get("warnings", []):
        lines += ["", "  WARNING  " + text]
    for text in plan.get("reproducibility", []):
        lines += ["", "  note     " + text]
    if drift:
        lines += ["", "  changed since the last approved plan:"]
        for row in drift[:30]:
            lines.append(f"    {row['field']}: {_short(row['approved'])} -> {_short(row['current'])}")
    return "\n".join(lines)


def _protocol_lines(plan: dict) -> list:
    d, sc = plan["declaration"], plan["scope"]
    m = d.get("measurement")
    if not m:
        return ["  order       not controlled: one order per seed, alternating by seed position; "
                "caches not reset",
                "              (no criterion is a timing metric; declare `measurement:` to "
                "control it anyway)"]
    cache = m["cache"]
    lines = [
        f"  order       {m['design']}, {m['replicates']} replicate(s), scheduler seed "
        f"{m['schedule_seed']}: {sc['attempts_note']}",
        f"  caches      {cache['profile']} profile; a new empty cache directory before every "
        "measured attempt"
        + (f"; {cache['warmup_runs']} untimed warm-up run(s) per attempt "
           f"({sc['untimed_warmup_runs_min']} in total)" if cache["profile"] == "warm" else "")
        + f"; downloaded model assets {cache['assets']}",
    ]
    if m["reset_hooks"]:
        lines.append("  reset       before every measured attempt: "
                     + "; ".join(f"{h['id']}: {h['command']}" for h in m["reset_hooks"]))
    lines.append("  schedule    " + "  ".join(
        f"{s['ordinal']}:{s['arm'][0]}/s{s['seed']}/r{s['replicate']}/{s['sequence']}"
        for s in plan["schedule"][:24])
        + (f"  … {len(plan['schedule']) - 24} more" if len(plan["schedule"]) > 24 else ""))
    lines.append("              (b = baseline, c = candidate; the full schedule is in the saved plan)")
    return lines


def _files_line(fs) -> str:
    if isinstance(fs, str):
        return fs
    return "; ".join(f"{k}: {v}" for k, v in fs.items())


def _short(value) -> str:
    text = json.dumps(value) if not isinstance(value, str) else value
    return text if len(text) <= 60 else text[:28] + "…" + text[-28:]


def render_summary(summary: dict) -> str:
    return "\n".join([
        f"run        {summary['run_id']}",
        f"execution  {summary['execution_status']}" + (f"  ({summary['detail']})" if summary.get("detail") else ""),
        f"validity   {summary['validity']}",
        f"verdict    {summary['verdict']}",
        f"reasons    {', '.join(summary['reasons']) or 'none'}",
        f"bundle     {os.path.relpath(summary['run_dir'])}  (digest {summary['bundle_digest'][:12]})",
    ])


# ── commands ──────────────────────────────────────────────────────────────

def cmd_demo(args):
    names = list(demomod.SCENARIOS) if args.all else [args.scenario]
    rows = []
    for name in names:
        dest = os.path.join(os.path.abspath(args.dir), name) if args.all else os.path.abspath(args.dir)
        path = demomod.create(dest, name)
        project = decl.project_dir(path)
        plan = planmod.build(path)
        planmod.save(project, plan)
        planmod.approve(project, plan["plan_digest"], "demo (auto-approved)")
        if not args.json and not args.all:
            print(f"Demo project: {os.path.relpath(dest)}   scenario: {name}")
            print(demomod.SCENARIOS[name]["story"])
            print()
            print(render_plan(plan, planmod.approval(project, plan["plan_digest"])))
            print("\nThe demo approves its own plan. In your project that is your step.\n")
        summary = runmod.Runner(path, log=(lambda m: None) if args.json or args.all else print).run()
        expected = demomod.SCENARIOS[name]["expect"]
        rows.append({"scenario": name, "expected": expected, "verdict": summary["verdict"],
                     "ok": summary["verdict"] == expected, "reasons": summary["reasons"],
                     "run_dir": summary["run_dir"], "run_id": summary["run_id"]})
    ok = all(r["ok"] for r in rows)
    if args.json:
        print(json.dumps({"ok": ok, "scenarios": rows}, indent=2))
    elif args.all:
        print(f"{'scenario':<10} {'expected':<11} {'verdict':<11} reasons")
        for r in rows:
            mark = "" if r["ok"] else "   <-- MISMATCH"
            print(f"{r['scenario']:<10} {r['expected']:<11} {r['verdict']:<11} {', '.join(r['reasons'])}{mark}")
    else:
        r = rows[0]
        with open(os.path.join(r["run_dir"], "report.md"), encoding="utf-8") as fh:
            print("\n" + fh.read())
        rel = os.path.relpath(os.path.abspath(args.dir))
        print(textwrap.dedent(f"""\
            Next, from {rel}/:
              remyx-validate verify                      # recheck the bundle and recompute the verdict
              remyx-validate record --decision revise --reason "latency over budget"
              # try the second candidate under the same declaration:
              sed -i 's/ref: candidate}}/ref: candidate-2}}/' impact.yaml
              remyx-validate plan      # read it, then:
              remyx-validate approve <digest> && remyx-validate run
            Other fixtures: remyx-validate demo --scenario {{{','.join(demomod.SCENARIOS)}}}"""))
    return 0 if ok else 1


INIT_TEMPLATE = """\
# Change Evidence Spec v0.1 declaration. Edit, then: remyx-validate plan
spec_version: "0.1"
objective: TODO one sentence on what this change is meant to improve
change:
  family: TODO            # e.g. optimizer, retrieval, prompt, model-migration
  hypothesis: TODO what you expect to move, and in which direction
  repo: {repo}
{pr_line}  baseline: {{ref: {baseline}}}{baseline_note}
  candidate: {{ref: {candidate}}}{candidate_note}

# Files the measurement depends on that must be identical for both arms.
# Mounted read-only under docker, digested into the plan either way.
harness:
  path: ./harness

# Runs once per arm inside its checkout; cached until the commit or these lines change.
# {{env}} is a per-arm directory outside the checkout.
setup:
  - python3 -m venv {{env}}/venv && {{env}}/venv/bin/pip install -q -e .

run:
  # Must write $VALIDATE_OUTPUT_DIR/results.json or print METRICS_JSON={{...}}.
  # The seed arrives as $VALIDATE_SEED, the arm name as $VALIDATE_ARM.
  command: {command}
  timeout_s: 1800
  retries: 0

# Per-arm environment. If the change is off by default, turn it on here for the candidate.
arms:
  baseline: {{env: {{}}}}
  candidate: {{env: {{}}}}

method:
  unit: seed              # seed: one paired unit per seed. example: per-example ids from the harness.
  seeds: [0, 1, 2, 3, 4]
  confidence: 0.95
  uncertainty: interval   # `point` judges on point estimates; use only for deterministic metrics

criteria:
  - id: TODO_primary
    metric: TODO_metric_name     # a key your harness reports
    role: target
    direction: higher
    aggregate: mean
    min_improvement: 0
  # - id: runtime
  #   metric: wall_s
  #   role: guardrail
  #   direction: lower
  #   max_regression: 0.10
  #   relative: true
  #   order_tolerance: 0.02  # run-order difference you would call unimportant (relative here)
  #   # A metric whose name or unit reads as a timing (latency, throughput, _ms,
  #   # _s, per_s, ...) is compared in both run orders by default. Say
  #   # `performance: false` if it is not a timing, `performance: true` if it is
  #   # one the name does not give away.

# Applied by default as soon as any criterion is a timing metric. Every seed
# runs both orders (baseline then candidate, and candidate then baseline),
# each attempt starts from a new empty cache directory, and the schedule is
# frozen in the plan. It doubles the measured attempts. Quality-only
# comparisons do not get it unless you uncomment this block.
# measurement:
#   design: crossover       # `single` is cheaper and reported as exploratory only
#   replicates: 1
#   schedule_seed: 0        # seeds the run order only, never the workload
#   cache:
#     profile: cold         # `warm`: reset, then `warmup_runs` untimed runs per arm
#   reset_hooks: []         # e.g. [{{id: flush-db, command: ./scripts/flush.sh}}]

validity:
  activation: []          # e.g. [{{metric: num_trainable_params}}]: must differ between arms
  held_constant: []       # e.g. [dataset_size]: must be equal between arms

{execution}

budget:
  max_wall_s: 7200
"""

INIT_EXECUTION_LOCAL = """\
# local runs the command as you, with your file and network access. Use it for
# code and setup commands you have read. docker is experimental.
execution:
  backend: local
  trust: trusted"""

INIT_EXECUTION_PR = """\
# This is someone's pull request. Its code and the setup commands above will
# run on your machine. Read the diff first, then replace `review-required`:
#   trust: trusted   + backend: local    you reviewed it; runs as your user
#   trust: untrusted + backend: docker   experimental; no network, no host env.
#                                        Not a security boundary for hostile code.
execution:
  backend: local
  trust: review-required"""

HARNESS_README = """\
# harness

Put the evaluation script and any fixed data here. Both arms run the same
files; the directory is digested into the plan, and a change after approval
stops the run.

The script must report results one of two ways:

1. write `$VALIDATE_OUTPUT_DIR/results.json`
2. print a final line `METRICS_JSON={"metric": 1.23}`

results.json, run-level numbers only:

    {"metrics": {"final_loss": 0.41, "wall_s": 12.0}}

results.json, with per-example observations (pair by `unit` id across arms):

    {"metrics": {"dataset_size": 200},
     "observations": [{"unit": "ex-0", "slice": "short", "metrics": {"correct": 1}}]}
"""


def cmd_init(args):
    target = os.path.abspath(os.path.join(args.dir, DEFAULT_FILE))
    project = os.path.dirname(target)
    if os.path.exists(target) and not args.force:
        raise CliError(f"{os.path.relpath(target)} exists; pass --force to overwrite")
    os.makedirs(project, exist_ok=True)
    repo, baseline, candidate = args.repo or ".", args.baseline or "main", args.candidate or "HEAD"
    execution = INIT_EXECUTION_LOCAL
    pr_line = baseline_note = candidate_note = ""
    info = {}
    if args.pr:
        info = gitops.pr_info(args.pr)
        repo = info["repo"]
        print(f"Fetching {repo} (first time only) …", file=sys.stderr)
        mirror = gitops.ensure_mirror(repo, project)
        head = gitops.resolve_ref(mirror, info["head_sha"])
        base_tip = gitops.resolve_ref(mirror, info["base_sha"])
        base = gitops.merge_base(mirror, base_tip, head) or base_tip
        gitops.resolve_ref(mirror, base)
        baseline, candidate = base, head
        execution = INIT_EXECUTION_PR
        pr_line = f"  pr: {info['url']}\n"
        baseline_note = f"   # merge-base with {info['base_ref']}"
        candidate_note = f"  # PR #{info['number']} head"
    text = INIT_TEMPLATE.format(repo=repo, baseline=baseline, candidate=candidate,
                                pr_line=pr_line, baseline_note=baseline_note,
                                candidate_note=candidate_note, execution=execution,
                                command=args.command or "TODO e.g. {env}/venv/bin/python {harness}/eval.py")
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(text)
    hdir = os.path.join(project, "harness")
    os.makedirs(hdir, exist_ok=True)
    readme = os.path.join(hdir, "README.md")
    if not os.path.exists(readme):
        with open(readme, "w", encoding="utf-8") as fh:
            fh.write(HARNESS_README)
    todo = ["objective and hypothesis", "criteria: name the metric your harness reports and its threshold"]
    if args.pr:
        todo.append("execution.trust: read the pull request and the setup commands, then choose "
                    "how to run it. Nothing runs until you do")
    if not args.command:
        todo.insert(1, "run.command, and an evaluation script under harness/")
    payload = {"declaration": target, "harness": hdir, "pr": info, "missing": todo}
    emit(args, payload, "\n".join(
        [f"Wrote {os.path.relpath(target)} and {os.path.relpath(hdir)}/"]
        + ([f"PR #{info['number']}: {info['title']}",
            f"  baseline  {baseline[:12]} (merge-base with {info['base_ref']})",
            f"  candidate {candidate[:12]} (PR head)"] if info else [])
        + ["Still to fill in before `plan` will accept it:"] + [f"  - {t}" for t in todo]))
    return 0


def cmd_validate(args):
    path = _decl_path(args)
    d = decl.load(path)
    payload = {"valid": True, "declaration_hash": decl.declaration_hash(d)}
    emit(args, payload, f"{args.file}: valid   declaration_hash {payload['declaration_hash'][:16]}")
    return 0


def cmd_plan(args):
    path = _decl_path(args)
    project = decl.project_dir(path)
    plan = planmod.build(path)
    planmod.save(project, plan)
    approved = planmod.approval(project, plan["plan_digest"])
    prior = planmod.latest_approved_plan(project, plan["declaration_file"])
    drift = planmod.diff(prior, plan) if prior and not approved else []
    text = render_plan(plan, approved, drift)
    if not approved:
        text += f"\n\nApprove with:  remyx-validate approve {plan['plan_digest'][:12]}"
    emit(args, {"plan": plan, "approved": bool(approved), "changed_since_approval": drift}, text)
    return 0


def cmd_approve(args):
    project = _project(args)
    full = planmod.find_saved(project, args.digest)
    rec = planmod.approve(project, full, args.approver or "")
    emit(args, rec, f"Approved plan {full[:12]} as {rec['approved_by']}.\nRun with:  remyx-validate run")
    return 0


def cmd_run(args):
    path = _decl_path(args)
    quiet = args.json
    try:
        summary = runmod.Runner(path, log=(lambda m: None) if quiet else
                                (lambda m: print(m, file=sys.stderr))).run()
    except planmod.NotApproved as exc:
        planmod.save(decl.project_dir(path), exc.plan)
        text = f"Refusing to run: {exc}."
        if exc.drift:
            text += "\nChanged since the approved plan:\n" + "\n".join(
                f"  {r['field']}: {_short(r['approved'])} -> {_short(r['current'])}" for r in exc.drift[:30])
        text += (f"\nReview with `remyx-validate plan`, then "
                 f"`remyx-validate approve {exc.plan['plan_digest'][:12]}`.")
        raise CliError(text, code=3, payload={"plan_digest": exc.plan["plan_digest"],
                                              "changed_since_approval": exc.drift})
    emit(args, summary, render_summary(summary)
         + f"\n\nReport:  {os.path.relpath(os.path.join(summary['run_dir'], 'report.md'))}")
    if args.strict:
        if summary["verdict"] == "FAIL":
            return 1
        if summary["verdict"] in ("INCOMPLETE", "NO_SIGNAL"):
            return 4
    return 0


def cmd_status(args):
    project = _project(args)
    if args.run_id:
        run_dir = runmod.find_run(project, args.run_id)
        with open(os.path.join(run_dir, "status.json"), encoding="utf-8") as fh:
            status = json.load(fh)
        emit(args, status, "\n".join(f"{k:<18} {v}" for k, v in sorted(status.items())))
        return 0
    runs = runmod.list_runs(project)
    if args.json:
        print(json.dumps({"runs": runs}, indent=2, sort_keys=True))
    elif not runs:
        print("no runs yet")
    else:
        print(f"{'run':<32} {'state':<8} {'execution':<17} {'verdict':<11} change")
        for r in runs[:args.limit]:
            print(f"{r['run_id']:<32} {r['state']:<8} {r.get('execution_status', ''):<17} "
                  f"{r.get('verdict', ''):<11} {r['change_id']}")
    return 0


def cmd_report(args):
    run_dir = runmod.find_run(_project(args), args.run_id)
    if args.json:
        print(json.dumps(bundle.load_evidence(run_dir), indent=2, sort_keys=True))
    else:
        with open(os.path.join(run_dir, "report.md"), encoding="utf-8") as fh:
            print(fh.read())
    return 0


def cmd_cancel(args):
    res = runmod.cancel(_project(args), args.run_id)
    emit(args, res, f"cancel sent to {res['run_id']}" if res["cancelled"]
         else f"{res['run_id']}: nothing to cancel ({res['reason']})")
    return 0


def cmd_record(args):
    project = _project(args)
    run_dir = runmod.find_run(project, args.run_id)
    try:
        doc = decisions.record(project, run_dir, args.decision, args.reason,
                               args.author or planmod.default_author(), args.supersedes or "")
    except ValueError as exc:
        raise CliError(str(exc))
    ev = doc["evidence"][0]
    emit(args, doc, f"Recorded {doc['decision_id']}: {doc['decision']} on {ev['run_id']} "
                    f"(verdict {ev['verdict']}) by {doc['author']}")
    return 0


def cmd_decisions(args):
    project = _project(args)
    rows = decisions.load(project)
    live = {d["decision_id"] for d in decisions.current(project)}
    if args.json:
        print(json.dumps({"decisions": rows, "current": sorted(live)}, indent=2, sort_keys=True))
    elif not rows:
        print("no decisions recorded")
    else:
        for d in rows:
            ev = d["evidence"][0]
            tag = "" if d["decision_id"] in live else "  (superseded)"
            print(f"{d['decision_id']}  {d['recorded_at']}  {d['decision']:<6} {ev['run_id']} "
                  f"[{ev['verdict']}]  {d['reason']}{tag}")
    return 0


def cmd_export(args):
    project = _project(args)
    run_dir = runmod.find_run(project, args.run_id)
    dest = os.path.abspath(args.output or (os.path.basename(run_dir) + ".tar.gz"))
    try:
        res = bundle.export(run_dir, dest, project, public=args.public, include_logs=args.include_logs)
    except bundle.ExportRefused as exc:
        raise CliError(str(exc))
    text = f"Wrote {os.path.relpath(dest)}. Nothing was uploaded.\n" + "\n".join(
        f"  {name}" for name in res["files"])
    if res["redacted"]:
        text += (f"\n{res['dropped']} other file(s) left out (logs, raw outputs, harness)."
                 f"\n{res['review']}")
    emit(args, res, text)
    return 0


def cmd_verify(args):
    target = args.path
    if not target or not os.path.exists(target):
        target = runmod.find_run(_project(args), target or "")
    res = bundle.verify(target)
    if res["ok"]:
        text = (f"OK  {res.get('run_id', '')}  consistent: files, identities and the full comparison "
                f"recomputed from the observations (verdict {res['verdict']}, digest "
                f"{res['bundle_digest'][:12]}{', redacted derivative' if res.get('redacted') else ''}).\n"
                f"Run order: {res.get('order_control', 'unknown')}.\n"
                "This checks the record against itself. It does not show who ran it or that "
                "the workload measured honestly, and it is not a reproduction.")
    else:
        text = "FAILED verification:\n" + "\n".join(f"  - {p}" for p in res["problems"])
    emit(args, res, text)
    return 0 if res["ok"] else 1


# ── parser ────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-f", "--file", default=DEFAULT_FILE, help="declaration file (default impact.yaml)")
    common.add_argument("--json", action="store_true", help="machine-readable output")

    p = argparse.ArgumentParser(prog="remyx-validate", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="Local only. No account, no upload, no telemetry. "
                                       "Run code and setup commands you trust: the local backend "
                                       "is not a sandbox and the docker backend is experimental.")
    p.add_argument("--version", action="version", version=f"remyx-validate {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, parents=[common], help=help_)
        sp.set_defaults(fn=fn)
        return sp

    sp = add("demo", cmd_demo, "run the offline CPU demo (about ten seconds)")
    sp.add_argument("--dir", default="validate-demo")
    sp.add_argument("--scenario", default=demomod.DEFAULT_SCENARIO, choices=list(demomod.SCENARIOS))
    sp.add_argument("--all", action="store_true", help="run every fixture and check its verdict")

    sp = add("init", cmd_init, "scaffold impact.yaml and harness/")
    sp.add_argument("--dir", default=".")
    sp.add_argument("--pr", help="GitHub pull request URL; pins base and head commits")
    sp.add_argument("--repo", help="git URL or local path (default .)")
    sp.add_argument("--baseline", help="baseline ref (default main)")
    sp.add_argument("--candidate", help="candidate ref (default HEAD)")
    sp.add_argument("--command", help="evaluation command")
    sp.add_argument("--force", action="store_true")

    add("validate", cmd_validate, "check the declaration against the spec")
    add("plan", cmd_plan, "resolve refs and show exactly what would run")

    sp = add("approve", cmd_approve, "approve a plan digest")
    sp.add_argument("digest", help="plan digest, 8+ characters")
    sp.add_argument("--approver")

    sp = add("run", cmd_run, "run the approved plan (as your user unless the backend is docker)")
    sp.add_argument("--strict", action="store_true", help="non-zero exit unless PASS or WARN")

    sp = add("status", cmd_status, "list runs, or show one")
    sp.add_argument("run_id", nargs="?")
    sp.add_argument("--limit", type=int, default=20)

    sp = add("report", cmd_report, "print a run's report (latest by default)")
    sp.add_argument("run_id", nargs="?", default="")

    sp = add("cancel", cmd_cancel, "stop a running run")
    sp.add_argument("run_id", nargs="?", default="")

    sp = add("record", cmd_record, "record a ship/revise/hold/drop decision")
    sp.add_argument("run_id", nargs="?", default="")
    sp.add_argument("--decision", required=True, choices=decisions.CHOICES)
    sp.add_argument("--reason", required=True)
    sp.add_argument("--author")
    sp.add_argument("--supersedes", help="decision id this one replaces")

    add("decisions", cmd_decisions, "list recorded decisions")

    sp = add("export", cmd_export, "write a bundle .tar.gz (never uploads)")
    sp.add_argument("run_id", nargs="?", default="")
    sp.add_argument("-o", "--output")
    sp.add_argument("--public", action="store_true", help="redact local paths and credential-shaped strings; drop logs")
    sp.add_argument("--include-logs", action="store_true")

    sp = add("verify", cmd_verify, "check a bundle and recompute its verdict")
    sp.add_argument("path", nargs="?", default="", help="bundle directory, .tar.gz, or run id")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except CliError as exc:
        _fail(args, str(exc), exc.code, exc.payload)
        return exc.code
    except decl.DeclarationError as exc:
        _fail(args, "The declaration is not valid:\n" + "\n".join(f"  - {p}" for p in exc.problems),
              2, {"problems": exc.problems})
        return 2
    except (planmod.PlanError, gitops.GitError, FileNotFoundError) as exc:
        _fail(args, str(exc), 2, {})
        return 2
    except KeyboardInterrupt:
        return 130


def _fail(args, message: str, code: int, payload: dict) -> None:
    if getattr(args, "json", False):
        print(json.dumps({"error": message, "exit_code": code, **payload}, indent=2, sort_keys=True))
    else:
        print(message, file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
