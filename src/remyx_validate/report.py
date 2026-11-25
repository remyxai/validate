"""report.md: the evidence for a person. Rendered only from evidence.json."""
from __future__ import annotations

EXEC_WORDS = {"COMPLETED": "Completed", "FAILED": "Failed", "TIMEOUT": "Timed out",
              "CANCELLED": "Cancelled", "BUDGET_EXCEEDED": "Stopped at budget",
              "PREFLIGHT_FAILED": "Stopped in preflight"}
VALIDITY_WORDS = {"VALID": "valid evidence", "INVALID": "evidence not valid",
                  "NOT_ASSESSED": "validity not assessed"}
STATUS_WORDS = {"met": "target met", "not_met": "target not met", "clear": "guardrail clear",
                "breached": "guardrail breached (veto)", "uncertain": "undecided at this confidence",
                "missing": "missing", "info": "informational",
                "exploratory": "exploratory only (run order or shared state not controlled)",
                "unassessed": "not assessed (run order moved it more than the tolerance)"}
ORDER_WORDS = {"within_tolerance": "inside tolerance", "exceeds_tolerance": "OUTSIDE tolerance",
               "unresolved": "not resolved", "not_gated": "reported, not gated",
               "not_assessed": "not assessed"}


def num(x, unit="") -> str:
    if x is None:
        return "n/a"
    if abs(x) >= 1000 or (x == int(x) and abs(x) >= 10):
        text = f"{x:,.1f}".rstrip("0").rstrip(".")
    else:
        text = f"{x:.4g}"
    return f"{text}{(' ' + unit) if unit else ''}"


def signed(x, unit="") -> str:
    if x is None:
        return "n/a"
    return ("+" if x >= 0 else "−") + num(abs(x), unit)


def interval(iv, unit="") -> str:
    if not iv:
        return "no interval"
    return f"[{num(iv[0])}, {num(iv[1])}]{(' ' + unit) if unit else ''}"


def headline(evidence: dict) -> str:
    comp, ex = evidence["comparison"], evidence["execution"]
    verdict = comp["verdict"]
    tags = []
    for r in comp["reasons"]:
        head = r.split(":")[0]
        label = {"guardrail_veto": "guardrail veto", "target_not_met": "target not met",
                 "control_drift": "control drift", "baseline_drift": "baseline drift",
                 "inactive_treatment": "inactive treatment",
                 "identical_arms_unverified": "identical arms, activation unverified",
                 "measured_zero": "measured zero", "uncertain": "underpowered",
                 "missing_metric": "missing metric",
                 "unpaired_measurements": "unpaired measurements",
                 "observations": "malformed observations",
                 "activation_missing": "activation witness missing",
                 "baseline_expect_missing": "baseline reading missing", "integrity": "integrity check failed",
                 "held_constant_missing": "held-constant metric missing",
                 "order_bias": "run-order effect outside tolerance",
                 "order_effect_unresolved": "order effect unresolved",
                 "exploratory": "exploratory performance evidence"}.get(head)
        if label and label not in tags:
            tags.append(label)
    suffix = f" ({', '.join(tags)})" if tags and verdict != "PASS" else ""
    return (f"{EXEC_WORDS.get(ex['status'], ex['status'])} · "
            f"{VALIDITY_WORDS[comp['validity']['status']]} · **{verdict}**{suffix}")


def _protocol_section(evidence: dict) -> list:
    """Run order and cache conditions. Evidence written before protocols
    existed has no `protocol` record and renders exactly as it did."""
    ex, comp = evidence["execution"], evidence["comparison"]
    p = ex.get("protocol")
    if not p:
        return []
    out = ["## Run order and caches", ""]
    if p["cache_isolation"] == "none":
        return out + [f"**Not order-controlled.** {p['note']}", ""]
    crossover = p["design"] == "crossover"
    out += [
        ("- design: crossover. Every seed and replicate ran both orders, so each arm was measured "
         "in first and in second position on the same seed" if crossover else
         "- design: single. Each seed ran in one order only. **Not order-controlled**; "
         "performance results are exploratory"),
        f"- {p['replicates']} replicate(s), {p['attempts_per_seed']} measured attempts per seed; "
        f"order drawn from scheduler seed {p['schedule_seed']}, independent of the workload seeds, "
        "and fixed in the approved plan",
        f"- cache profile: {p['cache_profile']}. Each measured attempt ran in a new process with a "
        "new, empty directory for: " + ", ".join(f"`{n}`" for n in p["isolated_layers"]),
    ]
    if p["cache_profile"] == "warm":
        out.append(f"- warm-up: {p['warmup_runs']} untimed run(s) of the same command per attempt, "
                   "for that attempt's own arm, after the reset. Not included in any metric")
    if p["reset_hooks"]:
        out.append("- reset hooks before every measured attempt: "
                   + ", ".join(f"`{h}`" for h in p["reset_hooks"]))
    out.append("- downloaded model assets: " + (
        "kept shared so they are not downloaded inside a timed attempt ("
        + ", ".join(f"`{k}`" for k in sorted(p["shared_immutable_assets"])) + ")"
        if p["shared_immutable_assets"] else
        "not shared with the attempts" if p["assets"] == "isolated" else
        "none passed by this backend"))
    out.append(f"- retries: {p['retry_policy']}")
    if p["declared_uncontrolled"]:
        out.append("- **declared as not reset:** " + ", ".join(p["declared_uncontrolled"]))
    out.append("- not controlled by the runner: " + "; ".join(p["not_controlled"]))
    host = p.get("host")
    if host:
        gpus = ", ".join(f"{g['name']} (driver {g['driver']})" for g in host["gpus"]) or "none detected"
        out.append(f"- machine (self-reported): {host['system']} {host['machine']}, "
                   f"{host['cpu_model'] or 'unknown CPU'}, {host['cpu_count']} logical CPUs; GPUs: {gpus}")
        out.append(f"- {p['concurrency']}; {p['timing']}")
    out.append("")
    ordered = [c for c in comp["criteria"] if c.get("order", {}).get("design") == "crossover"]
    if ordered:
        conf = int(round(evidence["declaration"]["resolved"]["method"]["confidence"] * 100))
        out += ["Change measured in each order (candidate minus baseline). AB is the baseline "
                "first; BA is the candidate first. The order effect is the AB value minus the BA "
                "value. First-position is the candidate when it ran first minus the baseline "
                "when it ran first.", "",
                f"| criterion | AB | BA | order effect | {conf}% interval | tolerance | result | "
                "first-position | second-position | drift | units |",
                "|---|---:|---:|---:|---|---:|---|---:|---:|---|---:|"]
        for c in ordered:
            o, u = c["order"], c.get("unit", "")
            tol = "none" if o["tolerance"] is None else "±" + num(o["tolerance"], u)
            out.append(f"| {c['id']} | {signed(o['delta_ab'], u)} | {signed(o['delta_ba'], u)} | "
                       f"{signed(o['sensitivity'], u)} | {interval(o['sensitivity_interval'], u)} | "
                       f"{tol} | {ORDER_WORDS[o['state']]} | {signed(o['first_position_delta'], u)} | "
                       f"{signed(o['second_position_delta'], u)} | "
                       + ("n/a" if o["drift_between_sequences"] is None else
                          f"{signed(o['drift_between_sequences'], u)} {interval(o['drift_interval'])}")
                       + f" | {o['n_units']} |")
        out += ["", "Drift is how much the metric moved between the first and the second pair of "
                "runs on a seed, both arms averaged. It is time passing, not the change, and is "
                "reported without a gate.", ""]
    return out


def render(evidence: dict) -> str:
    comp, ex, ch = evidence["comparison"], evidence["execution"], evidence["change"]
    d = evidence["declaration"]["resolved"]
    conf = int(round(d["method"]["confidence"] * 100))
    out = ["# Change evidence", "", headline(evidence), ""]
    if d.get("objective"):
        out += [f"**Objective.** {d['objective']}", ""]
    if ch.get("hypothesis"):
        out += [f"**Hypothesis.** {ch['hypothesis']}", ""]
    if ex.get("detail"):
        out += [f"**Execution note.** {ex['detail']}", ""]
    out += ["| | ref | commit |", "|---|---|---|",
            f"| baseline | `{ch['baseline']['ref']}` | `{ch['baseline']['sha'][:12]}` |",
            f"| candidate | `{ch['candidate']['ref']}` | `{ch['candidate']['sha'][:12]}` |", ""]
    for arm in ("baseline", "candidate"):
        env = d["arms"][arm]["env"]
        if env:
            out.append(f"- {arm} arm env: " + ", ".join(f"`{k}={v}`" for k, v in sorted(env.items())))
    if any(d["arms"][a]["env"] for a in d["arms"]):
        out.append("")

    out += ["## Criteria", "",
            f"| criterion | role | baseline | candidate | delta | {conf}% interval | pairs (matched/expected) | result |",
            "|---|---|---:|---:|---:|---|---:|---|"]
    for c in comp["criteria"]:
        u = c.get("unit", "")
        label = f"{c['id']} (`{c['metric']}`, {c['aggregate']}, {c['direction']} is better)"
        out.append(f"| {label} | {c['role']} | {num(c.get('baseline'), u)} | "
                   f"{num(c.get('candidate'), u)} | {signed(c.get('delta'), u)} | "
                   f"{interval(c.get('delta_interval'), u)} | {c['n_pairs']} "
                   f"({c['coverage']['matched']}/{c['coverage']['expected']}) | "
                   f"{STATUS_WORDS.get(c['status'], c['status'])} |")
    out.append("")
    for c in comp["criteria"]:
        for chk in c.get("checks", []):
            word = {"ok": "satisfied", "bad": "violated", "uncertain": "undecided"}[chk["state"]]
            extra = ""
            if c["role"] == "guardrail" and c.get("candidate_interval") and "candidate at" in chk["rule"]:
                extra = f"; candidate interval {interval(c['candidate_interval'], c.get('unit', ''))}"
            out.append(f"- `{c['id']}`: rule \"{chk['rule']}\" {word}{extra}")
    out.append("")

    sliced = [c for c in comp["criteria"] if c.get("slices")]
    if sliced:
        out += ["## Slices", "", "Point estimates only; slices are not gated.", "",
                "| criterion | slice | pairs | baseline | candidate | delta |", "|---|---|---:|---:|---:|---:|"]
        for c in sliced:
            for name, s in c["slices"].items():
                u = c.get("unit", "")
                out.append(f"| {c['id']} | {name} | {s['n_pairs']} | {num(s['baseline'], u)} | "
                           f"{num(s['candidate'], u)} | {signed(s['delta'], u)} |")
        out.append("")

    out += ["## Validity", ""]
    checks = comp["validity"]["checks"]
    if not checks:
        out.append("No validity checks ran." if comp["verdict"] == "INCOMPLETE"
                   else "No validity checks declared.")
    for chk in checks:
        out.append(f"- {'pass' if chk['passed'] else 'FAIL'} · {chk['check']}: {chk['detail']}")
    out.append("")

    out += _protocol_section(evidence)
    out += ["## Execution", "",
            f"- status: {ex['status']}; backend: {ex['backend']['backend']}; "
            f"wall time: {ex['wall_s']} s",
            f"- cost: " + (f"${evidence['cost']['reported_usd']:.4f} reported by the harness"
                           if evidence["cost"].get("reported_usd") is not None else "unknown")
            + (f"; reported-cost threshold ${evidence['cost']['cap_usd']} (checked after each "
               "attempt, not a spending limit)" if evidence["cost"].get("cap_usd") is not None else ""),
            ""]
    if ex.get("schedule") is not None:
        out += ["Attempts in the order they ran. `planned` is the position in the approved "
                "schedule. A discarded attempt belongs to a sequence that was run again and is "
                "not part of the comparison.", "",
                "| ran | planned | attempt | arm | seed | replicate | order | position | try | "
                "status | cache | warm-up | load before | seconds |",
                "|---:|---:|---|---|---:|---:|---|---:|---:|---|---|---:|---:|---:|"]
        for a in ex["attempts"]:
            iso = a.get("isolation") or {}
            cache = "new, empty" if iso.get("fresh") else "NOT clean"
            if iso.get("cleanup") not in (None, "removed"):
                cache += f"; cleanup {iso['cleanup']}"
            status = a["status"] + (" (discarded)" if a.get("discarded") else "")
            out.append(f"| {a['ordinal']} | {a['planned_ordinal']} | `{a['attempt_id']}` | {a['arm']} | "
                       f"{a['seed']} | {a['replicate']} | {a['sequence']} | {a['period']} | {a['try']} | "
                       f"{status} | {cache} | {len(iso.get('warmup', []))} | "
                       f"{(a.get('conditions') or {}).get('before', {}).get('load_1m', 'n/a')} | "
                       f"{a['duration_s']} |")
    else:
        out += ["| attempt | arm | seed | try | status | seconds |", "|---|---|---:|---:|---|---:|"]
        for a in ex["attempts"]:
            out.append(f"| `{a['attempt_id']}` | {a['arm']} | {a['seed']} | {a['try']} | "
                       f"{a['status']} | {a['duration_s']} |")
    for a in ex["attempts"]:
        if a.get("error"):
            out.append(f"\n`{a['attempt_id']}`: {a['error']}")
    out.append("")

    out += ["## Limitations", ""] + [f"- {text}" for text in comp["limitations"]] + [""]
    out += ["## Identities", "",
            f"- change_id: `{ch['change_id']}`",
            f"- declaration_hash: `{evidence['declaration']['declaration_hash']}`",
            f"- spec_hash: `{ex['spec_hash']}`",
            f"- plan_digest: `{ex['plan_digest']}`",
            f"- run_id: `{ex['run_id']}`",
            f"- approved by: {evidence['provenance'].get('approved_by', '')} "
            f"at {evidence['provenance'].get('approved_at', '')}", "",
            "## Decision", "",
            "The verdict is not a decision. Record one with "
            f"`remyx-validate record {ex['run_id']} --decision ship|revise|hold|drop --reason \"…\"`.", ""]
    return "\n".join(out)
