"""report.md: the evidence for a person. Rendered only from evidence.json."""
from __future__ import annotations

EXEC_WORDS = {"COMPLETED": "Completed", "FAILED": "Failed", "TIMEOUT": "Timed out",
              "CANCELLED": "Cancelled", "BUDGET_EXCEEDED": "Stopped at budget",
              "PREFLIGHT_FAILED": "Stopped in preflight"}
VALIDITY_WORDS = {"VALID": "valid evidence", "INVALID": "evidence not valid",
                  "NOT_ASSESSED": "validity not assessed"}
STATUS_WORDS = {"met": "target met", "not_met": "target not met", "clear": "guardrail clear",
                "breached": "guardrail breached (veto)", "uncertain": "undecided at this confidence",
                "missing": "missing", "info": "informational"}


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
                 "held_constant_missing": "held-constant metric missing"}.get(head)
        if label and label not in tags:
            tags.append(label)
    suffix = f" ({', '.join(tags)})" if tags and verdict != "PASS" else ""
    return (f"{EXEC_WORDS.get(ex['status'], ex['status'])} · "
            f"{VALIDITY_WORDS[comp['validity']['status']]} · **{verdict}**{suffix}")


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

    out += ["## Execution", "",
            f"- status: {ex['status']}; backend: {ex['backend']['backend']}; "
            f"wall time: {ex['wall_s']} s",
            f"- cost: " + (f"${evidence['cost']['reported_usd']:.4f} reported by the harness"
                           if evidence["cost"].get("reported_usd") is not None else "unknown")
            + (f"; reported-cost threshold ${evidence['cost']['cap_usd']} (checked after each "
               "attempt, not a spending limit)" if evidence["cost"].get("cap_usd") is not None else ""),
            "", "| attempt | arm | seed | try | status | seconds |", "|---|---|---:|---:|---|---:|"]
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
