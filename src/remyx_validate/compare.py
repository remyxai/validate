"""Paired comparison and the evidence-validity policy.

Four separate answers, never folded into one:

* execution status: did the workload finish (owned by the runner)
* validity: can this evidence support any claim at all
* verdict: PASS / WARN / FAIL / NO_SIGNAL / INCOMPLETE under the predeclared
  criteria
* decision: what a person chose to do, recorded elsewhere

``compare`` is a pure function of the resolved declaration, the observation
records, the execution status and the runner's integrity findings, so anyone
holding a bundle can recompute the verdict.

Verdict rules, first match wins:

1. INCOMPLETE: execution did not complete, an integrity check failed, or the
   observations are malformed (duplicates, undeclared seeds, conflicting
   slice labels).
2. NO_SIGNAL: a held-constant factor drifted or the baseline is outside its
   expected range.
3. INCOMPLETE: a gating metric, control or witness is missing, or more
   expected measurements are unpaired than allowed.
4. NO_SIGNAL: the activation witness is identical on any seed, or the arms
   are identical on every gating metric with no witness declared.
5. FAIL: a guardrail is breached or a target is not met, judged by the
   declared uncertainty rule.
6. WARN: no failure, but at least one gating criterion cannot be decided at
   the declared confidence.
7. PASS: every target met and every guardrail clear.

Coverage is judged against the declared seeds. Nothing absent is filled in.
"""
from __future__ import annotations

import zlib
from collections import defaultdict

from . import declaration as decl
from .stats import aggregate_fn, mean, paired_summary

EQ_REL_TOL = 1e-12
ARMS = ("baseline", "candidate")


def _same(a: float, b: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b)) or 1.0
    return abs(a - b) / scale <= EQ_REL_TOL


class _Index:
    """Observation records arranged for lookup, with identity problems noted.

    A record is identified by (arm, seed, unit, metric). Duplicates, seeds that
    were never declared and a unit carrying two slice labels are recorded as
    problems instead of being overwritten.
    """

    def __init__(self, observations, seeds):
        self.seeds = list(seeds)
        # unit_level[metric][arm][seed][unit] = value
        self.unit_level = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        # run_level[metric][arm][seed] = value
        self.run_level = defaultdict(lambda: defaultdict(dict))
        self.slices = {}
        self.problems = []
        seen, flagged = set(), set()
        declared = set(self.seeds)

        def problem(code, key, detail):
            if (code, key) not in flagged:
                flagged.add((code, key))
                self.problems.append({"check": "observations", "passed": False,
                                      "code": code, "detail": detail})

        for o in observations:
            arm, seed, unit, metric = o["arm"], o["seed"], o["unit"], o["metric"]
            if arm not in ARMS:
                problem("unexpected_arm", arm, f"observation for unknown arm {arm!r}")
                continue
            if seed not in declared:
                problem("unexpected_seed", seed, f"observations for seed {seed}, which the "
                                                 "declaration does not list")
                continue
            key = (arm, seed, unit, metric)
            if key in seen:
                problem("duplicate_observation", key,
                        f"more than one value for {metric!r} ({arm}, seed {seed}, unit {unit!r})")
                continue
            seen.add(key)
            if unit is None:
                self.run_level[metric][arm][seed] = o["value"]
            else:
                self.unit_level[metric][arm][seed][unit] = o["value"]
                label = o.get("slice")
                if label is not None:
                    if self.slices.setdefault(unit, label) != label:
                        problem("inconsistent_slice", unit,
                                f"unit {unit!r} is labelled with more than one slice")

    def has_unit_level(self, metric: str) -> bool:
        return any(units for arm in ARMS
                   for units in self.unit_level.get(metric, {}).get(arm, {}).values())

    def seed_value(self, metric: str, arm: str, seed: int):
        """A control or witness reading for one arm and seed: the run-level value,
        otherwise the mean over that seed's units, otherwise None."""
        run = self.run_level.get(metric, {}).get(arm, {})
        if seed in run:
            return run[seed]
        units = self.unit_level.get(metric, {}).get(arm, {}).get(seed)
        return mean(list(units.values())) if units else None


def _pair_by_seed(index: _Index, metric: str, aggregate: str) -> dict:
    """One pair per declared seed. A run-level value is used as reported. If a
    seed only has per-unit values, the two arms are first restricted to the
    units both reported, then aggregated."""
    agg = aggregate_fn(aggregate)
    base, cand, keys = [], [], []
    unit_union = unit_matched = 0
    for seed in index.seeds:
        rb = index.run_level.get(metric, {}).get("baseline", {}).get(seed)
        rc = index.run_level.get(metric, {}).get("candidate", {}).get(seed)
        if rb is not None and rc is not None:
            base.append(rb), cand.append(rc), keys.append(seed)
            continue
        if rb is not None or rc is not None:
            continue  # one arm per run, the other per unit: not comparable
        ub = index.unit_level.get(metric, {}).get("baseline", {}).get(seed, {})
        uc = index.unit_level.get(metric, {}).get("candidate", {}).get(seed, {})
        shared = sorted(set(ub) & set(uc))
        unit_union += len(set(ub) | set(uc))
        unit_matched += len(shared)
        if shared:
            base.append(agg([ub[u] for u in shared]))
            cand.append(agg([uc[u] for u in shared]))
            keys.append(seed)
    expected = len(index.seeds)
    seed_missing = (expected - len(keys)) / expected if expected else 1.0
    unit_missing = (unit_union - unit_matched) / unit_union if unit_union else 0.0
    coverage = {"basis": "declared seeds", "expected": expected, "matched": len(keys),
                "missing_rate": max(seed_missing, unit_missing)}
    if unit_union:
        coverage["units_within_seeds"] = {"expected": unit_union, "matched": unit_matched}
    return {"baseline": base, "candidate": cand, "keys": keys, "coverage": coverage,
            "pairing": "seed", "aggregate": "mean"}


def _pair_by_example(index: _Index, metric: str, aggregate: str, expected_units) -> dict:
    """One pair per example id. A (seed, example) cell counts only when both
    arms reported it; an example's value is the mean over its matched seeds, so
    both arms are always averaged over the same seeds."""
    ub = index.unit_level.get(metric, {}).get("baseline", {})
    uc = index.unit_level.get(metric, {}).get("candidate", {})
    union = set()
    for arm in (ub, uc):
        for units in arm.values():
            union |= set(units)
    matched = defaultdict(lambda: ([], []))
    cells = 0
    for seed in index.seeds:
        b, c = ub.get(seed, {}), uc.get(seed, {})
        for unit in set(b) & set(c):
            matched[unit][0].append(b[unit])
            matched[unit][1].append(c[unit])
            cells += 1
    keys = sorted(matched)
    n_units = max(len(union), expected_units or 0)
    expected = n_units * len(index.seeds)
    coverage = {
        "basis": ("declared seeds x declared expected_units" if expected_units else
                  "declared seeds x units observed in either arm"),
        "expected": expected, "matched": cells,
        "missing_rate": ((expected - cells) / expected) if expected else 1.0,
    }
    return {"baseline": [mean(matched[k][0]) for k in keys],
            "candidate": [mean(matched[k][1]) for k in keys],
            "keys": keys, "coverage": coverage, "pairing": "example", "aggregate": aggregate}


def _paired_values(index: _Index, criterion: dict, d: dict) -> dict:
    metric, aggregate = criterion["metric"], criterion["aggregate"]
    mode = criterion.get("pairing") or d["method"]["unit"]
    note = None
    if mode == "example" and not index.has_unit_level(metric):
        if criterion["role"] == "info":
            mode, note = "seed", "reported per run only, so it is paired by seed"
        else:
            out = _pair_by_example(index, metric, aggregate, None)
            out["note"] = ("the declared unit is `example` but this metric was reported per "
                           "run only. Report it per example, or set `pairing: seed` on the "
                           "criterion")
            return out
    if mode == "seed":
        out = _pair_by_seed(index, metric, aggregate)
    else:
        out = _pair_by_example(index, metric, aggregate, d.get("data", {}).get("expected_units"))
    out["note"] = note
    return out


def _fmt(x, unit=""):
    if x is None:
        return "n/a"
    return f"{x:.4g} {unit}" if unit else f"{x:.4g}"


def _evaluate_criterion(c: dict, index: _Index, d: dict, rng_base: int) -> dict:
    method = d["method"]
    sign = 1.0 if c["direction"] == "higher" else -1.0
    paired = _paired_values(index, c, d)
    base, cand, keys = paired["baseline"], paired["candidate"], paired["keys"]
    pairing, aggregate, coverage = paired["pairing"], paired["aggregate"], paired["coverage"]
    res = {
        "id": c["id"], "metric": c["metric"], "role": c["role"],
        "direction": c["direction"], "aggregate": c["aggregate"],
        "unit": c.get("unit", ""), "pairing": pairing,
        "n_pairs": len(keys), "coverage": coverage,
        "missing_rate": coverage["missing_rate"],
        "confidence": method["confidence"], "notes": [], "checks": [],
    }
    seed = rng_base + zlib.crc32(c["id"].encode("utf-8"))
    summary = paired_summary(base, cand, aggregate, method["confidence"], seed)
    for key in ("baseline", "candidate", "delta", "delta_interval",
                "baseline_interval", "candidate_interval", "estimator"):
        res[key] = summary[key]
    if summary["interval_note"]:
        res["notes"].append(summary["interval_note"])
    if paired.get("note"):
        res["notes"].append(paired["note"])
    if pairing == "seed" and c["aggregate"] != "mean" and "units_within_seeds" in coverage:
        res["notes"].append(f"{c['aggregate']} is taken within each run over the units both "
                            "arms reported; runs are then averaged and paired by seed")

    gating = c["role"] in ("target", "guardrail")
    if not keys:
        res["status"] = "missing" if gating else "info"
        res["notes"].append("metric not reported by both arms")
        return res
    if gating and res["missing_rate"] > method["max_missing_rate"]:
        res["status"] = "missing"
        res["notes"].append(
            f"{coverage['expected'] - coverage['matched']} of {coverage['expected']} expected "
            f"measurements ({coverage['basis']}) are not paired "
            f"({res['missing_rate']:.1%} > allowed {method['max_missing_rate']:.1%})")
        return res

    improvement = sign * res["delta"]
    interval = res["delta_interval"]
    imp_interval = None
    if interval:
        lo, hi = sign * interval[0], sign * interval[1]
        imp_interval = [min(lo, hi), max(lo, hi)]
    res["improvement"] = improvement
    res["improvement_interval"] = imp_interval
    res["relative_delta"] = (res["delta"] / abs(res["baseline"])) if res["baseline"] else None
    res["all_pairs_identical"] = all(_same(b, v) for b, v in zip(base, cand))
    res["slices"] = _slice_stats(index, keys, base, cand, c["aggregate"]) if pairing == "example" else {}
    if pairing == "example" and not d.get("data", {}).get("expected_units"):
        res["notes"].append("coverage is judged against the units either arm reported; a unit "
                            "missing from both arms cannot be seen. Declare `data.expected_units` "
                            "to check the count")

    if c["role"] == "info":
        res["status"] = "info"
        return res

    use_interval = method["uncertainty"] == "interval"
    scale = abs(res["baseline"]) if c.get("relative") else 1.0
    u = c.get("unit", "")
    outcomes = []

    def judge(label, ok_point, ok_sure, bad_sure):
        """ok_sure / bad_sure are None when there is no interval."""
        if not use_interval:
            state = "ok" if ok_point else "bad"
        elif ok_sure is None:
            state = "uncertain"
        elif ok_sure:
            state = "ok"
        elif bad_sure:
            state = "bad"
        else:
            state = "uncertain"
        res["checks"].append({"rule": label, "state": state})
        outcomes.append(state)

    if c["role"] == "target":
        need = c["min_improvement"] * scale
        strict = need == 0
        label = (f"improve by at least {_fmt(need, u)}" if need else "improve")
        point_ok = improvement > need if strict else improvement >= need
        if imp_interval:
            sure_ok = imp_interval[0] > need if strict else imp_interval[0] >= need
            sure_bad = imp_interval[1] < need if not strict else imp_interval[1] <= need
            judge(label, point_ok, sure_ok, sure_bad)
        else:
            judge(label, point_ok, None, None)
    else:
        ci = res["candidate_interval"]
        if "max" in c:
            judge(f"candidate at most {_fmt(c['max'], u)}", res["candidate"] <= c["max"],
                  None if not ci else ci[1] <= c["max"],
                  None if not ci else ci[0] > c["max"])
        if "min" in c:
            judge(f"candidate at least {_fmt(c['min'], u)}", res["candidate"] >= c["min"],
                  None if not ci else ci[0] >= c["min"],
                  None if not ci else ci[1] < c["min"])
        if "max_regression" in c:
            allow = c["max_regression"] * scale
            regression = -improvement
            judge(f"regress by at most {_fmt(allow, u)}", regression <= allow,
                  None if not imp_interval else -imp_interval[0] <= allow,
                  None if not imp_interval else -imp_interval[1] > allow)
    if c.get("relative"):
        res["notes"].append("relative threshold converted using the baseline point estimate")

    if "bad" in outcomes:
        res["status"] = "not_met" if c["role"] == "target" else "breached"
    elif "uncertain" in outcomes:
        res["status"] = "uncertain"
    else:
        res["status"] = "met" if c["role"] == "target" else "clear"
    return res


def _slice_stats(index, keys, base, cand, aggregate) -> dict:
    groups = defaultdict(lambda: ([], []))
    for key, b, v in zip(keys, base, cand):
        name = index.slices.get(key)
        if name is not None:
            groups[name][0].append(b)
            groups[name][1].append(v)
    agg = aggregate_fn(aggregate)
    out = {}
    for name in sorted(groups):
        b, v = groups[name]
        out[name] = {"n_pairs": len(b), "baseline": agg(b), "candidate": agg(v),
                     "delta": agg(v) - agg(b)}
    return out


def _per_seed(index: _Index, metric: str):
    """(seeds where both arms report, seeds where one is absent, values)."""
    values, absent = {}, []
    for seed in index.seeds:
        b, c = index.seed_value(metric, "baseline", seed), index.seed_value(metric, "candidate", seed)
        if b is None or c is None:
            absent.append(seed)
        else:
            values[seed] = (b, c)
    return values, absent


def _validity_checks(d: dict, index: _Index, criteria: list) -> list:
    """Controls and witnesses are judged on every declared seed. A reading
    absent from any seed is a missing control, never a passed check."""
    checks = []
    v = d["validity"]
    n = len(index.seeds)
    for name in v["held_constant"]:
        values, absent = _per_seed(index, name)
        if absent:
            checks.append({"check": "held_constant", "metric": name, "passed": False,
                           "code": "held_constant_missing",
                           "detail": f"`{name}` was declared held constant but is not reported by "
                                     f"both arms on {len(absent)} of {n} seed(s)"})
            continue
        off = [seed for seed, (b, c) in values.items() if not _same(b, c)]
        if off:
            b, c = values[off[0]]
            detail = (f"`{name}` differs between arms on {len(off)} of {n} seed(s); "
                      f"seed {off[0]}: baseline {_fmt(b)}, candidate {_fmt(c)}. "
                      "The arms were not run under the same conditions")
        else:
            detail = f"`{name}` is equal between arms on all {n} seed(s)"
        checks.append({"check": "held_constant", "metric": name, "passed": not off,
                       "code": "control_drift", "detail": detail})
    for item in v["baseline_expect"]:
        readings = [index.seed_value(item["metric"], "baseline", s) for s in index.seeds]
        if any(r is None for r in readings):
            checks.append({"check": "baseline_expect", "metric": item["metric"], "passed": False,
                           "code": "baseline_expect_missing",
                           "detail": f"baseline `{item['metric']}` is not reported on every seed"})
            continue
        got = mean(readings)
        ok = abs(got - item["value"]) <= item["tolerance"]
        checks.append({"check": "baseline_expect", "metric": item["metric"], "passed": ok,
                       "code": "baseline_drift",
                       "detail": f"baseline `{item['metric']}` is {_fmt(got)}, expected "
                                 f"{_fmt(item['value'])} ± {_fmt(item['tolerance'])}"})
    activation_ok = None
    for item in v["activation"]:
        values, absent = _per_seed(index, item["metric"])
        if absent:
            ok, code = False, "activation_missing"
            detail = (f"activation witness `{item['metric']}` is not reported by both arms on "
                      f"{len(absent)} of {n} seed(s)")
        else:
            idle = [seed for seed, (b, c) in values.items() if _same(b, c)]
            ok, code = not idle, "inactive_treatment"
            if idle:
                detail = (f"witness `{item['metric']}` is identical between arms on {len(idle)} of "
                          f"{n} seed(s), so the change did not take effect there")
            else:
                detail = f"witness `{item['metric']}` differs between arms on all {n} seed(s)"
        activation_ok = ok if activation_ok is None else (activation_ok and ok)
        checks.append({"check": "activation", "metric": item["metric"], "passed": ok,
                       "code": code, "detail": detail})

    gating = [c for c in criteria if c["role"] in ("target", "guardrail") and c.get("n_pairs")]
    if gating and all(c.get("all_pairs_identical") for c in gating):
        if activation_ok:
            checks.append({"check": "identical_arms", "passed": True, "code": "measured_zero",
                           "detail": "both arms scored identically on every gating metric and the "
                                     "activation witness shows the change was active: a measured "
                                     "zero on this data, not a proof of equivalence"})
        elif activation_ok is None:
            checks.append({"check": "identical_arms", "passed": False,
                           "code": "identical_arms_unverified",
                           "detail": "both arms scored identically on every gating metric and no "
                                     "activation witness is declared, so a zero effect cannot be told "
                                     "apart from a change that never ran. Declare "
                                     "`validity.activation` with a metric that must differ"})
    return checks


def compare(d: dict, observations: list, execution_status: str, integrity: list) -> dict:
    """Pure: same inputs, same comparison."""
    dhash = decl.declaration_hash(d)
    method = d["method"]
    index = _Index(observations, method["seeds"])
    rng_base = int(dhash[:12], 16)
    criteria = [_evaluate_criterion(c, index, d, rng_base) for c in d["criteria"]]
    reasons, limitations = [], []

    checks = list(integrity) + index.problems
    hard = [f for f in checks if not f.get("passed", False)]
    verdict = None

    if execution_status != "COMPLETED":
        verdict = "INCOMPLETE"
        reasons.append(f"execution_{execution_status.lower()}")
    if hard:
        verdict = "INCOMPLETE"
        reasons += [("integrity:" if f in integrity else "observations:") + f["code"] for f in hard]
    if verdict is None:
        # Order matters. A declared control that drifted explains unpaired
        # measurements, so it is reported first. Then anything missing: a
        # gating metric, a control or a witness. Activation is judged last, on
        # complete data.
        validity_checks = _validity_checks(d, index, criteria)
        checks += validity_checks
        failed = [c for c in validity_checks if not c["passed"]]
        drifted = [c for c in failed if c["code"] in ("control_drift", "baseline_drift")]
        absent = [c for c in failed if c["code"].endswith("_missing")]
        missing = [c for c in criteria if c["status"] == "missing"]
        if drifted:
            verdict, failed = "NO_SIGNAL", drifted
        elif missing or absent:
            verdict, failed = "INCOMPLETE", absent
            reasons += [("unpaired_measurements:" if c["n_pairs"] else "missing_metric:") + c["id"]
                        for c in missing]
        elif failed:
            verdict = "NO_SIGNAL"
        reasons += [f"{c['code']}" + (f":{c['metric']}" if c.get("metric") else "") for c in failed]

    if verdict is None:
        breached = [c for c in criteria if c["status"] == "breached"]
        not_met = [c for c in criteria if c["status"] == "not_met"]
        uncertain = [c for c in criteria if c["status"] == "uncertain"]
        if breached or not_met:
            verdict = "FAIL"
            reasons += [f"guardrail_veto:{c['id']}" for c in breached]
            reasons += [f"target_not_met:{c['id']}" for c in not_met]
        elif uncertain:
            verdict = "WARN"
            reasons += [f"uncertain:{c['id']}" for c in uncertain]
        else:
            verdict = "PASS"
            reasons += [f"target_met:{c['id']}" for c in criteria if c["status"] == "met"]
            reasons += [f"guardrail_clear:{c['id']}" for c in criteria if c["status"] == "clear"]
        if any(c.get("code") == "measured_zero" for c in checks):
            reasons.append("measured_zero")

    if verdict == "INCOMPLETE" and hard:
        validity = "INVALID"
    elif verdict == "INCOMPLETE":
        validity = "NOT_ASSESSED"
    elif verdict == "NO_SIGNAL":
        validity = "INVALID"
    else:
        validity = "VALID"

    limitations.append("Self-reported local evidence: produced and checksummed on the "
                       "machine that ran it. The checksums detect later edits; they do not "
                       "prove who ran what. This is not a remote attestation.")
    if d["execution"]["backend"] == "local":
        limitations.append("Local subprocess backend: the workload ran with the user's file "
                           "and network access and could have written its own results.")
    else:
        limitations.append("Docker backend: the harness and the candidate code share one "
                           "container, so candidate code can influence what the harness reports.")
    n_seeds = len(method["seeds"])
    if method["unit"] == "seed":
        limitations.append(f"Independent unit is the seed; {n_seeds} seed(s) were run.")
    else:
        limitations.append(f"Independent unit is the example, paired by id across arms; "
                           f"{n_seeds} seed(s). Example-level intervals do not cover "
                           "run-to-run training variance.")
    if method["uncertainty"] == "point":
        limitations.append("Declared uncertainty rule is `point`: criteria were judged on point "
                           "estimates with no interval.")
    if sum(1 for c in criteria if c["role"] != "info") > 1:
        limitations.append("Each interval is computed per criterion at the declared confidence. "
                           "They are not a joint statement over all criteria.")
    for c in criteria:
        for note in c["notes"]:
            limitations.append(f"{c['id']}: {note}")
    limitations.append("An offline comparison on the declared data. It does not predict a "
                       "production outcome.")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "validity": {"status": validity, "checks": checks},
        "criteria": criteria,
        "limitations": limitations,
        "inputs": {"execution_status": execution_status, "integrity": integrity},
    }
