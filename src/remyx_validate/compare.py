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
   declared uncertainty rule. A performance criterion whose order effect is
   outside its tolerance is unassessed and is never the cause of a FAIL.
   Then NO_SIGNAL if nothing failed and a gating performance criterion is
   unassessed for that reason.
6. WARN: no failure, but at least one gating criterion cannot be decided at
   the declared confidence.
7. PASS: every target met and every guardrail clear.

Coverage is judged against the declared seeds. Nothing absent is filled in.

Under a declared measurement protocol each seed is measured more than once:
once per replicate and, in the crossover design, once per run order (AB with
the baseline first, BA with the candidate first). Those runs are nested in
the seed. They are averaged within each independent unit before anything is
estimated, so they never count as extra units. A unit missing from any of its
runs is left out, because an unbalanced average would let run order back in.

For a criterion marked ``performance`` the crossover also yields the
treatment effect measured in each order. Their difference is the order
effect. A gating performance criterion declares a tolerance for it in
advance. An order effect confidently outside the tolerance leaves that
criterion unassessed: it cannot pass and it cannot fail. Other criteria are
still judged, so a quality guardrail that is breached is still a FAIL; with
nothing else failing the verdict is NO_SIGNAL. An order effect that cannot be
shown to be inside the tolerance keeps the criterion undecided. A difference
that is merely not significant is never read as no order effect.
"""
from __future__ import annotations

import zlib
from collections import defaultdict

from . import declaration as decl, schedule
from .stats import aggregate_fn, contrast_summary, mean, paired_summary

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


class _Strata:
    """The observations split into the runs each seed was measured in.

    With no measurement protocol there is one stratum and everything below
    behaves exactly as it did before the protocol existed. With one, a
    stratum is a (replicate, sequence) for the crossover design and a
    replicate for the single design.
    """

    def __init__(self, observations, d):
        self.seeds = list(d["method"]["seeds"])
        m = d.get("measurement")
        self.measured = bool(m)
        self.problems, self.slices = [], {}
        if not m:
            self.labels = [None]
            self.parts = {None: _Index(observations, self.seeds)}
        else:
            self.crossover = m["design"] == "crossover"
            orders = ("AB", "BA") if self.crossover else (None,)
            self.replicates = m["replicates"]
            self.labels = [(r, q) for r in range(self.replicates) for q in orders]
            rows = {label: [] for label in self.labels}
            stray = 0
            for o in observations:
                label = (o.get("replicate"), o.get("sequence") if self.crossover else None)
                if label in rows and o.get("sequence") in ("AB", "BA"):
                    rows[label].append(o)
                else:
                    stray += 1
            self.parts = {label: _Index(rows[label], self.seeds) for label in self.labels}
            if stray:
                self.problems.append({
                    "check": "observations", "passed": False, "code": "unscheduled_observation",
                    "detail": f"{stray} observation(s) carry no replicate and run order that the "
                              "declared measurement protocol schedules"})
        seen = set()
        for label in self.labels:
            part = self.parts[label]
            for problem in part.problems:
                if (problem["code"], problem["detail"]) not in seen:
                    seen.add((problem["code"], problem["detail"]))
                    self.problems.append(problem)
            for unit, name in part.slices.items():
                self.slices.setdefault(unit, name)

    def has_unit_level(self, metric: str) -> bool:
        return any(part.has_unit_level(metric) for part in self.parts.values())

    def cells(self, metric: str):
        """Per (seed, run): both arms' control or witness reading. Returns
        ({cell: (baseline, candidate)}, [cells where an arm is absent])."""
        values, absent = {}, []
        for label in self.labels:
            part = self.parts[label]
            for seed in self.seeds:
                cell = seed if label is None else (seed,) + label
                b = part.seed_value(metric, "baseline", seed)
                c = part.seed_value(metric, "candidate", seed)
                if b is None or c is None:
                    absent.append(cell)
                else:
                    values[cell] = (b, c)
        return values, absent

    def where(self, cell) -> str:
        if not self.measured:
            return f"seed {cell}"
        seed, replicate, order = cell
        return f"seed {seed} replicate {replicate}" + (f" order {order}" if order else "")

    @property
    def noun(self) -> str:
        return "measured pair(s)" if self.measured else "seed(s)"


def _merge(strata: _Strata, pair) -> dict:
    """Pair within each run, then average each unit over its runs. Only units
    present in every run are kept, so each arm is averaged over the same
    runs and, in a crossover, over both positions equally."""
    if not strata.measured:
        return pair(strata.parts[None])
    outs = {label: pair(strata.parts[label]) for label in strata.labels}
    cells = {label: {k: (b, c) for k, b, c in zip(o["keys"], o["baseline"], o["candidate"])}
             for label, o in outs.items()}
    union = set()
    for found in cells.values():
        union |= set(found)
    keys = sorted(k for k in union if all(k in cells[label] for label in strata.labels))
    first = outs[strata.labels[0]]
    expected = sum(o["coverage"]["expected"] for o in outs.values())
    matched = sum(o["coverage"]["matched"] for o in outs.values())
    excluded = len(union) - len(keys)
    rates = [o["coverage"]["missing_rate"] for o in outs.values()]
    rates.append((expected - matched) / expected if expected else 1.0)
    rates.append(excluded / len(union) if union else 0.0)
    coverage = {"basis": f"{first['coverage']['basis']} x {len(strata.labels)} measured run(s) per seed",
                "expected": expected, "matched": matched, "missing_rate": max(rates),
                "complete_units": len(keys), "incomplete_units": excluded}
    within = [o["coverage"]["units_within_seeds"] for o in outs.values()
              if "units_within_seeds" in o["coverage"]]
    if within:
        coverage["units_within_seeds"] = {"expected": sum(w["expected"] for w in within),
                                          "matched": sum(w["matched"] for w in within)}
    return {"baseline": [mean([cells[label][k][0] for label in strata.labels]) for k in keys],
            "candidate": [mean([cells[label][k][1] for label in strata.labels]) for k in keys],
            "keys": keys, "coverage": coverage, "pairing": first["pairing"],
            "aggregate": first["aggregate"], "cells": cells}


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


def _paired_values(index: _Strata, criterion: dict, d: dict) -> dict:
    metric, aggregate = criterion["metric"], criterion["aggregate"]
    mode = criterion.get("pairing") or d["method"]["unit"]
    note = None
    if mode == "example" and not index.has_unit_level(metric):
        if criterion["role"] == "info":
            mode, note = "seed", "reported per run only, so it is paired by seed"
        else:
            out = _merge(index, lambda part: _pair_by_example(part, metric, aggregate, None))
            out["note"] = ("the declared unit is `example` but this metric was reported per "
                           "run only. Report it per example, or set `pairing: seed` on the "
                           "criterion")
            return out
    if mode == "seed":
        out = _merge(index, lambda part: _pair_by_seed(part, metric, aggregate))
    else:
        expected_units = d.get("data", {}).get("expected_units")
        out = _merge(index, lambda part: _pair_by_example(part, metric, aggregate, expected_units))
    out["note"] = note
    return out


ORDER_EXPLAINED = ("AB is the baseline first and the candidate second; BA is the reverse. "
                   "sensitivity = (candidate − baseline in AB) − (candidate − baseline in BA)")


def _order_stats(c: dict, paired: dict, d: dict, strata: _Strata, baseline_point,
                 rng_seed: int) -> dict:
    """The treatment effect in each run order, their difference, and whether
    that difference is inside the tolerance declared for this criterion."""
    m, method = d["measurement"], d["method"]
    if not strata.crossover:
        return {"design": "single", "state": "not_assessed",
                "note": "each seed ran in one order only, so the order effect cannot be "
                        "separated from the change"}
    keys, cells, aggregate = paired["keys"], paired["cells"], paired["aggregate"]

    def column(arm: int, order: str) -> list:
        return [mean([cells[(r, order)][k][arm] for r in range(strata.replicates)]) for k in keys]

    b_ab, c_ab, b_ba, c_ba = column(0, "AB"), column(1, "AB"), column(0, "BA"), column(1, "BA")
    conf = method["confidence"]

    def contrast(columns, weights, salt):
        return contrast_summary(columns, weights, aggregate, conf, rng_seed + salt)

    ab = contrast([c_ab, b_ab], [1, -1], 1)
    ba = contrast([c_ba, b_ba], [1, -1], 2)
    sens = contrast([c_ab, b_ab, c_ba, b_ba], [1, -1, -1, 1], 3)
    first = contrast([c_ba, b_ab], [1, -1], 4)     # each arm measured in first position
    second = contrast([c_ab, b_ba], [1, -1], 5)    # each arm measured in second position
    drift = _drift(d, strata, paired, conf, rng_seed + 6)
    tolerance = None
    if "order_tolerance" in c:
        tolerance = c["order_tolerance"] * (abs(baseline_point) if c.get("relative") else 1.0)
    est, iv = sens["estimate"], sens["interval"]
    if tolerance is None:
        state = "not_gated"
    elif method["uncertainty"] == "point":
        state = "within_tolerance" if abs(est) <= tolerance else "exceeds_tolerance"
    elif not iv:
        state = "unresolved"
    elif -tolerance <= iv[0] and iv[1] <= tolerance:
        state = "within_tolerance"
    elif iv[0] > tolerance or iv[1] < -tolerance:
        state = "exceeds_tolerance"
    else:
        state = "unresolved"
    return {"design": "crossover", "n_units": len(keys), "replicates": strata.replicates,
            "delta_ab": ab["estimate"], "delta_ab_interval": ab["interval"],
            "delta_ba": ba["estimate"], "delta_ba_interval": ba["interval"],
            "sensitivity": est, "sensitivity_interval": iv, "estimator": sens["estimator"],
            "second_position_shift": est / 2.0,
            "first_position_delta": first["estimate"],
            "second_position_delta": second["estimate"],
            "drift_between_sequences": drift["estimate"], "drift_interval": drift["interval"],
            "tolerance": tolerance, "state": state, "definition": ORDER_EXPLAINED}


def _drift(d: dict, strata: _Strata, paired: dict, conf: float, rng_seed: int) -> dict:
    """How much the metric moved between the first and the second sequence a
    block ran, both arms averaged: later minus earlier. Both sequences hold
    one run of each arm, so the change itself cancels and what is left is
    time: warming hardware, a filling page cache, a neighbour's load. A
    diagnostic only. It needs to know which sequence each seed ran first, so
    it is reported when units are seeds or there is a single seed."""
    seeds = d["method"]["seeds"]
    lead = {}
    for group in schedule.groups(schedule.build(seeds, d["measurement"])):
        lead.setdefault((group[0]["seed"], group[0]["replicate"]), group[0]["sequence"])
    keys, cells = paired["keys"], paired["cells"]
    by_seed = paired["pairing"] == "seed"
    if not keys or not (by_seed or len(seeds) == 1):
        return {"estimate": None, "interval": None}
    earlier, later = [], []
    for k in keys:
        seed = k if by_seed else seeds[0]
        e, l = [], []
        for r in range(strata.replicates):
            first = lead[(seed, r)]
            second = "BA" if first == "AB" else "AB"
            e.append(mean(cells[(r, first)][k]))
            l.append(mean(cells[(r, second)][k]))
        earlier.append(mean(e))
        later.append(mean(l))
    return contrast_summary([later, earlier], [1, -1], "mean", conf, rng_seed)


def _fmt(x, unit=""):
    if x is None:
        return "n/a"
    return f"{x:.4g} {unit}" if unit else f"{x:.4g}"


def _evaluate_criterion(c: dict, index: _Strata, d: dict, rng_base: int) -> dict:
    res = _judge_criterion(c, index, d, rng_base)
    if not (index.measured and c.get("performance")) or "delta" not in res or res["delta"] is None:
        return res
    if res["status"] == "missing":
        return res
    paired = _paired_values(index, c, d)
    seed = rng_base + zlib.crc32(c["id"].encode("utf-8"))
    order = res["order"] = _order_stats(c, paired, d, index, res["baseline"], seed)
    if c["role"] == "info":
        return res
    u = c.get("unit", "")
    if order["state"] in ("within_tolerance", "exceeds_tolerance", "unresolved"):
        state = {"within_tolerance": "ok", "exceeds_tolerance": "bad", "unresolved": "uncertain"}
        res["checks"].append({"rule": f"order effect within ±{_fmt(order['tolerance'], u)}",
                              "state": state[order["state"]]})
        if order["state"] == "exceeds_tolerance":
            res["status"] = "unassessed"
            res["notes"].append("not assessed: the two run orders disagree by more than the "
                                "declared tolerance, so neither a pass nor a fail is supported")
        if order["state"] == "unresolved" and res["status"] in ("met", "clear"):
            res["status"] = "uncertain"
            res["notes"].append(
                "the order effect could not be shown to be inside the declared tolerance"
                + ("" if order["sensitivity_interval"] else
                   ": too few independent units for an interval"))
    uncontrolled = d["measurement"]["cache"]["uncontrolled"]
    if (order["state"] == "not_assessed" or uncontrolled) and res["status"] in ("met", "clear"):
        res["status"] = "exploratory"
        res["notes"].append(
            "exploratory only: " + ("each seed ran in one order" if order["state"] == "not_assessed"
                                    else "declared shared state was not reset ("
                                         + ", ".join(uncontrolled) + ")"))
    return res


def _judge_criterion(c: dict, index: _Strata, d: dict, rng_base: int) -> dict:
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
            f"({res['missing_rate']:.1%} > allowed {method['max_missing_rate']:.1%})"
            + (f"; {coverage['incomplete_units']} unit(s) lack a complete set of measured runs"
               if coverage.get("incomplete_units") else ""))
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


def _validity_checks(d: dict, index: _Strata, criteria: list) -> list:
    """Controls and witnesses are judged on every declared seed, and under a
    measurement protocol on every measured pair of that seed. A reading
    absent from any of them is a missing control, never a passed check."""
    checks = []
    v = d["validity"]
    n = len(index.seeds) * len(index.labels)
    noun = index.noun
    for name in v["held_constant"]:
        values, absent = index.cells(name)
        if absent:
            checks.append({"check": "held_constant", "metric": name, "passed": False,
                           "code": "held_constant_missing",
                           "detail": f"`{name}` was declared held constant but is not reported by "
                                     f"both arms on {len(absent)} of {n} {noun}"})
            continue
        off = [cell for cell, (b, c) in values.items() if not _same(b, c)]
        moved = []
        if index.measured and not off:
            for seed in index.seeds:
                readings = [b for cell, (b, _) in values.items() if cell[0] == seed]
                if any(not _same(readings[0], r) for r in readings[1:]):
                    moved.append(seed)
        if off:
            b, c = values[off[0]]
            detail = (f"`{name}` differs between arms on {len(off)} of {n} {noun}; "
                      f"{index.where(off[0])}: baseline {_fmt(b)}, candidate {_fmt(c)}. "
                      "The arms were not run under the same conditions")
        elif moved:
            detail = (f"`{name}` differs between repeated runs of seed {moved[0]}"
                      + (f" and {len(moved) - 1} other seed(s)" if len(moved) > 1 else "")
                      + ". Runs of the same seed did not do the same work")
        else:
            detail = f"`{name}` is equal between arms on all {n} {noun}"
        checks.append({"check": "held_constant", "metric": name, "passed": not (off or moved),
                       "code": "control_drift", "detail": detail})
    for item in v["baseline_expect"]:
        readings = [part.seed_value(item["metric"], "baseline", s)
                    for part in index.parts.values() for s in index.seeds]
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
        values, absent = index.cells(item["metric"])
        if absent:
            ok, code = False, "activation_missing"
            detail = (f"activation witness `{item['metric']}` is not reported by both arms on "
                      f"{len(absent)} of {n} {noun}")
        else:
            idle = [cell for cell, (b, c) in values.items() if _same(b, c)]
            ok, code = not idle, "inactive_treatment"
            if idle:
                detail = (f"witness `{item['metric']}` is identical between arms on {len(idle)} of "
                          f"{n} {noun}, so the change did not take effect there")
            else:
                detail = f"witness `{item['metric']}` differs between arms on all {n} {noun}"
        activation_ok = ok if activation_ok is None else (activation_ok and ok)
        checks.append({"check": "activation", "metric": item["metric"], "passed": ok,
                       "code": code, "detail": detail})

    for c in criteria:
        order = c.get("order") or {}
        if c["role"] == "info" or order.get("state") not in ("within_tolerance", "exceeds_tolerance"):
            continue
        u = c.get("unit", "")
        inside = order["state"] == "within_tolerance"
        iv = order["sensitivity_interval"]
        span = f" [{_fmt(iv[0])}, {_fmt(iv[1])}]" if iv else ""
        checks.append({
            "check": "order_effect", "metric": c["id"], "passed": inside,
            "code": "order_effect_within_tolerance" if inside else "order_bias",
            "detail": (f"`{c['id']}`: the change measured {_fmt(order['delta_ab'], u)} with the "
                       f"candidate second and {_fmt(order['delta_ba'], u)} with the candidate first; "
                       f"difference {_fmt(order['sensitivity'], u)}{span}, tolerance "
                       f"±{_fmt(order['tolerance'], u)}. "
                       + ("Run order did not move the result by more than the tolerance"
                          if inside else
                          "Run order moved the result by more than the tolerance, so the "
                          "declared cache and measurement conditions were not established"))})

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


def _protocol_limitations(d: dict) -> list:
    m = d.get("measurement")
    if not m:
        return []
    cache = m["cache"]
    runs = m["replicates"] * (2 if m["design"] == "crossover" else 1)
    out = []
    if m["design"] == "crossover":
        out.append("Run order: every seed ran both orders, baseline first and candidate first, "
                   "on a schedule fixed before approval. That balances position effects that "
                   "add the same amount to whichever arm runs second. It does not remove "
                   "carryover that differs by arm, or drift that is not steady over the run.")
    else:
        out.append("Run order: `design: single` ran each seed in one order only. Performance "
                   "results are exploratory and are not order-controlled evidence.")
    out.append(f"Each seed was measured {runs} time(s) per arm. Those runs are averaged within "
               "the independent unit and are not counted as extra units.")
    out.append("Caches: every measured attempt started from a new, empty directory for the "
               "cache locations listed in the execution protocol, in a new process. The OS "
               "page cache, hardware thermal and frequency state, GPU driver caches, remote "
               "services and other load on the machine were not controlled.")
    if cache["profile"] == "warm":
        out.append(f"Warm profile: after the reset each attempt ran {cache['warmup_runs']} untimed "
                   "warm-up run(s) of the same command for its own arm. Warm state does not pass "
                   "between arms. State held inside one process is the harness's to warm.")
    if cache["uncontrolled"]:
        out.append("Declared shared state that was not reset: " + ", ".join(cache["uncontrolled"])
                   + ". Performance results are exploratory.")
    return out


def compare(d: dict, observations: list, execution_status: str, integrity: list) -> dict:
    """Pure: same inputs, same comparison."""
    dhash = decl.declaration_hash(d)
    method = d["method"]
    index = _Strata(observations, d)
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
        failed = [c for c in validity_checks if not c["passed"] and c["code"] != "order_bias"]
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
        exploratory = [c for c in criteria if c["status"] == "exploratory"]
        biased = [c for c in criteria if c["status"] == "unassessed"]
        if breached or not_met:
            verdict = "FAIL"
            reasons += [f"guardrail_veto:{c['id']}" for c in breached]
            reasons += [f"target_not_met:{c['id']}" for c in not_met]
            reasons += [f"order_bias:{c['id']}" for c in biased]
        elif biased:
            verdict = "NO_SIGNAL"
            reasons += [f"order_bias:{c['id']}" for c in biased]
        elif uncertain or exploratory:
            verdict = "WARN"
            reasons += [f"uncertain:{c['id']}" for c in uncertain]
            reasons += [f"order_effect_unresolved:{c['id']}" for c in uncertain
                        if c.get("order", {}).get("state") == "unresolved"]
            reasons += [f"exploratory:{c['id']}" for c in exploratory]
        else:
            verdict = "PASS"
            reasons += [f"target_met:{c['id']}" for c in criteria if c["status"] == "met"]
            reasons += [f"guardrail_clear:{c['id']}" for c in criteria if c["status"] == "clear"]
        if verdict != "NO_SIGNAL" and any(c.get("code") == "measured_zero" for c in checks):
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
    limitations += _protocol_limitations(d)
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
