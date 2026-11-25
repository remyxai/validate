"""Run order, cache isolation and the order-effect check.

Three layers, none of which reads a wall clock:

* the schedule, as a pure function of the declaration
* the comparison, on simulated observations where the true effect, the
  second-position speedup and the seed-by-order interaction are all known
* the runner, on a probe harness that reports what it found in its caches
"""
import collections
import json
import os
import random
import subprocess
import sys
import textwrap
import time

import pytest
import yaml

from remyx_validate import (bundle, collect, compare, declaration as decl, demo, isolation,
                            plan as planmod, runner as runmod, schedule)
from conftest import load_evidence

CROSSOVER = {"design": "crossover", "replicates": 1, "schedule_seed": 3}


# ── schedule ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seeds", [[0], [4, 1, 7], [0, 1, 2, 3, 4, 5]])
@pytest.mark.parametrize("replicates", [1, 3])
def test_every_seed_and_replicate_runs_both_orders(seeds, replicates):
    m = dict(CROSSOVER, replicates=replicates)
    slots = schedule.build(seeds, m)
    assert schedule.problems(slots, seeds, m) == []
    assert len(slots) == 4 * len(seeds) * replicates
    positions = collections.Counter((s["arm"], s["period"]) for s in slots)
    assert len(set(positions.values())) == 1 and len(positions) == 4
    for seed in seeds:
        for rep in range(replicates):
            mine = {(s["sequence"], s["period"], s["arm"]) for s in slots
                    if (s["seed"], s["replicate"]) == (seed, rep)}
            assert mine == {("AB", 1, "baseline"), ("AB", 2, "candidate"),
                            ("BA", 1, "candidate"), ("BA", 2, "baseline")}


def test_schedule_is_reproducible_and_independent_of_the_workload_seeds():
    m = dict(CROSSOVER, replicates=2)
    assert schedule.build([3, 1, 2], m) == schedule.build([1, 2, 3], m)
    assert schedule.build([1, 2, 3], m) == schedule.build([1, 2, 3], dict(m))
    assert schedule.build([1, 2, 3], m) != schedule.build([1, 2, 3], dict(m, schedule_seed=4))

    def shape(seeds):
        """The schedule with each seed replaced by its rank."""
        rank = {s: i for i, s in enumerate(sorted(seeds))}
        return [(rank[s["seed"]], s["replicate"], s["sequence"], s["period"])
                for s in schedule.build(seeds, m)]

    # Renumbering the workload seeds (odd, even, large) moves nothing: the
    # order never comes from a seed's value or parity.
    assert shape([0, 1, 2, 3]) == shape([1, 3, 5, 7]) == shape([10, 20, 31, 42])


def test_which_order_goes_first_is_balanced_across_blocks():
    m = dict(CROSSOVER, replicates=1)
    slots = schedule.build(list(range(8)), m)
    leads = [g[0]["sequence"] for g in schedule.groups(slots)][::2]
    assert collections.Counter(leads) == {"AB": 4, "BA": 4}
    single = schedule.build(list(range(8)), dict(m, design="single"))
    assert collections.Counter(s["sequence"] for s in single) == {"AB": 8, "BA": 8}
    assert schedule.problems(single, list(range(8)), dict(m, design="single")) == []


def test_single_design_with_an_odd_seed_count_is_as_balanced_as_it_can_be():
    slots = schedule.build([0, 1, 2], dict(CROSSOVER, design="single"))
    counts = collections.Counter(g[0]["sequence"] for g in schedule.groups(slots))
    assert sorted(counts.values()) == [1, 2]


# ── declaration rules and identities ──────────────────────────────────────

def _decl(seeds, measurement=None, tol=1.0, role="target", **method):
    d = yaml.safe_load(demo.declaration_text("guardrail", protocol=False))
    d["method"].update({"unit": "seed", "seeds": list(seeds), **method})
    c = {"id": "latency", "metric": "latency_ms", "role": role, "direction": "lower",
         "performance": False}                         # the 0.1.0a0 behaviour, asked for
    if role == "guardrail":
        c["max_regression"] = 2.0
    if measurement is not None:
        c["performance"] = True
        if tol is not None:
            c["order_tolerance"] = tol
        d["measurement"] = dict(measurement)
    d["criteria"] = [c]
    d["data"] = {"identity": "simulated"}
    d["validity"] = {"activation": [{"metric": "witness"}], "held_constant": ["work"]}
    return decl.normalize(d)


def test_a_quality_only_declaration_is_untouched():
    """No timing criterion, no protocol: the same document and hashes as before."""
    raw = yaml.safe_load(demo.declaration_text("guardrail", protocol=False))
    raw["criteria"] = [c for c in raw["criteria"] if c["metric"] != "latency_ms"]
    d = decl.normalize(raw)
    assert "measurement" not in d and decl.planned_schedule(d) is None
    assert all("performance" not in c for c in d["criteria"])


def test_timing_criteria_get_the_protocol_by_default():
    raw = yaml.safe_load(demo.declaration_text("guardrail", protocol=False))
    assert raw["criteria"][1]["performance"] is False and "measurement" not in raw
    opted_out = decl.normalize(raw)
    assert "measurement" not in opted_out              # only because it says so
    del raw["criteria"][1]["performance"]              # what a user writes
    d = decl.normalize(raw)
    lat = d["criteria"][1]
    assert lat["performance"] is True and lat[decl.INFERRED] is True
    assert d["measurement"]["design"] == "crossover" and d["measurement"]["cache"]["profile"] == "cold"
    assert len(decl.planned_schedule(d)) == 4 * len(d["method"]["seeds"])
    problems = decl.semantic_problems(d)               # fail closed until a tolerance is declared
    assert len(problems) == 1 and "looks like a timing metric" in problems[0]
    assert "performance: false" in problems[0]
    raw["criteria"][1]["order_tolerance"] = 5
    assert decl.semantic_problems(decl.normalize(raw)) == []
    # The flag that says "inferred" is not part of what was asked.
    explicit = yaml.safe_load(yaml.safe_dump(raw))
    explicit["criteria"][1]["performance"] = True
    assert decl.declaration_hash(decl.normalize(raw)) == decl.declaration_hash(decl.normalize(explicit))


@pytest.mark.parametrize("metric,unit,timing", [
    ("latency_ms", "", True), ("steps_per_s", "", True), ("wall_s", "", True), ("ttft", "", True),
    ("tokens_per_second", "", True), ("response_time", "", True), ("p95", "ms", True),
    ("rate", "req/s", True), ("final_eval_loss", "", False), ("resolved", "", False),
    ("cost_usd", "", False), ("train_steps", "", False), ("dataset_size", "", False)])
def test_what_reads_as_a_timing_metric(metric, unit, timing):
    assert decl.looks_like_timing({"metric": metric, "unit": unit}) is timing


def test_an_informational_timing_needs_no_tolerance_and_is_still_crossed_over():
    raw = yaml.safe_load(demo.declaration_text("guardrail", protocol=False))
    raw["criteria"][1] = {"id": "throughput", "metric": "steps_per_s", "role": "info",
                          "direction": "higher"}
    d = decl.normalize(raw)
    assert decl.semantic_problems(d) == [] and d["measurement"]["design"] == "crossover"


def test_plan_says_when_a_timing_was_inferred_or_opted_out(tmp_path):
    def warnings(mutate):
        path = demo.create(str(tmp_path / "p"), "positive", protocol=False)
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        mutate(raw["criteria"][1])
        with open(path, "w") as fh:
            yaml.safe_dump(raw, fh)
        return planmod.build(path)["warnings"]

    assert any("declared `performance: false`" in w for w in warnings(lambda c: None))
    inferred = warnings(lambda c: (c.pop("performance"), c.update(order_tolerance=5)))
    assert any("treated as a timing metric" in w for w in inferred)


def test_explicit_performance_and_single_design():
    raw = yaml.safe_load(demo.declaration_text("guardrail", protocol=False))
    raw["criteria"][1]["performance"] = True
    problems = decl.semantic_problems(decl.normalize(raw))
    assert any("eclare `order_tolerance`" in p for p in problems)
    raw["criteria"][1]["order_tolerance"] = 5
    assert decl.semantic_problems(decl.normalize(raw)) == []
    raw["measurement"] = {"design": "single"}        # exploratory: nothing to gate
    del raw["criteria"][1]["order_tolerance"]
    assert decl.semantic_problems(decl.normalize(raw)) == []


def test_protocol_is_part_of_the_declaration_and_execution_identities():
    a, b = _decl([0, 1], CROSSOVER), _decl([0, 1], dict(CROSSOVER, schedule_seed=4))
    assert decl.declaration_hash(a) != decl.declaration_hash(b)
    shas = {"baseline": "0" * 40, "candidate": "1" * 40}
    assert decl.spec_hash(a, shas, None) != decl.spec_hash(b, shas, None)
    warm = _decl([0, 1], dict(CROSSOVER, cache={"profile": "warm"}))
    assert warm["measurement"]["cache"]["warmup_runs"] == 1
    assert decl.declaration_hash(warm) != decl.declaration_hash(a)


def test_isolated_cache_variables_cannot_be_set_by_an_arm():
    d = _decl([0], CROSSOVER)
    d["arms"]["candidate"]["env"]["XDG_CACHE_HOME"] = "/somewhere"
    assert any("set by the runner" in p for p in decl.semantic_problems(d))


# ── comparison on simulated runs ──────────────────────────────────────────

def _simulate(d, effect=0.0, second=0.0, seed_by_order=0.0, noise=0.0, rng=0, work=None,
              drift=0.0):
    """latency = a per-seed level + `effect` on the candidate + what running
    in second position adds (`second`, plus `seed_by_order` per unit of seed).
    Without a protocol the old rule is used: order alternates by seed index."""
    r = random.Random(rng)
    m, seeds = d.get("measurement"), d["method"]["seeds"]
    if m:
        slots = schedule.build(seeds, m)
    else:
        slots = [{"seed": s, "arm": schedule.ARM_OF[letter], "period": p}
                 for i, s in enumerate(seeds)
                 for p, letter in enumerate("AB" if i % 2 == 0 else "BA", 1)]
    rows = []
    for slot in slots:
        s, arm = slot["seed"], slot["arm"]
        y = 100.0 + 3.0 * s + (effect if arm == "candidate" else 0.0)
        if slot["period"] == 2:
            y += second + seed_by_order * s
        y += r.gauss(0.0, noise) if noise else 0.0
        y += drift * slot.get("ordinal", 0)            # the machine slowing down over the run
        tags = {k: slot[k] for k in ("replicate", "sequence", "period")} if m else {}
        readings = {"latency_ms": y, "witness": float(arm == "candidate"),
                    "work": (work or (lambda slot: 1000.0 + slot["seed"]))(slot)}
        for name, value in readings.items():
            rows.append({"arm": arm, "seed": s, "unit": None, "slice": None, "metric": name,
                         "value": float(value), "attempt_id": "att", **tags})
    return rows


def _run(d, rows=None, **sim):
    out = compare.compare(d, rows if rows is not None else _simulate(d, **sim), "COMPLETED", [])
    return out, out["criteria"][0]


@pytest.mark.parametrize("seeds", [[0, 1, 2, 3, 4], [0, 1, 2, 3]])
def test_alternating_by_seed_misattributes_a_second_run_speedup(seeds):
    """True effect is zero. Whichever arm runs second is faster, more so on
    larger seeds. One order per seed credits that speedup to an arm, even
    with as many AB seeds as BA seeds."""
    carry = dict(effect=0.0, second=-2.0, seed_by_order=-4.0)
    _, old = _run(_decl(seeds, uncertainty="point"), **carry)
    assert abs(old["delta"]) >= 2.0                    # an effect that is not there
    out, new = _run(_decl(seeds, CROSSOVER), **carry)
    assert new["delta"] == pytest.approx(0.0, abs=1e-9)  # the crossover cancels it
    assert new["order"]["sensitivity"] < -10
    assert out["verdict"] != "PASS"
    assert new["order"]["state"] in ("exceeds_tolerance", "unresolved")


def test_old_design_can_pass_a_change_that_does_nothing():
    carry = dict(effect=0.0, second=-2.0, seed_by_order=-4.0)
    out, _ = _run(_decl([0, 1, 2, 3, 4], uncertainty="point"), **carry)
    assert out["verdict"] == "PASS"                    # the bug this protocol exists for
    out, c = _run(_decl([0, 1, 2, 3, 4], CROSSOVER, uncertainty="point"), **carry)
    assert out["verdict"] == "NO_SIGNAL" and out["reasons"] == ["order_bias:latency"]
    assert out["validity"]["status"] == "INVALID"


@pytest.mark.parametrize("seeds", [[0, 1, 2, 3, 4], [0, 1, 2, 3]])
@pytest.mark.parametrize("effect,verdict", [(-5.0, "PASS"), (5.0, "FAIL"), (0.0, "WARN")])
def test_true_effects_are_recovered_when_order_does_not_matter(seeds, effect, verdict):
    out, c = _run(_decl(seeds, CROSSOVER), effect=effect, noise=0.1, rng=11)
    assert out["verdict"] == verdict, out["reasons"]
    assert c["order"]["state"] == "within_tolerance"
    assert c["delta"] == pytest.approx(effect, abs=0.3)
    assert c["delta_interval"][0] <= effect <= c["delta_interval"][1]
    assert c["n_pairs"] == len(seeds)                  # seeds, not attempts


def test_an_additive_position_effect_cancels_and_is_still_reported():
    """A constant second-run speedup leaves the balanced estimate exact. It
    is outside tolerance all the same, so it is not passed over in silence."""
    out, c = _run(_decl([0, 1, 2, 3], CROSSOVER), effect=-5.0, second=-3.0)
    o = c["order"]
    assert c["delta"] == pytest.approx(-5.0)
    assert o["delta_ab"] == pytest.approx(-8.0) and o["delta_ba"] == pytest.approx(-2.0)
    assert o["sensitivity"] == pytest.approx(-6.0)
    assert o["second_position_shift"] == pytest.approx(-3.0)
    assert o["first_position_delta"] == pytest.approx(-5.0)
    assert o["second_position_delta"] == pytest.approx(-5.0)
    assert out["verdict"] == "NO_SIGNAL" and out["reasons"] == ["order_bias:latency"]
    wide, c = _run(_decl([0, 1, 2, 3], CROSSOVER, tol=10.0), effect=-5.0, second=-3.0)
    assert wide["verdict"] == "PASS" and c["order"]["state"] == "within_tolerance"


def test_not_significant_is_not_the_same_as_no_order_effect():
    """Noisy order effect whose interval straddles the tolerance: undecided,
    never read as absent."""
    d = _decl([0, 1, 2, 3, 4, 5], CROSSOVER, tol=1.0)
    rows = _simulate(d, effect=-20.0)
    for row in rows:                                   # order effects of +-3 that average to 0
        if row["metric"] == "latency_ms" and row["period"] == 2:
            row["value"] += 1.5 if row["seed"] % 2 else -1.5
    out, c = _run(d, rows)
    assert c["order"]["sensitivity"] == pytest.approx(0.0, abs=1e-9)
    assert c["order"]["state"] == "unresolved" and c["status"] == "uncertain"
    assert out["verdict"] == "WARN"
    assert "order_effect_unresolved:latency" in out["reasons"]


def test_one_seed_gives_no_interval_and_no_pass():
    out, c = _run(_decl([0], CROSSOVER), effect=-5.0)
    assert c["n_pairs"] == 1 and c["delta_interval"] is None
    assert c["order"]["sensitivity_interval"] is None and c["order"]["state"] == "unresolved"
    assert out["verdict"] == "WARN"
    point, c = _run(_decl([0], CROSSOVER, uncertainty="point"), effect=-5.0)
    assert point["verdict"] == "PASS" and c["order"]["state"] == "within_tolerance"


def test_replicates_are_nested_in_the_seed_not_extra_units():
    one, a = _run(_decl([0, 1, 2], CROSSOVER), effect=-5.0, noise=0.1, rng=4)
    three, b = _run(_decl([0, 1, 2], dict(CROSSOVER, replicates=3)), effect=-5.0, noise=0.1, rng=4)
    assert a["n_pairs"] == b["n_pairs"] == 3 and b["order"]["n_units"] == 3
    assert a["coverage"]["expected"] == 6 and b["coverage"]["expected"] == 18
    assert three["verdict"] == "PASS"


def test_all_four_runs_of_a_seed_must_do_the_same_work():
    d = _decl([0, 1], CROSSOVER)
    ok, _ = _run(d, effect=-5.0)
    held = [c for c in ok["validity"]["checks"] if c["check"] == "held_constant"]
    assert held[0]["passed"] and "all 4 measured pair(s)" in held[0]["detail"]
    # The two arms agree inside each pair, but the BA pair did different work.
    drift, _ = _run(d, effect=-5.0, work=lambda s: 1000.0 + s["seed"] + (s["sequence"] == "BA"))
    assert drift["verdict"] == "NO_SIGNAL" and drift["reasons"] == ["control_drift:work"]
    arm, _ = _run(d, effect=-5.0, work=lambda s: 1000.0 + (s["arm"] == "candidate"))
    assert arm["reasons"] == ["control_drift:work"]


def test_an_incomplete_block_cannot_produce_a_performance_verdict():
    d = _decl([0, 1, 2], CROSSOVER)
    rows = [r for r in _simulate(d, effect=-5.0)
            if not (r["seed"] == 1 and r["sequence"] == "BA" and r["arm"] == "candidate")]
    out, c = _run(d, rows)
    assert out["verdict"] == "INCOMPLETE" and c["status"] == "missing"
    assert c["coverage"]["incomplete_units"] == 1 and "order" not in c
    # Allowing missing data still leaves seed 1 out: an unbalanced average
    # would let the order back in.
    lax = _decl([0, 1, 2], CROSSOVER, max_missing_rate=0.5)
    out, c = _run(lax, rows)
    assert c["n_pairs"] == 2 and c["coverage"]["complete_units"] == 2


def test_observations_outside_the_schedule_are_rejected():
    d = _decl([0, 1], CROSSOVER)
    rows = _simulate(d, effect=-5.0)
    untagged = [{k: v for k, v in r.items() if k not in ("replicate", "sequence", "period")}
                for r in rows]
    out, _ = _run(d, untagged)
    assert out["verdict"] == "INCOMPLETE"
    assert "observations:unscheduled_observation" in out["reasons"]
    out, _ = _run(d, rows + [dict(rows[0])])
    assert "observations:duplicate_observation" in out["reasons"]


def test_single_sequence_and_declared_shared_state_are_exploratory_only():
    single, c = _run(_decl([0, 1, 2, 3], dict(CROSSOVER, design="single"), tol=None),
                     effect=-5.0, noise=0.1)
    assert single["verdict"] == "WARN" and single["reasons"] == ["exploratory:latency"]
    assert c["status"] == "exploratory" and c["order"]["state"] == "not_assessed"
    assert any("not order-controlled" in text for text in single["limitations"])
    shared = dict(CROSSOVER, cache={"uncontrolled": ["provider prompt cache"]})
    out, c = _run(_decl([0, 1, 2, 3], shared), effect=-5.0, noise=0.1)
    assert out["verdict"] == "WARN" and c["status"] == "exploratory"
    worse, _ = _run(_decl([0, 1, 2, 3], shared), effect=5.0, noise=0.1)
    assert worse["verdict"] == "FAIL"                  # a regression is still a regression


def _with_quality(d, **criterion):
    extra = {"id": "accuracy", "metric": "accuracy", "direction": "higher", **criterion}
    d["criteria"].append(decl.normalize({"criteria": [extra]})["criteria"][0])
    return d


def _add_accuracy(rows, baseline, candidate):
    cells = {(r["arm"], r["seed"], r["replicate"], r["sequence"], r["period"]) for r in rows}
    for arm, seed, replicate, sequence, period in sorted(cells):
        rows.append({"arm": arm, "seed": seed, "unit": None, "slice": None, "metric": "accuracy",
                     "value": (candidate if arm == "candidate" else baseline) + 0.001 * seed,
                     "attempt_id": "att", "replicate": replicate, "sequence": sequence,
                     "period": period})
    return rows


def test_an_order_biased_latency_does_not_hide_or_cause_a_quality_verdict():
    """The latency criterion is unassessed. A quality guardrail that is
    breached on valid evidence is still a FAIL; with quality fine there is
    nothing to fail on and the result is NO_SIGNAL, never PASS."""
    carry = dict(effect=-5.0, second=-3.0)
    d = _with_quality(_decl([0, 1, 2, 3], CROSSOVER), role="guardrail", max_regression=0.01)
    worse, _ = _run(d, _add_accuracy(_simulate(d, **carry), 0.80, 0.60))
    by_id = {c["id"]: c for c in worse["criteria"]}
    assert by_id["latency"]["status"] == "unassessed" and by_id["accuracy"]["status"] == "breached"
    assert worse["verdict"] == "FAIL" and worse["validity"]["status"] == "VALID"
    assert worse["reasons"] == ["guardrail_veto:accuracy", "order_bias:latency"]
    fine, _ = _run(d, _add_accuracy(_simulate(d, **carry), 0.80, 0.80))
    assert fine["verdict"] == "NO_SIGNAL" and fine["reasons"] == ["order_bias:latency"]
    assert {c["id"]: c for c in fine["criteria"]}["accuracy"]["status"] == "clear"
    # A latency that looks like a regression in one order is not a FAIL either.
    slower, c = _run(_decl([0, 1, 2, 3], CROSSOVER), effect=5.0, second=-3.0)
    assert c["status"] == "unassessed" and slower["verdict"] == "NO_SIGNAL"


def test_steady_drift_over_the_run_is_cancelled_and_reported():
    """Every attempt is 0.25 ms slower than the one before it. The crossover
    removes it from the estimate. The drift column reports it: the second
    pair of runs on a seed starts two attempts after the first."""
    out, c = _run(_decl([0, 1, 2, 3, 4], CROSSOVER), effect=-5.0, drift=0.25)
    o = c["order"]
    assert c["delta"] == pytest.approx(-5.0) and out["verdict"] == "PASS"
    assert o["drift_between_sequences"] == pytest.approx(0.5)
    assert o["sensitivity"] == pytest.approx(0.5) and o["state"] == "within_tolerance"
    flat, c = _run(_decl([0, 1, 2, 3, 4], CROSSOVER), effect=-5.0)
    assert c["order"]["drift_between_sequences"] == pytest.approx(0.0, abs=1e-9)


def test_input_fingerprints_become_controls(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps(
        {"metrics": {"latency_ms": 12.5}, "fingerprints": {"inputs": "sha256:ab12", "rows": 240}}))
    a = collect.read(str(tmp_path), "")["metrics"]
    assert set(a) == {"latency_ms", "fingerprint.inputs", "fingerprint.rows"}
    (tmp_path / "results.json").write_text(json.dumps({"fingerprints": {"inputs": "sha256:ab13"}}))
    b = collect.read(str(tmp_path), "")["metrics"]
    assert a["fingerprint.inputs"] != b["fingerprint.inputs"]
    assert a["fingerprint.inputs"] == int(a["fingerprint.inputs"]) < 2 ** 48   # exact in JSON
    (tmp_path / "results.json").write_text(json.dumps({"fingerprints": {"inputs": ["a"]}}))
    with pytest.raises(collect.ResultsError):
        collect.read(str(tmp_path), "")
    # Same inputs for all four runs of a seed, or the result is not trusted.
    d = _decl([0, 1], CROSSOVER)
    d["validity"]["held_constant"] = ["fingerprint.inputs"]
    rows = _simulate(d, effect=-5.0)
    for row in rows:
        if row["metric"] == "work":
            row["metric"] = "fingerprint.inputs"
    assert _run(d, rows)[0]["verdict"] != "NO_SIGNAL"
    for row in rows:
        if row["metric"] == "fingerprint.inputs" and (row["seed"], row["sequence"], row["arm"]) == (1, "BA", "candidate"):
            row["value"] += 1
    assert _run(d, rows)[0]["reasons"] == ["control_drift:fingerprint.inputs"]


def test_quality_criteria_are_averaged_over_both_orders_without_an_order_gate():
    d = _decl([0, 1, 2], CROSSOVER)
    d["criteria"].append(decl.normalize({"criteria": [
        {"id": "work_info", "metric": "work", "role": "info", "direction": "higher"}]})["criteria"][0])
    out, _ = _run(d, effect=-5.0, noise=0.1)
    info = out["criteria"][1]
    assert "order" not in info and info["n_pairs"] == 3 and info["delta"] == pytest.approx(0.0)


def test_limitations_never_call_uncontrolled_conditions_controlled():
    out, _ = _run(_decl([0, 1], CROSSOVER), effect=-5.0)
    text = " ".join(out["limitations"])
    assert "were not controlled" in text and "does not remove" in text
    legacy, _ = _run(_decl([0, 1]), effect=-5.0)
    assert not any("Run order" in t for t in legacy["limitations"])


# ── cache isolation, on the filesystem ────────────────────────────────────

MEASUREMENT = decl.normalize({"measurement": {"cache": {"isolate": ["MYAPP_CACHE"]}}})["measurement"]


def test_a_cache_directory_is_new_empty_and_removed(tmp_path):
    owner = str(tmp_path / "run")
    root = os.path.join(owner, "attempts", "a1", "cache")
    paths = isolation.prepare(root, owner, MEASUREMENT)
    assert set(paths) == set(isolation.LAYERS) | {"MYAPP_CACHE"}
    assert isolation.is_clean(root, MEASUREMENT)
    with open(os.path.join(paths["XDG_CACHE_HOME"], "sentinel"), "w") as fh:
        fh.write("x")
    assert not isolation.is_clean(root, MEASUREMENT)
    with pytest.raises(isolation.IsolationError):
        isolation.prepare(root, owner, MEASUREMENT)    # never reuse a directory
    assert isolation.remove(root, owner) == "removed" and not os.path.exists(root)
    assert isolation.remove(root, owner) == "absent"


def test_removal_refuses_links_and_anything_outside_the_run(tmp_path):
    owner, outside = str(tmp_path / "run"), str(tmp_path / "home")
    os.makedirs(owner)
    os.makedirs(outside)
    keep = os.path.join(outside, "credentials")
    with open(keep, "w") as fh:
        fh.write("do not delete")
    assert isolation.remove(outside, owner).startswith("refused")
    assert isolation.remove(owner, owner).startswith("refused")
    link = os.path.join(owner, "cache")
    os.symlink(outside, link)
    assert isolation.remove(link, owner).startswith("refused")
    with pytest.raises(isolation.IsolationError):
        isolation.prepare(link, owner, MEASUREMENT)
    os.unlink(link)
    # A link planted inside a cache is unlinked, never followed.
    paths = isolation.prepare(link, owner, MEASUREMENT)
    os.symlink(outside, os.path.join(paths["TMPDIR"], "escape"))
    os.makedirs(os.path.join(paths["TMPDIR"], "locked"))
    open(os.path.join(paths["TMPDIR"], "locked", "f"), "w").close()
    os.chmod(os.path.join(paths["TMPDIR"], "locked"), 0o500)
    assert isolation.remove(link, owner) == "removed"
    assert os.path.isfile(keep)


def test_shared_assets_stay_where_they_already_are():
    got = isolation.shared_assets({"XDG_CACHE_HOME": "/c"})
    assert got == {"HF_HOME": "/c/huggingface", "TORCH_HOME": "/c/torch"}
    assert isolation.shared_assets({"HF_HOME": "/models"})["HF_HOME"] == "/models"


# ── the runner, on a probe that reports what it found in its caches ───────

PROBE = '''\
"""Counts files left in each isolated cache by an earlier process, then
leaves its own. Latency is a formula, never a clock."""
import json
import os
import sys
import time

arm, seed, phase = os.environ["VALIDATE_ARM"], int(os.environ["VALIDATE_SEED"]), os.environ["VALIDATE_PHASE"]
layers = os.environ["PROBE_LAYERS"].split(",")
found = 0
for name in layers:
    path = os.environ[name]
    assert path.startswith(os.environ["VALIDATE_CACHE_DIR"]), name
    found += len(os.listdir(path))
    open(os.path.join(path, "left-by-%s-%s-%d" % (phase, arm, os.getpid())), "w").close()
with open(os.path.join(os.environ["PROBE_SHARED"], "log"), "a") as fh:
    fh.write("%s %s %d %s %s %d\\n" % (phase, arm, seed, os.environ.get("VALIDATE_SEQUENCE", "-"),
                                   os.environ.get("VALIDATE_PERIOD", "-"), found))
flag = os.path.join(os.environ["PROBE_SHARED"], "failed-once")
if phase == "run" and os.environ.get("PROBE_FAIL") == "once" and arm == "candidate" \\
        and not os.path.exists(flag):
    open(flag, "w").close()
    sys.exit(3)
if phase == "run" and os.environ.get("PROBE_FAIL") == "hang":
    time.sleep(60)
hf = os.environ.get("HF_HOME", "")
print("METRICS_JSON=" + json.dumps({
    "latency_ms": 100.0 + 3 * seed - (5 if arm == "candidate" else 0),
    "files_from_earlier_processes": found,
    "assets_shared": int(bool(hf) and not hf.startswith(os.environ["VALIDATE_CACHE_DIR"])),
    "asset_present": int(os.path.isfile(os.path.join(os.environ["PROBE_SHARED"], "model.bin"))),
    "witness": int(arm == "candidate"), "work": 1000 + seed}))
'''

LAYER_NAMES = sorted(set(isolation.LAYERS) | {"MYAPP_CACHE"})


@pytest.fixture
def probe(tmp_path):
    """A project whose harness is the probe. Returns run(measurement, **options)."""

    def _run(measurement, fail="", retries=0, timeout_s=60, preflight=True, seeds=(0, 1),
             start=True):
        project = os.path.dirname(demo.create(str(tmp_path / "probe"), "positive"))
        shared = str(tmp_path / "shared")
        os.makedirs(shared, exist_ok=True)
        with open(os.path.join(shared, "model.bin"), "w") as fh:
            fh.write("immutable")
        with open(os.path.join(project, "harness", "eval.py"), "w") as fh:
            fh.write(PROBE)
        names = sorted(set(isolation.LAYERS) | set(measurement.get("cache", {}).get("isolate", [])))
        env = {"PROBE_LAYERS": ",".join(names), "PROBE_SHARED": shared, "PROBE_FAIL": fail}
        doc = {
            "spec_version": "0.1",
            "change": {"repo": "./app-repo", "baseline": {"ref": "main"}, "candidate": {"ref": "candidate"}},
            "harness": {"path": "./harness"},
            "run": {"command": "python3 {harness}/eval.py", "timeout_s": timeout_s, "retries": retries},
            "arms": {"baseline": {"env": env}, "candidate": {"env": env}},
            "method": {"unit": "seed", "seeds": list(seeds)},
            "criteria": [
                {"id": "latency", "metric": "latency_ms", "role": "target", "direction": "lower",
                 "performance": True, "order_tolerance": 1},
                {"id": "stale", "metric": "files_from_earlier_processes", "role": "info",
                 "direction": "lower"}],
            "validity": {"activation": [{"metric": "witness"}], "held_constant": ["work"]},
            "measurement": measurement,
            "budget": {"max_wall_s": 300},
        }
        if preflight:
            doc["preflight"] = {"command": "python3 {harness}/eval.py"}
        path = os.path.join(project, "impact.yaml")
        with open(path, "w") as fh:
            yaml.safe_dump(doc, fh)
        plan = planmod.build(path)
        planmod.save(project, plan)
        planmod.approve(project, plan["plan_digest"], "test")
        if not start:
            return path, shared
        summary = runmod.Runner(path).run()
        with open(os.path.join(shared, "log")) as fh:
            log = [line.split() for line in fh]
        return summary, load_evidence(summary), plan, log, shared

    return _run


def _leftover_caches(run_dir):
    return [base for base, dirs, _ in os.walk(run_dir) if os.path.basename(base) == "cache"]


def test_every_measured_attempt_starts_from_empty_caches(probe):
    """After a preflight that wrote into its own caches, across both arms,
    both orders and both seeds, no attempt sees a file from another process."""
    cache = {"isolate": ["MYAPP_CACHE"]}
    summary, ev, plan, log, shared = probe(dict(CROSSOVER, cache=cache))
    assert summary["verdict"] == "PASS", summary["reasons"]
    attempts = ev["execution"]["attempts"]
    assert len(attempts) == 8 and ev["execution"]["schedule"] == plan["schedule"]
    assert [a["planned_ordinal"] for a in attempts] == [a["ordinal"] for a in attempts] == list(range(1, 9))
    assert [(a["arm"], a["seed"], a["sequence"], a["period"]) for a in attempts] == \
        [(s["arm"], s["seed"], s["sequence"], s["period"]) for s in plan["schedule"]]
    assert all(a["isolation"]["fresh"] and a["isolation"]["cleanup"] == "removed" for a in attempts)
    assert [row[0] for row in log].count("preflight") == 2
    assert all(row[5] == "0" for row in log)           # preflight and every run
    stale = {c["id"]: c for c in ev["comparison"]["criteria"]}["stale"]
    assert stale["baseline"] == stale["candidate"] == 0
    assert _leftover_caches(summary["run_dir"]) == []
    assert os.path.isfile(os.path.join(shared, "model.bin"))
    assert bundle.verify(summary["run_dir"])["ok"]
    protocol = ev["execution"]["protocol"]
    assert protocol["order_controlled"] and "MYAPP_CACHE" in protocol["isolated_layers"]
    assert set(protocol["shared_immutable_assets"]) == {"HF_HOME", "TORCH_HOME"}
    assert any("page cache" in item for item in protocol["not_controlled"])
    assert protocol["host"]["cpu_count"] >= 1 and "hostname" not in protocol["host"]
    starts = [a["conditions"]["started_s"] for a in attempts]
    assert starts == sorted(starts) and all("before" in a["conditions"] for a in attempts)


def test_a_second_run_of_the_same_project_is_just_as_cold(probe):
    first = probe(dict(CROSSOVER, cache={"isolate": ["MYAPP_CACHE"]}), preflight=False)
    second = probe(dict(CROSSOVER, cache={"isolate": ["MYAPP_CACHE"]}), preflight=False)
    assert all(row[5] == "0" for row in first[3] + second[3])


def test_warm_profile_warms_each_arm_by_itself_and_equally(probe):
    cache = {"profile": "warm", "warmup_runs": 2, "isolate": ["MYAPP_CACHE"]}
    summary, ev, plan, log, _ = probe(dict(CROSSOVER, cache=cache), preflight=False)
    assert summary["verdict"] == "PASS", summary["reasons"]
    assert plan["scope"]["untimed_warmup_runs_min"] == 16
    runs = [row for row in log if row[0] == "run"]
    warmups = [row for row in log if row[0] == "warmup"]
    assert len(runs) == 8 and len(warmups) == 16
    # Each timed run finds exactly its own two warm-up runs, in every layer:
    # nothing from the other arm, the other order or the previous attempt.
    assert {row[5] for row in runs} == {str(2 * len(LAYER_NAMES))}
    assert sorted({row[5] for row in warmups}) == ["0", str(len(LAYER_NAMES))]
    for a in ev["execution"]["attempts"]:
        assert [w["status"] for w in a["isolation"]["warmup"]] == ["ok", "ok"]
    assert ev["observations"]["count"] == 8 * 6       # warm-up output is not evidence
    assert "## Run order and caches" in open(os.path.join(summary["run_dir"], "report.md")).read()
    assert bundle.verify(summary["run_dir"])["ok"]


def test_a_retry_reruns_the_whole_sequence_from_fresh_caches(probe):
    summary, ev, plan, log, _ = probe(dict(CROSSOVER, cache={"isolate": ["MYAPP_CACHE"]}),
                                      fail="once", retries=1, preflight=False, seeds=(0,))
    assert summary["verdict"] == "WARN"                # one seed: no interval
    attempts = ev["execution"]["attempts"]
    discarded = [a for a in attempts if a.get("discarded")]
    kept = [a for a in attempts if not a.get("discarded")]
    assert len(kept) == 4 and {a["try"] for a in discarded} == {1}
    assert any(a["status"] == "failed" for a in discarded)
    assert all(a["sequence"] == discarded[0]["sequence"] for a in discarded)
    retried = [a for a in kept if a["sequence"] == discarded[0]["sequence"]]
    assert [a["try"] for a in retried] == [2, 2] and [a["period"] for a in retried] == [1, 2]
    assert all(row[5] == "0" for row in log)           # the retry was cold too
    assert ev["observations"]["count"] == 4 * 6        # nothing from the abandoned try
    check = bundle.verify(summary["run_dir"])
    assert check["ok"], check["problems"]
    report = open(os.path.join(summary["run_dir"], "report.md")).read()
    assert "(discarded)" in report


def test_exhausted_retries_leave_the_block_incomplete(probe):
    summary, ev, *_ = probe(dict(CROSSOVER), fail="once", retries=0, preflight=False, seeds=(0,))
    assert summary["execution_status"] == "FAILED" and summary["verdict"] == "INCOMPLETE"
    assert "retry limit reached" in summary["detail"]
    assert all(a.get("discarded") for a in ev["execution"]["attempts"][-1:])
    assert bundle.verify(summary["run_dir"])["ok"]


def test_a_failed_reset_hook_stops_the_run_instead_of_measuring(probe):
    hooks = [{"id": "flush", "command": "exit 7", "timeout_s": 10}]
    summary, ev, plan, log, _ = probe(dict(CROSSOVER, reset_hooks=hooks), preflight=False, seeds=(0,))
    assert summary["execution_status"] == "FAILED" and summary["verdict"] == "INCOMPLETE"
    attempt = ev["execution"]["attempts"][0]
    assert attempt["status"] == "reset_failed" and "flush" in attempt["error"]
    assert attempt["isolation"]["reset_hooks"] == [
        {"id": "flush", "status": "failed", "duration_s": attempt["isolation"]["reset_hooks"][0]["duration_s"]}]
    assert [row for row in log if row[0] == "run"] == []   # nothing was timed
    assert any("flush" in w for w in plan["warnings"])


def test_a_killed_attempt_leaves_no_cache_behind(probe):
    summary, ev, *_ = probe(dict(CROSSOVER), fail="hang", timeout_s=2, preflight=False, seeds=(0,))
    assert summary["execution_status"] == "TIMEOUT"
    assert ev["execution"]["attempts"][0]["isolation"]["cleanup"] == "removed"
    assert _leftover_caches(summary["run_dir"]) == []


def test_cancelling_a_run_removes_the_cache_of_the_attempt_in_flight(probe, cli):
    path, shared = probe(dict(CROSSOVER), fail="hang", preflight=False, seeds=(0,), start=False)
    project = os.path.dirname(path)
    proc = subprocess.Popen([sys.executable, "-m", "remyx_validate.cli", "run"], cwd=project,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        log = os.path.join(shared, "log")
        deadline = time.time() + 60
        while not os.path.exists(log) and time.time() < deadline:
            time.sleep(0.1)
        assert os.path.exists(log), "the first attempt never started"
        runs = os.path.join(project, ".validate", "runs")
        assert _leftover_caches(runs), "the attempt in flight should own a cache directory"
        assert cli("cancel", cwd=project).returncode == 0
        proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    run_dir = runmod.find_run(project)
    with open(os.path.join(run_dir, "evidence.json")) as fh:
        ev = json.load(fh)
    assert ev["execution"]["status"] == "CANCELLED" and ev["comparison"]["verdict"] == "INCOMPLETE"
    assert ev["execution"]["attempts"][0]["status"] == "cancelled"
    assert _leftover_caches(run_dir) == []
    assert bundle.verify(run_dir)["ok"]


# ── the bundle ────────────────────────────────────────────────────────────

def _rewrite_evidence(run_dir, edit):
    path = os.path.join(run_dir, "evidence.json")
    with open(path) as fh:
        ev = json.load(fh)
    edit(ev)
    with open(path, "w") as fh:
        json.dump(ev, fh, indent=2, sort_keys=True)
        fh.write("\n")
    bundle.write_checksums(run_dir)


def test_verify_recomputes_the_order_decision_and_checks_the_schedule(run_scenario):
    summary, _ = run_scenario("carryover")
    assert summary["reasons"] == ["order_bias:p95_latency"]
    res = bundle.verify(summary["run_dir"])
    assert res["ok"] and res["recomputed_verdict"] == "NO_SIGNAL"
    assert res["order_control"] == "crossover, order controlled"

    def swap(ev):                                      # claim a different run order
        for a in ev["execution"]["attempts"]:
            a["sequence"] = "BA" if a["sequence"] == "AB" else "AB"

    _rewrite_evidence(summary["run_dir"], swap)
    assert not bundle.verify(summary["run_dir"])["ok"]


def test_verify_rejects_a_counted_attempt_without_a_clean_cache(run_scenario):
    summary, _ = run_scenario("crossover")
    _rewrite_evidence(summary["run_dir"],
                      lambda ev: ev["execution"]["attempts"][0]["isolation"].update(fresh=False))
    problems = bundle.verify(summary["run_dir"])["problems"]
    assert any("clean cache" in p for p in problems)


def test_carryover_is_caught_and_a_reset_hook_removes_it(run_scenario):
    """A shared service the runner's caches do not cover. Without a reset the
    two orders disagree by twice the speedup. With the declared hook they agree."""
    bad, _ = run_scenario("carryover")
    order = {c["id"]: c for c in load_evidence(bad)["comparison"]["criteria"]}["p95_latency"]["order"]
    assert order["sensitivity"] == pytest.approx(-80.0) and order["state"] == "exceeds_tolerance"
    assert order["first_position_delta"] == pytest.approx(order["second_position_delta"])
    good, _ = run_scenario("reset-hook")
    ev = load_evidence(good)
    order = {c["id"]: c for c in ev["comparison"]["criteria"]}["p95_latency"]["order"]
    assert good["verdict"] == "PASS" and order["sensitivity"] == pytest.approx(0.0, abs=1e-9)
    hooks = [h for a in ev["execution"]["attempts"] for h in a["isolation"]["reset_hooks"]]
    assert len(hooks) == 4 and {h["status"] for h in hooks} == {"ok"}


def test_cold_and_warm_reports_differ_only_by_the_declared_profile(run_scenario):
    cold, _ = run_scenario("crossover")
    warm, _ = run_scenario("warm")
    lat = lambda s: {c["id"]: c for c in load_evidence(s)["comparison"]["criteria"]}["p95_latency"]  # noqa: E731
    assert lat(cold)["delta"] == pytest.approx(lat(warm)["delta"])
    assert lat(warm)["baseline"] == pytest.approx(lat(cold)["baseline"] - 40.0)   # warm is faster
    for summary, profile in ((cold, "cold"), (warm, "warm")):
        text = open(os.path.join(summary["run_dir"], "report.md")).read()
        assert f"cache profile: {profile}" in text and "not controlled by the runner" in text


def test_runs_without_a_protocol_say_they_are_not_order_controlled(run_scenario):
    summary, _ = run_scenario("positive")
    ev = load_evidence(summary)
    assert ev["execution"]["protocol"]["order_controlled"] is False
    assert "schedule" not in ev["execution"]
    assert "**Not order-controlled.**" in open(os.path.join(summary["run_dir"], "report.md")).read()
    assert bundle.verify(summary["run_dir"])["order_control"] == "alternating, order not controlled"


def test_plan_states_the_schedule_and_its_cost_before_approval(tmp_path, cli):
    path = demo.create(str(tmp_path / "p"), "warm")
    plan = planmod.build(path)
    assert plan["scope"]["attempts_min"] == 4 and plan["scope"]["untimed_warmup_runs_min"] == 4
    assert plan["schedule"] == decl.planned_schedule(plan["declaration"])
    with open(path) as fh:
        text = fh.read()
    with open(path, "w") as fh:
        fh.write(text.replace("schedule_seed: 7 ", "schedule_seed: 8 "))
    assert planmod.build(path)["plan_digest"] != plan["plan_digest"]
    shown = cli("plan", cwd=os.path.dirname(path)).stdout
    assert "crossover" in shown and "schedule" in shown and "warm profile" in shown


def test_init_template_documents_the_protocol():
    from remyx_validate import cli as climod
    assert "measurement:" in climod.INIT_TEMPLATE and "order_tolerance" in climod.INIT_TEMPLATE
    assert textwrap.dedent(climod.INIT_TEMPLATE)       # still a plain string


def test_docker_attempts_get_the_same_isolation_and_warm_up(run_scenario):
    from conftest import has_docker
    if not has_docker():
        pytest.skip("docker not available")
    summary, _ = run_scenario("warm", edit=lambda text: text.replace(
        "backend: local\n  trust: trusted",
        "backend: docker\n  trust: untrusted\n  image: python:3.12-slim"))
    assert summary["verdict"] == "PASS", (summary["reasons"], summary["detail"])
    ev = load_evidence(summary)
    lat = {c["id"]: c for c in ev["comparison"]["criteria"]}["p95_latency"]
    assert lat["order"]["sensitivity"] == pytest.approx(0.0, abs=1e-9)
    assert all(a["isolation"]["fresh"] and a["isolation"]["cleanup"] == "removed"
               for a in ev["execution"]["attempts"])
    # Host paths are never handed to a container.
    assert ev["execution"]["protocol"]["shared_immutable_assets"] == {}
    assert bundle.verify(summary["run_dir"])["ok"]
