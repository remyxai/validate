"""Comparison policy on hand-built observations (no subprocesses)."""
import copy

import yaml

from remyx_validate import compare, declaration as decl, demo


def _decl(**method):
    d = yaml.safe_load(demo.declaration_text("guardrail"))
    d["method"].update({"unit": "seed", "seeds": [0, 1, 2, 3, 4], **method})
    d["criteria"] = [{"id": "loss", "metric": "loss", "role": "target", "direction": "lower"}]
    d["data"] = {"identity": "hand-built"}
    d["validity"] = {"activation": [{"metric": "witness"}], "held_constant": ["init_loss"]}
    return decl.normalize(d)


def _unit_obs(cells, metric="loss", slices=None):
    """cells: {(arm, seed, unit): value}. Adds a passing witness and control per (arm, seed)."""
    rows = []
    for (arm, seed, unit), value in cells.items():
        rows.append({"arm": arm, "seed": seed, "unit": unit,
                     "slice": (slices or {}).get((arm, seed, unit)),
                     "metric": metric, "value": float(value), "attempt_id": f"att_{arm}_s{seed}_t1"})
    for arm, seed in sorted({(a, s) for a, s, _ in cells}):
        for name, v in (("witness", arm == "candidate"), ("init_loss", 1.0)):
            rows.append({"arm": arm, "seed": seed, "unit": None, "slice": None, "metric": name,
                         "value": float(v), "attempt_id": f"att_{arm}_s{seed}_t1"})
    return rows


def _example_decl(seeds, **data):
    d = _decl(unit="example", seeds=seeds)
    d["data"] = dict(d.get("data", {}), **data)
    return d


def _obs(baseline, candidate, init=None, witness=(0, 1), extra=None):
    rows = []
    for arm, values in (("baseline", baseline), ("candidate", candidate)):
        for seed, value in enumerate(values):
            metrics = {"loss": value, "witness": witness[arm == "candidate"],
                       "init_loss": (init or {}).get((arm, seed), 1.0 + seed)}
            metrics.update((extra or {}).get(arm, {}))
            for name, v in metrics.items():
                rows.append({"arm": arm, "seed": seed, "unit": None, "slice": None,
                             "metric": name, "value": float(v), "attempt_id": f"att_{arm}_s{seed}_t1"})
    return rows


def run(d, rows, status="COMPLETED", integrity=()):
    return compare.compare(d, rows, status, list(integrity))


def test_lower_is_better_target_met_over_seeds():
    out = run(_decl(), _obs([1.0, 1.1, 0.9, 1.05, 0.95], [0.8, 0.85, 0.7, 0.9, 0.75]))
    c = out["criteria"][0]
    assert out["verdict"] == "PASS" and c["status"] == "met"
    assert c["pairing"] == "seed" and c["n_pairs"] == 5
    assert c["delta"] < 0 and c["improvement"] > 0 and c["improvement_interval"][0] > 0


def test_confident_regression_fails_and_noise_warns():
    worse = run(_decl(), _obs([1.0, 1.1, 0.9, 1.05, 0.95], [1.2, 1.3, 1.15, 1.2, 1.1]))
    assert worse["verdict"] == "FAIL" and worse["reasons"] == ["target_not_met:loss"]
    noisy = run(_decl(), _obs([1.0, 1.1, 0.9, 1.05, 0.95], [0.7, 1.4, 0.6, 1.3, 0.9]))
    assert noisy["verdict"] == "WARN" and noisy["reasons"] == ["uncertain:loss"]


def test_held_constant_may_vary_by_seed_but_not_by_arm():
    """init_loss differs seed to seed (1.0, 2.0, …) and is equal between arms: no drift."""
    ok = run(_decl(), _obs([1.0, 1.1, 0.9, 1.05, 0.95], [0.8, 0.85, 0.7, 0.9, 0.75]))
    assert all(c["passed"] for c in ok["validity"]["checks"])
    drift = run(_decl(), _obs([1.0, 1.1, 0.9, 1.05, 0.95], [0.8, 0.85, 0.7, 0.9, 0.75],
                              init={("candidate", 3): 9.0}))
    assert drift["verdict"] == "NO_SIGNAL" and drift["reasons"] == ["control_drift:init_loss"]
    assert drift["validity"]["status"] == "INVALID"


def test_inactive_witness_is_no_signal_even_when_numbers_differ():
    out = run(_decl(), _obs([1.0, 1.1, 0.9, 1.05, 0.95], [0.8, 0.85, 0.7, 0.9, 0.75], witness=(0, 0)))
    assert out["verdict"] == "NO_SIGNAL" and out["reasons"] == ["inactive_treatment:witness"]


def test_single_seed_has_no_interval_and_cannot_pass_on_interval_rule():
    d = _decl(seeds=[0])
    out = run(d, _obs([1.0], [0.5]))
    c = out["criteria"][0]
    assert c["delta_interval"] is None and c["status"] == "uncertain" and out["verdict"] == "WARN"
    point = run(_decl(seeds=[0], uncertainty="point"), _obs([1.0], [0.5]))
    assert point["verdict"] == "PASS"


def test_relative_regression_guardrail_and_absolute_limits():
    d = _decl()
    d["criteria"].append({"id": "speed", "metric": "steps_per_s", "role": "guardrail",
                          "direction": "higher", "aggregate": "mean", "relative": True,
                          "max_regression": 0.10})
    good = [0.8, 0.85, 0.7, 0.9, 0.75]
    base = [1.0, 1.1, 0.9, 1.05, 0.95]

    def with_speed(b, c):
        rows = _obs(base, good)
        for arm, values in (("baseline", b), ("candidate", c)):
            for seed, v in enumerate(values):
                rows.append({"arm": arm, "seed": seed, "unit": None, "slice": None,
                             "metric": "steps_per_s", "value": v, "attempt_id": "a"})
        return rows

    small = run(d, with_speed([100, 101, 99, 100, 100], [97, 98, 96, 97, 97]))      # about -3%
    assert small["verdict"] == "PASS"
    big = run(d, with_speed([100, 101, 99, 100, 100], [70, 71, 69, 70, 70]))        # about -30%
    assert big["verdict"] == "FAIL" and "guardrail_veto:speed" in big["reasons"]


def test_baseline_expectation_catches_control_drift():
    d = _decl()
    d["validity"]["baseline_expect"] = [{"metric": "loss", "value": 1.0, "tolerance": 0.05}]
    ok = run(d, _obs([1.0, 1.1, 0.9, 1.05, 0.95], [0.8, 0.85, 0.7, 0.9, 0.75]))
    assert ok["verdict"] == "PASS"
    off = run(d, _obs([2.0, 2.1, 1.9, 2.05, 1.95], [0.8, 0.85, 0.7, 0.9, 0.75]))
    assert off["verdict"] == "NO_SIGNAL" and off["reasons"] == ["baseline_drift:loss"]


def test_missing_metric_and_incomplete_execution():
    d = _decl()
    rows = [r for r in _obs([1.0, 1.1, 0.9, 1.05, 0.95], [0.8, 0.85, 0.7, 0.9, 0.75])
            if not (r["arm"] == "candidate" and r["metric"] == "loss")]
    out = run(d, rows)
    assert out["verdict"] == "INCOMPLETE" and out["reasons"] == ["missing_metric:loss"]
    timed_out = run(d, [], status="TIMEOUT")
    assert timed_out["verdict"] == "INCOMPLETE" and timed_out["validity"]["status"] == "NOT_ASSESSED"


# ── coverage regressions: none of these may PASS ──────────────────────────

GOOD_B, GOOD_C = [1.0, 1.1, 0.9, 1.05, 0.95], [0.8, 0.85, 0.7, 0.9, 0.75]


def test_both_arms_missing_a_declared_seed_is_incomplete():
    out = run(_decl(), _obs(GOOD_B[:4], GOOD_C[:4]))        # seed 4 never ran in either arm
    assert out["verdict"] == "INCOMPLETE"
    assert out["criteria"][0]["coverage"] == {"basis": "declared seeds", "expected": 5,
                                              "matched": 4, "missing_rate": 0.2}


def test_mismatched_seed_subsets_are_not_averaged_together():
    """Baseline has the example on seeds 0,1; candidate on seeds 1,2. Only seed 1
    is a matched cell, so the rest is unpaired and the run is incomplete."""
    cells = {}
    for u in ("a", "b", "c"):
        for s in (0, 1):
            cells[("baseline", s, u)] = 1.0
        for s in (1, 2):
            cells[("candidate", s, u)] = 0.5
    out = run(_example_decl([0, 1, 2]), _unit_obs(cells))
    c = out["criteria"][0]
    assert out["verdict"] == "INCOMPLETE"
    assert c["coverage"]["expected"] == 9 and c["coverage"]["matched"] == 3


def test_matched_seeds_only_feed_the_example_mean():
    cells = {}
    for i, u in enumerate("abcdefghij"):
        for s in (0, 1):
            cells[("baseline", s, u)] = 1.0 + i * 0.01 + s
            cells[("candidate", s, u)] = 0.5 + i * 0.02 + s
    out = run(_example_decl([0, 1], expected_units=10), _unit_obs(cells))
    c = out["criteria"][0]
    assert out["verdict"] == "PASS" and c["n_pairs"] == 10
    assert c["coverage"]["matched"] == 20 and c["coverage"]["missing_rate"] == 0


def test_example_missing_from_both_arms_needs_a_declared_count():
    cells = {(arm, 0, u): (1.0 if arm == "baseline" else 0.5 + i * 0.01)
             for arm in ("baseline", "candidate") for i, u in enumerate("abcdefghi")}   # 9 of 10
    blind = run(_example_decl([0]), _unit_obs(cells))
    assert any("cannot be seen" in text for text in blind["limitations"])
    seen = run(_example_decl([0], expected_units=10), _unit_obs(cells))
    assert seen["verdict"] == "INCOMPLETE" and seen["reasons"] == ["unpaired_measurements:loss"]


def test_held_constant_missing_on_one_seed_is_incomplete():
    rows = [r for r in _obs(GOOD_B, GOOD_C)
            if not (r["metric"] == "init_loss" and r["seed"] == 2 and r["arm"] == "candidate")]
    out = run(_decl(), rows)
    assert out["verdict"] == "INCOMPLETE" and out["reasons"] == ["held_constant_missing:init_loss"]


def test_witness_active_on_one_of_five_seeds_is_no_signal():
    rows = _obs(GOOD_B, GOOD_C)
    for r in rows:
        if r["metric"] == "witness" and r["arm"] == "candidate" and r["seed"] != 0:
            r["value"] = 0.0
    out = run(_decl(), rows)
    assert out["verdict"] == "NO_SIGNAL" and out["reasons"] == ["inactive_treatment:witness"]
    assert "4 of 5" in out["validity"]["checks"][-1]["detail"]


def test_witness_missing_on_a_seed_is_incomplete():
    rows = [r for r in _obs(GOOD_B, GOOD_C)
            if not (r["metric"] == "witness" and r["seed"] == 3 and r["arm"] == "baseline")]
    out = run(_decl(), rows)
    assert out["verdict"] == "INCOMPLETE" and out["reasons"] == ["activation_missing:witness"]


def test_duplicate_unexpected_and_conflicting_observations_are_rejected():
    rows = _obs(GOOD_B, GOOD_C)
    dup = run(_decl(), rows + [dict(rows[0])])
    assert dup["verdict"] == "INCOMPLETE" and "observations:duplicate_observation" in dup["reasons"]
    extra = run(_decl(), rows + [dict(rows[0], seed=99)])
    assert extra["verdict"] == "INCOMPLETE" and "observations:unexpected_seed" in extra["reasons"]
    cells = {(arm, 0, u): 1.0 for arm in ("baseline", "candidate") for u in "ab"}
    slices = {("baseline", 0, "a"): "short", ("candidate", 0, "a"): "long"}
    mixed = run(_example_decl([0]), _unit_obs(cells, slices=slices))
    assert mixed["verdict"] == "INCOMPLETE" and "observations:inconsistent_slice" in mixed["reasons"]
    assert dup["validity"]["status"] == "INVALID"


def test_example_unit_does_not_silently_fall_back_to_seeds():
    d = _decl(unit="example")
    out = run(d, _obs(GOOD_B, GOOD_C))                      # loss reported per run only
    assert out["verdict"] == "INCOMPLETE" and out["reasons"] == ["missing_metric:loss"]
    d["criteria"][0]["pairing"] = "seed"                    # the explicit choice
    assert run(d, _obs(GOOD_B, GOOD_C))["verdict"] == "PASS"


def test_per_unit_values_are_aligned_within_a_seed_before_aggregating():
    """Seed mode over per-example values: the candidate dropped its two worst
    examples. Aggregating each arm over what it reported would flatter it."""
    cells = {}
    for s in range(3):
        for i, u in enumerate("abcde"):
            cells[("baseline", s, u)] = 1.0 + i
            if i < 3:
                cells[("candidate", s, u)] = 1.0 + i
    out = run(_decl(seeds=[0, 1, 2]), _unit_obs(cells))
    c = out["criteria"][0]
    assert out["verdict"] == "INCOMPLETE"
    assert c["coverage"]["units_within_seeds"] == {"expected": 15, "matched": 9}
    assert c["delta"] == 0                                  # compared on shared units only


def test_zero_pairs_is_incomplete():
    out = run(_decl(), [])
    assert out["verdict"] == "INCOMPLETE" and out["criteria"][0]["n_pairs"] == 0


def test_measured_zero_with_active_witness_is_judged_not_dropped():
    out = run(_decl(), _obs(GOOD_B, GOOD_B))
    assert out["verdict"] == "FAIL" and out["reasons"] == ["target_not_met:loss", "measured_zero"]


def test_intervals_are_not_presented_as_joint():
    d = _decl()
    d["criteria"].append({"id": "loss2", "metric": "loss", "role": "guardrail", "direction": "lower",
                          "aggregate": "mean", "relative": False, "max": 5})
    out = run(d, _obs(GOOD_B, GOOD_C))
    assert any("not a joint statement" in text for text in out["limitations"])


def test_compare_is_pure():
    d = _decl()
    rows = _obs([1.0, 1.1, 0.9, 1.05, 0.95], [0.8, 0.85, 0.7, 0.9, 0.75])
    assert run(copy.deepcopy(d), list(rows)) == run(d, list(reversed(rows)))
