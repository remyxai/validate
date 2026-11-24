"""Estimators checked against values computed independently (scipy / numpy,
recorded here so the suite has no numeric dependencies)."""
import math

import pytest

from remyx_validate import stats


@pytest.mark.parametrize("p, df, expected", [
    (0.975, 1, 12.706204736),
    (0.975, 4, 2.776445105),
    (0.975, 9, 2.262157163),
    (0.975, 239, 1.969939406),
    (0.95, 10, 1.812461123),
    (0.995, 2, 9.924843201),
    (0.975, 2, 4.302652730),
    (0.999, 1, 318.308838986),
    (0.9995, 3, 12.923978637),
    (0.9995, 1000, 3.300283),
])
def test_t_ppf(p, df, expected):
    assert stats.t_ppf(p, df) == pytest.approx(expected, rel=1e-6)


def test_quantile_matches_numpy_linear():
    xs = [3, 1, 4, 1, 5, 9, 2, 6]
    assert stats.quantile(xs, 0.5) == pytest.approx(3.5)
    assert stats.quantile(xs, 0.95) == pytest.approx(7.95)
    assert stats.quantile([7], 0.95) == 7


def test_paired_t_interval_hand_checked():
    base = [10.0, 12.0, 9.0, 11.0, 13.0]
    cand = [11.0, 14.0, 9.5, 13.0, 13.5]
    # diffs 1, 2, .5, 2, .5 -> mean 1.2, sd 0.758288, se 0.339116, t(.975, 4) 2.776445
    s = stats.paired_summary(base, cand, "mean", 0.95, 0)
    assert s["delta"] == pytest.approx(1.2)
    assert s["delta_interval"][0] == pytest.approx(1.2 - 2.776445 * 0.339116, abs=1e-5)
    assert s["delta_interval"][1] == pytest.approx(1.2 + 2.776445 * 0.339116, abs=1e-5)
    assert s["estimator"] == "paired_mean_difference_t"


def test_no_interval_is_invented():
    one = stats.paired_summary([1.0], [2.0], "mean", 0.95, 0)
    assert one["delta"] == 1.0 and one["delta_interval"] is None and one["estimator"] == "point"
    few = stats.paired_summary([1.0] * 5, [2.0] * 5, "p95", 0.95, 0)
    assert few["delta_interval"] is None
    none = stats.paired_summary([], [], "mean", 0.95, 0)
    assert none["delta"] is None


def test_bootstrap_is_deterministic_and_seed_dependent():
    base = [float(i % 7) for i in range(60)]
    cand = [b + (i % 3) for i, b in enumerate(base)]
    a = stats.paired_summary(base, cand, "p95", 0.95, 123)
    b = stats.paired_summary(base, cand, "p95", 0.95, 123)
    assert a == b and a["estimator"] == "paired_bootstrap_percentile"
    assert a["delta_interval"][0] <= a["delta"] <= a["delta_interval"][1]


def test_unknown_aggregate():
    with pytest.raises(ValueError):
        stats.aggregate_fn("p150")
    assert stats.aggregate_fn("sum")([1, 2]) == 3 and not math.isnan(stats.aggregate_fn("median")([1, 2]))
