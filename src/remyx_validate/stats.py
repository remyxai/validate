"""Paired estimators with no third-party numerics.

Two interval methods, chosen by the aggregate a criterion declares:

* ``mean``: Student t interval on the per-unit paired differences.
* anything else (p50, p95, min, max, sum): paired percentile bootstrap, where
  each resample draws units and keeps both arms' values for the drawn unit.

Fewer than two paired units gives no interval. Nothing here manufactures one.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Optional, Sequence

BOOTSTRAP_RESAMPLES = 2000
MIN_BOOTSTRAP_UNITS = 10


# ── Student t quantile ────────────────────────────────────────────────────

def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(ln)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_ppf(p: float, df: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e9:
            break
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ── aggregates ────────────────────────────────────────────────────────────

def mean(xs: Sequence[float]) -> float:
    return math.fsum(xs) / len(xs)


def quantile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile (the numpy default, method 7)."""
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def aggregate_fn(name: str) -> Callable[[Sequence[float]], float]:
    name = name.lower()
    if name == "mean":
        return mean
    if name == "sum":
        return math.fsum
    if name == "min":
        return min
    if name == "max":
        return max
    if name == "median":
        return lambda xs: quantile(xs, 0.5)
    if name.startswith("p") and name[1:].replace(".", "", 1).isdigit():
        q = float(name[1:]) / 100.0
        if 0.0 <= q <= 1.0:
            return lambda xs: quantile(xs, q)
    raise ValueError(f"unknown aggregate {name!r}")


# ── intervals ─────────────────────────────────────────────────────────────

def t_interval(xs: Sequence[float], confidence: float) -> Optional[tuple]:
    n = len(xs)
    if n < 2:
        return None
    m = mean(xs)
    var = math.fsum((x - m) ** 2 for x in xs) / (n - 1)
    half = t_ppf(0.5 + confidence / 2.0, n - 1) * math.sqrt(var / n)
    return (m - half, m + half)


def paired_summary(baseline: Sequence[float], candidate: Sequence[float],
                   aggregate: str, confidence: float, rng_seed: int) -> dict:
    """Point estimates and intervals for two arms measured on the same units.

    ``baseline[i]`` and ``candidate[i]`` belong to the same independent unit.
    Returns the arm values, the delta (candidate minus baseline), and an
    interval for the delta and for each arm, or ``None`` with a reason.
    """
    n = len(baseline)
    if n != len(candidate):
        raise ValueError("arms must be paired")
    agg = aggregate_fn(aggregate)
    out = {
        "n_pairs": n,
        "aggregate": aggregate,
        "baseline": agg(baseline) if n else None,
        "candidate": agg(candidate) if n else None,
        "delta": None,
        "delta_interval": None,
        "baseline_interval": None,
        "candidate_interval": None,
        "estimator": None,
        "interval_note": None,
    }
    if n == 0:
        out["interval_note"] = "no paired units"
        return out
    out["delta"] = out["candidate"] - out["baseline"]
    if n < 2:
        out["estimator"] = "point"
        out["interval_note"] = "one paired unit: no interval can be estimated"
        return out

    if aggregate.lower() == "mean":
        diffs = [c - b for b, c in zip(baseline, candidate)]
        out["estimator"] = "paired_mean_difference_t"
        out["delta"] = mean(diffs)
        out["delta_interval"] = list(t_interval(diffs, confidence))
        out["baseline_interval"] = list(t_interval(baseline, confidence))
        out["candidate_interval"] = list(t_interval(candidate, confidence))
        if n < 5:
            out["interval_note"] = (
                f"t interval on {n} paired units: wide, and it assumes roughly "
                "normal paired differences")
        return out

    if n < MIN_BOOTSTRAP_UNITS:
        out["estimator"] = "point"
        out["interval_note"] = (
            f"{aggregate} over {n} paired units: below the {MIN_BOOTSTRAP_UNITS} "
            "needed for a bootstrap interval")
        return out

    rng = random.Random(rng_seed)
    deltas, bs, cs = [], [], []
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = [rng.randrange(n) for _ in range(n)]
        b = agg([baseline[i] for i in idx])
        c = agg([candidate[i] for i in idx])
        bs.append(b)
        cs.append(c)
        deltas.append(c - b)
    a = (1.0 - confidence) / 2.0
    out["estimator"] = "paired_bootstrap_percentile"
    out["delta_interval"] = [quantile(deltas, a), quantile(deltas, 1 - a)]
    out["baseline_interval"] = [quantile(bs, a), quantile(bs, 1 - a)]
    out["candidate_interval"] = [quantile(cs, a), quantile(cs, 1 - a)]
    out["interval_note"] = (
        f"percentile bootstrap, {BOOTSTRAP_RESAMPLES} paired resamples; tail "
        "quantiles from small samples are unstable")
    return out
