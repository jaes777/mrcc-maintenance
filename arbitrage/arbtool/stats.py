"""Small, dependency-free statistics used across the system.

Deliberately plain Python: the whole tool runs on a stock Python install with
nothing to download, which matters more here than raw speed.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


def mean(xs: Sequence[float]) -> float:
    if not xs:
        raise ValueError("mean of empty sequence")
    return sum(xs) / len(xs)


def stdev(xs: Sequence[float], ddof: int = 1) -> float:
    n = len(xs)
    if n - ddof <= 0:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - ddof))


def median(xs: Sequence[float]) -> float:
    if not xs:
        raise ValueError("median of empty sequence")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def percentile(xs: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile, ``p`` in [0, 100]."""
    if not xs:
        raise ValueError("percentile of empty sequence")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * max(0.0, min(100.0, p)) / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def zscore(value: float, window: Sequence[float]) -> Optional[float]:
    """How unusual ``value`` is versus ``window``, in standard deviations."""
    if len(window) < 3:
        return None
    sd = stdev(window)
    if sd <= 1e-12:
        return None
    return (value - mean(window)) / sd


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    xs, ys = xs[-n:], ys[-n:]
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx <= 1e-12 or dy <= 1e-12:
        return 0.0
    return num / (dx * dy)


def ols(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    """Least-squares fit ``y = intercept + slope * x``."""
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0, 0.0
    xs, ys = xs[-n:], ys[-n:]
    mx, my = mean(xs), mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 1e-12:
        return my, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return my - slope * mx, slope


def half_life(series: Sequence[float]) -> Optional[float]:
    """Estimated days for a deviation to decay halfway back to its mean.

    Fits the Ornstein-Uhlenbeck-style regression ``Δs_t = a + b * s_{t-1}``.
    A negative ``b`` means the series pulls back towards its mean; the half-life
    is ``-ln(2) / ln(1 + b)``. ``None`` means "no evidence of mean reversion",
    which is a reason to leave the pair alone.
    """
    if len(series) < 20:
        return None
    lagged = list(series[:-1])
    deltas = [series[i + 1] - series[i] for i in range(len(series) - 1)]
    _, b = ols(lagged, deltas)
    if b >= -1e-9 or (1.0 + b) <= 0:
        return None
    hl = -math.log(2.0) / math.log(1.0 + b)
    if not math.isfinite(hl) or hl <= 0:
        return None
    return hl


def adf_like_stat(series: Sequence[float]) -> Optional[float]:
    """A Dickey-Fuller style t-statistic on the mean-reversion coefficient.

    Not a substitute for a full ADF test with proper critical values, but the
    sign and magnitude are informative: values below roughly -2.9 correspond to
    the conventional 5% rejection region for a unit root, i.e. evidence that the
    spread is stationary rather than wandering away forever.
    """
    n = len(series)
    if n < 30:
        return None
    lagged = list(series[:-1])
    deltas = [series[i + 1] - series[i] for i in range(n - 1)]
    a, b = ols(lagged, deltas)
    resid = [d - (a + b * l) for d, l in zip(deltas, lagged)]
    dof = len(deltas) - 2
    if dof <= 0:
        return None
    sigma2 = sum(r * r for r in resid) / dof
    mx = mean(lagged)
    sxx = sum((x - mx) ** 2 for x in lagged)
    if sxx <= 1e-12 or sigma2 <= 1e-18:
        return None
    se = math.sqrt(sigma2 / sxx)
    if se <= 1e-18:
        return None
    return b / se


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Confidence interval for a win rate.

    The plain ``wins / trades`` figure is a point estimate that says nothing
    about sample size: 6 wins from 10 and 600 from 1000 both read "60%". The
    Wilson interval keeps the sample size visible, and the system gates on the
    lower bound so small samples cannot pass by luck.
    """
    if trials <= 0:
        return 0.0, 1.0
    z = _z_for_confidence(confidence)
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = p + z * z / (2 * trials)
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return max(0.0, lo), min(1.0, hi)


def _z_for_confidence(confidence: float) -> float:
    table = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.975: 2.2414, 0.99: 2.5758}
    best = min(table, key=lambda c: abs(c - confidence))
    return table[best]


def binomial_tail_pvalue(successes: int, trials: int, p0: float) -> float:
    """P(X >= successes) under Binomial(trials, p0).

    Answers "could this win rate have happened by chance if the true edge were
    only p0?" Small values mean the result is hard to explain as luck.
    """
    if trials <= 0:
        return 1.0
    successes = max(0, min(successes, trials))
    total = 0.0
    for k in range(successes, trials + 1):
        total += math.comb(trials, k) * (p0 ** k) * ((1 - p0) ** (trials - k))
    return min(1.0, max(0.0, total))


def max_drawdown(equity: Sequence[float]) -> float:
    """Worst peak-to-trough fall, as a positive percentage."""
    if not equity:
        return 0.0
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst * 100.0


def sharpe(returns: Sequence[float], periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    sd = stdev(returns)
    if sd <= 1e-12:
        return 0.0
    return mean(returns) / sd * math.sqrt(periods_per_year)


def rolling(series: Sequence[float], window: int) -> List[List[float]]:
    return [list(series[max(0, i - window + 1): i + 1]) for i in range(len(series))]
