"""Odds conversion and overround removal (devigging).

Bookmakers' quoted prices do not sum to 100%. A metro Australian win market
typically books to 115-130%; Betfair sits near 101-102%. That excess is the
operator's margin, and it must be stripped out before the market's opinion
can be used as a probability.

How you strip it matters more than most people assume, because the margin
is not spread evenly. The favourite-longshot bias means far more margin is
loaded onto a 50/1 shot than onto a 2/1 favourite. Three methods are
provided:

  proportional  p_i = q_i / sum(q)
                Simple, standard, and biased: it assumes uniform margin,
                so it overstates longshots and understates favourites.

  power         p_i = q_i^k, with k solved so the probabilities sum to 1.
                Because q_i < 1, raising to k > 1 shrinks longshots more
                than favourites -- exactly the correction required. Cannot
                produce a probability outside [0, 1]. Clarke, Kovalchik &
                Ingram (2017) found it beat the multiplicative method
                universally across three large bookmaker datasets.

  shin          Assumes a fraction z of money comes from insiders and the
                bookmaker prices defensively against them. Solve for z so
                the probabilities sum to 1. Designed with racing in mind;
                Strumbelj (2014) found it more accurate than basic
                normalisation.

DEFAULTS
--------
Shin for bookmaker prices, proportional for exchange prices (where the book
is already near 100% and there is little margin structure to unwind).
Which is genuinely best is an empirical question for your data -- use
`compare_methods` to settle it by out-of-sample log loss rather than taste.
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Sequence

import numpy as np
from scipy.optimize import brentq

Method = Literal["proportional", "power", "shin"]


def decimal_to_probability(odds: Sequence[Optional[float]]) -> np.ndarray:
    """Raw implied probabilities, 1/odds. Missing prices become NaN."""
    values = []
    for price in odds:
        if price is None or price <= 1.0 or not math.isfinite(price):
            values.append(float("nan"))
        else:
            values.append(1.0 / price)
    return np.array(values, dtype=float)


def booksum(odds: Sequence[Optional[float]]) -> float:
    """Total book percentage. 1.18 means an 18% overround."""
    raw = decimal_to_probability(odds)
    return float(np.nansum(raw))


def overround(odds: Sequence[Optional[float]]) -> float:
    return booksum(odds) - 1.0


# --------------------------------------------------------------------------
# Devigging methods
# --------------------------------------------------------------------------

def _proportional(raw: np.ndarray) -> np.ndarray:
    total = np.nansum(raw)
    if total <= 0:
        return raw
    return raw / total


def _power(raw: np.ndarray) -> np.ndarray:
    """Solve sum(q_i^k) = 1 for k."""
    valid = np.isfinite(raw) & (raw > 0)
    if valid.sum() < 2:
        return _proportional(raw)

    q = raw[valid]

    def excess(k: float) -> float:
        return float(np.sum(q ** k) - 1.0)

    try:
        # sum(q^k) decreases monotonically in k for q < 1, so the root is
        # bracketed once the upper bound drives the sum below 1.
        lower, upper = 0.5, 1.0
        for _ in range(60):
            if excess(upper) < 0:
                break
            upper *= 1.5
        else:
            return _proportional(raw)
        k = brentq(excess, lower, upper, maxiter=200)
    except (ValueError, RuntimeError):
        return _proportional(raw)

    out = np.full_like(raw, float("nan"))
    out[valid] = q ** k
    return out


def _shin(raw: np.ndarray) -> np.ndarray:
    """Solve Shin's insider-trading model for z."""
    valid = np.isfinite(raw) & (raw > 0)
    if valid.sum() < 2:
        return _proportional(raw)

    q = raw[valid]
    total = float(q.sum())
    if total <= 1.0:
        return _proportional(raw)

    def probs_for(z: float) -> np.ndarray:
        # As z -> 0 the expression tends to q / sqrt(booksum), NOT q /
        # booksum: the 4(1-z) term inside the root leaves a square root of
        # the normaliser behind. Getting this limit wrong makes the solver
        # think there is no root and silently fall back to proportional.
        if z <= 1e-12:
            return q / math.sqrt(total)
        inner = z * z + 4.0 * (1.0 - z) * (q * q) / total
        return (np.sqrt(inner) - z) / (2.0 * (1.0 - z))

    def excess(z: float) -> float:
        return float(probs_for(z).sum() - 1.0)

    try:
        if excess(0.0) <= 0:
            return _proportional(raw)
        upper = 0.99
        if excess(upper) > 0:
            return _proportional(raw)
        z = brentq(excess, 0.0, upper, maxiter=200)
    except (ValueError, RuntimeError):
        return _proportional(raw)

    out = np.full_like(raw, float("nan"))
    out[valid] = probs_for(z)
    return out


_METHODS = {
    "proportional": _proportional,
    "power": _power,
    "shin": _shin,
}


def devig(odds: Sequence[Optional[float]], method: Method = "shin") -> np.ndarray:
    """Convert decimal odds to fair probabilities summing to 1.

    Runners with no price get NaN and are excluded from the normalisation,
    which is the correct treatment: an unpriced runner is unknown, not
    impossible.
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown devig method {method!r}; "
                         f"choose from {sorted(_METHODS)}")
    raw = decimal_to_probability(odds)
    if not np.any(np.isfinite(raw)):
        return raw

    result = _METHODS[method](raw)

    # Renormalise defensively: power and Shin solve to sum 1 numerically,
    # but rounding and fallbacks can leave a small residual.
    total = np.nansum(result)
    if total > 0:
        result = result / total
    return result


def shin_insider_fraction(odds: Sequence[Optional[float]]) -> Optional[float]:
    """The fitted z from Shin's model -- the estimated share of money
    coming from informed traders. Typically 0.01-0.05 in liquid markets,
    higher in thin ones. Useful as a market-quality diagnostic."""
    raw = decimal_to_probability(odds)
    valid = np.isfinite(raw) & (raw > 0)
    if valid.sum() < 2:
        return None
    q = raw[valid]
    total = float(q.sum())
    if total <= 1.0:
        return 0.0

    def excess(z: float) -> float:
        if z <= 1e-12:
            return float((q / math.sqrt(total)).sum() - 1.0)
        inner = z * z + 4.0 * (1.0 - z) * (q * q) / total
        return float(((np.sqrt(inner) - z) / (2.0 * (1.0 - z))).sum() - 1.0)

    try:
        if excess(0.0) <= 0 or excess(0.99) > 0:
            return None
        return float(brentq(excess, 0.0, 0.99, maxiter=200))
    except (ValueError, RuntimeError):
        return None


# --------------------------------------------------------------------------
# Empirical method selection
# --------------------------------------------------------------------------

def compare_methods(races_odds: Sequence[Sequence[Optional[float]]],
                    winner_indices: Sequence[int]) -> dict[str, float]:
    """Mean negative log-likelihood of the actual winners under each method.

    Lower is better. Run this over a large historical sample and use
    whichever method wins on your own data rather than trusting a default.
    """
    scores: dict[str, float] = {}
    for name in _METHODS:
        total, count = 0.0, 0
        for odds, winner in zip(races_odds, winner_indices):
            probs = devig(odds, name)  # type: ignore[arg-type]
            if winner >= len(probs):
                continue
            p = probs[winner]
            if not np.isfinite(p) or p <= 0:
                continue
            total += -math.log(p)
            count += 1
        scores[name] = total / count if count else float("inf")
    return scores


# --------------------------------------------------------------------------
# Venue economics
# --------------------------------------------------------------------------

def betfair_commission(state: Optional[str] = None) -> float:
    """Betfair Australia market base rate, charged on net market winnings.

    8% on Australian racing generally; 10% for NSW and ACT racing because
    of the race-fields fee structure. These are configurable because
    Betfair has changed them before and will again -- the AU rate rose
    from 7% to 8% in March 2025.
    """
    if state and state.upper() in ("NSW", "ACT"):
        return 0.10
    return 0.08


# Australian national tote takeout by pool. Verify against your state's
# totalizator rules; these move.
TOTE_TAKEOUT = {
    "win": 0.145,
    "place": 0.145,
    "quinella": 0.20,
    "exacta": 0.20,
    "trifecta": 0.215,
    "first_four": 0.23,
    "quaddie": 0.205,
    "running_double": 0.20,
}


def breakeven_probability(odds: float, commission: float = 0.0) -> float:
    """Win probability needed to break even at these odds.

    With exchange commission on winnings, the effective net odds shrink,
    so the required probability rises.
    """
    if odds <= 1.0:
        return 1.0
    net = (odds - 1.0) * (1.0 - commission)
    return 1.0 / (1.0 + net)


def tote_dividend(pool_total: float, runner_pool: float, stake: float,
                  takeout: float) -> float:
    """Tote dividend per $1, accounting for your own stake diluting the pool.

    This is why parimutuel betting scales badly: your money goes into the
    same pool you are trying to win from, so a strategy that is profitable
    at $50 can be losing at $1,000 in a thin country pool. Model it rather
    than assuming the displayed approximate price.
    """
    if runner_pool + stake <= 0:
        return 0.0
    return (1.0 - takeout) * (pool_total + stake) / (runner_pool + stake)
