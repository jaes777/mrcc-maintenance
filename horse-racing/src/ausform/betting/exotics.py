"""Multi-position probabilities and Australian exotic bet types.

Given win probabilities, we need the chance that a particular horse runs
second, or that two specific horses fill the first two placings in order,
and so on. The standard tool is Harville (1973), which models the finishing
order as sequential sampling without replacement:

    P(i 1st, j 2nd) = p_i * p_j / (1 - p_i)

THE HARVILLE BIAS, AND WHY IT MATTERS HERE
------------------------------------------
Harville is known to be wrong in a specific, systematic direction: it
*overstates* how often a strong favourite fills second or third, and
understates longshots. Empirically the favourite runs second only about
80% as often as Harville implies, and third about 65% as often.

This is not academic hair-splitting. It is exactly the calculation behind
every place bet, quinella and trifecta you might place, so an uncorrected
Harville will systematically overprice favourites in the place market --
and the place market is where beginners bet most.

The fix used here is the Lo & Bacon-Shone discounted model, which applies a
power discount at each stage:

    P(i 1st, j 2nd) = [p_i / sum_h p_h] * [p_j^L1 / sum_{h!=i} p_h^L1]

with L1, L2 < 1. Setting L1 = L2 = 1 recovers Harville exactly. Because
raising to a power below 1 compresses the distribution toward uniform, a
high-probability horse receives less of the second/third-place mass --
precisely the required correction.

Published exponents differ by dataset (roughly 0.76 in one thoroughbred
study; 0.89 and 0.80 for second and third in another), which is why they
are parameters here rather than constants. `fit_discount_exponents` will
estimate them from your own results, and you should use it.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

log = logging.getLogger(__name__)

# Below this, the fit is noise. Repeat-sampling standard deviation on
# lambda_2nd is ~0.13 at 100 races and ~0.03 at 2000.
_MIN_FIT_RACES = 2000

# Starting values from the literature. Fit your own before betting exotics.
DEFAULT_LAMBDA_2ND = 0.85
DEFAULT_LAMBDA_3RD = 0.75


# --------------------------------------------------------------------------
# Australian place rules
# --------------------------------------------------------------------------

def places_paid(starters: int) -> int:
    """How many place dividends an Australian thoroughbred race pays.

        8 or more starters -> 3 places
        5, 6 or 7 starters -> 2 places
        4 or fewer         -> no place betting

    Note for tote bets the count is fixed at declaration time, so a field
    cut from 8 to 7 by a late scratching still pays three. Fixed-odds
    operators differ; check the specific bookmaker's rules.
    """
    if starters >= 8:
        return 3
    if starters >= 5:
        return 2
    return 0


# --------------------------------------------------------------------------
# Ordered finishing probabilities
# --------------------------------------------------------------------------

def _normalise(probs: np.ndarray) -> np.ndarray:
    total = probs.sum()
    return probs / total if total > 0 else probs


def ordered_probability(
    probs: Sequence[float],
    order: Sequence[int],
    lambda_2nd: float = DEFAULT_LAMBDA_2ND,
    lambda_3rd: float = DEFAULT_LAMBDA_3RD,
) -> float:
    """P(the runners in `order` finish in exactly that order).

    Stage 1 uses the raw win probabilities; stage 2 applies the
    `lambda_2nd` discount; stage 3 and beyond apply `lambda_3rd`.
    """
    p = np.asarray(probs, dtype=float)
    remaining = list(range(len(p)))
    result = 1.0

    for stage, runner in enumerate(order):
        if runner not in remaining:
            return 0.0
        exponent = 1.0 if stage == 0 else (lambda_2nd if stage == 1 else lambda_3rd)
        pool = np.power(np.clip(p[remaining], 1e-12, None), exponent)
        total = pool.sum()
        if total <= 0:
            return 0.0
        result *= float(pool[remaining.index(runner)] / total)
        remaining.remove(runner)

    return result


def position_probabilities(
    probs: Sequence[float],
    max_position: int = 3,
    lambda_2nd: float = DEFAULT_LAMBDA_2ND,
    lambda_3rd: float = DEFAULT_LAMBDA_3RD,
    monte_carlo_threshold: int = 14,
    n_samples: int = 40000,
    seed: int = 0,
) -> np.ndarray:
    """Matrix of P(runner i finishes in position k), shape (n, max_position).

    Exact enumeration is used for small fields; beyond
    `monte_carlo_threshold` runners the number of ordered tuples explodes,
    so we sample the sequential process instead. The sampler draws from the
    same discounted model, so the two paths agree up to Monte Carlo error.
    """
    p = _normalise(np.asarray(probs, dtype=float))
    n = len(p)
    if n == 0:
        return np.zeros((0, max_position))

    depth = min(max_position, n)
    result = np.zeros((n, max_position))

    # Depth 3 covers every place market in Australian racing and is by far
    # the hottest path, so it gets a closed-form vectorised solution
    # instead of enumerating n*(n-1)*(n-2) orderings in Python. Same
    # numbers, orders of magnitude faster.
    if depth <= 3:
        return _positions_fast(p, max_position, depth, lambda_2nd, lambda_3rd)

    if n <= monte_carlo_threshold:
        for order in itertools.permutations(range(n), depth):
            probability = ordered_probability(p, order, lambda_2nd, lambda_3rd)
            if probability <= 0:
                continue
            for position, runner in enumerate(order):
                result[runner, position] += probability
        return result

    rng = np.random.default_rng(seed)
    for _ in range(n_samples):
        remaining = list(range(n))
        for stage in range(depth):
            exponent = 1.0 if stage == 0 else (lambda_2nd if stage == 1 else lambda_3rd)
            pool = np.power(np.clip(p[remaining], 1e-12, None), exponent)
            pool = pool / pool.sum()
            pick = rng.choice(len(remaining), p=pool)
            result[remaining[pick], stage] += 1
            remaining.pop(pick)
    return result / n_samples


def _positions_fast(p: np.ndarray, max_position: int, depth: int,
                    lambda_2nd: float, lambda_3rd: float) -> np.ndarray:
    """Closed-form position probabilities for the first three placings.

    Derivation, writing q = p^lambda_2nd and r = p^lambda_3rd:

      P(i 1st) = p_i

      P(i 2nd) = sum_{j != i} p_j * q_i / (S_q - q_j)
               = q_i * [ A - p_i / (S_q - q_i) ],  A = sum_j p_j/(S_q - q_j)

      P(i 3rd) = sum_{j != i} sum_{k != i,j}
                   p_j * q_k/(S_q - q_j) * r_i/(S_r - r_j - r_k)

    The first two are O(n); the third is O(n^2) and fully vectorised.
    """
    n = len(p)
    result = np.zeros((n, max_position))
    result[:, 0] = p
    if depth == 1 or n < 2:
        return result

    q = np.power(np.clip(p, 1e-12, None), lambda_2nd)
    sum_q = q.sum()
    denom_q = sum_q - q                      # denominator once j is removed
    safe_q = np.where(np.abs(denom_q) < 1e-15, 1e-15, denom_q)

    a_total = float(np.sum(p / safe_q))
    result[:, 1] = q * (a_total - p / safe_q)

    if depth == 2 or n < 3 or max_position < 3:
        return result

    r = np.power(np.clip(p, 1e-12, None), lambda_3rd)
    sum_r = r.sum()

    # weight[j, k] = P(j 1st) * P(k 2nd | j 1st), zero on the diagonal.
    weight = (p[:, None] * q[None, :]) / safe_q[:, None]
    np.fill_diagonal(weight, 0.0)

    # Denominator for third place once j and k are gone.
    third_denom = sum_r - r[:, None] - r[None, :]
    third_denom = np.where(np.abs(third_denom) < 1e-15, 1e-15, third_denom)

    scaled = weight / third_denom            # (j, k)
    column_total = scaled.sum()              # over all valid (j, k)
    per_j = scaled.sum(axis=1)               # sum over k, for each j
    per_k = scaled.sum(axis=0)               # sum over j, for each k

    # Exclude the terms where i equals j or k, then weight by r_i.
    result[:, 2] = r * (column_total - per_j - per_k + np.diagonal(scaled))
    return np.clip(result, 0.0, 1.0)


def place_probabilities(
    probs: Sequence[float],
    starters: Optional[int] = None,
    lambda_2nd: float = DEFAULT_LAMBDA_2ND,
    lambda_3rd: float = DEFAULT_LAMBDA_3RD,
    **kwargs,
) -> np.ndarray:
    """P(each runner finishes within the paying places), using AU rules."""
    p = np.asarray(probs, dtype=float)
    n_places = places_paid(starters if starters is not None else len(p))
    if n_places == 0:
        return np.zeros(len(p))
    positions = position_probabilities(
        p, max_position=n_places, lambda_2nd=lambda_2nd,
        lambda_3rd=lambda_3rd, **kwargs)
    return positions.sum(axis=1)


# --------------------------------------------------------------------------
# Exotic bet probabilities
# --------------------------------------------------------------------------

def exacta_probability(probs, first: int, second: int, **kwargs) -> float:
    """First and second in exact order."""
    return ordered_probability(probs, [first, second], **kwargs)


def quinella_probability(probs, a: int, b: int, **kwargs) -> float:
    """First and second in either order."""
    return (ordered_probability(probs, [a, b], **kwargs)
            + ordered_probability(probs, [b, a], **kwargs))


def trifecta_probability(probs, first: int, second: int, third: int, **kwargs) -> float:
    return ordered_probability(probs, [first, second, third], **kwargs)


def first_four_probability(probs, a: int, b: int, c: int, d: int, **kwargs) -> float:
    return ordered_probability(probs, [a, b, c, d], **kwargs)


def multi_leg_probability(leg_probabilities: Iterable[float]) -> float:
    """Quaddie / running double: independent legs multiply.

    Independence is a genuine assumption. It is usually reasonable across
    different races, but breaks down when the same conditions drive several
    legs -- a track that turns heavy mid-afternoon shifts every remaining
    race toward wet-track horses at once.
    """
    result = 1.0
    for probability in leg_probabilities:
        result *= probability
    return result


# --------------------------------------------------------------------------
# Combinations and cost
# --------------------------------------------------------------------------

def box_combinations(bet_type: str, k: int) -> int:
    """Number of combinations in a boxed bet of `k` selections."""
    positions = {"quinella": 2, "exacta": 2, "trifecta": 3, "first_four": 4}
    if bet_type not in positions:
        raise ValueError(f"Unknown exotic {bet_type!r}")
    depth = positions[bet_type]
    if k < depth:
        return 0
    if bet_type == "quinella":  # unordered
        return k * (k - 1) // 2
    result = 1
    for i in range(depth):
        result *= (k - i)
    return result


def box_cost(bet_type: str, k: int, unit: float = 1.0) -> float:
    return box_combinations(bet_type, k) * unit


@dataclass
class FlexiBet:
    """Flexi (percentage) betting, standard on Australian totes.

    You nominate what you want to spend; the tote gives you that fraction
    of every combination, and pays you that fraction of the dividend. It is
    how anyone bets a first four without needing $1,680.
    """

    bet_type: str
    n_selections: int
    outlay: float
    unit: float = 1.0

    @property
    def combinations(self) -> int:
        return box_combinations(self.bet_type, self.n_selections)

    @property
    def full_cost(self) -> float:
        return self.combinations * self.unit

    @property
    def percentage(self) -> float:
        if self.full_cost <= 0:
            return 0.0
        return self.outlay / self.full_cost

    def payout(self, dividend: float) -> float:
        """What a winning combination actually returns you."""
        return self.percentage * dividend

    def describe(self) -> str:
        return (f"{self.bet_type} box of {self.n_selections} = "
                f"{self.combinations} combos, full cost ${self.full_cost:,.2f}; "
                f"${self.outlay:,.2f} outlay = {self.percentage:.1%} of the dividend")


# --------------------------------------------------------------------------
# Fitting the discount exponents
# --------------------------------------------------------------------------

def fit_discount_exponents(
    races: Sequence[tuple[Sequence[float], Sequence[int]]],
    initial: tuple[float, float] = (DEFAULT_LAMBDA_2ND, DEFAULT_LAMBDA_3RD),
) -> tuple[float, float]:
    """Estimate lambda_2nd and lambda_3rd from observed finishing orders.

    `races` is a sequence of (win_probabilities, actual_finish_order),
    where the order lists runner indices from first to third.

    Do run this. The published exponents vary by dataset, and using
    somebody else's numbers on your data is how you end up systematically
    overpricing favourites in the place market.
    """
    usable = [(np.asarray(p, dtype=float), list(order)[:3])
              for p, order in races if len(order) >= 3 and len(p) >= 4]
    if len(usable) < _MIN_FIT_RACES:
        log.warning(
            "Only %d usable races to fit the discount exponents (need %d); "
            "keeping the defaults %s. At 100 races the fitted lambda has a "
            "standard deviation of about 0.13, which is wider than the whole "
            "effect being estimated.", len(usable), _MIN_FIT_RACES, initial)
        return initial

    def negative_log_likelihood(params: np.ndarray) -> float:
        lam2, lam3 = float(params[0]), float(params[1])
        if not (0.05 < lam2 <= 1.5 and 0.05 < lam3 <= 1.5):
            return 1e9
        total = 0.0
        for probs, order in usable:
            # Condition on the winner: we are fitting how the *remaining*
            # placings fall out, not re-fitting the win model.
            probability = ordered_probability(probs, order, lam2, lam3)
            first = probs[order[0]] / probs.sum() if probs.sum() > 0 else 1e-12
            conditional = probability / max(first, 1e-12)
            total += -math.log(max(conditional, 1e-12))
        return total / len(usable)

    result = minimize(negative_log_likelihood, np.array(initial),
                      method="Nelder-Mead",
                      options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-4})
    lam2, lam3 = float(result.x[0]), float(result.x[1])
    return (min(1.5, max(0.05, lam2)), min(1.5, max(0.05, lam3)))
