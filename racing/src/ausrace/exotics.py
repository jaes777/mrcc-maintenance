"""Turning win probabilities into finishing-order probabilities.

A model gives you one number per horse: the probability it wins. Every other
bet type - place, quinella, exacta, trifecta, first four, same-race multi -
needs the probability of a particular *finishing order*, which does not follow
from the win probabilities alone. You need a model of how the rest of the field
sorts itself out behind the winner.

**Harville (1973)** is the standard closed form. It assumes that once the
winner is removed, the remaining horses' relative chances are unchanged:

    P(i 1st, j 2nd, k 3rd) = p_i * p_j/(1-p_i) * p_k/(1-p_i-p_j)

Harville is systematically biased: it **overestimates** the probability of a
short-priced favourite running 2nd or 3rd. The reason is intuitive - Harville
assumes a beaten favourite was still nearly the best horse in the race, whereas
in reality a favourite that fails to win has usually run badly rather than
narrowly missed. Pricing trifectas off raw Harville is the most common way that
model-driven exotic betting loses money, because it makes you overbet
favourite-anchored combinations.

**Discounted Harville (Lo & Bacon-Shone)** is the fix, and the default here. It
raises the conditional probabilities to a power lambda < 1, which flattens the
distribution for the minor placings:

    P(i 1st, j 2nd) = p_i * [ p_j^L2 / sum_{m != i} p_m^L2 ]

with L1 = 1 > L2 > L3 > L4. Setting every lambda to 1 recovers Harville
exactly. The published starting values (L2 = 0.81, L3 = 0.65) come from Hong
Kong data; `fit_lambdas` re-fits them on Australian results, which you should
do, because AU field sizes and class structure differ.

**Stern (1990)** gives each horse a Gamma(shape=r) running time; r = 1 is
exactly Harville and larger r moves towards Henery's normal model. Stern
reported r around 2 as a good fit. It is included for comparison and evaluated
by Monte Carlo, since it has no closed form.

References:
  Harville, D.A. (1973) JASA 68(342), 312-316.
  Henery, R.J. (1981) JRSS-B 43(1).
  Stern, H. (1990) "Models for distributions on permutations", JASA 85(410).
  Lo, V. & Bacon-Shone, J. (1994, 2008), Management Science 41(6), 1048.
"""

from __future__ import annotations

from itertools import permutations
from typing import Iterable, Sequence

import numpy as np

# Discount exponents for positions 1, 2, 3, 4. Position 1 is always 1.0 by
# definition. These are the published Lo & Bacon-Shone starting values; refit
# them on your own data with `fit_lambdas`.
DEFAULT_LAMBDAS: tuple[float, float, float, float] = (1.0, 0.81, 0.65, 0.55)

# Stern's Gamma shape. r = 1 reproduces Harville exactly.
DEFAULT_STERN_R = 2.0


def normalise(p: Sequence[float]) -> np.ndarray:
    """Return probabilities summing to 1, with non-finite and negative entries zeroed."""
    arr = np.asarray(p, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    arr[arr < 0] = 0.0
    total = arr.sum()
    if total <= 0:
        return np.full(len(arr), 1.0 / max(len(arr), 1))
    return arr / total


def _lam(lambdas: Sequence[float] | None, position: int) -> float:
    """Discount exponent for a 1-indexed finishing position."""
    lambdas = DEFAULT_LAMBDAS if lambdas is None else lambdas
    if position <= 1:
        return 1.0
    idx = min(position - 1, len(lambdas) - 1)
    return float(lambdas[idx])


# --------------------------------------------------------------------------
# Closed-form ordering probabilities
# --------------------------------------------------------------------------

def order_probability(
    p: Sequence[float],
    order: Sequence[int],
    lambdas: Sequence[float] | None = DEFAULT_LAMBDAS,
) -> float:
    """P(the horses in `order` finish 1st, 2nd, ... in exactly that sequence).

    Pass `lambdas=None` or all-ones for plain Harville.
    """
    probs = normalise(p)
    n = len(probs)
    used = np.zeros(n, dtype=bool)
    result = 1.0

    for position, horse in enumerate(order, start=1):
        if used[horse]:
            raise ValueError(f"Index {horse} appears twice in the finishing order.")
        if position == 1:
            result *= probs[horse]
        else:
            lam = _lam(lambdas, position)
            weights = np.where(used, 0.0, probs ** lam)
            total = weights.sum()
            if total <= 1e-15:
                return 0.0
            result *= weights[horse] / total
        used[horse] = True
        if result <= 0:
            return 0.0
    return float(result)


def harville_order_prob(p: Sequence[float], order: Sequence[int]) -> float:
    """Plain Harville, no discounting. Kept for comparison and testing."""
    return order_probability(p, order, lambdas=(1.0, 1.0, 1.0, 1.0))


def exacta_matrix(
    p: Sequence[float],
    lambdas: Sequence[float] | None = DEFAULT_LAMBDAS,
) -> np.ndarray:
    """M[i, j] = P(i wins, j runs 2nd). Diagonal is zero."""
    probs = normalise(p)
    n = len(probs)
    lam = _lam(lambdas, 2)
    weights = probs ** lam
    total = weights.sum()

    # Denominator for winner i is the total weight excluding i.
    denom = total - weights
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = probs[:, None] * (weights[None, :] / denom[:, None])
    np.fill_diagonal(matrix, 0.0)
    matrix[~np.isfinite(matrix)] = 0.0
    return matrix


def position_matrix(
    p: Sequence[float],
    max_position: int = 3,
    lambdas: Sequence[float] | None = DEFAULT_LAMBDAS,
) -> np.ndarray:
    """P[i, k] = P(horse i finishes in position k+1), exact, for k up to 2.

    Position 4 falls back to Monte Carlo because the exact sum is O(n^4) and
    the accuracy gained is not worth the runtime inside a betting loop.
    """
    probs = normalise(p)
    n = len(probs)
    max_position = int(min(max_position, n))
    out = np.zeros((n, max_position))
    out[:, 0] = probs

    if max_position >= 2:
        lam2 = _lam(lambdas, 2)
        w2 = probs ** lam2
        denom2 = w2.sum() - w2                      # excluding the winner
        with np.errstate(divide="ignore", invalid="ignore"):
            contrib = probs[:, None] * (w2[None, :] / denom2[:, None])
        contrib[~np.isfinite(contrib)] = 0.0
        np.fill_diagonal(contrib, 0.0)
        out[:, 1] = contrib.sum(axis=0)

    if max_position >= 3:
        lam2, lam3 = _lam(lambdas, 2), _lam(lambdas, 3)
        w2, w3 = probs ** lam2, probs ** lam3
        tot2, tot3 = w2.sum(), w3.sum()
        for a in range(n):                          # a wins
            d2 = tot2 - w2[a]
            if d2 <= 1e-15:
                continue
            pa = probs[a]
            if pa <= 0:
                continue
            for b in range(n):                      # b runs 2nd
                if b == a:
                    continue
                d3 = tot3 - w3[a] - w3[b]
                if d3 <= 1e-15:
                    continue
                lead = pa * (w2[b] / d2)
                if lead <= 0:
                    continue
                share = lead * w3 / d3
                share[a] = 0.0
                share[b] = 0.0
                out[:, 2] += share

    if max_position >= 4:
        mc = simulate_orders(probs, n_sims=40_000, lambdas=lambdas, seed=0)
        counts = np.bincount(mc[:, 3], minlength=n)
        out[:, 3] = counts / len(mc)

    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------
# Monte Carlo - the universal path for any bet whose payoff is a predicate
# on the finishing order (boxed bets, standouts, same-race multis)
# --------------------------------------------------------------------------

def simulate_orders(
    p: Sequence[float],
    n_sims: int = 40_000,
    positions: int | None = None,
    lambdas: Sequence[float] | None = DEFAULT_LAMBDAS,
    seed: int | None = 0,
) -> np.ndarray:
    """Sample finishing orders from the discounted-Harville model.

    Sampling is sequential and exact for this model: draw the winner from p,
    then draw each subsequent position from the discounted weights of whoever
    is left. Returns an (n_sims, positions) array of horse indices.
    """
    probs = normalise(p)
    n = len(probs)
    positions = n if positions is None else int(min(positions, n))
    rng = np.random.default_rng(seed)

    out = np.zeros((n_sims, positions), dtype=np.int32)
    available = np.ones((n_sims, n), dtype=bool)

    for k in range(positions):
        lam = _lam(lambdas, k + 1)
        weights = np.where(available, probs[None, :] ** lam, 0.0)
        totals = weights.sum(axis=1, keepdims=True)
        totals[totals <= 0] = 1.0
        cumulative = np.cumsum(weights / totals, axis=1)
        draws = rng.random((n_sims, 1))
        chosen = (cumulative < draws).sum(axis=1)
        chosen = np.clip(chosen, 0, n - 1)
        out[:, k] = chosen
        available[np.arange(n_sims), chosen] = False

    return out


def stern_simulate(
    p: Sequence[float],
    n_sims: int = 40_000,
    r: float = DEFAULT_STERN_R,
    seed: int | None = 0,
    calibrate: bool = True,
) -> np.ndarray:
    """Sample finishing orders under Stern's Gamma running-time model.

    Each horse gets a running time Gamma(shape=r, rate=lambda_i); lowest time
    wins. r = 1 is exponential and reproduces Harville exactly.

    With `calibrate` set, the rates are adjusted by fixed-point iteration so
    the simulated win probabilities match the input win probabilities. Without
    it, r != 1 would silently distort the win probabilities as well as the
    minor placings, which is not what you want - the win probabilities came
    from the model and should be preserved.
    """
    probs = normalise(p)
    n = len(probs)
    rng = np.random.default_rng(seed)
    rates = probs.copy()

    if calibrate and abs(r - 1.0) > 1e-9:
        for _ in range(25):
            times = rng.gamma(shape=r, scale=1.0 / np.maximum(rates, 1e-12),
                              size=(8_000, n))
            simulated = np.bincount(np.argmin(times, axis=1), minlength=n) / 8_000
            simulated = np.maximum(simulated, 1e-6)
            rates *= (probs / simulated) ** 0.5
            rates = np.maximum(rates, 1e-9)
            if np.max(np.abs(simulated - probs)) < 0.002:
                break

    times = rng.gamma(shape=r, scale=1.0 / np.maximum(rates, 1e-12), size=(n_sims, n))
    return np.argsort(times, axis=1)


# --------------------------------------------------------------------------
# Bet-level probabilities
# --------------------------------------------------------------------------

def places_paid(n_runners: int) -> int:
    """Number of place dividends paid, from the *final* starter count.

    Australian terms, consistent across TAB and the major corporates:
      8 or more starters -> 3 places
      5, 6 or 7 starters -> 2 places
      fewer than 5       -> no place market (tote refunds; some fixed-odds
                            books instead treat place stakes as win stakes)

    The count that matters is starters after scratchings, not the nominated
    field. A 9-horse race that loses two runners becomes a 2-place race, and
    getting this wrong silently corrupts every place-bet expected value.
    """
    n = int(n_runners)
    if n >= 8:
        return 3
    if n >= 5:
        return 2
    return 0


def place_probabilities(
    p: Sequence[float],
    n_runners: int | None = None,
    lambdas: Sequence[float] | None = DEFAULT_LAMBDAS,
) -> tuple[np.ndarray, int]:
    """Probability each horse is paid as a placegetter, plus the number of
    paid places."""
    probs = normalise(p)
    n = len(probs) if n_runners is None else int(n_runners)
    paid = places_paid(n)
    if paid == 0:
        return np.zeros(len(probs)), 0
    positions = position_matrix(probs, max_position=paid, lambdas=lambdas)
    return np.clip(positions.sum(axis=1), 0.0, 1.0), paid


def quinella_probability(p, i: int, j: int, lambdas=DEFAULT_LAMBDAS) -> float:
    """P(i and j fill the first two placings in either order)."""
    matrix = exacta_matrix(p, lambdas=lambdas)
    return float(matrix[i, j] + matrix[j, i])


def exacta_probability(p, first: int, second: int, lambdas=DEFAULT_LAMBDAS) -> float:
    """P(`first` wins and `second` runs 2nd, in that exact order)."""
    return float(exacta_matrix(p, lambdas=lambdas)[first, second])


def trifecta_probability(p, order: Sequence[int], lambdas=DEFAULT_LAMBDAS) -> float:
    """P(the three given horses finish 1-2-3 in exactly that order)."""
    if len(order) != 3:
        raise ValueError("A trifecta needs exactly three selections.")
    return order_probability(p, order, lambdas=lambdas)


def first_four_probability(p, order: Sequence[int], lambdas=DEFAULT_LAMBDAS) -> float:
    """P(the four given horses finish 1-2-3-4 in exactly that order)."""
    if len(order) != 4:
        raise ValueError("A first four needs exactly four selections.")
    return order_probability(p, order, lambdas=lambdas)


def boxed_probability(
    p: Sequence[float],
    selections: Sequence[int],
    positions: int,
    lambdas: Sequence[float] | None = DEFAULT_LAMBDAS,
) -> float:
    """P(a boxed bet collects) - the top `positions` finishers all come from
    `selections`, in any order.

    Exact by enumeration. A box over k selections covers k!/(k-positions)!
    combinations, which is why boxing wide gets expensive so fast: five
    runners boxed in a first four is 120 combinations, not 5.
    """
    selections = list(dict.fromkeys(int(s) for s in selections))
    if len(selections) < positions:
        return 0.0
    return float(sum(order_probability(p, order, lambdas=lambdas)
                     for order in permutations(selections, positions)))


def standout_probability(
    p: Sequence[float],
    position_sets: Sequence[Sequence[int]],
    lambdas: Sequence[float] | None = DEFAULT_LAMBDAS,
) -> float:
    """P(a bet with a different selection set per finishing position collects).

    This covers standouts and bankers as well as boxes: pass the same set for
    every position to get a box, or a single horse for position 1 and a wider
    set below it to get a standout.
    """
    from itertools import product
    total = 0.0
    for combo in product(*position_sets):
        if len(set(combo)) != len(combo):
            continue
        total += order_probability(p, combo, lambdas=lambdas)
    return float(total)


def combination_count(position_sets: Sequence[Sequence[int]]) -> int:
    """Number of valid combinations, i.e. the unit cost of the ticket."""
    from itertools import product
    return sum(1 for combo in product(*position_sets) if len(set(combo)) == len(combo))


def box_combinations(n_selections: int, positions: int) -> int:
    """Ordered combinations covered by a box (= the unit cost).

    Sanity anchors: 3 horses boxed in a trifecta is 6; 4 horses is 24;
    5 horses is 60.
    """
    if n_selections < positions:
        return 0
    total = 1
    for k in range(positions):
        total *= n_selections - k
    return total


def same_race_multi_probability(
    p: Sequence[float],
    legs: Sequence[tuple[int, int]],
    n_sims: int = 60_000,
    lambdas: Sequence[float] | None = DEFAULT_LAMBDAS,
    seed: int | None = 0,
) -> float:
    """P(every leg of a same-race multi lands).

    `legs` is a list of (horse_index, max_finishing_position) - so (3, 2) means
    "horse 3 finishes top 2".

    **Never multiply the legs together.** Legs in a same-race multi are on
    different horses competing for overlapping finishing slots, so they are
    *negatively* correlated: if one of your horses takes a top-2 slot, that is
    one fewer slot for the other. Multiplying gives a joint probability that is
    too high, which makes the bookmaker's price look better than it is.

    Concretely, in a 10-horse race with every horse at 10%: two horses each
    have a 30% chance of running top 3, and the naive product says 9%. The true
    joint probability is 6.67% - the naive figure overstates it by a third.
    """
    if not legs:
        return 0.0
    max_pos = max(pos for _, pos in legs)
    orders = simulate_orders(p, n_sims=n_sims, positions=max_pos,
                             lambdas=lambdas, seed=seed)
    hit = np.ones(n_sims, dtype=bool)
    for horse, max_finish in legs:
        placed = (orders[:, :max_finish] == horse).any(axis=1)
        hit &= placed
    return float(hit.mean())


# --------------------------------------------------------------------------
# Fitting the discount parameters to your own results
# --------------------------------------------------------------------------

def fit_lambdas(
    races: Iterable[tuple[Sequence[float], Sequence[int]]],
    max_position: int = 3,
    grid: Sequence[float] = tuple(np.round(np.arange(0.40, 1.31, 0.03), 2)),
) -> tuple[list[float], dict]:
    """Fit the discount exponents by maximum likelihood on observed finishes.

    `races` yields (win_probabilities, actual_finishing_order), where the order
    lists horse indices for 1st, 2nd, 3rd... Positions are fitted one at a
    time, because the exponent for position k only enters the likelihood of
    position k given the ones above it - which makes this fast and stable.

    A fitted lambda below 1 confirms the Harville bias is present in your data.
    A fitted lambda at or above 1 would mean it is not, which is worth
    investigating before you trust it.
    """
    races = [(normalise(p), list(order)) for p, order in races]
    races = [(p, o) for p, o in races if len(o) >= 2]
    if not races:
        return list(DEFAULT_LAMBDAS), {"n_races": 0}

    lambdas = [1.0]
    diagnostics: dict = {"n_races": len(races), "per_position": {}}

    for position in range(2, max_position + 1):
        scores: dict[float, float] = {}
        for candidate in grid:
            total, count = 0.0, 0
            for probs, order in races:
                if len(order) < position:
                    continue
                used = np.zeros(len(probs), dtype=bool)
                used[order[: position - 1]] = True
                weights = np.where(used, 0.0, probs ** float(candidate))
                denom = weights.sum()
                if denom <= 1e-15:
                    continue
                actual = order[position - 1]
                if actual >= len(probs) or used[actual]:
                    continue
                total += np.log(max(weights[actual] / denom, 1e-12))
                count += 1
            scores[float(candidate)] = total / count if count else float("-inf")

        best = max(scores, key=scores.get)
        lambdas.append(best)
        diagnostics["per_position"][position] = {
            "lambda": best,
            "log_likelihood": scores[best],
            "harville_log_likelihood": scores.get(1.0, float("nan")),
            "improvement_over_harville": scores[best] - scores.get(1.0, float("nan")),
        }

    while len(lambdas) < 4:
        lambdas.append(lambdas[-1] * 0.85)

    return lambdas, diagnostics


def compare_models(p: Sequence[float], lambdas=DEFAULT_LAMBDAS) -> "object":
    """Side-by-side place probabilities under Harville and discounted Harville.

    Useful for seeing the size of the correction: the favourite's place
    probability should drop and the longshots' should rise.
    """
    import pandas as pd
    probs = normalise(p)
    n = len(probs)
    paid = max(places_paid(n), 3)
    plain = position_matrix(probs, paid, lambdas=(1.0, 1.0, 1.0, 1.0)).sum(axis=1)
    disc = position_matrix(probs, paid, lambdas=lambdas).sum(axis=1)
    return pd.DataFrame({
        "win_prob": probs,
        "place_harville": plain,
        "place_discounted": disc,
        "difference": disc - plain,
    }).sort_values("win_prob", ascending=False)
