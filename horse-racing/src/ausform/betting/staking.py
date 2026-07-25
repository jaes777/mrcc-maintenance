"""Stake sizing: Kelly, and why you should not use all of it.

THE KELLY CRITERION
-------------------
For a single bet at decimal odds o with win probability p, the fraction of
bankroll that maximises long-run growth is

    f* = (p*o - 1) / (o - 1)

The numerator is the edge. No edge, no bet -- and if the edge is negative
the formula returns a negative fraction, which is the maths telling you to
lay it, not to back it smaller.

WHY FULL KELLY IS DANGEROUS
---------------------------
Two independent reasons, and the second matters far more in racing.

1. Volatility. Under full Kelly the probability of your bankroll halving
   at some point is 50%. Betting a fraction c of Kelly gives a growth rate
   of c(2-c) times the maximum, and a halving probability of 0.5^((2-c)/c).
   Half Kelly keeps 75% of the growth while cutting the chance of a halving
   from 50% to 12.5%. That is an extremely good trade.

2. Estimation error -- the decisive one. Kelly assumes p is *known*. Yours
   is an estimate with real variance, and f* is convex in p, so plugging in
   a noisy estimate systematically *over*-bets. Card counters, who know
   their probabilities almost exactly, still bet fractions of 0.2-0.8. A
   horse racing model, whose probabilities are far less certain, has no
   business betting full Kelly.

The default here is quarter Kelly with hard caps on top. That is not
timidity; it is the appropriate response to using estimated probabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

log = logging.getLogger(__name__)


@dataclass
class StakingPolicy:
    """Bankroll rules. Every limit here exists to survive a bad run."""

    kelly_fraction: float = 0.25
    max_bet_pct: float = 0.02       # of bankroll, per bet
    max_race_pct: float = 0.05      # total exposure in one race
    max_day_pct: float = 0.10       # total exposure in one day
    min_edge: float = 0.05          # skip anything thinner than this
    min_probability: float = 0.02   # ignore hopeless outsiders
    min_stake: float = 1.0          # below this, not worth the friction
    round_to: float = 0.50

    def validate(self) -> None:
        if not 0 < self.kelly_fraction <= 1.0:
            raise ValueError("kelly_fraction must be in (0, 1].")
        if self.kelly_fraction > 0.5:
            log.warning(
                "kelly_fraction=%.2f is aggressive. Above 0.5 the risk of a "
                "deep drawdown rises sharply, and model probability error "
                "makes the true risk worse than the formula suggests.",
                self.kelly_fraction)


def kelly_fraction(probability: float, odds: float, commission: float = 0.0) -> float:
    """Optimal bankroll fraction for one bet. Zero when there is no edge.

    `commission` is charged on winnings (as on Betfair), which reduces the
    net odds and therefore the optimal stake.
    """
    if odds <= 1.0 or not 0.0 < probability < 1.0:
        return 0.0
    net_odds = (odds - 1.0) * (1.0 - commission)
    if net_odds <= 0:
        return 0.0
    edge = probability * (1.0 + net_odds) - 1.0
    if edge <= 0:
        return 0.0
    return edge / net_odds


def expected_value(probability: float, odds: float, commission: float = 0.0) -> float:
    """Expected profit per $1 staked. 0.05 means a 5% edge."""
    if odds <= 1.0:
        return -1.0
    net_odds = (odds - 1.0) * (1.0 - commission)
    return probability * (1.0 + net_odds) - 1.0


def kelly_multiple_runners(
    probabilities: Sequence[float],
    odds: Sequence[float],
    commission: float = 0.0,
    max_total: float = 0.5,
) -> np.ndarray:
    """Simultaneous stakes on several runners in the SAME race.

    Because at most one can win, these bets are mutually exclusive rather
    than independent, and sizing them one at a time over-stakes the race.
    The correct problem is to maximise expected log wealth jointly:

        maximise  sum_i p_i * ln(1 - sum_j f_j + f_i * o_i)

    which is concave, so a constrained optimiser finds the global optimum
    reliably. Runners with no edge naturally receive zero.

    The exact objective, including the term for "none of the runners I
    backed wins", is

        maximise  sum_i p_i * ln(1 - F + f_i * a_i) + (1 - sum_i p_i) * ln(1 - F)

    where F = sum_j f_j and a_i = 1 + (o_i - 1)(1 - commission) is the gross
    return per unit staked after commission.

    A NOTE ON WHICH RUNNERS QUALIFY
    -------------------------------
    It is tempting to pre-filter to runners with a standalone edge, i.e.
    p_i * a_i > 1. That is WRONG, and subtly so. The first-order condition
    for runner i to earn a positive stake is p_i * a_i > W0, where
    W0 = 1 - F is the bankroll left unbet. Since W0 < 1 whenever you bet
    anything at all, the true inclusion threshold is strictly *below* 1: a
    runner with no standalone edge can still belong in the portfolio as a
    hedge, because the money already committed elsewhere lowers the bar.
    Filtering at 1 both drops those runners and mis-sizes the survivors,
    which then absorb the whole allocation.

    So the filter here only removes runners that are unbettable in
    principle (no price, no probability), and the optimiser decides the
    rest. `probabilities` is used as supplied and is NOT renormalised: if
    it sums to less than 1 the remainder is treated as "none of these
    wins", which is correct for a partial field but a silent under-stake if
    the caller merely passed unnormalised numbers. Pass a full, normalised
    field.
    """
    p = np.asarray(probabilities, dtype=float)
    o = np.asarray(odds, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0)

    net = 1.0 + (o - 1.0) * (1.0 - commission)
    viable = (o > 1.0) & (p > 0) & np.isfinite(p) & np.isfinite(o) & (net > 1.0)
    if not viable.any():
        return np.zeros(n)

    index = np.where(viable)[0]
    p_v, net_v = p[index], net[index]

    # If nothing has an edge even at a full bankroll, there is no bet.
    # (With W0 = 1 the condition p*a > W0 is the strictest it can be, so
    # failing it for every runner means the optimum really is all zeros.)
    if not np.any(p_v * net_v > 1.0):
        return np.zeros(n)

    def negative_growth(f: np.ndarray) -> float:
        total = f.sum()
        wealth = 1.0 - total + f * net_v
        if np.any(wealth <= 1e-9):
            return 1e9
        # Any probability mass on runners we did not back loses the stake.
        losing = max(0.0, 1.0 - p_v.sum())
        residual = 1.0 - total
        if residual <= 1e-9:
            return 1e9
        return -(float(p_v @ np.log(wealth)) + losing * float(np.log(residual)))

    constraints = [{"type": "ineq", "fun": lambda f: max_total - f.sum()}]
    bounds = [(0.0, max_total) for _ in index]
    start = np.full(len(index), min(0.01, max_total / max(1, len(index))))

    result = minimize(negative_growth, start, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 500, "ftol": 1e-12})

    stakes = np.zeros(n)
    if result.success:
        stakes[index] = np.maximum(0.0, result.x)
    else:
        # Fall back to independent Kelly, scaled to respect the cap.
        independent = np.array([kelly_fraction(p[i], o[i], commission)
                                for i in index])
        total = independent.sum()
        if total > max_total:
            independent *= max_total / total
        stakes[index] = independent
        log.debug("Joint Kelly did not converge; used scaled independent Kelly.")
    return stakes


@dataclass
class Stake:
    """A sized bet, ready to place."""

    selection: str
    bet_type: str
    odds: float
    probability: float
    edge: float
    kelly: float
    amount: float
    reason: str = ""


def size_bets(
    candidates: Sequence[tuple[str, str, float, float]],
    bankroll: float,
    policy: StakingPolicy,
    commission: float = 0.0,
    already_staked_today: float = 0.0,
) -> list[Stake]:
    """Turn (selection, bet_type, probability, odds) tuples into stakes.

    Applies, in order: the minimum edge and probability filters, fractional
    Kelly, the per-bet cap, the per-race cap, the per-day cap, and finally
    rounding and the minimum stake. Anything that fails a filter is simply
    not bet -- there is no "bet it smaller because it is close".
    """
    policy.validate()

    qualifying: list[tuple[str, str, float, float, float]] = []
    for selection, bet_type, probability, odds in candidates:
        if probability < policy.min_probability or odds <= 1.0:
            continue
        edge = expected_value(probability, odds, commission)
        if edge < policy.min_edge:
            continue
        qualifying.append((selection, bet_type, probability, odds, edge))

    # Win bets within one race are mutually exclusive -- at most one can
    # land -- so sizing them independently over-stakes the race. Solve the
    # joint log-growth problem for those, and size everything else (place
    # bets, which can all win together) independently.
    win_bets = [q for q in qualifying if q[1] == "win"]

    joint: dict[str, float] = {}
    if len(win_bets) > 1:
        fractions = kelly_multiple_runners(
            [q[2] for q in win_bets], [q[3] for q in win_bets],
            commission=commission, max_total=policy.max_race_pct)
        joint = {q[0]: float(f) for q, f in zip(win_bets, fractions)}

    stakes: list[Stake] = []
    for selection, bet_type, probability, odds, edge in qualifying:
        full = kelly_fraction(probability, odds, commission)
        if bet_type == "win" and selection in joint:
            fraction = joint[selection] * policy.kelly_fraction
        else:
            fraction = full * policy.kelly_fraction

        fraction = min(fraction, policy.max_bet_pct)
        amount = fraction * bankroll
        if amount < policy.min_stake:
            continue

        stakes.append(Stake(
            selection=selection,
            bet_type=bet_type,
            odds=odds,
            probability=probability,
            edge=edge,
            kelly=full,
            amount=amount,
            reason=(f"model {probability:.1%} vs market "
                    f"{1/odds:.1%} at ${odds:.2f}"),
        ))

    # Per-race cap: scale everything down proportionally rather than
    # dropping bets, which would bias selection toward whatever was sized
    # first.
    race_cap = policy.max_race_pct * bankroll
    total = sum(s.amount for s in stakes)
    if total > race_cap and total > 0:
        scale = race_cap / total
        for stake in stakes:
            stake.amount *= scale

    # Per-day cap.
    day_cap = policy.max_day_pct * bankroll
    remaining = max(0.0, day_cap - already_staked_today)
    total = sum(s.amount for s in stakes)
    if total > remaining and total > 0:
        scale = remaining / total
        for stake in stakes:
            stake.amount *= scale

    final: list[Stake] = []
    for stake in stakes:
        stake.amount = round(stake.amount / policy.round_to) * policy.round_to
        if stake.amount >= policy.min_stake:
            final.append(stake)
    return final


def drawdown_risk(kelly_fraction_used: float, target: float = 0.5) -> float:
    """P(bankroll ever falls to `target` of its starting value).

    Standard diffusion approximation: P = target^((2-c)/c). At full Kelly
    the chance of ever halving is 50%; at quarter Kelly it is under 2%.
    """
    c = max(1e-6, min(1.0, kelly_fraction_used))
    return float(target ** ((2.0 - c) / c))


def growth_rate_fraction(kelly_fraction_used: float) -> float:
    """Share of the maximum growth rate retained: c(2-c)."""
    c = max(0.0, min(1.0, kelly_fraction_used))
    return c * (2.0 - c)


def bets_needed_to_prove_edge(yield_pct: float, odds_sd: float = 3.0,
                              t_stat: float = 2.0) -> int:
    """How many bets before a claimed edge is statistically real.

    N = (t * sigma / mu)^2. For a 5% yield with the payout variability of
    a typical mixed win portfolio, this is around 14,000 bets. Anyone
    claiming a proven edge off a few hundred bets -- in a backtest or live
    -- is reporting noise.
    """
    if yield_pct <= 0:
        return 0
    return int(round((t_stat * odds_sd / yield_pct) ** 2))
