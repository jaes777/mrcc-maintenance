"""Odds handling, expected value, and stake sizing.

Everything here is arithmetic on probabilities and prices - no modelling. It is
kept separate from the model on purpose: the model's job is to produce honest
probabilities, and this module's job is to decide whether any price on offer is
worth taking, and for how much.

The single most important idea: a bet is worth making when your probability is
higher than the price implies, *after* removing the bookmaker's margin. Picking
winners is not the same thing as making money, and this module is where the
difference is enforced.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Sequence

import numpy as np

BetType = Literal[
    "WIN", "PLACE", "EACH_WAY", "QUINELLA", "EXACTA",
    "TRIFECTA", "FIRST_FOUR", "BOXED_TRIFECTA", "BOXED_EXACTA", "BOXED_FIRST_FOUR",
]

# Betfair Australia charges commission on NET MARKET WINNINGS, not turnover -
# which is why the exchange is the only realistic venue for a model-driven
# operation. A corporate bookmaker's 115-130% book takes its margin out of
# every bet whether you win or lose.
#
# The Market Base Rate varies by controlling body and changes over time:
#   Racing NSW / ACT      ~10%
#   Most other AU racing   ~8%
#   Some states / NZ       ~6%
# Read the actual rate from the market's Rules via the API rather than trusting
# any of these; they move with each body's race-fields fees.
DEFAULT_EXCHANGE_COMMISSION = 0.08
EXCHANGE_COMMISSION_BY_STATE = {
    "NSW": 0.10, "ACT": 0.10,
    "VIC": 0.08, "QLD": 0.08, "SA": 0.08, "WA": 0.08, "TAS": 0.08, "NT": 0.08,
}

# Typical Australian fixed-odds book percentages, for reference when judging
# whether a price is worth taking at all.
TYPICAL_BOOK_METRO = 1.15      # to 1.30
TYPICAL_BOOK_COUNTRY = 1.25    # to 1.40


# --------------------------------------------------------------------------
# Odds <-> probability
# --------------------------------------------------------------------------

def implied_probability(odds_decimal: float | Sequence[float]) -> np.ndarray:
    """Raw implied probability 1/odds. Sums to more than 1 across a field -
    that excess is the bookmaker's margin (the 'overround')."""
    odds = np.asarray(odds_decimal, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = 1.0 / odds
    probs[~np.isfinite(probs)] = np.nan
    return probs


def overround(odds_decimal: Sequence[float]) -> float:
    """Total book percentage. 1.18 means the book is set to 118% - the
    bookmaker's built-in edge is roughly 15% of turnover on that market."""
    probs = implied_probability(odds_decimal)
    return float(np.nansum(probs))


def devig(
    odds_decimal: Sequence[float],
    method: Literal["proportional", "power", "shin"] = "shin",
) -> np.ndarray:
    """Strip the bookmaker's margin out of a set of odds to recover the
    market's implied probabilities.

    Three methods, in increasing order of realism:

    - **proportional**: divide every implied probability by the overround.
      Simple, and wrong in a specific way - it assumes the margin is spread
      evenly, but bookmakers load far more margin onto longshots.

    - **power**: find k with sum((1/o_i)^k) = 1. Handles the longshot loading
      much better than proportional.

    - **shin**: Shin's (1993) model, which derives the margin from the
      bookmaker protecting itself against insider money. Consistently the best
      performer in published comparisons of de-vigging methods, and the default
      here. See Strumbelj (2014), "On determining probability forecasts from
      betting odds", Int. J. Forecasting 30(4).
    """
    odds = np.asarray(odds_decimal, dtype=float)
    valid = np.isfinite(odds) & (odds > 1.0)
    out = np.full(len(odds), np.nan)
    if valid.sum() == 0:
        return out

    raw = 1.0 / odds[valid]
    book = raw.sum()

    if book <= 0:
        return out

    if method == "proportional":
        out[valid] = raw / book
        return out

    if method == "power":
        # Solve sum(raw^k) = 1 for k by bisection. k > 1 when the book is
        # over 100%, which compresses longshots more than favourites.
        # 60 halvings takes the bracket below 1e-16 - already exact in float64.
        lo, hi = 0.05, 10.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if np.sum(raw ** mid) > 1.0:
                lo = mid
            else:
                hi = mid
        out[valid] = raw ** ((lo + hi) / 2)
        out[valid] /= out[valid].sum()
        return out

    # Shin's method. Solve for the insider-trading proportion z in [0, 1) such
    # that the recovered probabilities sum to 1.
    def recovered(z: float) -> np.ndarray:
        if z <= 1e-9:
            return raw / book
        disc = z * z + 4.0 * (1.0 - z) * (raw * raw) / book
        return (np.sqrt(np.maximum(disc, 0.0)) - z) / (2.0 * (1.0 - z))

    lo, hi = 0.0, 0.99
    for _ in range(60):
        mid = (lo + hi) / 2
        if recovered(mid).sum() > 1.0:
            lo = mid
        else:
            hi = mid
    probs = recovered((lo + hi) / 2)
    total = probs.sum()
    out[valid] = probs / total if total > 0 else raw / book
    return out


def fair_odds(probability: float | Sequence[float]) -> np.ndarray | float:
    """The break-even price for a given probability.

    Returns a float for a scalar input and an array for a sequence, so it can
    be used interchangeably in both contexts.
    """
    scalar = np.isscalar(probability) or (
        isinstance(probability, np.ndarray) and probability.ndim == 0
    )
    p = np.atleast_1d(np.asarray(probability, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        odds = 1.0 / p
    odds = np.where(np.isfinite(odds), odds, np.inf)
    return float(odds[0]) if scalar else odds


# --------------------------------------------------------------------------
# Expected value
# --------------------------------------------------------------------------

def expected_value(
    probability: float,
    odds_decimal: float,
    commission: float = 0.0,
) -> float:
    """Expected profit per $1 staked.

    0.05 means that, if the probability is right, you make 5c per dollar
    staked on average over a large number of identical bets. Negative means
    the bet loses money no matter how confident it feels.
    """
    if not np.isfinite(odds_decimal) or odds_decimal <= 1.0:
        return float("-inf")
    profit_if_win = (odds_decimal - 1.0) * (1.0 - commission)
    return float(probability * profit_if_win - (1.0 - probability))


def edge_ratio(probability: float, odds_decimal: float) -> float:
    """How much shorter your assessed price is than the offered price.

    1.20 means the horse should be $5 and you're being offered $6.
    """
    if not np.isfinite(odds_decimal) or odds_decimal <= 0 or probability <= 0:
        return 0.0
    return float(probability * odds_decimal)


# --------------------------------------------------------------------------
# Stake sizing
# --------------------------------------------------------------------------

def kelly_fraction(
    probability: float,
    odds_decimal: float,
    commission: float = 0.0,
) -> float:
    """Full-Kelly stake as a fraction of bankroll.

    Kelly maximises the long-run growth rate of a bankroll. It is also brutally
    aggressive - full Kelly on racing regularly produces 50%+ drawdowns even
    when the edge is real, and it is unforgiving if your probabilities are
    even slightly optimistic (which they always are). Use `staking_plan`,
    which applies a fraction of this.
    """
    if not np.isfinite(odds_decimal) or odds_decimal <= 1.0:
        return 0.0
    b = (odds_decimal - 1.0) * (1.0 - commission)
    if b <= 0:
        return 0.0
    f = (probability * (b + 1.0) - 1.0) / b
    return float(max(f, 0.0))


def staking_plan(
    probability: float,
    odds_decimal: float,
    bankroll: float,
    kelly_fraction_used: float = 0.25,
    max_stake_pct: float = 0.02,
    min_stake: float = 1.0,
    commission: float = 0.0,
) -> float:
    """Recommended stake in dollars.

    Defaults to quarter-Kelly capped at 2% of bankroll. Both caps matter:
      - Quarter Kelly cuts the drawdown roughly fourfold at the cost of about
        a quarter of the growth rate, which is a trade almost everyone should take.
      - The hard 2% cap protects you from the case where the model is simply
        wrong about one horse and Kelly wants to bet the house on it.
    """
    full = kelly_fraction(probability, odds_decimal, commission=commission)
    if full <= 0:
        return 0.0
    fraction = min(full * kelly_fraction_used, max_stake_pct)
    stake = bankroll * fraction
    return float(round(stake, 2)) if stake >= min_stake else 0.0


def simultaneous_kelly(
    probabilities: Sequence[float],
    odds_decimal: Sequence[float],
    commission: float = 0.0,
    kelly_fraction_used: float = 0.25,
    max_total_pct: float = 0.05,
) -> np.ndarray:
    """Kelly stakes across several mutually exclusive runners in one race.

    Betting two horses in the same race is not two independent bets - only one
    can win, so the stakes interact. This solves the multi-outcome Kelly
    problem by the standard method: repeatedly drop the least attractive
    selection until every remaining one clears the threshold set by the others.

    Returns fractions of bankroll, one per runner.
    """
    p = np.asarray(probabilities, dtype=float)
    o = np.asarray(odds_decimal, dtype=float)
    n = len(p)
    stakes = np.zeros(n)

    b = (o - 1.0) * (1.0 - commission)
    viable = np.isfinite(b) & (b > 0) & np.isfinite(p) & (p > 0)
    if not viable.any():
        return stakes

    # Rank by expected return per unit; the optimal set is always a prefix.
    order = np.argsort(-(p * (b + 1.0)))
    order = [i for i in order if viable[i]]

    best_set: list[int] = []
    for k in range(1, len(order) + 1):
        candidate = order[:k]
        sum_p = p[candidate].sum()
        sum_inv = np.sum(1.0 / (b[candidate] + 1.0))
        denom = 1.0 - sum_inv
        if denom <= 1e-9:
            break
        reserve = (1.0 - sum_p) / denom          # fraction held back in cash
        f = p[candidate] - reserve / (b[candidate] + 1.0)
        if np.all(f > 0):
            best_set = candidate
            stakes = np.zeros(n)
            stakes[candidate] = f
        else:
            break

    if not best_set:
        return np.zeros(n)

    stakes *= kelly_fraction_used
    total = stakes.sum()
    if total > max_total_pct:
        stakes *= max_total_pct / total
    return stakes


# --------------------------------------------------------------------------
# Bet recommendation
# --------------------------------------------------------------------------

@dataclass
class BetRecommendation:
    """One suggested bet, with the reasoning attached."""

    bet_type: str
    selection: str              # human-readable, e.g. "3 Fast Lad" or "3-7-1"
    selection_ids: list[str]
    model_probability: float
    offered_odds: float
    fair_odds: float
    expected_value: float       # profit per $1 staked
    edge_pct: float             # EV as a percentage
    stake: float
    confidence: str             # HIGH / MEDIUM / LOW
    rationale: str
    combinations: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


# Beyond these prices, a model's probability estimate cannot be trusted enough
# to bet on. The reason is arithmetic: at $101 the model is claiming a ~1%
# chance, and being wrong by half a percentage point - which is nothing in
# absolute terms - moves the expected value by 50%. Published data backs this
# up: horses at 30/1 return about 63c in the dollar and those at 100/1+ about
# 39c. An "edge" found at these prices is nearly always model error.
MAX_ODDS_WIN = 51.0
MAX_ODDS_PLACE = 26.0


def _confidence(ev: float, probability: float, odds: float, n_sample_support: int,
                price_is_estimated: bool = False) -> str:
    """A blunt three-way label. Deliberately conservative.

    Long-priced horses need a much bigger apparent edge to be trustworthy,
    because a small absolute error in a small probability is a huge relative
    error - the classic way a model 'finds value' that isn't there.
    """
    if n_sample_support < 500:
        return "LOW"
    # An estimated price is a guess about what a pool will pay. It can never
    # justify better than LOW confidence, however large the apparent edge.
    if price_is_estimated:
        return "LOW"
    if odds >= 15:
        return "HIGH" if ev > 0.25 else ("MEDIUM" if ev > 0.15 else "LOW")
    if odds >= 6:
        return "HIGH" if ev > 0.15 else ("MEDIUM" if ev > 0.08 else "LOW")
    return "HIGH" if ev > 0.08 else ("MEDIUM" if ev > 0.04 else "LOW")


def recommend_win_bets(
    runner_ids: Sequence[str],
    labels: Sequence[str],
    probabilities: Sequence[float],
    odds_decimal: Sequence[float],
    bankroll: float = 1000.0,
    min_edge: float = 0.05,
    commission: float = 0.0,
    kelly_fraction_used: float = 0.25,
    max_stake_pct: float = 0.02,
    n_sample_support: int = 100_000,
    max_odds: float = MAX_ODDS_WIN,
) -> list[BetRecommendation]:
    """Win bets whose expected value clears `min_edge`.

    `min_edge` of 0.05 means we only bet when we think we are getting at least
    5% the better of it. That threshold is doing a lot of work: set it to 0 and
    you will bet almost every race and lose to the margin, because small
    apparent edges are usually model error rather than real value.
    """
    out: list[BetRecommendation] = []
    p = np.asarray(probabilities, dtype=float)
    o = np.asarray(odds_decimal, dtype=float)

    for i, rid in enumerate(runner_ids):
        if not np.isfinite(o[i]) or o[i] <= 1.0 or not np.isfinite(p[i]):
            continue
        if o[i] > max_odds:
            # Too long to trust. See MAX_ODDS_WIN.
            continue
        ev = expected_value(p[i], o[i], commission=commission)
        if ev < min_edge:
            continue
        stake = staking_plan(
            p[i], o[i], bankroll,
            kelly_fraction_used=kelly_fraction_used,
            max_stake_pct=max_stake_pct,
            commission=commission,
        )
        if stake <= 0:
            continue
        out.append(
            BetRecommendation(
                bet_type="WIN",
                selection=str(labels[i]),
                selection_ids=[str(rid)],
                model_probability=float(p[i]),
                offered_odds=float(o[i]),
                fair_odds=float(fair_odds(p[i])),
                expected_value=float(ev),
                edge_pct=float(ev * 100),
                stake=stake,
                confidence=_confidence(ev, p[i], o[i], n_sample_support),
                rationale=(
                    f"Model rates this a {p[i]:.1%} chance (fair price "
                    f"${1/p[i]:.2f}); available at ${o[i]:.2f}. "
                    f"That is {ev*100:.1f}c expected profit per $1 staked."
                ),
            )
        )
    return sorted(out, key=lambda b: -b.expected_value)


def recommend_place_bets(
    runner_ids: Sequence[str],
    labels: Sequence[str],
    place_probabilities: Sequence[float],
    place_odds: Sequence[float],
    bankroll: float = 1000.0,
    min_edge: float = 0.04,
    commission: float = 0.0,
    kelly_fraction_used: float = 0.25,
    max_stake_pct: float = 0.02,
    places_paid: int = 3,
    n_sample_support: int = 100_000,
    max_odds: float = MAX_ODDS_PLACE,
    price_is_estimated: bool = False,
) -> list[BetRecommendation]:
    """Place bets clearing `min_edge`.

    Place markets are typically less efficiently priced than win markets, but
    they also carry a bigger margin, so the two effects partly cancel.
    """
    out: list[BetRecommendation] = []
    p = np.asarray(place_probabilities, dtype=float)
    o = np.asarray(place_odds, dtype=float)

    for i, rid in enumerate(runner_ids):
        if not np.isfinite(o[i]) or o[i] <= 1.0 or not np.isfinite(p[i]):
            continue
        if o[i] > max_odds:
            # Too long to trust. See MAX_ODDS_WIN.
            continue
        ev = expected_value(p[i], o[i], commission=commission)
        if ev < min_edge:
            continue
        stake = staking_plan(
            p[i], o[i], bankroll,
            kelly_fraction_used=kelly_fraction_used,
            max_stake_pct=max_stake_pct,
            commission=commission,
        )
        if stake <= 0:
            continue
        out.append(
            BetRecommendation(
                bet_type="PLACE",
                selection=str(labels[i]),
                selection_ids=[str(rid)],
                model_probability=float(p[i]),
                offered_odds=float(o[i]),
                fair_odds=float(fair_odds(p[i])),
                expected_value=float(ev),
                edge_pct=float(ev * 100),
                stake=stake,
                confidence=_confidence(ev, p[i], o[i], n_sample_support,
                                       price_is_estimated=price_is_estimated),
                rationale=(
                    f"Model rates this a {p[i]:.1%} chance of finishing top "
                    f"{places_paid} (fair place price ${1/p[i]:.2f}); "
                    f"available at ${o[i]:.2f}."
                ),
            )
        )
    return sorted(out, key=lambda b: -b.expected_value)


def drawdown_probability(kelly_fraction_used: float, fraction_of_bankroll: float) -> float:
    """P(bankroll ever falls to `fraction_of_bankroll` of its starting value).

    Under the standard log-normal approximation for fractional-Kelly betting:

        P(ever reach x * W0) = x^(2/f - 1)

    This assumes your edge is real and your probabilities are correct. If they
    are optimistic - and a model's own estimates of its edge almost always are,
    because you bet precisely where the model is most confident and therefore
    most likely to be overconfident - the real risk is worse than this.
    """
    f = float(np.clip(kelly_fraction_used, 1e-6, 1.0))
    x = float(np.clip(fraction_of_bankroll, 1e-6, 1.0))
    return float(min(x ** (2.0 / f - 1.0), 1.0))


def kelly_growth_fraction(kelly_fraction_used: float) -> float:
    """Share of full-Kelly growth rate retained: f(2 - f)."""
    f = float(np.clip(kelly_fraction_used, 0.0, 2.0))
    return float(f * (2.0 - f))


def kelly_drawdown_note(kelly_fraction_used: float) -> str:
    """A plain-English statement of the variance the chosen staking implies."""
    f = kelly_fraction_used
    p_half = drawdown_probability(f, 0.5)
    p_third = drawdown_probability(f, 1 / 3)
    growth = kelly_growth_fraction(f)

    label = ("Full Kelly" if f >= 0.9 else
             "Half Kelly" if f >= 0.45 else
             "Quarter Kelly" if f >= 0.2 else
             f"{f:.0%} Kelly")

    advice = ""
    if f >= 0.9:
        advice = (" Almost nobody should use this: a 50% chance of halving your "
                  "bankroll at some point is not a risk most people have priced in.")
    elif f <= 0.3:
        advice = (" This is the sane setting. Even Bill Benter, with a proven "
                  "edge, lost about 20% of his capital in one of his first five "
                  "seasons using fractional Kelly.")

    return (
        f"{label}. Assuming the edge is real, the chance of your bankroll ever "
        f"halving is about {p_half:.1%}, and of falling to a third about "
        f"{p_third:.1%}. You keep roughly {growth:.0%} of the full-Kelly growth "
        f"rate.{advice}"
    )
