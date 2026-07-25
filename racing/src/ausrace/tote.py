"""Tote (parimutuel) pricing, takeout, breakage and flexi betting.

Fixed odds tell you the price before you bet. The tote does not: your dividend
depends on how much money everyone else put on, and it is only known after the
race. That makes tote expected value a different calculation, and it is where
most naive betting tools quietly go wrong.

The takeout is the reason this matters so much. TAB removes roughly 14-15% from
the win pool before paying anyone, and **21-23% from trifecta and first-four
pools**. Compare that to a fixed-odds book at 115-125%. Exotic pools are, in
pure expected-value terms, the most expensive bets available in Australian
racing, and any tool that recommends trifectas without accounting for a 21%
haircut is misleading you.

The single most useful formula here:

    Edge = (1 - takeout) * (p_model / q_market) - 1

Your edge is your probability divided by the market's, haircut by the takeout.
At a 14.5% win takeout, your model must be at least **17% more confident than
the pool** before the bet even breaks even. At the 23% first-four takeout, it
needs to be nearly **30%** more confident. That threshold is the honest reason
most punters lose.

Rates below are the published NSW TAB schedule, which is being harmonised under
the national tote merger. They are configuration, not constants - check them
against your operator before relying on the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Takeout by pool type. Sources: NSW TAB schedule; TAB quotes an overall range
# of 14.25%-25% across bet types. The national-tote rollout has moved several
# of these (trifecta 21 -> 21.5, first four 22.5 -> 23, quaddie 20 -> 20.5),
# so verify against your operator rather than trusting these.
TAKEOUT: dict[str, float] = {
    "WIN": 0.145,
    "PLACE": 0.1425,
    "DUET": 0.145,
    "QUINELLA": 0.1475,
    "EXACTA": 0.165,
    "DOUBLE": 0.17,
    "QUADDIE": 0.205,
    "TRIFECTA": 0.215,
    "FIRST_FOUR": 0.23,
    "BIG6": 0.25,
}

# Australian dividends are rounded DOWN, normally to the nearest 10c, with a
# minimum dividend around $1.00-$1.04. Breakage makes the effective takeout
# higher than the nominal rate, and it bites hardest on short-priced runners
# where 10c is a large fraction of the dividend.
BREAKAGE_INCREMENT = 0.10
MINIMUM_DIVIDEND = 1.04


def apply_breakage(dividend: float,
                   increment: float = BREAKAGE_INCREMENT,
                   minimum: float = MINIMUM_DIVIDEND) -> float:
    """Round a theoretical dividend down the way a tote actually pays it."""
    if not np.isfinite(dividend) or dividend <= 0:
        return 0.0
    rounded = np.floor(dividend / increment) * increment
    return float(max(rounded, minimum))


def tote_dividend(
    pool: float,
    money_on_selection: float,
    takeout: float,
    your_stake: float = 0.0,
    n_places: int = 1,
    breakage: bool = True,
) -> float:
    """Dividend per $1, given the pool and the money on your selection.

    Australian dividends *include* the stake, so $3.50 means $3.50 back per
    dollar. Your own stake is added to both the pool and the money on your
    selection, because betting into a pool dilutes the very dividend you are
    trying to collect - a first-order effect on thin exotic pools, not a
    rounding detail.

    For place pools the net pool is split into `n_places` equal sub-pools.
    """
    total_pool = float(pool) + float(your_stake)
    on_selection = float(money_on_selection) + float(your_stake)
    if on_selection <= 0 or total_pool <= 0:
        return 0.0
    net = total_pool * (1.0 - takeout)
    if n_places > 1:
        net /= n_places
    dividend = net / on_selection
    return apply_breakage(dividend) if breakage else float(dividend)


def implied_dividend(market_probability: float, takeout: float,
                     breakage: bool = True) -> float:
    """Dividend implied by a market probability and a takeout.

    This is the workhorse when you cannot see the pool: if the market thinks a
    horse is a 20% chance and the takeout is 14.5%, the dividend will be about
    (1 - 0.145) / 0.20 = $4.28.
    """
    if not np.isfinite(market_probability) or market_probability <= 0:
        return 0.0
    dividend = (1.0 - takeout) / market_probability
    return apply_breakage(dividend) if breakage else float(dividend)


def tote_edge(model_probability: float, market_probability: float,
              takeout: float) -> float:
    """Expected profit per $1 staked into a tote pool.

    Edge = (1 - takeout) * (p_model / q_market) - 1
    """
    if not np.isfinite(market_probability) or market_probability <= 0:
        return float("-inf")
    return float((1.0 - takeout) * (model_probability / market_probability) - 1.0)


def required_confidence(takeout: float) -> float:
    """How much more confident than the market you must be just to break even.

    Returns the ratio p_model / q_market at which edge is exactly zero. At a
    14.5% win takeout this is 1.17; at the 23% first-four takeout it is 1.30.
    """
    return float(1.0 / (1.0 - takeout))


# --------------------------------------------------------------------------
# Flexi betting
# --------------------------------------------------------------------------

@dataclass
class FlexiBet:
    combinations: int
    outlay: float
    unit: float = 1.0

    @property
    def full_cost(self) -> float:
        return self.combinations * self.unit

    @property
    def percentage(self) -> float:
        """Your share of the declared dividend, as a fraction."""
        if self.full_cost <= 0:
            return 0.0
        return self.outlay / self.full_cost

    def payout(self, declared_dividend: float) -> float:
        """What a winning combination actually returns you."""
        return self.percentage * float(declared_dividend)

    def describe(self) -> str:
        return (
            f"{self.combinations} combinations, full cost ${self.full_cost:,.2f}. "
            f"Betting ${self.outlay:,.2f} = {self.percentage:.1%} flexi, so you "
            f"collect {self.percentage:.1%} of the declared dividend."
        )


def flexi_percentage(combinations: int, outlay: float, unit: float = 1.0) -> float:
    """Flexi fraction: outlay / (combinations x unit). TAB's minimum is 1%."""
    full = combinations * unit
    return float(outlay / full) if full > 0 else 0.0


def flexi_roi(combination_probs: dict, dividends: dict, combinations: int) -> float:
    """Return on investment for a multi-combination ticket, per dollar staked.

    **Flexi percentage does not appear in this formula, and that is the point.**
    Taking 10% flexi instead of 100% scales your outlay and your return by the
    same factor, so the ROI is identical. Flexi is a bankroll control, never a
    source of edge - "it's only $36 instead of $360" does not make a bad ticket
    good. Rank tickets by this ROI, then size them with Kelly.
    """
    if combinations <= 0:
        return float("-inf")
    expected = sum(prob * dividends.get(combo, 0.0)
                   for combo, prob in combination_probs.items())
    return float(expected / combinations - 1.0)


def marginal_combination_value(prob: float, dividend: float) -> float:
    """Expected value of adding one more combination to a ticket.

    Adding a fifth horse to a boxed trifecta takes it from 24 combinations to
    60 - you are buying 36 extra combinations. Each one is only worth having if
    it is individually positive. Boxing by habit is how a ticket with two good
    combinations and thirty bad ones ends up losing money.
    """
    return float(prob * dividend - 1.0)


# --------------------------------------------------------------------------
# Promotional payouts
# --------------------------------------------------------------------------

def best_tote_value_note() -> str:
    """Why Best Tote and Top Fluc cannot be valued from a point estimate."""
    return (
        "Best Tote pays the highest of the three TAB dividends, so its value is "
        "E[max(D1, D2, D3)], which by Jensen's inequality is strictly greater "
        "than the best single expected dividend whenever the three totes differ "
        "at all. Valuing it as 'the best tote I expect' systematically "
        "understates it. The gap is widest on longshots and in small pools - "
        "which is exactly where these promotions are offered. Estimating it "
        "properly needs the joint distribution of the three dividends from "
        "historical data."
    )


def deduction_note() -> str:
    """How scratchings affect an already-struck fixed-odds bet."""
    return (
        "Bets struck BEFORE final fields are all-in: no refund and no deductions "
        "if a runner is scratched. Bets struck AFTER final fields get the "
        "scratched runner refunded, and deductions are applied to the remaining "
        "runners in cents per dollar, based on the scratched runner's price at "
        "withdrawal. Tote bets are simply refunded with no deduction. This "
        "matters for expected value: an all-in price is worse than the same "
        "number taken after final fields."
    )
