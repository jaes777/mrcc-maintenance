"""What a trade actually costs.

This module exists because it is where most "arbitrage" ideas die. A price gap
of 40 basis points between two listings looks like free money until you add
commission, the bid/ask spread on both legs, currency conversion, stamp duty,
stock borrow, and the slippage you eat when the two exchanges are not even open
at the same time. The system therefore refuses to evaluate a signal without
charging every one of those costs first.

All figures are in basis points (1 bp = 0.01%).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .universe import PairSpec, VENUES, Venue


@dataclass
class CostModel:
    """Per-trade frictions. Defaults lean towards a retail cost structure."""

    fx_conversion_bps: float = 15.0        # spread a retail broker takes on FX
    non_overlap_slippage_bps: float = 12.0  # legs cannot be filled simultaneously
    impact_bps_per_1pct_adv: float = 8.0    # market impact for larger orders
    min_commission_ccy: float = 0.0         # per-order floor, in base currency
    borrow_multiplier: float = 1.0          # scale venue borrow rates
    extra_slippage_bps: float = 0.0         # global stress knob for beta testing

    def venue(self, code: str) -> Venue:
        return VENUES[code]

    # --- individual components --------------------------------------------

    def entry_cost_bps(self, venue_code: str, side: str, needs_fx: bool) -> float:
        """Cost of opening one leg: commission + half spread + tax + FX."""
        v = self.venue(venue_code)
        cost = v.commission_bps + v.half_spread_bps + self.extra_slippage_bps
        cost += v.buy_tax_bps if side == "long" else v.sell_tax_bps
        if needs_fx:
            cost += self.fx_conversion_bps
        return cost

    def exit_cost_bps(self, venue_code: str, side: str, needs_fx: bool) -> float:
        """Cost of closing one leg. Closing a long is a sell, and vice versa."""
        v = self.venue(venue_code)
        cost = v.commission_bps + v.half_spread_bps + self.extra_slippage_bps
        cost += v.sell_tax_bps if side == "long" else v.buy_tax_bps
        if needs_fx:
            cost += self.fx_conversion_bps
        return cost

    def borrow_bps(self, venue_code: str, hold_days: float) -> float:
        v = self.venue(venue_code)
        return v.borrow_bps_per_year * self.borrow_multiplier * max(0.0, hold_days) / 365.0

    def timing_penalty_bps(self, pair: PairSpec) -> float:
        """Charged when the two exchanges are never open at the same time.

        With no overlap you place the second leg hours after the first, during
        which the price can move against you. This is real money, so it is
        charged as a cost rather than assumed away.
        """
        return 0.0 if pair.sessions_overlap else self.non_overlap_slippage_bps

    # --- combined ----------------------------------------------------------

    def round_trip_bps(self, pair: PairSpec, hold_days: float, base_currency: str,
                       allow_short: bool = True) -> Dict[str, float]:
        """Total cost of one complete pair trade, itemised.

        Returned in basis points *of one leg's notional*. A two-leg trade is
        charged on both legs, so the total below is what the price gap must
        exceed just to break even.
        """
        a, b = pair.a, pair.b
        a_fx = VENUES[a.venue].currency != base_currency
        b_fx = VENUES[b.venue].currency != base_currency

        # Which leg is bought and which is sold flips with the direction of the
        # gap, but the total does not: over a complete round trip each leg is
        # bought exactly once and sold exactly once either way, so every
        # commission, spread crossing and transaction tax is paid regardless.
        # Leg A is priced as the long side purely so the breakdown has labels.
        items: Dict[str, float] = {
            "entry_a": self.entry_cost_bps(a.venue, "long", a_fx),
            "exit_a": self.exit_cost_bps(a.venue, "long", a_fx),
        }
        if allow_short:
            items["entry_b"] = self.entry_cost_bps(b.venue, "short", b_fx)
            items["exit_b"] = self.exit_cost_bps(b.venue, "short", b_fx)
            items["borrow"] = self.borrow_bps(b.venue, hold_days)
        items["timing"] = self.timing_penalty_bps(pair)
        items["total"] = sum(items.values())
        return items

    def breakeven_bps(self, pair: PairSpec, hold_days: float, base_currency: str,
                      allow_short: bool = True) -> float:
        """The minimum gap, in bps, that makes a trade worth doing at all."""
        return self.round_trip_bps(pair, hold_days, base_currency, allow_short)["total"]

    def stressed(self, slippage_multiplier: float = 2.0,
                 borrow_multiplier: float = 2.0) -> "CostModel":
        """A harsher copy of this model, used by the beta-test agents."""
        return CostModel(
            fx_conversion_bps=self.fx_conversion_bps * slippage_multiplier,
            non_overlap_slippage_bps=self.non_overlap_slippage_bps * slippage_multiplier,
            impact_bps_per_1pct_adv=self.impact_bps_per_1pct_adv * slippage_multiplier,
            min_commission_ccy=self.min_commission_ccy,
            borrow_multiplier=borrow_multiplier,
            extra_slippage_bps=self.extra_slippage_bps + 3.0 * (slippage_multiplier - 1.0),
        )

    def explain(self, pair: PairSpec, hold_days: float, base_currency: str,
                allow_short: bool = True) -> List[str]:
        """Plain-English breakdown, for the report and the CLI."""
        items = self.round_trip_bps(pair, hold_days, base_currency, allow_short)
        a, b = pair.a.symbol, pair.b.symbol
        lines = [
            f"Cost of one complete {pair.pair_id} trade held {hold_days:.0f} days "
            f"(basis points of one leg):"
        ]
        labels = {
            "entry_a": f"opening the {a} leg",
            "exit_a": f"closing the {a} leg",
            "entry_b": f"opening the {b} leg",
            "exit_b": f"closing the {b} leg",
            "borrow": "borrowing shares in order to sell short",
            "timing": "the two exchanges never being open at the same time",
        }
        for key, label in labels.items():
            if key in items and items[key]:
                lines.append(f"  {items[key]:6.1f} bp  {label}")
        lines.append(
            f"  {items['total']:6.1f} bp  TOTAL — the mispricing must be bigger "
            f"than this before the trade makes any money"
        )
        lines.append(
            "  (Which leg you buy and which you sell flips with the direction of "
            "the gap;"
        )
        lines.append(
            "   the total does not, because each leg is bought once and sold once "
            "either way.)"
        )
        return lines


@dataclass
class CostOverrides:
    """User overrides applied on top of the venue defaults."""

    commission_bps: Dict[str, float] = field(default_factory=dict)
    half_spread_bps: Dict[str, float] = field(default_factory=dict)

    def apply(self) -> None:
        for code, value in self.commission_bps.items():
            if code in VENUES:
                VENUES[code] = _replace_venue(VENUES[code], commission_bps=value)
        for code, value in self.half_spread_bps.items():
            if code in VENUES:
                VENUES[code] = _replace_venue(VENUES[code], half_spread_bps=value)


def _replace_venue(venue: Venue, **changes) -> Venue:
    import dataclasses
    return dataclasses.replace(venue, **changes)
