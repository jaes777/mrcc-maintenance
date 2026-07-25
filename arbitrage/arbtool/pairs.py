"""Turning two raw price feeds into one comparable, tradeable spread.

Two listings of the same company are quoted in different currencies and often in
different bundle sizes (one US ADR can represent 2, 5 or 8 ordinary shares). Only
after converting both to "price of one ordinary share, in one currency" can they
be compared at all. Everything downstream depends on getting this right, so the
share ratio is re-derived from the data rather than trusted from configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .data.providers import Bar, CachingProvider, DataError
from .stats import adf_like_stat, correlation, half_life, median, stdev
from .universe import PairSpec, VENUES


@dataclass
class PairSeries:
    """Aligned, currency-normalised history for one cross-listed pair."""

    pair: PairSpec
    dates: List[str]
    a_local: List[float]
    b_local: List[float]
    fx_a: List[float]
    fx_b: List[float]
    a_base: List[float]      # price of ONE ordinary share, in the base currency
    b_base: List[float]
    log_spread: List[float]  # ln(a_base) - ln(b_base); zero means exact parity
    base_currency: str

    def __len__(self) -> int:
        return len(self.dates)

    def slice(self, start_index: int, end_index: int) -> "PairSeries":
        s = slice(start_index, end_index)
        return PairSeries(
            self.pair, self.dates[s], self.a_local[s], self.b_local[s],
            self.fx_a[s], self.fx_b[s], self.a_base[s], self.b_base[s],
            self.log_spread[s], self.base_currency,
        )

    def index_of(self, iso_date: str) -> Optional[int]:
        try:
            return self.dates.index(iso_date)
        except ValueError:
            return None

    def gap_bps(self, i: int) -> float:
        """Signed size of the mispricing at bar ``i``, in basis points."""
        return self.log_spread[i] * 10_000.0


@dataclass
class PairQuality:
    """Verdict on whether a pair is fit to trade, and why."""

    pair_id: str
    tradeable: bool
    bars: int
    empirical_ratio: float
    configured_ratio: float
    ratio_error_pct: float
    return_correlation: float
    spread_halflife: Optional[float]
    spread_volatility_bps: float
    stationarity_stat: Optional[float]
    breakeven_bps: float
    median_abs_gap_bps: float
    opportunity_ratio: float          # typical gap divided by cost to trade it
    sessions_overlap: bool
    both_shortable: bool
    reasons: List[str] = field(default_factory=list)

    def summary(self) -> str:
        verdict = "TRADEABLE" if self.tradeable else "REJECTED"
        return (
            f"{self.pair_id:<6} {verdict:<10} bars={self.bars:<5} "
            f"corr={self.return_correlation:+.2f} "
            f"half-life={_fmt(self.spread_halflife)} "
            f"gap/cost={self.opportunity_ratio:.2f} "
            + ("" if self.tradeable else "| " + "; ".join(self.reasons))
        )


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f}d"


def _to_map(bars: Sequence[Bar]) -> Dict[str, float]:
    return {b.date: b.close for b in bars if b.close and b.close > 0}


def _forward_fill(dates: Sequence[str], values: Dict[str, float]) -> List[Optional[float]]:
    """Carry the last known value forward across missing days.

    Exchanges have different holidays, so one leg regularly has a bar when the
    other does not. Carrying forward is the honest choice: it uses the last
    price that genuinely existed, and never a future one.
    """
    out: List[Optional[float]] = []
    last: Optional[float] = None
    for d in dates:
        if d in values:
            last = values[d]
        out.append(last)
    return out


def build_pair_series(pair: PairSpec, provider: CachingProvider, start: str, end: str,
                      base_currency: str = "USD",
                      max_forward_fill_days: int = 5) -> PairSeries:
    """Fetch both legs plus FX, align them, and produce the spread."""
    a_bars = provider.fetch_leg(pair.a, start, end)
    b_bars = provider.fetch_leg(pair.b, start, end)
    if len(a_bars) < 30 or len(b_bars) < 30:
        raise DataError(
            f"{pair.pair_id}: not enough history "
            f"({len(a_bars)} and {len(b_bars)} bars) to analyse anything"
        )

    ccy_a = VENUES[pair.a.venue].currency
    ccy_b = VENUES[pair.b.venue].currency
    fx_a_bars = provider.fetch_fx(ccy_a, base_currency, start, end)
    fx_b_bars = provider.fetch_fx(ccy_b, base_currency, start, end)

    a_map, b_map = _to_map(a_bars), _to_map(b_bars)
    # Only days where BOTH venues actually traded become decision points.
    common = sorted(set(a_map) & set(b_map))
    if len(common) < 30:
        raise DataError(
            f"{pair.pair_id}: only {len(common)} days where both venues traded"
        )

    fx_a = _forward_fill(common, _to_map(fx_a_bars))
    fx_b = _forward_fill(common, _to_map(fx_b_bars))

    dates: List[str] = []
    a_local: List[float] = []
    b_local: List[float] = []
    fa: List[float] = []
    fb: List[float] = []
    a_base: List[float] = []
    b_base: List[float] = []
    spread: List[float] = []

    for i, day in enumerate(common):
        rate_a, rate_b = fx_a[i], fx_b[i]
        if rate_a is None or rate_b is None or rate_a <= 0 or rate_b <= 0:
            continue  # no exchange rate yet: the day is simply not comparable
        pa = a_map[day] * rate_a / pair.a.ordinary_shares
        pb = b_map[day] * rate_b / pair.b.ordinary_shares
        if pa <= 0 or pb <= 0:
            continue
        dates.append(day)
        a_local.append(a_map[day])
        b_local.append(b_map[day])
        fa.append(rate_a)
        fb.append(rate_b)
        a_base.append(pa)
        b_base.append(pb)
        spread.append(math.log(pa) - math.log(pb))

    if len(dates) < 30:
        raise DataError(f"{pair.pair_id}: too few usable days after alignment")

    return PairSeries(pair, dates, a_local, b_local, fa, fb, a_base, b_base,
                      spread, base_currency)


def infer_ratio(series: PairSeries) -> float:
    """Re-derive the share ratio implied by the prices themselves.

    If the configured ratio is right, this comes back close to it. If a company
    has changed its ADR ratio or done a split, this catches it — which matters,
    because a wrong ratio makes a perfectly normal pair look like a permanent
    100% mispricing and would otherwise generate confident, catastrophic trades.
    """
    ratios = [
        (a * fa) / (b * fb)
        for a, b, fa, fb in zip(series.a_local, series.b_local, series.fx_a, series.fx_b)
        if b > 0 and fb > 0
    ]
    return median(ratios) if ratios else float("nan")


def nearest_sensible_ratio(value: float) -> float:
    """Snap an inferred ratio to the round numbers issuers actually use."""
    candidates = [0.125, 0.2, 0.25, 1 / 3, 0.5, 2 / 3, 1.0, 1.5, 2.0, 3.0, 4.0,
                  5.0, 6.0, 8.0, 10.0, 20.0]
    if not math.isfinite(value) or value <= 0:
        return float("nan")
    return min(candidates, key=lambda c: abs(math.log(c) - math.log(value)))


def assess_pair(series: PairSeries, cost_model, signal_cfg, base_currency: str,
                allow_short: bool, ratio_tolerance_pct: float = 20.0) -> PairQuality:
    """Decide whether a pair is worth trading, with explicit reasons."""
    reasons: List[str] = []
    pair = series.pair

    configured = pair.a.ordinary_shares / pair.b.ordinary_shares
    empirical = infer_ratio(series)
    ratio_error = (
        abs(empirical - configured) / configured * 100.0
        if configured > 0 and math.isfinite(empirical) else float("inf")
    )

    returns_a = _log_returns(series.a_base)
    returns_b = _log_returns(series.b_base)
    corr = correlation(returns_a, returns_b)
    hl = half_life(series.log_spread)
    adf = adf_like_stat(series.log_spread)
    spread_vol_bps = stdev(series.log_spread) * 10_000.0
    median_gap_bps = median([abs(s) * 10_000.0 for s in series.log_spread])

    hold_days = hl if hl else float(signal_cfg.max_hold_days)
    breakeven = cost_model.breakeven_bps(pair, hold_days, base_currency, allow_short)
    # Two legs are traded, so the gap has to cover cost on both sides of it.
    opportunity = (spread_vol_bps * signal_cfg.entry_z) / breakeven if breakeven > 0 else 0.0

    if len(series) < 250:
        reasons.append(f"only {len(series)} usable days of history (want 250+)")
    if ratio_error > ratio_tolerance_pct:
        suggestion = nearest_sensible_ratio(empirical)
        reasons.append(
            f"share ratio looks wrong: data implies {empirical:.4g} but config says "
            f"{configured:.4g} ({ratio_error:.0f}% off; nearest standard ratio "
            f"{suggestion:.4g}) — fix the ratio before trading this pair"
        )
    if corr < 0.5:
        reasons.append(
            f"the two listings only move together {corr:.2f} — that is too loose "
            f"for them to be treated as the same asset"
        )
    if hl is None:
        reasons.append("the gap shows no tendency to close (no mean reversion)")
    elif hl < signal_cfg.min_halflife:
        reasons.append(f"gap closes in {hl:.1f} days — too fast to trade after costs")
    elif hl > signal_cfg.max_halflife:
        reasons.append(f"gap takes {hl:.1f} days to close — capital tied up too long")
    if adf is not None and adf > -2.0:
        reasons.append(
            f"the gap behaves like a random walk (stat {adf:.2f}) rather than "
            f"something that returns to normal"
        )
    if opportunity < 1.2:
        reasons.append(
            f"typical mispricing is only {opportunity:.2f}x the cost of trading it "
            f"— not enough edge to survive fees"
        )
    if allow_short and not pair.both_shortable:
        reasons.append(
            "one leg cannot reliably be sold short, so the hedge cannot be built"
        )

    return PairQuality(
        pair_id=pair.pair_id,
        tradeable=not reasons,
        bars=len(series),
        empirical_ratio=empirical,
        configured_ratio=configured,
        ratio_error_pct=ratio_error,
        return_correlation=corr,
        spread_halflife=hl,
        spread_volatility_bps=spread_vol_bps,
        stationarity_stat=adf,
        breakeven_bps=breakeven,
        median_abs_gap_bps=median_gap_bps,
        opportunity_ratio=opportunity,
        sessions_overlap=pair.sessions_overlap,
        both_shortable=pair.both_shortable,
        reasons=reasons,
    )


def _log_returns(prices: Sequence[float]) -> List[float]:
    return [
        math.log(prices[i] / prices[i - 1])
        for i in range(1, len(prices))
        if prices[i] > 0 and prices[i - 1] > 0
    ]


def ratio_report(series: PairSeries) -> Tuple[float, float, str]:
    """(configured, empirical, human-readable verdict)."""
    configured = series.pair.a.ordinary_shares / series.pair.b.ordinary_shares
    empirical = infer_ratio(series)
    if not math.isfinite(empirical) or configured <= 0:
        return configured, empirical, "could not be determined from the data"
    error = abs(empirical - configured) / configured * 100.0
    if error <= 5:
        verdict = "matches the data"
    elif error <= 20:
        verdict = f"slightly off ({error:.0f}%) — worth checking"
    else:
        verdict = (
            f"WRONG by {error:.0f}% — data suggests "
            f"{nearest_sensible_ratio(empirical):.4g}"
        )
    return configured, empirical, verdict
