"""The simulation engine and the walk-forward protocol.

Two rules govern everything in this file:

1. **A signal seen at today's close is acted on at the next bar's price.** You
   cannot trade at a price you only learned about after the market closed.
2. **Parameters are chosen on one slice of history and scored on the next.**
   The scored slice is never used to pick anything. Results from the fitting
   slice are reported but never gated on, because any strategy can be made to
   look wonderful on the data used to build it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from .costs import CostModel
from .metrics import Metrics, compute_metrics, equity_from_trades
from .models import ModelSpec, score
from .pairs import PairSeries
from .portfolio import Trade

EXECUTION_LAG_BARS = 1


@dataclass
class BacktestContext:
    cost_model: CostModel
    base_currency: str = "USD"
    notional_per_leg: float = 10_000.0
    allow_short: bool = True
    fx_hedged: bool = False
    execution_lag_bars: int = EXECUTION_LAG_BARS


def _days_between(d0: str, d1: str) -> int:
    a = datetime.strptime(d0, "%Y-%m-%d").date()
    b = datetime.strptime(d1, "%Y-%m-%d").date()
    return max(0, (b - a).days)


def _prices(series: PairSeries, index: int, fx_index: Optional[int]) -> Tuple[float, float]:
    """Base-currency price of one ordinary share on each leg.

    ``fx_index`` pins the exchange rate to an earlier bar, which is how a
    currency-hedged position is modelled: the share price still moves, the
    exchange rate no longer does.
    """
    if fx_index is None:
        return series.a_base[index], series.b_base[index]
    a = series.a_local[index] * series.fx_a[fx_index] / series.pair.a.ordinary_shares
    b = series.b_local[index] * series.fx_b[fx_index] / series.pair.b.ordinary_shares
    return a, b


def simulate(series: PairSeries, spec: ModelSpec, ctx: BacktestContext,
             start_index: int = 0, end_index: Optional[int] = None,
             fold: int = -1, out_of_sample: bool = True,
             scores: Optional[Sequence[Optional[float]]] = None) -> List[Trade]:
    """Run one model over one slice of one pair's history."""
    n = len(series)
    end_index = n if end_index is None else min(end_index, n)
    if scores is None:
        scores = score(series.log_spread, spec)

    trades: List[Trade] = []
    lag = max(1, ctx.execution_lag_bars)
    long_only = not ctx.allow_short

    open_pos: Optional[Dict] = None
    i = start_index
    while i < end_index:
        z = scores[i]
        if z is None:
            i += 1
            continue

        if open_pos is None:
            if abs(z) >= spec.entry_z:
                fill = i + lag
                if fill >= end_index:
                    break
                open_pos = _open(series, ctx, spec, fill, z, long_only)
            i += 1
            continue

        # Position is open: decide whether this bar's reading closes it.
        bars_held = i - open_pos["entry_index"]
        reason = None
        if abs(z) <= spec.exit_z:
            reason = "converged"
        elif abs(z) >= spec.stop_z:
            reason = "stop"
        elif bars_held >= spec.max_hold_days:
            reason = "time"

        if reason:
            fill = min(i + lag, end_index - 1)
            if fill <= open_pos["entry_index"]:
                i += 1
                continue
            trades.append(_close(series, ctx, spec, open_pos, fill, z, reason,
                                 fold, out_of_sample))
            open_pos = None
        i += 1

    if open_pos is not None and end_index - 1 > open_pos["entry_index"]:
        last = end_index - 1
        final_z = scores[last] if scores[last] is not None else open_pos["entry_z"]
        trades.append(_close(series, ctx, spec, open_pos, last, final_z,
                             "end_of_data", fold, out_of_sample))
    return trades


def _open(series: PairSeries, ctx: BacktestContext, spec: ModelSpec,
          index: int, z: float, long_only: bool) -> Dict:
    fx_index = index if ctx.fx_hedged else None
    price_a, price_b = _prices(series, index, fx_index)
    direction = "long_b_short_a" if z > 0 else "long_a_short_b"
    return {
        "entry_index": index,
        "entry_date": series.dates[index],
        "entry_z": z,
        "entry_gap_bps": series.gap_bps(index),
        "entry_a": price_a,
        "entry_b": price_b,
        "shares_a": ctx.notional_per_leg / price_a,
        "shares_b": ctx.notional_per_leg / price_b,
        "direction": direction,
        "fx_index": fx_index,
        "long_only": long_only,
    }


def _close(series: PairSeries, ctx: BacktestContext, spec: ModelSpec, pos: Dict,
           index: int, z: float, reason: str, fold: int, oos: bool) -> Trade:
    price_a, price_b = _prices(series, index, pos["fx_index"])
    side_a = -1 if pos["direction"] == "long_b_short_a" else 1
    side_b = -side_a

    gross = side_a * pos["shares_a"] * (price_a - pos["entry_a"])
    if not pos["long_only"]:
        gross += side_b * pos["shares_b"] * (price_b - pos["entry_b"])
    else:
        # Long-only keeps just the cheap leg, so if the cheap leg is A the
        # position is the A leg alone; otherwise it is the B leg alone.
        gross = (pos["shares_a"] * (price_a - pos["entry_a"]) if side_a > 0
                 else pos["shares_b"] * (price_b - pos["entry_b"]))

    hold_days = _days_between(pos["entry_date"], series.dates[index])
    cost_bps = ctx.cost_model.round_trip_bps(
        series.pair, hold_days, ctx.base_currency, allow_short=not pos["long_only"]
    )["total"]
    costs = cost_bps / 10_000.0 * ctx.notional_per_leg
    net = gross - costs

    return Trade(
        pair_id=series.pair.pair_id,
        model=spec.key(),
        direction=pos["direction"],
        entry_date=pos["entry_date"],
        exit_date=series.dates[index],
        entry_z=pos["entry_z"],
        exit_z=z,
        entry_gap_bps=pos["entry_gap_bps"],
        exit_gap_bps=series.gap_bps(index),
        notional_per_leg=ctx.notional_per_leg,
        gross_pnl=gross,
        costs=costs,
        net_pnl=net,
        return_bps=net / ctx.notional_per_leg * 10_000.0,
        hold_days=hold_days,
        exit_reason=reason,
        fold=fold,
        out_of_sample=oos,
        long_only=pos["long_only"],
    )


# --------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------


@dataclass
class Fold:
    index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    chosen: Optional[ModelSpec]
    train_metrics: Optional[Metrics]
    test_metrics: Optional[Metrics]
    candidates_tried: int
    note: str = ""

    def summary(self) -> str:
        if self.chosen is None:
            return f"fold {self.index}: no model qualified on training data ({self.note})"
        train = self.train_metrics.win_rate * 100 if self.train_metrics else 0
        test = self.test_metrics.win_rate * 100 if self.test_metrics else 0
        trades = self.test_metrics.trades if self.test_metrics else 0
        return (
            f"fold {self.index}: {self.test_start}..{self.test_end} "
            f"chose {self.chosen.name} (lb={self.chosen.lookback}, "
            f"in={self.chosen.entry_z}, out={self.chosen.exit_z}) | "
            f"fitted win {train:.0f}% -> unseen win {test:.0f}% over {trades} trades"
        )


@dataclass
class WalkForwardResult:
    pair_id: str
    folds: List[Fold]
    oos_trades: List[Trade]
    in_sample_trades: List[Trade]
    oos_metrics: Metrics
    degradation_pct: float = 0.0     # how much win rate fell from fitted to unseen
    note: str = ""

    @property
    def usable(self) -> bool:
        return bool(self.oos_trades)


def walk_forward(series: PairSeries, ctx: BacktestContext, specs: Sequence[ModelSpec],
                 train_days: int, test_days: int, step_days: int,
                 purge_days: int = 2, min_folds: int = 3,
                 min_train_trades: int = 5, capital: float = 100_000.0,
                 confidence: float = 0.95) -> WalkForwardResult:
    """Fit on the past, score on the future, repeatedly."""
    n = len(series)
    folds: List[Fold] = []
    oos: List[Trade] = []
    ins: List[Trade] = []

    # Scores depend only on the model spec, so they are computed once per spec
    # over the whole series and sliced per fold. Each score is still causal.
    cache: Dict[str, List[Optional[float]]] = {}

    def scores_for(spec: ModelSpec) -> List[Optional[float]]:
        key = f"{spec.name}|{spec.lookback}|{spec.extra}"
        if key not in cache:
            cache[key] = score(series.log_spread, spec)
        return cache[key]

    fold_index = 0
    train_start = 0
    while True:
        train_end = train_start + train_days
        test_start = train_end + purge_days
        test_end = test_start + test_days
        if test_end > n:
            break

        candidates: List[Tuple[float, ModelSpec, Metrics]] = []
        for spec in specs:
            train_trades = simulate(series, spec, ctx, train_start, train_end,
                                    fold=fold_index, out_of_sample=False,
                                    scores=scores_for(spec))
            if len(train_trades) < min_train_trades:
                continue
            m = compute_metrics(train_trades,
                                equity_from_trades(train_trades, capital),
                                confidence)
            # Reward consistent per-trade profit, and credit sample size so a
            # single lucky trade cannot win the selection.
            candidates.append((m.expectancy_bps * math.sqrt(m.trades), spec, m))

        if not candidates:
            folds.append(Fold(fold_index, series.dates[train_start],
                              series.dates[train_end - 1], series.dates[test_start],
                              series.dates[test_end - 1], None, None, None,
                              len(specs), "no parameter set produced enough trades"))
            train_start += step_days
            fold_index += 1
            continue

        best_objective = max(c[0] for c in candidates)
        # Among near-equal performers prefer the highest reliable win rate, which
        # is the objective the user actually asked for.
        threshold = best_objective * 0.9 if best_objective > 0 else best_objective * 1.1
        shortlist = [c for c in candidates if c[0] >= threshold] or candidates
        _, chosen_spec, chosen_metrics = max(
            shortlist, key=lambda c: (c[2].win_rate_lower, c[0])
        )

        ins.extend(simulate(series, chosen_spec, ctx, train_start, train_end,
                            fold_index, False, scores_for(chosen_spec)))
        test_trades = simulate(series, chosen_spec, ctx, test_start, test_end,
                               fold_index, True, scores_for(chosen_spec))
        oos.extend(test_trades)
        test_metrics = compute_metrics(test_trades,
                                       equity_from_trades(test_trades, capital),
                                       confidence) if test_trades else None

        folds.append(Fold(fold_index, series.dates[train_start],
                          series.dates[train_end - 1], series.dates[test_start],
                          series.dates[test_end - 1], chosen_spec, chosen_metrics,
                          test_metrics, len(specs)))
        train_start += step_days
        fold_index += 1

    oos_metrics = compute_metrics(oos, equity_from_trades(oos, capital), confidence)
    in_metrics = compute_metrics(ins, equity_from_trades(ins, capital), confidence)
    degradation = ((in_metrics.win_rate - oos_metrics.win_rate) * 100.0
                   if ins and oos else 0.0)

    note = ""
    if len(folds) < min_folds:
        note = (f"only {len(folds)} out-of-sample windows were possible with this "
                f"much history — at least {min_folds} are needed for a credible result")
    return WalkForwardResult(series.pair.pair_id, folds, oos, ins, oos_metrics,
                             degradation, note)
