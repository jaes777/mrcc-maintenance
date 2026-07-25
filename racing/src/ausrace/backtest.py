"""Walk-forward backtesting.

The only honest way to test a racing model is to make it predict races it has
never seen, using only information that existed before those races ran. That
means splitting by **date**, never by row.

Random train/test splits on racing data are invalid, and they fail in three
separate ways that all inflate the result:

1. **Temporal leakage** - the model learns from next month to predict last
   month.
2. **Entity leakage** - a horse's fifth start is in training while its third
   start is in test, so any career-aggregate feature carries the test outcome
   inside it. This one alone can manufacture an apparent 50-60% strike rate,
   and it is by far the most common way amateur racing models fool their
   authors.
3. **Broken fields** - splitting runners rather than races leaves incomplete
   fields, which makes within-race normalisation meaningless.

This module makes all three impossible: it splits on dates, keeps whole races
together, and the feature layer builds history strictly forward in time.

A useful warning sign to remember: if walk-forward scores are suspiciously
stable across folds, that usually means leakage rather than a robust model.
Real racing performance fluctuates a lot from period to period.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .evaluate import (
    bets_needed, calibration_table, flat_stake_roi, full_report,
    market_baseline, roi_by_odds_band, roi_significance, top1_strike_rate,
    winner_log_loss,
)
from .model import RaceModel


@dataclass
class BacktestResult:
    """Everything a walk-forward run produced."""

    predictions: pd.DataFrame
    folds: pd.DataFrame
    report: dict
    calibration: pd.DataFrame
    odds_bands: pd.DataFrame
    baseline: dict
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        r = self.report
        lines = [
            "=" * 72,
            "WALK-FORWARD BACKTEST",
            "=" * 72,
            f"Races tested:        {r['n_races']:,}",
            f"Runners:             {r['n_runners']:,}",
            "",
            "PROBABILITY QUALITY (lower log loss is better)",
            f"  Model (blended):   {r['log_loss']:.4f}",
        ]
        if "log_loss_market_only" in r:
            lines.append(f"  Market alone:      {r['log_loss_market_only']:.4f}")
        if "log_loss_model_only" in r:
            lines.append(f"  Model alone:       {r['log_loss_model_only']:.4f}")
        if "beats_market" in r:
            verdict = "YES" if r["beats_market"] else "NO"
            lines.append(f"  Beats the market?  {verdict}")
            if not r["beats_market"]:
                lines.append(
                    "  -> The model adds nothing the odds did not already say. "
                    "This is the most common outcome and it is real information."
                )
        lines += [
            "",
            "STRIKE RATE",
            f"  Top pick wins:     {r['top1_strike_rate']:.1%}",
            f"  Winner in top 3:   {r['top3_strike_rate']:.1%}",
        ]
        if self.baseline:
            lines.append(f"  Favourite wins:    {self.baseline.get('favourite_strike_rate', float('nan')):.1%}"
                         "   <- the bar to clear")
        lines += [
            "",
            "MONEY",
            f"  Bets placed:       {r['roi_n_bets']:,}",
            f"  ROI:               {r['roi']:+.2%}" if np.isfinite(r.get("roi", np.nan))
            else "  ROI:               n/a (no qualifying bets)",
            f"  {r['roi_verdict']}",
        ]
        if self.warnings:
            lines += ["", "WARNINGS"]
            lines += [f"  ! {w}" for w in self.warnings]
        lines.append("=" * 72)
        return "\n".join(lines)


def walk_forward(
    frame: pd.DataFrame,
    initial_train_days: int = 730,
    test_days: int = 90,
    step_days: Optional[int] = None,
    min_train_races: int = 2000,
    use_market: bool = True,
    min_edge: float = 0.05,
    commission: float = 0.0,
    params: Optional[dict] = None,
    verbose: bool = True,
) -> BacktestResult:
    """Expanding-window walk-forward backtest.

    Train on everything up to date T, predict the next `test_days`, roll
    forward, repeat. The training window expands rather than sliding, because
    racing form is fairly stationary and more data helps.

    `frame` must already have features (see `features.build_features`).
    """
    step_days = step_days or test_days
    data = frame[frame["won"].notna()].copy()
    data["race_datetime"] = pd.to_datetime(data["race_datetime"])
    data = data.sort_values("race_datetime").reset_index(drop=True)

    if data.empty:
        raise ValueError("No resolved races to backtest.")

    warnings: list[str] = []
    start_date = data["race_datetime"].min()
    end_date = data["race_datetime"].max()
    span_days = (end_date - start_date).days

    if span_days < initial_train_days + test_days:
        # Not enough history for the requested windows - fall back to a
        # proportional split rather than silently producing nothing.
        initial_train_days = max(int(span_days * 0.6), 1)
        test_days = max(int(span_days * 0.2), 1)
        step_days = test_days
        warnings.append(
            f"Data spans only {span_days} days; windows reduced to "
            f"{initial_train_days}d train / {test_days}d test. Short histories "
            f"give unreliable backtests."
        )

    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    cutoff = start_date + timedelta(days=initial_train_days)
    fold = 0

    while cutoff < end_date:
        test_end = cutoff + timedelta(days=test_days)
        train = data[data["race_datetime"] < cutoff]
        test = data[(data["race_datetime"] >= cutoff) & (data["race_datetime"] < test_end)]

        if test.empty:
            cutoff += timedelta(days=step_days)
            continue
        if train["race_id"].nunique() < min_train_races:
            cutoff += timedelta(days=step_days)
            continue

        # Hold out the last 15% of the training window (by date) to fit the
        # market blend and drive early stopping. Using a random slice here
        # would reintroduce exactly the leakage this module exists to prevent.
        split = train["race_datetime"].quantile(0.85)
        fit_part = train[train["race_datetime"] < split]
        valid_part = train[train["race_datetime"] >= split]
        if fit_part["race_id"].nunique() < min_train_races // 2:
            fit_part, valid_part = train, None

        model = RaceModel(use_market=use_market, params=params or RaceModel().params)
        try:
            model.fit(fit_part, valid=valid_part)
        except Exception as exc:
            warnings.append(f"Fold {fold} failed to train: {exc}")
            cutoff += timedelta(days=step_days)
            continue

        predicted = model.predict(test)
        predicted["fold"] = fold
        predictions.append(predicted)

        fold_report = full_report(predicted, min_edge=min_edge, commission=commission)
        fold_rows.append({
            "fold": fold,
            "train_end": cutoff,
            "test_start": test.race_datetime.min(),
            "test_end": test.race_datetime.max(),
            "train_races": int(train["race_id"].nunique()),
            "test_races": int(test["race_id"].nunique()),
            "log_loss": fold_report["log_loss"],
            "log_loss_market": fold_report.get("log_loss_market_only", np.nan),
            "top1": fold_report["top1_strike_rate"],
            "roi": fold_report["roi"],
            "n_bets": fold_report["roi_n_bets"],
            "alpha": model.alpha,
            "beta": model.beta,
        })

        if verbose:
            print(
                f"  fold {fold:>2}  train<{cutoff.date()}  "
                f"test {test.race_datetime.min().date()}..{test.race_datetime.max().date()}  "
                f"races={test['race_id'].nunique():>5}  "
                f"logloss={fold_report['log_loss']:.4f}  "
                f"top1={fold_report['top1_strike_rate']:.1%}  "
                f"bets={fold_report['roi_n_bets']:>4}  "
                f"roi={fold_report['roi']:+.1%}" if np.isfinite(fold_report.get("roi", np.nan))
                else f"  fold {fold:>2}  (no qualifying bets)"
            )

        cutoff += timedelta(days=step_days)
        fold += 1

    if not predictions:
        raise RuntimeError(
            "No folds completed. There is not enough history, or "
            "min_train_races is set higher than the data supports."
        )

    combined = pd.concat(predictions, ignore_index=True)
    folds = pd.DataFrame(fold_rows)

    report = full_report(combined, min_edge=min_edge, commission=commission)
    baseline = market_baseline(combined)

    # --- Honesty checks -------------------------------------------------
    if report["roi_n_bets"] > 0 and np.isfinite(report.get("roi", np.nan)) and report["roi"] > 0:
        bets = combined[combined["won"].notna() & combined["odds_decimal"].notna()]
        average_odds = float(bets["odds_decimal"].median()) if len(bets) else 5.0
        required = bets_needed(max(report["roi"], 0.01), average_odds)
        if required > 0 and report["roi_n_bets"] < required:
            warnings.append(
                f"The measured ROI of {report['roi']:+.1%} rests on "
                f"{report['roi_n_bets']:,} bets. Demonstrating an edge of that "
                f"size at these prices needs roughly {required:,} bets. This "
                f"result is not yet distinguishable from luck."
            )

    if len(folds) >= 3:
        roi_by_fold = folds["roi"].dropna()
        if len(roi_by_fold) >= 3:
            if (roi_by_fold > 0).all():
                warnings.append(
                    "Every fold is profitable. In racing that is unusual enough "
                    "to be worth double-checking for leakage before believing it."
                )
            if roi_by_fold.std() < 0.01:
                warnings.append(
                    "ROI is near-identical across folds. Real racing results "
                    "fluctuate; suspiciously stable folds usually mean leakage."
                )

    if "beats_market" in report and not report["beats_market"]:
        warnings.append(
            "The model does not beat the market's own log loss. Any positive "
            "ROI below is very unlikely to survive live betting."
        )

    return BacktestResult(
        predictions=combined,
        folds=folds,
        report=report,
        calibration=calibration_table(combined),
        odds_bands=roi_by_odds_band(combined, min_edge=min_edge),
        baseline=baseline,
        warnings=warnings,
    )


def simulate_bankroll(
    predictions: pd.DataFrame,
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 0.02,
    min_edge: float = 0.05,
    commission: float = 0.0,
) -> pd.DataFrame:
    """Replay the bets chronologically to see the actual bankroll path.

    ROI hides the thing that actually ends betting operations, which is the
    drawdown along the way. A strategy that finishes +8% having been 40% down
    at one point is not one most people can sit through.
    """
    from .betting import expected_value, staking_plan

    data = predictions[
        predictions["won"].notna() & predictions["p_win"].notna()
        & predictions["odds_decimal"].notna() & (predictions["odds_decimal"] > 1)
        & ~predictions["scratched"].astype(bool)
    ].sort_values("race_datetime").copy()

    balance = bankroll
    peak = bankroll
    rows = []

    for race_id, race in data.groupby("race_id", sort=False):
        staked = 0.0
        returned = 0.0
        for _, runner in race.iterrows():
            ev = expected_value(runner["p_win"], runner["odds_decimal"], commission)
            if ev < min_edge:
                continue
            stake = staking_plan(
                runner["p_win"], runner["odds_decimal"], balance,
                kelly_fraction_used=kelly_fraction,
                max_stake_pct=max_stake_pct, commission=commission,
            )
            if stake <= 0:
                continue
            staked += stake
            if runner["won"] == 1:
                returned += stake * (1 + (runner["odds_decimal"] - 1) * (1 - commission))

        if staked <= 0:
            continue

        balance += returned - staked
        peak = max(peak, balance)
        rows.append({
            "race_datetime": race["race_datetime"].iloc[0],
            "race_id": race_id,
            "staked": staked,
            "returned": returned,
            "profit": returned - staked,
            "balance": balance,
            "peak": peak,
            "drawdown": (peak - balance) / peak if peak > 0 else 0.0,
        })
        if balance <= 0:
            rows[-1]["note"] = "BANKRUPT"
            break

    return pd.DataFrame(rows)


def bankroll_summary(path: pd.DataFrame, starting: float = 1000.0) -> dict:
    """Headline numbers from a bankroll simulation."""
    if path.empty:
        return {"note": "No bets were placed."}
    final = float(path["balance"].iloc[-1])
    return {
        "starting_bankroll": starting,
        "final_bankroll": final,
        "total_return_pct": (final - starting) / starting,
        "races_bet": int(len(path)),
        "total_staked": float(path["staked"].sum()),
        "roi_on_turnover": float(path["profit"].sum() / path["staked"].sum()),
        "max_drawdown_pct": float(path["drawdown"].max()),
        "longest_losing_run": int(_longest_run(path["profit"] < 0)),
        "went_broke": bool(final <= 0),
    }


def _longest_run(flags: pd.Series) -> int:
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest
