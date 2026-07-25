"""Evaluation metrics.

The metric that matters is NOT top-1 accuracy. It is included, but only as
a sanity check, because it is the number most likely to mislead you:

  * In an average Australian field of about eleven, always predicting
    "loses" for every runner scores ~91% on the runner-level binary label.
    Published papers claiming 95%+ "accuracy" for race prediction are
    almost always reporting this, and it means nothing.
  * The market favourite wins roughly 31-35% of Australian races. That is
    close to the *ceiling* for top-1 accuracy, not a baseline to beat --
    if the market is near-efficient, a model that knew the true
    probabilities exactly would still only pick the winner about a third
    of the time. The rest is irreducible randomness in the event itself.
  * So a top-1 accuracy far above ~36% is not a triumph. It is a leak.
    Treat it as a bug report and go looking for the future information
    that got into your features.

The metrics that actually matter, in order:

  1. Log loss against the devigged market baseline. This is the real test.
     A model that cannot beat the market's own probabilities out of sample
     has no business placing a bet. Expect to beat it by tenths of a
     percent to a couple of percent -- Benter's celebrated improvement was
     R^2 0.1245 -> 0.1396.
  2. Calibration. When you say 20%, does it happen 20% of the time? Value
     lives in the 2-10% band and that is where miscalibration is most
     expensive.
  3. Yield (ROI) by edge threshold, with bet counts attached. A +40% yield
     on 60 bets is noise, not an edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class RacePrediction:
    """Model output for one race, paired with what actually happened."""

    race_id: str
    date: str
    probabilities: np.ndarray
    market_probabilities: Optional[np.ndarray]
    odds: np.ndarray
    winner_index: Optional[int]
    runner_names: list[str] = field(default_factory=list)
    finish_order: Optional[list[int]] = None
    field_size: int = 0


# --------------------------------------------------------------------------
# Probability quality
# --------------------------------------------------------------------------

def log_loss(predictions: Sequence[RacePrediction],
             use_market: bool = False) -> float:
    """Mean negative log-likelihood of the actual winner.

    This is a *race-level multiclass* log loss: one term per race, not one
    per runner. Lower is better.

    A probability of exactly zero on the actual winner is NOT skipped. It
    is the single worst thing a model can do -- it declared the outcome
    impossible -- and dropping those races quietly deletes the model's
    catastrophic failures from its own scorecard. They are floored at
    1e-15 and counted, which is brutal, and correct. (Zero is reachable in
    practice: the softmax underflows when the strength gap is large.)
    Only genuinely non-finite values are skipped, and those are counted
    separately so a silently broken run cannot look like a clean one.
    """
    total, count = 0.0, 0
    for prediction in predictions:
        if prediction.winner_index is None:
            continue
        probs = (prediction.market_probabilities if use_market
                 else prediction.probabilities)
        if probs is None or prediction.winner_index >= len(probs):
            continue
        p = probs[prediction.winner_index]
        if not np.isfinite(p):
            continue
        total += -math.log(max(float(p), 1e-15))
        count += 1
    return total / count if count else float("nan")


def count_broken(predictions: Sequence[RacePrediction],
                 use_market: bool = False) -> int:
    """Races whose probability vector is unusable (non-finite or not
    summing to 1). Reported so a broken run is visibly broken."""
    broken = 0
    for prediction in predictions:
        if prediction.winner_index is None:
            continue
        probs = (prediction.market_probabilities if use_market
                 else prediction.probabilities)
        if probs is None:
            continue
        if not np.all(np.isfinite(probs)) or abs(float(np.sum(probs)) - 1.0) > 1e-6:
            broken += 1
    return broken


def brier_score(predictions: Sequence[RacePrediction],
                use_market: bool = False) -> float:
    """Multiclass Brier score, averaged over races.

    Less sensitive than log loss to one catastrophic overconfident call,
    so worth reporting alongside it.
    """
    total, count = 0.0, 0
    for prediction in predictions:
        if prediction.winner_index is None:
            continue
        probs = (prediction.market_probabilities if use_market
                 else prediction.probabilities)
        if probs is None:
            continue
        # A non-finite vector is a broken prediction, not a bad one.
        # Substituting zeros would score it 1.0 -- merely "wrong" -- and
        # let a numerically broken model masquerade as a mediocre one.
        if not np.all(np.isfinite(probs)):
            continue
        outcome = np.zeros(len(probs))
        if prediction.winner_index < len(probs):
            outcome[prediction.winner_index] = 1.0
        total += float(np.sum((np.asarray(probs) - outcome) ** 2))
        count += 1
    return total / count if count else float("nan")


def top1_accuracy(predictions: Sequence[RacePrediction],
                  use_market: bool = False) -> float:
    """Share of races where the highest-probability runner won.

    Sanity metric only. See the module docstring before celebrating.
    """
    hits, count = 0, 0
    for prediction in predictions:
        if prediction.winner_index is None:
            continue
        probs = (prediction.market_probabilities if use_market
                 else prediction.probabilities)
        if probs is None or not np.any(np.isfinite(probs)):
            continue
        pick = int(np.nanargmax(probs))
        hits += int(pick == prediction.winner_index)
        count += 1
    return hits / count if count else float("nan")


def uniform_baseline_log_loss(predictions: Sequence[RacePrediction]) -> float:
    """Log loss of guessing 1/field_size for everyone. The floor of
    competence: anything worse than this is actively harmful."""
    total, count = 0.0, 0
    for prediction in predictions:
        n = len(prediction.probabilities)
        if prediction.winner_index is None or n == 0:
            continue
        total += math.log(n)
        count += 1
    return total / count if count else float("nan")


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

@dataclass
class CalibrationBin:
    low: float
    high: float
    count: int
    mean_predicted: float
    observed_rate: float

    @property
    def error(self) -> float:
        return self.mean_predicted - self.observed_rate


def calibration_curve(
    predictions: Sequence[RacePrediction],
    edges: Sequence[float] = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20,
                              0.30, 0.45, 0.65, 1.0),
    use_market: bool = False,
) -> list[CalibrationBin]:
    """Predicted vs realised win rate, bucketed by predicted probability.

    Buckets are deliberately narrow at the bottom: most runners and most
    claimed value sit below 10%, and that is where being wrong costs the
    most.
    """
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(len(edges) - 1)]

    for prediction in predictions:
        if prediction.winner_index is None:
            continue
        probs = (prediction.market_probabilities if use_market
                 else prediction.probabilities)
        if probs is None:
            continue
        for index, p in enumerate(probs):
            if not np.isfinite(p):
                continue
            won = int(index == prediction.winner_index)
            for b in range(len(edges) - 1):
                if edges[b] <= p < edges[b + 1] or (b == len(edges) - 2 and p == 1.0):
                    buckets[b].append((float(p), won))
                    break

    result: list[CalibrationBin] = []
    for b, items in enumerate(buckets):
        if not items:
            continue
        predicted = [p for p, _ in items]
        wins = [w for _, w in items]
        result.append(CalibrationBin(
            low=edges[b], high=edges[b + 1], count=len(items),
            mean_predicted=float(np.mean(predicted)),
            observed_rate=float(np.mean(wins)),
        ))
    return result


def expected_calibration_error(predictions: Sequence[RacePrediction],
                               use_market: bool = False) -> float:
    """Sample-weighted mean absolute gap between predicted and realised."""
    bins = calibration_curve(predictions, use_market=use_market)
    total = sum(b.count for b in bins)
    if not total:
        return float("nan")
    return sum(b.count * abs(b.error) for b in bins) / total


# --------------------------------------------------------------------------
# Betting returns
# --------------------------------------------------------------------------

@dataclass
class YieldResult:
    threshold: float
    bets: int
    staked: float
    returned: float
    wins: int
    # Profit per $1 staked, one entry per bet. Kept so significance can be
    # assessed from the realised distribution rather than a guess at it.
    profits: list[float] = field(default_factory=list)

    @property
    def profit(self) -> float:
        return self.returned - self.staked

    @property
    def yield_pct(self) -> float:
        return self.profit / self.staked if self.staked else 0.0

    @property
    def strike_rate(self) -> float:
        return self.wins / self.bets if self.bets else 0.0

    @property
    def profit_sd(self) -> float:
        """Standard deviation of per-bet profit.

        Win betting has brutal variance: most bets lose $1 and the
        occasional one returns $20, so the sd typically runs 2-5x the mean
        stake. This is the number that decides how long it takes to know
        whether an edge is real.
        """
        if len(self.profits) < 2:
            return float("nan")
        return float(np.std(self.profits, ddof=1))

    @property
    def t_statistic(self) -> float:
        sd = self.profit_sd
        if not np.isfinite(sd) or sd <= 0 or self.bets < 2:
            return 0.0
        return float(np.mean(self.profits) * math.sqrt(self.bets) / sd)

    def bootstrap_interval(self, confidence: float = 0.95,
                           n_resamples: int = 2000,
                           seed: int = 0) -> tuple[float, float]:
        """Percentile bootstrap interval for the yield.

        A t-test is a poor fit here. Per-bet profit is about -1 some 85% of
        the time with an occasional +20, so the distribution is violently
        skewed and the normal approximation flatters small samples. A
        bootstrap makes no distributional assumption.
        """
        if self.bets < 20 or not self.profits:
            return (float("nan"), float("nan"))
        rng = np.random.default_rng(seed)
        sample = np.asarray(self.profits, dtype=float)
        draws = rng.choice(sample, size=(n_resamples, len(sample)), replace=True)
        means = draws.mean(axis=1)
        tail = (1.0 - confidence) / 2.0
        return (float(np.quantile(means, tail)),
                float(np.quantile(means, 1.0 - tail)))

    @property
    def is_significant(self) -> bool:
        """Whether the yield is distinguishable from zero.

        Uses a bootstrap interval that must exclude zero, and demands a
        minimum sample. Read a False as "this sample cannot tell a real
        edge from luck" -- not as "the strategy is bad". Most honest
        backtests never reach True.

        IMPORTANT: this is evaluated at several nested edge thresholds on
        overlapping bets, with no multiplicity correction. Testing five
        thresholds means roughly a 1-in-4 chance that at least one shows
        "significant" on pure noise. Treat a single significant row among
        several as unremarkable, and never as a green light.
        """
        if self.bets < 200:
            return False
        low, high = self.bootstrap_interval()
        if not (np.isfinite(low) and np.isfinite(high)):
            return False
        return low > 0.0 or high < 0.0

    def bets_needed(self, assumed_yield: float = 0.03) -> int:
        """Bets required to detect an edge of `assumed_yield`, at the
        payout variance actually observed.

        The assumed yield is deliberately a *parameter*, not the realised
        one. Deriving the required sample size from the sample's own point
        estimate is circular -- it reduces algebraically to 4n/t^2, which
        is smaller than n exactly when the sample got lucky, so it would
        always tell you that you already have enough evidence at the
        precise moment you do not. The default of 3% is a plausible real
        edge on an exchange; pass your own if you prefer.
        """
        sd = self.profit_sd
        if not np.isfinite(sd) or sd <= 0 or assumed_yield <= 0:
            return 0
        return int(round((2.0 * sd / assumed_yield) ** 2))


def yield_by_threshold(
    predictions: Sequence[RacePrediction],
    thresholds: Sequence[float] = (0.0, 0.02, 0.05, 0.10, 0.20),
    commission: float = 0.0,
    stake: float = 1.0,
) -> list[YieldResult]:
    """Flat-stake return from backing every runner whose edge clears each
    threshold, where edge = p*odds - 1.

    Flat staking is used deliberately: it isolates whether the *model* has
    an edge, without a staking plan flattering or masking it.
    """
    results: list[YieldResult] = []
    for threshold in thresholds:
        bets = wins = 0
        staked = returned = 0.0
        profits: list[float] = []
        for prediction in predictions:
            if prediction.winner_index is None:
                continue
            for index, p in enumerate(prediction.probabilities):
                odds = prediction.odds[index] if index < len(prediction.odds) else np.nan
                if not np.isfinite(p) or not np.isfinite(odds) or odds <= 1.0:
                    continue
                net_odds = (odds - 1.0) * (1.0 - commission)
                edge = p * (1.0 + net_odds) - 1.0
                if edge < threshold:
                    continue
                bets += 1
                staked += stake
                if index == prediction.winner_index:
                    wins += 1
                    returned += stake * (1.0 + net_odds)
                    profits.append(net_odds)
                else:
                    profits.append(-1.0)
        results.append(YieldResult(threshold, bets, staked, returned, wins, profits))
    return results


@dataclass
class BankrollPath:
    """Bankroll trajectory from a staking plan."""

    values: list[float]
    max_drawdown: float
    final: float
    peak: float

    @property
    def total_return_pct(self) -> float:
        if not self.values:
            return 0.0
        return (self.final - self.values[0]) / self.values[0]


def simulate_bankroll(
    predictions: Sequence[RacePrediction],
    starting_bankroll: float = 1000.0,
    kelly_fraction_used: float = 0.25,
    min_edge: float = 0.05,
    max_bet_pct: float = 0.02,
    commission: float = 0.0,
) -> BankrollPath:
    """Walk a staking plan through the predictions in order.

    Reports maximum drawdown, which is the number that actually determines
    whether a strategy is survivable. A plan with a good final return and a
    70% drawdown is one you would have abandoned long before the recovery.
    """
    bankroll = starting_bankroll
    values = [bankroll]
    peak = bankroll
    max_drawdown = 0.0

    for prediction in predictions:
        if prediction.winner_index is None or bankroll <= 0:
            continue
        staked_this_race = 0.0
        payout = 0.0

        for index, p in enumerate(prediction.probabilities):
            odds = prediction.odds[index] if index < len(prediction.odds) else np.nan
            if not np.isfinite(p) or not np.isfinite(odds) or odds <= 1.0:
                continue
            net_odds = (odds - 1.0) * (1.0 - commission)
            edge = p * (1.0 + net_odds) - 1.0
            if edge < min_edge or net_odds <= 0:
                continue
            full_kelly = edge / net_odds
            fraction = min(full_kelly * kelly_fraction_used, max_bet_pct)
            amount = max(0.0, fraction * bankroll)
            if amount <= 0:
                continue
            staked_this_race += amount
            if index == prediction.winner_index:
                payout += amount * (1.0 + net_odds)

        bankroll += payout - staked_this_race
        values.append(bankroll)
        peak = max(peak, bankroll)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - bankroll) / peak)

    return BankrollPath(values=values, max_drawdown=max_drawdown,
                        final=bankroll, peak=peak)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def comparable_subset(predictions: Sequence[RacePrediction]
                      ) -> list[RacePrediction]:
    """Races where BOTH a model and a market probability exist.

    Comparing a model averaged over every race against a market averaged
    over only the priced ones is not a comparison -- unpriced races are
    typically small, odd fields, and whichever side is scored on them is
    being judged on a different problem. Every head-to-head figure in the
    report is computed on this intersection.
    """
    return [
        p for p in predictions
        if p.winner_index is not None
        and p.market_probabilities is not None
        and np.all(np.isfinite(p.market_probabilities))
    ]


def full_report(predictions: Sequence[RacePrediction],
                commission: float = 0.0) -> dict:
    """Everything, with the market baseline alongside every model figure."""
    shared = comparable_subset(predictions)
    has_market = bool(shared)

    report = {
        "races": len([p for p in predictions if p.winner_index is not None]),
        "model": {
            "log_loss": log_loss(predictions),
            "brier": brier_score(predictions),
            "top1_accuracy": top1_accuracy(predictions),
            "calibration_error": expected_calibration_error(predictions),
        },
        "baselines": {
            "uniform_log_loss": uniform_baseline_log_loss(predictions),
        },
        "broken_predictions": count_broken(predictions),
        "yield": [
            {
                "edge_threshold": r.threshold,
                "bets": r.bets,
                "strike_rate": r.strike_rate,
                "yield_pct": r.yield_pct,
                "profit": r.profit,
                "profit_sd": r.profit_sd,
                "yield_ci_95": r.bootstrap_interval(),
                "statistically_significant": r.is_significant,
                "bets_needed_to_detect_3pct_edge": r.bets_needed(0.03),
            }
            for r in yield_by_threshold(predictions, commission=commission)
        ],
        "calibration": [
            {
                "band": f"{b.low:.0%}-{b.high:.0%}",
                "count": b.count,
                "predicted": b.mean_predicted,
                "observed": b.observed_rate,
            }
            for b in calibration_curve(predictions)
        ],
    }

    if has_market:
        # Head-to-head figures are computed on the shared subset only, so
        # the two sides are scored on exactly the same races.
        market_ll = log_loss(shared, use_market=True)
        model_ll_shared = log_loss(shared)

        report["baselines"].update({
            "market_log_loss": market_ll,
            "market_brier": brier_score(shared, use_market=True),
            "market_top1_accuracy": top1_accuracy(shared, use_market=True),
            "market_calibration_error": expected_calibration_error(
                shared, use_market=True),
        })
        report["model"].update({
            "log_loss_on_priced_races": model_ll_shared,
            "top1_accuracy_on_priced_races": top1_accuracy(shared),
        })
        report["comparable_races"] = len(shared)

        if np.isfinite(market_ll) and market_ll > 0 and np.isfinite(model_ll_shared):
            report["model"]["log_loss_improvement_vs_market_pct"] = (
                100.0 * (1.0 - model_ll_shared / market_ll))

    return report
