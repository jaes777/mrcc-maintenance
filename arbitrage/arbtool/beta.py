"""Independent beta-test agents that attack a strategy before it is trusted.

A backtest is the strategy grading its own homework. These agents are the second
opinion: each one takes a strategy that already passed the walk-forward gate and
tries to break it. A strategy that survives them is not guaranteed to work — but
one that fails any of them is definitely not ready.

Two kinds of reviewer:

* Deterministic adversaries (below) — reproducible, run offline, no cost.
* An optional Claude reviewer (:func:`llm_review`) — reads the numbers and
  argues with them in plain English. Requires the ``anthropic`` package and an
  API key; the suite runs fine without it.
"""

from __future__ import annotations

import copy
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .backtest import BacktestContext, WalkForwardResult, simulate
from .metrics import compute_metrics, equity_from_trades, tail_summary
from .models import ModelSpec, score
from .pairs import PairSeries
from .stats import binomial_tail_pvalue, stdev


@dataclass
class BetaFinding:
    """One adversarial test and what it concluded."""

    name: str
    passed: bool
    severity: str          # info | warning | critical
    headline: str
    detail: str
    numbers: Dict[str, float] = field(default_factory=dict)

    def line(self) -> str:
        mark = "PASS" if self.passed else ("WARN" if self.severity == "warning" else "FAIL")
        return f"[{mark}] {self.name}: {self.headline}"


@dataclass
class BetaReport:
    findings: List[BetaFinding]
    reviewer_notes: Optional[str] = None

    @property
    def passed(self) -> bool:
        return all(f.passed or f.severity == "info" for f in self.findings)

    @property
    def critical_failures(self) -> List[BetaFinding]:
        return [f for f in self.findings if not f.passed and f.severity == "critical"]

    def summary(self) -> str:
        lines = [f.line() for f in self.findings]
        verdict = "BETA TESTS PASSED" if self.passed else "BETA TESTS FAILED"
        lines.append(f"--- {verdict} ---")
        if self.reviewer_notes:
            lines.append("")
            lines.append("Independent AI reviewer:")
            lines.append(self.reviewer_notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Individual adversaries
# --------------------------------------------------------------------------


def audit_lookahead(series: PairSeries, spec: ModelSpec) -> BetaFinding:
    """Prove the model cannot see the future.

    Recomputes each score using only the data available at the time and checks
    it matches the score produced when the model was handed the whole history.
    Any mismatch means the backtest was trading on information that did not
    exist yet — the single most common way a strategy looks brilliant on paper
    and loses money live.
    """
    full = score(series.log_spread, spec)
    checkpoints = [i for i in range(len(series) - 1, max(spec.lookback, 20), -37)][:12]
    mismatches = 0
    worst = 0.0
    for i in checkpoints:
        truncated = score(series.log_spread[: i + 1], spec)
        a, b = full[i], truncated[i]
        if a is None and b is None:
            continue
        if a is None or b is None:
            mismatches += 1
            continue
        diff = abs(a - b)
        worst = max(worst, diff)
        if diff > 1e-9:
            mismatches += 1

    passed = mismatches == 0
    return BetaFinding(
        name="look-ahead audit",
        passed=passed,
        severity="critical",
        headline=("no future information is used"
                  if passed else
                  f"{mismatches} of {len(checkpoints)} scores changed when future "
                  f"data was hidden — the backtest is invalid"),
        detail=("Each signal was recomputed using only the bars that existed at "
                "the time and compared against the original. Identical results "
                "mean the strategy is causal."),
        numbers={"checkpoints": len(checkpoints), "mismatches": mismatches,
                 "largest_difference": worst},
    )


def stress_costs(series: PairSeries, spec: ModelSpec, ctx: BacktestContext,
                 start: int, end: int, multiplier: float = 2.0) -> BetaFinding:
    """Double every trading cost and see whether the edge survives."""
    base_trades = simulate(series, spec, ctx, start, end)
    stressed_ctx = copy.copy(ctx)
    stressed_ctx.cost_model = ctx.cost_model.stressed(multiplier, multiplier)
    stressed_trades = simulate(series, spec, stressed_ctx, start, end)

    base = compute_metrics(base_trades)
    stressed = compute_metrics(stressed_trades)
    passed = stressed.trades > 0 and stressed.expectancy_bps > 0

    return BetaFinding(
        name="cost shock",
        passed=passed,
        severity="critical",
        headline=(f"still profitable with costs {multiplier:.0f}x higher "
                  f"({stressed.expectancy_bps:+.1f} bp per trade)"
                  if passed else
                  f"loses money once costs are {multiplier:.0f}x higher "
                  f"({stressed.expectancy_bps:+.1f} bp per trade) — the 'edge' is "
                  f"just an optimistic fee assumption"),
        detail=("Real commissions, spreads and borrow rates are worse than "
                "modelled more often than they are better. A strategy that only "
                "works at the modelled cost has no margin for reality."),
        numbers={"base_expectancy_bps": base.expectancy_bps,
                 "stressed_expectancy_bps": stressed.expectancy_bps,
                 "base_win_rate": base.win_rate, "stressed_win_rate": stressed.win_rate},
    )


def stress_execution_lag(series: PairSeries, spec: ModelSpec, ctx: BacktestContext,
                         start: int, end: int, lag: int = 3) -> BetaFinding:
    """Delay every fill and see how much of the edge was pure speed."""
    base = compute_metrics(simulate(series, spec, ctx, start, end))
    slow_ctx = copy.copy(ctx)
    slow_ctx.execution_lag_bars = lag
    slow = compute_metrics(simulate(series, spec, slow_ctx, start, end))

    decay = (base.expectancy_bps - slow.expectancy_bps)
    passed = slow.trades > 0 and slow.expectancy_bps > 0

    return BetaFinding(
        name="execution delay",
        passed=passed,
        severity="warning",
        headline=(f"survives a {lag}-day delay before each trade is filled"
                  if passed else
                  f"needs same-day execution — a {lag}-day delay turns it into a "
                  f"loser ({slow.expectancy_bps:+.1f} bp)"),
        detail=("Cross-listed pairs on non-overlapping exchanges cannot be traded "
                "simultaneously. A strategy whose edge disappears with a delay is "
                "competing on speed, and a retail account will always lose that race."),
        numbers={"base_expectancy_bps": base.expectancy_bps,
                 "delayed_expectancy_bps": slow.expectancy_bps,
                 "decay_bps": decay},
    )


def stress_data_gaps(series: PairSeries, spec: ModelSpec, ctx: BacktestContext,
                     start: int, end: int, drop_fraction: float = 0.05,
                     seed: int = 11, runs: int = 5) -> BetaFinding:
    """Randomly delete bars — real feeds have gaps, halts and bad prints."""
    outcomes: List[float] = []
    win_rates: List[float] = []
    for run in range(runs):
        rng = random.Random(seed + run)
        keep = [i for i in range(len(series)) if rng.random() > drop_fraction]
        if len(keep) < 100:
            continue
        gapped = PairSeries(
            series.pair,
            [series.dates[i] for i in keep],
            [series.a_local[i] for i in keep],
            [series.b_local[i] for i in keep],
            [series.fx_a[i] for i in keep],
            [series.fx_b[i] for i in keep],
            [series.a_base[i] for i in keep],
            [series.b_base[i] for i in keep],
            [series.log_spread[i] for i in keep],
            series.base_currency,
        )
        scaled_start = int(start * len(keep) / max(1, len(series)))
        scaled_end = int(end * len(keep) / max(1, len(series)))
        m = compute_metrics(simulate(gapped, spec, ctx, scaled_start, scaled_end))
        if m.trades:
            outcomes.append(m.expectancy_bps)
            win_rates.append(m.win_rate)

    if not outcomes:
        return BetaFinding("data gaps", False, "warning",
                           "no trades survived when bars were dropped",
                           "The strategy depends on a perfectly complete data feed.",
                           {})
    worst = min(outcomes)
    spread_bps = max(outcomes) - worst
    passed = worst > 0
    return BetaFinding(
        name="data gaps",
        passed=passed,
        severity="warning",
        headline=(f"tolerates {drop_fraction:.0%} missing bars "
                  f"(worst run {worst:+.1f} bp)"
                  if passed else
                  f"breaks when {drop_fraction:.0%} of bars go missing "
                  f"(worst run {worst:+.1f} bp)"),
        detail=("Live feeds drop bars: halts, holidays, vendor outages. A result "
                "that only holds on flawless data will not repeat."),
        numbers={"worst_expectancy_bps": worst, "spread_bps": spread_bps,
                 "runs": len(outcomes)},
    )


def null_hypothesis_test(series: PairSeries, spec: ModelSpec, ctx: BacktestContext,
                         start: int, end: int, seed: int = 23, runs: int = 8) -> BetaFinding:
    """Run the strategy on markets with the mean reversion removed.

    The gap is replaced by a random walk of identical volatility, so the data
    looks the same but contains no convergence to trade. A strategy that still
    appears to win here is not detecting anything real — it is an artifact of
    the rules themselves, and the original result means nothing.
    """
    # Scale the walk so it wanders about as far overall as the real spread ever
    # does. Using the real spread's *daily* volatility would send a random walk
    # to absurd levels over years of data and produce meaningless prices.
    n = len(series)
    total_sd = stdev(series.log_spread) or 0.005
    vol = total_sd / math.sqrt(max(2, n))
    fake_wins = 0
    fake_trades = 0
    expectancies: List[float] = []

    for run in range(runs):
        rng = random.Random(seed + run * 7919)
        walk = [series.log_spread[0]]
        for _ in range(n - 1):
            walk.append(walk[-1] + rng.gauss(0.0, vol))
        fake = PairSeries(
            series.pair, series.dates, series.a_local, series.b_local,
            series.fx_a, series.fx_b,
            [b * math.exp(s) for b, s in zip(series.b_base, walk)],
            series.b_base, walk, series.base_currency,
        )
        trades = simulate(fake, spec, ctx, start, end)
        m = compute_metrics(trades)
        fake_wins += m.wins
        fake_trades += m.trades
        if m.trades:
            expectancies.append(m.expectancy_bps)

    fake_win_rate = fake_wins / fake_trades if fake_trades else 0.0
    mean_expectancy = sum(expectancies) / len(expectancies) if expectancies else 0.0
    # Profit is the test, not win rate: these rules win more than half the time
    # even on pure noise, because they take many small wins and few large losses.
    passed = mean_expectancy <= 0.0

    return BetaFinding(
        name="null-market test",
        passed=passed,
        severity="critical",
        headline=(f"loses money ({mean_expectancy:+.1f} bp per trade) on markets "
                  f"with the mean reversion removed — good, the real edge is real"
                  if passed else
                  f"still makes {mean_expectancy:+.1f} bp per trade on markets that "
                  f"contain no opportunity — the result is an artifact of the rules"),
        detail=("The gap was replaced with a random walk that wanders as far as the "
                "real one but never pulls back. There is nothing to exploit in that "
                "data, so the strategy should lose the cost of trading. Note the "
                f"null win rate of {fake_win_rate:.0%}: these rules win most of the "
                "time even on pure noise, which is the clearest possible "
                "demonstration that win rate alone proves nothing."),
        numbers={"null_win_rate": fake_win_rate, "null_trades": fake_trades,
                 "null_expectancy_bps": mean_expectancy},
    )


def parameter_sensitivity(series: PairSeries, base_spec: ModelSpec,
                          ctx: BacktestContext, start: int, end: int) -> BetaFinding:
    """Nudge the settings. A real edge is a plateau, not a needle.

    If performance collapses when the entry threshold moves by a quarter of a
    standard deviation, the chosen settings were fitted to noise.
    """
    results: List[float] = []
    for d_entry in (-0.25, 0.0, 0.25):
        for d_lookback in (-15, 0, 15):
            spec = ModelSpec(
                name=base_spec.name,
                lookback=max(15, base_spec.lookback + d_lookback),
                entry_z=max(0.5, base_spec.entry_z + d_entry),
                exit_z=base_spec.exit_z,
                stop_z=base_spec.stop_z,
                max_hold_days=base_spec.max_hold_days,
                extra=dict(base_spec.extra),
            )
            m = compute_metrics(simulate(series, spec, ctx, start, end))
            if m.trades:
                results.append(m.expectancy_bps)

    if not results:
        return BetaFinding("parameter sensitivity", False, "warning",
                           "no nearby settings produce any trades",
                           "The chosen settings sit on an isolated spike.", {})

    positive = sum(1 for r in results if r > 0)
    fraction = positive / len(results)
    passed = fraction >= 0.6
    return BetaFinding(
        name="parameter sensitivity",
        passed=passed,
        severity="warning",
        headline=(f"{positive}/{len(results)} nearby settings also profitable — "
                  f"a stable plateau"
                  if passed else
                  f"only {positive}/{len(results)} nearby settings are profitable — "
                  f"the chosen numbers were fitted to noise"),
        detail=("Entry threshold and lookback were nudged around the chosen values. "
                "A genuine effect does not vanish when a parameter moves slightly."),
        numbers={"profitable_variants": positive, "variants_tested": len(results),
                 "fraction": fraction,
                 "worst_bps": min(results), "best_bps": max(results)},
    )


def worst_period_stress(wf: WalkForwardResult, min_trades: int = 5) -> BetaFinding:
    """Report the single worst out-of-sample window, not the average.

    Averages hide the period that would have made a real person stop trading.
    Windows with only a handful of trades are ignored: a 0% win rate over one
    trade is noise, and treating it as a finding would be theatre.
    """
    by_fold: Dict[int, List] = {}
    for trade in wf.oos_trades:
        by_fold.setdefault(trade.fold, []).append(trade)
    by_fold = {k: v for k, v in by_fold.items() if len(v) >= min_trades}
    if not by_fold:
        return BetaFinding(
            "worst period", True, "info",
            f"no single window had {min_trades}+ trades, so no window is judged",
            "Individual windows were too thin to say anything about.", {})

    worst_fold, worst_metrics = None, None
    for fold, trades in by_fold.items():
        m = compute_metrics(trades, equity_from_trades(trades, 100_000.0))
        if worst_metrics is None or m.net_pnl < worst_metrics.net_pnl:
            worst_fold, worst_metrics = fold, m

    passed = worst_metrics.win_rate >= 0.40 and worst_metrics.max_drawdown_pct < 25
    return BetaFinding(
        name="worst period",
        passed=passed,
        severity="warning",
        headline=(f"worst window still respectable: {worst_metrics.win_rate:.0%} win "
                  f"rate over {worst_metrics.trades} trades, "
                  f"{worst_metrics.max_drawdown_pct:.1f}% drawdown"
                  if passed else
                  f"worst window was ugly: {worst_metrics.win_rate:.0%} win rate over "
                  f"{worst_metrics.trades} trades, "
                  f"{worst_metrics.max_drawdown_pct:.1f}% drawdown — expect to live "
                  f"through this again"),
        detail=("Long-run averages are not what you experience. This is the single "
                f"worst out-of-sample stretch with at least {min_trades} trades in it."),
        numbers={"fold": float(worst_fold), "win_rate": worst_metrics.win_rate,
                 "net_pnl": worst_metrics.net_pnl,
                 "max_drawdown_pct": worst_metrics.max_drawdown_pct},
    )


def multiple_testing_check(wf: WalkForwardResult, configs_tried: int) -> BetaFinding:
    """Discount the result for how many settings were tried before it was found.

    Test enough combinations and one will look excellent by chance alone. The
    p-value is adjusted for the number of attempts so the claim is honest.
    """
    m = wf.oos_metrics
    raw_p = binomial_tail_pvalue(m.wins, m.trades, 0.5)
    # The family being corrected for is the parameter search: every setting that
    # was tried before this one was chosen. The walk-forward folds are not extra
    # independent searches — they are the test, so they are not counted again.
    effective = max(1, configs_tried)
    # Sidak correction: probability that no attempt would have looked this good.
    adjusted = 1.0 - (1.0 - raw_p) ** effective if raw_p < 1 else 1.0
    passed = adjusted < 0.05

    return BetaFinding(
        name="multiple-testing check",
        passed=passed,
        severity="critical",
        headline=(f"result survives being discounted for {effective} attempts "
                  f"(adjusted p={adjusted:.4f})"
                  if passed else
                  f"after accounting for {effective} settings tried, this result is "
                  f"explainable by luck (adjusted p={adjusted:.3f})"),
        detail=("Searching many parameter combinations guarantees that some will "
                "look good by chance, so the p-value is corrected for the size of "
                "that search. The correction assumes every setting was an "
                "independent attempt, which they are not — they are variations on "
                "each other — so this is deliberately the harshest reading. If it "
                "fails, the fix is more out-of-sample data or a smaller search, "
                "not a looser test."),
        numbers={"raw_p": raw_p, "adjusted_p": adjusted,
                 "attempts": float(effective), "oos_trades": float(m.trades)},
    )


def tail_risk_check(wf: WalkForwardResult) -> BetaFinding:
    """Compare the worst losses against the typical win.

    Mean-reversion strategies win often and lose rarely but violently. A high
    win rate paired with a fat left tail is the classic shape of a strategy that
    makes money for years and then gives it all back at once.
    """
    tail = tail_summary(wf.oos_trades)
    if not tail:
        return BetaFinding("tail risk", False, "warning", "no trades to analyse",
                           "Nothing to measure.", {})
    ratio = tail["loss_tail_vs_avg_win"]
    passed = ratio < 6.0
    return BetaFinding(
        name="tail risk",
        passed=passed,
        severity="warning",
        headline=(f"worst 1% of trades lose {ratio:.1f}x the average win — "
                  f"survivable" if passed else
                  f"worst 1% of trades lose {ratio:.1f}x the average win — a high "
                  f"win rate here is hiding a rare, very large loss"),
        detail=("Win rate counts how often you win. This counts how much you lose "
                "when you are wrong. The second number is what closes accounts."),
        numbers=tail,
    )


# --------------------------------------------------------------------------
# Suite
# --------------------------------------------------------------------------


def run_beta_suite(series: PairSeries, wf: WalkForwardResult, ctx: BacktestContext,
                   configs_tried: int, oos_start: int, oos_end: int) -> BetaReport:
    """Run every adversary against a strategy that passed the walk-forward."""
    chosen = next((f.chosen for f in reversed(wf.folds) if f.chosen), None)
    findings: List[BetaFinding] = []

    if chosen is None:
        findings.append(BetaFinding(
            "strategy present", False, "critical", "no model was ever selected",
            "The walk-forward never found a workable configuration.", {}))
        return BetaReport(findings)

    findings.append(audit_lookahead(series, chosen))
    findings.append(stress_costs(series, chosen, ctx, oos_start, oos_end))
    findings.append(stress_execution_lag(series, chosen, ctx, oos_start, oos_end))
    findings.append(stress_data_gaps(series, chosen, ctx, oos_start, oos_end))
    findings.append(null_hypothesis_test(series, chosen, ctx, oos_start, oos_end))
    findings.append(parameter_sensitivity(series, chosen, ctx, oos_start, oos_end))
    findings.append(worst_period_stress(wf))
    findings.append(multiple_testing_check(wf, configs_tried))
    findings.append(tail_risk_check(wf))
    return BetaReport(findings)


# --------------------------------------------------------------------------
# Optional independent AI reviewer
# --------------------------------------------------------------------------

REVIEWER_SYSTEM = """You are an independent, sceptical quantitative reviewer.

You are shown the results of a cross-listing statistical-arbitrage strategy that
has already passed its own automated tests. Your job is to find the reasons it
might still fail with real money. Assume the author is fooling themselves until
the numbers prove otherwise.

Be specific and quantitative. Refer to the actual figures given. Do not praise.
Prioritise: survivorship and data quality, whether costs are plausibly modelled,
whether the sample is large enough to support the win-rate claim, capacity and
liquidity, what happens in a crisis, and whether the win rate is masking a fat
left tail.

Finish with a single line: VERDICT: DEPLOY-TO-PAPER or VERDICT: DO-NOT-DEPLOY,
followed by one sentence of justification. Keep the whole reply under 400 words
and write it so a non-specialist can follow it."""


def llm_review(payload: Dict, model: str = "claude-opus-5",
               timeout: float = 120.0) -> Optional[str]:
    """Ask Claude to critique the strategy report.

    Optional. Returns ``None`` (with a printable reason) when the ``anthropic``
    package or an API key is missing, so the beta suite never depends on it.
    """
    try:
        import anthropic  # type: ignore
    except ImportError:
        return None

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None

    import json

    client = anthropic.Anthropic(timeout=timeout)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            system=REVIEWER_SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            messages=[{
                "role": "user",
                "content": (
                    "Review this cross-listing arbitrage strategy report and tell me "
                    "why it might fail with real money:\n\n"
                    + json.dumps(payload, indent=2, default=str)
                ),
            }],
        )
    except Exception as exc:  # network, auth, rate limit — never fatal
        return f"(independent reviewer unavailable: {exc})"

    if response.stop_reason == "refusal":
        return "(independent reviewer declined to answer this request)"

    return "\n".join(b.text for b in response.content if b.type == "text").strip() or None


def reviewer_availability() -> str:
    """Plain-English explanation of whether the AI reviewer can run."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return ("The optional AI reviewer is not installed. To enable it: "
                "pip install anthropic, then set ANTHROPIC_API_KEY.")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return ("The optional AI reviewer is installed but has no API key. "
                "Set ANTHROPIC_API_KEY to enable it.")
    return "The optional AI reviewer is available and will be asked for a second opinion."
