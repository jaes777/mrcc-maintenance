"""The agents, and the orchestrator that runs them.

Each agent owns one job and hands its output to the next over a shared message
bus. Every message is recorded, so any conclusion can be traced back through the
chain that produced it — which matters more here than in most systems, because
the output is a recommendation to risk money.

    Scraper  ->  Normaliser  ->  PairSelector  ->  Analyst
                                                     |
                             Risk  <-  Backtester ---+
                              |
                          Evaluator  ->  BetaTester  ->  Execution

The Evaluator is the gatekeeper: nothing reaches Execution without passing the
win-rate, profit-factor and drawdown gate, and then surviving the beta agents.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .backtest import BacktestContext, WalkForwardResult, walk_forward
from .beta import BetaReport, llm_review, run_beta_suite
from .config import Config
from .costs import CostModel
from .data.providers import CachingProvider, DataError, build_provider, today_iso
from .metrics import Metrics, compute_metrics
from .models import ModelSpec, grid, score
from .pairs import (PairQuality, PairSeries, assess_pair, build_pair_series,
                    ratio_report)
from .portfolio import (PaperBroker, PortfolioResult, PortfolioRules, Trade,
                        combine_portfolio, orders_for_signal)
from .universe import PairSpec, selected_pairs, synthetic_universe


# --------------------------------------------------------------------------
# Message bus
# --------------------------------------------------------------------------


@dataclass
class Message:
    sender: str
    topic: str
    summary: str
    payload: Dict[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def line(self) -> str:
        return f"{self.at}  {self.sender:<14} {self.topic:<18} {self.summary}"


class MessageBus:
    """Records everything the agents say to each other."""

    def __init__(self, echo: Optional[Callable[[str], None]] = None):
        self.messages: List[Message] = []
        self.echo = echo

    def publish(self, sender: str, topic: str, summary: str, **payload) -> Message:
        message = Message(sender, topic, summary, payload)
        self.messages.append(message)
        if self.echo:
            self.echo(message.line())
        return message

    def by_topic(self, topic: str) -> List[Message]:
        return [m for m in self.messages if m.topic == topic]

    def transcript(self) -> str:
        return "\n".join(m.line() for m in self.messages)


class Agent:
    """Base class. Every agent has a name and talks through the bus."""

    name = "agent"

    def __init__(self, bus: MessageBus, config: Config):
        self.bus = bus
        self.config = config

    def say(self, topic: str, summary: str, **payload) -> None:
        self.bus.publish(self.name, topic, summary, **payload)


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------


class ScraperAgent(Agent):
    """Collects raw prices and exchange rates from every configured source."""

    name = "scraper"

    def __init__(self, bus: MessageBus, config: Config, provider: CachingProvider):
        super().__init__(bus, config)
        self.provider = provider

    def collect(self, pairs: List[PairSpec]) -> Dict[str, PairSeries]:
        end = self.config.end or today_iso()
        out: Dict[str, PairSeries] = {}
        for pair in pairs:
            try:
                series = build_pair_series(pair, self.provider, self.config.start, end,
                                           self.config.base_currency)
            except DataError as exc:
                self.say("data.missing", f"{pair.pair_id}: {exc}", pair=pair.pair_id)
                continue
            out[pair.pair_id] = series
            self.say(
                "data.collected",
                f"{pair.pair_id}: {len(series)} usable days "
                f"({series.dates[0]} to {series.dates[-1]})",
                pair=pair.pair_id, bars=len(series),
            )
        if not out:
            self.say("data.empty", "no pair produced usable data")
        return out


class NormalizerAgent(Agent):
    """Verifies that the two listings really are the same asset.

    Prices are already converted to one currency per ordinary share upstream;
    this agent's job is to catch the case where the configured share ratio is
    wrong, which would otherwise present a bookkeeping error as a permanent
    100% arbitrage.
    """

    name = "normaliser"

    def verify(self, series_map: Dict[str, PairSeries]) -> Dict[str, PairSeries]:
        clean: Dict[str, PairSeries] = {}
        for pair_id, series in series_map.items():
            configured, empirical, verdict = ratio_report(series)
            if verdict.startswith("WRONG"):
                self.say("ratio.rejected", f"{pair_id}: share ratio {verdict}",
                         pair=pair_id, configured=configured, empirical=empirical)
                continue
            clean[pair_id] = series
            self.say("ratio.verified",
                     f"{pair_id}: share ratio {configured:.4g} {verdict}",
                     pair=pair_id, configured=configured, empirical=empirical)
        return clean


class PairSelectorAgent(Agent):
    """Decides which markets are worth trading between, and says why not."""

    name = "pair-selector"

    def __init__(self, bus: MessageBus, config: Config, cost_model: CostModel):
        super().__init__(bus, config)
        self.cost_model = cost_model

    def select(self, series_map: Dict[str, PairSeries]) -> Tuple[List[str], List[PairQuality]]:
        assessments: List[PairQuality] = []
        for pair_id, series in series_map.items():
            quality = assess_pair(series, self.cost_model, self.config.signal,
                                  self.config.base_currency, self.config.risk.allow_short)
            assessments.append(quality)
            topic = "pair.accepted" if quality.tradeable else "pair.rejected"
            self.say(topic, quality.summary(), pair=pair_id,
                     opportunity=quality.opportunity_ratio)

        assessments.sort(key=lambda q: (-q.tradeable, -q.opportunity_ratio))
        chosen = [q.pair_id for q in assessments if q.tradeable]
        self.say("pair.shortlist",
                 f"{len(chosen)} of {len(assessments)} pairs are worth trading: "
                 + (", ".join(chosen) if chosen else "none"),
                 pairs=chosen)
        return chosen, assessments


class AnalystAgent(Agent):
    """Fits candidate models on old data and scores them on unseen data."""

    name = "analyst"

    def __init__(self, bus: MessageBus, config: Config, ctx: BacktestContext):
        super().__init__(bus, config)
        self.ctx = ctx

    def build_grid(self) -> List[ModelSpec]:
        return grid(self.config.search_models, self.config.search_lookback,
                    self.config.search_entry_z, self.config.search_exit_z,
                    self.config.signal.stop_z, self.config.signal.max_hold_days)

    def analyse(self, series_map: Dict[str, PairSeries],
                pair_ids: List[str]) -> Dict[str, WalkForwardResult]:
        specs = self.build_grid()
        self.say("model.grid", f"testing {len(specs)} model configurations per pair",
                 configurations=len(specs))

        results: Dict[str, WalkForwardResult] = {}
        wf = self.config.walk_forward
        for pair_id in pair_ids:
            series = series_map[pair_id]
            result = walk_forward(
                series, self.ctx, specs, wf.train_days, wf.test_days, wf.step_days,
                wf.purge_days, wf.min_folds, capital=self.config.risk.capital,
                confidence=self.config.gate.confidence,
            )
            results[pair_id] = result
            if not result.folds:
                self.say("model.insufficient",
                         f"{pair_id}: not enough history for even one out-of-sample "
                         f"window", pair=pair_id)
                continue
            for fold in result.folds:
                self.say("model.fold", f"{pair_id} {fold.summary()}", pair=pair_id)
            self.say("model.result",
                     f"{pair_id} unseen data: {result.oos_metrics.headline()}",
                     pair=pair_id, win_rate=result.oos_metrics.win_rate,
                     trades=result.oos_metrics.trades)
            if result.degradation_pct > 15:
                self.say("model.warning",
                         f"{pair_id}: win rate fell {result.degradation_pct:.0f} "
                         f"points from fitted to unseen data — a sign of overfitting",
                         pair=pair_id, degradation=result.degradation_pct)
        return results


class RiskAgent(Agent):
    """Applies position limits, exposure caps and the daily loss halt."""

    name = "risk"

    def rules(self) -> PortfolioRules:
        r = self.config.risk
        notional = r.capital * r.notional_per_leg_pct / 100.0
        return PortfolioRules(
            capital=r.capital,
            notional_per_leg=notional,
            max_concurrent=r.max_concurrent_positions,
            max_gross_exposure=r.capital * r.max_gross_exposure_pct / 100.0,
            daily_loss_halt=r.capital * r.daily_loss_halt_pct / 100.0,
            max_loss_per_trade=r.capital * r.max_loss_per_trade_pct / 100.0,
        )

    def combine(self, results: Dict[str, WalkForwardResult]) -> PortfolioResult:
        all_trades: List[Trade] = []
        for result in results.values():
            all_trades.extend(result.oos_trades)
        rules = self.rules()
        portfolio = combine_portfolio(all_trades, rules)
        self.say("risk.applied",
                 f"{len(portfolio.trades)} trades fit inside the account's limits; "
                 f"{len(portfolio.skipped)} were skipped",
                 accepted=len(portfolio.trades), skipped=len(portfolio.skipped))
        if portfolio.halted_days:
            self.say("risk.halt",
                     f"the daily loss limit stopped trading on "
                     f"{len(portfolio.halted_days)} day(s)",
                     days=portfolio.halted_days)
        return portfolio


@dataclass
class GateResult:
    passed: bool
    metrics: Metrics
    reasons: List[str]
    checks: Dict[str, Tuple[bool, str]]

    def report(self) -> str:
        lines = ["Promotion gate:"]
        for name, (ok, detail) in self.checks.items():
            lines.append(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
        lines.append(f"  => {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)


class EvaluatorAgent(Agent):
    """The gatekeeper. Decides whether a strategy may trade at all.

    The user's requirement was a minimum 60% win rate. Nobody can promise that
    in advance, so it is enforced here instead: the *lower bound* of a 95%
    confidence interval on out-of-sample trades must clear the threshold, on a
    sample large enough to mean something. Win rate alone is not sufficient —
    profit factor, expectancy and drawdown are checked too, because a strategy
    can win 80% of the time and still lose money.
    """

    name = "evaluator"

    def evaluate(self, portfolio: PortfolioResult) -> GateResult:
        gate = self.config.gate
        metrics = compute_metrics(portfolio.trades, portfolio.equity, gate.confidence,
                                  portfolio.daily_returns)
        checks: Dict[str, Tuple[bool, str]] = {}

        ok_n = metrics.trades >= gate.min_oos_trades
        checks["enough unseen trades"] = (
            ok_n, f"{metrics.trades} trades (need {gate.min_oos_trades})")

        ok_wr = metrics.win_rate_lower >= gate.min_win_rate
        checks[f"win rate at least {gate.min_win_rate:.0%}"] = (
            ok_wr,
            f"measured {metrics.win_rate:.1%}, but the honest lower bound is "
            f"{metrics.win_rate_lower:.1%} (need {gate.min_win_rate:.0%})")

        ok_pf = metrics.profit_factor >= gate.min_profit_factor
        checks["profit factor"] = (
            ok_pf, f"{metrics.profit_factor:.2f} (need {gate.min_profit_factor:.2f})")

        ok_exp = metrics.expectancy_bps >= gate.min_expectancy_bps
        checks["profit per trade"] = (
            ok_exp,
            f"{metrics.expectancy_bps:+.1f} basis points "
            f"(need {gate.min_expectancy_bps:+.1f})")

        ok_dd = metrics.max_drawdown_pct <= gate.max_drawdown_pct
        checks["worst drawdown"] = (
            ok_dd,
            f"{metrics.max_drawdown_pct:.1f}% (limit {gate.max_drawdown_pct:.1f}%)")

        ok_luck = metrics.luck_pvalue < 0.05 if metrics.trades else False
        checks["not explainable by luck"] = (
            ok_luck, f"p = {metrics.luck_pvalue:.4f} (need below 0.05)")

        reasons = [f"{name}: {detail}" for name, (ok, detail) in checks.items() if not ok]
        passed = not reasons

        self.say("gate.result",
                 ("strategy PASSED the promotion gate"
                  if passed else
                  f"strategy REJECTED — {len(reasons)} check(s) failed"),
                 passed=passed, win_rate=metrics.win_rate,
                 win_rate_lower=metrics.win_rate_lower)
        for reason in reasons:
            self.say("gate.reason", reason)
        return GateResult(passed, metrics, reasons, checks)


class BetaTestAgent(Agent):
    """Runs the adversarial reviewers against anything that passed the gate."""

    name = "beta-tester"

    def __init__(self, bus: MessageBus, config: Config, ctx: BacktestContext,
                 use_llm_reviewer: bool = False):
        super().__init__(bus, config)
        self.ctx = ctx
        self.use_llm_reviewer = use_llm_reviewer

    def test(self, series_map: Dict[str, PairSeries],
             results: Dict[str, WalkForwardResult],
             configs_tried: int) -> Dict[str, BetaReport]:
        reports: Dict[str, BetaReport] = {}
        for pair_id, wf in results.items():
            if not wf.oos_trades:
                continue
            series = series_map[pair_id]
            first_test = min((f.test_start for f in wf.folds if f.chosen), default=None)
            if first_test is None:
                continue
            start = series.index_of(first_test) or 0
            report = run_beta_suite(series, wf, self.ctx, configs_tried, start, len(series))
            reports[pair_id] = report
            for finding in report.findings:
                self.say("beta.finding", f"{pair_id} {finding.line()}",
                         pair=pair_id, passed=finding.passed, test=finding.name)
            self.say("beta.result",
                     f"{pair_id}: {'passed' if report.passed else 'FAILED'} the "
                     f"adversarial review", pair=pair_id, passed=report.passed)
        return reports

    def independent_review(self, payload: Dict) -> Optional[str]:
        if not self.use_llm_reviewer:
            return None
        self.say("beta.reviewer", "asking an independent AI reviewer for a second opinion")
        notes = llm_review(payload)
        if notes:
            self.say("beta.reviewer.done", "independent reviewer responded")
        else:
            self.say("beta.reviewer.skip",
                     "independent AI reviewer unavailable (no anthropic package or API key)")
        return notes


class ExecutionAgent(Agent):
    """Turns a live signal into orders. Paper by default; live is not implemented."""

    name = "execution"

    def __init__(self, bus: MessageBus, config: Config, ctx: BacktestContext):
        super().__init__(bus, config)
        self.ctx = ctx
        self.broker = PaperBroker(config.risk.capital)

    def current_signals(self, series_map: Dict[str, PairSeries],
                        results: Dict[str, WalkForwardResult],
                        approved: List[str]) -> List[Dict]:
        """What the strategy would do today, for the pairs that were approved."""
        signals: List[Dict] = []
        for pair_id in approved:
            series = series_map.get(pair_id)
            wf = results.get(pair_id)
            if not series or not wf:
                continue
            spec = next((f.chosen for f in reversed(wf.folds) if f.chosen), None)
            if spec is None:
                continue
            scores = score(series.log_spread, spec)
            last = len(series) - 1
            z = scores[last]
            if z is None:
                self.say("signal.none", f"{pair_id}: not enough recent data to score",
                         pair=pair_id)
                continue
            gap_bps = series.gap_bps(last)
            if abs(z) < spec.entry_z:
                self.say("signal.flat",
                         f"{pair_id}: gap is {gap_bps:+.0f} bp ({z:+.2f} standard "
                         f"deviations) — normal, no trade", pair=pair_id, z=z)
                continue

            notional = self.config.risk.capital * self.config.risk.notional_per_leg_pct / 100
            direction = "long_b_short_a" if z > 0 else "long_a_short_b"
            orders = orders_for_signal(
                pair_id, series.dates[last], direction,
                series.a_base[last], series.b_base[last],
                notional / series.a_base[last], notional / series.b_base[last],
                long_only=not self.config.risk.allow_short,
            )
            for order in orders:
                self.broker.submit(order)
            signals.append({"pair": pair_id, "date": series.dates[last], "z": z,
                            "gap_bps": gap_bps, "direction": direction,
                            "orders": orders})
            self.say("signal.trade",
                     f"{pair_id}: gap is {gap_bps:+.0f} bp ({z:+.2f} standard "
                     f"deviations) — {'sell' if z > 0 else 'buy'} the "
                     f"{series.pair.a.symbol} leg", pair=pair_id, z=z)
        return signals


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


@dataclass
class ResearchOutcome:
    config: Config
    provider_name: str
    used_synthetic: bool
    series: Dict[str, PairSeries]
    assessments: List[PairQuality]
    walk_forward: Dict[str, WalkForwardResult]
    portfolio: PortfolioResult
    gate: GateResult
    beta: Dict[str, BetaReport]
    reviewer_notes: Optional[str]
    signals: List[Dict]
    approved_pairs: List[str]
    transcript: str
    elapsed_seconds: float

    @property
    def deployable(self) -> bool:
        """Cleared for paper trading — and paper trading only.

        A single weak pair does not veto the rest: what is required is that the
        portfolio-level gate passed and that at least one pair survived the
        adversarial agents. Only those surviving pairs are ever traded.
        """
        if not self.gate.passed:
            return False
        if not self.config.gate.require_beta_pass:
            return True
        return bool(self.approved_pairs)


class Orchestrator:
    """Wires the agents together and runs one complete research cycle."""

    def __init__(self, config: Config, echo: Optional[Callable[[str], None]] = None,
                 use_llm_reviewer: bool = False):
        problems = config.validate()
        if problems:
            raise ValueError("Configuration problems:\n  - " + "\n  - ".join(problems))
        self.config = config
        self.bus = MessageBus(echo)
        self.use_llm_reviewer = use_llm_reviewer

    def run(self) -> ResearchOutcome:
        started = time.time()
        cfg = self.config
        provider = build_provider(cfg.provider, cfg.data_dir, cfg.seed)
        cost_model = CostModel()
        ctx = BacktestContext(
            cost_model=cost_model,
            base_currency=cfg.base_currency,
            notional_per_leg=cfg.risk.capital * cfg.risk.notional_per_leg_pct / 100.0,
            allow_short=cfg.risk.allow_short,
            fx_hedged=cfg.risk.fx_hedged,
        )

        if cfg.provider == "synthetic":
            universe = list(synthetic_universe().values())
            if cfg.pairs:
                universe = [p for p in universe if p.pair_id in cfg.pairs]
        else:
            universe = selected_pairs(cfg.pairs)

        self.bus.publish("orchestrator", "run.start",
                         f"researching {len(universe)} cross-listed pairs from "
                         f"{cfg.start} using the '{cfg.provider}' data source")

        scraper = ScraperAgent(self.bus, cfg, provider)
        series_map = scraper.collect(universe)
        if not series_map:
            raise DataError(
                "No market data could be collected. If you have no internet access, "
                "run with --provider synthetic to exercise the system offline."
            )

        normaliser = NormalizerAgent(self.bus, cfg)
        series_map = normaliser.verify(series_map)

        selector = PairSelectorAgent(self.bus, cfg, cost_model)
        shortlist, assessments = selector.select(series_map)

        analyst = AnalystAgent(self.bus, cfg, ctx)
        configs_tried = len(analyst.build_grid())
        wf_results = analyst.analyse(series_map, shortlist)

        risk = RiskAgent(self.bus, cfg)
        portfolio = risk.combine(wf_results)

        evaluator = EvaluatorAgent(self.bus, cfg)
        gate = evaluator.evaluate(portfolio)

        beta_reports: Dict[str, BetaReport] = {}
        reviewer_notes: Optional[str] = None
        beta_agent = BetaTestAgent(self.bus, cfg, ctx, self.use_llm_reviewer)
        if gate.passed:
            beta_reports = beta_agent.test(series_map, wf_results, configs_tried)
            reviewer_notes = beta_agent.independent_review({
                "gate": {name: detail for name, (_, detail) in gate.checks.items()},
                "metrics": gate.metrics.to_dict(),
                "pairs": [q.pair_id for q in assessments if q.tradeable],
                "beta_findings": {
                    pid: [f.__dict__ for f in rep.findings]
                    for pid, rep in beta_reports.items()
                },
            })
            for report in beta_reports.values():
                report.reviewer_notes = reviewer_notes
        else:
            self.bus.publish("orchestrator", "beta.skipped",
                             "strategy failed the gate, so adversarial testing was "
                             "not run — there is nothing worth stress-testing")

        approved = [pid for pid, rep in beta_reports.items() if rep.passed] \
            if gate.passed else []

        execution = ExecutionAgent(self.bus, cfg, ctx)
        signals = execution.current_signals(series_map, wf_results, approved) \
            if approved else []
        if not approved:
            self.bus.publish("orchestrator", "execution.blocked",
                             "no pair is approved for trading, so no orders were "
                             "generated — this is the system working as intended")

        elapsed = time.time() - started
        self.bus.publish("orchestrator", "run.end",
                         f"finished in {elapsed:.1f}s")

        return ResearchOutcome(
            config=cfg,
            provider_name=cfg.provider,
            used_synthetic=cfg.provider == "synthetic",
            series=series_map,
            assessments=assessments,
            walk_forward=wf_results,
            portfolio=portfolio,
            gate=gate,
            beta=beta_reports,
            reviewer_notes=reviewer_notes,
            signals=signals,
            approved_pairs=approved,
            transcript=self.bus.transcript(),
            elapsed_seconds=elapsed,
        )
