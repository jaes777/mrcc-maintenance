"""Tests for the arbitrage system.

The important ones are the correctness tests: that the engine cannot see the
future, that costs are actually charged, and that the promotion gate rejects
results it should reject. A bug in any of those would make every number the
system prints a lie, which is worse than useless.

Run with:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arbtool import stats  # noqa: E402
from arbtool.agents import EvaluatorAgent, MessageBus, Orchestrator, RiskAgent  # noqa: E402
from arbtool.backtest import BacktestContext, simulate, walk_forward  # noqa: E402
from arbtool.beta import audit_lookahead, null_hypothesis_test  # noqa: E402
from arbtool.config import Config  # noqa: E402
from arbtool.costs import CostModel  # noqa: E402
from arbtool.data.providers import SyntheticProvider, business_days  # noqa: E402
from arbtool.metrics import compute_metrics  # noqa: E402
from arbtool.models import ModelSpec, grid, score  # noqa: E402
from arbtool.pairs import PairSeries, infer_ratio  # noqa: E402
from arbtool.portfolio import (LiveBroker, PortfolioRules, Trade,  # noqa: E402
                               combine_portfolio, orders_for_signal)
from arbtool.universe import UNIVERSE, VENUES, synthetic_universe  # noqa: E402


def make_series(spread, base_price=100.0, pair=None):
    """Build a PairSeries whose gap follows the supplied spread exactly."""
    pair = pair or synthetic_universe(1)["SYN1"]
    dates = business_days("2015-01-01", "2035-01-01")[: len(spread)]
    b = [base_price] * len(spread)
    a = [base_price * math.exp(s) for s in spread]
    ones = [1.0] * len(spread)
    return PairSeries(pair, dates, a, b, ones, ones, a, b, list(spread), "USD")


def oscillating(n=900, amplitude=0.01, period=20):
    """A perfectly mean-reverting spread — the easiest possible market.

    A pure sine wave has a maximum z-score of sqrt(2) against its own rolling
    window, so tests that use it must set an entry threshold below that or no
    trade will ever fire. ``TRADEABLE_SPEC`` below does.
    """
    return [amplitude * math.sin(2 * math.pi * i / period) for i in range(n)]


# Entry threshold chosen to sit under a sine wave's sqrt(2) ceiling.
TRADEABLE_SPEC = ModelSpec(name="zscore", lookback=60, entry_z=1.2, exit_z=0.2,
                           stop_z=4.0, max_hold_days=20)


class TestStats(unittest.TestCase):
    def test_wilson_penalises_small_samples(self):
        small_lo, _ = stats.wilson_interval(6, 10)
        large_lo, _ = stats.wilson_interval(600, 1000)
        self.assertLess(small_lo, large_lo)
        self.assertLess(small_lo, 0.6)      # 6/10 must not read as a 60% edge
        self.assertGreater(large_lo, 0.55)

    def test_wilson_bounds_are_ordered(self):
        for wins, n in [(0, 10), (5, 10), (10, 10), (1, 1000)]:
            lo, hi = stats.wilson_interval(wins, n)
            self.assertLessEqual(lo, wins / n + 1e-9)
            self.assertGreaterEqual(hi, wins / n - 1e-9)
            self.assertTrue(0.0 <= lo <= hi <= 1.0)

    def test_half_life_detects_mean_reversion(self):
        phi = 0.9
        series, value = [], 1.0
        for _ in range(400):
            series.append(value)
            value *= phi
        hl = stats.half_life(series)
        self.assertIsNotNone(hl)
        self.assertAlmostEqual(hl, -math.log(2) / math.log(phi), delta=0.5)

    def test_half_life_is_none_for_a_trend(self):
        self.assertIsNone(stats.half_life([float(i) for i in range(400)]))

    def test_max_drawdown(self):
        self.assertAlmostEqual(stats.max_drawdown([100, 120, 60, 90]), 50.0)
        self.assertAlmostEqual(stats.max_drawdown([100, 110, 120]), 0.0)

    def test_binomial_tail(self):
        self.assertLess(stats.binomial_tail_pvalue(90, 100, 0.5), 1e-9)
        self.assertGreater(stats.binomial_tail_pvalue(50, 100, 0.5), 0.4)


class TestCosts(unittest.TestCase):
    def setUp(self):
        self.model = CostModel()

    def test_uk_stamp_duty_is_charged(self):
        shel = UNIVERSE["SHEL"]
        items = self.model.round_trip_bps(shel, 10, "USD")
        self.assertGreater(items["total"], VENUES["LSE"].buy_tax_bps)

    def test_borrow_cost_grows_with_holding_period(self):
        pair = UNIVERSE["BHP"]
        short_hold = self.model.breakeven_bps(pair, 1, "USD")
        long_hold = self.model.breakeven_bps(pair, 100, "USD")
        self.assertGreater(long_hold, short_hold)

    def test_non_overlapping_sessions_are_penalised(self):
        bhp = UNIVERSE["BHP"]      # ASX vs NYSE never overlap
        ry = UNIVERSE["RY"]        # TSX and NYSE overlap
        self.assertFalse(bhp.sessions_overlap)
        self.assertTrue(ry.sessions_overlap)
        self.assertGreater(self.model.timing_penalty_bps(bhp), 0)
        self.assertEqual(self.model.timing_penalty_bps(ry), 0)

    def test_long_only_costs_less_than_a_hedged_pair(self):
        pair = UNIVERSE["BHP"]
        self.assertLess(self.model.breakeven_bps(pair, 10, "USD", allow_short=False),
                        self.model.breakeven_bps(pair, 10, "USD", allow_short=True))

    def test_stressed_model_is_harsher(self):
        pair = UNIVERSE["BHP"]
        self.assertGreater(self.model.stressed().breakeven_bps(pair, 10, "USD"),
                           self.model.breakeven_bps(pair, 10, "USD"))


class TestModels(unittest.TestCase):
    def test_every_model_is_causal(self):
        """The critical invariant: no score may depend on a future bar."""
        spread = oscillating(400)
        for name in ("zscore", "ratio", "kalman", "hybrid"):
            spec = ModelSpec(name=name, lookback=40)
            full = score(spread, spec)
            for i in (120, 250, 399):
                truncated = score(spread[: i + 1], spec)
                self.assertEqual(
                    full[i], truncated[i],
                    f"model '{name}' changed its bar-{i} score when future data "
                    f"was removed — it is looking ahead",
                )

    def test_grid_never_produces_exit_above_entry(self):
        specs = grid(["zscore"], [30, 60], [1.5, 2.0], [0.0, 0.5, 1.0, 2.5], 4.0, 20)
        self.assertTrue(specs)
        for spec in specs:
            self.assertLess(spec.exit_z, spec.entry_z)

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(KeyError):
            score([0.0] * 50, ModelSpec(name="does-not-exist"))

    def test_scores_align_with_input_length(self):
        spread = oscillating(200)
        self.assertEqual(len(score(spread, ModelSpec(name="zscore"))), len(spread))


class TestPairSeries(unittest.TestCase):
    def test_infer_ratio_recovers_a_known_ratio(self):
        n = 300
        pair = synthetic_universe(1)["SYN1"]
        dates = business_days("2015-01-01", "2030-01-01")[:n]
        b_local = [50.0] * n
        a_local = [250.0] * n        # 1 unit of A == 5 of B
        ones = [1.0] * n
        series = PairSeries(pair, dates, a_local, b_local, ones, ones,
                            a_local, b_local, [math.log(5.0)] * n, "USD")
        self.assertAlmostEqual(infer_ratio(series), 5.0, places=6)

    def test_gap_in_basis_points(self):
        series = make_series([0.0, 0.01, -0.01])
        self.assertAlmostEqual(series.gap_bps(1), 100.0, places=6)
        self.assertAlmostEqual(series.gap_bps(2), -100.0, places=6)


class TestBacktest(unittest.TestCase):
    def setUp(self):
        self.ctx = BacktestContext(CostModel(), "USD", 10_000.0, allow_short=True)

    def test_no_trades_when_the_gap_never_widens(self):
        series = make_series([0.0] * 500)
        trades = simulate(series, ModelSpec(name="zscore", lookback=60), self.ctx)
        self.assertEqual(trades, [])

    def test_mean_reverting_market_produces_winning_trades(self):
        series = make_series(oscillating(900, amplitude=0.02, period=30))
        spec = TRADEABLE_SPEC
        trades = simulate(series, spec, self.ctx)
        self.assertGreater(len(trades), 5)
        metrics = compute_metrics(trades)
        self.assertGreater(metrics.win_rate, 0.5)

    def test_costs_are_actually_deducted(self):
        series = make_series(oscillating(900, amplitude=0.02, period=30))
        spec = TRADEABLE_SPEC
        for trade in simulate(series, spec, self.ctx):
            self.assertGreater(trade.costs, 0.0)
            self.assertAlmostEqual(trade.net_pnl, trade.gross_pnl - trade.costs, places=6)

    def test_execution_happens_after_the_signal(self):
        """A signal seen at bar i must never be filled at bar i."""
        series = make_series(oscillating(900, amplitude=0.02, period=30))
        spec = TRADEABLE_SPEC
        scores = score(series.log_spread, spec)
        for trade in simulate(series, spec, self.ctx, scores=scores):
            fill = series.index_of(trade.entry_date)
            signal_z = scores[fill - 1]
            self.assertIsNotNone(signal_z)
            self.assertGreaterEqual(abs(signal_z), spec.entry_z)

    def test_higher_costs_reduce_profit(self):
        series = make_series(oscillating(900, amplitude=0.02, period=30))
        spec = TRADEABLE_SPEC
        cheap = compute_metrics(simulate(series, spec, self.ctx))
        expensive_ctx = BacktestContext(CostModel().stressed(5.0, 5.0), "USD", 10_000.0)
        expensive = compute_metrics(simulate(series, spec, expensive_ctx))
        self.assertLess(expensive.expectancy_bps, cheap.expectancy_bps)

    def test_direction_is_short_the_rich_leg(self):
        series = make_series(oscillating(900, amplitude=0.02, period=30))
        spec = TRADEABLE_SPEC
        for trade in simulate(series, spec, self.ctx):
            if trade.entry_z > 0:      # A is expensive
                self.assertEqual(trade.direction, "long_b_short_a")
            else:
                self.assertEqual(trade.direction, "long_a_short_b")

    def test_walk_forward_only_scores_unseen_data(self):
        series = make_series(oscillating(2000, amplitude=0.02, period=30))
        specs = grid(["zscore"], [30, 60], [1.2, 1.35], [0.0, 0.5], 4.0, 20)
        result = walk_forward(series, self.ctx, specs, 400, 100, 100)
        self.assertTrue(result.folds)
        self.assertTrue(all(t.out_of_sample for t in result.oos_trades))
        for fold in result.folds:
            self.assertLess(fold.train_end, fold.test_start)


class TestPortfolio(unittest.TestCase):
    def _trade(self, pair, entry, exit_, pnl):
        return Trade(pair, "m", "long_b_short_a", entry, exit_, 2.0, 0.1, 200, 10,
                     10_000.0, pnl + 20, 20, pnl, pnl / 10_000 * 10_000, 5, "converged")

    def test_concurrency_limit_is_enforced(self):
        rules = PortfolioRules(100_000, 10_000, 2, 1e9, 1e9, 1e9)
        trades = [self._trade(f"P{i}", "2020-01-01", "2020-06-01", 100)
                  for i in range(5)]
        result = combine_portfolio(trades, rules)
        self.assertEqual(len(result.trades), 2)
        self.assertEqual(len(result.skipped), 3)

    def test_gross_exposure_limit_is_enforced(self):
        rules = PortfolioRules(100_000, 10_000, 10, 20_000, 1e9, 1e9)
        trades = [self._trade(f"P{i}", "2020-01-01", "2020-06-01", 100)
                  for i in range(4)]
        result = combine_portfolio(trades, rules)
        self.assertEqual(len(result.trades), 1)  # 1 trade == 2 legs == 20,000 gross

    def test_per_trade_loss_is_capped(self):
        rules = PortfolioRules(100_000, 10_000, 5, 1e9, 1e9, 2_000)
        result = combine_portfolio([self._trade("P", "2020-01-01", "2020-02-01", -9_000)],
                                   rules)
        self.assertEqual(result.trades[0].net_pnl, -2_000)
        self.assertEqual(result.trades[0].exit_reason, "risk_stop")

    def test_orders_are_one_buy_and_one_short(self):
        orders = orders_for_signal("BHP", "2024-01-02", "long_b_short_a",
                                   50.0, 25.0, 200.0, 400.0, long_only=False)
        self.assertEqual(len(orders), 2)
        self.assertEqual({o["side"] for o in orders}, {"BUY", "SELL SHORT"})
        buy = next(o for o in orders if o["side"] == "BUY")
        self.assertEqual(buy["leg"], "B")     # A is the rich leg, so B is bought

    def test_long_only_produces_a_single_buy(self):
        orders = orders_for_signal("BHP", "2024-01-02", "long_a_short_b",
                                   50.0, 25.0, 200.0, 400.0, long_only=True)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["side"], "BUY")
        self.assertEqual(orders[0]["leg"], "A")

    def test_live_broker_refuses_to_run(self):
        with self.assertRaises(NotImplementedError):
            LiveBroker()


class TestGate(unittest.TestCase):
    def _evaluate(self, trades, **gate_overrides):
        config = Config()
        for key, value in gate_overrides.items():
            setattr(config.gate, key, value)
        agent = EvaluatorAgent(MessageBus(), config)
        portfolio = combine_portfolio(
            trades, RiskAgent(MessageBus(), config).rules())
        return agent.evaluate(portfolio)

    def _winners(self, n, wins, pnl=100.0):
        trades = []
        for i in range(n):
            day = f"2020-{1 + i // 20:02d}-{1 + i % 20:02d}"
            amount = pnl if i < wins else -pnl
            trades.append(Trade(f"P{i%3}", "m", "long_b_short_a", day, day, 2.0, 0.1,
                                200, 10, 10_000.0, amount + 20, 20, amount,
                                amount / 10_000 * 10_000, 3, "converged"))
        return trades

    def test_small_sample_is_rejected_even_at_100_percent(self):
        result = self._evaluate(self._winners(10, 10), min_oos_trades=30)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["enough unseen trades"][0])

    def test_sixty_percent_on_a_thin_sample_fails_the_lower_bound(self):
        """6 wins from 10 reads as 60% but must not pass a 60% requirement."""
        result = self._evaluate(self._winners(10, 6), min_oos_trades=5,
                                min_win_rate=0.60)
        self.assertAlmostEqual(result.metrics.win_rate, 0.60)
        self.assertLess(result.metrics.win_rate_lower, 0.60)
        self.assertFalse(result.passed)

    def test_strong_large_sample_passes(self):
        result = self._evaluate(self._winners(200, 150), min_oos_trades=30)
        self.assertTrue(result.passed, result.reasons)

    def test_high_win_rate_with_fat_losses_is_rejected(self):
        """80% wins of 1 unit against 20% losses of 10 units loses money."""
        trades = []
        for i in range(100):
            day = f"2020-{1 + i // 25:02d}-{1 + i % 25:02d}"
            amount = 100.0 if i % 5 else -1_000.0
            trades.append(Trade("P", "m", "long_b_short_a", day, day, 2.0, 0.1, 200, 10,
                                10_000.0, amount + 20, 20, amount, amount / 10_000 * 1e4,
                                3, "converged"))
        result = self._evaluate(trades, min_oos_trades=30)
        self.assertGreaterEqual(result.metrics.win_rate, 0.79)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["profit factor"][0])


class TestBetaAgents(unittest.TestCase):
    def test_lookahead_audit_passes_a_causal_model(self):
        series = make_series(oscillating(800))
        finding = audit_lookahead(series, ModelSpec(name="zscore", lookback=60))
        self.assertTrue(finding.passed, finding.detail)

    def test_lookahead_audit_catches_a_cheating_model(self):
        """Register a deliberately non-causal model and confirm it is caught."""
        from arbtool import models

        @models.register("_cheater_for_tests")
        def _cheater(spread, spec):
            # Uses the whole series' mean — information from the future.
            mean = sum(spread) / len(spread)
            sd = stats.stdev(spread) or 1.0
            return [None if i < spec.lookback else (spread[i] - mean) / sd
                    for i in range(len(spread))]

        try:
            series = make_series(oscillating(800) +
                                 [0.05 + 0.001 * i for i in range(200)])
            finding = audit_lookahead(series, ModelSpec(name="_cheater_for_tests",
                                                        lookback=60))
            self.assertFalse(finding.passed)
        finally:
            models._REGISTRY.pop("_cheater_for_tests", None)

    def test_null_market_shows_no_edge(self):
        series = make_series(oscillating(1200, amplitude=0.02, period=30))
        ctx = BacktestContext(CostModel(), "USD", 10_000.0)
        spec = TRADEABLE_SPEC
        finding = null_hypothesis_test(series, spec, ctx, 100, len(series), runs=4)
        self.assertTrue(finding.passed, finding.headline)


class TestConfig(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.json")
            original = Config(start="2019-01-01")
            original.risk.capital = 55_000
            original.gate.min_win_rate = 0.65
            original.save(path)
            loaded = Config.load(path)
            self.assertEqual(loaded.start, "2019-01-01")
            self.assertEqual(loaded.risk.capital, 55_000)
            self.assertEqual(loaded.gate.min_win_rate, 0.65)

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(ValueError):
            Config.from_dict({"start": "2019-01-01", "typo_key": 1})

    def test_validation_catches_contradictions(self):
        config = Config()
        config.signal.entry_z = 0.5
        config.signal.exit_z = 2.0
        self.assertTrue(any("entry_z" in p for p in config.validate()))

    def test_live_mode_requires_acknowledgement(self):
        config = Config()
        config.execution_mode = "live"
        self.assertTrue(any("live_trading_acknowledged" in p
                            for p in config.validate()))


class TestSyntheticProvider(unittest.TestCase):
    def test_deterministic(self):
        a = SyntheticProvider(seed=3).fetch_prices("SYN1.A", "2020-01-01", "2021-01-01")
        b = SyntheticProvider(seed=3).fetch_prices("SYN1.A", "2020-01-01", "2021-01-01")
        self.assertEqual([x.close for x in a], [x.close for x in b])

    def test_legs_track_each_other(self):
        provider = SyntheticProvider(seed=5)
        a = provider.fetch_prices("SYN2.A", "2018-01-01", "2022-01-01")
        b = provider.fetch_prices("SYN2.B", "2018-01-01", "2022-01-01")
        gaps = [abs(math.log(x.close / y.close)) for x, y in zip(a, b)]
        # A realistic cross-listing gap: mostly well under 1%.
        self.assertLess(sum(gaps) / len(gaps), 0.05)

    def test_business_days_excludes_weekends(self):
        days = business_days("2024-01-01", "2024-01-14")
        self.assertEqual(len(days), 10)
        self.assertNotIn("2024-01-06", days)


class TestEndToEnd(unittest.TestCase):
    def test_full_pipeline_runs_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(provider="synthetic", start="2016-01-01",
                            data_dir=tmp, pairs=["SYN1", "SYN2"])
            config.walk_forward.train_days = 400
            config.walk_forward.test_days = 120
            config.walk_forward.step_days = 120
            config.search_models = ["zscore"]
            config.search_lookback = [60]
            config.search_entry_z = [2.0]
            config.search_exit_z = [0.5]

            outcome = Orchestrator(config).run()
            self.assertEqual(len(outcome.series), 2)
            self.assertTrue(outcome.transcript)
            self.assertIsInstance(outcome.deployable, bool)

            from arbtool.report import write_json, write_report
            html_path = write_report(outcome, os.path.join(tmp, "r.html"))
            json_path = write_json(outcome, os.path.join(tmp, "r.json"))
            self.assertTrue(os.path.getsize(html_path) > 2000)
            self.assertTrue(os.path.getsize(json_path) > 200)

    def test_orchestrator_rejects_a_bad_config(self):
        config = Config(provider="synthetic")
        config.risk.capital = -1
        with self.assertRaises(ValueError):
            Orchestrator(config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
