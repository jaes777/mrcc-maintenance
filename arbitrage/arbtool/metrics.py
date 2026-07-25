"""Scoring a set of trades.

Win rate is deliberately not the only number here. A strategy that wins 90% of
the time and loses ten times its average win on the other 10% is a losing
strategy, and that shape — many small wins, rare large losses — is exactly what
mean-reversion trading produces naturally. Profit factor, expectancy and
drawdown are reported next to the win rate for that reason.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from .stats import (binomial_tail_pvalue, max_drawdown, mean, percentile, sharpe,
                    stdev, wilson_interval)


@dataclass
class Metrics:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    win_rate_lower: float = 0.0     # 95% lower confidence bound — the gated number
    win_rate_upper: float = 0.0
    luck_pvalue: float = 1.0        # chance of this result if the true rate were 50%
    net_pnl: float = 0.0
    gross_pnl: float = 0.0
    total_costs: float = 0.0
    profit_factor: float = 0.0
    expectancy_bps: float = 0.0
    avg_win_bps: float = 0.0
    avg_loss_bps: float = 0.0
    worst_trade_bps: float = 0.0
    best_trade_bps: float = 0.0
    payoff_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    avg_hold_days: float = 0.0
    exposure_days: int = 0
    cost_share_of_gross: float = 0.0
    exit_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def headline(self) -> str:
        return (
            f"{self.trades} trades | win rate {self.win_rate * 100:.1f}% "
            f"(95% CI {self.win_rate_lower * 100:.1f}–{self.win_rate_upper * 100:.1f}%) "
            f"| profit factor {self.profit_factor:.2f} "
            f"| net {self.net_pnl:,.0f} | max drawdown {self.max_drawdown_pct:.1f}%"
        )


def compute_metrics(trades: Sequence, equity: Optional[Sequence[float]] = None,
                    confidence: float = 0.95,
                    daily_returns: Optional[Sequence[float]] = None) -> Metrics:
    """Score a list of :class:`arbtool.portfolio.Trade`."""
    m = Metrics()
    m.trades = len(trades)
    if not trades:
        return m

    net = [t.net_pnl for t in trades]
    bps = [t.return_bps for t in trades]
    wins = [x for x in net if x > 0]
    losses = [x for x in net if x <= 0]

    m.wins, m.losses = len(wins), len(losses)
    m.win_rate = m.wins / m.trades
    m.win_rate_lower, m.win_rate_upper = wilson_interval(m.wins, m.trades, confidence)
    m.luck_pvalue = binomial_tail_pvalue(m.wins, m.trades, 0.5)

    m.net_pnl = sum(net)
    m.gross_pnl = sum(t.gross_pnl for t in trades)
    m.total_costs = sum(t.costs for t in trades)
    gross_wins = sum(wins)
    gross_losses = abs(sum(losses))
    m.profit_factor = (gross_wins / gross_losses) if gross_losses > 1e-9 else (
        float("inf") if gross_wins > 0 else 0.0)

    m.expectancy_bps = mean(bps)
    win_bps = [b for b in bps if b > 0]
    loss_bps = [b for b in bps if b <= 0]
    m.avg_win_bps = mean(win_bps) if win_bps else 0.0
    m.avg_loss_bps = mean(loss_bps) if loss_bps else 0.0
    m.payoff_ratio = (abs(m.avg_win_bps / m.avg_loss_bps)
                      if abs(m.avg_loss_bps) > 1e-9 else float("inf"))
    m.worst_trade_bps = min(bps)
    m.best_trade_bps = max(bps)
    m.avg_hold_days = mean([t.hold_days for t in trades])
    m.exposure_days = int(sum(t.hold_days for t in trades))
    m.cost_share_of_gross = (m.total_costs / abs(m.gross_pnl) * 100.0
                             if abs(m.gross_pnl) > 1e-9 else 0.0)

    reasons: Dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    m.exit_reasons = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))

    if equity:
        m.max_drawdown_pct = max_drawdown(equity)
    if daily_returns:
        m.sharpe = sharpe(daily_returns)
    return m


def tail_summary(trades: Sequence) -> Dict[str, float]:
    """The part of the distribution that actually ends accounts."""
    if not trades:
        return {}
    bps = sorted(t.return_bps for t in trades)
    return {
        "p01_bps": percentile(bps, 1),
        "p05_bps": percentile(bps, 5),
        "median_bps": percentile(bps, 50),
        "p95_bps": percentile(bps, 95),
        "p99_bps": percentile(bps, 99),
        "worst_bps": bps[0],
        "stdev_bps": stdev(bps),
        "loss_tail_vs_avg_win": (
            abs(percentile(bps, 1)) / mean([b for b in bps if b > 0])
            if any(b > 0 for b in bps) else float("inf")
        ),
    }


def equity_from_trades(trades: Sequence, starting_capital: float) -> List[float]:
    """Equity curve stepped at each trade close (used when no daily marks exist)."""
    equity = [starting_capital]
    running = starting_capital
    for t in sorted(trades, key=lambda x: x.exit_date):
        running += t.net_pnl
        equity.append(running)
    return equity
