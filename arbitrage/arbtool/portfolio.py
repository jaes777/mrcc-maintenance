"""Trades, positions, brokers and the portfolio-level rules.

Two brokers exist:

* :class:`PaperBroker` — records orders and marks them against real prices, with
  no money involved. This is the default and covers everything the research
  pipeline needs.
* :class:`LiveBroker` — an unimplemented interface. It deliberately refuses to
  run. Connecting real money is a decision a person has to make consciously,
  with their own broker's API, after reading the risk notes; it is not something
  this tool should do on anyone's behalf.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Sequence


@dataclass
class Trade:
    """One complete round trip: both legs opened, both legs closed."""

    pair_id: str
    model: str
    direction: str            # "long_b_short_a" (A is rich) or "long_a_short_b"
    entry_date: str
    exit_date: str
    entry_z: float
    exit_z: float
    entry_gap_bps: float
    exit_gap_bps: float
    notional_per_leg: float
    gross_pnl: float
    costs: float
    net_pnl: float
    return_bps: float         # net P&L as bps of one leg's notional
    hold_days: int
    exit_reason: str          # converged | stop | time | end_of_data
    fold: int = -1
    out_of_sample: bool = True
    long_only: bool = False

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    def to_dict(self) -> Dict:
        return asdict(self)

    def describe(self) -> str:
        verb = "WIN " if self.is_win else "LOSS"
        return (
            f"{verb} {self.pair_id} {self.entry_date}->{self.exit_date} "
            f"({self.hold_days}d) gap {self.entry_gap_bps:+.0f}bp -> "
            f"{self.exit_gap_bps:+.0f}bp, net {self.net_pnl:+,.0f} "
            f"({self.return_bps:+.0f}bp) [{self.exit_reason}]"
        )


@dataclass
class OpenPosition:
    pair_id: str
    model: str
    direction: str
    entry_index: int
    entry_date: str
    entry_z: float
    entry_a: float            # base-currency price per ordinary share at entry
    entry_b: float
    shares_a: float
    shares_b: float
    notional_per_leg: float
    entry_cost: float
    long_only: bool = False

    def side_a(self) -> int:
        return -1 if self.direction == "long_b_short_a" else 1

    def side_b(self) -> int:
        return -self.side_a()

    def mark(self, price_a: float, price_b: float) -> float:
        """Unrealised P&L at the given prices, before exit costs."""
        pnl = self.side_a() * self.shares_a * (price_a - self.entry_a)
        if not self.long_only:
            pnl += self.side_b() * self.shares_b * (price_b - self.entry_b)
        return pnl


class PaperBroker:
    """Records intended orders and settles them at observed prices. No money."""

    mode = "paper"

    def __init__(self, starting_capital: float):
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.orders: List[Dict] = []
        self.trades: List[Trade] = []

    def submit(self, order: Dict) -> Dict:
        order = dict(order)
        order["submitted_at"] = datetime.now().isoformat(timespec="seconds")
        order["status"] = "filled_paper"
        self.orders.append(order)
        return order

    def record_trade(self, trade: Trade) -> None:
        self.trades.append(trade)
        self.cash += trade.net_pnl

    @property
    def equity(self) -> float:
        return self.cash

    def blotter(self) -> str:
        if not self.orders:
            return "No orders."
        lines = ["date        pair    side        leg   qty        price      note"]
        for o in self.orders:
            lines.append(
                f"{o.get('date',''):<11} {o.get('pair_id',''):<7} "
                f"{o.get('side',''):<11} {o.get('leg',''):<5} "
                f"{o.get('quantity',0):<10.2f} {o.get('price',0):<10.4f} "
                f"{o.get('note','')}"
            )
        return "\n".join(lines)


class LiveBroker:
    """Placeholder for a real brokerage connection. Intentionally inert.

    To trade real money you would implement :meth:`submit` against your own
    broker's API and supply your own credentials. That step is left to you on
    purpose: an automated system that can move real money should only ever be
    switched on by someone who understands what it will do when it is wrong.
    """

    mode = "live"

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "Live trading is not implemented, by design.\n"
            "\n"
            "This tool researches, tests and paper-trades cross-listing spreads.\n"
            "It does not place real orders. To do that you would need to:\n"
            "  1. open an account with a broker that supports every venue in the\n"
            "     pair AND permits short selling on the expensive leg;\n"
            "  2. implement LiveBroker.submit() against that broker's API;\n"
            "  3. replace the modelled costs in costs.py with your real, quoted\n"
            "     commissions, FX spreads and borrow rates;\n"
            "  4. run in paper mode long enough to collect a genuine out-of-sample\n"
            "     record — the backtest is not that record;\n"
            "  5. understand that a 60% win rate is not the same as making money,\n"
            "     and that the losing 40% can be far larger than the winning 60%.\n"
            "\n"
            "See RISK.md before going anywhere near this."
        )


@dataclass
class PortfolioRules:
    capital: float
    notional_per_leg: float
    max_concurrent: int
    max_gross_exposure: float
    daily_loss_halt: float
    max_loss_per_trade: float


@dataclass
class PortfolioResult:
    trades: List[Trade]
    skipped: List[Dict]
    equity_dates: List[str]
    equity: List[float]
    daily_returns: List[float]
    halted_days: List[str]

    def to_dict(self) -> Dict:
        return {
            "trades": [t.to_dict() for t in self.trades],
            "skipped": self.skipped,
            "equity_dates": self.equity_dates,
            "equity": self.equity,
            "halted_days": self.halted_days,
        }


def combine_portfolio(all_trades: Sequence[Trade], rules: PortfolioRules) -> PortfolioResult:
    """Apply portfolio-level limits to trades generated independently per pair.

    Each pair is analysed on its own, which can produce more simultaneous trades
    than the account could ever fund. This replays them in date order and drops
    the ones that would have breached a limit — capital, concurrency, or the
    daily loss halt — so the reported result is one an account could actually
    have achieved.
    """
    ordered = sorted(all_trades, key=lambda t: (t.entry_date, t.pair_id))
    accepted: List[Trade] = []
    skipped: List[Dict] = []
    open_until: List[str] = []       # exit dates of currently-open trades
    daily_pnl: Dict[str, float] = {}
    halted: List[str] = []

    for trade in ordered:
        open_until = [d for d in open_until if d > trade.entry_date]

        if len(open_until) >= rules.max_concurrent:
            skipped.append({"trade": trade.pair_id, "date": trade.entry_date,
                            "reason": "position limit reached"})
            continue
        gross = (len(open_until) + 1) * rules.notional_per_leg * (1 if trade.long_only else 2)
        if gross > rules.max_gross_exposure:
            skipped.append({"trade": trade.pair_id, "date": trade.entry_date,
                            "reason": "gross exposure limit reached"})
            continue
        if daily_pnl.get(trade.entry_date, 0.0) <= -rules.daily_loss_halt:
            if trade.entry_date not in halted:
                halted.append(trade.entry_date)
            skipped.append({"trade": trade.pair_id, "date": trade.entry_date,
                            "reason": "daily loss halt in force"})
            continue

        capped = trade
        if trade.net_pnl < -rules.max_loss_per_trade:
            # The per-trade stop would have cut the loss before it reached this.
            capped = _cap_loss(trade, rules.max_loss_per_trade)
        accepted.append(capped)
        open_until.append(capped.exit_date)
        daily_pnl[capped.exit_date] = daily_pnl.get(capped.exit_date, 0.0) + capped.net_pnl

    equity_dates, equity, daily_returns = _equity_curve(accepted, rules.capital)
    return PortfolioResult(accepted, skipped, equity_dates, equity, daily_returns, halted)


def _cap_loss(trade: Trade, max_loss: float) -> Trade:
    scaled = Trade(**trade.to_dict())
    scaled.net_pnl = -max_loss
    scaled.return_bps = -max_loss / trade.notional_per_leg * 10_000.0
    scaled.exit_reason = "risk_stop"
    return scaled


def _equity_curve(trades: Sequence[Trade], capital: float):
    """Daily equity, marked when trades close."""
    if not trades:
        return [], [capital], []
    by_day: Dict[str, float] = {}
    for t in trades:
        by_day[t.exit_date] = by_day.get(t.exit_date, 0.0) + t.net_pnl
    days = sorted(by_day)
    equity = [capital]
    dates = [days[0]]
    running = capital
    for day in days:
        running += by_day[day]
        equity.append(running)
        dates.append(day)
    returns = [
        (equity[i] - equity[i - 1]) / equity[i - 1]
        for i in range(1, len(equity)) if equity[i - 1] > 0
    ]
    return dates, equity, returns


def orders_for_signal(pair_id: str, date: str, direction: str, price_a: float,
                      price_b: float, shares_a: float, shares_b: float,
                      long_only: bool) -> List[Dict]:
    """The concrete orders a broker would receive for one pair trade."""
    a_is_rich = direction == "long_b_short_a"
    buy_leg = {"leg": "B", "quantity": shares_b, "price": price_b} if a_is_rich else \
              {"leg": "A", "quantity": shares_a, "price": price_a}
    sell_leg = {"leg": "A", "quantity": shares_a, "price": price_a} if a_is_rich else \
               {"leg": "B", "quantity": shares_b, "price": price_b}

    def order(spec: Dict, side: str, note: str) -> Dict:
        return {"date": date, "pair_id": pair_id, "leg": spec["leg"], "side": side,
                "quantity": round(spec["quantity"], 4),
                "price": round(spec["price"], 4), "note": note}

    orders = [order(buy_leg, "BUY", "cheap listing")]
    if long_only:
        orders[0]["note"] = "cheap listing (long-only mode: unhedged)"
    else:
        orders.append(order(sell_leg, "SELL SHORT", "expensive listing"))
    return orders


def save_trades(trades: Sequence[Trade], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([t.to_dict() for t in trades], handle, indent=2)
