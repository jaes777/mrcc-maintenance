"""Configuration for the cross-market arbitrage research system.

Everything the system does is driven by this one object, so a single JSON file
fully describes (and reproduces) a run.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List


@dataclass
class GateConfig:
    """The promotion gate. A strategy is only allowed to trade if it passes.

    ``min_win_rate`` is applied to the *lower bound* of a 95% Wilson confidence
    interval on out-of-sample trades, not to the raw sample win rate. Requiring
    the lower bound is what stops a lucky 7-out-of-10 run from being mistaken
    for a 70% edge.
    """

    min_win_rate: float = 0.60
    confidence: float = 0.95
    min_oos_trades: int = 30
    min_profit_factor: float = 1.20
    max_drawdown_pct: float = 15.0
    min_expectancy_bps: float = 1.0
    require_beta_pass: bool = True


@dataclass
class WalkForwardConfig:
    """Out-of-sample protocol.

    History is cut into consecutive folds. Parameters are fitted on ``train_days``
    and then applied, untouched, to the following ``test_days``. Only the test
    segments are ever scored. This is the single most important defence against
    a model that merely memorised the past.
    """

    train_days: int = 500
    test_days: int = 125
    step_days: int = 125
    min_folds: int = 3
    purge_days: int = 2  # dropped between train and test to avoid leakage


@dataclass
class RiskConfig:
    capital: float = 100_000.0
    notional_per_leg_pct: float = 10.0     # % of capital per leg of one pair trade
    max_concurrent_positions: int = 5
    max_gross_exposure_pct: float = 150.0
    max_loss_per_trade_pct: float = 2.0    # hard stop, % of capital
    daily_loss_halt_pct: float = 4.0       # kill switch for the day
    allow_short: bool = True               # false => long-the-cheap-leg only
    fx_hedged: bool = False


@dataclass
class SignalConfig:
    lookback: int = 60          # bars used for the rolling spread mean/std
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    max_hold_days: int = 20
    min_halflife: float = 1.0   # reject pairs that mean-revert implausibly fast
    max_halflife: float = 40.0  # ...or too slowly to trade


@dataclass
class Config:
    base_currency: str = "USD"
    pairs: List[str] = field(default_factory=list)      # empty => whole universe
    provider: str = "auto"                              # auto|stooq|yahoo|synthetic
    start: str = "2018-01-01"
    end: str = ""                                       # empty => today
    data_dir: str = "data"
    seed: int = 7

    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    gate: GateConfig = field(default_factory=GateConfig)

    # Execution safety. Live trading stays off unless BOTH are switched on and a
    # broker adapter has been implemented by the user.
    execution_mode: str = "paper"           # paper|live
    live_trading_acknowledged: bool = False

    # Grid searched during the training half of each walk-forward fold.
    search_lookback: List[int] = field(default_factory=lambda: [30, 60, 90, 120])
    search_entry_z: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0])
    search_exit_z: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    search_models: List[str] = field(default_factory=lambda: ["zscore", "ratio", "kalman"])

    # --- (de)serialisation -------------------------------------------------

    _NESTED = {
        "signal": SignalConfig,
        "risk": RiskConfig,
        "walk_forward": WalkForwardConfig,
        "gate": GateConfig,
    }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(
                f"Unknown config key(s): {', '.join(unknown)}. "
                f"Valid keys: {', '.join(sorted(known))}"
            )
        kwargs: Dict[str, Any] = {}
        for key, value in raw.items():
            sub = cls._NESTED.get(key)
            if sub is not None:
                if not isinstance(value, dict):
                    raise ValueError(f"Config key '{key}' must be an object")
                sub_known = {f.name for f in fields(sub)}
                sub_unknown = sorted(set(value) - sub_known)
                if sub_unknown:
                    raise ValueError(
                        f"Unknown key(s) in '{key}': {', '.join(sub_unknown)}"
                    )
                kwargs[key] = sub(**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    def validate(self) -> List[str]:
        """Return a list of human-readable problems (empty means the config is sane)."""
        problems: List[str] = []
        s, r, w, g = self.signal, self.risk, self.walk_forward, self.gate
        if s.entry_z <= s.exit_z:
            problems.append("signal.entry_z must be greater than signal.exit_z")
        if s.stop_z <= s.entry_z:
            problems.append("signal.stop_z must be greater than signal.entry_z")
        if s.lookback < 10:
            problems.append("signal.lookback must be at least 10 bars")
        if r.capital <= 0:
            problems.append("risk.capital must be positive")
        if not 0 < r.notional_per_leg_pct <= 100:
            problems.append("risk.notional_per_leg_pct must be in (0, 100]")
        if r.max_concurrent_positions < 1:
            problems.append("risk.max_concurrent_positions must be at least 1")
        if w.train_days < 100:
            problems.append("walk_forward.train_days must be at least 100")
        if w.test_days < 20:
            problems.append("walk_forward.test_days must be at least 20")
        if not 0 < g.min_win_rate < 1:
            problems.append("gate.min_win_rate must be a fraction between 0 and 1")
        if g.min_oos_trades < 20:
            problems.append(
                "gate.min_oos_trades below 20 cannot support a credible win-rate claim"
            )
        if self.execution_mode == "live" and not self.live_trading_acknowledged:
            problems.append(
                "execution_mode 'live' requires live_trading_acknowledged = true"
            )
        return problems
