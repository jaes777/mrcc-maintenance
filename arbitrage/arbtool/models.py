"""Candidate models for deciding when a gap is abnormally wide.

Each model answers one question: *given only what was known at the time, how
unusual is today's gap?* The answer is a z-score — the number of standard
deviations the gap sits away from its own recent normal.

Every model here is strictly causal. Bar ``i``'s score uses bars ``0..i`` and
nothing later. That discipline is what separates a backtest from a fantasy, and
it is verified by an automated look-ahead audit in :mod:`arbtool.beta`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .stats import mean, stdev


@dataclass
class ModelSpec:
    """A model plus the parameters it was fitted with."""

    name: str
    lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    max_hold_days: int = 20
    extra: Dict[str, float] = field(default_factory=dict)

    def key(self) -> str:
        return (f"{self.name}|lb={self.lookback}|in={self.entry_z}"
                f"|out={self.exit_z}|stop={self.stop_z}")

    def describe(self) -> str:
        return (
            f"{self.name} model — measures the gap against the last "
            f"{self.lookback} trading days; opens a trade when the gap is "
            f"{self.entry_z} standard deviations wide, closes it at "
            f"{self.exit_z}, gives up at {self.stop_z} or after "
            f"{self.max_hold_days} days"
        )


ScoreFn = Callable[[Sequence[float], ModelSpec], List[Optional[float]]]
_REGISTRY: Dict[str, ScoreFn] = {}


def register(name: str) -> Callable[[ScoreFn], ScoreFn]:
    def wrap(fn: ScoreFn) -> ScoreFn:
        _REGISTRY[name] = fn
        return fn
    return wrap


def available_models() -> List[str]:
    return sorted(_REGISTRY)


def score(spread: Sequence[float], spec: ModelSpec) -> List[Optional[float]]:
    try:
        fn = _REGISTRY[spec.name]
    except KeyError:
        raise KeyError(
            f"Unknown model '{spec.name}'. Available: {', '.join(available_models())}"
        ) from None
    scores = fn(spread, spec)
    if len(scores) != len(spread):
        raise RuntimeError(f"model '{spec.name}' returned a misaligned score series")
    return scores


@register("zscore")
def _zscore(spread: Sequence[float], spec: ModelSpec) -> List[Optional[float]]:
    """Fixed-window z-score: the plain, hard-to-beat baseline."""
    out: List[Optional[float]] = []
    window = max(10, spec.lookback)
    for i in range(len(spread)):
        if i < window:
            out.append(None)
            continue
        history = spread[i - window:i]          # excludes today: no self-reference
        sd = stdev(history)
        out.append(None if sd <= 1e-9 else (spread[i] - mean(history)) / sd)
    return out


@register("ratio")
def _ratio(spread: Sequence[float], spec: ModelSpec) -> List[Optional[float]]:
    """Median-anchored score: less easily dragged around by one big outlier."""
    out: List[Optional[float]] = []
    window = max(10, spec.lookback)
    for i in range(len(spread)):
        if i < window:
            out.append(None)
            continue
        history = sorted(spread[i - window:i])
        mid = history[len(history) // 2]
        # Median absolute deviation, scaled so it matches a standard deviation
        # for well-behaved data.
        mad = sorted(abs(x - mid) for x in history)[len(history) // 2]
        scale = mad * 1.4826
        if scale <= 1e-9:
            scale = stdev(history)
        out.append(None if scale <= 1e-9 else (spread[i] - mid) / scale)
    return out


@register("kalman")
def _kalman(spread: Sequence[float], spec: ModelSpec) -> List[Optional[float]]:
    """Adaptive score that lets the 'normal' level drift.

    Cross-listing premiums are not constant — index rebalances, tax changes and
    capital controls move them permanently. A fixed window treats such a shift
    as a huge opportunity and keeps betting against it. This version updates its
    idea of normal continuously, so it adapts instead of fighting.
    """
    alpha = float(spec.extra.get("alpha", 2.0 / (max(10, spec.lookback) + 1.0)))
    out: List[Optional[float]] = []
    level: Optional[float] = None
    variance = 0.0
    warmup = max(10, spec.lookback // 2)
    for i, value in enumerate(spread):
        if level is None:
            level, variance = value, 0.0
            out.append(None)
            continue
        deviation = value - level
        sd = math.sqrt(variance) if variance > 0 else 0.0
        out.append(None if (i < warmup or sd <= 1e-9) else deviation / sd)
        # Update AFTER scoring, so today's value never informs today's score.
        variance = (1 - alpha) * (variance + alpha * deviation * deviation)
        level = level + alpha * deviation
    return out


@register("hybrid")
def _hybrid(spread: Sequence[float], spec: ModelSpec) -> List[Optional[float]]:
    """Agreement filter: trades only when the fixed and adaptive views concur.

    Fewer trades, but the ones that survive are the ones both a stable and an
    adaptive view of 'normal' call extreme. Designed for win rate rather than
    trade count.
    """
    fixed = _zscore(spread, spec)
    adaptive = _kalman(spread, spec)
    out: List[Optional[float]] = []
    for f, a in zip(fixed, adaptive):
        if f is None or a is None or (f > 0) != (a > 0):
            out.append(None)
        else:
            out.append(f if abs(f) < abs(a) else a)  # the more conservative view
    return out


def grid(models: Sequence[str], lookbacks: Sequence[int], entries: Sequence[float],
         exits: Sequence[float], stop_z: float, max_hold_days: int) -> List[ModelSpec]:
    """Every parameter combination the training phase will try."""
    specs: List[ModelSpec] = []
    for name in models:
        if name not in _REGISTRY:
            raise KeyError(f"Unknown model '{name}'. Available: {available_models()}")
        for lookback in lookbacks:
            for entry in entries:
                for exit_ in exits:
                    if exit_ >= entry:
                        continue
                    specs.append(ModelSpec(
                        name=name, lookback=int(lookback), entry_z=float(entry),
                        exit_z=float(exit_), stop_z=float(stop_z),
                        max_hold_days=int(max_hold_days),
                    ))
    return specs
