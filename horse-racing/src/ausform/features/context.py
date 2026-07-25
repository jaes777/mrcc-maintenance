"""Point-in-time rolling statistics.

THE LEAKAGE PROBLEM
-------------------
The single easiest way to build a horse racing model that looks brilliant
and loses money is to leak the future into the past. A jockey's season
strike rate, a track's average winning time, a horse's "career record" --
compute any of these over the whole dataset and your backtest will be
using facts that nobody could have known on race day. The model will look
astonishing and be worthless.

This module exists to make that mistake structurally hard. It maintains
running aggregates that are only ever updated by `observe()`, which the
caller must invoke *after* extracting features for a race. Feature code
can therefore only ever see races that have already happened.

The intended usage pattern, and the only safe one:

    context = RollingContext()
    for race in races_in_strict_date_order:
        features = build_features(race, context)   # reads past only
        ...
        context.observe(race)                      # now folds it in

`observe()` refuses to accept a race dated earlier than the most recent
one it has seen, which turns an ordering mistake into an exception rather
than a silently optimistic backtest.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from ..types import Race


@dataclass
class _Tally:
    starts: int = 0
    wins: int = 0
    places: int = 0

    def rate(self, kind: str, prior_rate: float, prior_weight: float) -> float:
        """Smoothed strike rate.

        A jockey with 1 start from 1 win is not a 100% strike-rate jockey.
        We shrink towards the population base rate, with the prior counting
        as `prior_weight` pseudo-starts, so small samples behave sensibly.
        """
        hits = self.wins if kind == "win" else self.places
        denominator = self.starts + prior_weight
        if denominator <= 0:
            # No starts and no prior to fall back on. The honest answer is
            # the population base rate, not a crash and not zero.
            return prior_rate
        return (hits + prior_rate * prior_weight) / denominator


@dataclass
class _TimeStats:
    """Running mean/variance of winning times for a (track, distance, going)
    bucket, via Welford's algorithm."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    @property
    def sd(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))


# Returned by every read for an unseen jockey/trainer/sire/barrier. Shared
# and never mutated -- reads must not create dictionary entries, or merely
# querying the context would grow it and corrupt summary().
_EMPTY = _Tally()


class RollingContext:
    """Accumulates everything the feature builder needs to know about the past."""

    def __init__(self, prior_weight: float = 30.0):
        self.prior_weight = prior_weight
        self._last_date: Optional[_dt.date] = None

        self.jockeys: dict[str, _Tally] = defaultdict(_Tally)
        self.trainers: dict[str, _Tally] = defaultdict(_Tally)
        self.combos: dict[tuple[str, str], _Tally] = defaultdict(_Tally)
        self.sires: dict[str, _Tally] = defaultdict(_Tally)
        self.sires_wet: dict[str, _Tally] = defaultdict(_Tally)
        self.barriers: dict[tuple[str, int, int], _Tally] = defaultdict(_Tally)

        # Population base rates, used as shrinkage targets.
        self._total_starts = 0
        self._total_wins = 0
        self._total_places = 0

        # Winning-time baselines keyed by (track, distance, going band).
        self.times: dict[tuple[str, int, str], _TimeStats] = defaultdict(_TimeStats)

    # -- base rates --------------------------------------------------------

    @property
    def base_win_rate(self) -> float:
        if self._total_starts == 0:
            return 0.10
        return self._total_wins / self._total_starts

    @property
    def base_place_rate(self) -> float:
        if self._total_starts == 0:
            return 0.30
        return self._total_places / self._total_starts

    # -- reads (safe: only reflect observed history) -----------------------

    def jockey_win_rate(self, name: Optional[str]) -> float:
        if not name:
            return self.base_win_rate
        return self.jockeys.get(name, _EMPTY).rate(
            "win", self.base_win_rate, self.prior_weight)

    def jockey_place_rate(self, name: Optional[str]) -> float:
        if not name:
            return self.base_place_rate
        return self.jockeys.get(name, _EMPTY).rate(
            "place", self.base_place_rate, self.prior_weight)

    def jockey_starts(self, name: Optional[str]) -> int:
        return self.jockeys.get(name, _EMPTY).starts if name else 0

    def trainer_win_rate(self, name: Optional[str]) -> float:
        if not name:
            return self.base_win_rate
        return self.trainers.get(name, _EMPTY).rate(
            "win", self.base_win_rate, self.prior_weight)

    def trainer_place_rate(self, name: Optional[str]) -> float:
        if not name:
            return self.base_place_rate
        return self.trainers.get(name, _EMPTY).rate(
            "place", self.base_place_rate, self.prior_weight)

    def combo_win_rate(self, jockey: Optional[str], trainer: Optional[str]) -> float:
        if not jockey or not trainer:
            return self.base_win_rate
        # Shrink harder: jockey/trainer pairings have small samples.
        return self.combos.get((jockey, trainer), _EMPTY).rate(
            "win", self.base_win_rate, self.prior_weight * 0.5)

    def sire_wet_win_rate(self, sire: Optional[str]) -> float:
        """How the sire's progeny go on rain-affected ground.

        Wet-track aptitude is strongly heritable and is one of the few
        genuinely useful breeding signals for flat racing.
        """
        if not sire:
            return self.base_win_rate
        return self.sires_wet.get(sire, _EMPTY).rate(
            "win", self.base_win_rate, self.prior_weight * 0.7)

    def barrier_win_rate(self, track_code: str, distance_m: int, barrier: Optional[int]
                         ) -> float:
        """Historical strike rate from this gate at this track and trip.

        Distance is bucketed to 200m so the sample is not spread too thin.
        """
        if barrier is None:
            return self.base_win_rate
        key = (track_code, self._distance_bucket(distance_m), min(barrier, 20))
        return self.barriers.get(key, _EMPTY).rate(
            "win", self.base_win_rate, self.prior_weight)

    # Below this many observed winning times, a bucket's mean and standard
    # deviation are too unstable to standardise against: a spuriously small
    # sd turns an ordinary run into a +10 sigma "rating".
    MIN_TIME_SAMPLES = 8

    def time_baseline(self, track_code: str, distance_m: int, going: str
                      ) -> Optional[_TimeStats]:
        stats = self.times.get((track_code, self._distance_bucket(distance_m), going))
        if stats is None or stats.n < self.MIN_TIME_SAMPLES or stats.sd <= 1e-6:
            return None
        return stats

    @staticmethod
    def _distance_bucket(distance_m: int) -> int:
        return int(round(distance_m / 200.0) * 200)

    @staticmethod
    def _going_band(condition: Optional[int]) -> str:
        if condition is None:
            return "unknown"
        if condition <= 4:
            return "dry"
        if condition <= 7:
            return "soft"
        return "heavy"

    # -- write -------------------------------------------------------------

    def observe(self, race: Race) -> None:
        """Fold a completed race into the running aggregates.

        Must be called only after features for this race have been built.
        """
        if self._last_date is not None and race.date < self._last_date:
            raise ValueError(
                f"RollingContext received race dated {race.date} after having "
                f"already observed {self._last_date}. Races must be processed in "
                f"strict date order or the statistics will leak future "
                f"information into past predictions."
            )
        self._last_date = race.date

        if not race.is_resulted:
            return

        going = self._going_band(race.track_condition)
        bucket = self._distance_bucket(race.distance_m)
        is_wet = race.track_condition is not None and race.track_condition >= 5

        winning_time: Optional[float] = None

        for runner in race.active_runners:
            position = runner.result_position
            if position is None:
                continue
            won = position == 1
            placed = position <= 3

            self._total_starts += 1
            self._total_wins += int(won)
            self._total_places += int(placed)

            for tally in self._tallies_for(runner, race, bucket, is_wet):
                tally.starts += 1
                tally.wins += int(won)
                tally.places += int(placed)

            if won and runner.result_time_s:
                winning_time = runner.result_time_s

        if winning_time:
            self.times[(race.track.code, bucket, going)].update(winning_time)

    def _tallies_for(self, runner, race, bucket: int, is_wet: bool) -> list[_Tally]:
        tallies = [self.jockeys[runner.jockey] if runner.jockey else _Tally(),
                   self.trainers[runner.trainer] if runner.trainer else _Tally()]
        if runner.jockey and runner.trainer:
            tallies.append(self.combos[(runner.jockey, runner.trainer)])
        if runner.sire:
            tallies.append(self.sires[runner.sire])
            if is_wet:
                tallies.append(self.sires_wet[runner.sire])
        if runner.barrier is not None:
            tallies.append(
                self.barriers[(race.track.code, bucket, min(runner.barrier, 20))])
        return tallies

    def observe_winning_time(self, race: Race, seconds: float) -> None:
        """Record a winning time explicitly, for feeds that supply it
        separately from the runner history."""
        going = self._going_band(race.track_condition)
        bucket = self._distance_bucket(race.distance_m)
        self.times[(race.track.code, bucket, going)].update(seconds)

    def summary(self) -> dict:
        return {
            "races_observed_through": self._last_date.isoformat() if self._last_date else None,
            "runner_records": self._total_starts,
            "base_win_rate": round(self.base_win_rate, 4),
            "base_place_rate": round(self.base_place_rate, 4),
            "jockeys_seen": len(self.jockeys),
            "trainers_seen": len(self.trainers),
            "time_buckets": len(self.times),
        }
