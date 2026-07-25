"""Paper trading: honest forward testing against real races.

A backtest can be wrong in a hundred quiet ways. A forward test cannot,
provided one rule is obeyed:

    RECORD THE PREDICTION AND THE PRICE **BEFORE** THE RACE IS RUN,
    AND SETTLE AT THAT RECORDED PRICE.

That rule is the whole point of this module, and it is why settlement here
deliberately ignores the starting price. If you evaluate at SP you are
scoring a bet you could not have placed: the price moved after you decided,
and in racing it moved *because* of information you did not have. Late
money is informed money. A forward test settled at SP will look better than
reality and teach you the wrong lesson.

So every row stores `price_at_decision` and `decision_time`, and every
payout is computed from `price_at_decision`. The starting price is recorded
too, but only as a diagnostic -- the gap between the two tells you how much
the market moved against you, which is itself a useful signal about whether
your edges are real or just stale prices.

WHAT TO EXPECT
--------------
Most days will produce no bets. That is correct behaviour, not a fault.
And a few weeks of forward testing cannot prove an edge -- it takes on the
order of 14,000 bets to distinguish a genuine 5% yield from luck. What a
few weeks CAN do, and what this is for, is catch the things that are
actually catchable in a short window:

  * the pipeline breaking on real data shapes
  * probabilities that are badly calibrated against real outcomes
  * the model systematically disagreeing with the market in one direction
  * prices that have vanished by the time you would have bet

Treat the running yield as noise until the bet count is in the thousands.
Treat the calibration curve as informative much sooner.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .analyse import RaceAnalysis
from .types import Race

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_runs (
    race_id            TEXT NOT NULL,
    horse_id           TEXT NOT NULL,
    race_date          TEXT NOT NULL,
    track              TEXT,
    race_number        INTEGER,
    horse_name         TEXT,
    saddlecloth        INTEGER,

    -- Everything below is recorded strictly before the race is run.
    decision_time      TEXT NOT NULL,
    model_win_prob     REAL,
    market_win_prob    REAL,
    model_place_prob   REAL,
    price_at_decision  REAL,
    place_price_at_decision REAL,
    edge               REAL,
    bet_type           TEXT,
    stake              REAL DEFAULT 0,

    -- Filled in after the race.
    settled            INTEGER DEFAULT 0,
    result_position    INTEGER,
    starting_price     REAL,
    payout             REAL,
    profit             REAL,
    settled_at         TEXT,

    PRIMARY KEY (race_id, horse_id, bet_type)
);
CREATE INDEX IF NOT EXISTS idx_paper_date    ON paper_runs(race_date);
CREATE INDEX IF NOT EXISTS idx_paper_settled ON paper_runs(settled);
"""


@dataclass
class PaperSummary:
    """Running performance of the forward test."""

    predictions: int
    settled_predictions: int
    bets: int
    settled_bets: int
    staked: float
    returned: float
    wins: int

    log_loss: Optional[float] = None
    market_log_loss: Optional[float] = None
    top1_accuracy: Optional[float] = None
    market_top1_accuracy: Optional[float] = None
    calibration: list[tuple[str, int, float, float]] = None
    mean_price_drift: Optional[float] = None
    profits: list[float] = None

    @property
    def profit(self) -> float:
        return self.returned - self.staked

    @property
    def yield_pct(self) -> Optional[float]:
        return self.profit / self.staked if self.staked else None

    @property
    def strike_rate(self) -> Optional[float]:
        return self.wins / self.settled_bets if self.settled_bets else None

    def yield_interval(self, confidence: float = 0.95,
                       n_resamples: int = 2000, seed: int = 0
                       ) -> tuple[Optional[float], Optional[float]]:
        """Bootstrap interval for the running yield."""
        if not self.profits or len(self.profits) < 20:
            return (None, None)
        rng = np.random.default_rng(seed)
        sample = np.asarray(self.profits, dtype=float)
        draws = rng.choice(sample, size=(n_resamples, len(sample)), replace=True)
        means = draws.mean(axis=1)
        tail = (1.0 - confidence) / 2.0
        return (float(np.quantile(means, tail)),
                float(np.quantile(means, 1.0 - tail)))

    def verdict(self) -> str:
        """A deliberately deflationary reading of the numbers so far."""
        if self.settled_bets == 0:
            return ("No settled bets yet. If this persists, that is the tool "
                    "working: with a 15-30% overround, most races genuinely "
                    "contain no value.")
        low, high = self.yield_interval()
        if low is None:
            return (f"{self.settled_bets} settled bets -- far too few to read "
                    f"anything into the yield. Keep going.")
        if low <= 0 <= high:
            return (f"Yield {self.yield_pct:+.1%}, but the 95% interval spans "
                    f"{low:+.1%} to {high:+.1%}. That includes zero, so this is "
                    f"consistent with having no edge at all. Nothing is proven.")
        if high < 0:
            return (f"Yield {self.yield_pct:+.1%}, interval {low:+.1%} to "
                    f"{high:+.1%}, entirely below zero. On this evidence the "
                    f"strategy is losing money. Stop and investigate.")
        return (f"Yield {self.yield_pct:+.1%}, interval {low:+.1%} to "
                f"{high:+.1%}, entirely above zero. Encouraging, but "
                f"{self.settled_bets} bets is still a small sample for a claim "
                f"this expensive to be wrong about -- keep testing before "
                f"changing how you stake.")


class PaperTrader:
    """Records pre-race predictions and settles them against real results."""

    def __init__(self, path: str | Path = "paper.db"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "PaperTrader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- recording ---------------------------------------------------------

    def record(self, analysis: RaceAnalysis,
               decision_time: Optional[_dt.datetime] = None) -> int:
        """Log a race's assessment before it is run.

        Every runner is recorded, not only the ones we would bet. Betting
        rows tell you about profit; the full field tells you whether the
        probabilities are any good, which you learn from far sooner.
        """
        race = analysis.race
        now = decision_time or _dt.datetime.now(_dt.timezone.utc)

        # Feeds are inconsistent about timezone awareness, so normalise
        # before comparing rather than letting a naive/aware mismatch
        # crash the one check that keeps the forward test honest.
        start = race.start_time
        if start is not None and start.tzinfo is None:
            start = start.replace(tzinfo=_dt.timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=_dt.timezone.utc)

        if start is not None and now >= start:
            log.warning(
                "Recording %s at %s, at or after its %s start time. A forward "
                "test is only meaningful if the decision precedes the race.",
                race.race_id, now.isoformat(), race.start_time.isoformat())

        staked = {(s.selection, s.bet_type): s.amount for s in analysis.stakes}
        rows = []
        for assessment in analysis.assessments:
            key = f"#{assessment.number} {assessment.name}"
            for bet_type in ("win", "place"):
                amount = staked.get((key, bet_type), 0.0)
                price = (assessment.win_odds if bet_type == "win"
                         else assessment.place_odds)
                # Only keep a place row if we actually staked it, to avoid
                # doubling the size of the prediction table for nothing.
                if bet_type == "place" and amount <= 0:
                    continue
                rows.append((
                    race.race_id,
                    assessment.name.upper(),
                    race.date.isoformat(),
                    race.track.name,
                    race.race_number,
                    assessment.name,
                    assessment.number,
                    now.isoformat(),
                    assessment.model_win_prob,
                    assessment.market_win_prob,
                    assessment.model_place_prob,
                    price,
                    assessment.place_odds,
                    assessment.win_edge if bet_type == "win" else assessment.place_edge,
                    bet_type,
                    amount,
                ))

        self.conn.executemany(
            """INSERT OR REPLACE INTO paper_runs (
                   race_id, horse_id, race_date, track, race_number,
                   horse_name, saddlecloth, decision_time, model_win_prob,
                   market_win_prob, model_place_prob, price_at_decision,
                   place_price_at_decision, edge, bet_type, stake
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows)
        self.conn.commit()
        return len(rows)

    # -- settlement --------------------------------------------------------

    def settle_race(self, race: Race) -> int:
        """Settle a recorded race against its actual result.

        Payouts use `price_at_decision`, never the starting price. The SP is
        stored alongside purely so you can see how far the market moved.
        """
        if not race.is_resulted:
            return 0

        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        n_places = _places_for(race)
        settled = 0

        for runner in race.runners:
            for bet_type in ("win", "place"):
                row = self.conn.execute(
                    """SELECT * FROM paper_runs
                       WHERE race_id = ? AND horse_id = ? AND bet_type = ?
                         AND settled = 0""",
                    (race.race_id, runner.name.upper(), bet_type)).fetchone()
                if row is None:
                    continue

                position = runner.result_position
                price = row["price_at_decision"]
                stake = row["stake"] or 0.0

                won = (position == 1 if bet_type == "win"
                       else position is not None and n_places > 0
                       and position <= n_places)

                if stake > 0 and price and price > 1.0 and won:
                    payout = stake * price
                else:
                    payout = 0.0

                self.conn.execute(
                    """UPDATE paper_runs
                       SET settled = 1, result_position = ?, starting_price = ?,
                           payout = ?, profit = ?, settled_at = ?
                       WHERE race_id = ? AND horse_id = ? AND bet_type = ?""",
                    (position, runner.fixed_win_odds, payout, payout - stake,
                     now, race.race_id, runner.name.upper(), bet_type))
                settled += 1

        self.conn.commit()
        return settled

    def settle_many(self, races: Iterable[Race]) -> int:
        return sum(self.settle_race(race) for race in races)

    def pending_races(self) -> list[tuple[str, str]]:
        """(race_id, race_date) for everything recorded but not yet settled."""
        rows = self.conn.execute(
            """SELECT DISTINCT race_id, race_date FROM paper_runs
               WHERE settled = 0 ORDER BY race_date""").fetchall()
        return [(r["race_id"], r["race_date"]) for r in rows]

    # -- reporting ---------------------------------------------------------

    def summary(self, since: Optional[_dt.date] = None) -> PaperSummary:
        clause, params = "", []
        if since:
            clause = "WHERE race_date >= ?"
            params = [since.isoformat()]

        rows = self.conn.execute(
            f"SELECT * FROM paper_runs {clause}", params).fetchall()
        win_rows = [r for r in rows if r["bet_type"] == "win"]
        settled_win = [r for r in win_rows if r["settled"]]

        bet_rows = [r for r in rows if (r["stake"] or 0) > 0]
        settled_bets = [r for r in bet_rows if r["settled"]]

        staked = sum(r["stake"] for r in settled_bets)
        returned = sum(r["payout"] or 0.0 for r in settled_bets)
        wins = sum(1 for r in settled_bets if (r["payout"] or 0) > 0)
        profits = [(r["profit"] or 0.0) / r["stake"]
                   for r in settled_bets if r["stake"]]

        # Probability quality over the full field, grouped by race.
        by_race: dict[str, list[sqlite3.Row]] = {}
        for row in settled_win:
            by_race.setdefault(row["race_id"], []).append(row)

        model_ll, market_ll, model_hits, market_hits, scored = 0.0, 0.0, 0, 0, 0
        for runners in by_race.values():
            winner = [r for r in runners if r["result_position"] == 1]
            if not winner or len(runners) < 2:
                continue
            scored += 1
            model_p = max(1e-15, winner[0]["model_win_prob"] or 1e-15)
            model_ll += -math.log(model_p)
            best = max(runners, key=lambda r: r["model_win_prob"] or 0.0)
            model_hits += int(best["result_position"] == 1)

            market_ps = [r["market_win_prob"] for r in runners]
            if all(p is not None for p in market_ps):
                market_ll += -math.log(max(1e-15, winner[0]["market_win_prob"]))
                fav = max(runners, key=lambda r: r["market_win_prob"] or 0.0)
                market_hits += int(fav["result_position"] == 1)

        drift = [
            math.log(r["starting_price"] / r["price_at_decision"])
            for r in settled_win
            if r["starting_price"] and r["price_at_decision"]
            and r["starting_price"] > 1 and r["price_at_decision"] > 1
        ]

        return PaperSummary(
            predictions=len(win_rows),
            settled_predictions=len(settled_win),
            bets=len(bet_rows),
            settled_bets=len(settled_bets),
            staked=staked,
            returned=returned,
            wins=wins,
            log_loss=model_ll / scored if scored else None,
            market_log_loss=market_ll / scored if scored else None,
            top1_accuracy=model_hits / scored if scored else None,
            market_top1_accuracy=market_hits / scored if scored else None,
            calibration=self._calibration(settled_win),
            mean_price_drift=float(np.mean(drift)) if drift else None,
            profits=profits,
        )

    @staticmethod
    def _calibration(rows: Sequence[sqlite3.Row]
                     ) -> list[tuple[str, int, float, float]]:
        edges = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 1.01]
        out = []
        for i in range(len(edges) - 1):
            bucket = [r for r in rows
                      if r["model_win_prob"] is not None
                      and edges[i] <= r["model_win_prob"] < edges[i + 1]]
            if not bucket:
                continue
            predicted = float(np.mean([r["model_win_prob"] for r in bucket]))
            observed = float(np.mean([1.0 if r["result_position"] == 1 else 0.0
                                      for r in bucket]))
            out.append((f"{edges[i]:.0%}-{edges[i+1]:.0%}",
                        len(bucket), predicted, observed))
        return out


def _places_for(race: Race) -> int:
    from .betting.exotics import places_paid
    return places_paid(len([r for r in race.runners if not r.scratched]))
