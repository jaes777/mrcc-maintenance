"""Tests for the forward-testing (paper trading) harness.

The critical property under test: a bet must settle at the price recorded
before the race, never at the starting price. Settling at SP scores a bet
you could not have placed, and would make every forward test look better
than reality.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from ausform.analyse import RaceAnalysis, RunnerAssessment
from ausform.betting.staking import Stake
from ausform.paper import PaperTrader
from ausform.tracks import FLEMINGTON
from ausform.types import Race, Runner


def _race(race_id="R1", start_hour=14) -> Race:
    race = Race(race_id=race_id, track=FLEMINGTON, date=_dt.date(2024, 6, 1),
                race_number=1, distance_m=1200,
                start_time=_dt.datetime(2024, 6, 1, start_hour, 0,
                                        tzinfo=_dt.timezone.utc))
    race.runners = [
        Runner(horse_id="A", name="Alpha", number=1, fixed_win_odds=3.00,
               fixed_place_odds=1.50),
        Runner(horse_id="B", name="Bravo", number=2, fixed_win_odds=5.00,
               fixed_place_odds=2.00),
        Runner(horse_id="C", name="Charlie", number=3, fixed_win_odds=9.00,
               fixed_place_odds=3.00),
    ]
    return race


def _analysis(race: Race, stake_on: str | None = "#1 Alpha",
              stake_amount: float = 10.0, price: float = 3.00) -> RaceAnalysis:
    assessments = [
        RunnerAssessment(
            number=r.number, name=r.name, barrier=None, jockey=None,
            model_win_prob=p, market_win_prob=1 / (r.fixed_win_odds or 1),
            model_place_prob=min(0.99, p * 2.2),
            win_odds=r.fixed_win_odds, place_odds=r.fixed_place_odds,
            win_edge=0.1, place_edge=0.05,
            fair_win_odds=1 / p, rank=i + 1)
        for i, (r, p) in enumerate(zip(race.runners, [0.45, 0.30, 0.25]))
    ]
    stakes = ([Stake(selection=stake_on, bet_type="win", odds=price,
                     probability=0.45, edge=0.35, kelly=0.1,
                     amount=stake_amount)]
              if stake_on else [])
    return RaceAnalysis(race=race, assessments=assessments, stakes=stakes,
                        n_places=0, market_overround=0.18)


def test_records_every_runner_not_just_bets(tmp_path):
    """The full field is what tells you whether the probabilities are any
    good, and you learn that far sooner than you learn about profit."""
    with PaperTrader(tmp_path / "p.db") as trader:
        race = _race()
        assert trader.record(_analysis(race)) == 3
        summary = trader.summary()
        assert summary.predictions == 3
        assert summary.bets == 1


def test_settles_at_recorded_price_not_starting_price(tmp_path):
    """THE test. The horse is backed at $3.00 and starts at $10.00; the
    payout must reflect $3.00."""
    race = _race()
    with PaperTrader(tmp_path / "p.db") as trader:
        trader.record(_analysis(race, stake_amount=10.0, price=3.00),
                      decision_time=_dt.datetime(2024, 6, 1, 12, 0,
                                                 tzinfo=_dt.timezone.utc))

        # The race is run: our horse wins, but its price blew out to $10.
        race.runners[0].result_position = 1
        race.runners[0].fixed_win_odds = 10.00
        race.runners[1].result_position = 2
        race.runners[2].result_position = 3
        trader.settle_race(race)

        summary = trader.summary()
        assert summary.settled_bets == 1
        assert summary.returned == pytest.approx(30.0), (
            "payout must use the $3.00 recorded at decision time, not the "
            "$10.00 starting price")
        assert summary.profit == pytest.approx(20.0)


def test_losing_bet_returns_nothing(tmp_path):
    race = _race()
    with PaperTrader(tmp_path / "p.db") as trader:
        trader.record(_analysis(race, stake_amount=10.0))
        race.runners[0].result_position = 3
        race.runners[1].result_position = 1
        race.runners[2].result_position = 2
        trader.settle_race(race)

        summary = trader.summary()
        assert summary.returned == 0.0
        assert summary.profit == pytest.approx(-10.0)
        assert summary.strike_rate == 0.0


def test_price_drift_is_measured(tmp_path):
    """The gap between decision price and SP says whether the market moved
    against you -- a useful early signal that edges are stale."""
    race = _race()
    with PaperTrader(tmp_path / "p.db") as trader:
        trader.record(_analysis(race))
        for runner, sp, pos in zip(race.runners, [6.00, 10.00, 18.00], [1, 2, 3]):
            runner.fixed_win_odds = sp        # every price doubled
            runner.result_position = pos
        trader.settle_race(race)

        summary = trader.summary()
        # log(6/3) = log(2) ~ 0.693, i.e. prices drifted.
        assert summary.mean_price_drift == pytest.approx(0.693, abs=0.02)


def test_pending_tracks_unsettled_races(tmp_path):
    with PaperTrader(tmp_path / "p.db") as trader:
        trader.record(_analysis(_race("R1")))
        trader.record(_analysis(_race("R2")))
        assert len(trader.pending_races()) == 2

        settled = _race("R1")
        for runner, pos in zip(settled.runners, [1, 2, 3]):
            runner.result_position = pos
        trader.settle_race(settled)

        pending = trader.pending_races()
        assert [race_id for race_id, _ in pending] == ["R2"]


def test_warns_when_decision_is_not_before_the_race(tmp_path, caplog):
    """Recording a 'prediction' after the jump is not a forward test."""
    race = _race(start_hour=12)
    with PaperTrader(tmp_path / "p.db") as trader:
        with caplog.at_level("WARNING"):
            trader.record(_analysis(race),
                          decision_time=_dt.datetime(2024, 6, 1, 13, 0,
                                                     tzinfo=_dt.timezone.utc))
    assert any("precedes the race" in record.getMessage()
               for record in caplog.records)


def test_naive_and_aware_datetimes_do_not_crash(tmp_path):
    """Feeds are inconsistent about timezones; the honesty check must not
    be the thing that breaks."""
    race = _race()
    race.start_time = _dt.datetime(2024, 6, 1, 14, 0)   # naive
    with PaperTrader(tmp_path / "p.db") as trader:
        assert trader.record(_analysis(race),
                             decision_time=_dt.datetime(2024, 6, 1, 12, 0)) == 3


def test_verdict_refuses_to_celebrate_a_small_sample(tmp_path):
    race = _race()
    with PaperTrader(tmp_path / "p.db") as trader:
        trader.record(_analysis(race, stake_amount=10.0))
        race.runners[0].result_position = 1
        race.runners[1].result_position = 2
        race.runners[2].result_position = 3
        trader.settle_race(race)

        verdict = trader.summary().verdict()
        assert "too few" in verdict.lower()


def test_verdict_when_nothing_qualified(tmp_path):
    with PaperTrader(tmp_path / "p.db") as trader:
        trader.record(_analysis(_race(), stake_on=None))
        verdict = trader.summary().verdict()
        assert "no settled bets" in verdict.lower()
        assert "working" in verdict.lower()
