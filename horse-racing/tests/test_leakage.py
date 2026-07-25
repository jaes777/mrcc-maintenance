"""Leakage tests.

These are the most important tests in the suite. A racing model that leaks
future information looks excellent and loses money, and the failure is
silent. Everything here exists to make that failure loud instead.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pytest

from ausform.data.simulator import SeasonSimulator
from ausform.features import RollingContext, build_features
from ausform.types import PastRun, Race, Runner
from ausform.tracks import FLEMINGTON


def _make_runner(horse_id: str, history_dates: list[_dt.date]) -> Runner:
    return Runner(
        horse_id=horse_id,
        name=horse_id,
        number=1,
        history=[
            PastRun(date=d, track_code="FLEM", distance_m=1200,
                    finish_position=1, field_size=10)
            for d in history_dates
        ],
    )


def test_runs_before_excludes_same_day_and_future():
    """A horse's form must not include the race being predicted."""
    race_date = _dt.date(2024, 6, 1)
    runner = _make_runner("H1", [
        _dt.date(2024, 5, 1),   # past  -> visible
        race_date,              # today -> MUST be hidden
        _dt.date(2024, 7, 1),   # future-> MUST be hidden
    ])
    visible = runner.runs_before(race_date)
    assert len(visible) == 1
    assert visible[0].date == _dt.date(2024, 5, 1)


def test_context_rejects_out_of_order_races():
    """Processing races out of date order must raise, not silently leak."""
    context = RollingContext()
    later = Race(race_id="B", track=FLEMINGTON, date=_dt.date(2024, 6, 2),
                 race_number=1, distance_m=1200)
    earlier = Race(race_id="A", track=FLEMINGTON, date=_dt.date(2024, 6, 1),
                   race_number=1, distance_m=1200)

    context.observe(later)
    with pytest.raises(ValueError, match="strict date order"):
        context.observe(earlier)


def test_jockey_strike_rate_is_point_in_time():
    """A jockey's strike rate must reflect only races already observed."""
    context = RollingContext(prior_weight=0.0)

    def race_with_winner(day: int, winner_jockey: str) -> Race:
        race = Race(race_id=f"R{day}", track=FLEMINGTON,
                    date=_dt.date(2024, 6, day), race_number=1, distance_m=1200)
        race.runners = [
            Runner(horse_id="a", name="a", number=1, jockey=winner_jockey,
                   result_position=1),
            Runner(horse_id="b", name="b", number=2, jockey="Loser",
                   result_position=2),
        ]
        return race

    # Before observing anything, the rate is the population prior.
    assert context.jockey_win_rate("Winner") == pytest.approx(context.base_win_rate)

    context.observe(race_with_winner(1, "Winner"))
    after_one = context.jockey_win_rate("Winner")
    context.observe(race_with_winner(2, "Winner"))
    after_two = context.jockey_win_rate("Winner")

    assert after_one == pytest.approx(1.0)
    assert after_two == pytest.approx(1.0)
    assert context.jockey_win_rate("Loser") == pytest.approx(0.0)


def test_speed_baseline_excludes_current_race():
    """Winning-time baselines must not contain the race being rated.

    Otherwise a horse's speed rating is computed against a benchmark that
    includes its own performance -- a subtle, powerful leak.
    """
    context = RollingContext()
    race = Race(race_id="R1", track=FLEMINGTON, date=_dt.date(2024, 6, 1),
                race_number=1, distance_m=1200, track_condition=4)
    race.runners = [
        Runner(horse_id="a", name="a", number=1, result_position=1,
               result_time_s=70.0),
        Runner(horse_id="b", name="b", number=2, result_position=2,
               result_time_s=70.5),
    ]

    # Nothing observed yet -> no baseline exists.
    assert context.time_baseline("FLEM", 1200, "dry") is None

    build_features(race, context)
    assert context.time_baseline("FLEM", 1200, "dry") is None, (
        "building features must not populate the baseline")

    context.observe(race)
    # One sample is still below the minimum needed to standardise safely.
    assert context.time_baseline("FLEM", 1200, "dry") is None


def test_features_identical_whether_or_not_future_exists():
    """The decisive test.

    Build features for a race twice: once from a dataset that stops at that
    race, and once from a dataset that also contains everything after it.
    If any future information leaks in, the two feature matrices differ.
    """
    simulator = SeasonSimulator(seed=42, n_horses=1500)
    races = simulator.simulate_season(
        _dt.date(2023, 1, 1), days=80, meetings_per_day=2, races_per_meeting=6)
    assert len(races) > 100

    target_index = len(races) // 2
    target = races[target_index]

    def features_for(subset):
        context = RollingContext()
        result = None
        for race in subset:
            fs = build_features(race, context)
            if race.race_id == target.race_id:
                result = fs
            context.observe(race)
        return result

    truncated = features_for(races[:target_index + 1])
    complete = features_for(races)

    assert truncated is not None and complete is not None
    np.testing.assert_allclose(
        np.nan_to_num(truncated.matrix, nan=-999.0),
        np.nan_to_num(complete.matrix, nan=-999.0),
        rtol=1e-10,
        err_msg="Features changed when future races were present: LEAKAGE.",
    )


def test_simulator_private_information_is_invisible():
    """The simulator's private-information term must not be recoverable.

    It exists precisely to represent what the market knows and the model
    cannot. If any feature correlated strongly with it, the simulator would
    be handing the model an unfair advantage and every backtest number
    would be optimistic.
    """
    simulator = SeasonSimulator(seed=7, n_horses=2000, private_info_sd=0.6)
    races = simulator.simulate_season(
        _dt.date(2023, 1, 1), days=40, meetings_per_day=2, races_per_meeting=6)

    # The private term is drawn fresh per run and never stored on the
    # Runner or PastRun, so no feature can reference it.
    runner = races[-1].runners[0]
    assert not hasattr(runner, "private")
    for past in runner.history:
        assert not hasattr(past, "private")
