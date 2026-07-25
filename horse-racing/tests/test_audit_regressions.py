"""Regression tests for bugs found by independent adversarial review.

Each test here corresponds to a specific defect that was shipped, found by
an audit, and fixed. They exist so those defects cannot come back quietly.
Several of them are the kind that make a betting model look *better* than
it is, which is the dangerous direction.
"""

from __future__ import annotations

import datetime as _dt
import math

import numpy as np
import pytest
from scipy.optimize import minimize

from ausform.backtest.metrics import (
    RacePrediction,
    brier_score,
    comparable_subset,
    count_broken,
    log_loss,
    yield_by_threshold,
)
from ausform.betting.odds import devig, devig_with_method
from ausform.betting.staking import kelly_multiple_runners
from ausform.data.simulator import SeasonSimulator
from ausform.features import RollingContext, build_features
from ausform.model.blend import BlendedRace, MarketBlend
from ausform.types import PastRun, Runner


# --------------------------------------------------------------------------
# staking: the joint-Kelly viability filter
# --------------------------------------------------------------------------

def _log_growth(f, p, o) -> float:
    f = np.asarray(f, dtype=float)
    total = f.sum()
    reserve = 1.0 - total
    wealth = reserve + f * np.asarray(o, dtype=float)
    if reserve <= 1e-12 or np.any(wealth <= 1e-12):
        return -1e9
    return float(np.asarray(p) @ np.log(wealth)
                 + max(0.0, 1.0 - sum(p)) * math.log(reserve))


def _reference_optimum(p, o, cap=0.5, starts=25):
    best = None
    rng = np.random.default_rng(0)
    for _ in range(starts):
        start = rng.random(len(p)) * cap / len(p)
        result = minimize(lambda f: -_log_growth(f, p, o), start, method="SLSQP",
                          bounds=[(0.0, cap)] * len(p),
                          constraints=[{"type": "ineq",
                                        "fun": lambda f: cap - f.sum()}],
                          options={"maxiter": 900, "ftol": 1e-14})
        if result.success and (best is None
                               or -result.fun > _log_growth(best, p, o)):
            best = np.maximum(0.0, result.x)
    return best


def test_joint_kelly_includes_hedges_without_standalone_edge():
    """A runner with p*odds < 1 can still belong in the portfolio.

    The inclusion condition is p_i*a_i > W0, the unbet reserve, which is
    below 1 once you bet anything. Filtering at 1 dropped these runners and
    mis-sized the survivors. This case was sub-optimal by 0.022 of log
    growth before the fix.
    """
    p, o = [0.8639, 0.0829], [2.968, 4.598]
    assert p[1] * o[1] < 1.0, "the hedge must have no standalone edge"

    stakes = kelly_multiple_runners(p, o, max_total=0.9)
    assert stakes[1] > 0.0, "the hedge was excluded from the portfolio"

    reference = _reference_optimum(p, o, cap=0.9)
    assert _log_growth(stakes, p, o) >= _log_growth(reference, p, o) - 1e-9


def test_joint_kelly_matches_reference_optimum_across_many_races():
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(60):
        n = int(rng.integers(2, 7))
        p = rng.random(n)
        p = p / p.sum() * rng.uniform(0.85, 1.0)
        o = 1 + rng.random(n) * 8
        stakes = kelly_multiple_runners(list(p), list(o), max_total=0.5)
        reference = _reference_optimum(list(p), list(o), cap=0.5)
        worst = max(worst, _log_growth(reference, p, o) - _log_growth(stakes, p, o))
    assert worst < 1e-6, f"joint Kelly is sub-optimal by {worst:.2e}"


def test_joint_kelly_still_declines_an_edgeless_race():
    assert np.all(kelly_multiple_runners([0.2, 0.2, 0.2], [3.0, 3.0, 4.0]) == 0.0)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _prediction(probs, winner, odds=None, market=None, race_id="r"):
    probs = np.asarray(probs, dtype=float)
    return RacePrediction(
        race_id=race_id, date="2024-01-01", probabilities=probs,
        market_probabilities=None if market is None else np.asarray(market, float),
        odds=np.asarray(odds if odds is not None else [2.0] * len(probs), float),
        winner_index=winner, field_size=len(probs))


def test_log_loss_does_not_hide_zero_probability_winners():
    """Declaring the winner impossible is the worst thing a model can do.

    Those races were previously skipped, which deleted the model's
    catastrophic failures from its own scorecard.
    """
    good = [_prediction([0.5, 0.5], 0, race_id=f"g{i}") for i in range(10)]
    disaster = [_prediction([0.0, 1.0], 0, race_id="bad")]

    without = log_loss(good)
    with_disaster = log_loss(good + disaster)

    assert with_disaster > without, (
        "a race where the model gave the winner zero chance did not worsen "
        "the score -- it is being silently dropped")
    assert with_disaster > 3.0


def test_brier_does_not_score_broken_predictions_as_merely_wrong():
    """An all-NaN prediction is broken, not 'a bit wrong'. Substituting
    zeros scored it 1.0, better than an honest uniform guess."""
    broken = [_prediction([np.nan, np.nan, np.nan, np.nan], 0)]
    assert math.isnan(brier_score(broken))
    assert count_broken(broken) == 1


def test_bets_needed_is_not_circular():
    """It must not be derived from the realised yield.

    The old formula reduced to 4n/t^2, so it always claimed you already had
    enough evidence exactly when the sample got lucky.
    """
    def sample(win_rate: float, tag: str):
        # Back the $6.00 runner every time; vary only how often it wins, so
        # the two samples differ in realised yield but not much in variance.
        predictions = []
        for i in range(400):
            won = (i % 100) < int(win_rate * 100)
            predictions.append(
                _prediction([0.5, 0.5], 0 if won else 1,
                            odds=[6.0, 1.2], race_id=f"{tag}{i}"))
        return yield_by_threshold(predictions, thresholds=(0.0,))[0]

    lucky = sample(0.30, "w")     # +80% yield: a huge lucky run
    unlucky = sample(0.14, "l")   # -16% yield

    assert lucky.yield_pct > unlucky.yield_pct + 0.5
    assert lucky.profit_sd > 0 and unlucky.profit_sd > 0

    # Both samples are 400 bets with similar payout variance, so the
    # required sample size must be similar -- and far above 400. The old
    # formula reduced to 4n/t^2 and handed the lucky sample a number BELOW
    # its own bet count, i.e. "you already have proof".
    assert lucky.bets_needed(0.03) > 400
    assert unlucky.bets_needed(0.03) > 400
    ratio = lucky.bets_needed(0.03) / unlucky.bets_needed(0.03)
    assert 0.4 < ratio < 2.5, (
        "required sample size still depends strongly on how lucky the "
        "sample happened to be")


def test_report_compares_model_and_market_on_the_same_races():
    predictions = [
        _prediction([0.5, 0.3, 0.2], 0, market=[0.4, 0.4, 0.2], race_id="priced"),
        _prediction([0.9, 0.1], 0, market=None, race_id="unpriced"),
    ]
    shared = comparable_subset(predictions)
    assert [p.race_id for p in shared] == ["priced"]


# --------------------------------------------------------------------------
# features and context
# --------------------------------------------------------------------------

def test_context_reads_do_not_create_entries():
    """Querying the context must not grow it, or summary() lies and memory
    grows without bound."""
    context = RollingContext()
    for name in ("A", "B", "C"):
        context.jockey_win_rate(name)
        context.trainer_win_rate(name)
        context.sire_wet_win_rate(name)
    context.barrier_win_rate("FLEM", 1200, 5)

    assert context.summary()["jockeys_seen"] == 0
    assert context.summary()["trainers_seen"] == 0


def test_runs_before_returns_most_recent_first():
    """Every feature assumes history[0] is the last start. A provider
    returning oldest-first would silently invert them all."""
    oldest = _dt.date(2024, 1, 1)
    newest = _dt.date(2024, 6, 1)
    runner = Runner(
        horse_id="h", name="h", number=1,
        history=[  # deliberately supplied oldest-first
            PastRun(date=oldest, track_code="FLEM", distance_m=1200),
            PastRun(date=_dt.date(2024, 3, 1), track_code="FLEM", distance_m=1400),
            PastRun(date=newest, track_code="FLEM", distance_m=1600),
        ])
    history = runner.runs_before(_dt.date(2024, 7, 1))
    assert [r.date for r in history] == [newest, _dt.date(2024, 3, 1), oldest]


# --------------------------------------------------------------------------
# blend
# --------------------------------------------------------------------------

def test_blend_reports_when_it_was_never_fitted():
    """Default weights must not be presented as estimates."""
    blend = MarketBlend()
    blend.fit([])                      # nothing to fit on
    assert blend.fitted is False
    assert blend.n_races_fitted == 0
    assert "not fitted" in blend.describe()


def test_blend_marks_itself_fitted_on_adequate_data():
    rng = np.random.default_rng(3)
    races = []
    for _ in range(600):
        size = int(rng.integers(6, 12))
        market = rng.random(size) ** 2
        market = market / market.sum()
        fundamental = rng.random(size)
        fundamental = fundamental / fundamental.sum()
        races.append(BlendedRace(fundamental=fundamental, market=market,
                                 winner_index=int(rng.choice(size, p=market))))
    blend = MarketBlend().fit(races)
    assert blend.fitted is True
    assert blend.n_races_fitted == 600


# --------------------------------------------------------------------------
# odds
# --------------------------------------------------------------------------

def test_devig_reports_silent_fallback_to_proportional():
    """Shin degrades to proportional on an underround book. That is fine,
    but it silently changes the market baseline, so it must be reportable."""
    underround = [2.9, 4.6, 7.4, 11.0, 18.0, 30.0]
    _, method = devig_with_method(underround, "shin")
    assert method == "proportional"

    overround = [2.2, 3.4, 5.5, 8.0, 13.0, 21.0, 34.0, 51.0]
    _, method = devig_with_method(overround, "shin")
    assert method == "shin"


def test_power_handles_heavily_underround_books():
    """The old hard lower bracket of 0.5 rejected books needing k < 0.5."""
    probs = devig([8.0, 30.0, 100.0], "power")
    proportional = devig([8.0, 30.0, 100.0], "proportional")
    assert probs.sum() == pytest.approx(1.0)
    assert not np.allclose(probs, proportional), (
        "power silently fell back to proportional on an underround book")


# --------------------------------------------------------------------------
# simulator realism
# --------------------------------------------------------------------------

def test_weight_is_not_a_readout_of_latent_ability():
    """Weight was previously derived from `horse.ability`, giving a single
    feature correlating 0.96 with the hidden truth and making the whole
    evaluation vacuous. It must now come only from the public record."""
    simulator = SeasonSimulator(seed=11, n_horses=3000)
    races = simulator.simulate_season(
        _dt.date(2022, 1, 1), days=60, meetings_per_day=3, races_per_meeting=8)

    ability = {h.horse_id: h.ability for h in simulator.horses}
    context = RollingContext()
    weights, abilities = [], []
    for race in races:
        features = build_features(race, context)
        context.observe(race)
        column = features.names.index("weight_kg")
        for row, runner in enumerate(features.runners):
            weights.append(features.matrix[row, column])
            abilities.append(ability[runner.horse_id])

    weights = np.array(weights)
    abilities = np.array(abilities)
    mask = np.isfinite(weights) & np.isfinite(abilities)
    correlation = abs(np.corrcoef(weights[mask], abilities[mask])[0, 1])

    assert correlation < 0.6, (
        f"weight_kg correlates {correlation:.2f} with latent ability; it is "
        f"leaking the simulator's hidden truth into a feature")


def test_simulated_market_has_realistic_favourite_longshot_gradient():
    """Blind flat betting must return WORSE on longshots than on
    short-priced runners, as it does in reality. An earlier calibration had
    market noise swamping the bias and produced the opposite gradient,
    which would have rewarded a model for backing outsiders."""
    simulator = SeasonSimulator(seed=31, n_horses=7000)
    races = simulator.simulate_season(
        _dt.date(2022, 1, 1), days=110, meetings_per_day=4, races_per_meeting=8)

    def band_return(low, high):
        staked = returned = 0.0
        for race in races:
            live = [r for r in race.runners if r.fixed_win_odds]
            if len(live) < 5:
                continue
            probs = devig([r.fixed_win_odds for r in live], "shin")
            for index, runner in enumerate(live):
                if low <= probs[index] < high:
                    staked += 1.0
                    if runner.result_position == 1:
                        returned += runner.fixed_win_odds
        return (returned - staked) / staked if staked else float("nan")

    longshots = band_return(0.02, 0.05)
    short_priced = band_return(0.30, 0.60)

    assert longshots < short_priced - 0.05, (
        f"longshots returned {longshots:.1%} vs {short_priced:.1%} on short "
        f"prices; the favourite-longshot bias is inverted")
    assert longshots < -0.15
