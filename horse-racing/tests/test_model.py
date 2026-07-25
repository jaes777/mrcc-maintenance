"""Model tests: correctness of the likelihood, and that signal is recovered."""

from __future__ import annotations

import datetime as _dt
import math

import numpy as np
import pytest

from ausform.backtest import WalkForwardBacktest, WalkForwardConfig
from ausform.backtest.metrics import RacePrediction, top1_accuracy
from ausform.data.simulator import SeasonSimulator
from ausform.model import ConditionalLogit, MarketBlend, RaceObservation
from ausform.model.blend import BlendedRace


def _synthetic_races(n_races=600, n_features=4, seed=0):
    """Races generated from a known linear model, for exact recovery tests."""
    rng = np.random.default_rng(seed)
    true_beta = np.array([1.5, -0.8, 0.4, 0.0])[:n_features]
    observations = []
    for _ in range(n_races):
        size = int(rng.integers(6, 13))
        features = rng.normal(size=(size, n_features))
        scores = features @ true_beta
        # Gumbel noise makes the argmax exactly multinomial-logit distributed.
        gumbel = rng.gumbel(size=size)
        order = np.argsort(-(scores + gumbel))
        observations.append(RaceObservation(
            features=features,
            winner_index=int(order[0]),
            finish_order=[int(i) for i in order],
        ))
    return observations, true_beta


def test_conditional_logit_recovers_known_coefficients():
    observations, true_beta = _synthetic_races(n_races=3000, seed=1)
    model = ConditionalLogit(l2=1e-4).fit(observations)
    # Features are standardised internally, so compare directions not scale.
    recovered = model.coefficients / np.linalg.norm(model.coefficients)
    expected = true_beta / np.linalg.norm(true_beta)
    np.testing.assert_allclose(recovered, expected, atol=0.12)


def test_conditional_logit_gradient_matches_numerical():
    """Analytic gradient must match finite differences, or the optimiser
    converges to the wrong place while looking fine."""
    observations, _ = _synthetic_races(n_races=60, seed=2)
    model = ConditionalLogit(l2=0.5)
    stacked = np.vstack([o.features for o in observations])
    model._standardise(stacked, learn=True)
    design = [model._standardise(o.features) for o in observations]
    orders = [[o.winner_index] for o in observations]

    def loss_and_grad(beta):
        total, grad = 0.0, np.zeros_like(beta)
        for matrix, order in zip(design, orders):
            remaining = list(range(matrix.shape[0]))
            for position in order:
                sub = matrix[remaining]
                scores = sub @ beta
                peak = scores.max()
                exp_scores = np.exp(scores - peak)
                denominator = exp_scores.sum()
                probs = exp_scores / denominator
                local = remaining.index(position)
                total += scores[local] - (peak + np.log(denominator))
                grad += sub[local] - probs @ sub
                remaining.remove(position)
        total -= 0.5 * float(beta @ beta)
        grad -= 2 * 0.5 * beta
        return -total, -grad

    beta = np.array([0.3, -0.2, 0.5, 0.1])
    _, analytic = loss_and_grad(beta)

    numerical = np.zeros_like(beta)
    eps = 1e-6
    for i in range(len(beta)):
        up, down = beta.copy(), beta.copy()
        up[i] += eps
        down[i] -= eps
        numerical[i] = (loss_and_grad(up)[0] - loss_and_grad(down)[0]) / (2 * eps)

    np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-6)


def test_probabilities_sum_to_one_for_any_field_size():
    observations, _ = _synthetic_races(n_races=300, seed=3)
    model = ConditionalLogit().fit(observations)
    rng = np.random.default_rng(9)
    for size in (2, 5, 8, 14, 24):
        probs = model.predict_proba(rng.normal(size=(size, 4)))
        assert probs.sum() == pytest.approx(1.0)
        assert np.all(probs >= 0)


def test_plackett_luce_depth_uses_more_signal():
    """Fitting on the first three placings should not be worse than fitting
    on winners alone, given the same data."""
    observations, true_beta = _synthetic_races(n_races=400, seed=4)
    win_only = ConditionalLogit(l2=0.1).fit(observations, plackett_luce_depth=1)
    exploded = ConditionalLogit(l2=0.1).fit(observations, plackett_luce_depth=3)

    expected = true_beta / np.linalg.norm(true_beta)

    def error(model):
        recovered = model.coefficients / np.linalg.norm(model.coefficients)
        return float(np.linalg.norm(recovered - expected))

    assert error(exploded) <= error(win_only) + 0.05


def test_market_blend_closed_form():
    """combine() must implement p ∝ f^alpha * pi^beta."""
    blend = MarketBlend(alpha=0.4, beta=0.9)
    fundamental = np.array([0.5, 0.3, 0.2])
    market = np.array([0.4, 0.35, 0.25])

    got = blend.combine(fundamental, market)
    expected = (fundamental ** 0.4) * (market ** 0.9)
    expected = expected / expected.sum()
    np.testing.assert_allclose(got, expected, rtol=1e-9)


def test_market_blend_recovers_weights():
    """If outcomes come purely from the market probabilities, the fit should
    put the weight on the market."""
    rng = np.random.default_rng(6)
    races = []
    for _ in range(1500):
        size = int(rng.integers(6, 12))
        market = rng.random(size) ** 2
        market = market / market.sum()
        fundamental = rng.random(size)
        fundamental = fundamental / fundamental.sum()
        winner = int(rng.choice(size, p=market))
        races.append(BlendedRace(fundamental=fundamental, market=market,
                                 winner_index=winner))

    blend = MarketBlend().fit(races)
    assert blend.beta > blend.alpha
    assert blend.beta > 0.7
    assert blend.alpha < 0.3


def test_blend_falls_back_without_market():
    blend = MarketBlend(alpha=0.3, beta=0.9)
    fundamental = np.array([0.6, 0.25, 0.15])
    np.testing.assert_allclose(blend.combine(fundamental, None), fundamental)


def test_model_beats_uniform_but_not_absurdly_on_simulated_data():
    """End-to-end sanity.

    The model must beat random guessing. It must NOT wildly beat the market
    -- if it does, something is leaking, and this test is here to catch
    exactly that.
    """
    simulator = SeasonSimulator(seed=21, n_horses=6000)
    races = simulator.simulate_season(
        _dt.date(2022, 1, 1), days=260, meetings_per_day=3, races_per_meeting=8)

    config = WalkForwardConfig(train_days=150, test_days=60, min_train_races=400)
    result = WalkForwardBacktest(config).run(races)
    assert result.windows >= 1

    report = result.report()
    model_ll = report["model"]["log_loss"]
    uniform_ll = report["baselines"]["uniform_log_loss"]
    market_ll = report["baselines"]["market_log_loss"]

    assert model_ll < uniform_ll, "model is worse than random guessing"

    improvement = 100.0 * (1.0 - model_ll / market_ll)
    assert improvement > -3.0, "model is far worse than the market"
    assert improvement < 15.0, (
        f"model beats the market by {improvement:.1f}% -- implausible, "
        f"check for leakage")

    # Top-1 accuracy near the structural ceiling, not above it.
    assert 0.20 < report["model"]["top1_accuracy"] < 0.45


def test_metrics_compare_like_with_like():
    """Model and market log-loss must be computed over the same races."""
    predictions = [
        RacePrediction(
            race_id="1", date="2024-01-01",
            probabilities=np.array([0.5, 0.3, 0.2]),
            market_probabilities=np.array([0.4, 0.4, 0.2]),
            odds=np.array([2.0, 2.5, 5.0]),
            winner_index=0, field_size=3),
        RacePrediction(
            race_id="2", date="2024-01-02",
            probabilities=np.array([0.6, 0.4]),
            market_probabilities=None,     # no market for this race
            odds=np.array([1.7, 2.3]),
            winner_index=1, field_size=2),
    ]
    # Race 2 has no market, so the head-to-head must be scored on race 1
    # alone -- otherwise the two sides are averaged over different races.
    from ausform.backtest.metrics import comparable_subset, full_report

    shared = comparable_subset(predictions)
    assert [p.race_id for p in shared] == ["1"]

    report = full_report(predictions)
    assert report["comparable_races"] == 1
    # The comparison figure must come from the shared subset, and race 1's
    # model probability for the winner is 0.5.
    assert report["model"]["log_loss_on_priced_races"] == pytest.approx(
        -math.log(0.5))
    assert report["baselines"]["market_log_loss"] == pytest.approx(
        -math.log(0.4))
    assert np.isfinite(report["model"]["log_loss_improvement_vs_market_pct"])
    assert 0.0 <= top1_accuracy(predictions) <= 1.0
