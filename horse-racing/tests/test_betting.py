"""Tests for odds handling, exotics and staking."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from ausform.betting.exotics import (
    box_combinations,
    box_cost,
    FlexiBet,
    fit_discount_exponents,
    ordered_probability,
    place_probabilities,
    places_paid,
    position_probabilities,
    quinella_probability,
)
from ausform.betting.odds import (
    booksum,
    breakeven_probability,
    devig,
    decimal_to_probability,
    shin_insider_fraction,
)
from ausform.betting.staking import (
    StakingPolicy,
    bets_needed_to_prove_edge,
    drawdown_risk,
    expected_value,
    growth_rate_fraction,
    kelly_fraction,
    kelly_multiple_runners,
    size_bets,
)

BOOK = [2.2, 3.4, 5.5, 8.0, 13.0, 21.0, 34.0, 51.0]


# --- devigging ------------------------------------------------------------

@pytest.mark.parametrize("method", ["proportional", "power", "shin"])
def test_devig_sums_to_one(method):
    probs = devig(BOOK, method)
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(probs > 0)


def test_devig_corrects_favourite_longshot_bias():
    """Power and Shin must move probability toward the favourite and away
    from longshots, relative to naive proportional scaling."""
    proportional = devig(BOOK, "proportional")
    for method in ("power", "shin"):
        adjusted = devig(BOOK, method)
        assert adjusted[0] > proportional[0], f"{method} did not lift the favourite"
        assert adjusted[-1] < proportional[-1], f"{method} did not cut the longshot"


def test_shin_insider_fraction_in_plausible_range():
    z = shin_insider_fraction(BOOK)
    assert z is not None
    assert 0.0 < z < 0.15


def test_devig_handles_missing_and_degenerate_input():
    assert np.isnan(decimal_to_probability([None, 0.5, 1.0])).all()
    result = devig([2.0, None, 4.0], "shin")
    assert np.isnan(result[1])
    assert np.nansum(result) == pytest.approx(1.0)
    assert devig([], "shin").size == 0


def test_booksum_and_breakeven():
    assert booksum([2.0, 2.0]) == pytest.approx(1.0)
    assert breakeven_probability(5.0) == pytest.approx(0.2)
    # Commission on winnings raises the probability you need.
    assert breakeven_probability(5.0, 0.08) > 0.2


# --- exotics --------------------------------------------------------------

def test_places_paid_australian_rules():
    assert places_paid(12) == 3
    assert places_paid(8) == 3
    assert places_paid(7) == 2
    assert places_paid(5) == 2
    assert places_paid(4) == 0
    assert places_paid(2) == 0


def test_fast_positions_match_brute_force():
    """The vectorised closed form must equal exhaustive enumeration."""
    rng = np.random.default_rng(11)
    for _ in range(8):
        n = int(rng.integers(4, 11))
        p = rng.random(n) ** 2
        p = p / p.sum()
        for lam2, lam3 in [(1.0, 1.0), (0.85, 0.75), (0.65, 0.5)]:
            fast = position_probabilities(p, 3, lambda_2nd=lam2, lambda_3rd=lam3)
            slow = np.zeros((n, 3))
            for order in itertools.permutations(range(n), 3):
                probability = ordered_probability(p, order, lam2, lam3)
                for position, runner in enumerate(order):
                    slow[runner, position] += probability
            np.testing.assert_allclose(fast, slow, atol=1e-9)


def test_position_columns_sum_to_one():
    p = np.array([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
    positions = position_probabilities(p, 3)
    np.testing.assert_allclose(positions.sum(axis=0), np.ones(3), atol=1e-9)
    assert np.all(positions.sum(axis=1) <= 1.0 + 1e-9)


def test_lambda_one_recovers_exact_harville():
    """With no discount, the model must reduce to textbook Harville."""
    p = np.array([0.5, 0.3, 0.15, 0.05])
    for i, j in itertools.permutations(range(4), 2):
        expected = p[i] * p[j] / (1 - p[i])
        actual = ordered_probability(p, [i, j], 1.0, 1.0)
        assert actual == pytest.approx(expected, rel=1e-9)


def test_discount_moves_place_mass_off_the_favourite():
    """The whole point of the discount: Harville overstates how often a
    strong favourite fills a place."""
    # Eight starters, so three places are paid and the totals sum to 3.
    p = np.array([0.44, 0.2, 0.15, 0.1, 0.06, 0.03, 0.01, 0.01])
    harville = place_probabilities(p, len(p), lambda_2nd=1.0, lambda_3rd=1.0)
    discounted = place_probabilities(p, len(p), lambda_2nd=0.85, lambda_3rd=0.75)

    assert discounted[0] < harville[0], "favourite should get less place mass"
    assert discounted[-1] > harville[-1], "longshot should get more"
    assert discounted.sum() == pytest.approx(3.0, abs=1e-6)


def test_place_probability_respects_field_size():
    p = np.array([0.4, 0.3, 0.2, 0.1, 0.05, 0.05])
    p = p / p.sum()
    two_places = place_probabilities(p, starters=6)
    assert two_places.sum() == pytest.approx(2.0, abs=1e-6)
    assert place_probabilities(p, starters=4).sum() == pytest.approx(0.0)


@pytest.mark.parametrize("k,expected", [(3, 6), (4, 24), (5, 60), (6, 120),
                                        (7, 210), (8, 336)])
def test_boxed_trifecta_costs_match_published(k, expected):
    assert box_cost("trifecta", k) == expected


@pytest.mark.parametrize("k,expected", [(4, 24), (5, 120), (6, 360),
                                        (7, 840), (8, 1680)])
def test_boxed_first_four_costs_match_published(k, expected):
    assert box_cost("first_four", k) == expected


def test_quinella_box_is_unordered():
    for k in range(2, 10):
        assert box_combinations("quinella", k) == k * (k - 1) // 2


def test_flexi_percentage():
    flexi = FlexiBet("first_four", 6, outlay=50.0)
    assert flexi.combinations == 360
    assert flexi.percentage == pytest.approx(50.0 / 360.0)
    assert flexi.payout(2000.0) == pytest.approx(2000.0 * 50.0 / 360.0)


def test_fit_discount_exponents_recovers_known_values():
    """Generate finishing orders from known exponents and recover them."""
    rng = np.random.default_rng(5)
    true_lam2, true_lam3 = 0.80, 0.65
    races = []
    for _ in range(1200):
        n = int(rng.integers(8, 13))
        p = rng.random(n) ** 2
        p = p / p.sum()
        remaining = list(range(n))
        order = []
        for stage in range(3):
            exponent = 1.0 if stage == 0 else (true_lam2 if stage == 1 else true_lam3)
            pool = np.power(p[remaining], exponent)
            pool = pool / pool.sum()
            pick = rng.choice(len(remaining), p=pool)
            order.append(remaining.pop(pick))
        races.append((p, order))

    lam2, lam3 = fit_discount_exponents(races)
    assert lam2 == pytest.approx(true_lam2, abs=0.12)
    assert lam3 == pytest.approx(true_lam3, abs=0.15)


# --- staking --------------------------------------------------------------

def test_kelly_matches_formula_and_brute_force():
    p, odds = 0.30, 5.0
    assert kelly_fraction(p, odds) == pytest.approx((p * odds - 1) / (odds - 1))

    # Brute-force the log-growth maximum and compare.
    fractions = np.linspace(0.0, 0.6, 60001)
    growth = p * np.log(1 - fractions + fractions * odds) + (1 - p) * np.log(1 - fractions)
    assert fractions[int(np.argmax(growth))] == pytest.approx(
        kelly_fraction(p, odds), abs=1e-3)


def test_kelly_zero_without_edge():
    assert kelly_fraction(0.19, 5.0) == 0.0   # exactly break-even is 0.20
    assert kelly_fraction(0.10, 5.0) == 0.0
    assert expected_value(0.20, 5.0) == pytest.approx(0.0)


def test_kelly_commission_reduces_stake():
    assert kelly_fraction(0.30, 5.0, 0.08) < kelly_fraction(0.30, 5.0)


def test_kelly_multiple_runners_beats_independent_sizing():
    """Joint sizing must not exceed sane exposure and must maximise growth."""
    probs = [0.40, 0.25, 0.12]
    odds = [3.0, 5.0, 12.0]
    stakes = kelly_multiple_runners(probs, odds)
    assert np.all(stakes >= 0)
    assert stakes.sum() < 1.0

    def growth(f):
        total = f.sum()
        wealth = 1 - total + f * np.array(odds)
        residual = 1 - total
        if residual <= 0 or np.any(wealth <= 0):
            return -1e9
        return (np.array(probs) @ np.log(wealth)
                + (1 - sum(probs)) * math.log(residual))

    best = growth(stakes)
    rng = np.random.default_rng(3)
    for _ in range(4000):
        candidate = stakes + rng.normal(0, 0.01, size=len(stakes))
        candidate = np.clip(candidate, 0, 0.5)
        if candidate.sum() < 0.95:
            assert growth(candidate) <= best + 1e-6


def test_fractional_kelly_risk_relationships():
    assert drawdown_risk(1.0) == pytest.approx(0.5)
    assert drawdown_risk(0.5) == pytest.approx(0.125)
    assert growth_rate_fraction(0.5) == pytest.approx(0.75)
    assert growth_rate_fraction(1.0) == pytest.approx(1.0)


def test_bets_needed_matches_published_figure():
    assert bets_needed_to_prove_edge(0.05, odds_sd=3.0, t_stat=2.0) == 14400


def test_size_bets_respects_caps():
    policy = StakingPolicy(kelly_fraction=0.25, max_bet_pct=0.02,
                           max_race_pct=0.05, min_edge=0.05)
    candidates = [(f"H{i}", "win", 0.35, 4.0) for i in range(10)]
    stakes = size_bets(candidates, bankroll=1000.0, policy=policy)
    assert sum(s.amount for s in stakes) <= 0.05 * 1000.0 + 1e-6
    assert all(s.amount <= 0.02 * 1000.0 + 1e-6 for s in stakes)


def test_size_bets_skips_thin_edges():
    policy = StakingPolicy(min_edge=0.05)
    # 0.21 * 5.0 - 1 = 0.05 edge exactly at threshold; 0.205 is below.
    assert size_bets([("H", "win", 0.205, 5.0)], 1000.0, policy) == []
