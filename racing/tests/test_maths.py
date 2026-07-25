"""Tests for the parts that must be exactly right.

The probability and money arithmetic is provable, so it is tested against
closed-form answers and known identities rather than against a fixture. If
these pass, the maths is correct regardless of what any data source does.
"""

import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ausrace import betting, exotics, tote  # noqa: E402
from ausrace.schema import track_condition_band, track_condition_to_number  # noqa: E402


# --------------------------------------------------------------------------
# Ordering models
# --------------------------------------------------------------------------

def test_harville_orderings_sum_to_one():
    """Every possible finishing order must have total probability 1."""
    p = np.array([0.35, 0.25, 0.20, 0.12, 0.08])
    total = sum(exotics.harville_order_prob(p, order)
                for order in permutations(range(5)))
    assert total == pytest.approx(1.0, abs=1e-9)


def test_discounted_harville_orderings_sum_to_one():
    """The discount must not break normalisation."""
    p = np.array([0.40, 0.22, 0.18, 0.12, 0.08])
    total = sum(exotics.order_probability(p, order, exotics.DEFAULT_LAMBDAS)
                for order in permutations(range(5)))
    assert total == pytest.approx(1.0, abs=1e-9)


def test_lambda_one_recovers_harville():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    for order in [(0, 1, 2), (2, 0, 3), (3, 2, 1)]:
        assert exotics.order_probability(p, order, (1.0, 1.0, 1.0, 1.0)) == pytest.approx(
            exotics.harville_order_prob(p, order), rel=1e-12
        )


def test_harville_exacta_closed_form():
    """P(i 1st, j 2nd) = p_i * p_j / (1 - p_i)."""
    p = np.array([0.5, 0.3, 0.2])
    matrix = exotics.exacta_matrix(p, lambdas=(1.0, 1.0, 1.0, 1.0))
    assert matrix[0, 1] == pytest.approx(0.5 * 0.3 / 0.5)
    assert matrix[2, 0] == pytest.approx(0.2 * 0.5 / 0.8)
    assert matrix[1, 1] == 0.0


def test_position_matrix_rows_and_columns():
    """Each position's probabilities sum to 1 across horses; each horse's
    probabilities across positions cannot exceed 1."""
    p = np.array([0.30, 0.22, 0.18, 0.12, 0.10, 0.08])
    for lambdas in [(1.0, 1.0, 1.0, 1.0), exotics.DEFAULT_LAMBDAS]:
        positions = exotics.position_matrix(p, max_position=3, lambdas=lambdas)
        for col in range(3):
            assert positions[:, col].sum() == pytest.approx(1.0, abs=1e-6)
        assert (positions.sum(axis=1) <= 1.0 + 1e-9).all()


def test_discount_reduces_favourite_place_probability():
    """The whole point of the discount: favourites place less often than plain
    Harville predicts, and longshots place more."""
    p = np.array([0.50, 0.15, 0.12, 0.10, 0.07, 0.04, 0.02])
    plain = exotics.position_matrix(p, 3, lambdas=(1.0, 1.0, 1.0, 1.0)).sum(axis=1)
    discounted = exotics.position_matrix(p, 3, lambdas=exotics.DEFAULT_LAMBDAS).sum(axis=1)
    assert discounted[0] < plain[0]      # favourite
    assert discounted[-1] > plain[-1]    # longshot


def test_monte_carlo_matches_closed_form():
    """Sampling must agree with the exact calculation."""
    p = np.array([0.35, 0.25, 0.20, 0.12, 0.08])
    orders = exotics.simulate_orders(p, n_sims=200_000, positions=3, seed=1)

    exact = exotics.order_probability(p, (0, 1, 2), exotics.DEFAULT_LAMBDAS)
    sampled = float(((orders[:, 0] == 0) & (orders[:, 1] == 1) & (orders[:, 2] == 2)).mean())
    assert sampled == pytest.approx(exact, abs=0.004)

    # Simulated win probabilities must reproduce the inputs.
    win_counts = np.bincount(orders[:, 0], minlength=5) / len(orders)
    assert win_counts == pytest.approx(p, abs=0.005)


def test_boxed_probability_equals_sum_of_permutations():
    p = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    selections = [0, 1, 2]
    manual = sum(exotics.order_probability(p, order, exotics.DEFAULT_LAMBDAS)
                 for order in permutations(selections, 3))
    assert exotics.boxed_probability(p, selections, 3) == pytest.approx(manual)


def test_box_combination_counts():
    """The published anchors: 3 horses = 6, 4 = 24, 5 = 60."""
    assert exotics.box_combinations(3, 3) == 6
    assert exotics.box_combinations(4, 3) == 24
    assert exotics.box_combinations(5, 3) == 60
    assert exotics.box_combinations(6, 4) == 360
    assert exotics.box_combinations(2, 3) == 0


def test_standout_combination_count():
    """One banker to win, three horses for 2nd and 3rd = 3 x 2 = 6."""
    assert exotics.combination_count([[5], [1, 2, 3], [1, 2, 3]]) == 6


def test_places_paid_thresholds():
    """Australian place terms, from the final starter count."""
    assert exotics.places_paid(20) == 3
    assert exotics.places_paid(8) == 3
    assert exotics.places_paid(7) == 2
    assert exotics.places_paid(5) == 2
    assert exotics.places_paid(4) == 0
    assert exotics.places_paid(2) == 0


def test_place_probabilities_sum_to_places_paid():
    """Across the field, place probabilities must sum to the number of places."""
    p = np.array([0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04])
    probs, paid = exotics.place_probabilities(p)
    assert paid == 3
    assert probs.sum() == pytest.approx(3.0, abs=1e-6)

    small = np.array([0.4, 0.3, 0.2, 0.1])
    probs_small, paid_small = exotics.place_probabilities(small)
    assert paid_small == 0
    assert probs_small.sum() == 0.0


def test_same_race_multi_is_not_the_product_of_legs():
    """Legs in one race are negatively correlated - multiplying overstates it.

    Ten runners at 10% each: two horses each have a 30% chance of running top
    3, but the joint probability is C(8,1)/C(10,3) = 8/120 = 6.67%, not 9%.
    """
    p = np.full(10, 0.1)
    joint = exotics.same_race_multi_probability(
        p, [(0, 3), (1, 3)], n_sims=300_000, seed=3
    )
    assert joint == pytest.approx(8 / 120, abs=0.004)
    assert joint < 0.3 * 0.3          # strictly less than the naive product


def test_fitted_lambdas_recover_a_known_value():
    """Generate finishes from a known lambda and check the fit finds it."""
    rng = np.random.default_rng(11)
    true_lambdas = (1.0, 0.70, 0.70, 0.70)
    races = []
    for _ in range(1200):
        raw = rng.random(9) + 0.05
        p = raw / raw.sum()
        order = exotics.simulate_orders(
            p, n_sims=1, positions=3, lambdas=true_lambdas, seed=int(rng.integers(1e9))
        )[0]
        races.append((p, list(order)))

    fitted, _ = exotics.fit_lambdas(races, max_position=2)
    assert fitted[1] == pytest.approx(0.70, abs=0.15)


# --------------------------------------------------------------------------
# Odds and expected value
# --------------------------------------------------------------------------

def test_devig_methods_all_normalise():
    odds = [2.5, 4.0, 6.0, 11.0, 21.0, 51.0]
    for method in ["proportional", "power", "shin"]:
        probs = betting.devig(odds, method=method)
        assert np.nansum(probs) == pytest.approx(1.0, abs=1e-6), method
        assert (probs[np.isfinite(probs)] > 0).all(), method


def test_devig_preserves_ordering():
    odds = [2.0, 3.0, 8.0, 20.0]
    for method in ["proportional", "power", "shin"]:
        probs = betting.devig(odds, method=method)
        assert list(np.argsort(-probs)) == [0, 1, 2, 3], method


def test_shin_and_power_shrink_longshots_more_than_proportional():
    """Both should take proportionally more margin out of the longshots.

    Uses a realistic Australian book of about 122%. The effect only exists on
    an over-round book - on an under-round one the correction runs the other
    way, which is correct behaviour and not what we are testing here.
    """
    odds = [2.5, 3.5, 5.0, 7.0, 10.0, 18.0, 30.0]
    assert betting.overround(odds) > 1.15
    proportional = betting.devig(odds, "proportional")
    for method in ["power", "shin"]:
        adjusted = betting.devig(odds, method)
        # Longshot's share of the book falls; favourite's rises.
        assert adjusted[-1] < proportional[-1], method
        assert adjusted[0] > proportional[0], method


def test_devig_on_fair_book_is_identity():
    """A book already at 100% must come back unchanged."""
    probs = np.array([0.5, 0.3, 0.2])
    odds = 1 / probs
    for method in ["proportional", "power", "shin"]:
        assert betting.devig(odds, method) == pytest.approx(probs, abs=1e-4), method


def test_overround():
    assert betting.overround([2.0, 2.0]) == pytest.approx(1.0)
    assert betting.overround([1.9, 1.9]) == pytest.approx(2 / 1.9)


def test_expected_value_signs():
    # Fair bet: probability exactly matches the price.
    assert betting.expected_value(0.25, 4.0) == pytest.approx(0.0)
    # Better than the price.
    assert betting.expected_value(0.30, 4.0) == pytest.approx(0.2)
    # Worse than the price.
    assert betting.expected_value(0.20, 4.0) == pytest.approx(-0.2)


def test_commission_reduces_expected_value():
    plain = betting.expected_value(0.30, 4.0, commission=0.0)
    charged = betting.expected_value(0.30, 4.0, commission=0.10)
    assert charged < plain
    # 10% of the (4.0 - 1) profit, taken 30% of the time.
    assert plain - charged == pytest.approx(0.30 * 3.0 * 0.10)


def test_kelly_matches_the_textbook_example():
    """A 5% edge at $5.00 justifies a 1.25% bet at full Kelly.

    f = (p*d - 1) / (d - 1) = (0.21*5 - 1) / 4 = 0.0125
    """
    assert betting.kelly_fraction(0.21, 5.0) == pytest.approx(0.0125)


def test_kelly_is_zero_without_an_edge():
    assert betting.kelly_fraction(0.20, 5.0) == pytest.approx(0.0)
    assert betting.kelly_fraction(0.15, 5.0) == 0.0


def test_staking_plan_respects_both_caps():
    # Kelly would want a huge bet here; the 2% cap must bind.
    stake = betting.staking_plan(0.90, 5.0, bankroll=1000, kelly_fraction_used=1.0,
                                 max_stake_pct=0.02)
    assert stake == pytest.approx(20.0)

    # Quarter Kelly on the textbook example: 0.0125 / 4 = 0.3125% of $1000.
    stake = betting.staking_plan(0.21, 5.0, bankroll=1000, kelly_fraction_used=0.25,
                                 max_stake_pct=0.02, min_stake=0.0)
    assert stake == pytest.approx(3.13, abs=0.01)


def test_simultaneous_kelly_never_exceeds_the_cap():
    p = np.array([0.40, 0.30, 0.20, 0.10])
    odds = np.array([3.0, 4.5, 7.0, 15.0])
    stakes = betting.simultaneous_kelly(p, odds, kelly_fraction_used=0.25,
                                        max_total_pct=0.05)
    assert stakes.sum() <= 0.05 + 1e-9
    assert (stakes >= 0).all()


def test_simultaneous_kelly_matches_direct_numerical_optimisation():
    """The closed form must equal the true maximiser of expected log wealth.

    This is the real test of the multi-outcome Kelly solution: optimise
    E[log(final wealth)] numerically over the stake vector and check the
    analytic answer lands on the same point. Note that the joint solution can
    stake *more* in total than the sum of independent Kelly fractions, because
    backing several runners in one race is a partial hedge - only one of them
    can lose you the others' stakes.
    """
    from scipy.optimize import minimize

    p = np.array([0.40, 0.25, 0.15])
    odds = np.array([2.8, 4.5, 8.0])

    def negative_log_growth(f):
        if f.min() < 0 or f.sum() >= 1:
            return 1e6
        total = 0.0
        for i in range(len(p)):
            wealth = 1 + f[i] * (odds[i] - 1) - (f.sum() - f[i])
            if wealth <= 0:
                return 1e6
            total += p[i] * np.log(wealth)
        total += (1 - p.sum()) * np.log(1 - f.sum())
        return -total

    numerical = minimize(
        negative_log_growth, np.full(3, 0.05), method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000},
    ).x
    analytic = betting.simultaneous_kelly(p, odds, kelly_fraction_used=1.0,
                                          max_total_pct=1.0)
    assert analytic == pytest.approx(numerical, abs=1e-5)


def test_simultaneous_kelly_excludes_negative_edge_runners():
    """A runner priced worse than the model rates it must get nothing."""
    p = np.array([0.40, 0.25, 0.05])
    odds = np.array([2.8, 4.5, 3.0])      # third horse is badly over-bet
    stakes = betting.simultaneous_kelly(p, odds, kelly_fraction_used=1.0,
                                        max_total_pct=1.0)
    assert stakes[2] == 0.0
    assert stakes[0] > 0 and stakes[1] > 0


def test_drawdown_probability_matches_the_formula():
    """P(ever reach x*W0) = x^(2/f - 1). Full Kelly halving = 50%."""
    assert betting.drawdown_probability(1.0, 0.5) == pytest.approx(0.5)
    assert betting.drawdown_probability(0.5, 0.5) == pytest.approx(0.125)
    assert betting.drawdown_probability(0.25, 0.5) == pytest.approx(0.5 ** 7, abs=1e-6)


def test_fractional_kelly_growth():
    """f(2 - f): half Kelly keeps 75% of the growth rate."""
    assert betting.kelly_growth_fraction(1.0) == pytest.approx(1.0)
    assert betting.kelly_growth_fraction(0.5) == pytest.approx(0.75)
    assert betting.kelly_growth_fraction(0.25) == pytest.approx(0.4375)


# --------------------------------------------------------------------------
# Tote
# --------------------------------------------------------------------------

def test_tote_dividend_matches_the_pool_formula():
    """$10,000 pool, $1,000 on the horse, 14.5% takeout.
    Dividend = 10000 * 0.855 / 1000 = $8.55."""
    dividend = tote.tote_dividend(10_000, 1_000, 0.145, breakage=False)
    assert dividend == pytest.approx(8.55)


def test_own_stake_dilutes_the_dividend():
    without = tote.tote_dividend(10_000, 1_000, 0.145, your_stake=0, breakage=False)
    with_stake = tote.tote_dividend(10_000, 1_000, 0.145, your_stake=500, breakage=False)
    assert with_stake < without


def test_breakage_always_rounds_down():
    assert tote.apply_breakage(8.59) == pytest.approx(8.50)
    assert tote.apply_breakage(8.50) == pytest.approx(8.50)
    assert tote.apply_breakage(1.01) == pytest.approx(tote.MINIMUM_DIVIDEND)


def test_place_pool_splits_between_placegetters():
    single = tote.tote_dividend(9_000, 1_000, 0.1425, n_places=1, breakage=False)
    split = tote.tote_dividend(9_000, 1_000, 0.1425, n_places=3, breakage=False)
    assert split == pytest.approx(single / 3)


def test_required_confidence_thresholds():
    """At 14.5% win takeout you must be 17% more confident than the pool;
    at the 23% first-four takeout, 30%."""
    assert tote.required_confidence(0.145) == pytest.approx(1.1696, abs=1e-3)
    assert tote.required_confidence(0.23) == pytest.approx(1.2987, abs=1e-3)


def test_tote_edge_is_zero_at_the_break_even_ratio():
    takeout = 0.145
    ratio = tote.required_confidence(takeout)
    assert tote.tote_edge(0.20 * ratio, 0.20, takeout) == pytest.approx(0.0)


def test_flexi_percentage_and_payout():
    """Six horses boxed in a first four is 360 combinations; $36 is 10% flexi."""
    combos = exotics.box_combinations(6, 4)
    assert combos == 360
    bet = tote.FlexiBet(combinations=combos, outlay=36.0)
    assert bet.percentage == pytest.approx(0.10)
    assert bet.payout(5_000.0) == pytest.approx(500.0)


def test_flexi_does_not_change_roi():
    """The whole point: flexi scales outlay and return together."""
    probs = {(0, 1, 2): 0.02, (0, 2, 1): 0.015, (1, 0, 2): 0.01}
    dividends = {(0, 1, 2): 60.0, (0, 2, 1): 80.0, (1, 0, 2): 90.0}
    roi_full = tote.flexi_roi(probs, dividends, combinations=6)
    roi_small = tote.flexi_roi(probs, dividends, combinations=6)
    assert roi_full == roi_small


# --------------------------------------------------------------------------
# Schema helpers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("Good 4", 4.0), ("GOOD4", 4.0), ("Soft 7", 7.0), ("Heavy 10", 10.0),
    ("Firm 2", 2.0), (5, 5.0), (5.0, 5.0),
])
def test_track_condition_parsing(value, expected):
    assert track_condition_to_number(value) == pytest.approx(expected)


def test_unknown_track_condition_is_nan_not_a_guess():
    assert np.isnan(track_condition_to_number(None))
    assert np.isnan(track_condition_to_number(""))
    assert np.isnan(track_condition_to_number("Synthetic"))


@pytest.mark.parametrize("value,band", [
    ("Good 4", "GOOD"), ("Soft 5", "SOFT"), ("Soft 7", "SOFT"),
    ("Heavy 8", "HEAVY"), ("Firm 1", "FIRM"), (None, "UNKNOWN"),
])
def test_track_condition_bands(value, band):
    assert track_condition_band(value) == band


def test_fair_odds_handles_scalars_and_arrays():
    """Regression: a scalar probability used to raise a TypeError, which broke
    every place-bet recommendation."""
    assert betting.fair_odds(0.25) == pytest.approx(4.0)
    assert isinstance(betting.fair_odds(0.25), float)
    assert betting.fair_odds(0.0) == np.inf
    assert betting.fair_odds([0.5, 0.25, 0.0]) == pytest.approx([2.0, 4.0, np.inf])


def test_no_win_bets_recommended_beyond_the_odds_cap():
    """A 200/1 shot must never be recommended, however large the apparent edge.

    Regression: the tool once recommended a $228 place bet at HIGH confidence
    off an estimated dividend. A tiny absolute error on a tiny probability is a
    huge relative error, and published data shows 100/1+ runners return about
    39c in the dollar.
    """
    bets = betting.recommend_win_bets(
        ["a"], ["Longshot"], [0.05], [200.0], bankroll=1000, min_edge=0.05,
    )
    assert bets == []

    place = betting.recommend_place_bets(
        ["a"], ["Longshot"], [0.035], [228.6], bankroll=1000, min_edge=0.04,
    )
    assert place == []


def test_estimated_prices_never_earn_high_confidence():
    """An estimated dividend is a guess about a pool, not a quoted price."""
    quoted = betting.recommend_win_bets(
        ["a"], ["Horse"], [0.40], [4.0], bankroll=1000, min_edge=0.05,
    )
    assert quoted and quoted[0].confidence == "HIGH"

    estimated = betting.recommend_place_bets(
        ["a"], ["Horse"], [0.70], [2.5], bankroll=1000, min_edge=0.04,
        price_is_estimated=True,
    )
    assert estimated and estimated[0].confidence == "LOW"
