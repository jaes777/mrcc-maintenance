"""Feature engineering.

Every feature here is computed from information available *before* the race
is run: the runner's own form lines dated strictly earlier, and a
`RollingContext` that has only observed earlier races.

Features are grouped by what a form analyst would actually look at:

  form      -- has this horse been running well lately, and against whom
  class     -- is it going up or down in grade
  speed     -- how fast has it actually run, adjusted for track and going
  fitness   -- freshness, spell, run-through-the-campaign
  suitability -- distance, going, track, barrier
  people    -- jockey, trainer, and the pairing
  physical  -- weight, age, sex
  market    -- what the money thinks (optional; see `include_market`)

The market block is kept separate and optional because it is by far the
strongest single predictor, and mixing it in unthinkingly makes it
impossible to tell whether the rest of the model has learnt anything. The
recommended setup trains a form-only model, then blends it with the market
explicitly -- see `ausform.model.blend`.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..types import PastRun, Race, Runner
from .context import RollingContext

# Sentinel used where a value genuinely does not exist (e.g. a first-starter
# has no "days since last run"). Gradient boosters handle NaN natively and
# route it down its own branch, which is more honest than imputing a zero.
MISSING = float("nan")


@dataclass
class FeatureSet:
    """Feature matrix for one race, aligned to `runners`."""

    matrix: np.ndarray          # shape (n_runners, n_features)
    names: list[str]
    runners: list[Runner]
    race: Race

    @property
    def n_runners(self) -> int:
        return len(self.runners)

    def column(self, name: str) -> np.ndarray:
        return self.matrix[:, self.names.index(name)]


def _safe_div(numerator: float, denominator: float, default: float = MISSING) -> float:
    return numerator / denominator if denominator else default


def _mean(values: Sequence[float]) -> float:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return sum(clean) / len(clean) if clean else MISSING


# --------------------------------------------------------------------------
# Form
# --------------------------------------------------------------------------

def _finish_percentile(run: PastRun) -> Optional[float]:
    """Where the horse finished, scaled 1.0 (won) to 0.0 (last).

    Raw finishing position is misleading across field sizes: third of five
    is a worse run than third of eighteen. This normalises it.
    """
    if run.finish_position is None or not run.field_size or run.field_size < 2:
        return None
    return 1.0 - (run.finish_position - 1) / (run.field_size - 1)


def _form_features(runs: list[PastRun]) -> dict[str, float]:
    if not runs:
        return {
            "starts": 0.0,
            "wins": 0.0,
            "places": 0.0,
            "win_rate": MISSING,
            "place_rate": MISSING,
            "avg_finish_pct_3": MISSING,
            "avg_finish_pct_5": MISSING,
            "best_finish_pct_5": MISSING,
            "last_finish_pct": MISSING,
            "form_trend": MISSING,
            "avg_beaten_lengths_3": MISSING,
            "consistency": MISSING,
            "is_first_starter": 1.0,
        }

    wins = sum(1 for r in runs if r.won)
    places = sum(1 for r in runs if r.placed)
    pct = [p for p in (_finish_percentile(r) for r in runs) if p is not None]

    recent3 = pct[:3]
    recent5 = pct[:5]
    older = pct[3:8]

    beaten = [r.margin_l for r in runs[:3] if r.margin_l is not None]

    # Trend: recent form minus slightly older form. Positive means the
    # horse is on the up, which handicappers weight heavily.
    trend = MISSING
    if recent3 and older:
        trend = _mean(recent3) - _mean(older)

    consistency = MISSING
    if len(recent5) >= 3:
        consistency = -float(np.std(recent5))  # higher (less negative) = steadier

    return {
        "starts": float(len(runs)),
        "wins": float(wins),
        "places": float(places),
        "win_rate": _safe_div(wins, len(runs)),
        "place_rate": _safe_div(places, len(runs)),
        "avg_finish_pct_3": _mean(recent3),
        "avg_finish_pct_5": _mean(recent5),
        "best_finish_pct_5": max(recent5) if recent5 else MISSING,
        "last_finish_pct": pct[0] if pct else MISSING,
        "form_trend": trend,
        "avg_beaten_lengths_3": _mean(beaten),
        "consistency": consistency,
        "is_first_starter": 0.0,
    }


# --------------------------------------------------------------------------
# Class
# --------------------------------------------------------------------------

_CLASS_KEYWORDS = [
    ("group 1", 13.0), ("g1", 13.0),
    ("group 2", 12.0), ("g2", 12.0),
    ("group 3", 11.0), ("g3", 11.0),
    ("listed", 10.0), ("lr", 10.0),
    ("bm84", 8.4), ("bm 84", 8.4),
    ("bm78", 7.8), ("bm 78", 7.8),
    ("bm70", 7.0), ("bm 70", 7.0),
    ("bm64", 6.4), ("bm 64", 6.4),
    ("bm58", 5.8), ("bm 58", 5.8),
    ("class 6", 6.0), ("class 5", 5.5), ("class 4", 5.0),
    ("class 3", 4.5), ("class 2", 4.0), ("class 1", 3.5),
    ("maiden", 2.0), ("mdn", 2.0),
]


def class_score(text: Optional[str]) -> float:
    """Map a race-class string onto a rough numeric ladder.

    Australian class nomenclature is inconsistent across states, so this is
    deliberately fuzzy. Prize money (below) is the more reliable signal;
    this exists to catch cases where prize money is missing.
    """
    if not text:
        return MISSING
    lowered = text.lower()
    for keyword, score in _CLASS_KEYWORDS:
        if keyword in lowered:
            return score
    return MISSING


def _class_features(runs: list[PastRun], race: Race) -> dict[str, float]:
    today_money = race.prize_money
    today_class = class_score(race.class_level)

    recent_money = [r.prize_money_total for r in runs[:5] if r.prize_money_total]
    recent_class = [class_score(r.class_level) for r in runs[:5]]
    recent_class = [c for c in recent_class if not math.isnan(c)]

    # Class change: positive means stepping up in grade today.
    money_change = MISSING
    if today_money and recent_money:
        money_change = math.log(today_money / max(1.0, _mean(recent_money)))

    class_change = MISSING
    if not math.isnan(today_class) and recent_class:
        class_change = today_class - _mean(recent_class)

    # Highest grade the horse has actually won at.
    won_class = [class_score(r.class_level) for r in runs if r.won]
    won_class = [c for c in won_class if not math.isnan(c)]

    return {
        "race_class_score": today_class,
        "race_prize_money_log": math.log(today_money) if today_money else MISSING,
        "avg_class_last5": _mean(recent_class) if recent_class else MISSING,
        "class_change": class_change,
        "prize_money_change_log": money_change,
        "best_winning_class": max(won_class) if won_class else MISSING,
    }


# --------------------------------------------------------------------------
# Speed
# --------------------------------------------------------------------------

def _speed_features(runs: list[PastRun], race: Race,
                    context: RollingContext) -> dict[str, float]:
    """Time-based ratings, normalised against the track/distance/going norm.

    A raw race time is meaningless on its own -- 1200m at Flemington on a
    Good 3 is not comparable to 1200m at Wyong on a Heavy 9. Each run is
    converted into standard deviations faster than the baseline for its
    own bucket, which makes times comparable across the country.
    """
    ratings: list[float] = []
    for run in runs[:8]:
        if not run.race_time_s:
            continue
        baseline = context.time_baseline(
            run.track_code, run.distance_m,
            RollingContext._going_band(run.track_condition))
        if baseline is None or baseline.sd <= 0:
            continue
        # Negative time difference = faster than average = good, so flip sign.
        rating = (baseline.mean - run.race_time_s) / baseline.sd
        # Clip to a plausible range. Beyond about 5 sd the value is almost
        # always a timing error or a freak track variant, not a fast horse,
        # and left unclipped it would dominate every downstream average.
        ratings.append(max(-5.0, min(5.0, rating)))

    sectionals = [r.last_600m_s for r in runs[:5] if r.last_600m_s]

    return {
        "speed_rating_best": max(ratings) if ratings else MISSING,
        "speed_rating_avg3": _mean(ratings[:3]) if ratings else MISSING,
        "speed_rating_last": ratings[0] if ratings else MISSING,
        "speed_ratings_available": float(len(ratings)),
        # Faster closing sectional is better, so negate for consistency
        # with the "higher is better" convention used throughout.
        "sectional_600_best": -min(sectionals) if sectionals else MISSING,
        "sectional_600_avg": -_mean(sectionals) if sectionals else MISSING,
    }


# --------------------------------------------------------------------------
# Fitness / freshness
# --------------------------------------------------------------------------

def _fitness_features(runs: list[PastRun], race_date: _dt.date) -> dict[str, float]:
    if not runs:
        return {
            "days_since_run": MISSING,
            "log_days_since_run": MISSING,
            "is_first_up": 1.0,
            "is_second_up": 0.0,
            "runs_this_prep": 0.0,
            "runs_last_90d": 0.0,
            "career_days": MISSING,
        }

    days = (race_date - runs[0].date).days
    # A gap over ~90 days is conventionally treated as a spell in Australia.
    is_first_up = 1.0 if days >= 90 else 0.0

    # Count runs since the last spell to locate the horse in its campaign.
    runs_this_prep = 0
    previous = race_date
    for run in runs:
        if (previous - run.date).days >= 90:
            break
        runs_this_prep += 1
        previous = run.date

    is_second_up = 1.0 if runs_this_prep == 1 and not is_first_up else 0.0
    runs_90 = sum(1 for r in runs if (race_date - r.date).days <= 90)
    career_days = (race_date - runs[-1].date).days

    return {
        "days_since_run": float(days),
        "log_days_since_run": math.log1p(max(0, days)),
        "is_first_up": is_first_up,
        "is_second_up": is_second_up,
        "runs_this_prep": float(runs_this_prep),
        "runs_last_90d": float(runs_90),
        "career_days": float(career_days),
    }


# --------------------------------------------------------------------------
# Suitability: distance, going, track, barrier
# --------------------------------------------------------------------------

def _suitability_features(runs: list[PastRun], runner: Runner, race: Race,
                          context: RollingContext) -> dict[str, float]:
    distance = race.distance_m
    condition = race.track_condition

    # -- distance
    at_distance = [r for r in runs if abs(r.distance_m - distance) <= 100]
    dist_wins = sum(1 for r in at_distance if r.won)
    winning_distances = [r.distance_m for r in runs if r.won]
    avg_distance = _mean([float(r.distance_m) for r in runs[:6]])

    distance_change = MISSING
    if runs:
        distance_change = distance - runs[0].distance_m

    best_distance_gap = MISSING
    if winning_distances:
        best_distance_gap = abs(distance - _mean([float(d) for d in winning_distances]))

    # -- going
    wet = [r for r in runs if r.track_condition and r.track_condition >= 5]
    dry = [r for r in runs if r.track_condition and r.track_condition <= 4]
    is_wet_today = 1.0 if (condition and condition >= 5) else 0.0

    wet_win_rate = _safe_div(sum(1 for r in wet if r.won), len(wet))
    dry_win_rate = _safe_div(sum(1 for r in dry if r.won), len(dry))

    # Positive means the horse is relatively better on rain-affected ground.
    going_edge = MISSING
    if not math.isnan(wet_win_rate) and not math.isnan(dry_win_rate):
        going_edge = wet_win_rate - dry_win_rate

    # -- track
    at_track = [r for r in runs if r.track_code == race.track.code]
    track_wins = sum(1 for r in at_track if r.won)

    # -- barrier
    barrier = runner.barrier
    field_size = max(2, race.field_size)
    barrier_pct = MISSING
    if barrier is not None:
        barrier_pct = (barrier - 1) / (field_size - 1)

    # Interaction: a wide gate costs far more on a tight track over a short
    # trip, where there is no time to cross or recover.
    straight = race.track.straight_m or 400
    barrier_cost = MISSING
    if barrier is not None:
        tightness = 400.0 / straight
        trip = 1600.0 / max(distance, 800)
        barrier_cost = -barrier_pct * tightness * trip

    return {
        "starts_at_distance": float(len(at_distance)),
        "win_rate_at_distance": _safe_div(dist_wins, len(at_distance)),
        "distance_change": float(distance_change) if distance_change is not MISSING else MISSING,
        "distance_vs_avg": (distance - avg_distance) if not math.isnan(avg_distance) else MISSING,
        "best_distance_gap": best_distance_gap,
        "race_distance": float(distance),
        "is_wet_track": is_wet_today,
        "track_condition": float(condition) if condition else MISSING,
        "wet_win_rate": wet_win_rate,
        "dry_win_rate": dry_win_rate,
        "going_edge": going_edge,
        "wet_starts": float(len(wet)),
        "starts_at_track": float(len(at_track)),
        "win_rate_at_track": _safe_div(track_wins, len(at_track)),
        "barrier": float(barrier) if barrier is not None else MISSING,
        "barrier_pct": barrier_pct,
        "barrier_cost": barrier_cost,
        "barrier_historic_win_rate": context.barrier_win_rate(
            race.track.code, distance, barrier),
        "field_size": float(field_size),
        "sire_wet_win_rate": (context.sire_wet_win_rate(runner.sire)
                              if is_wet_today else MISSING),
    }


# --------------------------------------------------------------------------
# People and physical
# --------------------------------------------------------------------------

def _people_features(runner: Runner, runs: list[PastRun],
                     context: RollingContext) -> dict[str, float]:
    # Has this jockey ridden the horse before, and how did it go?
    same_jockey = [r for r in runs if r.jockey and r.jockey == runner.jockey]
    jockey_switch = 1.0 if (runs and runs[0].jockey and
                            runs[0].jockey != runner.jockey) else 0.0

    return {
        "jockey_win_rate": context.jockey_win_rate(runner.jockey),
        "jockey_place_rate": context.jockey_place_rate(runner.jockey),
        "jockey_experience": math.log1p(context.jockey_starts(runner.jockey)),
        "trainer_win_rate": context.trainer_win_rate(runner.trainer),
        "trainer_place_rate": context.trainer_place_rate(runner.trainer),
        "combo_win_rate": context.combo_win_rate(runner.jockey, runner.trainer),
        "jockey_rode_before": float(len(same_jockey)),
        "jockey_win_rate_on_horse": _safe_div(
            sum(1 for r in same_jockey if r.won), len(same_jockey)),
        "jockey_switch": jockey_switch,
    }


def _physical_features(runner: Runner, runs: list[PastRun]) -> dict[str, float]:
    weight = runner.weight_kg
    weight_change = MISSING
    if weight and runs and runs[0].weight_kg:
        weight_change = weight - runs[0].weight_kg

    sex = (runner.sex or "").lower()[:1]

    return {
        "weight_kg": float(weight) if weight else MISSING,
        "weight_change": weight_change,
        "apprentice_claim": float(runner.apprentice_claim_kg or 0.0),
        "age": float(runner.age) if runner.age else MISSING,
        "is_mare_or_filly": 1.0 if sex in ("m", "f") else 0.0,
        "is_gelding": 1.0 if sex == "g" else 0.0,
        # Provider-supplied ratings, if the feed carries them.
        "early_speed_rating": (float(runner.early_speed_rating)
                               if runner.early_speed_rating else MISSING),
        "provider_form_rating": (float(runner.form_rating)
                                 if runner.form_rating else MISSING),
    }


# --------------------------------------------------------------------------
# Market
# --------------------------------------------------------------------------

def _market_features(runner: Runner, race: Race) -> dict[str, float]:
    """Market-derived features.

    Kept optional and clearly separated. The market is the single best
    predictor of a horse race that exists -- it aggregates every stable
    whisper, every professional's opinion and every piece of information
    the feeds do not carry. A model that includes it will look far better
    than one that does not, but that improvement is not *your* edge.
    """
    odds = runner.fixed_win_odds or runner.tote_win_odds
    if not odds or odds <= 1.0:
        return {
            "market_prob_raw": MISSING,
            "market_log_odds": MISSING,
            "market_rank": MISSING,
            "odds_drift": MISSING,
        }

    raw_prob = 1.0 / odds
    priced = [r for r in race.active_runners
              if (r.fixed_win_odds or r.tote_win_odds)]
    ranked = sorted(priced, key=lambda r: (r.fixed_win_odds or r.tote_win_odds))
    rank = ranked.index(runner) + 1 if runner in ranked else MISSING

    # Price movement from the opening line. Firming (shortening) horses win
    # more often than drifting ones -- late money is informed money.
    drift = MISSING
    if runner.opening_odds and runner.opening_odds > 1.0:
        drift = math.log(runner.opening_odds / odds)

    return {
        "market_prob_raw": raw_prob,
        "market_log_odds": math.log(raw_prob / max(1e-9, 1 - raw_prob)),
        "market_rank": float(rank) if rank is not MISSING else MISSING,
        "odds_drift": drift,
    }


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def runner_features(runner: Runner, race: Race, context: RollingContext,
                    include_market: bool = False) -> dict[str, float]:
    """All features for one runner. Reads only pre-race information."""
    runs = runner.runs_before(race.date)

    values: dict[str, float] = {}
    values.update(_form_features(runs))
    values.update(_class_features(runs, race))
    values.update(_speed_features(runs, race, context))
    values.update(_fitness_features(runs, race.date))
    values.update(_suitability_features(runs, runner, race, context))
    values.update(_people_features(runner, runs, context))
    values.update(_physical_features(runner, runs))
    if include_market:
        values.update(_market_features(runner, race))
    return values


def build_features(race: Race, context: RollingContext,
                   include_market: bool = False) -> FeatureSet:
    """Feature matrix for every non-scratched runner in `race`.

    Also adds within-race relative features. This matters more than it
    might seem: horse racing is a competition, so what predicts the winner
    is not a horse's absolute quality but its quality *relative to the
    others in this particular race*. A speed rating of 2.0 is a standout
    in a maiden and unremarkable in a Group 1.
    """
    runners = race.active_runners
    if not runners:
        return FeatureSet(np.zeros((0, 0)), [], [], race)

    rows = [runner_features(r, race, context, include_market) for r in runners]
    names = sorted(rows[0].keys())
    matrix = np.array([[row.get(name, MISSING) for name in names] for row in rows],
                      dtype=float)

    relative_names, relative_matrix = _relative_features(matrix, names)

    return FeatureSet(
        matrix=np.hstack([matrix, relative_matrix]),
        names=names + relative_names,
        runners=runners,
        race=race,
    )


# Features where a horse's standing relative to its rivals carries more
# information than the raw value.
_RELATIVE_BASE = [
    "speed_rating_best", "speed_rating_avg3", "avg_finish_pct_3",
    "avg_finish_pct_5", "win_rate", "place_rate", "jockey_win_rate",
    "trainer_win_rate", "weight_kg", "class_change", "best_winning_class",
    "days_since_run", "win_rate_at_distance", "barrier_cost",
    "race_prize_money_log", "form_trend",
]


def _relative_features(matrix: np.ndarray, names: list[str]
                       ) -> tuple[list[str], np.ndarray]:
    """Z-score and rank each selected feature within the race."""
    out_names: list[str] = []
    columns: list[np.ndarray] = []

    for base in _RELATIVE_BASE:
        if base not in names:
            continue
        values = matrix[:, names.index(base)]
        finite = values[np.isfinite(values)]

        if finite.size >= 2:
            mean = float(finite.mean())
            sd = float(finite.std())
            z = (values - mean) / sd if sd > 1e-9 else np.zeros_like(values)
            # Rank within race, scaled to [0, 1]; higher value = higher rank.
            order = np.argsort(np.argsort(np.where(np.isfinite(values), values, -np.inf)))
            rank = order / max(1, len(values) - 1)
        else:
            z = np.full_like(values, MISSING)
            rank = np.full_like(values, MISSING)

        z = np.where(np.isfinite(values), z, MISSING)
        rank = np.where(np.isfinite(values), rank, MISSING)

        out_names.append(f"rel_z_{base}")
        columns.append(z)
        out_names.append(f"rel_rank_{base}")
        columns.append(rank)

    if not columns:
        return [], np.zeros((matrix.shape[0], 0))
    return out_names, np.column_stack(columns)
