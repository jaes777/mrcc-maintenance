"""A synthetic Australian racing season.

WHY THIS EXISTS
---------------
Real, licensed Australian form data is expensive and cannot be bundled
here. But a prediction system needs to be *proved correct* before anyone
trusts it with money, and you cannot debug a model when you have no idea
what the right answer was.

This module generates a season where the ground truth is known by
construction: every horse has a latent ability and set of preferences that
we choose, races are decided by those latent values plus noise, and a
simulated betting market prices them with realistic distortions. That lets
us verify the machinery end to end -- does the feature builder recover the
signal, does the model calibrate, does the backtester report honest ROI,
does the staking plan avoid ruin.

WHAT THIS IS NOT
----------------
It is NOT evidence that the model will work on real races. Accuracy
figures measured here describe this simulator, not Australian racing.
The simulator is deliberately *more* predictable than reality. Any claim
about live performance must come from a backtest on real historical data.
"""

from __future__ import annotations

import datetime as _dt
import math
import random

import numpy as np
from scipy.special import ndtr
from dataclasses import dataclass, field
from typing import Optional

from ..tracks import all_tracks
from ..types import (
    Meeting,
    PastRun,
    Race,
    Runner,
    Track,
    Weather,
)

# Class ladder from lowest to highest. Higher index = better horses,
# bigger prize money, and a tighter spread of ability within the race.
CLASS_LADDER = [
    "Maiden",
    "Class 1",
    "Class 2",
    "Class 3",
    "BM58",
    "BM64",
    "BM70",
    "BM78",
    "BM84",
    "Listed",
    "Group 3",
    "Group 2",
    "Group 1",
]

COMMON_DISTANCES = [1000, 1100, 1200, 1300, 1400, 1600, 1800, 2000, 2040, 2400, 3200]

# A thoroughbred backing up inside a week is rare and usually deliberate.
_MIN_DAYS_BETWEEN_RUNS = 7

# Being badly wrong at the trip is a big handicap, not an infinite one.
_MAX_DISTANCE_PENALTY = 3.0

# Australian fixed-odds books cap somewhere around $301. Quoting beyond
# that is not something any real market does, and unbounded prices wreck
# every downstream metric that touches a payout.
_MAX_ODDS = 301.0
_MIN_ODDS = 1.04


@dataclass
class SimHorse:
    """A horse with hidden truth attached."""

    horse_id: str
    name: str
    ability: float          # latent quality, roughly N(0, 1)
    wet_affinity: float     # >0 improves on rain-affected ground
    optimal_distance: int    # metres where the horse is at its best
    distance_tolerance: float  # how quickly it falls away from optimum
    improvement_rate: float  # per-start improvement while lightly raced
    consistency: float      # lower = more erratic run to run
    birth_year: int
    sire: str
    starts: int = 0
    history: list[PastRun] = field(default_factory=list)
    current_class: int = 0
    days_since_run: int = 60
    retired: bool = False


@dataclass
class SimJockey:
    name: str
    skill: float  # in the same units as horse ability, but much smaller


@dataclass
class SimTrainer:
    name: str
    skill: float


_FIRST = ["Ripper", "Bold", "Golden", "Silent", "Northern", "Red", "Lucky", "Storm",
          "Midnight", "Coastal", "Iron", "Velvet", "Wild", "Royal", "Swift", "Dusty",
          "Electric", "Autumn", "Sacred", "Rebel", "Gallant", "Crimson", "Outback",
          "Marble", "Thunder", "Whisper", "Granite", "Feather", "Copper", "Sapphire"]
_SECOND = ["Ambition", "Runner", "Legend", "Dancer", "Miss", "Express", "Prince",
           "Rocket", "Belle", "Warrior", "Star", "Chief", "Spirit", "Bullet", "Queen",
           "Arrow", "Vision", "Flame", "Knight", "Prospect", "Empire", "Charm",
           "Sonnet", "Tempest", "Voyager", "Halo", "Ranger", "Echo", "Bandit", "Comet"]

_JOCKEY_SURNAMES = ["McDonald", "Bowman", "Berry", "Lane", "Zahra", "Williams",
                    "Allen", "Rawiller", "Boss", "Avdulla", "Schofield", "Clark",
                    "Hall", "Nolen", "Dunn", "Purton", "Baker", "Collett", "Dee",
                    "Thornton", "Currie", "Hope", "Yendall", "Maskiell"]

_TRAINER_SURNAMES = ["Waterhouse", "Hayes", "Cummings", "Maher", "Waller", "Price",
                     "Freedman", "Snowden", "Gollan", "Munce", "Weir", "Moody",
                     "O'Brien", "Kent", "Ryan", "Bott", "Eustace", "McEvoy"]


class SeasonSimulator:
    """Generates a chronologically consistent racing season.

    Races are run in date order and each horse's form history is built from
    races it has actually contested in this simulation. That matters: it
    means a backtest over simulator output exercises the same
    point-in-time logic that live data would, including horses with one
    start, first-up runs, and class changes.
    """

    def __init__(
        self,
        seed: int = 7,
        n_horses: int = 9000,
        n_jockeys: int = 60,
        n_trainers: int = 40,
        market_efficiency: float = 0.93,
        market_overround: float = 1.18,
        noise_scale: float = 1.80,
        private_info_sd: float = 0.55,
        market_sees_private: float = 0.85,
        favourite_longshot_exponent: float = 0.86,
    ):
        """
        n_horses: population size. Should be large relative to the number of
            races you intend to simulate -- Australian racing runs roughly
            19,000 races a year off about 30,000 individual horses, i.e.
            each horse starts 6-9 times a season. Too small a population
            makes every horse a 50-start veteran and inflates form quality.
        market_efficiency: how much of the true signal the simulated market
            captures (1.0 = perfectly efficient and unbeatable, 0 = random).
            Real win markets are highly but not perfectly efficient.
        favourite_longshot_exponent: strength of the favourite-longshot
            bias, applied as p ** exponent then renormalised. Below 1 it
            lifts longshots' implied chances above the truth, which is what
            punters actually do.

            These two are calibrated together, because they push blind
            flat-bet return in opposite directions and only their net
            effect is observable. At the defaults, betting every runner
            blind returns about -29% in the 2-5% probability band rising to
            about -11% at 30-60%, which is the shape and magnitude real
            Australian racing shows. An earlier setting (0.82 / 0.91) had
            market noise swamping the bias and produced a gradient running
            the *wrong way* -- longshots returning better than favourites
            -- which would have quietly rewarded a model for backing
            outsiders.
        market_overround: total book percentage. Australian tote win pools
            and fixed-odds books typically run 1.15-1.25.
        noise_scale: multiplier on run-to-run variability. This is the dial
            that sets how predictable racing is. It is calibrated so the
            market favourite wins about a third of races, which is what
            actually happens in Australia. Turning it down makes the
            simulator easy and the results meaningless.

            Calibration at the defaults (seed 11, 75 days, ~1,700 races):
            the market favourite wins 32.2% and places 66%, and the
            devigged market scores a log-loss of about 1.91. Published
            Australian figures put the favourite near 32% winning and
            60-65% placing. Do not turn this down to make the model look
            better -- it is the dial that decides whether the whole
            evaluation means anything.
        private_info_sd: size of the per-run "private information" term --
            how the horse worked on Tuesday, whether it ate up, that the
            stable is quietly confident, a niggling problem the vet
            cleared. This affects the result but appears in NO feature the
            model can compute, because no data feed carries it.
        market_sees_private: the fraction of that private term the betting
            market prices in. This is the single most important realism
            knob in the simulator. Without it, the model and the market see
            exactly the same information, the model wins easily, and the
            backtest reports fantasy returns. Real betting markets contain
            people who talk to stables, and that is precisely why they are
            hard to beat.
        """
        self.rng = random.Random(seed)
        # Separate numpy stream for the vectorised Monte Carlo pricing.
        self._nprng = np.random.default_rng(seed)
        self.market_efficiency = market_efficiency
        self.market_overround = market_overround
        self.noise_scale = noise_scale
        self.private_info_sd = private_info_sd
        self.market_sees_private = market_sees_private
        self.fl_exponent = favourite_longshot_exponent
        self.tracks: list[Track] = [t for t in all_tracks() if t.latitude != 0.0]

        self.horses = [self._make_horse(i) for i in range(n_horses)]
        self.jockeys = [
            SimJockey(f"{self.rng.choice('ABCDEFGHJKLMNPRSTW')} {self.rng.choice(_JOCKEY_SURNAMES)}",
                      self.rng.gauss(0, 0.16))
            for _ in range(n_jockeys)
        ]
        self.trainers = [
            SimTrainer(f"{self.rng.choice('ABCDEFGHJKLMNPRSTW')} {self.rng.choice(_TRAINER_SURNAMES)}",
                       self.rng.gauss(0, 0.13))
            for _ in range(n_trainers)
        ]
        self._used_names: set[str] = set()
        self._race_counter = 0

    # -- construction ------------------------------------------------------

    def _make_horse(self, index: int) -> SimHorse:
        rng = self.rng
        name = f"{rng.choice(_FIRST)} {rng.choice(_SECOND)}"
        return SimHorse(
            horse_id=f"H{index:05d}",
            name=f"{name} ({index % 97})" if index > 60 else name,
            ability=rng.gauss(0, 1.0),
            wet_affinity=rng.gauss(0, 0.45),
            optimal_distance=rng.choice(COMMON_DISTANCES),
            distance_tolerance=rng.uniform(250, 700),
            improvement_rate=abs(rng.gauss(0.05, 0.04)),
            consistency=rng.uniform(0.55, 1.15),
            birth_year=2019 - rng.randint(0, 5),
            sire=f"{rng.choice(_FIRST)} {rng.choice(_SECOND)}",
        )

    # -- the physics of a run ---------------------------------------------

    def _barrier_penalty(self, barrier: int, track: Track, distance_m: int,
                         field_size: int) -> float:
        """Wide gates cost more on tight tracks and over short trips.

        On a course like Moonee Valley (173m straight) a wide draw over
        1200m is a genuine handicap; at Flemington down the straight six it
        barely matters. Scaled by how far out the gate is relative to the
        field size so an eight-horse field is not punished like a sixteen.
        """
        if barrier <= 1:
            return 0.0
        straight = track.straight_m or 400
        tightness = max(0.4, min(1.8, 400.0 / straight))
        # Short races give less time to recover from a wide run.
        trip_factor = max(0.5, min(1.6, 1600.0 / max(distance_m, 800)))
        relative = (barrier - 1) / max(1, field_size - 1)
        return -0.55 * tightness * trip_factor * (relative ** 1.5)

    def _distance_penalty(self, horse: SimHorse, distance_m: int) -> float:
        """Penalty for running outside the horse's best trip.

        Capped deliberately. An unbounded quadratic sends a sprinter
        entered over 3200m to a strength of -100, which in turn produces
        astronomical odds that no real market would ever quote. A horse
        badly out of its distance range runs poorly; it does not cease to
        exist.
        """
        gap = abs(distance_m - horse.optimal_distance)
        raw = 1.4 * (gap / horse.distance_tolerance) ** 2
        return -min(raw, _MAX_DISTANCE_PENALTY)

    def _condition_effect(self, horse: SimHorse, condition: int) -> float:
        """Wet-track ability only expresses itself once the ground softens."""
        wetness = max(0, condition - 4) / 6.0  # 0 on Good, 1 on Heavy 10
        return horse.wet_affinity * wetness * 1.6

    def _weight_penalty(self, weight_kg: float, distance_m: float) -> float:
        """Weight bites harder the further they go."""
        excess = weight_kg - 55.0
        return -0.055 * excess * (distance_m / 1600.0)

    def _freshness_effect(self, days: int) -> float:
        """First-up horses are typically underdone; long spells cost more.

        The sweet spot in Australian racing is roughly a 14-28 day back-up.
        """
        if days <= 7:
            return -0.18   # too quick a back-up
        if days <= 35:
            return 0.06
        if days <= 70:
            return -0.05
        if days <= 120:
            return -0.22   # first-up off a spell
        return -0.38

    def _experience_effect(self, horse: SimHorse) -> float:
        """Lightly raced horses still improving; older ones plateau."""
        return horse.improvement_rate * min(horse.starts, 12)

    # -- race generation ---------------------------------------------------

    def _pick_field(self, class_index: int, size: int,
                    distance_m: Optional[int] = None) -> list[SimHorse]:
        """Choose horses whose current class is near this race's class.

        Real fields are not random draws from the horse population. Two
        constraints matter and both are enforced here:

        1. Horses are placed by their connections into races they can win,
           so ability spread within a race is compressed. That is what
           makes the simulated market hard to beat -- as it should be.
        2. A thoroughbred cannot back up every few days. Minimum spacing is
           enforced and horses are weighted towards the 14-35 day window
           that dominates real Australian form lines.
        """
        candidates = [
            h for h in self.horses
            if not h.retired
            and h.days_since_run >= _MIN_DAYS_BETWEEN_RUNS
            and abs(h.current_class - class_index) <= 1
        ]
        if len(candidates) < size:
            candidates = [
                h for h in self.horses
                if not h.retired and h.days_since_run >= _MIN_DAYS_BETWEEN_RUNS
            ]
        if len(candidates) < size:
            return []

        # Weight by how "due" each horse is, so form lines cluster where
        # real ones do rather than spreading uniformly over the spell, and
        # by how well the trip suits. Trainers do not enter a 1000m
        # sprinter in the Cup; leaving that out produces fields containing
        # runners with no conceivable chance and a market to match.
        weights = [
            self._readiness(h.days_since_run) * self._distance_appeal(h, distance_m)
            for h in candidates
        ]
        if sum(weights) <= 0:
            weights = [1.0] * len(candidates)
        chosen: list[SimHorse] = []
        pool = list(candidates)
        pool_weights = list(weights)
        for _ in range(min(size, len(pool))):
            pick = self.rng.choices(range(len(pool)), weights=pool_weights)[0]
            chosen.append(pool.pop(pick))
            pool_weights.pop(pick)
        return chosen

    @staticmethod
    def _distance_appeal(horse: SimHorse, distance_m: Optional[int]) -> float:
        """How likely connections are to enter this horse over this trip."""
        if distance_m is None:
            return 1.0
        gap = abs(distance_m - horse.optimal_distance)
        return math.exp(-((gap / max(200.0, horse.distance_tolerance)) ** 2))

    @staticmethod
    def _readiness(days: int) -> float:
        """Relative likelihood a horse is entered, given days since its run."""
        if days < _MIN_DAYS_BETWEEN_RUNS:
            return 0.0
        if days <= 35:
            return 1.0
        if days <= 90:
            return 0.35   # spelling
        if days <= 200:
            return 0.15   # coming back from a longer let-up
        return 0.05

    def _true_performance(
        self,
        horse: SimHorse,
        jockey: SimJockey,
        trainer: SimTrainer,
        barrier: int,
        weight_kg: float,
        race: Race,
        field_size: int,
    ) -> float:
        return (
            horse.ability
            + self._experience_effect(horse)
            + jockey.skill
            + trainer.skill
            + self._barrier_penalty(barrier, race.track, race.distance_m, field_size)
            + self._distance_penalty(horse, race.distance_m)
            + self._condition_effect(horse, race.track_condition or 4)
            + self._weight_penalty(weight_kg, race.distance_m)
            + self._freshness_effect(horse.days_since_run)
            + self.rng.gauss(0, horse.consistency)
        )

    def _true_win_probabilities(self, strengths: list[float],
                                sds: list[float]) -> np.ndarray:
        """The genuine win probabilities implied by the race process.

        The race is decided by performance_i = strength_i + N(0, sd_i),
        with the winner being the argmax. The probability that runner i
        wins is therefore

            P(i) = integral phi_i(x) * product_{j != i} Phi_j(x) dx

        which is evaluated here by numerical quadrature.

        Two earlier attempts got this wrong in instructive ways.

        The first approximated it with a softmax at a single field-average
        noise level, but the race uses a *per-horse* sd. That mismatch is a
        systematic mispricing that varies with price, and it showed up as a
        20-point spread in blind flat-bet return across odds bands, where a
        correctly formed book must be flat.

        The second used Monte Carlo. That is unbiased but *noisy*, and
        noise in a price is not harmless: a runner lands in a low-priced
        band partly because its estimate happened to come out low, so
        conditional on the band the true chance is higher than the price
        implies. That errors-in-variables effect alone put a 5-point tilt
        into the longshot end of a market that was otherwise perfect.

        Quadrature has neither problem: it is deterministic and accurate to
        about 1e-9, so a control market with no deliberate bias comes out
        genuinely flat.
        """
        mu = np.asarray(strengths, dtype=float)
        sd = np.maximum(1e-9, np.asarray(sds, dtype=float))

        lo = float((mu - 9.0 * sd).min())
        hi = float((mu + 9.0 * sd).max())
        grid = np.linspace(lo, hi, 2048)

        z = (grid[:, None] - mu[None, :]) / sd[None, :]
        log_cdf = np.log(np.clip(ndtr(z), 1e-300, None))
        pdf = np.exp(-0.5 * z * z) / (sd[None, :] * math.sqrt(2.0 * math.pi))

        # product over j != i, computed in logs for stability
        others = np.exp(log_cdf.sum(axis=1)[:, None] - log_cdf)
        probs = np.trapezoid(pdf * others, grid, axis=0)

        probs = np.clip(probs, 1e-12, None)
        return probs / probs.sum()

    def _simulate_market(self, true_probs: np.ndarray) -> list[float]:
        """Produce decimal odds from the true win probabilities.

        The market observes the truth with *unbiased* noise in log-odds
        space, controlled by `market_efficiency`. Then a favourite-longshot
        bias is applied -- the well documented tendency for punters to
        overbet outsiders -- and finally an overround.

        The noise is deliberately unbiased: shrinking the signal by an
        efficiency factor would flatten every price in a systematic,
        trivially exploitable direction and make the model look far
        cleverer than it is.
        """
        log_p = np.log(np.clip(true_probs, 1e-12, None))
        sigma = math.sqrt(max(0.0, 1.0 / max(0.05, self.market_efficiency) - 1.0))
        noisy = log_p + self._nprng.standard_normal(len(log_p)) * sigma

        exp_vals = np.exp(noisy - noisy.max())
        probs = exp_vals / exp_vals.sum()

        biased = probs ** self.fl_exponent
        biased = biased / biased.sum()

        return [
            min(_MAX_ODDS,
                max(_MIN_ODDS, round(float(1.0 / (p * self.market_overround)), 2)))
            for p in biased
        ]

    def simulate_race(
        self,
        date: _dt.date,
        track: Track,
        race_number: int,
        class_index: Optional[int] = None,
        distance_m: Optional[int] = None,
        field_size: Optional[int] = None,
        track_condition: Optional[int] = None,
    ) -> Race:
        rng = self.rng
        self._race_counter += 1

        class_index = class_index if class_index is not None else min(
            len(CLASS_LADDER) - 1, int(abs(rng.gauss(0, 3.2))))
        distance_m = distance_m or rng.choice(COMMON_DISTANCES)
        field_size = field_size or rng.randint(6, 16)
        if track_condition is None:
            # Most Australian racing happens on Good ground.
            track_condition = rng.choices(
                [3, 4, 5, 6, 7, 8, 9],
                weights=[22, 34, 16, 11, 8, 6, 3],
            )[0]

        weather = Weather(
            temperature_c=round(rng.gauss(19, 6), 1),
            rainfall_mm_24h=round(max(0.0, (track_condition - 4) * rng.uniform(1.5, 5.0)), 1),
            rainfall_mm_72h=round(max(0.0, (track_condition - 4) * rng.uniform(3.0, 9.0)), 1),
            humidity_pct=round(min(100, max(20, rng.gauss(60, 15))), 1),
            wind_speed_kmh=round(abs(rng.gauss(12, 7)), 1),
        )

        race = Race(
            race_id=f"SIM{self._race_counter:07d}",
            track=track,
            date=date,
            race_number=race_number,
            distance_m=distance_m,
            name=f"{CLASS_LADDER[class_index]} Handicap ({distance_m}m)",
            start_time=_dt.datetime.combine(date, _dt.time(12, 0)) +
            _dt.timedelta(minutes=35 * race_number),
            track_condition=track_condition,
            class_level=CLASS_LADDER[class_index],
            prize_money=float(15000 * (1.55 ** class_index)),
            weather=weather,
        )

        field_horses = self._pick_field(class_index, field_size, distance_m)
        actual_size = len(field_horses)
        if actual_size < 4:
            return race  # not enough runners; caller will skip

        barriers = list(range(1, actual_size + 1))
        rng.shuffle(barriers)

        strengths: list[float] = []       # what the market can perceive
        performances: list[float] = []    # what actually decides the race
        pairs: list[tuple[SimHorse, SimJockey, SimTrainer, int, float]] = []

        for i, horse in enumerate(field_horses):
            jockey = rng.choice(self.jockeys)
            trainer = rng.choice(self.trainers)
            barrier = barriers[i]
            weight_kg = self._handicap_weight(horse, rng)

            expected = (
                horse.ability
                + self._experience_effect(horse)
                + jockey.skill
                + trainer.skill
                + self._barrier_penalty(barrier, track, distance_m, actual_size)
                + self._distance_penalty(horse, distance_m)
                + self._condition_effect(horse, track_condition)
                + self._weight_penalty(weight_kg, distance_m)
                + self._freshness_effect(horse.days_since_run)
            )
            # Private information: real, affects the result, and invisible
            # to every feature the model can build. The market prices most
            # of it; the model cannot price any of it.
            private = rng.gauss(0, self.private_info_sd)

            strengths.append(expected + self.market_sees_private * private)
            performances.append(
                expected + private
                + rng.gauss(0, horse.consistency * self.noise_scale)
            )
            pairs.append((horse, jockey, trainer, barrier, weight_kg))

        per_horse_sd = [h.consistency * self.noise_scale for h in field_horses]
        true_probs = self._true_win_probabilities(strengths, per_horse_sd)
        odds = self._simulate_market(true_probs)
        place_odds = self._simulate_place_market(odds, actual_size)

        # Higher performance finishes in front.
        order = sorted(range(actual_size), key=lambda i: -performances[i])
        placings = {idx: pos + 1 for pos, idx in enumerate(order)}

        winner_time = self._race_time(distance_m, track_condition, performances[order[0]])

        for i, (horse, jockey, trainer, barrier, weight_kg) in enumerate(pairs):
            position = placings[i]
            margin = round(max(0.0, (performances[order[0]] - performances[i]) * 2.6), 2)
            finish_time = round(winner_time + margin * 0.16, 2)
            runner = Runner(
                horse_id=horse.horse_id,
                name=horse.name,
                number=i + 1,
                barrier=barrier,
                weight_kg=weight_kg,
                jockey=jockey.name,
                trainer=trainer.name,
                age=date.year - horse.birth_year,
                sire=horse.sire,
                fixed_win_odds=odds[i],
                fixed_place_odds=place_odds[i],
                result_position=position,
                result_time_s=finish_time,
                result_margin_l=margin,
                history=list(horse.history),  # snapshot of form *before* this race
            )
            race.runners.append(runner)

            # Record this run into the horse's form for future races.
            horse.history.insert(0, PastRun(
                date=date,
                track_code=track.code,
                distance_m=distance_m,
                finish_position=position,
                field_size=actual_size,
                margin_l=margin,
                barrier=barrier,
                weight_kg=weight_kg,
                jockey=jockey.name,
                track_condition=track_condition,
                class_level=CLASS_LADDER[class_index],
                prize_money_total=race.prize_money,
                starting_price=odds[i],
                race_time_s=finish_time,
                last_600m_s=round(rng.gauss(34.5, 0.9) - performances[i] * 0.18, 2),
            ))
            horse.history = horse.history[:30]
            horse.starts += 1
            horse.days_since_run = 0

            # Horses that win move up in class; those that keep losing drop.
            if position == 1 and rng.random() < 0.62:
                horse.current_class = min(len(CLASS_LADDER) - 1, horse.current_class + 1)
            elif position > actual_size * 0.7 and rng.random() < 0.18:
                horse.current_class = max(0, horse.current_class - 1)
            if horse.starts > 45 and rng.random() < 0.06:
                horse.retired = True

        return race

    @staticmethod
    def _handicap_weight(horse: SimHorse, rng: random.Random) -> float:
        """Weight allotted by the handicapper today.

        This deliberately does NOT read `horse.ability`. An earlier version
        did, and the result was a feature correlating 0.96 with the latent
        ability term -- a single column handing the model 92% of the hidden
        truth, available even to a first-starter with no form at all. That
        made "can the model recover the signal?" a vacuous question and
        rendered the fitted blend weights uninterpretable.

        Real handicappers work from the public record: what the horse has
        beaten, and what it carried when it did. So the weight here is
        built from *observed past finishing positions* -- lagged, noisy,
        and unavailable before a horse has raced -- exactly like the real
        thing. It remains correlated with ability, because good horses do
        win and do get weighted, but only through the form the model can
        also see.
        """
        recent = [r for r in horse.history[:6] if r.finish_position and r.field_size]
        if not recent:
            # No public record yet: near the bottom of the handicap.
            return round(54.0 + rng.uniform(-0.5, 1.0), 1)

        # Mean finishing percentile, 1.0 for winning, 0.0 for running last.
        scores = [1.0 - (r.finish_position - 1) / max(1, r.field_size - 1)
                  for r in recent]
        rating = sum(scores) / len(scores)

        # Handicappers also weight on class: winning better races costs more.
        class_bump = 0.35 * min(6, horse.current_class)

        return round(53.0 + 4.0 * rating + class_bump + rng.gauss(0, 1.1), 1)

    def _simulate_place_market(self, win_odds: list[float],
                               field_size: int) -> list[Optional[float]]:
        """Derive place prices from the win market.

        A naive "place odds = a fraction of win odds" rule is badly wrong
        and produces a market in which almost every runner looks like
        value -- which is exactly the sort of artefact that makes a betting
        model look profitable when it is not.

        Instead the place market is built the way a real one is: convert
        win prices to probabilities, derive genuine place probabilities
        under the discounted-Harville model, then load an overround on top.
        """
        from ..betting.exotics import place_probabilities, places_paid

        n_places = places_paid(field_size)
        if n_places == 0:
            return [None] * len(win_odds)

        raw = [1.0 / o for o in win_odds]
        total = sum(raw)
        win_probs = [p / total for p in raw]

        place_probs = place_probabilities(win_probs, field_size)

        # Place books carry a margin too, typically a little tighter per
        # place than the win book.
        margin = 1.0 + (self.market_overround - 1.0) * 0.75
        prices: list[Optional[float]] = []
        for p in place_probs:
            if p <= 0:
                prices.append(_MAX_ODDS)
                continue
            price = 1.0 / (min(0.995, float(p)) * margin)
            prices.append(round(min(_MAX_ODDS, max(1.02, price)), 2))
        return prices

    @staticmethod
    def _race_time(distance_m: int, condition: int, winner_strength: float) -> float:
        """Plausible winning time. Wet ground is slower."""
        base_speed = 16.6  # metres per second, roughly
        going_drag = 1.0 - 0.012 * max(0, condition - 4)
        speed = base_speed * going_drag * (1.0 + 0.006 * winner_strength)
        return round(distance_m / speed, 2)

    # -- season ------------------------------------------------------------

    def simulate_season(
        self,
        start: _dt.date,
        days: int = 365,
        meetings_per_day: int = 3,
        races_per_meeting: int = 8,
    ) -> list[Race]:
        """Run a whole season in date order and return every race."""
        races: list[Race] = []
        current = start
        for _ in range(days):
            # Ageing: every horse gets a day older since its last run.
            for horse in self.horses:
                horse.days_since_run += 1

            # Australian racing is concentrated on weekends but runs midweek too.
            is_weekend = current.weekday() >= 5
            todays_meetings = meetings_per_day if is_weekend else max(1, meetings_per_day - 1)

            for track in self.rng.sample(self.tracks, min(todays_meetings, len(self.tracks))):
                # One track condition for the whole meeting, as in reality.
                condition = self.rng.choices(
                    [3, 4, 5, 6, 7, 8, 9], weights=[22, 34, 16, 11, 8, 6, 3])[0]
                for race_number in range(1, races_per_meeting + 1):
                    race = self.simulate_race(
                        current, track, race_number, track_condition=condition)
                    if race.runners:
                        races.append(race)
            current += _dt.timedelta(days=1)
        return races

    def to_meetings(self, races: list[Race]) -> list[Meeting]:
        """Group a flat race list back into meetings."""
        grouped: dict[tuple[str, _dt.date], Meeting] = {}
        for race in races:
            key = (race.track.code, race.date)
            if key not in grouped:
                grouped[key] = Meeting(
                    meeting_id=f"{race.track.code}-{race.date.isoformat()}",
                    track=race.track,
                    date=race.date,
                    track_condition=race.track_condition,
                )
            grouped[key].races.append(race)
        return list(grouped.values())
