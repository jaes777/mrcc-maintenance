"""A race simulator, for testing the pipeline.

**Nothing produced here is real.** These are invented horses running invented
races. The simulator exists so the whole pipeline - features, model, blending,
backtesting, bet selection - can be exercised end to end and its arithmetic
verified, in an environment with no access to real racing data.

Any performance figure measured on synthetic data tells you that the code
works. It tells you **nothing whatsoever** about how the model will perform on
real races, and it must never be presented as a prediction of real performance.

What it does model, because these are the properties the pipeline needs to be
tested against:

- Horses with a persistent latent ability plus per-race noise, so that form
  genuinely carries signal but does not determine the result.
- Realistic Australian field sizes, averaging about ten runners.
- Jockeys and trainers of differing quality, but with most of their apparent
  strike-rate differences driven by the horses they get - which is the real
  confounding that makes raw jockey statistics so misleading.
- Weight allocated by ability, reproducing the trap where higher weight
  correlates *positively* with winning.
- A market that is calibrated but noisy, carries an overround, and shows the
  favourite-longshot bias.
- Track conditions that some horses handle and others do not.

The `market_skill` parameter controls how much the market knows. At 1.0 the
market sees the true probabilities and cannot be beaten; at 0.0 it is noise.
Real racing markets sit close to the top of that range, which is why the
default is deliberately high.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

TRACKS = [
    ("Flemington", "VIC"), ("Caulfield", "VIC"), ("Moonee Valley", "VIC"),
    ("Randwick", "NSW"), ("Rosehill", "NSW"), ("Warwick Farm", "NSW"),
    ("Eagle Farm", "QLD"), ("Doomben", "QLD"), ("Morphettville", "SA"),
    ("Ascot", "WA"), ("Sandown", "VIC"), ("Newcastle", "NSW"),
]
DISTANCES = [1000, 1100, 1200, 1400, 1600, 1800, 2000, 2400]
CLASSES = ["Maiden", "BM58", "BM64", "BM70", "BM78", "Listed", "Group 3", "Group 1"]


@dataclass
class SimConfig:
    n_horses: int = 3000
    n_races: int = 6000
    start_date: datetime = datetime(2021, 1, 1)
    days_span: int = 1500
    mean_field_size: float = 9.6
    ability_sd: float = 1.0
    race_noise_sd: float = 0.80     # how much luck there is in a race
    market_skill: float = 0.92      # 1.0 = market knows the truth exactly
    market_noise_sd: float = 0.25
    overround: float = 1.16
    longshot_bias: float = 0.10     # how much longshots are over-bet
    seed: int = 7


def simulate(config: SimConfig | None = None) -> pd.DataFrame:
    """Generate a synthetic racing dataset in the canonical schema.

    Returns a row-per-runner frame with results, prices, and everything the
    feature layer needs.
    """
    cfg = config or SimConfig()
    rng = np.random.default_rng(cfg.seed)

    # --- population ------------------------------------------------------
    ability = rng.normal(0, cfg.ability_sd, cfg.n_horses)
    # Some horses genuinely improve on wet ground and others hate it.
    wet_affinity = rng.normal(0, 0.45, cfg.n_horses)
    # Preferred distance, so distance suitability is a real effect.
    preferred_distance = rng.choice(DISTANCES, cfg.n_horses)
    distance_tolerance = rng.uniform(200, 700, cfg.n_horses)
    horse_age_start = rng.integers(2, 6, cfg.n_horses)
    sexes = rng.choice(["G", "M", "C", "F", "H"], cfg.n_horses, p=[.45, .25, .12, .13, .05])

    n_jockeys, n_trainers = 220, 350
    jockey_skill = rng.normal(0, 0.18, n_jockeys)      # small true effect
    trainer_skill = rng.normal(0, 0.15, n_trainers)
    # Better trainers get better horses - this is the confounding that makes
    # raw trainer strike rates look far more predictive than they are.
    trainer_horse_bias = rng.normal(0, 0.5, n_trainers)
    horse_trainer = rng.integers(0, n_trainers, cfg.n_horses)
    ability += trainer_horse_bias[horse_trainer]

    rows: list[dict] = []
    horse_starts = np.zeros(cfg.n_horses, dtype=int)
    horse_last_run = np.full(cfg.n_horses, np.nan)

    for race_index in range(cfg.n_races):
        day_offset = int(race_index / cfg.n_races * cfg.days_span)
        race_dt = cfg.start_date + timedelta(
            days=day_offset, hours=int(rng.integers(11, 18)), minutes=int(rng.integers(0, 60))
        )
        track, state = TRACKS[rng.integers(0, len(TRACKS))]
        distance = int(rng.choice(DISTANCES))
        race_class = CLASSES[min(int(abs(rng.normal(0, 2))), len(CLASSES) - 1)]
        prizemoney = float(np.exp(9.5 + CLASSES.index(race_class) * 0.55))

        # Track condition: mostly Good, sometimes wet.
        condition = float(np.clip(rng.gamma(shape=2.2, scale=1.7) + 2.0, 1, 10))
        is_wet = condition >= 5

        field_size = int(np.clip(rng.poisson(cfg.mean_field_size - 2) + 3, 4, 20))
        field = rng.choice(cfg.n_horses, size=field_size, replace=False)

        # --- each runner's true strength this race ------------------------
        strength = ability[field].copy()
        if is_wet:
            strength += wet_affinity[field] * ((condition - 4) / 3.0)

        distance_penalty = np.abs(distance - preferred_distance[field]) / distance_tolerance[field]
        strength -= 0.35 * distance_penalty

        jockeys = rng.integers(0, n_jockeys, field_size)
        # Better horses attract better jockeys - more confounding.
        jockeys = np.where(
            rng.random(field_size) < 0.4,
            np.argsort(-jockey_skill)[: max(field_size, 1)][:field_size],
            jockeys,
        )
        strength += jockey_skill[jockeys]
        trainers = horse_trainer[field]
        strength += trainer_skill[trainers]

        barriers = rng.permutation(field_size) + 1
        # Inside barriers help, and help more over short trips.
        barrier_effect = -0.03 * (barriers - 1) * (1.0 if distance <= 1200 else 0.4)
        strength += barrier_effect

        # Weight is allocated by ability. This deliberately reproduces the trap
        # where carrying more weight correlates POSITIVELY with winning, even
        # though its causal effect is negative.
        weights = 54.0 + 4.0 * (strength - strength.mean()) / max(strength.std(), 1e-6)
        weights = np.clip(weights, 52.0, 62.0)
        strength -= 0.10 * (weights - weights.mean())

        days_since = np.where(
            np.isnan(horse_last_run[field]),
            np.nan,
            (race_dt - cfg.start_date).days - horse_last_run[field],
        )
        # A long spell costs a little first-up.
        fresh_penalty = np.where(np.isfinite(days_since) & (days_since > 90), -0.2, 0.0)
        strength += fresh_penalty

        # --- true win probabilities and the result ------------------------
        # Gumbel noise on strength gives an exact multinomial-logit structure,
        # which is the standard model for a race.
        scale = cfg.race_noise_sd
        true_logits = strength / scale
        true_p = np.exp(true_logits - true_logits.max())
        true_p /= true_p.sum()

        noisy = strength + rng.gumbel(0, scale, field_size)
        finish_order = np.argsort(-noisy)
        finish_position = np.empty(field_size, dtype=int)
        finish_position[finish_order] = np.arange(1, field_size + 1)
        margins = (noisy[finish_order[0]] - noisy) * 2.0

        # --- the market ---------------------------------------------------
        # The market sees a blend of the truth and noise, then a margin and a
        # longshot bias are applied on top.
        market_logits = (
            cfg.market_skill * true_logits
            + (1 - cfg.market_skill) * rng.normal(0, 1, field_size)
            + rng.normal(0, cfg.market_noise_sd, field_size)
        )
        market_p = np.exp(market_logits - market_logits.max())
        market_p /= market_p.sum()

        # Favourite-longshot bias: longshots are backed more than they deserve,
        # so their implied probability is inflated and their price is shorter
        # than it should be. Raising to a power below 1 flattens the
        # distribution, which does exactly that.
        biased = market_p ** (1.0 - cfg.longshot_bias)
        biased /= biased.sum()
        # Apply the margin. The book must sum to MORE than 1 - that is the
        # bookmaker's edge - so implied probabilities are scaled UP by the
        # overround and the prices come down accordingly.
        odds = 1.0 / (biased * cfg.overround)
        odds = np.round(np.clip(odds, 1.04, 999.0), 2)

        race_id = f"SYN{race_index:06d}"
        for slot in range(field_size):
            horse_idx = int(field[slot])
            rows.append({
                "race_id": race_id,
                "race_datetime": race_dt,
                "track": track,
                "state": state,
                "race_number": int(race_index % 9) + 1,
                "distance_m": distance,
                "track_condition": f"{'Heavy' if condition >= 8 else 'Soft' if condition >= 5 else 'Good'} {int(round(condition))}",
                "race_class": race_class,
                "prizemoney": prizemoney,
                "runner_id": f"H{horse_idx:05d}",
                "horse": f"Horse {horse_idx:05d}",
                "tab_number": slot + 1,
                "barrier": int(barriers[slot]),
                "weight_kg": round(float(weights[slot]), 1),
                "jockey": f"J{int(jockeys[slot]):03d}",
                "trainer": f"T{int(trainers[slot]):03d}",
                "age": int(horse_age_start[horse_idx] + day_offset / 365),
                "sex": sexes[horse_idx],
                "days_since_last_run": float(days_since[slot]) if np.isfinite(days_since[slot]) else np.nan,
                "odds_decimal": float(odds[slot]),
                "bsp": float(odds[slot]),
                "finish_position": int(finish_position[slot]),
                "margin_l": round(float(max(margins[slot], 0.0)), 2),
                "scratched": False,
                "_true_p": float(true_p[slot]),   # ground truth, for testing only
            })
            horse_starts[horse_idx] += 1
            horse_last_run[horse_idx] = (race_dt - cfg.start_date).days

    return pd.DataFrame(rows)


def simulate_upcoming_race(seed: int = 42, field_size: int = 10) -> pd.DataFrame:
    """A single unresolved race, for testing the prediction path.

    Same schema, but `finish_position` is NaN because it has not run yet.
    """
    rng = np.random.default_rng(seed)
    track, state = TRACKS[rng.integers(0, len(TRACKS))]
    race_dt = datetime.now() + timedelta(hours=3)
    strength = rng.normal(0, 1.0, field_size)
    logits = strength / 1.6
    p = np.exp(logits - logits.max())
    p /= p.sum()
    odds = np.round(np.clip(1.16 / p, 1.04, 999), 2)

    return pd.DataFrame([{
        "race_id": "UPCOMING001",
        "race_datetime": race_dt,
        "track": track,
        "state": state,
        "race_number": 5,
        "distance_m": 1400,
        "track_condition": "Good 4",
        "race_class": "BM70",
        "prizemoney": 50000.0,
        "runner_id": f"H{9000 + i:05d}",
        "horse": f"Horse {9000 + i:05d}",
        "tab_number": i + 1,
        "barrier": i + 1,
        "weight_kg": round(float(55 + rng.normal(0, 2)), 1),
        "jockey": f"J{int(rng.integers(0, 200)):03d}",
        "trainer": f"T{int(rng.integers(0, 300)):03d}",
        "age": int(rng.integers(3, 8)),
        "sex": "G",
        "odds_decimal": float(odds[i]),
        "finish_position": np.nan,
        "scratched": False,
    } for i in range(field_size)])
