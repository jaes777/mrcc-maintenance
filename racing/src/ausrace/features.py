"""Feature engineering.

This is where "every variable about the horse, the race and the day" becomes
numbers a model can use.

The overriding constraint here is **no leakage**. Every feature for a race must
be computed using only information that existed before that race jumped. This
sounds obvious and is violated constantly - the usual way is to compute a
horse's career win rate over the whole dataset and then use it to predict a
race that is included in that career. A model built that way looks
extraordinary in backtest and loses money immediately in real life.

The guarantee here is structural rather than a matter of care: races are
processed strictly in chronological order, features are emitted from the
running state, and only *then* is the running state updated with what happened.
It is not possible for a future result to reach a past feature.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .betting import devig
from .schema import UNKNOWN_LOSS_POSITION, conform, has_full_finishing_order

# Bayesian shrinkage strength for strike rates. A jockey with 2 wins from 3
# rides does not have a 67% strike rate; shrinking towards the population mean
# with a prior weight of K rides stops the model chasing tiny samples.
SHRINKAGE_HORSE = 4.0
SHRINKAGE_JOCKEY = 40.0
SHRINKAGE_TRAINER = 40.0
SHRINKAGE_COMBO = 25.0

# Distance bands used for "does this horse handle this trip".
DISTANCE_BANDS = [(0, 1200, "SPRINT"), (1200, 1600, "MILE"),
                  (1600, 2100, "MIDDLE"), (2100, 9999, "STAYING")]


def _missing(value: Any) -> bool:
    """True for None, NaN or an empty string - the shapes 'no data' takes here."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return value == ""


def distance_band(metres: float) -> str:
    if not np.isfinite(metres):
        return "UNKNOWN"
    for low, high, name in DISTANCE_BANDS:
        if low <= metres < high:
            return name
    return "UNKNOWN"


@dataclass
class _Record:
    """Running tally for one entity (horse, jockey, trainer, or a pairing)."""

    starts: int = 0
    wins: int = 0
    places: int = 0
    # Starts for which we actually know whether the horse placed. On a
    # winner-only data source this stays at zero and the place rate correctly
    # falls back to the population prior instead of inventing a number.
    place_starts: int = 0
    prize: float = 0.0
    by_key: dict[Any, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    recent: deque = field(default_factory=lambda: deque(maxlen=10))
    last_datetime: Any = None
    last_weight: float = float("nan")
    last_odds: float = float("nan")

    def strike(self, prior: float, k: float) -> float:
        return (self.wins + prior * k) / (self.starts + k)

    def place_strike(self, prior: float, k: float) -> float:
        return (self.places + prior * k) / (self.place_starts + k)

    def keyed_strike(self, key: Any, prior: float, k: float) -> float:
        starts, wins = self.by_key.get(key, [0, 0])
        return (wins + prior * k) / (starts + k)

    def keyed_starts(self, key: Any) -> int:
        return self.by_key.get(key, [0, 0])[0]


FEATURE_COLUMNS = [
    # --- market (by far the strongest single signal) ----------------------
    "mkt_prob", "mkt_log_odds", "mkt_rank", "mkt_prob_rel", "field_overround",
    # --- horse form ------------------------------------------------------
    "h_starts", "h_win_rate", "h_place_rate", "h_prize_per_start",
    "h_last_finish", "h_finish_avg3", "h_finish_avg5", "h_best_finish5",
    "h_days_since_run", "h_first_up", "h_second_up", "h_spell_long",
    "h_form_momentum", "h_avg_margin3",
    # --- horse suitability ------------------------------------------------
    "h_dist_starts", "h_dist_win_rate", "h_track_starts", "h_track_win_rate",
    "h_going_starts", "h_going_win_rate", "h_class_starts", "h_class_win_rate",
    "h_dist_change", "h_wet_specialist",
    # --- weight / class movement -----------------------------------------
    "h_weight", "h_weight_rel", "h_weight_change", "h_class_move",
    "h_last_odds_log", "h_odds_drift",
    # --- barrier ---------------------------------------------------------
    "barrier", "barrier_rel", "barrier_wide", "barrier_inside",
    # --- people ----------------------------------------------------------
    "j_rides", "j_win_rate", "j_place_rate", "j_track_win_rate",
    "t_runners", "t_win_rate", "t_place_rate", "t_track_win_rate",
    "jt_starts", "jt_win_rate",
    # --- race context ----------------------------------------------------
    "field_size", "distance_m", "condition_num", "is_wet", "log_prize",
    "temp_c", "rain_mm_24h", "wind_kph", "humidity_pct",
    "age", "is_mare", "is_gelding",
]


def build_features(
    frame: pd.DataFrame,
    devig_method: str = "shin",
    include_market: bool = True,
) -> pd.DataFrame:
    """Add the feature columns to a canonical runner frame.

    Handles historical rows (which have results) and upcoming rows (which do
    not) in the same pass, so an upcoming race automatically gets features
    built from every prior race in the dataset.

    Set `include_market=False` to build a pure-fundamentals model that ignores
    the betting market. That model will be substantially less accurate, but it
    is the only way to find out whether you know anything the market doesn't.
    """
    data = conform(frame, strict=True)
    data = data.sort_values(["race_datetime", "race_id"], kind="mergesort").reset_index(drop=True)

    # Whether this dataset records real finishing positions at all. On a
    # winner-only source, place statistics are censored: we can see that a
    # winner placed, but never that a loser did. Counting only the observable
    # half would turn the "place rate" into a disguised win rate, so place
    # tracking is switched off entirely and the feature falls back to the
    # population prior - which is honest about knowing nothing.
    full_order = has_full_finishing_order(data)

    if data["finish_position"].notna().any() and not full_order:
        print(
            "[features] This dataset records only WINNERS, not full finishing "
            "order (Betfair's free files are like this). Form features based on "
            "finishing position - h_last_finish, h_finish_avg3, h_finish_avg5, "
            "h_best_finish5, h_form_momentum, h_place_rate - will be mostly "
            "empty, and place probabilities cannot be validated against results. "
            "A source with full finishing order (e.g. Punting Form) is a large "
            "upgrade if you want those features."
        )

    horses: dict[str, _Record] = defaultdict(_Record)
    jockeys: dict[str, _Record] = defaultdict(_Record)
    trainers: dict[str, _Record] = defaultdict(_Record)
    combos: dict[tuple[str, str], _Record] = defaultdict(_Record)

    # Population base rates, updated as we sweep forward. Starting values are
    # the long-run averages for Australian racing: an average field is around
    # 10 runners, so a random horse wins about 10% of the time and places
    # (top 3) about 30% of the time.
    pop = {"starts": 0, "wins": 0, "places": 0, "place_starts": 0}

    rows: list[dict] = []

    for race_id, race in data.groupby("race_id", sort=False):
        # Base rates from every race that has already been processed - never
        # including the current one, which is why they are computed here at the
        # top of the loop rather than after the update block below.
        base_win = (pop["wins"] + 100 * 0.10) / (pop["starts"] + 1000)
        base_place = (pop["places"] + 300 * 0.10) / (pop["place_starts"] + 1000)

        live = race[~race["scratched"]]
        n_live = max(len(live), 1)
        race_dt = race["race_datetime"].iloc[0]
        distance = float(race["distance_m"].iloc[0]) if pd.notna(race["distance_m"].iloc[0]) else np.nan
        track = str(race["track"].iloc[0])
        band = distance_band(distance)
        going = str(race["condition_band"].iloc[0])
        race_class = str(race["race_class"].iloc[0]) if pd.notna(race["race_class"].iloc[0]) else "UNKNOWN"
        prize = float(race["prizemoney"].iloc[0]) if pd.notna(race["prizemoney"].iloc[0]) else np.nan

        # --- market ------------------------------------------------------
        odds = race["odds_decimal"].to_numpy(dtype=float)
        if include_market and np.isfinite(odds).sum() >= 2:
            mkt = devig(odds, method=devig_method)
            book = float(np.nansum(1.0 / odds[np.isfinite(odds) & (odds > 1)]))
        else:
            mkt = np.full(len(race), np.nan)
            book = np.nan
        rank = pd.Series(mkt).rank(ascending=False, method="min").to_numpy()

        weights = race["weight_kg"].to_numpy(dtype=float)
        mean_weight = np.nanmean(weights) if np.isfinite(weights).any() else np.nan
        barriers = race["barrier"].to_numpy(dtype=float)

        # Pull every column the inner loop needs out as a plain numpy array
        # first. `iterrows()` builds a fresh pandas Series per row, which was
        # roughly a third of the total runtime on a full dataset.
        col = {
            name: race[name].to_numpy()
            for name in ("runner_id", "jockey", "trainer", "sex")
        }
        num = {
            name: race[name].to_numpy(dtype=float)
            for name in ("days_since_last_run", "weight_kg", "barrier",
                         "condition_num", "is_wet", "temp_c", "rain_mm_24h",
                         "wind_kph", "humidity_pct", "age")
        }

        for pos in range(len(race)):
            hid = str(col["runner_id"][pos])
            raw_jockey, raw_trainer = col["jockey"][pos], col["trainer"][pos]
            jockey = str(raw_jockey) if not _missing(raw_jockey) else "UNKNOWN_J"
            trainer = str(raw_trainer) if not _missing(raw_trainer) else "UNKNOWN_T"

            h = horses[hid]
            j = jockeys[jockey]
            t = trainers[trainer]
            c = combos[(jockey, trainer)]

            # ---- days since last run, computed from our own history so it
            # ---- is correct even when the source omits it.
            if h.last_datetime is not None and pd.notna(race_dt):
                days = (race_dt - h.last_datetime).days
            else:
                days = num["days_since_last_run"][pos]

            recent = list(h.recent)                    # most recent last
            finishes = [r["finish"] for r in recent if np.isfinite(r["finish"])]
            last3 = finishes[-3:]
            last5 = finishes[-5:]
            margins = [r["margin"] for r in recent[-3:] if np.isfinite(r["margin"])]

            # Momentum: are the last two runs better than the three before?
            if len(finishes) >= 5:
                momentum = float(np.mean(finishes[-5:-2]) - np.mean(finishes[-2:]))
            else:
                momentum = np.nan

            last_weight = h.last_weight
            weight = num["weight_kg"][pos]
            prev_dist = recent[-1]["distance"] if recent else np.nan

            row = {
                "race_id": race_id,
                "runner_id": hid,
                # --- market ---
                "mkt_prob": mkt[pos],
                "mkt_log_odds": np.log(odds[pos]) if np.isfinite(odds[pos]) and odds[pos] > 1 else np.nan,
                "mkt_rank": rank[pos],
                "mkt_prob_rel": mkt[pos] * n_live if np.isfinite(mkt[pos]) else np.nan,
                "field_overround": book,
                # --- horse form ---
                "h_starts": h.starts,
                "h_win_rate": h.strike(base_win, SHRINKAGE_HORSE),
                "h_place_rate": h.place_strike(base_place, SHRINKAGE_HORSE),
                "h_prize_per_start": h.prize / h.starts if h.starts else np.nan,
                "h_last_finish": finishes[-1] if finishes else np.nan,
                "h_finish_avg3": float(np.mean(last3)) if last3 else np.nan,
                "h_finish_avg5": float(np.mean(last5)) if last5 else np.nan,
                "h_best_finish5": float(np.min(last5)) if last5 else np.nan,
                "h_days_since_run": days,
                "h_first_up": 1.0 if (np.isfinite(days) and days >= 60) else 0.0,
                "h_second_up": 1.0 if (recent and np.isfinite(recent[-1]["days_before"])
                                       and recent[-1]["days_before"] >= 60) else 0.0,
                "h_spell_long": 1.0 if (np.isfinite(days) and days >= 180) else 0.0,
                "h_form_momentum": momentum,
                "h_avg_margin3": float(np.mean(margins)) if margins else np.nan,
                # --- suitability ---
                "h_dist_starts": h.keyed_starts(("d", band)),
                "h_dist_win_rate": h.keyed_strike(("d", band), base_win, SHRINKAGE_HORSE),
                "h_track_starts": h.keyed_starts(("t", track)),
                "h_track_win_rate": h.keyed_strike(("t", track), base_win, SHRINKAGE_HORSE),
                "h_going_starts": h.keyed_starts(("g", going)),
                "h_going_win_rate": h.keyed_strike(("g", going), base_win, SHRINKAGE_HORSE),
                "h_class_starts": h.keyed_starts(("c", race_class)),
                "h_class_win_rate": h.keyed_strike(("c", race_class), base_win, SHRINKAGE_HORSE),
                "h_dist_change": (distance - prev_dist) if (np.isfinite(distance) and
                                                            np.isfinite(prev_dist)) else np.nan,
                "h_wet_specialist": (h.keyed_strike(("g", "SOFT"), base_win, SHRINKAGE_HORSE)
                                     + h.keyed_strike(("g", "HEAVY"), base_win, SHRINKAGE_HORSE)) / 2
                                    - h.keyed_strike(("g", "GOOD"), base_win, SHRINKAGE_HORSE),
                # --- weight / class ---
                "h_weight": weight,
                "h_weight_rel": weight - mean_weight if np.isfinite(mean_weight) else np.nan,
                "h_weight_change": weight - last_weight if np.isfinite(last_weight) else np.nan,
                "h_class_move": (np.log1p(prize) - np.log1p(recent[-1]["prize"]))
                                if (recent and np.isfinite(prize) and np.isfinite(recent[-1]["prize"]))
                                else np.nan,
                "h_last_odds_log": np.log(h.last_odds) if np.isfinite(h.last_odds) else np.nan,
                "h_odds_drift": (np.log(odds[pos]) - np.log(h.last_odds))
                                if (np.isfinite(h.last_odds) and np.isfinite(odds[pos]) and odds[pos] > 1)
                                else np.nan,
                # --- barrier ---
                "barrier": num["barrier"][pos],
                "barrier_rel": num["barrier"][pos] / n_live,
                "barrier_wide": 1.0 if num["barrier"][pos] > 0.7 * n_live else 0.0,
                "barrier_inside": 1.0 if num["barrier"][pos] <= 3 else 0.0,
                # --- people ---
                "j_rides": j.starts,
                "j_win_rate": j.strike(base_win, SHRINKAGE_JOCKEY),
                "j_place_rate": j.place_strike(base_place, SHRINKAGE_JOCKEY),
                "j_track_win_rate": j.keyed_strike(("t", track), base_win, SHRINKAGE_JOCKEY),
                "t_runners": t.starts,
                "t_win_rate": t.strike(base_win, SHRINKAGE_TRAINER),
                "t_place_rate": t.place_strike(base_place, SHRINKAGE_TRAINER),
                "t_track_win_rate": t.keyed_strike(("t", track), base_win, SHRINKAGE_TRAINER),
                "jt_starts": c.starts,
                "jt_win_rate": c.strike(base_win, SHRINKAGE_COMBO),
                # --- race context ---
                "field_size": n_live,
                "distance_m": distance,
                "condition_num": num["condition_num"][pos],
                "is_wet": num["is_wet"][pos],
                "log_prize": np.log1p(prize) if np.isfinite(prize) else np.nan,
                "temp_c": num["temp_c"][pos],
                "rain_mm_24h": num["rain_mm_24h"][pos],
                "wind_kph": num["wind_kph"][pos],
                "humidity_pct": num["humidity_pct"][pos],
                "age": num["age"][pos],
                "is_mare": 1.0 if str(col["sex"][pos]).upper().startswith(("M", "F")) else 0.0,
                "is_gelding": 1.0 if str(col["sex"][pos]).upper().startswith("G") else 0.0,
            }
            rows.append(row)

        # ---- Only now, after every feature in this race has been emitted,
        # ---- do we fold this race's results into the running state.
        finishes_col = race["finish_position"].to_numpy(dtype=float)
        scratched_col = race["scratched"].to_numpy(dtype=bool)
        margin_col = race["margin_l"].to_numpy(dtype=float)
        odds_col = race["odds_decimal"].to_numpy(dtype=float)

        for idx in range(len(race)):
            finish = finishes_col[idx]
            if np.isnan(finish) or scratched_col[idx]:
                continue

            hid = str(col["runner_id"][idx])
            raw_jockey, raw_trainer = col["jockey"][idx], col["trainer"][idx]
            jockey = str(raw_jockey) if not _missing(raw_jockey) else "UNKNOWN_J"
            trainer = str(raw_trainer) if not _missing(raw_trainer) else "UNKNOWN_T"

            # A winner-only source records every loser as UNKNOWN_LOSS_POSITION.
            # Counting that as "finished 99th" would poison every form feature
            # built on finishing position, and would also wrongly record the
            # horse as not having placed when we simply do not know.
            position_known = finish != UNKNOWN_LOSS_POSITION
            won = 1 if finish == 1 else 0
            if full_order and position_known:
                placed = 1 if finish <= 3 else 0
            else:
                # Either this runner's position is unknown, or the whole source
                # is winner-only and place data is censored. Either way it must
                # not be counted, in neither the numerator nor the denominator.
                placed = None
            h = horses[hid]
            prev_dt = h.last_datetime

            for record, key_sets in (
                (h, [("d", band), ("t", track), ("g", going), ("c", race_class)]),
                (jockeys[jockey], [("t", track)]),
                (trainers[trainer], [("t", track)]),
                (combos[(jockey, trainer)], []),
            ):
                record.starts += 1
                record.wins += won
                if placed is not None:
                    record.places += placed
                    record.place_starts += 1
                for key in key_sets:
                    entry = record.by_key[key]
                    entry[0] += 1
                    entry[1] += won

            h.prize += float(prize) if np.isfinite(prize) else 0.0
            h.recent.append({
                # NaN, not 99, when the source does not know the real position.
                "finish": float(finish) if position_known else np.nan,
                "margin": margin_col[idx],
                "distance": distance,
                "prize": prize,
                "days_before": (race_dt - prev_dt).days if prev_dt is not None else np.nan,
            })
            h.last_datetime = race_dt
            h.last_weight = num["weight_kg"][idx]
            h.last_odds = odds_col[idx]

            pop["starts"] += 1
            pop["wins"] += won
            if placed is not None:
                pop["places"] += placed
                pop["place_starts"] += 1

    features = pd.DataFrame(rows)
    merged = data.merge(features, on=["race_id", "runner_id"], how="left", validate="one_to_one")

    merged["won"] = (merged["finish_position"] == 1).astype(float)
    merged.loc[merged["finish_position"].isna(), "won"] = np.nan

    # `placed` is only knowable when the source records real finishing
    # positions. On a winner-only source it stays NaN rather than being
    # silently set to 0, which would train a place model on a fiction.
    merged["placed"] = (merged["finish_position"] <= 3).astype(float)
    merged.loc[merged["finish_position"].isna(), "placed"] = np.nan
    merged.loc[merged["finish_position"] == UNKNOWN_LOSS_POSITION, "placed"] = np.nan

    return merged


def feature_matrix(frame: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Extract the numeric feature block, keeping only columns that exist and
    actually vary. Constant columns are dropped - they cost training time and
    contribute nothing."""
    columns = columns or FEATURE_COLUMNS
    present = [c for c in columns if c in frame.columns]
    matrix = frame[present].astype(float)
    varying = [c for c in present if matrix[c].nunique(dropna=True) > 1]
    return matrix[varying], varying


def assert_no_leakage(frame: pd.DataFrame) -> list[str]:
    """Sanity checks that catch the classic leakage mistakes.

    Not a proof, but it catches the errors that actually happen: a feature
    correlating suspiciously with the result, or a horse's career-start count
    failing to increase monotonically through time.
    """
    problems: list[str] = []
    if "won" not in frame.columns or frame["won"].isna().all():
        return ["No results present - cannot check for leakage."]

    resolved = frame[frame["won"].notna()]

    # A horse's prior-starts count must never decrease over time.
    ordered = resolved.sort_values("race_datetime")
    for hid, group in ordered.groupby("runner_id"):
        if len(group) < 2:
            continue
        starts = group["h_starts"].to_numpy()
        if np.any(np.diff(starts) < 0):
            problems.append(
                f"h_starts decreases over time for runner {hid} - history is "
                f"not being accumulated in chronological order."
            )
            break

    # Any single feature correlating above ~0.5 with the outcome is
    # implausible in racing and almost certainly leakage.
    for col in FEATURE_COLUMNS:
        if col not in resolved.columns:
            continue
        series = resolved[col]
        if series.notna().sum() < 100 or series.nunique() < 3:
            continue
        corr = series.corr(resolved["won"])
        if np.isfinite(corr) and abs(corr) > 0.5:
            problems.append(
                f"Feature '{col}' correlates {corr:.2f} with the result. "
                f"Nothing in racing is that predictive - check for leakage."
            )

    # The first appearance of every horse must show zero prior starts.
    first = resolved.sort_values("race_datetime").groupby("runner_id").head(1)
    bad_first = (first["h_starts"] > 0).sum()
    if bad_first:
        problems.append(f"{bad_first} horse(s) have prior starts recorded on their first appearance.")

    return problems
