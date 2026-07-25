"""Canonical data schema for Australian thoroughbred racing.

Every data source adapter must emit rows conforming to `RUNNER_COLUMNS`.
Keeping one schema means we can mix sources (Betfair, a form CSV, manual entry)
without the rest of the pipeline caring where a row came from.

Design note on units: all distances are metres, all weights are kilograms, all
times are seconds, all prices are DECIMAL odds (3.50 = $3.50, i.e. stake back
plus 2.50 profit). Australian racing quotes decimal odds, so we never convert.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

# Sentinel finishing position meaning "did not win, and the exact position is
# not known". Betfair's published files record only whether a runner won, so
# every loser comes back as this value. It matters a great deal downstream:
# a horse's average finishing position is meaningless if every loss is recorded
# as 99th, so the feature layer must treat this as missing rather than as a
# real position. See `features.build_features`.
UNKNOWN_LOSS_POSITION = 99.0


def has_full_finishing_order(frame: pd.DataFrame) -> bool:
    """Whether the dataset records real finishing positions or only winners.

    Returns False for a winner-only source such as the Betfair files, in which
    case every form feature based on finishing position will be unavailable and
    the model has substantially less to work with.
    """
    resolved = frame.loc[frame["finish_position"].notna(), "finish_position"]
    if resolved.empty:
        return False
    real_positions = resolved[(resolved > 1) & (resolved != UNKNOWN_LOSS_POSITION)]
    return len(real_positions) > 0.05 * len(resolved)


# --------------------------------------------------------------------------
# Track condition
# --------------------------------------------------------------------------
# Australian tracks are rated on a 1-10 scale. The rating is published before
# racing and revised during the day. Mapping to the named categories:
TRACK_CONDITION_SCALE: dict[str, tuple[int, int]] = {
    "FIRM": (1, 2),
    "GOOD": (3, 4),
    "SOFT": (5, 7),
    "HEAVY": (8, 10),
    "SYNTHETIC": (0, 0),  # Synthetic surfaces are not rated on the 1-10 scale.
}


def track_condition_to_number(condition: str | int | float | None) -> float:
    """Normalise a track condition to its numeric rating.

    Accepts "Good 4", "GOOD4", "Soft 7", "Heavy", 4, 4.0, None.
    Returns NaN when unknown rather than guessing, so that downstream code can
    treat 'unknown going' as missing instead of silently assuming Good.
    """
    if condition is None:
        return float("nan")
    if isinstance(condition, (int, float)) and not isinstance(condition, bool):
        value = float(condition)
        return value if 1 <= value <= 10 else float("nan")

    text = str(condition).strip().upper()
    if not text:
        return float("nan")

    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        value = float(digits[:2])
        if 1 <= value <= 10:
            return value

    # No number attached - fall back to the midpoint of the named band.
    letters = "".join(ch for ch in text if ch.isalpha())
    for name, (low, high) in TRACK_CONDITION_SCALE.items():
        if letters.startswith(name):
            if name == "SYNTHETIC":
                return float("nan")
            return (low + high) / 2.0
    return float("nan")


def track_condition_band(condition: str | int | float | None) -> str:
    """Return FIRM/GOOD/SOFT/HEAVY/UNKNOWN for a condition value."""
    value = track_condition_to_number(condition)
    if np.isnan(value):
        return "UNKNOWN"
    for name, (low, high) in TRACK_CONDITION_SCALE.items():
        if name == "SYNTHETIC":
            continue
        if low <= value <= high:
            return name
    return "UNKNOWN"


# --------------------------------------------------------------------------
# Canonical columns
# --------------------------------------------------------------------------
# Columns the pipeline relies on. Adapters fill what they can; missing optional
# columns become NaN and the feature layer handles their absence explicitly.

RACE_KEY_COLUMNS = ["race_id"]

REQUIRED_COLUMNS = [
    "race_id",          # unique id for the race
    "race_datetime",    # tz-aware or naive local datetime of the jump
    "track",            # course name, e.g. "Flemington"
    "runner_id",        # stable id for the horse (name-based fallback allowed)
    "horse",            # horse name
]

OPTIONAL_COLUMNS = [
    # --- race context -----------------------------------------------------
    "state",            # NSW / VIC / QLD / SA / WA / TAS / NT / ACT
    "race_number",
    "distance_m",
    "track_condition",  # raw string as published, e.g. "Good 4"
    "surface",          # TURF / SYNTHETIC
    "race_class",       # e.g. "BM78", "Group 1", "Maiden"
    "prizemoney",       # total race prizemoney in AUD
    "field_size",
    "rail_position",    # e.g. "+6m" - meaningfully changes track bias
    # --- runner ----------------------------------------------------------
    "barrier",
    "weight_kg",        # weight carried including jockey
    "jockey",
    "trainer",
    "age",
    "sex",              # G / M / C / F / H / R
    "days_since_last_run",
    "career_starts",
    "career_wins",
    "career_places",
    "gear_change",      # e.g. "Blinkers first time"
    # --- market ----------------------------------------------------------
    "odds_decimal",     # best available fixed odds at time of assessment
    "bsp",              # Betfair Starting Price (settled, historical only)
    "market_prob",      # de-vigged market probability, computed downstream
    # --- weather (joined from a weather source) ---------------------------
    "temp_c",
    "rain_mm_24h",
    "wind_kph",
    "humidity_pct",
    # --- result (historical rows only; NaN for upcoming races) ------------
    "finish_position",  # 1 = winner; NaN if scratched/did not finish
    "margin_l",         # lengths behind winner
    "race_time_s",      # winner's time for the race
    "scratched",        # bool
]

RUNNER_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


@dataclass
class Runner:
    """A single horse in a single race."""

    runner_id: str
    horse: str
    barrier: Optional[int] = None
    weight_kg: Optional[float] = None
    jockey: Optional[str] = None
    trainer: Optional[str] = None
    odds_decimal: Optional[float] = None
    scratched: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class Race:
    """A race and its field."""

    race_id: str
    race_datetime: datetime
    track: str
    runners: list[Runner]
    distance_m: Optional[int] = None
    track_condition: Optional[str] = None
    race_class: Optional[str] = None
    state: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def live_runners(self) -> list[Runner]:
        return [r for r in self.runners if not r.scratched]

    def to_frame(self) -> pd.DataFrame:
        """Flatten to the canonical row-per-runner frame."""
        rows = []
        for runner in self.runners:
            row = {
                "race_id": self.race_id,
                "race_datetime": self.race_datetime,
                "track": self.track,
                "state": self.state,
                "distance_m": self.distance_m,
                "track_condition": self.track_condition,
                "race_class": self.race_class,
                "runner_id": runner.runner_id,
                "horse": runner.horse,
                "barrier": runner.barrier,
                "weight_kg": runner.weight_kg,
                "jockey": runner.jockey,
                "trainer": runner.trainer,
                "odds_decimal": runner.odds_decimal,
                "scratched": runner.scratched,
            }
            row.update(runner.extra)
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame["field_size"] = len(self.live_runners)
        return frame


def empty_frame() -> pd.DataFrame:
    """An empty frame with every canonical column present."""
    return pd.DataFrame({col: pd.Series(dtype="object") for col in RUNNER_COLUMNS})


def conform(frame: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Coerce an arbitrary frame to the canonical schema.

    Adds any missing optional columns as NaN, drops unknown columns, and casts
    the columns whose dtype the rest of the pipeline depends on. Raises when a
    required column is absent and `strict` is set.
    """
    frame = frame.copy()

    missing_required = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing_required and strict:
        raise ValueError(
            f"Missing required column(s): {missing_required}. "
            f"Every data source must supply {REQUIRED_COLUMNS}."
        )

    for col in RUNNER_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan

    # Preserve any extra columns a source supplies - features may use them -
    # but put the canonical ones first for readability.
    extras = [c for c in frame.columns if c not in RUNNER_COLUMNS]
    frame = frame[RUNNER_COLUMNS + extras]

    frame["race_datetime"] = pd.to_datetime(frame["race_datetime"], errors="coerce")
    frame["race_id"] = frame["race_id"].astype(str)
    frame["runner_id"] = frame["runner_id"].astype(str)
    frame["horse"] = frame["horse"].astype(str)

    numeric = [
        "distance_m", "prizemoney", "field_size", "barrier", "weight_kg", "age",
        "days_since_last_run", "career_starts", "career_wins", "career_places",
        "odds_decimal", "bsp", "market_prob", "temp_c", "rain_mm_24h",
        "wind_kph", "humidity_pct", "finish_position", "margin_l", "race_time_s",
        "race_number",
    ]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame["scratched"] = frame["scratched"].fillna(False).astype(bool)

    # Derived, always-present helpers.
    frame["condition_num"] = frame["track_condition"].map(track_condition_to_number)
    frame["condition_band"] = frame["track_condition"].map(track_condition_band)
    frame["is_wet"] = (frame["condition_num"] >= 5).astype("float")
    frame.loc[frame["condition_num"].isna(), "is_wet"] = np.nan

    return frame.sort_values(["race_datetime", "race_id", "runner_id"]).reset_index(drop=True)


def validate(frame: pd.DataFrame) -> list[str]:
    """Return a list of human-readable data-quality problems.

    This is deliberately noisy: bad racing data is the single most common cause
    of a model that looks brilliant in backtest and loses money live.
    """
    problems: list[str] = []

    if frame.empty:
        return ["Dataset is empty."]

    if frame["race_datetime"].isna().any():
        n = int(frame["race_datetime"].isna().sum())
        problems.append(f"{n} row(s) have an unparseable race_datetime.")

    dupes = frame.duplicated(subset=["race_id", "runner_id"]).sum()
    if dupes:
        problems.append(f"{dupes} duplicate (race_id, runner_id) row(s).")

    # A finished race should have exactly one winner.
    finished = frame[frame["finish_position"].notna()]
    if not finished.empty:
        winners = finished[finished["finish_position"] == 1].groupby("race_id").size()
        races_with_results = finished["race_id"].nunique()
        no_winner = races_with_results - (winners == 1).sum()
        if no_winner > 0:
            problems.append(
                f"{no_winner} finished race(s) do not have exactly one runner "
                f"with finish_position == 1 (dead heats or bad data)."
            )

    # Field sizes outside this range are almost always parse errors.
    sizes = frame[~frame["scratched"]].groupby("race_id").size()
    odd_sizes = sizes[(sizes < 2) | (sizes > 24)]
    if len(odd_sizes):
        problems.append(
            f"{len(odd_sizes)} race(s) have an implausible field size "
            f"(min {int(sizes.min())}, max {int(sizes.max())})."
        )

    # Odds sanity.
    odds = frame["odds_decimal"].dropna()
    if len(odds):
        bad = ((odds <= 1.0) | (odds > 1000)).sum()
        if bad:
            problems.append(f"{bad} row(s) have odds outside the plausible 1.01-1000 range.")

    coverage = 1.0 - frame["odds_decimal"].isna().mean()
    if coverage < 0.5:
        problems.append(
            f"Only {coverage:.0%} of runners have odds. The market blend is the "
            f"single strongest signal available - without odds the model is much weaker."
        )

    return problems
