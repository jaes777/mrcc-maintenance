"""Core domain objects for Australian thoroughbred racing.

Everything downstream (features, models, betting) speaks in terms of these
types, so adding a new data provider only means writing a translator into
this vocabulary.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional


class Surface(str, Enum):
    TURF = "turf"
    SYNTHETIC = "synthetic"  # e.g. Geelong / Ballarat all-weather
    DIRT = "dirt"


class Rail(str, Enum):
    """Rail position relative to the true course."""

    TRUE = "true"
    OUT = "out"
    IN = "in"


# The Australian track rating scale. Since 2014 all codes use the same
# four words with a 1-10 numeric qualifier: Firm 1-2, Good 3-4, Soft 5-7,
# Heavy 8-10. Lower number = firmer/faster ground.
TRACK_CONDITION_SCALE: dict[str, tuple[int, int]] = {
    "firm": (1, 2),
    "good": (3, 4),
    "soft": (5, 7),
    "heavy": (8, 10),
}


def condition_to_number(text: str) -> Optional[int]:
    """Parse 'Soft 6', 'Good(4)', 'HEAVY 10' etc. into the 1-10 numeric scale.

    Falls back to the midpoint of the named band when no digit is supplied,
    and returns None when the text is unrecognised, so callers can decide
    whether to impute or drop.
    """
    if not text:
        return None
    lowered = text.strip().lower()
    digits = "".join(ch for ch in lowered if ch.isdigit())
    if digits:
        value = int(digits)
        if 1 <= value <= 10:
            return value
    for word, (low, high) in TRACK_CONDITION_SCALE.items():
        if word in lowered:
            return (low + high) // 2
    return None


def condition_band(number: Optional[int]) -> Optional[str]:
    """Inverse of `condition_to_number`: 6 -> 'soft'."""
    if number is None:
        return None
    for word, (low, high) in TRACK_CONDITION_SCALE.items():
        if low <= number <= high:
            return word
    return None


@dataclass(frozen=True)
class Track:
    """A racecourse. Coordinates drive the weather lookup."""

    code: str
    name: str
    state: str
    latitude: float
    longitude: float
    surface: Surface = Surface.TURF
    circumference_m: Optional[int] = None
    straight_m: Optional[int] = None
    direction: Optional[str] = None  # "left" or "right" handed


@dataclass
class Weather:
    """Conditions at (or forecast for) race time."""

    temperature_c: Optional[float] = None
    rainfall_mm_24h: Optional[float] = None
    rainfall_mm_72h: Optional[float] = None
    humidity_pct: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    is_forecast: bool = False


@dataclass
class PastRun:
    """One historical start for a horse.

    This is the raw form line. Fields are all optional because different
    providers expose different depth, and the feature layer is written to
    degrade gracefully rather than crash on a missing sectional.
    """

    date: _dt.date
    track_code: str
    distance_m: int
    finish_position: Optional[int] = None
    field_size: Optional[int] = None
    margin_l: Optional[float] = None  # lengths behind winner; 0 for winner
    barrier: Optional[int] = None
    weight_kg: Optional[float] = None
    jockey: Optional[str] = None
    track_condition: Optional[int] = None  # 1-10 scale
    class_level: Optional[str] = None
    prize_money_total: Optional[float] = None
    starting_price: Optional[float] = None  # decimal odds
    race_time_s: Optional[float] = None
    last_600m_s: Optional[float] = None
    surface: Surface = Surface.TURF
    # Explanatory flags a form guide would carry
    barrier_trial: bool = False
    scratched: bool = False

    @property
    def won(self) -> bool:
        return self.finish_position == 1

    @property
    def placed(self) -> bool:
        """Top-3. Note this ignores field-size rules for place betting;
        those live in the betting module where they matter."""
        return self.finish_position is not None and self.finish_position <= 3


@dataclass
class Runner:
    """A horse entered in a specific race."""

    horse_id: str
    name: str
    number: int  # saddlecloth
    barrier: Optional[int] = None
    weight_kg: Optional[float] = None
    jockey: Optional[str] = None
    trainer: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    sire: Optional[str] = None
    dam_sire: Optional[str] = None
    gear_changes: list[str] = field(default_factory=list)
    scratched: bool = False
    apprentice_claim_kg: float = 0.0

    # Provider-supplied ratings. Useful as features but never trusted
    # blindly: they are somebody else's model, with unknown construction.
    early_speed_rating: Optional[float] = None
    early_speed_band: Optional[str] = None  # e.g. LEADER / MIDFIELD / BACKMARKER
    form_rating: Optional[float] = None
    last_5_starts: Optional[str] = None  # e.g. "3x1204"

    # Market. Populated by the odds adapter; may be absent pre-market.
    fixed_win_odds: Optional[float] = None  # decimal, includes stake
    fixed_place_odds: Optional[float] = None
    tote_win_odds: Optional[float] = None
    exchange_back_odds: Optional[float] = None
    exchange_lay_odds: Optional[float] = None
    opening_odds: Optional[float] = None

    # Form history, most recent first. Populated by the form adapter.
    history: list[PastRun] = field(default_factory=list)

    # Filled in after the race is run; None beforehand.
    result_position: Optional[int] = None
    result_time_s: Optional[float] = None
    result_margin_l: Optional[float] = None

    def runs_before(self, cutoff: _dt.date) -> list[PastRun]:
        """Point-in-time form, most recent first.

        Used everywhere in feature building to make leakage structurally
        difficult: you cannot see a run that had not happened when the race
        was framed.

        The sort is not decorative. Every caller assumes `[0]` is the last
        start -- days since run, last finishing position, weight change,
        jockey switch, the most recent three runs. A provider that hands
        back oldest-first would silently invert all of those with no error
        anywhere, so the ordering is enforced here rather than trusted.
        """
        recent = [r for r in self.history if r.date < cutoff and not r.scratched]
        recent.sort(key=lambda r: r.date, reverse=True)
        return recent


@dataclass
class Race:
    """A single race, with everything known about it at analysis time."""

    race_id: str
    track: Track
    date: _dt.date
    race_number: int
    distance_m: int
    name: str = ""
    start_time: Optional[_dt.datetime] = None
    track_condition: Optional[int] = None  # 1-10
    rail: Rail = Rail.TRUE
    rail_offset_m: float = 0.0
    class_level: Optional[str] = None
    prize_money: Optional[float] = None
    age_restriction: Optional[str] = None
    sex_restriction: Optional[str] = None
    is_handicap: bool = True
    weather: Optional[Weather] = None
    runners: list[Runner] = field(default_factory=list)
    # Provider key needed to re-request this race (TAB uses a 3-letter code).
    venue_mnemonic: Optional[str] = None

    @property
    def field_size(self) -> int:
        return len([r for r in self.runners if not r.scratched])

    @property
    def active_runners(self) -> list[Runner]:
        return [r for r in self.runners if not r.scratched]

    @property
    def is_resulted(self) -> bool:
        return any(r.result_position is not None for r in self.runners)

    @property
    def winner(self) -> Optional[Runner]:
        for runner in self.runners:
            if runner.result_position == 1:
                return runner
        return None

    def runner_by_number(self, number: int) -> Optional[Runner]:
        for runner in self.runners:
            if runner.number == number:
                return runner
        return None


@dataclass
class Meeting:
    """A day's racing at one track."""

    meeting_id: str
    track: Track
    date: _dt.date
    races: list[Race] = field(default_factory=list)
    rail: Rail = Rail.TRUE
    track_condition: Optional[int] = None

    def __iter__(self) -> Iterable[Race]:
        return iter(self.races)
