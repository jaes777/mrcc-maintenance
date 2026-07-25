"""Betfair public historical data for Australian racing.

This is the most useful *free and legally clean* source of real Australian
race data. Betfair publishes it deliberately, for modellers, with no login
and no key -- unlike Racing Australia, whose terms of use expressly forbid
automated access.

Two products are supported:

1. **Betfair Starting Price (BSP) daily CSVs.** One file per day per
   country, win and place, going back to the beginning of the Exchange.
       https://promo.betfair.com/betfairsp/prices/dwbfpricesauswin{DDMMYYYY}.csv
   Columns: EVENT_ID, MENU_HINT, EVENT_NAME, EVENT_DT, SELECTION_ID,
   SELECTION_NAME, WIN_LOSE, BSP, PPWAP, MORNINGWAP, PPMAX, PPMIN, IPMAX,
   IPMIN, MORNINGTRADEDVOL, PPTRADEDVOL, IPTRADEDVOL

2. **ANZ Thoroughbred yearly/monthly extracts** published on Betfair's
   data-scientists site.

WHAT THIS GIVES YOU, AND WHAT IT DOES NOT
-----------------------------------------
It gives you real outcomes (WIN_LOSE) and a genuinely excellent market
consensus price (BSP), for hundreds of thousands of Australian runners.
That is enough to calibrate and validate the betting engine against real
prices, and to measure whether a strategy would actually have made money.

It does NOT give you barrier, weight, jockey, trainer, sectionals or track
condition. Those come from a form provider. Treat Betfair as the outcome
and price spine, and join richer form onto it when you have a feed.
"""

from __future__ import annotations

import csv
import datetime as _dt
import io
import logging
import re
from dataclasses import dataclass
from typing import Iterator, Optional

import requests

from ..tracks import get_track, unknown_track
from ..types import Race, Runner

log = logging.getLogger(__name__)

BSP_URL = "https://promo.betfair.com/betfairsp/prices/dwbfprices{country}{market}{date}.csv"

# "AUS / Randwick (AUS) 25th Jul" -> track name
_MENU_TRACK = re.compile(r"/\s*([^(/]+?)\s*(?:\(|$)")
# "R7 1200m Grp1" / "R3 2040m Hcap" -> race number and distance
_EVENT_RACE = re.compile(r"\bR(?:ace)?\s*(\d+)\b", re.IGNORECASE)
_EVENT_DIST = re.compile(r"\b(\d{3,4})\s*m\b", re.IGNORECASE)
# "7. Bold Ambition" -> saddlecloth and name
_SELECTION = re.compile(r"^\s*(\d+)[.)]?\s+(.*?)\s*$")


@dataclass
class BspRow:
    """One runner in one Betfair market."""

    event_id: str
    event_date: _dt.date
    track_name: str
    race_number: Optional[int]
    distance_m: Optional[int]
    saddlecloth: Optional[int]
    horse_name: str
    won: bool
    bsp: Optional[float]
    morning_wap: Optional[float]
    preplay_wap: Optional[float]
    preplay_volume: Optional[float]


def _f(value: str) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


class BetfairHistorical:
    """Downloads and parses Betfair's free public price files."""

    def __init__(self, session: Optional[requests.Session] = None, timeout: int = 60):
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "ausform/0.1")
        self.timeout = timeout

    def bsp_url(self, date: _dt.date, market: str = "win",
                country: str = "aus") -> str:
        """Build the daily BSP file URL. Date format is DDMMYYYY."""
        if market not in ("win", "place"):
            raise ValueError("market must be 'win' or 'place'")
        return BSP_URL.format(country=country, market=market,
                              date=date.strftime("%d%m%Y"))

    def fetch_day(self, date: _dt.date, market: str = "win",
                  country: str = "aus") -> list[BspRow]:
        """All runners priced on `date`. Empty list if the file is absent
        (no racing that day, or not yet published)."""
        url = self.bsp_url(date, market, country)
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 404:
                return []
            response.raise_for_status()
        except requests.RequestException as exc:
            log.warning("Betfair BSP fetch failed for %s: %s", date, exc)
            return []
        return list(self.parse_bsp(response.text))

    def fetch_range(self, start: _dt.date, end: _dt.date,
                    market: str = "win", country: str = "aus") -> Iterator[BspRow]:
        """Iterate every runner between two dates inclusive."""
        current = start
        while current <= end:
            yield from self.fetch_day(current, market, country)
            current += _dt.timedelta(days=1)

    # -- parsing -----------------------------------------------------------

    @staticmethod
    def parse_bsp(text: str) -> Iterator[BspRow]:
        """Parse the BSP CSV body. Tolerant of column additions/reordering."""
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            menu = (row.get("MENU_HINT") or "").strip()
            event = (row.get("EVENT_NAME") or "").strip()
            selection = (row.get("SELECTION_NAME") or "").strip()
            if not selection:
                continue

            track_match = _MENU_TRACK.search(menu)
            track_name = track_match.group(1).strip() if track_match else menu

            race_match = _EVENT_RACE.search(event)
            dist_match = _EVENT_DIST.search(event)
            sel_match = _SELECTION.match(selection)

            event_dt = row.get("EVENT_DT") or ""
            parsed_date = None
            for fmt in ("%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M",
                        "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S"):
                try:
                    parsed_date = _dt.datetime.strptime(event_dt.strip(), fmt).date()
                    break
                except ValueError:
                    continue
            if parsed_date is None:
                continue

            win_lose = (row.get("WIN_LOSE") or "").strip()

            yield BspRow(
                event_id=(row.get("EVENT_ID") or "").strip(),
                event_date=parsed_date,
                track_name=track_name,
                race_number=int(race_match.group(1)) if race_match else None,
                distance_m=int(dist_match.group(1)) if dist_match else None,
                saddlecloth=int(sel_match.group(1)) if sel_match else None,
                horse_name=(sel_match.group(2) if sel_match else selection),
                won=win_lose in ("1", "1.0", "WIN", "win"),
                bsp=_f(row.get("BSP", "")),
                morning_wap=_f(row.get("MORNINGWAP", "")),
                preplay_wap=_f(row.get("PPWAP", "")),
                preplay_volume=_f(row.get("PPTRADEDVOL", "")),
            )

    @staticmethod
    def to_races(rows: list[BspRow]) -> list[Race]:
        """Group BSP rows into Race objects.

        Finishing positions beyond the winner are unknown from this source,
        so runners get position 1 for the winner and None otherwise. That
        is sufficient for win-market modelling and for measuring realised
        ROI, but not for exotics -- which need the full finishing order.
        """
        grouped: dict[str, list[BspRow]] = {}
        for row in rows:
            grouped.setdefault(row.event_id or f"{row.track_name}-{row.event_date}", []
                               ).append(row)

        races: list[Race] = []
        for event_id, group in grouped.items():
            head = group[0]
            track = get_track(head.track_name) or unknown_track(head.track_name)
            race = Race(
                race_id=event_id,
                track=track,
                date=head.event_date,
                race_number=head.race_number or 0,
                distance_m=head.distance_m or 0,
            )
            for index, row in enumerate(group, start=1):
                race.runners.append(Runner(
                    horse_id=row.horse_name.upper(),
                    name=row.horse_name,
                    number=row.saddlecloth or index,
                    # BSP is a genuine post-market consensus price and the
                    # best single free predictor available.
                    fixed_win_odds=row.bsp,
                    opening_odds=row.morning_wap,
                    result_position=1 if row.won else None,
                ))
            races.append(race)
        return races
