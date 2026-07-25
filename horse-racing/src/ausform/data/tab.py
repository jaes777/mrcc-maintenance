"""TAB Australia odds adapter.

TAB exposes a public JSON service that the tab.com.au site itself consumes.
It requires no key, which makes it the most practical free source of
Australian fixed-odds and tote prices.

Two caveats, stated plainly:

1. It is an *undocumented* service. Field names and paths can change
   without notice, so every accessor here is defensive: a shape change
   should degrade to "no odds" rather than a stack trace.
2. Automated access is governed by TAB's terms of use. Keep request rates
   low and personal. `min_interval_s` throttles by default.

The parser is written against the documented-by-observation response shape
and is covered by fixture tests in tests/fixtures/. It has NOT been
verified against the live service from this build environment, which has
no outbound access to tab.com.au -- see README "Data sources".
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from typing import Any, Iterable, Optional

import requests

from ..tracks import get_track, unknown_track
from ..types import Meeting, Race, Rail, Runner, condition_to_number
from .base import OddsProvider

log = logging.getLogger(__name__)

BASE_URL = "https://api.beta.tab.com.au/v1/tab-info-service"
JURISDICTIONS = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"]

# `jurisdiction` selects a pricing jurisdiction, not a geography: the feed
# returns international meetings too. This filters to domestic racing.
_AU_LOCATIONS = {"NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT", "AUS"}


def _first(mapping: Any, *keys: str, default: Any = None) -> Any:
    """Return the first present, non-null key. Feeds rename fields; this
    keeps the parser working across minor variations."""
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _to_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


class TabClient(OddsProvider):
    """Reads meetings, fields and prices from the TAB info service."""

    def __init__(
        self,
        jurisdiction: str = "NSW",
        session: Optional[requests.Session] = None,
        timeout: int = 20,
        min_interval_s: float = 1.0,
    ):
        self.jurisdiction = jurisdiction
        self.session = session or requests.Session()
        self.session.headers.setdefault(
            "User-Agent", "ausform/0.1 (personal race analysis)")
        self.timeout = timeout
        self.min_interval_s = min_interval_s
        self._last_request = 0.0

    # -- HTTP --------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        url = f"{BASE_URL}{path}"
        query = {"jurisdiction": self.jurisdiction}
        query.update(params or {})
        try:
            response = self.session.get(url, params=query, timeout=self.timeout)
            self._last_request = time.monotonic()
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("TAB request failed (%s): %s", url, exc)
            return None

    # -- public API --------------------------------------------------------

    def meetings(self, date: _dt.date) -> list[Meeting]:
        """Thoroughbred meetings scheduled on `date`."""
        payload = self._get(f"/racing/dates/{date.isoformat()}/meetings",
                            {"returnOffers": "false", "returnPromo": "false"})
        if not payload:
            return []
        return list(self._parse_meetings(payload, date))

    def race(self, date: _dt.date, venue_mnemonic: str, race_number: int,
             race_type: str = "R") -> Optional[Race]:
        """One race with runners and prices."""
        path = (f"/racing/dates/{date.isoformat()}/meetings/{race_type}/"
                f"{venue_mnemonic}/races/{race_number}")
        payload = self._get(path, {"returnPromo": "false"})
        if not payload:
            return None
        return self._parse_race(payload, date, venue_mnemonic, race_number)

    def attach_odds(self, race: Race) -> Race:
        """Refresh prices on an existing Race in place.

        Matching is by saddlecloth number, which is stable within a race
        and is what every Australian feed keys on.
        """
        venue = race.venue_mnemonic or race.track.code[:3]
        fresh = self.race(race.date, venue, race.race_number)
        if fresh is None:
            log.info("No TAB prices for %s R%d", race.track.name, race.race_number)
            return race
        prices = {r.number: r for r in fresh.runners}
        for runner in race.runners:
            source = prices.get(runner.number)
            if source is None:
                continue
            runner.fixed_win_odds = source.fixed_win_odds
            runner.fixed_place_odds = source.fixed_place_odds
            runner.tote_win_odds = source.tote_win_odds
            if source.scratched:
                runner.scratched = True
        return race

    # -- parsing -----------------------------------------------------------

    def _parse_meetings(self, payload: dict, date: _dt.date) -> Iterable[Meeting]:
        for item in _first(payload, "meetings", default=[]) or []:
            race_type = _first(item, "raceType", "meetingType", default="")
            # R = thoroughbred; G = greyhound; H = harness. We only do gallops.
            if str(race_type).upper() not in ("R", "T", "THOROUGHBRED"):
                continue
            # Australian racing only; the same feed carries UK/US meetings.
            location = str(_first(item, "location", default="") or "").upper()
            if location and location not in _AU_LOCATIONS:
                continue

            venue_name = _first(item, "meetingName", "venueName", default="")
            mnemonic = _first(item, "venueMnemonic")
            track = get_track(venue_name) or unknown_track(venue_name, location or "??")

            # The meeting object carries going and rail, which are two of
            # the most useful race-level features available anywhere.
            condition = condition_to_number(
                str(_first(item, "trackCondition", default="") or ""))
            rail_text = str(_first(item, "railPosition", default="") or "").lower()
            rail = Rail.TRUE
            if "out" in rail_text:
                rail = Rail.OUT
            elif "in" in rail_text:
                rail = Rail.IN

            meeting = Meeting(
                meeting_id=str(_first(item, "meetingId",
                                      default=f"{venue_name}-{date.isoformat()}")),
                track=track,
                date=date,
                rail=rail,
                track_condition=condition,
            )
            for race_item in _first(item, "races", default=[]) or []:
                race = self._parse_race_summary(race_item, track, date)
                if race is not None:
                    race.venue_mnemonic = mnemonic
                    race.track_condition = condition
                    race.rail = rail
                    meeting.races.append(race)
            yield meeting

    def _parse_race_summary(self, item: dict, track, date: _dt.date) -> Optional[Race]:
        number = _first(item, "raceNumber", "number")
        if number is None:
            return None
        distance = _first(item, "raceDistance", "distance", default=0)
        start = _first(item, "raceStartTime", "startTime")
        start_time = None
        if isinstance(start, str):
            try:
                start_time = _dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
            except ValueError:
                start_time = None
        return Race(
            race_id=str(_first(item, "raceId", default=f"{track.code}-{date}-{number}")),
            track=track,
            date=date,
            race_number=int(number),
            distance_m=int(distance or 0),
            name=_first(item, "raceName", default=""),
            start_time=start_time,
            class_level=_first(item, "raceClassConditions", "raceClass"),
        )

    def _parse_race(self, payload: dict, date: _dt.date, venue: str,
                    race_number: int) -> Optional[Race]:
        venue_name = _first(payload, "meetingName", "venueName", default=venue)
        track = get_track(venue_name) or unknown_track(venue_name)
        distance = _first(payload, "raceDistance", "distance", default=0)

        race = Race(
            race_id=str(_first(payload, "raceId",
                               default=f"{venue}-{date}-{race_number}")),
            track=track,
            date=date,
            race_number=race_number,
            distance_m=int(distance or 0),
            name=_first(payload, "raceName", default=""),
            class_level=_first(payload, "raceClassConditions", "raceClass"),
        )

        for item in _first(payload, "runners", default=[]) or []:
            runner = self._parse_runner(item)
            if runner is not None:
                race.runners.append(runner)
        return race

    def _parse_runner(self, item: dict) -> Optional[Runner]:
        number = _first(item, "runnerNumber", "number")
        name = _first(item, "runnerName", "name")
        if number is None or name is None:
            return None

        # Prices live under fixedOdds / parimutuel sub-objects.
        fixed = _first(item, "fixedOdds", default={}) or {}
        tote = _first(item, "parimutuel", default={}) or {}

        scratched = bool(
            _first(item, "scratched", default=False)
            or str(_first(fixed, "bettingStatus", default="")).lower() == "closed"
            or _first(item, "scratchedTime") is not None
        )

        return Runner(
            horse_id=str(_first(item, "runnerId", default=f"{name}")).strip(),
            name=str(name).strip(),
            number=int(number),
            barrier=_first(item, "barrierNumber", "barrier"),
            weight_kg=_to_float(_first(item, "handicapWeight", "weight")),
            jockey=_first(item, "riderDriverName", "jockey"),
            trainer=_first(item, "trainerName", "trainer"),
            scratched=scratched,
            fixed_win_odds=_to_float(_first(fixed, "returnWin", "winPrice")),
            fixed_place_odds=_to_float(_first(fixed, "returnPlace", "placePrice")),
            tote_win_odds=_to_float(_first(tote, "returnWin", "winPrice")),
            opening_odds=_to_float(_first(fixed, "returnWinOpen", "openPrice")),
            early_speed_rating=_to_float(_first(item, "earlySpeedRating")),
            early_speed_band=_first(item, "earlySpeedRatingBand"),
            form_rating=_to_float(_first(item, "dfsFormRating")),
            last_5_starts=_first(item, "last5Starts"),
        )
