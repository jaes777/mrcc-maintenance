"""Weather via Open-Meteo.

Open-Meteo is free and needs no API key, which matters because weather is
one of the few genuinely free inputs. Two endpoints are used: the archive
for historical meetings (model training) and the forecast for upcoming
ones.

Rain is the variable that actually matters. It moves the track rating,
which in turn reshuffles the whole field: a wet-track specialist that is
unplaceable on Good 3 can be a standout on Heavy 9. We therefore pull
cumulative rainfall over the preceding 24 and 72 hours rather than just
conditions at the moment of the race.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Optional

import requests

from ..types import Track, Weather

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_HOURLY_VARS = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m"


class WeatherClient:
    """Fetches race-time weather for a track.

    Results are memoised per (track, date) because a single meeting has
    eight or nine races that all want the same day's data.
    """

    def __init__(self, session: Optional[requests.Session] = None, timeout: int = 20):
        self.session = session or requests.Session()
        self.timeout = timeout
        self._cache: dict[tuple[str, _dt.date, int], Optional[Weather]] = {}

    def for_race(
        self,
        track: Track,
        date: _dt.date,
        hour: int = 14,
    ) -> Optional[Weather]:
        """Weather at `hour` local time on `date` at `track`.

        Returns None rather than raising when the lookup fails, so a
        network hiccup degrades the prediction instead of killing it.
        """
        if track.latitude == 0.0 and track.longitude == 0.0:
            log.debug("No coordinates for %s; skipping weather", track.name)
            return None

        key = (track.code, date, hour)
        if key in self._cache:
            return self._cache[key]

        today = _dt.date.today()
        is_forecast = date >= today
        try:
            weather = (
                self._fetch_forecast(track, date, hour)
                if is_forecast
                else self._fetch_archive(track, date, hour)
            )
        except (requests.RequestException, ValueError, KeyError) as exc:
            log.warning("Weather lookup failed for %s %s: %s", track.name, date, exc)
            weather = None

        self._cache[key] = weather
        return weather

    # -- internals ---------------------------------------------------------

    def _fetch_archive(self, track: Track, date: _dt.date, hour: int) -> Optional[Weather]:
        # Reach back three days so we can total antecedent rainfall.
        start = date - _dt.timedelta(days=3)
        params = {
            "latitude": track.latitude,
            "longitude": track.longitude,
            "start_date": start.isoformat(),
            "end_date": date.isoformat(),
            "hourly": _HOURLY_VARS,
            "timezone": "auto",
        }
        payload = self._get(ARCHIVE_URL, params)
        return self._parse(payload, date, hour, is_forecast=False)

    def _fetch_forecast(self, track: Track, date: _dt.date, hour: int) -> Optional[Weather]:
        today = _dt.date.today()
        days_ahead = (date - today).days
        if days_ahead > 15:
            log.debug("%s is beyond the forecast horizon", date)
            return None
        params = {
            "latitude": track.latitude,
            "longitude": track.longitude,
            "hourly": _HOURLY_VARS,
            "past_days": 3,  # gives us antecedent rain for track-condition context
            "forecast_days": max(1, min(16, days_ahead + 1)),
            "timezone": "auto",
        }
        payload = self._get(FORECAST_URL, params)
        return self._parse(payload, date, hour, is_forecast=True)

    def _get(self, url: str, params: dict) -> dict:
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _parse(
        payload: dict,
        date: _dt.date,
        hour: int,
        is_forecast: bool,
    ) -> Optional[Weather]:
        hourly = payload.get("hourly") or {}
        times: list[str] = hourly.get("time") or []
        if not times:
            return None

        target = f"{date.isoformat()}T{hour:02d}:00"
        try:
            index = times.index(target)
        except ValueError:
            # Fall back to the closest available hour on the target date.
            same_day = [i for i, t in enumerate(times) if t.startswith(date.isoformat())]
            if not same_day:
                return None
            index = min(same_day, key=lambda i: abs(int(times[i][11:13]) - hour))

        def at(name: str) -> Optional[float]:
            series = hourly.get(name)
            if not series or index >= len(series):
                return None
            value = series[index]
            return float(value) if value is not None else None

        def rain_over(hours: int) -> Optional[float]:
            series = hourly.get("precipitation")
            if not series:
                return None
            start = max(0, index - hours)
            window = [v for v in series[start:index + 1] if v is not None]
            return round(sum(float(v) for v in window), 2) if window else None

        return Weather(
            temperature_c=at("temperature_2m"),
            humidity_pct=at("relative_humidity_2m"),
            wind_speed_kmh=at("wind_speed_10m"),
            wind_direction_deg=at("wind_direction_10m"),
            rainfall_mm_24h=rain_over(24),
            rainfall_mm_72h=rain_over(72),
            is_forecast=is_forecast,
        )


def estimate_condition_shift(weather: Optional[Weather]) -> float:
    """Rough nudge to the track rating implied by recent rain.

    This is a *heuristic fallback* for when the official rating is not yet
    published, not a substitute for it. Track curators, irrigation and
    drainage vary enormously by venue, so treat the output as a prior with
    wide error bars: it answers "is this likely to be wetter than posted?",
    not "the track will be a Soft 6".

    Returns a positive number of rating points to add (wetter = higher).
    """
    if weather is None:
        return 0.0
    rain_24 = weather.rainfall_mm_24h or 0.0
    rain_72 = weather.rainfall_mm_72h or 0.0
    # Recent rain matters far more than rain that has had time to drain.
    effective = rain_24 + 0.35 * max(0.0, rain_72 - rain_24)
    if effective < 1:
        return 0.0
    if effective < 5:
        return 0.5
    if effective < 12:
        return 1.5
    if effective < 25:
        return 3.0
    if effective < 50:
        return 4.5
    return 6.0
