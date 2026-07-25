"""Weather, and the track-condition problem.

Track condition is one of the most important variables in Australian racing -
some horses are transformed by rain and others are ruined by it - and it is
also the biggest hole in the free data. Betfair's published files carry the
track, distance, prices and result, but *not* the going.

So we approximate it. Open-Meteo gives free historical and forecast rainfall
for any latitude/longitude, and cumulative rainfall over the preceding days is
what actually drives a track rating. `estimate_track_condition` turns rainfall
into an estimated 1-10 rating.

Be clear about what that is: an estimate, not the official rating. A real
rating comes from a penetrometer reading taken by a track manager, and it
depends on drainage, soil type, recent renovation and irrigation, none of which
rainfall captures. Treat the estimate as a useful feature and not as fact. If
you get access to real ratings (Punting Form, or a licensed feed), use those
instead and `fit_condition_model` will tell you how far off the proxy was.

Open-Meteo's free tier is non-commercial use only, data is CC-BY 4.0, and no
API key is needed. https://open-meteo.com/en/terms
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 45

# Approximate coordinates for Australian racecourses, accurate to within about
# a kilometre. That is far finer than the weather model's ~10km grid, so the
# imprecision does not matter for rainfall. Add your own tracks here - the
# lookup is a plain dictionary on purpose so it is easy to edit.
RACECOURSES: dict[str, tuple[float, float, str]] = {
    # Victoria
    "FLEMINGTON": (-37.7889, 144.9089, "Australia/Melbourne"),
    "CAULFIELD": (-37.8817, 145.0417, "Australia/Melbourne"),
    "MOONEE VALLEY": (-37.7658, 144.9317, "Australia/Melbourne"),
    "SANDOWN": (-37.9469, 145.1636, "Australia/Melbourne"),
    "SANDOWN HILLSIDE": (-37.9469, 145.1636, "Australia/Melbourne"),
    "SANDOWN LAKESIDE": (-37.9469, 145.1636, "Australia/Melbourne"),
    "GEELONG": (-38.1656, 144.3492, "Australia/Melbourne"),
    "BALLARAT": (-37.5364, 143.8622, "Australia/Melbourne"),
    "BENDIGO": (-36.7644, 144.3047, "Australia/Melbourne"),
    "CRANBOURNE": (-38.1108, 145.2864, "Australia/Melbourne"),
    "PAKENHAM": (-38.0442, 145.5397, "Australia/Melbourne"),
    "MORNINGTON": (-38.2261, 145.0483, "Australia/Melbourne"),
    "WERRIBEE": (-37.8942, 144.6592, "Australia/Melbourne"),
    "WARRNAMBOOL": (-38.3803, 142.4886, "Australia/Melbourne"),
    "SALE": (-38.1094, 147.0703, "Australia/Melbourne"),
    "SEYMOUR": (-37.0272, 145.1400, "Australia/Melbourne"),
    "KILMORE": (-37.2939, 144.9500, "Australia/Melbourne"),
    "WANGARATTA": (-36.3583, 146.3208, "Australia/Melbourne"),
    "ECHUCA": (-36.1408, 144.7519, "Australia/Melbourne"),
    "SWAN HILL": (-35.3389, 143.5544, "Australia/Melbourne"),
    # New South Wales / ACT
    "RANDWICK": (-33.9042, 151.2306, "Australia/Sydney"),
    "ROYAL RANDWICK": (-33.9042, 151.2306, "Australia/Sydney"),
    "ROSEHILL": (-33.8214, 151.0244, "Australia/Sydney"),
    "ROSEHILL GARDENS": (-33.8214, 151.0244, "Australia/Sydney"),
    "WARWICK FARM": (-33.9086, 150.9403, "Australia/Sydney"),
    "CANTERBURY": (-33.9078, 151.1136, "Australia/Sydney"),
    "NEWCASTLE": (-32.8925, 151.7264, "Australia/Sydney"),
    "KEMBLA GRANGE": (-34.4664, 150.7972, "Australia/Sydney"),
    "GOSFORD": (-33.4297, 151.3406, "Australia/Sydney"),
    "HAWKESBURY": (-33.6083, 150.8281, "Australia/Sydney"),
    "WYONG": (-33.2836, 151.4272, "Australia/Sydney"),
    "SCONE": (-32.0472, 150.8642, "Australia/Sydney"),
    "MUSWELLBROOK": (-32.2650, 150.8892, "Australia/Sydney"),
    "DUBBO": (-32.2519, 148.6053, "Australia/Sydney"),
    "WAGGA": (-35.1225, 147.3733, "Australia/Sydney"),
    "WAGGA WAGGA": (-35.1225, 147.3733, "Australia/Sydney"),
    "ALBURY": (-36.0806, 146.9219, "Australia/Sydney"),
    "GRAFTON": (-29.6875, 152.9331, "Australia/Sydney"),
    "COFFS HARBOUR": (-30.3072, 153.1033, "Australia/Sydney"),
    "PORT MACQUARIE": (-31.4278, 152.8811, "Australia/Sydney"),
    "TAMWORTH": (-31.0839, 150.9264, "Australia/Sydney"),
    "CANBERRA": (-35.3169, 149.1494, "Australia/Sydney"),
    "GOULBURN": (-34.7492, 149.7261, "Australia/Sydney"),
    "NOWRA": (-34.8794, 150.6011, "Australia/Sydney"),
    "TAREE": (-31.9081, 152.4531, "Australia/Sydney"),
    # Queensland
    "EAGLE FARM": (-27.4297, 153.0733, "Australia/Brisbane"),
    "DOOMBEN": (-27.4247, 153.0642, "Australia/Brisbane"),
    "GOLD COAST": (-28.0136, 153.3639, "Australia/Brisbane"),
    "SUNSHINE COAST": (-26.7822, 153.0894, "Australia/Brisbane"),
    "IPSWICH": (-27.6169, 152.7594, "Australia/Brisbane"),
    "TOOWOOMBA": (-27.5619, 151.9539, "Australia/Brisbane"),
    "ROCKHAMPTON": (-23.3789, 150.5058, "Australia/Brisbane"),
    "TOWNSVILLE": (-19.3122, 146.7581, "Australia/Brisbane"),
    "CAIRNS": (-16.9200, 145.7600, "Australia/Brisbane"),
    "MACKAY": (-21.1428, 149.1747, "Australia/Brisbane"),
    "TOOWOOMBA CLIFFORD PARK": (-27.5619, 151.9539, "Australia/Brisbane"),
    # South Australia
    "MORPHETTVILLE": (-34.9756, 138.5439, "Australia/Adelaide"),
    "MURRAY BRIDGE": (-35.1258, 139.2733, "Australia/Adelaide"),
    "GAWLER": (-34.5983, 138.7522, "Australia/Adelaide"),
    "BALAKLAVA": (-34.1461, 138.4131, "Australia/Adelaide"),
    "STRATHALBYN": (-35.2597, 138.8931, "Australia/Adelaide"),
    # Western Australia
    "ASCOT": (-31.9139, 115.9269, "Australia/Perth"),
    "BELMONT": (-31.9436, 115.9081, "Australia/Perth"),
    "BELMONT PARK": (-31.9436, 115.9081, "Australia/Perth"),
    "BUNBURY": (-33.3339, 115.6389, "Australia/Perth"),
    "NORTHAM": (-31.6539, 116.6689, "Australia/Perth"),
    "PINJARRA": (-32.6297, 115.8722, "Australia/Perth"),
    "ALBANY": (-34.9878, 117.8564, "Australia/Perth"),
    "GERALDTON": (-28.7756, 114.6144, "Australia/Perth"),
    "KALGOORLIE": (-30.7594, 121.4653, "Australia/Perth"),
    # Tasmania
    "HOBART": (-42.8322, 147.2917, "Australia/Hobart"),
    "ELWICK": (-42.8322, 147.2917, "Australia/Hobart"),
    "LAUNCESTON": (-41.4064, 147.1425, "Australia/Hobart"),
    "MOWBRAY": (-41.4064, 147.1425, "Australia/Hobart"),
    "DEVONPORT": (-41.1794, 146.3489, "Australia/Hobart"),
    # Northern Territory
    "DARWIN": (-12.4058, 130.8631, "Australia/Darwin"),
    "FANNIE BAY": (-12.4058, 130.8631, "Australia/Darwin"),
    "ALICE SPRINGS": (-23.7036, 133.8828, "Australia/Darwin"),
}


def lookup_course(track: str) -> Optional[tuple[float, float, str]]:
    """Find a racecourse's coordinates, tolerant of naming variations."""
    if not track:
        return None
    key = str(track).strip().upper()
    if key in RACECOURSES:
        return RACECOURSES[key]
    # Betfair sometimes appends a suffix, e.g. "Sandown (AUS)".
    cleaned = key.split("(")[0].strip()
    if cleaned in RACECOURSES:
        return RACECOURSES[cleaned]
    for name, value in RACECOURSES.items():
        if cleaned.startswith(name) or name.startswith(cleaned):
            return value
    return None


def fetch_daily_weather(
    latitude: float,
    longitude: float,
    start: date,
    end: date,
    timezone: str = "Australia/Sydney",
    forecast: bool = False,
) -> pd.DataFrame:
    """Daily weather for one location.

    Uses the archive (reanalysis) endpoint for past dates and the forecast
    endpoint for future ones. The archive has a lag of roughly five days, so
    very recent dates come from the forecast endpoint's past-days window.
    """
    daily = [
        "precipitation_sum", "rain_sum", "precipitation_hours",
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "windspeed_10m_max", "et0_fao_evapotranspiration",
    ]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(daily),
        "timezone": timezone,
    }

    if forecast:
        url = FORECAST_URL
        params["past_days"] = 14
        params["forecast_days"] = 16
    else:
        url = ARCHIVE_URL
        params["start_date"] = start.isoformat()
        params["end_date"] = end.isoformat()

    response = requests.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    if "daily" not in payload:
        raise ValueError(f"Open-Meteo returned no daily block: {payload.get('reason', payload)}")

    frame = pd.DataFrame(payload["daily"])
    frame["date"] = pd.to_datetime(frame["time"]).dt.date
    return frame.drop(columns=["time"])


def with_rainfall_history(weather: pd.DataFrame) -> pd.DataFrame:
    """Add the cumulative-rainfall windows that actually drive track ratings.

    Three days matters most - that is the water still in the surface. Seven and
    fourteen day totals capture how saturated the profile underneath is, which
    is why a track can stay Soft for days after the rain stops.
    """
    frame = weather.sort_values("date").copy()
    rain = frame["precipitation_sum"].fillna(0.0)
    frame["rain_1d"] = rain
    frame["rain_3d"] = rain.rolling(3, min_periods=1).sum()
    frame["rain_7d"] = rain.rolling(7, min_periods=1).sum()
    frame["rain_14d"] = rain.rolling(14, min_periods=1).sum()
    if "et0_fao_evapotranspiration" in frame.columns:
        et0 = frame["et0_fao_evapotranspiration"].fillna(0.0)
        # Net water balance: rain in, evaporation out. Drying is what turns a
        # Soft 7 back into a Good 4 over a few warm days.
        frame["water_balance_7d"] = (rain - et0).rolling(7, min_periods=1).sum()
    return frame


def estimate_track_condition(
    rain_1d: float,
    rain_3d: float,
    rain_7d: float,
    water_balance_7d: float | None = None,
) -> float:
    """Estimated track rating on the Australian 1-10 scale.

    A deliberately simple, transparent piecewise model, tuned to the published
    scale rather than fitted to data:

      3-4  Good    - the default in dry weather
      5-7  Soft    - meaningful rain in the last few days
      8-10 Heavy   - heavy or sustained rain

    The heuristic weights recent rain most heavily and lets the seven-day total
    add a saturation penalty. `fit_condition_model` replaces these constants
    with fitted ones once you have real ratings to fit against.

    This is an estimate. It will be wrong on individual days - most obviously
    when a track is irrigated in dry weather, which no rainfall model can see.
    """
    rain_1d = float(rain_1d or 0.0)
    rain_3d = float(rain_3d or 0.0)
    rain_7d = float(rain_7d or 0.0)

    score = 3.5                                  # a dry track sits at Good 3-4
    score += 0.45 * min(rain_1d, 25.0)           # today's rain dominates
    score += 0.18 * min(max(rain_3d - rain_1d, 0.0), 40.0)
    score += 0.06 * min(max(rain_7d - rain_3d, 0.0), 60.0)

    if water_balance_7d is not None and np.isfinite(water_balance_7d):
        # A strongly negative balance means the track has been drying out.
        score += 0.03 * float(np.clip(water_balance_7d, -40.0, 40.0))

    return float(np.clip(score, 1.0, 10.0))


def attach_weather(
    frame: pd.DataFrame,
    cache: dict | None = None,
    estimate_condition: bool = True,
    quiet: bool = False,
) -> pd.DataFrame:
    """Join weather onto a canonical runner frame, one request per track.

    Requests are batched by (track, date-range) and cached, so a season of data
    for 40 tracks costs 40 requests rather than one per race.
    """
    frame = frame.copy()
    cache = cache if cache is not None else {}

    frame["_date"] = pd.to_datetime(frame["race_datetime"], errors="coerce").dt.date
    unknown_tracks: set[str] = set()

    for track, group in frame.groupby("track"):
        course = lookup_course(track)
        if course is None:
            unknown_tracks.add(str(track))
            continue
        lat, lon, tz = course

        dates = group["_date"].dropna()
        if dates.empty:
            continue
        start = min(dates) - timedelta(days=15)   # room for the 14-day window
        end = max(dates)
        today = date.today()

        key = (round(lat, 3), round(lon, 3), start, end)
        if key not in cache:
            try:
                if end > today:
                    weather = fetch_daily_weather(lat, lon, start, end, tz, forecast=True)
                else:
                    weather = fetch_daily_weather(lat, lon, start, min(end, today), tz)
                cache[key] = with_rainfall_history(weather)
            except Exception as exc:
                if not quiet:
                    print(f"[weather] {track}: {exc}")
                cache[key] = None
        weather = cache[key]
        if weather is None:
            continue

        lookup = weather.set_index("date")
        idx = group.index
        dates_for_rows = frame.loc[idx, "_date"]

        for column, target in (
            ("temperature_2m_mean", "temp_c"),
            ("precipitation_sum", "rain_mm_24h"),
            ("windspeed_10m_max", "wind_kph"),
            ("rain_3d", "rain_mm_3d"),
            ("rain_7d", "rain_mm_7d"),
            ("rain_14d", "rain_mm_14d"),
            ("water_balance_7d", "water_balance_7d"),
        ):
            if column in lookup.columns:
                frame.loc[idx, target] = dates_for_rows.map(lookup[column]).to_numpy()

    if estimate_condition:
        needs = frame["track_condition"].isna() if "track_condition" in frame.columns \
            else pd.Series(True, index=frame.index)
        have_rain = frame.get("rain_mm_24h", pd.Series(np.nan, index=frame.index)).notna()
        target = needs & have_rain
        if target.any():
            estimates = [
                estimate_track_condition(
                    row.get("rain_mm_24h"), row.get("rain_mm_3d"),
                    row.get("rain_mm_7d"), row.get("water_balance_7d"),
                )
                for _, row in frame.loc[target].iterrows()
            ]
            frame.loc[target, "condition_estimated"] = estimates
            frame.loc[target, "track_condition"] = [round(e) for e in estimates]
            frame.loc[target, "condition_is_estimate"] = True

    if unknown_tracks and not quiet:
        print(f"[weather] No coordinates for {len(unknown_tracks)} track(s); "
              f"add them to RACECOURSES: {sorted(unknown_tracks)[:10]}")

    return frame.drop(columns=["_date"])


def fit_condition_model(actual: Iterable[float], estimated: Iterable[float]) -> dict:
    """Compare estimated ratings against real ones, once you have real ones.

    Returns the mean absolute error and the proportion landing in the right
    named band, which is the number that matters - being one point out inside
    "Soft" is far less harmful than calling a Heavy 8 track Good 4.
    """
    a = np.asarray(list(actual), dtype=float)
    e = np.asarray(list(estimated), dtype=float)
    mask = np.isfinite(a) & np.isfinite(e)
    if mask.sum() == 0:
        return {"n": 0}
    a, e = a[mask], e[mask]

    def band(values):
        return np.select(
            [values <= 2, values <= 4, values <= 7],
            ["FIRM", "GOOD", "SOFT"], default="HEAVY",
        )

    return {
        "n": int(mask.sum()),
        "mean_absolute_error": float(np.mean(np.abs(a - e))),
        "bias": float(np.mean(e - a)),
        "band_accuracy": float(np.mean(band(a) == band(e))),
        "correlation": float(np.corrcoef(a, e)[0, 1]) if len(a) > 2 else float("nan"),
    }
