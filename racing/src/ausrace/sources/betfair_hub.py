"""Betfair's free, publicly published Australian racing data.

This is the backbone of the whole tool, for one reason: it is the only source
of several years of Australian thoroughbred results *with prices attached* that
is free, requires no account, and that Betfair publishes explicitly for people
building models. Everything else either costs money or would breach someone's
terms of service.

Two files are used:

1. **ANZ Thoroughbreds market CSVs** - one row per runner per race, with the
   Betfair Starting Price, the best price available at the jump, traded volume,
   and the result. Monthly files from 2026, annual zips before that.
   https://betfair-datascientists.github.io/data/dataListing/

2. **Betfair SP promo files** - a daily settled file, used to keep the dataset
   current between monthly publications.
   https://promo.betfair.com/betfairsp/prices

A deliberate omission, stated up front: **neither file carries the track
condition**, which is one of the most important variables in Australian racing.
The weather module fills that gap approximately, using rainfall history as a
proxy. "Approximately" is doing real work in that sentence - see
`weather.estimate_track_condition`.

What this module will not do: scrape racingaustralia.horse, racing.com,
punters.com.au or racenet.com.au. Racing Australia's terms of use prohibit
automated collection by name, and the others take the same position. The
scrapers are easy to write and that is not the point.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

HUB_BASE = "https://betfair-datascientists.github.io/data/assets"
BSP_BASE = "https://promo.betfair.com/betfairsp/prices"

USER_AGENT = "ausrace-model/1.0 (personal racing model; contact: local user)"
TIMEOUT = 60


def _get(url: str) -> requests.Response:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    response.raise_for_status()
    return response


# --------------------------------------------------------------------------
# ANZ Thoroughbreds monthly / annual files
# --------------------------------------------------------------------------

def hub_monthly_url(year: int, month: int) -> str:
    return f"{HUB_BASE}/ANZ_Thoroughbreds_{year}_{month:02d}.csv"


def hub_annual_url(year: int) -> str:
    return f"{HUB_BASE}/ANZ_Thoroughbreds_{year}.zip"


def fetch_hub_month(year: int, month: int, cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Download one month of ANZ thoroughbred market data."""
    url = hub_monthly_url(year, month)
    if cache_dir:
        cached = Path(cache_dir) / f"ANZ_Thoroughbreds_{year}_{month:02d}.csv"
        if cached.exists():
            return pd.read_csv(cached)
        cached.parent.mkdir(parents=True, exist_ok=True)
        text = _get(url).text
        cached.write_text(text)
        return pd.read_csv(io.StringIO(text))
    return pd.read_csv(io.StringIO(_get(url).text))


def fetch_hub_year(year: int, cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Download and unpack one annual zip of ANZ thoroughbred market data."""
    url = hub_annual_url(year)
    if cache_dir:
        cached = Path(cache_dir) / f"ANZ_Thoroughbreds_{year}.zip"
        cached.parent.mkdir(parents=True, exist_ok=True)
        if not cached.exists():
            cached.write_bytes(_get(url).content)
        payload = cached.read_bytes()
    else:
        payload = _get(url).content

    frames = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            if name.lower().endswith(".csv"):
                with archive.open(name) as fh:
                    frames.append(pd.read_csv(fh))
    if not frames:
        raise ValueError(f"No CSV found inside {url}")
    return pd.concat(frames, ignore_index=True)


def fetch_hub_range(
    start: date,
    end: date,
    cache_dir: str | Path | None = "data/raw/betfair_hub",
    on_error: str = "warn",
) -> pd.DataFrame:
    """Fetch every ANZ thoroughbred file covering [start, end].

    Uses annual zips for complete past years and monthly CSVs otherwise, which
    is both faster and fewer requests. Months that 404 (not yet published) are
    skipped with a warning rather than aborting the whole load.
    """
    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for year in range(start.year, end.year + 1):
        whole_year = start <= date(year, 1, 1) and end >= date(year, 12, 31)
        if whole_year:
            try:
                frames.append(fetch_hub_year(year, cache_dir=cache_dir))
                continue
            except Exception as exc:
                errors.append(f"{year} annual: {exc}")

        first = 1 if year > start.year else start.month
        last = 12 if year < end.year else end.month
        for month in range(first, last + 1):
            try:
                frames.append(fetch_hub_month(year, month, cache_dir=cache_dir))
            except Exception as exc:
                errors.append(f"{year}-{month:02d}: {exc}")

    if errors and on_error == "raise":
        raise RuntimeError("Failed to fetch: " + "; ".join(errors))
    if errors and on_error == "warn":
        print(f"[betfair_hub] {len(errors)} file(s) unavailable "
              f"(usually not yet published): {errors[:5]}")

    if not frames:
        raise RuntimeError(
            "No Betfair Hub data could be downloaded. Check your internet "
            "connection, and confirm the files exist at "
            "https://betfair-datascientists.github.io/data/dataListing/"
        )

    combined = pd.concat(frames, ignore_index=True)
    mask = pd.to_datetime(combined["LOCAL_MEETING_DATE"], errors="coerce", dayfirst=True).dt.date
    return combined[(mask >= start) & (mask <= end)].reset_index(drop=True)


def parse_hub(frame: pd.DataFrame, country: str | None = "AU") -> pd.DataFrame:
    """Convert Betfair Hub columns to the canonical schema.

    Prices: `WIN_BSP` is the Betfair Starting Price, the price the bet was
    actually settled at. `BEST_AVAIL_BACK_AT_SCHEDULED_OFF` is the best price
    that was on offer at the scheduled jump time. The BSP is the honest choice
    for backtesting, because it is a price you could actually have taken - it
    is what a bet placed at SP would have got. Backtesting against the best
    price ever available during the day is a classic way to invent an edge that
    does not exist.
    """
    frame = frame.copy()
    frame.columns = [c.strip().upper() for c in frame.columns]

    required = {"WIN_MARKET_ID", "SELECTION_ID", "SELECTION_NAME", "LOCAL_MEETING_DATE"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Betfair Hub file is missing expected column(s): {sorted(missing)}. "
            f"Betfair may have changed the format - got: {sorted(frame.columns)[:20]}"
        )

    if country and "STATE_CODE" in frame.columns:
        # NZ state codes are distinguishable; keep AU only when asked.
        nz_codes = {"NZ", "NZL"}
        if country.upper() in {"AU", "AUS"}:
            frame = frame[~frame["STATE_CODE"].astype(str).str.upper().isin(nz_codes)]

    meeting_date = pd.to_datetime(frame["LOCAL_MEETING_DATE"], errors="coerce", dayfirst=True)
    if "SCHEDULED_RACE_TIME" in frame.columns:
        race_dt = pd.to_datetime(frame["SCHEDULED_RACE_TIME"], errors="coerce", dayfirst=True)
        race_dt = race_dt.fillna(meeting_date)
    else:
        race_dt = meeting_date

    out = pd.DataFrame({
        "race_id": frame["WIN_MARKET_ID"].astype(str),
        "race_datetime": race_dt,
        "track": frame.get("TRACK", pd.Series("UNKNOWN", index=frame.index)).astype(str).str.strip(),
        "state": frame.get("STATE_CODE"),
        "race_number": pd.to_numeric(frame.get("RACE_NO"), errors="coerce"),
        "distance_m": pd.to_numeric(frame.get("DISTANCE"), errors="coerce"),
        "race_class": frame.get("RACE_TYPE"),
        "runner_id": frame["SELECTION_ID"].astype(str),
        "horse": frame["SELECTION_NAME"].astype(str).str.strip(),
        "bsp": pd.to_numeric(frame.get("WIN_BSP"), errors="coerce"),
    })

    # Strip the tab number that Betfair prefixes onto runner names ("7. Fast Lad").
    out["horse"] = out["horse"].str.replace(r"^\s*\d+\.\s*", "", regex=True)

    if "TAB_NUMBER" in frame.columns:
        out["tab_number"] = pd.to_numeric(frame["TAB_NUMBER"], errors="coerce")

    # Result. WIN_RESULT is 1 for the winner, 0 otherwise. Betfair gives us the
    # winner but not the full finishing order, so places beyond first are
    # unknown from this source alone.
    if "WIN_RESULT" in frame.columns:
        result = pd.to_numeric(frame["WIN_RESULT"], errors="coerce")
        out["finish_position"] = np.where(result == 1, 1.0, np.nan)
        # A runner with a known result that did not win finished somewhere
        # other than first; encode that as 99 so `won` is correctly False
        # while making clear the exact position is unknown.
        out.loc[(result == 0), "finish_position"] = 99.0
    if "PLACE_RESULT" in frame.columns:
        place_result = pd.to_numeric(frame["PLACE_RESULT"], errors="coerce")
        out["place_result"] = place_result

    # Price to bet at.
    best_at_off = pd.to_numeric(
        frame.get("BEST_AVAIL_BACK_AT_SCHEDULED_OFF"), errors="coerce"
    ) if "BEST_AVAIL_BACK_AT_SCHEDULED_OFF" in frame.columns else pd.Series(np.nan, index=frame.index)
    out["odds_decimal"] = out["bsp"].where(out["bsp"].notna() & (out["bsp"] > 1.0), best_at_off)
    out["best_avail_at_off"] = best_at_off

    for col in ("WIN_PREPLAY_VOLUME", "WIN_PREPLAY_WEIGHTED_AVERAGE_PRICE_TAKEN",
                "WIN_PREPLAY_MIN_PRICE_TAKEN", "WIN_PREPLAY_MAX_PRICE_TAKEN"):
        if col in frame.columns:
            out[col.lower()] = pd.to_numeric(frame[col], errors="coerce")

    out["scratched"] = False
    out = out[out["race_id"].notna() & out["runner_id"].notna()]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Daily BSP promo files
# --------------------------------------------------------------------------

def bsp_url(day: date, market: str = "win", region: str = "aus") -> str:
    """URL for a daily Betfair SP file. Note the date format is DDMMYYYY."""
    if market not in {"win", "place"}:
        raise ValueError("market must be 'win' or 'place'")
    return f"{BSP_BASE}/dwbfprices{region}{market}{day.strftime('%d%m%Y')}.csv"


def fetch_bsp_day(day: date, market: str = "win", region: str = "aus") -> pd.DataFrame:
    """One day of settled Betfair SP data."""
    return pd.read_csv(io.StringIO(_get(bsp_url(day, market, region)).text))


def fetch_bsp_range(start: date, end: date, market: str = "win",
                    region: str = "aus", quiet: bool = False) -> pd.DataFrame:
    """Every daily BSP file between two dates. Missing days are skipped."""
    frames, missing = [], 0
    day = start
    while day <= end:
        try:
            frames.append(fetch_bsp_day(day, market=market, region=region))
        except Exception:
            missing += 1
        day += timedelta(days=1)
    if not quiet and missing:
        print(f"[betfair_bsp] {missing} day(s) had no file (dark days or not yet published).")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# Harness and greyhound meetings share the 'aus' file, so thoroughbred meetings
# have to be separated out via MENU_HINT. These markers appear in the menu hint
# for the other two codes.
NON_THOROUGHBRED_MARKERS = ("(GRYD)", "(HARN)", "GREY", "HARNESS", "TROT", "PACE")


def parse_bsp(frame: pd.DataFrame, thoroughbred_only: bool = True) -> pd.DataFrame:
    """Convert a Betfair SP file to the canonical schema."""
    frame = frame.copy()
    frame.columns = [c.strip().upper() for c in frame.columns]

    required = {"EVENT_ID", "SELECTION_ID", "SELECTION_NAME", "BSP", "WIN_LOSE"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"BSP file missing column(s): {sorted(missing)}")

    if thoroughbred_only and "MENU_HINT" in frame.columns:
        hint = frame["MENU_HINT"].astype(str).str.upper()
        frame = frame[~hint.str.contains("|".join(NON_THOROUGHBRED_MARKERS), regex=True, na=False)]

    event_dt = pd.to_datetime(frame.get("EVENT_DT"), errors="coerce", dayfirst=True)

    # MENU_HINT looks like "AUS / Randwick (AUS) 5th Jul"; the track is the
    # part after the last '/' with the country suffix removed.
    track = (
        frame.get("MENU_HINT", pd.Series("", index=frame.index))
        .astype(str).str.split("/").str[-1]
        .str.replace(r"\(.*?\)", "", regex=True)
        .str.replace(r"\d+\w{2}\s+\w+$", "", regex=True)
        .str.strip()
    )

    out = pd.DataFrame({
        "race_id": frame["EVENT_ID"].astype(str),
        "race_datetime": event_dt,
        "track": track,
        "runner_id": frame["SELECTION_ID"].astype(str),
        "horse": frame["SELECTION_NAME"].astype(str)
                 .str.replace(r"^\s*\d+\.\s*", "", regex=True).str.strip(),
        "bsp": pd.to_numeric(frame["BSP"], errors="coerce"),
        "odds_decimal": pd.to_numeric(frame["BSP"], errors="coerce"),
        "scratched": False,
    })

    won = pd.to_numeric(frame["WIN_LOSE"], errors="coerce")
    out["finish_position"] = np.where(won == 1, 1.0, np.where(won == 0, 99.0, np.nan))

    # EVENT_NAME usually carries the race number and distance, e.g. "R5 1400m".
    if "EVENT_NAME" in frame.columns:
        name = frame["EVENT_NAME"].astype(str)
        out["race_number"] = pd.to_numeric(
            name.str.extract(r"R(\d+)", expand=False), errors="coerce")
        out["distance_m"] = pd.to_numeric(
            name.str.extract(r"(\d{3,4})\s*[mM]\b", expand=False), errors="coerce")

    for col, target in (("PPWAP", "preplay_vwap"), ("MORNINGWAP", "morning_vwap"),
                        ("PPMAX", "preplay_max"), ("PPMIN", "preplay_min"),
                        ("PPTRADEDVOL", "preplay_volume")):
        if col in frame.columns:
            out[target] = pd.to_numeric(frame[col], errors="coerce")

    return out[out["race_id"].notna()].reset_index(drop=True)


def load_australian_history(
    start: date,
    end: date,
    cache_dir: str | Path | None = "data/raw/betfair_hub",
) -> pd.DataFrame:
    """The convenience entry point: everything Betfair publishes for AU racing
    between two dates, in canonical form."""
    raw = fetch_hub_range(start, end, cache_dir=cache_dir)
    return parse_hub(raw, country="AU")
