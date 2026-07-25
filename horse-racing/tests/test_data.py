"""Data layer tests: parsers, store, track registry, simulator realism."""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pytest

from ausform.data.betfair import BetfairHistorical
from ausform.data.simulator import SeasonSimulator
from ausform.data.store import RaceStore
from ausform.data.tab import TabClient
from ausform.data.weather import WeatherClient, estimate_condition_shift
from ausform.tracks import FLEMINGTON, get_track
from ausform.types import Weather, condition_band, condition_to_number

BSP_CSV = """EVENT_ID,MENU_HINT,EVENT_NAME,EVENT_DT,SELECTION_ID,SELECTION_NAME,WIN_LOSE,BSP,PPWAP,MORNINGWAP,PPMAX,PPMIN,IPMAX,IPMIN,MORNINGTRADEDVOL,PPTRADEDVOL,IPTRADEDVOL
991,AUS / Randwick (AUS) 15th Mar,R7 1400m Grp3,15-03-2025 05:30,55,7. Bold Ambition,1,4.35,4.5,5.2,6.0,4.1,3.2,1.01,120.5,3400.2,900.1
991,AUS / Randwick (AUS) 15th Mar,R7 1400m Grp3,15-03-2025 05:30,56,2. Silent Prince,0,9.80,10.2,11.0,13.0,9.0,40.0,2.0,80.0,1200.0,300.0
992,AUS / Flemington (AUS) 15th Mar,R1 1200m Mdn,15-03-2025 03:05,57,1. Golden Belle,0,2.50,2.6,2.8,3.0,2.4,5.0,1.5,300.0,5000.0,1000.0
"""


# --- track registry -------------------------------------------------------

def test_track_lookup_by_code_name_and_alias():
    assert get_track("FLEM") is FLEMINGTON
    assert get_track("Flemington") is FLEMINGTON
    assert get_track("the valley").code == "MVAL"
    assert get_track("Moonee Valley").code == "MVAL"
    assert get_track("Royal Randwick").code == "RAND"
    assert get_track("nonexistent track") is None
    assert get_track("") is None


def test_track_coordinates_are_in_australia():
    from ausform.tracks import all_tracks
    for track in all_tracks():
        assert -45 < track.latitude < -10, f"{track.name} latitude looks wrong"
        assert 112 < track.longitude < 155, f"{track.name} longitude looks wrong"


# --- track condition ------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Good 4", 4), ("Soft 6", 6), ("HEAVY 10", 10), ("Firm 2", 2),
    ("Good(3)", 3), ("soft", 6), ("good", 3), ("heavy", 9), ("", None),
    ("unknown", None),
])
def test_condition_parsing(text, expected):
    assert condition_to_number(text) == expected


def test_condition_band_round_trip():
    assert condition_band(4) == "good"
    assert condition_band(6) == "soft"
    assert condition_band(9) == "heavy"
    assert condition_band(1) == "firm"
    assert condition_band(None) is None


# --- Betfair --------------------------------------------------------------

def test_bsp_url_format():
    client = BetfairHistorical()
    url = client.bsp_url(_dt.date(2025, 3, 15))
    assert url.endswith("dwbfpricesauswin15032025.csv")
    assert client.bsp_url(_dt.date(2025, 3, 15), "place").endswith(
        "dwbfpricesausplace15032025.csv")
    with pytest.raises(ValueError):
        client.bsp_url(_dt.date(2025, 3, 15), "trifecta")


def test_bsp_parsing():
    rows = list(BetfairHistorical.parse_bsp(BSP_CSV))
    assert len(rows) == 3

    first = rows[0]
    assert first.track_name == "Randwick"
    assert first.race_number == 7
    assert first.distance_m == 1400
    assert first.saddlecloth == 7
    assert first.horse_name == "Bold Ambition"
    assert first.won is True
    assert first.bsp == pytest.approx(4.35)
    assert first.event_date == _dt.date(2025, 3, 15)

    assert rows[1].won is False


def test_bsp_to_races_groups_by_event():
    races = BetfairHistorical.to_races(list(BetfairHistorical.parse_bsp(BSP_CSV)))
    assert len(races) == 2
    randwick = next(r for r in races if r.track.code == "RAND")
    assert len(randwick.runners) == 2
    assert randwick.winner.name == "Bold Ambition"


def test_bsp_parsing_tolerates_junk():
    junk = ("EVENT_ID,MENU_HINT,EVENT_NAME,EVENT_DT,SELECTION_NAME,WIN_LOSE,BSP\n"
            "1,x,y,not-a-date,1. Horse,1,2.0\n"
            "2,AUS / Randwick (AUS),R1 1200m,15-03-2025 05:30,,1,2.0\n")
    rows = list(BetfairHistorical.parse_bsp(junk))
    assert rows == []  # both rows are unusable and must be dropped, not crash


# --- TAB ------------------------------------------------------------------

def test_tab_runner_parsing():
    client = TabClient()
    runner = client._parse_runner({
        "runnerName": "STORM FORCE TEN",
        "runnerNumber": 1,
        "barrierNumber": 4,
        "handicapWeight": 58.5,
        "riderDriverName": "J McDonald",
        "trainerName": "C Waller",
        "earlySpeedRating": 89,
        "earlySpeedRatingBand": "LEADER",
        "last5Starts": "31204",
        "fixedOdds": {"returnWin": 3.0, "returnPlace": 1.24,
                      "returnWinOpen": 2.9, "bettingStatus": "Open"},
        "parimutuel": {"returnWin": 3.4},
    })
    assert runner is not None
    assert runner.name == "STORM FORCE TEN"
    assert runner.number == 1
    assert runner.barrier == 4
    assert runner.weight_kg == pytest.approx(58.5)
    assert runner.jockey == "J McDonald"
    assert runner.fixed_win_odds == pytest.approx(3.0)
    assert runner.fixed_place_odds == pytest.approx(1.24)
    assert runner.tote_win_odds == pytest.approx(3.4)
    assert runner.early_speed_rating == pytest.approx(89)
    assert runner.scratched is False


def test_tab_detects_scratchings_and_bad_rows():
    client = TabClient()
    scratched = client._parse_runner({
        "runnerName": "X", "runnerNumber": 2, "scratched": True,
        "fixedOdds": {}, "parimutuel": {},
    })
    assert scratched.scratched is True
    assert client._parse_runner({"runnerNumber": 3}) is None
    assert client._parse_runner({}) is None


def test_tab_filters_to_australian_thoroughbreds():
    client = TabClient()
    payload = {"meetings": [
        {"raceType": "R", "meetingName": "Randwick", "location": "NSW",
         "venueMnemonic": "RAN", "trackCondition": "Good 4",
         "railPosition": "True", "races": [
             {"raceNumber": 1, "raceDistance": 1200, "raceName": "Test"}]},
        {"raceType": "G", "meetingName": "Wentworth Park", "location": "NSW",
         "races": []},                                  # greyhounds: drop
        {"raceType": "R", "meetingName": "Ascot UK", "location": "GBR",
         "races": []},                                  # overseas: drop
    ]}
    meetings = list(client._parse_meetings(payload, _dt.date(2025, 3, 15)))
    assert len(meetings) == 1
    assert meetings[0].track.code == "RAND"
    assert meetings[0].track_condition == 4
    assert meetings[0].races[0].venue_mnemonic == "RAN"


# --- weather --------------------------------------------------------------

def test_weather_parsing_and_rain_windows():
    payload = {"hourly": {
        "time": [f"2025-03-15T{h:02d}:00" for h in range(24)],
        "temperature_2m": [15.0] * 24,
        "relative_humidity_2m": [70.0] * 24,
        "precipitation": [1.0] * 24,
        "wind_speed_10m": [12.0] * 24,
        "wind_direction_10m": [180.0] * 24,
    }}
    weather = WeatherClient._parse(payload, _dt.date(2025, 3, 15), 14, False)
    assert weather is not None
    assert weather.temperature_c == pytest.approx(15.0)
    # 15 hourly readings of 1mm up to and including hour 14.
    assert weather.rainfall_mm_24h == pytest.approx(15.0)


def test_weather_returns_none_when_date_absent():
    payload = {"hourly": {"time": ["2025-03-14T14:00"], "temperature_2m": [10.0]}}
    assert WeatherClient._parse(payload, _dt.date(2025, 3, 15), 14, False) is None


def test_condition_shift_monotonic_in_rain():
    dry = estimate_condition_shift(Weather(rainfall_mm_24h=0.0, rainfall_mm_72h=0.0))
    damp = estimate_condition_shift(Weather(rainfall_mm_24h=8.0, rainfall_mm_72h=10.0))
    soaked = estimate_condition_shift(Weather(rainfall_mm_24h=60.0, rainfall_mm_72h=80.0))
    assert dry == 0.0
    assert 0 < damp < soaked
    assert estimate_condition_shift(None) == 0.0


# --- store ----------------------------------------------------------------

def test_store_round_trip(tmp_path):
    simulator = SeasonSimulator(seed=3, n_horses=800)
    races = simulator.simulate_season(
        _dt.date(2023, 1, 1), days=20, meetings_per_day=2, races_per_meeting=4)

    db = tmp_path / "test.db"
    with RaceStore(db) as store:
        stored = store.import_races(races)
        assert stored > 0
        stats = store.stats()
        assert stats["races"] == len(races)

        rebuilt = list(store.iter_races())
        assert len(rebuilt) == len(races)
        assert any(r.winner is not None for r in rebuilt)


def test_store_csv_import_and_validation(tmp_path):
    csv_path = tmp_path / "form.csv"
    csv_path.write_text(
        "race_id,date,track,race_number,distance_m,horse_id,horse_name,"
        "finish_position,barrier,weight_kg,jockey,track_condition\n"
        "R1,2024-01-05,Flemington,1,1200,H1,Bold Ambition,1,3,57.5,J Smith,Good 4\n"
        "R1,2024-01-05,Flemington,1,1200,H2,Silent Prince,2,7,56.0,A Jones,Good 4\n"
    )
    db = tmp_path / "t.db"
    with RaceStore(db) as store:
        imported, skipped = store.import_csv(csv_path)
        assert imported == 2 and skipped == 0
        races = list(store.iter_races())
        assert len(races) == 1
        assert races[0].track.code == "FLEM"
        assert races[0].track_condition == 4
        assert races[0].winner.name == "Bold Ambition"


def test_store_rejects_csv_missing_required_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("race_id,date\nR1,2024-01-05\n")
    with RaceStore(tmp_path / "x.db") as store:
        with pytest.raises(ValueError, match="missing required columns"):
            store.import_csv(bad)


def test_store_history_is_point_in_time(tmp_path):
    csv_path = tmp_path / "form.csv"
    csv_path.write_text(
        "race_id,date,track,race_number,distance_m,horse_id,horse_name,finish_position\n"
        "R1,2024-01-05,Flemington,1,1200,H1,Horse One,1\n"
        "R2,2024-02-05,Flemington,1,1200,H1,Horse One,2\n"
        "R3,2024-03-05,Flemington,1,1200,H1,Horse One,3\n"
    )
    with RaceStore(tmp_path / "t.db") as store:
        store.import_csv(csv_path)
        history = store.horse_history("H1", _dt.date(2024, 2, 5))
        assert len(history) == 1                       # only January is visible
        assert history[0].date == _dt.date(2024, 1, 5)


# --- simulator realism ----------------------------------------------------

def test_simulator_produces_realistic_market():
    """Guards the calibration. If these drift, every downstream number
    becomes meaningless."""
    simulator = SeasonSimulator(seed=11, n_horses=6000)
    races = simulator.simulate_season(
        _dt.date(2022, 1, 1), days=75, meetings_per_day=3, races_per_meeting=8)

    odds = np.array([r.fixed_win_odds for race in races for r in race.runners])
    assert odds.max() <= 301.0, "no real book quotes beyond about $301"
    assert odds.min() >= 1.0
    assert 5.0 < np.median(odds) < 40.0

    favourite_wins = total = 0
    for race in races:
        live = [r for r in race.runners if r.fixed_win_odds]
        if len(live) < 5:
            continue
        favourite = min(live, key=lambda r: r.fixed_win_odds)
        total += 1
        favourite_wins += int(favourite.result_position == 1)

    strike = favourite_wins / total
    assert 0.28 < strike < 0.38, (
        f"favourite strike rate {strike:.1%} is outside the realistic "
        f"Australian band of roughly 31-35%")


def test_simulator_horses_do_not_race_too_often():
    simulator = SeasonSimulator(seed=5, n_horses=6000)
    simulator.simulate_season(_dt.date(2022, 1, 1), days=365,
                              meetings_per_day=3, races_per_meeting=8)
    starts = [h.starts for h in simulator.horses if h.starts > 0]
    assert np.mean(starts) < 20, "horses are racing far more often than reality"


def test_simulator_history_is_snapshot_before_the_race():
    simulator = SeasonSimulator(seed=8, n_horses=1200)
    races = simulator.simulate_season(
        _dt.date(2023, 1, 1), days=60, meetings_per_day=2, races_per_meeting=5)
    for race in races[-40:]:
        for runner in race.runners:
            for past in runner.history:
                assert past.date < race.date, (
                    "a runner's history contains the race being run")
