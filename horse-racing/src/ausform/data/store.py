"""Local historical store (SQLite) plus CSV import/export.

The model is only as good as the history behind it. This module is the
place you pour whatever historical form you can get -- a purchased feed, a
Kaggle dump, years of your own scraped results -- into one normalised
shape the rest of the tool understands.

CSV SCHEMA
----------
One row per runner per race. Required columns:

    race_id, date, track, race_number, distance_m, horse_id, horse_name,
    finish_position

Optional but strongly recommended (each materially improves the model):

    field_size, barrier, weight_kg, jockey, trainer, track_condition,
    class_level, prize_money, starting_price, margin_l, race_time_s,
    last_600m_s, age, sex, sire, scratched, surface, rail

Dates are ISO (YYYY-MM-DD). `track` accepts a code, full name or common
alias. `track_condition` accepts either the 1-10 number or text such as
"Soft 6". `starting_price` is decimal odds including stake (4.50, not 3/1).
"""

from __future__ import annotations

import csv
import datetime as _dt
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator, Optional

from ..tracks import get_track, unknown_track
from ..types import PastRun, Race, Runner, Surface, condition_to_number

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "race_id", "date", "track", "race_number", "distance_m",
    "horse_id", "horse_name", "finish_position",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    race_id         TEXT NOT NULL,
    date            TEXT NOT NULL,
    track_code      TEXT NOT NULL,
    track_name      TEXT,
    race_number     INTEGER,
    distance_m      INTEGER,
    class_level     TEXT,
    prize_money     REAL,
    track_condition INTEGER,
    surface         TEXT,
    rail            TEXT,
    field_size      INTEGER,
    horse_id        TEXT NOT NULL,
    horse_name      TEXT,
    saddlecloth     INTEGER,
    barrier         INTEGER,
    weight_kg       REAL,
    jockey          TEXT,
    trainer         TEXT,
    age             INTEGER,
    sex             TEXT,
    sire            TEXT,
    finish_position INTEGER,
    margin_l        REAL,
    starting_price  REAL,
    race_time_s     REAL,
    last_600m_s     REAL,
    scratched       INTEGER DEFAULT 0,
    PRIMARY KEY (race_id, horse_id)
);
CREATE INDEX IF NOT EXISTS idx_runs_horse_date ON runs(horse_id, date);
CREATE INDEX IF NOT EXISTS idx_runs_date       ON runs(date);
CREATE INDEX IF NOT EXISTS idx_runs_race       ON runs(race_id);
CREATE INDEX IF NOT EXISTS idx_runs_jockey     ON runs(jockey, date);
CREATE INDEX IF NOT EXISTS idx_runs_trainer    ON runs(trainer, date);
"""


def _as_int(value) -> Optional[int]:
    if value in (None, "", "-"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value) -> Optional[float]:
    if value in (None, "", "-"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_date(value) -> Optional[_dt.date]:
    if isinstance(value, _dt.date):
        return value
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d %b %Y"):
        try:
            return _dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


class RaceStore:
    """SQLite-backed archive of historical runs."""

    def __init__(self, path: str | Path = "ausform.db"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "RaceStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- ingest ------------------------------------------------------------

    def import_csv(self, csv_path: str | Path, strict: bool = False) -> tuple[int, int]:
        """Load a CSV of historical runs. Returns (imported, skipped).

        With strict=False (default) unparseable rows are logged and skipped
        so one bad line in a 300k-row dump does not abort the import.
        """
        path = Path(csv_path)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = REQUIRED_COLUMNS - columns
            if missing:
                raise ValueError(
                    f"CSV is missing required columns: {sorted(missing)}. "
                    f"See the schema in ausform/data/store.py.")
            return self._insert_rows(reader, strict=strict)

    def _insert_rows(self, rows: Iterable[dict], strict: bool) -> tuple[int, int]:
        imported = skipped = 0
        batch: list[tuple] = []
        for line_no, row in enumerate(rows, start=2):
            try:
                batch.append(self._row_to_tuple(row))
                imported += 1
            except (ValueError, TypeError, KeyError) as exc:
                if strict:
                    raise ValueError(f"Row {line_no}: {exc}") from exc
                log.warning("Skipping row %d: %s", line_no, exc)
                skipped += 1
            if len(batch) >= 5000:
                self._flush(batch)
                batch = []
        if batch:
            self._flush(batch)
        self.conn.commit()
        return imported, skipped

    def _flush(self, batch: list[tuple]) -> None:
        self.conn.executemany(
            """INSERT OR REPLACE INTO runs (
                   race_id, date, track_code, track_name, race_number, distance_m,
                   class_level, prize_money, track_condition, surface, rail,
                   field_size, horse_id, horse_name, saddlecloth, barrier,
                   weight_kg, jockey, trainer, age, sex, sire, finish_position,
                   margin_l, starting_price, race_time_s, last_600m_s, scratched
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            batch,
        )

    @staticmethod
    def _row_to_tuple(row: dict) -> tuple:
        date = _as_date(row.get("date"))
        if date is None:
            raise ValueError(f"unparseable date {row.get('date')!r}")

        track_text = (row.get("track") or "").strip()
        track = get_track(track_text) or unknown_track(track_text or "UNKNOWN")

        condition_raw = row.get("track_condition")
        condition = (_as_int(condition_raw)
                     if str(condition_raw).strip().isdigit()
                     else condition_to_number(str(condition_raw or "")))

        horse_id = (row.get("horse_id") or row.get("horse_name") or "").strip()
        if not horse_id:
            raise ValueError("missing horse_id/horse_name")

        return (
            str(row["race_id"]).strip(),
            date.isoformat(),
            track.code,
            track.name,
            _as_int(row.get("race_number")),
            _as_int(row.get("distance_m")),
            (row.get("class_level") or None),
            _as_float(row.get("prize_money")),
            condition,
            (row.get("surface") or Surface.TURF.value),
            (row.get("rail") or None),
            _as_int(row.get("field_size")),
            horse_id,
            (row.get("horse_name") or horse_id).strip(),
            _as_int(row.get("saddlecloth")),
            _as_int(row.get("barrier")),
            _as_float(row.get("weight_kg")),
            (row.get("jockey") or None),
            (row.get("trainer") or None),
            _as_int(row.get("age")),
            (row.get("sex") or None),
            (row.get("sire") or None),
            _as_int(row.get("finish_position")),
            _as_float(row.get("margin_l")),
            _as_float(row.get("starting_price")),
            _as_float(row.get("race_time_s")),
            _as_float(row.get("last_600m_s")),
            1 if str(row.get("scratched", "")).strip().lower() in ("1", "true", "yes") else 0,
        )

    def import_races(self, races: Iterable[Race]) -> int:
        """Persist Race objects (e.g. simulator output) into the store."""
        batch = []
        for race in races:
            for runner in race.runners:
                batch.append((
                    race.race_id, race.date.isoformat(), race.track.code,
                    race.track.name, race.race_number, race.distance_m,
                    race.class_level, race.prize_money, race.track_condition,
                    race.track.surface.value, race.rail.value, race.field_size,
                    runner.horse_id, runner.name, runner.number, runner.barrier,
                    runner.weight_kg, runner.jockey, runner.trainer, runner.age,
                    runner.sex, runner.sire, runner.result_position, None,
                    runner.fixed_win_odds, None, None,
                    1 if runner.scratched else 0,
                ))
        self._flush(batch)
        self.conn.commit()
        return len(batch)

    # -- read --------------------------------------------------------------

    def horse_history(self, horse_id: str, before: _dt.date) -> list[PastRun]:
        """Point-in-time form for one horse, most recent first.

        `before` is exclusive, which is the whole point: it makes it
        impossible to accidentally read a horse's result in the race you
        are trying to predict.
        """
        cursor = self.conn.execute(
            """SELECT * FROM runs
               WHERE horse_id = ? AND date < ? AND scratched = 0
               ORDER BY date DESC""",
            (horse_id, before.isoformat()),
        )
        return [self._row_to_pastrun(row) for row in cursor.fetchall()]

    @staticmethod
    def _row_to_pastrun(row: sqlite3.Row) -> PastRun:
        return PastRun(
            date=_dt.date.fromisoformat(row["date"]),
            track_code=row["track_code"],
            distance_m=row["distance_m"] or 0,
            finish_position=row["finish_position"],
            field_size=row["field_size"],
            margin_l=row["margin_l"],
            barrier=row["barrier"],
            weight_kg=row["weight_kg"],
            jockey=row["jockey"],
            track_condition=row["track_condition"],
            class_level=row["class_level"],
            prize_money_total=row["prize_money"],
            starting_price=row["starting_price"],
            race_time_s=row["race_time_s"],
            last_600m_s=row["last_600m_s"],
        )

    def iter_races(
        self,
        start: Optional[_dt.date] = None,
        end: Optional[_dt.date] = None,
        with_history: bool = True,
    ) -> Iterator[Race]:
        """Rebuild Race objects in date order, each runner carrying only
        the form available before that race."""
        clauses, params = [], []
        if start:
            clauses.append("date >= ?")
            params.append(start.isoformat())
        if end:
            clauses.append("date <= ?")
            params.append(end.isoformat())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        cursor = self.conn.execute(
            f"SELECT * FROM runs {where} ORDER BY date, race_id, saddlecloth", params)

        current_id: Optional[str] = None
        rows: list[sqlite3.Row] = []
        for row in cursor:
            if current_id is not None and row["race_id"] != current_id:
                yield self._build_race(rows, with_history)
                rows = []
            current_id = row["race_id"]
            rows.append(row)
        if rows:
            yield self._build_race(rows, with_history)

    def _build_race(self, rows: list[sqlite3.Row], with_history: bool) -> Race:
        head = rows[0]
        date = _dt.date.fromisoformat(head["date"])
        track = get_track(head["track_code"]) or unknown_track(
            head["track_name"] or head["track_code"])
        race = Race(
            race_id=head["race_id"],
            track=track,
            date=date,
            race_number=head["race_number"] or 0,
            distance_m=head["distance_m"] or 0,
            track_condition=head["track_condition"],
            class_level=head["class_level"],
            prize_money=head["prize_money"],
        )
        for row in rows:
            runner = Runner(
                horse_id=row["horse_id"],
                name=row["horse_name"],
                number=row["saddlecloth"] or 0,
                barrier=row["barrier"],
                weight_kg=row["weight_kg"],
                jockey=row["jockey"],
                trainer=row["trainer"],
                age=row["age"],
                sex=row["sex"],
                sire=row["sire"],
                scratched=bool(row["scratched"]),
                fixed_win_odds=row["starting_price"],
                result_position=row["finish_position"],
                result_time_s=row["race_time_s"],
                result_margin_l=row["margin_l"],
            )
            if with_history:
                runner.history = self.horse_history(runner.horse_id, date)
            race.runners.append(runner)
        return race

    def stats(self) -> dict:
        row = self.conn.execute(
            """SELECT COUNT(*) AS runs,
                      COUNT(DISTINCT race_id) AS races,
                      COUNT(DISTINCT horse_id) AS horses,
                      MIN(date) AS first_date,
                      MAX(date) AS last_date
               FROM runs"""
        ).fetchone()
        return dict(row)
