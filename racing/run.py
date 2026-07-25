#!/usr/bin/env python3
"""Australian horse racing analysis tool.

Run `python run.py` with no arguments to see what it can do.

Commands, in the order you would normally use them:

    python run.py demo            See the whole thing work on simulated races
    python run.py fetch           Download real Australian racing data
    python run.py backtest        Test the model honestly on real past races
    python run.py train           Build a model from the downloaded data
    python run.py predict FILE    Analyse an upcoming race
    python run.py check           Check the maths and the setup

Gamble responsibly. National Gambling Helpline 1800 858 858 (free, 24/7).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BANNER = """
======================================================================
  Australian Racing Model
======================================================================
  This tool estimates probabilities and finds bets where the price on
  offer is better than those probabilities justify.

  It cannot pick 60% winners. Nothing can. The market favourite - the
  most-backed horse in every race - wins about a third of the time, and
  the market is well calibrated. Read HONEST_ANSWER.md before you use
  this to bet money.

  Gamble responsibly. Help: 1800 858 858 (free, confidential, 24/7)
  Self-exclude from all licensed AU wagering: betstop.gov.au
======================================================================
"""


def cmd_check(args) -> int:
    """Verify the installation and the arithmetic."""
    print(BANNER)
    print("Checking Python packages...")
    missing = []
    for package in ["pandas", "numpy", "sklearn", "lightgbm", "scipy", "requests"]:
        try:
            __import__(package)
            print(f"  ok    {package}")
        except ImportError:
            print(f"  MISSING  {package}")
            missing.append(package)
    if missing:
        print(f"\nInstall the missing packages with:\n  pip install -r requirements.txt")
        return 1

    print("\nChecking network access to the data sources...")
    import requests
    for name, url in [
        ("Betfair data hub", "https://betfair-datascientists.github.io/data/dataListing/"),
        ("Betfair SP files", "https://promo.betfair.com/betfairsp/prices"),
        ("Open-Meteo weather", "https://archive-api.open-meteo.com/v1/archive"
                               "?latitude=-37.79&longitude=144.91&start_date=2024-01-01"
                               "&end_date=2024-01-02&daily=precipitation_sum&timezone=Australia/Melbourne"),
    ]:
        try:
            response = requests.get(url, timeout=25)
            print(f"  {'ok   ' if response.status_code < 400 else 'FAIL '} {name} "
                  f"(HTTP {response.status_code})")
        except Exception as exc:
            print(f"  FAIL  {name}: {type(exc).__name__}")
            print(f"        If you are behind a firewall or VPN, that is the likely cause.")

    print("\nRunning the arithmetic tests...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q"],
        capture_output=True, text=True,
    )
    print("  " + (result.stdout.strip().splitlines() or ["no output"])[-1])
    return 0 if result.returncode == 0 else 1


def cmd_demo(args) -> int:
    """Run the whole pipeline on simulated races."""
    from ausrace import backtest, features, recommend
    from ausrace.sources.synthetic import SimConfig, simulate

    print(BANNER)
    print("DEMO MODE - these are SIMULATED races, not real ones.")
    print("It shows that the code works. It says nothing about real racing.\n")

    print(f"Simulating {args.races:,} races...")
    raw = simulate(SimConfig(n_races=args.races, n_horses=max(args.races // 2, 500), seed=7))
    print(f"  {raw['race_id'].nunique():,} races, {len(raw):,} runners, "
          f"average field {raw.groupby('race_id').size().mean():.1f}")

    print("\nBuilding features (this is the slow part)...")
    feat = features.build_features(raw)
    print(f"  {feat.shape[0]:,} rows x {feat.shape[1]} columns")

    leaks = features.assert_no_leakage(feat)
    print(f"\nLeakage check: {'CLEAN' if not leaks else 'PROBLEMS FOUND'}")
    for leak in leaks:
        print(f"  ! {leak}")

    print("\nWalk-forward backtest...")
    result = backtest.walk_forward(
        feat, initial_train_days=args.train_days, test_days=args.test_days,
        min_train_races=args.min_races, verbose=True,
    )
    print("\n" + result.summary())

    print("\nCALIBRATION (does 20% actually win 20%?)")
    print(result.calibration.to_string(index=False))

    if not result.predictions.empty:
        print("\nBANKROLL SIMULATION ($1,000 start, quarter Kelly)")
        path = backtest.simulate_bankroll(result.predictions, bankroll=1000)
        summary = backtest.bankroll_summary(path)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key:<24} {value:,.4f}")
            else:
                print(f"  {key:<24} {value}")

    # Show the per-race output on the most recent race.
    last_race_id = result.predictions["race_id"].iloc[-1]
    last = result.predictions[result.predictions["race_id"] == last_race_id]
    print("\n\nEXAMPLE RACE OUTPUT")
    print(recommend.format_race_report(
        recommend.recommend_for_race(last, recommend.Settings(bankroll=1000)),
        race_name=f"{last['track'].iloc[0]} (simulated)",
    ))
    return 0


def cmd_fetch(args) -> int:
    """Download real Australian racing data."""
    from ausrace.sources import betfair_hub, weather

    print(BANNER)
    end = date.today() if args.to is None else date.fromisoformat(args.to)
    start = (end - timedelta(days=args.days)) if args.since is None \
        else date.fromisoformat(args.since)

    print(f"Downloading Betfair ANZ thoroughbred data, {start} to {end}.")
    print("Source: https://betfair-datascientists.github.io/data/dataListing/")
    print("This is free, public, and published by Betfair for exactly this purpose.\n")

    raw = betfair_hub.fetch_hub_range(start, end, cache_dir=args.cache)
    frame = betfair_hub.parse_hub(raw, country="AU")
    print(f"\n  {frame['race_id'].nunique():,} races, {len(frame):,} runners")
    print(f"  {frame['track'].nunique()} tracks")
    print(f"  date range {frame['race_datetime'].min()} .. {frame['race_datetime'].max()}")

    if args.weather:
        print("\nAttaching weather from Open-Meteo (free, non-commercial use)...")
        print("Betfair's files carry no track condition, so rainfall is used as a proxy.")
        frame = weather.attach_weather(frame)
        estimated = frame.get("condition_is_estimate")
        if estimated is not None:
            print(f"  Track condition ESTIMATED for {int(estimated.sum()):,} runners.")
            print("  These are estimates from rainfall, not official ratings.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out) if out.suffix == ".parquet" else frame.to_pickle(out)
    print(f"\nSaved to {out}")
    print(f"\nNext: python run.py backtest --data {out}")
    return 0


def cmd_backtest(args) -> int:
    """Test the model on real past races, honestly."""
    from ausrace import backtest, features, schema

    print(BANNER)
    frame = _load(args.data)
    print(f"Loaded {frame['race_id'].nunique():,} races, {len(frame):,} runners.\n")

    problems = schema.validate(schema.conform(frame))
    if problems:
        print("DATA QUALITY WARNINGS:")
        for problem in problems:
            print(f"  ! {problem}")
        print()

    print("Building features...")
    feat = features.build_features(frame)

    leaks = features.assert_no_leakage(feat)
    print(f"Leakage check: {'CLEAN' if not leaks else 'PROBLEMS FOUND'}")
    for leak in leaks:
        print(f"  ! {leak}")
    print()

    result = backtest.walk_forward(
        feat,
        initial_train_days=args.train_days,
        test_days=args.test_days,
        min_train_races=args.min_races,
        use_market=not args.no_market,
        min_edge=args.min_edge,
        commission=args.commission,
        verbose=True,
    )
    print("\n" + result.summary())

    print("\nCALIBRATION")
    print(result.calibration.to_string(index=False))
    if not result.odds_bands.empty:
        print("\nROI BY PRICE BRACKET")
        print(result.odds_bands.to_string(index=False))

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    result.folds.to_csv(reports / "backtest_folds.csv", index=False)
    result.calibration.to_csv(reports / "backtest_calibration.csv", index=False)
    with open(reports / "backtest_report.json", "w") as fh:
        json.dump({"report": result.report, "baseline": result.baseline,
                   "warnings": result.warnings}, fh, indent=2, default=str)
    print(f"\nDetailed results written to {reports}/")
    return 0


def cmd_train(args) -> int:
    """Train a model on all available data and save it."""
    from ausrace import features
    from ausrace.model import RaceModel

    print(BANNER)
    frame = _load(args.data)
    feat = features.build_features(frame)
    resolved = feat[feat["won"].notna()]

    # Hold out the most recent slice by date for the blend and early stopping.
    split = resolved["race_datetime"].quantile(0.85)
    train = resolved[resolved["race_datetime"] < split]
    valid = resolved[resolved["race_datetime"] >= split]

    print(f"Training on {train['race_id'].nunique():,} races "
          f"(validating on {valid['race_id'].nunique():,} more recent ones)...")
    model = RaceModel(use_market=not args.no_market).fit(train, valid=valid)
    model.save(args.out)

    print(f"\nSaved to {args.out}")
    print(f"  market blend weights: model {model.alpha:.3f}, market {model.beta:.3f}")
    if model.beta > model.alpha:
        print("  The market weight is higher than the model weight. That is the "
              "normal, healthy result: the odds know more than the form does.")
    print("\nTop features by information gain:")
    print(model.importance(15).to_string(index=False))
    return 0


def cmd_predict(args) -> int:
    """Analyse an upcoming race from a CSV."""
    from ausrace import features, recommend
    from ausrace.model import RaceModel

    print(BANNER)
    upcoming = _load(args.race)

    required = ["race_id", "race_datetime", "track", "runner_id", "horse"]
    missing = [c for c in required if c not in upcoming.columns]
    if missing:
        raise SystemExit(
            f"\nYour race file is missing these column(s): {', '.join(missing)}\n"
            f"It needs at least: {', '.join(required)}\n"
            f"It has: {', '.join(map(str, upcoming.columns))}\n"
            f"See the example in README.md."
        )
    if upcoming.empty:
        raise SystemExit("\nYour race file has no runners in it.")

    print(f"Loaded {len(upcoming)} runners.\n")

    history = _load(args.history) if args.history else None
    if history is not None:
        combined = pd.concat([history, upcoming], ignore_index=True)
    else:
        print("No history supplied - form features will be empty and the model "
              "will lean almost entirely on the market price.\n")
        combined = upcoming

    feat = features.build_features(combined)
    target = feat[feat["race_id"].isin(upcoming["race_id"].astype(str).unique())]

    model = RaceModel.load(args.model)
    predicted = model.predict(target)

    settings = recommend.Settings(
        bankroll=args.bankroll,
        kelly_fraction=args.kelly,
        min_edge_win=args.min_edge,
    )
    for race_id, race in predicted.groupby("race_id"):
        result = recommend.recommend_for_race(race, settings)
        print(recommend.format_race_report(
            result, race_name=f"{race['track'].iloc[0]} race {race_id}"))
        print()
    return 0


def _load(path: str | Path) -> pd.DataFrame:
    """Load a data file, with error messages a non-programmer can act on."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"\nCould not find the file: {path}\n"
            f"Check the spelling and the folder you are in.\n"
            f"If you have not downloaded data yet, run:  python run.py fetch"
        )
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        return pd.read_pickle(path)
    except Exception as exc:
        raise SystemExit(
            f"\nCould not read {path}.\n"
            f"The file may be corrupted, empty, or not the format the name "
            f"suggests.\nDetail: {type(exc).__name__}: {exc}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Australian horse racing analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("check", help="verify installation, network and maths")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("demo", help="run the whole pipeline on simulated races")
    p.add_argument("--races", type=int, default=4000)
    p.add_argument("--train-days", type=int, default=700)
    p.add_argument("--test-days", type=int, default=150)
    p.add_argument("--min-races", type=int, default=1000)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("fetch", help="download real Australian racing data")
    p.add_argument("--days", type=int, default=1460, help="how far back to go")
    p.add_argument("--since", default=None, help="start date YYYY-MM-DD")
    p.add_argument("--to", default=None, help="end date YYYY-MM-DD")
    p.add_argument("--out", default="data/raw/au_racing.pkl")
    p.add_argument("--cache", default="data/raw/betfair_hub")
    p.add_argument("--weather", action="store_true",
                   help="attach weather and estimate track condition")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("backtest", help="walk-forward test on real past races")
    p.add_argument("--data", default="data/raw/au_racing.pkl")
    p.add_argument("--train-days", type=int, default=730)
    p.add_argument("--test-days", type=int, default=90)
    p.add_argument("--min-races", type=int, default=2000)
    p.add_argument("--min-edge", type=float, default=0.05)
    p.add_argument("--commission", type=float, default=0.0,
                   help="0 for a bookmaker; 0.08-0.10 for Betfair")
    p.add_argument("--no-market", action="store_true",
                   help="build a pure form model that ignores the odds")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("train", help="train and save a model")
    p.add_argument("--data", default="data/raw/au_racing.pkl")
    p.add_argument("--out", default="models/model.pkl")
    p.add_argument("--no-market", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="analyse an upcoming race")
    p.add_argument("race", help="CSV of the runners in the upcoming race")
    p.add_argument("--model", default="models/model.pkl")
    p.add_argument("--history", default="data/raw/au_racing.pkl")
    p.add_argument("--bankroll", type=float, default=1000.0)
    p.add_argument("--kelly", type=float, default=0.25)
    p.add_argument("--min-edge", type=float, default=0.05)
    p.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    if not getattr(args, "func", None):
        print(BANNER)
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
