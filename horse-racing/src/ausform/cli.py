"""Command line interface.

    ausform demo          run the whole pipeline on simulated data
    ausform backtest      walk-forward evaluation on stored races
    ausform import-csv    load historical form into the local store
    ausform fetch-betfair download free Betfair Australian results
    ausform train         fit a model and save it
    ausform analyse       assess a race and suggest bets
    ausform serve         start the local web dashboard
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

from . import __version__

log = logging.getLogger("ausform")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


BANNER = """
ausform -- Australian thoroughbred race analysis

A reminder before you use any of this: the goal is not to pick more
winners than the market. It is to price a minority of runners slightly
better than the market and bet only those. Most races should produce no
bet at all.
"""


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------

def cmd_demo(args: argparse.Namespace) -> int:
    from .analyse import analyse_race
    from .backtest import WalkForwardBacktest, WalkForwardConfig
    from .data.simulator import SeasonSimulator
    from .features import RollingContext, build_features

    print(BANNER)
    print(f"Simulating {args.days} days of racing...")
    simulator = SeasonSimulator(seed=args.seed, n_horses=args.horses)
    races = simulator.simulate_season(
        _dt.date(2022, 1, 1), days=args.days,
        meetings_per_day=3, races_per_meeting=8)
    print(f"  {len(races):,} races, "
          f"{sum(r.field_size for r in races):,} runners\n")

    print("Running walk-forward backtest (this is the honest kind)...")
    config = WalkForwardConfig(
        train_days=args.train_days, test_days=60, min_train_races=500)
    result = WalkForwardBacktest(config).run(races)
    report = result.report(commission=0.08)
    _print_report(report)

    print("\n" + "=" * 68)
    print("SAMPLE RACE ANALYSIS")
    print("=" * 68)

    context = RollingContext()
    split = int(len(races) * 0.7)
    for race in races[:split]:
        build_features(race, context)
        context.observe(race)

    from .model import TwoStageModel, observations_from_featuresets
    feature_sets = []
    context2 = RollingContext()
    for race in races[:split]:
        feature_sets.append(build_features(race, context2))
        context2.observe(race)
    observations = observations_from_featuresets(feature_sets)
    odds_by_race = {
        r.race_id: [x.fixed_win_odds for x in r.active_runners]
        for r in races[:split]
    }
    model = TwoStageModel.create()
    model.fit(observations, odds_by_race, feature_names=feature_sets[0].names)

    target = races[split + 5]
    analysis = analyse_race(target, model, context)
    _print_analysis(analysis)
    return 0


def _print_report(report: dict) -> None:
    model = report["model"]
    base = report["baselines"]
    print(f"\n  Races evaluated : {report['races']:,} "
          f"over {report.get('windows', 0)} walk-forward windows")
    print("\n  PROBABILITY QUALITY  (log-loss, lower is better)")
    print(f"    guessing at random        : {base['uniform_log_loss']:.4f}")
    if "market_log_loss" in base:
        print(f"    the betting market        : {base['market_log_loss']:.4f}")
    print(f"    this model                : {model['log_loss']:.4f}")
    if "log_loss_improvement_vs_market_pct" in model:
        improvement = model["log_loss_improvement_vs_market_pct"]
        print(f"    -> model beats market by  : {improvement:+.2f}%")
        if improvement > 5:
            print("       WARNING: beating the market by more than a few percent "
                  "almost always means leakage, not skill.")

    print(f"\n  Top-1 accuracy   : model {model['top1_accuracy']:.1%}", end="")
    if "market_top1_accuracy" in base:
        print(f"   market {base['market_top1_accuracy']:.1%}")
    else:
        print()
    print("    (~33% is roughly the ceiling: that is how often the favourite")
    print("     wins, and most of the rest is irreducible randomness)")

    if "blend" in report:
        blend = report["blend"]
        print(f"\n  Blend weights    : model {blend['mean_alpha_model_weight']:.3f}, "
              f"market {blend['mean_beta_market_weight']:.3f}")

    print("\n  FLAT-STAKE YIELD")
    for row in report["yield"]:
        flag = "significant" if row["statistically_significant"] else "NOT significant"
        print(f"    edge > {row['edge_threshold']:>4.0%} : "
              f"{row['bets']:>6,} bets, strike {row['strike_rate']:>5.1%}, "
              f"yield {row['yield_pct']:>+7.1%}  [{flag}]")
        if not row["statistically_significant"] and row.get("bets_needed_for_significance"):
            print(f"{'':18}would need ~{row['bets_needed_for_significance']:,} "
                  f"bets to prove")


def _print_analysis(analysis) -> None:
    print("\n" + analysis.summary() + "\n")
    header = (f"{'#':>3} {'Horse':<24} {'Bar':>3} {'Model':>7} {'Market':>7} "
              f"{'Fair$':>7} {'Odds':>7} {'Edge':>7}")
    print(header)
    print("-" * len(header))
    for a in analysis.assessments[:12]:
        market = f"{a.market_win_prob:.1%}" if a.market_win_prob else "  -  "
        odds = f"{a.win_odds:.2f}" if a.win_odds else "  -  "
        edge = f"{a.win_edge:+.1%}" if a.win_edge is not None else "  -  "
        print(f"{a.number:>3} {a.name[:24]:<24} {a.barrier or 0:>3} "
              f"{a.model_win_prob:>7.1%} {market:>7} "
              f"{a.fair_win_odds:>7.2f} {odds:>7} {edge:>7}")

    top = analysis.top_pick
    if top:
        print(f"\n  Top rated: #{top.number} {top.name}")
        for note in top.notes[:5]:
            print(f"    - {note}")

    print("\n  RECOMMENDED BETS")
    if analysis.stakes:
        for stake in analysis.stakes:
            print(f"    {stake.bet_type.upper():>5} {stake.selection} "
                  f"${stake.amount:.2f} @ ${stake.odds:.2f} "
                  f"(edge {stake.edge:+.1%})")
    else:
        print("    No bet.")
    for warning in analysis.warnings:
        print(f"    ! {warning}")


# --------------------------------------------------------------------------
# data commands
# --------------------------------------------------------------------------

def cmd_import_csv(args: argparse.Namespace) -> int:
    from .data.store import RaceStore
    with RaceStore(args.db) as store:
        imported, skipped = store.import_csv(args.path, strict=args.strict)
        print(f"Imported {imported:,} runner records ({skipped:,} skipped).")
        print(json.dumps(store.stats(), indent=2, default=str))
    return 0


def cmd_fetch_betfair(args: argparse.Namespace) -> int:
    from .data.betfair import BetfairHistorical
    from .data.store import RaceStore

    client = BetfairHistorical()
    start = _dt.date.fromisoformat(args.start)
    end = _dt.date.fromisoformat(args.end)

    rows = list(client.fetch_range(start, end))
    if not rows:
        print("No data returned. Betfair publishes one file per day; "
              "check the dates and your network access.")
        return 1

    races = client.to_races(rows)
    print(f"Fetched {len(rows):,} runner records across {len(races):,} races.")

    with RaceStore(args.db) as store:
        stored = store.import_races(races)
        print(f"Stored {stored:,} records in {args.db}.")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from .backtest import WalkForwardBacktest, WalkForwardConfig
    from .data.store import RaceStore

    with RaceStore(args.db) as store:
        races = list(store.iter_races())
    if not races:
        print(f"No races in {args.db}. Import some first.")
        return 1

    print(f"Backtesting {len(races):,} races from {args.db}...")
    config = WalkForwardConfig(train_days=args.train_days,
                              test_days=args.test_days,
                              min_train_races=args.min_races)
    result = WalkForwardBacktest(config).run(
        races, progress=lambda m: print("  " + m))
    report = result.report(commission=args.commission)
    _print_report(report)

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nFull report written to {args.json}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .data.store import RaceStore
    from .features import RollingContext, build_features
    from .model import TwoStageModel, observations_from_featuresets

    with RaceStore(args.db) as store:
        races = sorted(store.iter_races(), key=lambda r: (r.date, r.race_id))
    if not races:
        print(f"No races in {args.db}.")
        return 1

    context = RollingContext()
    feature_sets = []
    for race in races:
        feature_sets.append(build_features(race, context))
        context.observe(race)

    observations = observations_from_featuresets(feature_sets)
    odds_by_race = {
        r.race_id: [x.fixed_win_odds or x.tote_win_odds for x in r.active_runners]
        for r in races
    }
    model = TwoStageModel.create(l2=args.l2)
    model.fit(observations, odds_by_race, feature_names=feature_sets[0].names)

    print(f"Fitted on {len(observations):,} races.")
    print(f"  {model.blend.describe()}")
    print("\n  Strongest coefficients:")
    for name, value in model.fundamental.top_coefficients(15):
        print(f"    {name:<38} {value:+.4f}")

    Path(args.out).write_bytes(pickle.dumps({"model": model, "context": context}))
    print(f"\nSaved to {args.out}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Run: pip install 'ausform[web]'")
        return 1
    from .web.app import create_app

    app = create_app(db_path=args.db, model_path=args.model)
    print(f"Dashboard on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ausform",
        description="Australian thoroughbred race analysis and bet evaluation.")
    parser.add_argument("--version", action="version", version=f"ausform {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the full pipeline on simulated data")
    demo.add_argument("--days", type=int, default=540)
    demo.add_argument("--horses", type=int, default=9000)
    demo.add_argument("--seed", type=int, default=11)
    demo.add_argument("--train-days", type=int, default=300)
    demo.set_defaults(func=cmd_demo)

    imp = sub.add_parser("import-csv", help="load historical form from CSV")
    imp.add_argument("path")
    imp.add_argument("--db", default="ausform.db")
    imp.add_argument("--strict", action="store_true")
    imp.set_defaults(func=cmd_import_csv)

    bf = sub.add_parser("fetch-betfair",
                        help="download free Betfair Australian race results")
    bf.add_argument("--start", required=True, help="YYYY-MM-DD")
    bf.add_argument("--end", required=True, help="YYYY-MM-DD")
    bf.add_argument("--db", default="ausform.db")
    bf.set_defaults(func=cmd_fetch_betfair)

    back = sub.add_parser("backtest", help="walk-forward evaluation")
    back.add_argument("--db", default="ausform.db")
    back.add_argument("--train-days", type=int, default=365)
    back.add_argument("--test-days", type=int, default=30)
    back.add_argument("--min-races", type=int, default=500)
    back.add_argument("--commission", type=float, default=0.08)
    back.add_argument("--json", help="write the full report here")
    back.set_defaults(func=cmd_backtest)

    train = sub.add_parser("train", help="fit a model and save it")
    train.add_argument("--db", default="ausform.db")
    train.add_argument("--out", default="model.pkl")
    train.add_argument("--l2", type=float, default=1.0)
    train.set_defaults(func=cmd_train)

    serve = sub.add_parser("serve", help="start the web dashboard")
    serve.add_argument("--db", default="ausform.db")
    serve.add_argument("--model", default="model.pkl")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
