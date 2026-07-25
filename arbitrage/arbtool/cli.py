"""Command line interface.

Designed to be usable by someone who has never written a line of code. Every
command explains what it is about to do, and every number it prints is labelled
in words rather than jargon.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .agents import Orchestrator
from .beta import reviewer_availability
from .config import Config
from .costs import CostModel
from .data.providers import DataError, build_provider, today_iso
from .pairs import build_pair_series, ratio_report
from .report import write_json, write_report
from .universe import UNIVERSE, VENUES, get_pair, synthetic_universe

BANNER = """
Cross-market arbitrage research tool
------------------------------------
This tool researches, tests and PAPER trades price gaps between two stock
exchange listings of the same company. It does not place real orders.
"""


def _echo(line: str) -> None:
    print(f"  {line}")


def _load_config(args) -> Config:
    config = Config.load(args.config) if args.config else Config()
    if getattr(args, "provider", None):
        config.provider = args.provider
    if getattr(args, "start", None):
        config.start = args.start
    if getattr(args, "end", None):
        config.end = args.end
    if getattr(args, "pairs", None):
        config.pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    if getattr(args, "capital", None):
        config.risk.capital = args.capital
    if getattr(args, "win_rate", None):
        config.gate.min_win_rate = args.win_rate
    if getattr(args, "long_only", False):
        config.risk.allow_short = False
    if getattr(args, "data_dir", None):
        config.data_dir = args.data_dir
    return config


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_explain(args) -> int:
    print(BANNER)
    print("""WHAT THIS ACTUALLY DOES

Some companies are listed on two stock exchanges at once. BHP trades in Sydney
and in New York. Shell trades in London and in New York. Both listings are
claims on the same company, so once you convert the currencies and adjust for
the fact that one "share" on one exchange can represent several on the other,
the two prices must track each other closely.

Sometimes they drift apart. That gap is the opportunity: you buy the cheap
listing, sell the expensive one, and wait for them to come back together. You
do not care whether the company goes up or down — only whether the gap closes.

WHY IT IS HARDER THAN IT SOUNDS

  * Every trade pays commission, twice, on two legs.
  * You cross the bid/ask spread on both legs.
  * You convert currency, and your broker takes a cut.
  * The UK charges 0.5% stamp duty on share purchases.
  * Selling short means borrowing the shares, which costs money every day.
  * Sydney and New York are never open at the same time, so you cannot trade
    both legs at the same instant.

Add those up and a gap of 0.4% can easily be a losing trade. This tool models
every one of those costs before it calls anything an opportunity.

ABOUT YOUR 60% WIN RATE REQUIREMENT

Nobody can promise a win rate in advance. What this tool does instead is
measure it honestly and refuse to trade unless it clears your bar:

  1. Settings are chosen using old data only.
  2. They are then applied, untouched, to later data the model never saw.
  3. Only those unseen trades are scored.
  4. The score used is not the raw win rate — it is the lower end of a 95%
     confidence interval, so a lucky run of 7 wins from 10 cannot pass.
  5. It must also make money per trade, have a decent profit factor, and stay
     inside your drawdown limit.

And one warning worth repeating: a high win rate is not the same as making
money. This kind of strategy wins small, often, and loses big, rarely. That is
why the gate checks four other things beside the win rate.

WHAT IT WILL NEVER DO

It will not place a real order. Connecting real money is a decision you have to
make deliberately, with your own broker, after reading RISK.md.

COMMANDS

  python3 -m arbtool demo       run everything offline on generated data
  python3 -m arbtool check      test whether real market data is reachable
  python3 -m arbtool pairs      list the company pairs it knows about
  python3 -m arbtool costs BHP  show what one BHP trade actually costs
  python3 -m arbtool research   the real thing: research, test, report
  python3 -m arbtool signals    what the approved strategies say to do today
""")
    return 0


def cmd_check(args) -> int:
    print(BANNER)
    config = _load_config(args)
    print(f"Checking whether market data is reachable using '{config.provider}'...\n")
    provider = build_provider(config.provider, config.data_dir, config.seed)
    end = config.end or today_iso()

    ok = 0
    for pair in list(UNIVERSE.values())[:3]:
        for leg in (pair.a, pair.b):
            try:
                bars = provider.fetch_leg(leg, config.start, end)
                print(f"  OK    {leg.symbol:<12} {len(bars)} days "
                      f"({bars[0].date} to {bars[-1].date})")
                ok += 1
            except DataError as exc:
                print(f"  FAIL  {leg.symbol:<12} {exc}")

    print()
    if ok == 0:
        print("No market data could be downloaded. Common causes:\n"
              "  * no internet connection\n"
              "  * a firewall or corporate proxy blocking the data sites\n"
              "  * the free data source is temporarily down\n\n"
              "You can still exercise the whole system offline:\n"
              "  python3 -m arbtool demo")
        return 1

    summary = provider.cache_summary()
    print(f"Data is reachable. Local cache now holds {summary['bars']:,} daily bars "
          f"across {summary['symbols']} symbols.")
    print(f"\n{reviewer_availability()}")
    return 0


def cmd_pairs(args) -> int:
    print(BANNER)
    universe = synthetic_universe() if args.provider == "synthetic" else UNIVERSE
    print(f"{len(universe)} cross-listed pairs are configured.\n")
    for pair in universe.values():
        overlap = "sessions overlap" if pair.sessions_overlap else "NO session overlap"
        short = "both shortable" if pair.both_shortable else "one leg hard to short"
        print(f"  {pair.describe()}")
        print(f"      {overlap}; {short}")
        if pair.notes:
            print(f"      note: {pair.notes}")
    print("\nVenue cost assumptions (basis points, one side):")
    for code, venue in VENUES.items():
        if code == "SYNTH":
            continue
        extras = []
        if venue.buy_tax_bps:
            extras.append(f"buy tax {venue.buy_tax_bps:.0f}")
        if venue.sell_tax_bps:
            extras.append(f"sell tax {venue.sell_tax_bps:.0f}")
        if not venue.shortable:
            extras.append("not shortable")
        print(f"  {code:<9} {venue.currency}  commission {venue.commission_bps:.1f}, "
              f"half-spread {venue.half_spread_bps:.1f}"
              + (f", {', '.join(extras)}" if extras else ""))
    print("\nThese are estimates. Replace them with your broker's real numbers in "
          "arbtool/universe.py before trusting any result.")
    return 0


def cmd_costs(args) -> int:
    print(BANNER)
    config = _load_config(args)
    pair = get_pair(args.pair.upper())
    model = CostModel()
    for hold in (5, 10, 20):
        for line in model.explain(pair, hold, config.base_currency,
                                  config.risk.allow_short):
            print("  " + line)
        print()
    print("  Read that bottom line carefully: it is how far apart the two prices\n"
          "  must be before the trade is worth doing at all. Most of the time they\n"
          "  are closer together than that, which is why most days there is no trade.")
    return 0


def cmd_ratio(args) -> int:
    print(BANNER)
    config = _load_config(args)
    provider = build_provider(config.provider, config.data_dir, config.seed)
    end = config.end or today_iso()
    pairs = [get_pair(p.upper()) for p in args.pairs.split(",")] if args.pairs \
        else list(UNIVERSE.values())

    print("Checking each configured share ratio against what the prices imply.\n")
    for pair in pairs:
        try:
            series = build_pair_series(pair, provider, config.start, end,
                                       config.base_currency)
        except DataError as exc:
            print(f"  {pair.pair_id:<6} could not check: {exc}")
            continue
        configured, empirical, verdict = ratio_report(series)
        print(f"  {pair.pair_id:<6} configured {configured:<8.4g} "
              f"data implies {empirical:<8.4g}  {verdict}")
    return 0


def cmd_research(args) -> int:
    print(BANNER)
    config = _load_config(args)
    problems = config.validate()
    if problems:
        print("Configuration problems:")
        for problem in problems:
            print(f"  - {problem}")
        return 2

    print(f"Researching {len(config.pairs) or 'all'} pair(s) from {config.start} "
          f"using '{config.provider}' data.")
    print(f"Requirement: win rate of at least {config.gate.min_win_rate:.0%} on "
          f"unseen data, measured at the 95% confidence lower bound.\n")

    orchestrator = Orchestrator(config, echo=_echo if args.verbose else None,
                                use_llm_reviewer=args.ai_review)
    try:
        outcome = orchestrator.run()
    except DataError as exc:
        print(f"\nCould not run: {exc}")
        return 1

    print("\n" + "=" * 72)
    print(outcome.gate.report())
    print("=" * 72)

    if outcome.beta:
        print("\nAdversarial beta tests:")
        for pair_id, report in outcome.beta.items():
            print(f"\n  {pair_id}")
            for line in report.summary().splitlines():
                print(f"    {line}")

    print("\nVERDICT: ", end="")
    if outcome.deployable:
        print("cleared for PAPER trading.")
        print("  Nothing here has touched real money, and this tool cannot make it.")
        print("  The honest next step is to run it forward on live prices for a few")
        print("  months and check the real out-of-sample record against this one.")
    else:
        print("NOT fit to trade.")
        print("  This is the normal, expected outcome. Most ideas do not survive an")
        print("  honest test — which is exactly why running one is worth the effort.")

    if outcome.used_synthetic:
        print("\n  Reminder: this run used GENERATED data. It proves the system works.")
        print("  It says nothing about real markets.")

    report_path = args.report or os.path.join(config.data_dir, "report.html")
    os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".", exist_ok=True)
    write_report(outcome, report_path)
    json_path = os.path.splitext(report_path)[0] + ".json"
    write_json(outcome, json_path)
    print(f"\nFull report written to: {report_path}")
    print(f"Machine-readable copy:  {json_path}")
    print("Open the HTML file in any web browser.")
    return 0 if outcome.deployable else 3


def cmd_demo(args) -> int:
    print(BANNER)
    print("Running the whole system offline on generated markets.\n"
          "No internet needed. No real company is involved. This exists so you can\n"
          "watch every agent do its job before you trust it with real data.\n")
    args.provider = "synthetic"
    args.verbose = True
    if not getattr(args, "start", None):
        args.start = "2015-01-01"
    return cmd_research(args)


def cmd_signals(args) -> int:
    print(BANNER)
    config = _load_config(args)
    orchestrator = Orchestrator(config, echo=None, use_llm_reviewer=False)
    try:
        outcome = orchestrator.run()
    except DataError as exc:
        print(f"Could not run: {exc}")
        return 1

    if not outcome.deployable:
        print("No strategy is currently approved, so there are no signals.")
        print("Run 'research' to see exactly which check failed.")
        return 3
    if not outcome.signals:
        print("No trade today: every approved pair is trading within its normal range.")
        print("That is the usual answer. Waiting is most of this strategy.")
        return 0

    print("PAPER trade signals for today:\n")
    for s in outcome.signals:
        print(f"  {s['pair']}  gap {s['gap_bps']:+.0f} basis points "
              f"({s['z']:+.2f} standard deviations from normal)")
        for order in s["orders"]:
            print(f"      {order['side']:<11} {order['quantity']:>10.2f} "
                  f"of leg {order['leg']} at {order['price']:.4f}  ({order['note']})")
    print("\nThese are paper orders. Nothing was sent to a broker.")
    return 0


def cmd_config(args) -> int:
    config = Config()
    config.save(args.out)
    print(f"Wrote a default configuration to {args.out}.")
    print("Edit it in any text editor, then run:")
    print(f"  python3 -m arbtool research --config {args.out}")
    return 0


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arbtool",
        description="Research, test and paper-trade cross-listing arbitrage.",
    )
    sub = parser.add_subparsers(dest="command")

    def common(p, with_data: bool = True):
        p.add_argument("--config", help="path to a JSON configuration file")
        if with_data:
            p.add_argument("--provider", choices=["auto", "stooq", "yahoo", "synthetic"],
                           help="where to get prices from")
            p.add_argument("--start", help="first date to study, YYYY-MM-DD")
            p.add_argument("--end", help="last date to study, YYYY-MM-DD")
            p.add_argument("--pairs", help="comma-separated pair ids, e.g. BHP,SHEL")
            p.add_argument("--data-dir", help="where to keep the price cache")
        p.add_argument("--capital", type=float, help="account size to simulate")
        p.add_argument("--win-rate", type=float,
                       help="required win rate as a fraction, e.g. 0.60")
        p.add_argument("--long-only", action="store_true",
                       help="never sell short (buys the cheap leg only — this is a "
                            "directional bet, not an arbitrage)")

    p = sub.add_parser("explain", help="what this does, in plain English")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("check", help="test whether market data is reachable")
    common(p)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("pairs", help="list the cross-listed pairs it knows")
    p.add_argument("--provider", default="auto",
                   choices=["auto", "stooq", "yahoo", "synthetic"])
    p.set_defaults(func=cmd_pairs)

    p = sub.add_parser("costs", help="show what one trade really costs")
    p.add_argument("pair", help="pair id, e.g. BHP")
    common(p, with_data=False)
    p.set_defaults(func=cmd_costs)

    p = sub.add_parser("ratio", help="verify configured share ratios against the data")
    common(p)
    p.set_defaults(func=cmd_ratio)

    p = sub.add_parser("research", help="run the full research pipeline")
    common(p)
    p.add_argument("--report", help="where to write the HTML report")
    p.add_argument("--verbose", action="store_true",
                   help="print every agent message as it happens")
    p.add_argument("--ai-review", action="store_true",
                   help="also ask an independent Claude reviewer to critique the "
                        "result (needs the anthropic package and an API key)")
    p.set_defaults(func=cmd_research)

    p = sub.add_parser("demo", help="run the whole system offline on generated data")
    common(p)
    p.add_argument("--report", help="where to write the HTML report")
    p.add_argument("--ai-review", action="store_true")
    p.set_defaults(func=cmd_demo, verbose=True)

    p = sub.add_parser("signals", help="what the approved strategies say to do today")
    common(p)
    p.set_defaults(func=cmd_signals)

    p = sub.add_parser("init-config", help="write a configuration file you can edit")
    p.add_argument("--out", default="arbtool.config.json")
    p.set_defaults(func=cmd_config)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        print("\nNew here? Start with:  python3 -m arbtool explain")
        return 0
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
