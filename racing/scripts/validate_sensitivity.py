"""Sensitivity validation: does the tool find an edge only when one exists?

A betting tool that always reports a profit is broken. So is one that always
reports a loss. This script runs the whole pipeline against simulated worlds
where we *know* the right answer, and checks that the verdict tracks reality:

  1. Efficient market, bookmaker margin   -> should find NO edge
  2. Efficient market, exchange margin    -> should find little or no edge
  3. Weak market, bookmaker margin        -> should find a REAL edge
  4. Weak market, exchange margin         -> should find a LARGE edge

Because these are simulated worlds, the ground truth is known. That is the
whole point - it is the only way to test a betting model's verdict rather than
just its arithmetic.

Nothing here says anything about real racing. It tests the tool, not the sport.
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ausrace import backtest, features  # noqa: E402
from ausrace.sources.synthetic import SimConfig, simulate  # noqa: E402

SCENARIOS = [
    ("Efficient market, bookmaker book (116%)", 0.92, 1.16, "no edge"),
    ("Efficient market, exchange book (102%)", 0.92, 1.02, "little or no edge"),
    ("Weak market, bookmaker book (116%)", 0.70, 1.16, "real edge"),
    ("Weak market, exchange book (102%)", 0.70, 1.02, "large edge"),
]


def run(n_races: int = 5000, seed: int = 11) -> pd.DataFrame:
    rows = []
    for name, skill, book, expectation in SCENARIOS:
        started = time.time()
        print(f"\n--- {name}  (expecting: {expectation})")

        raw = simulate(SimConfig(
            n_races=n_races, n_horses=2500, seed=seed,
            market_skill=skill, overround=book,
        ))
        feat = features.build_features(raw)

        leaks = features.assert_no_leakage(feat)
        if leaks:
            print(f"    LEAKAGE DETECTED: {leaks}")

        result = backtest.walk_forward(
            feat, initial_train_days=600, test_days=180,
            min_train_races=1000, verbose=False,
        )
        report = result.report

        rows.append({
            "scenario": name,
            "expected": expectation,
            "market_skill": skill,
            "book": book,
            "log_loss_model": round(report["log_loss"], 4),
            "log_loss_market": round(report.get("log_loss_market_only", float("nan")), 4),
            "beats_market": report.get("beats_market"),
            "top1": f"{report['top1_strike_rate']:.1%}",
            "fav_strike": f"{result.baseline.get('favourite_strike_rate', float('nan')):.1%}",
            "n_bets": report["roi_n_bets"],
            "roi": f"{report['roi']:+.1%}" if pd.notna(report.get("roi")) else "n/a",
            "p_value": round(report.get("roi_p_value", float("nan")), 4),
            "leakage": "none" if not leaks else "DETECTED",
            "seconds": round(time.time() - started),
        })
        print(f"    log loss {report['log_loss']:.4f} vs market "
              f"{report.get('log_loss_market_only', float('nan')):.4f} | "
              f"{report['roi_n_bets']} bets | ROI "
              f"{report['roi']:+.1%}" if pd.notna(report.get("roi")) else "no bets")
        for warning in result.warnings:
            print(f"    ! {warning}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    table = run()
    print("\n" + "=" * 110)
    print("SENSITIVITY VALIDATION (simulated worlds - says nothing about real racing)")
    print("=" * 110)
    print(table.to_string(index=False))
    Path("reports").mkdir(exist_ok=True)
    table.to_csv("reports/sensitivity_validation.csv", index=False)
    print("\nSaved to reports/sensitivity_validation.csv")
