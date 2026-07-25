"""The betting recommendation engine.

Takes a race with model win probabilities attached and returns the bets worth
making, across every Australian bet type, ranked by expected value.

The guiding principle is that **most races should produce no bet at all**. A
tool that finds a bet in every race is not finding value, it is finding the
bookmaker's margin. If you run this over a full Saturday and it suggests two
bets, that is the system working correctly.

Exotics are held to a much higher bar than win bets, for two reasons that
compound. First, tote takeout on trifectas and first fours is 21-23% versus
14.5% on the win pool, so you need roughly double the edge to break even.
Second, exotic probabilities are the product of several estimates, so model
error compounds across the legs. The thresholds below reflect both.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations as iter_combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import exotics, tote
from .betting import (
    BetRecommendation, expected_value, fair_odds, kelly_drawdown_note,
    recommend_place_bets, recommend_win_bets, simultaneous_kelly, staking_plan,
)


@dataclass
class Settings:
    """Everything you might want to tune, in one place."""

    bankroll: float = 1000.0
    kelly_fraction: float = 0.25          # quarter Kelly
    max_stake_pct: float = 0.02           # never more than 2% on one bet
    max_race_exposure_pct: float = 0.05   # never more than 5% on one race

    min_edge_win: float = 0.05
    min_edge_place: float = 0.04
    min_edge_exotic: float = 0.20         # exotics must clear a far higher bar

    exchange_commission: float = 0.06     # Betfair MBR; read it per market
    use_tote_for_exotics: bool = True

    # Discount exponents for the ordering model. Refit these on your own data
    # with exotics.fit_lambdas before trusting exotic recommendations.
    lambdas: tuple = exotics.DEFAULT_LAMBDAS

    max_exotic_selections: int = 4        # cap how wide a box may go
    n_sample_support: int = 100_000       # size of the data the model was fitted on


def _labels(frame: pd.DataFrame) -> list[str]:
    """Human-readable runner labels: tab number and name where available."""
    out = []
    for _, row in frame.iterrows():
        number = row.get("tab_number")
        if pd.notna(number):
            out.append(f"{int(number)}. {row['horse']}")
        else:
            out.append(str(row["horse"]))
    return out


def recommend_for_race(
    race: pd.DataFrame,
    settings: Settings | None = None,
    place_odds: Optional[Sequence[float]] = None,
    exotic_dividends: Optional[dict] = None,
) -> dict:
    """Every worthwhile bet in one race.

    `race` is one race's runners with a `p_win` column. Scratched runners are
    dropped and the probabilities renormalised, which also recalculates the
    number of paid places - a race that drops from 8 starters to 7 pays two
    places, not three, and that changes every place-bet calculation.

    `exotic_dividends` optionally maps a tuple of runner indices to an observed
    or estimated dividend. Without it, dividends are estimated from the market's
    own win probabilities put through the same ordering model - which is more
    robust than it sounds, because the ordering model's bias largely cancels
    when you take the ratio of your probability to the market's.
    """
    settings = settings or Settings()

    live = race[~race["scratched"].astype(bool)].copy().reset_index(drop=True)
    n = len(live)
    if n < 2:
        return {"bets": [], "note": "Fewer than 2 runners - no market.", "n_runners": n}

    p = exotics.normalise(live["p_win"].to_numpy(dtype=float))
    live["p_win_normalised"] = p
    odds = live["odds_decimal"].to_numpy(dtype=float)
    labels = _labels(live)
    ids = live["runner_id"].astype(str).tolist()

    market = live["p_market"].to_numpy(dtype=float) if "p_market" in live.columns \
        else np.full(n, np.nan)

    bets: list[BetRecommendation] = []
    notes: list[str] = []

    # ---- Win -----------------------------------------------------------
    bets += recommend_win_bets(
        ids, labels, p, odds,
        bankroll=settings.bankroll,
        min_edge=settings.min_edge_win,
        commission=0.0,                     # fixed-odds bookmaker, no commission
        kelly_fraction_used=settings.kelly_fraction,
        max_stake_pct=settings.max_stake_pct,
        n_sample_support=settings.n_sample_support,
    )

    # ---- Place ---------------------------------------------------------
    place_p, paid = exotics.place_probabilities(p, n_runners=n, lambdas=settings.lambdas)
    live["p_place"] = place_p
    live["places_paid"] = paid

    if paid == 0:
        notes.append(
            f"{n} starters - no place market is formed under 5 runners."
        )
    elif place_odds is not None:
        bets += recommend_place_bets(
            ids, labels, place_p, np.asarray(place_odds, dtype=float),
            bankroll=settings.bankroll,
            min_edge=settings.min_edge_place,
            kelly_fraction_used=settings.kelly_fraction,
            max_stake_pct=settings.max_stake_pct,
            places_paid=paid,
            n_sample_support=settings.n_sample_support,
        )
    elif np.isfinite(market).sum() >= 2:
        # No place prices supplied - estimate them from the tote place pool.
        market_place, _ = exotics.place_probabilities(
            exotics.normalise(np.nan_to_num(market, nan=0.0)),
            n_runners=n, lambdas=settings.lambdas,
        )
        estimated = np.array([
            tote.implied_dividend(q, tote.TAKEOUT["PLACE"]) for q in market_place
        ])
        found = recommend_place_bets(
            ids, labels, place_p, estimated,
            bankroll=settings.bankroll,
            min_edge=settings.min_edge_place,
            kelly_fraction_used=settings.kelly_fraction,
            max_stake_pct=settings.max_stake_pct,
            places_paid=paid,
            n_sample_support=settings.n_sample_support,
            price_is_estimated=True,
        )
        for bet in found:
            bet.rationale += (
                " Place dividend ESTIMATED from the tote pool, not a real "
                "quoted price - check the actual price before betting."
            )
        bets += found
        notes.append("Place prices were estimated from the tote, not quoted.")

    # ---- Exotics -------------------------------------------------------
    if np.isfinite(market).sum() >= 3 and n >= 4:
        bets += _recommend_exotics(
            p, market, ids, labels, n, settings, exotic_dividends
        )
    elif n >= 4:
        notes.append("No market prices - exotic value cannot be assessed.")

    # ---- Exposure cap --------------------------------------------------
    bets.sort(key=lambda b: -b.expected_value)
    cap = settings.bankroll * settings.max_race_exposure_pct
    total = sum(b.stake for b in bets)
    if total > cap and total > 0:
        scale = cap / total
        for bet in bets:
            bet.stake = round(bet.stake * scale, 2)
        notes.append(
            f"Stakes scaled by {scale:.0%} to keep total race exposure within "
            f"{settings.max_race_exposure_pct:.0%} of bankroll."
        )
    bets = [b for b in bets if b.stake > 0]

    if not bets:
        notes.append(
            "No bet. The prices on offer do not beat the model by enough to be "
            "worth the risk. This is the normal outcome for most races."
        )

    return {
        "bets": [b.to_dict() for b in bets],
        "n_runners": n,
        "places_paid": paid,
        "runners": live[[
            "runner_id", "horse", "p_win_normalised", "p_place", "odds_decimal"
        ]].rename(columns={"p_win_normalised": "p_win"}).to_dict("records"),
        "notes": notes,
        "staking_note": kelly_drawdown_note(settings.kelly_fraction),
        "total_stake": round(sum(b.stake for b in bets), 2),
    }


def _recommend_exotics(
    p: np.ndarray,
    market: np.ndarray,
    ids: list[str],
    labels: list[str],
    n: int,
    settings: Settings,
    supplied_dividends: Optional[dict],
) -> list[BetRecommendation]:
    """Quinella, exacta, trifecta and first four, where the edge justifies it.

    Dividends come from the market's own win probabilities pushed through the
    same ordering model. Because both your probability and the market's go
    through identical machinery, a misspecified lambda largely cancels in the
    ratio - so the *edge* estimate is far more trustworthy than the absolute
    dividend estimate.
    """
    out: list[BetRecommendation] = []
    market_p = exotics.normalise(np.nan_to_num(market, nan=0.0))
    if not np.isfinite(market_p).all() or market_p.sum() <= 0:
        return out

    # Only consider the runners the model rates most highly; every extra
    # selection multiplies the combinations and dilutes the ticket.
    top = list(np.argsort(-p)[: min(settings.max_exotic_selections + 2, n)])

    def price(combo: tuple, bet_type: str) -> tuple[float, float, float]:
        """(model probability, market probability, dividend) for a combination."""
        if len(combo) == 2 and bet_type == "QUINELLA":
            mine = exotics.quinella_probability(p, combo[0], combo[1], settings.lambdas)
            theirs = exotics.quinella_probability(market_p, combo[0], combo[1], settings.lambdas)
        else:
            mine = exotics.order_probability(p, combo, settings.lambdas)
            theirs = exotics.order_probability(market_p, combo, settings.lambdas)
        if supplied_dividends and combo in supplied_dividends:
            dividend = float(supplied_dividends[combo])
        else:
            dividend = tote.implied_dividend(theirs, tote.TAKEOUT[bet_type])
        return mine, theirs, dividend

    candidates: list[tuple] = []
    for a, b in iter_combinations(top, 2):
        candidates.append(("QUINELLA", (a, b)))
        candidates.append(("EXACTA", (a, b)))
        candidates.append(("EXACTA", (b, a)))
    for combo in iter_combinations(top[:5], 3):
        for order in ((combo[0], combo[1], combo[2]), (combo[1], combo[0], combo[2])):
            candidates.append(("TRIFECTA", order))

    for bet_type, combo in candidates:
        mine, theirs, dividend = price(combo, bet_type)
        if not np.isfinite(dividend) or dividend <= 1.0 or mine <= 0:
            continue
        ev = mine * dividend - 1.0
        if ev < settings.min_edge_exotic:
            continue

        stake = staking_plan(
            mine, dividend, settings.bankroll,
            kelly_fraction_used=settings.kelly_fraction * 0.5,   # extra caution
            max_stake_pct=settings.max_stake_pct * 0.5,
        )
        if stake <= 0:
            continue

        joiner = "-" if bet_type != "QUINELLA" else " & "
        selection = joiner.join(labels[i] for i in combo)
        takeout = tote.TAKEOUT[bet_type]

        out.append(BetRecommendation(
            bet_type=bet_type,
            selection=selection,
            selection_ids=[ids[i] for i in combo],
            model_probability=float(mine),
            offered_odds=float(dividend),
            fair_odds=float(fair_odds(mine)),
            expected_value=float(ev),
            edge_pct=float(ev * 100),
            stake=stake,
            confidence="LOW",   # exotic estimates never earn better than this
            combinations=1,
            rationale=(
                f"Model rates this combination at {mine:.3%} against the "
                f"market's {theirs:.3%}. Estimated dividend ${dividend:,.2f} "
                f"after {takeout:.1%} tote takeout. "
                f"DIVIDEND IS AN ESTIMATE - the real one is only known after "
                f"the race, and pool movements can change it substantially."
            ),
        ))

    return sorted(out, key=lambda b: -b.expected_value)[:5]


def evaluate_box(
    p: Sequence[float],
    selections: Sequence[int],
    positions: int,
    market: Sequence[float],
    labels: Sequence[str] | None = None,
    bet_type: str = "TRIFECTA",
    lambdas=exotics.DEFAULT_LAMBDAS,
    unit: float = 1.0,
    outlay: float | None = None,
) -> dict:
    """Assess a specific boxed ticket, combination by combination.

    Reports which individual combinations are worth having and which are
    dragging the ticket down. Adding a selection to a box is only worthwhile if
    the combinations it brings with it are individually positive - going from
    four horses to five in a trifecta buys 36 extra combinations, and if those
    are all negative they can turn a good ticket into a bad one.
    """
    p = exotics.normalise(p)
    market_p = exotics.normalise(market)
    selections = list(dict.fromkeys(int(s) for s in selections))
    labels = labels or [str(i) for i in range(len(p))]
    takeout = tote.TAKEOUT.get(bet_type, 0.21)

    from itertools import permutations
    rows = []
    for combo in permutations(selections, positions):
        mine = exotics.order_probability(p, combo, lambdas)
        theirs = exotics.order_probability(market_p, combo, lambdas)
        dividend = tote.implied_dividend(theirs, takeout)
        rows.append({
            "combination": "-".join(labels[i] for i in combo),
            "model_prob": mine,
            "market_prob": theirs,
            "est_dividend": dividend,
            "ev_per_unit": mine * dividend - 1.0,
        })

    table = pd.DataFrame(rows).sort_values("ev_per_unit", ascending=False)
    n_combos = len(table)
    full_cost = n_combos * unit
    flexi = tote.FlexiBet(n_combos, outlay if outlay is not None else full_cost, unit)

    total_expected = float((table["model_prob"] * table["est_dividend"]).sum())
    roi = total_expected / n_combos - 1.0 if n_combos else float("nan")

    positive = table[table["ev_per_unit"] > 0]
    trimmed_roi = float("nan")
    if len(positive):
        trimmed_roi = float(
            (positive["model_prob"] * positive["est_dividend"]).sum() / len(positive) - 1.0
        )

    return {
        "bet_type": bet_type,
        "combinations": n_combos,
        "full_cost": full_cost,
        "flexi": flexi.describe(),
        "hit_probability": float(table["model_prob"].sum()),
        "roi_per_dollar": roi,
        "positive_combinations": int(len(positive)),
        "roi_if_only_positive_taken": trimmed_roi,
        "detail": table,
        "verdict": (
            f"ROI {roi:+.1%} per dollar. "
            + (f"Only {len(positive)} of {n_combos} combinations are individually "
               f"positive - taking just those would return {trimmed_roi:+.1%}. "
               if len(positive) and len(positive) < n_combos else "")
            + f"Note the {takeout:.1%} tote takeout on this pool."
        ),
    }


def format_race_report(result: dict, race_name: str = "") -> str:
    """Plain-English summary of one race, for someone who does not read code."""
    lines: list[str] = []
    if race_name:
        lines.append(f"=== {race_name} ===")
    lines.append(f"{result['n_runners']} runners, {result.get('places_paid', 0)} place dividends paid.")
    lines.append("")

    runners = pd.DataFrame(result.get("runners", []))
    if not runners.empty:
        runners = runners.sort_values("p_win", ascending=False)
        lines.append("Model assessment:")
        lines.append(f"  {'Horse':<28} {'Win%':>7} {'Place%':>8} {'Fair $':>8} {'Market $':>9}")
        for _, row in runners.iterrows():
            fair = 1 / row["p_win"] if row["p_win"] > 0 else float("inf")
            offered = row.get("odds_decimal")
            offered_text = f"${offered:.2f}" if pd.notna(offered) else "-"
            lines.append(
                f"  {str(row['horse'])[:28]:<28} {row['p_win']:>6.1%} "
                f"{row.get('p_place', float('nan')):>7.1%} ${fair:>7.2f} {offered_text:>9}"
            )
        lines.append("")

    bets = result.get("bets", [])
    if bets:
        lines.append(f"RECOMMENDED BETS (total ${result.get('total_stake', 0):,.2f}):")
        for bet in bets:
            lines.append(
                f"  [{bet['confidence']}] {bet['bet_type']} - {bet['selection']}"
            )
            lines.append(
                f"      ${bet['stake']:,.2f} at ${bet['offered_odds']:.2f} "
                f"(fair ${bet['fair_odds']:.2f}, edge {bet['edge_pct']:+.1f}%)"
            )
            lines.append(f"      {bet['rationale']}")
    else:
        lines.append("NO BET.")

    for note in result.get("notes", []):
        lines.append(f"  Note: {note}")
    if result.get("staking_note"):
        lines.append(f"  Staking: {result['staking_note']}")

    return "\n".join(lines)
