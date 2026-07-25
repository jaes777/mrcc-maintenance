"""Model evaluation, honestly done.

Three metrics matter, in this order:

1. **Log loss on the winner** - how much probability the model put on the horse
   that actually won. This is the metric to optimise. It is sensitive to the
   whole distribution, not just the top pick.

2. **ROI** - whether betting the model's opinion actually made money at the
   prices available. A model can have excellent log loss and still lose money,
   because the market's prices already reflect that accuracy.

3. **Top-1 strike rate** - how often the highest-rated horse won. This is the
   number people ask about and the least useful of the three, because you can
   trivially raise it by always picking the favourite and still go broke.

The significance machinery here exists because racing ROI is extremely noisy.
A few hundred bets tells you essentially nothing. `roi_significance` makes that
concrete rather than leaving it as a caveat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Probability quality
# --------------------------------------------------------------------------

def winner_log_loss(frame: pd.DataFrame, prob_col: str = "p_win") -> float:
    """Mean -log(probability assigned to the actual winner), per race.

    Lower is better. For scale: assigning every horse an equal chance in an
    average 10-horse field scores about 2.30.
    """
    winners = frame[(frame["won"] == 1) & frame[prob_col].notna()]
    if winners.empty:
        return float("nan")
    return float(-np.mean(np.log(np.clip(winners[prob_col], 1e-12, 1.0))))


def brier_score(frame: pd.DataFrame, prob_col: str = "p_win") -> float:
    """Mean squared error of the win probabilities across all runners."""
    sub = frame[frame["won"].notna() & frame[prob_col].notna()]
    if sub.empty:
        return float("nan")
    return float(np.mean((sub[prob_col] - sub["won"]) ** 2))


def top1_strike_rate(frame: pd.DataFrame, prob_col: str = "p_win") -> float:
    """How often the model's highest-rated runner won."""
    sub = frame[frame["won"].notna() & frame[prob_col].notna()]
    if sub.empty:
        return float("nan")
    top = sub.loc[sub.groupby("race_id")[prob_col].idxmax()]
    return float(top["won"].mean())


def topk_strike_rate(frame: pd.DataFrame, k: int = 3, prob_col: str = "p_win") -> float:
    """How often the winner was among the model's top k."""
    sub = frame[frame["won"].notna() & frame[prob_col].notna()].copy()
    if sub.empty:
        return float("nan")
    sub["rank"] = sub.groupby("race_id")[prob_col].rank(ascending=False, method="first")
    hits = sub[(sub["rank"] <= k) & (sub["won"] == 1)]
    return float(len(hits) / sub["race_id"].nunique())


def calibration_table(
    frame: pd.DataFrame,
    prob_col: str = "p_win",
    bins: tuple[float, ...] = (0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 1.0),
) -> pd.DataFrame:
    """Predicted vs actual win rate by probability band.

    A well-calibrated model wins about 20% of the races where it says 20%. If
    the actual column sits consistently below the predicted column, the model
    is overconfident and every expected-value calculation built on it is wrong.
    """
    sub = frame[frame["won"].notna() & frame[prob_col].notna()].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["band"] = pd.cut(sub[prob_col], bins=list(bins), include_lowest=True)
    table = sub.groupby("band", observed=True).agg(
        runners=("won", "size"),
        predicted=(prob_col, "mean"),
        actual=("won", "mean"),
    ).reset_index()
    table["difference"] = table["actual"] - table["predicted"]
    # Binomial standard error on the observed rate, so you can see whether a
    # gap is real or just a small bucket.
    table["std_error"] = np.sqrt(
        table["actual"] * (1 - table["actual"]) / table["runners"].clip(lower=1)
    )
    table["calibrated"] = table["difference"].abs() <= 2 * table["std_error"]
    return table


# --------------------------------------------------------------------------
# Money
# --------------------------------------------------------------------------

@dataclass
class ROIResult:
    n_bets: int
    total_staked: float
    total_return: float
    profit: float
    roi: float                 # profit / staked
    strike_rate: float
    std_error: float           # standard error of the ROI estimate
    t_stat: float
    p_value_one_sided: float
    verdict: str

    def summary(self) -> str:
        return (
            f"{self.n_bets} bets, ${self.total_staked:,.0f} staked, "
            f"${self.profit:+,.0f} profit, ROI {self.roi:+.2%}, "
            f"strike rate {self.strike_rate:.1%}. {self.verdict}"
        )


def roi_significance(returns_per_unit: np.ndarray, stakes: np.ndarray | None = None) -> ROIResult:
    """ROI plus an honest statement of whether it is distinguishable from zero.

    `returns_per_unit` is profit per $1 staked for each bet: -1 for a loser,
    (odds - 1) for a winner.

    The verdict is deliberately harsh. Racing returns have enormous variance -
    a single 40/1 winner moves a 500-bet ROI by 8 percentage points - so a
    positive ROI over a small sample is the expected outcome of luck, not
    evidence of an edge.
    """
    r = np.asarray(returns_per_unit, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return ROIResult(0, 0, 0, 0, float("nan"), float("nan"),
                         float("nan"), float("nan"), float("nan"),
                         "No bets placed.")

    s = np.ones(n) if stakes is None else np.asarray(stakes, dtype=float)[:n]
    staked = float(s.sum())
    profit = float((r * s).sum())
    roi = profit / staked if staked else float("nan")
    strike = float((r > 0).mean())

    std = float(np.std(r, ddof=1)) if n > 1 else float("nan")
    se = std / np.sqrt(n) if n > 1 else float("nan")
    t = roi / se if (np.isfinite(se) and se > 0) else float("nan")

    # One-sided normal approximation for "is the true ROI above zero".
    from scipy.stats import norm
    p = float(1 - norm.cdf(t)) if np.isfinite(t) else float("nan")

    if n < 300:
        verdict = (
            f"MEANINGLESS SAMPLE: {n} bets proves nothing either way. "
            f"You need thousands."
        )
    elif not np.isfinite(p):
        verdict = "Could not compute significance."
    elif p < 0.01:
        verdict = f"Statistically significant at p={p:.3f}, but see the sample-size note."
    elif p < 0.05:
        verdict = f"Weakly significant (p={p:.3f}). Treat with suspicion."
    else:
        verdict = (
            f"NOT significant (p={p:.3f}). This result is consistent with having "
            f"no edge at all."
        )

    return ROIResult(n, staked, staked + profit, profit, roi, strike, se, t, p, verdict)


def bet_return_std(average_odds: float) -> float:
    """Standard deviation of profit per $1 staked, at fair odds.

    For a unit stake at decimal odds d, the return is (d-1) on a win and -1 on
    a loss. At fair odds p = 1/d:

        Var[R] = p(d-1)^2 + (1-p)(1)^2 = [(d-1)^2 + (d-1)]/d = d - 1

    so the standard deviation is simply sqrt(d - 1). Betting at $5 has a
    per-bet standard deviation of 2.0 - forty times the size of a 5% edge,
    which is the whole reason racing results take so long to interpret.
    """
    return float(np.sqrt(max(average_odds - 1.0, 1e-9)))


def bets_needed(true_roi: float, average_odds: float = 5.0, power: float = 0.8,
                alpha: float = 0.05) -> int:
    """How many bets are needed to demonstrate an edge of `true_roi`.

        n = ((z_alpha + z_power) * sqrt(d - 1) / roi)^2

    This function exists to make one point concrete. Proving a 5% edge at $5
    average odds takes roughly 9,900 bets. At a realistic rate of a few
    thousand bets a year, that is two to three years of live betting. Anyone
    showing you a profitable 200-bet sample is showing you noise.
    """
    from scipy.stats import norm
    if true_roi <= 0:
        return -1
    z_alpha = norm.ppf(1 - alpha)
    z_power = norm.ppf(power)
    sigma = bet_return_std(average_odds)
    return int(np.ceil(((z_alpha + z_power) * sigma / true_roi) ** 2))


def sample_size_table(rois=(0.02, 0.05, 0.10),
                      odds=(3.0, 5.0, 8.0, 10.0, 21.0)) -> pd.DataFrame:
    """Bets required to prove various edges at various average prices."""
    rows = []
    for roi in rois:
        for d in odds:
            rows.append({
                "true_roi": roi,
                "average_odds": d,
                "per_bet_std": round(bet_return_std(d), 2),
                "bets_needed": bets_needed(roi, d),
            })
    return pd.DataFrame(rows)


def flat_stake_roi(
    frame: pd.DataFrame,
    prob_col: str = "p_win",
    odds_col: str = "odds_decimal",
    min_edge: float = 0.05,
    commission: float = 0.0,
) -> ROIResult:
    """ROI from flat-staking every runner whose expected value clears `min_edge`."""
    sub = frame[
        frame["won"].notna() & frame[prob_col].notna() & frame[odds_col].notna()
        & (frame[odds_col] > 1.0) & ~frame["scratched"].astype(bool)
    ].copy()
    if sub.empty:
        return roi_significance(np.array([]))

    profit_if_win = (sub[odds_col] - 1.0) * (1.0 - commission)
    sub["ev"] = sub[prob_col] * profit_if_win - (1 - sub[prob_col])
    bets = sub[sub["ev"] >= min_edge]
    if bets.empty:
        return roi_significance(np.array([]))

    returns = np.where(
        bets["won"] == 1,
        (bets[odds_col] - 1.0) * (1.0 - commission),
        -1.0,
    )
    return roi_significance(returns)


def roi_by_odds_band(
    frame: pd.DataFrame,
    prob_col: str = "p_win",
    odds_col: str = "odds_decimal",
    min_edge: float = 0.05,
    bands: tuple[float, ...] = (1.0, 2.0, 3.5, 6.0, 11.0, 21.0, 51.0, 1000.0),
) -> pd.DataFrame:
    """Break ROI down by price bracket.

    Concentrated losses at long odds are the classic symptom of a model that
    overestimates small probabilities - the most common failure mode there is.
    """
    sub = frame[
        frame["won"].notna() & frame[prob_col].notna() & frame[odds_col].notna()
        & (frame[odds_col] > 1.0) & ~frame["scratched"].astype(bool)
    ].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["ev"] = sub[prob_col] * (sub[odds_col] - 1) - (1 - sub[prob_col])
    bets = sub[sub["ev"] >= min_edge].copy()
    if bets.empty:
        return pd.DataFrame()

    bets["band"] = pd.cut(bets[odds_col], bins=list(bands))
    bets["ret"] = np.where(bets["won"] == 1, bets[odds_col] - 1.0, -1.0)

    table = bets.groupby("band", observed=True).agg(
        bets=("ret", "size"),
        winners=("won", "sum"),
        roi=("ret", "mean"),
        avg_odds=(odds_col, "mean"),
        avg_prob=(prob_col, "mean"),
    ).reset_index()
    table["strike_rate"] = table["winners"] / table["bets"]
    table["std_error"] = bets.groupby("band", observed=True)["ret"].std().to_numpy() / np.sqrt(table["bets"])
    return table


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def market_baseline(frame: pd.DataFrame, odds_col: str = "odds_decimal") -> dict:
    """What the market alone achieves on the same races.

    This is the bar. A model that does not beat these numbers has added
    nothing, and the honest response is to say so rather than to report the
    model's numbers in isolation.
    """
    sub = frame[frame["won"].notna() & frame[odds_col].notna() & (frame[odds_col] > 1)
                & ~frame["scratched"].astype(bool)].copy()
    if sub.empty:
        return {}

    favourites = sub.loc[sub.groupby("race_id")[odds_col].idxmin()]
    fav_returns = np.where(favourites["won"] == 1, favourites[odds_col] - 1.0, -1.0)
    fav_roi = roi_significance(fav_returns)

    if "mkt_prob" in sub.columns and sub["mkt_prob"].notna().any():
        mkt_ll = winner_log_loss(sub, "mkt_prob")
    else:
        mkt_ll = float("nan")

    return {
        "favourite_strike_rate": float(favourites["won"].mean()),
        "favourite_roi": fav_roi.roi,
        "favourite_n": fav_roi.n_bets,
        "market_log_loss": mkt_ll,
        "mean_overround": float(sub.groupby("race_id")
                                .apply(lambda g: (1 / g[odds_col]).sum(), include_groups=False)
                                .mean()),
        "n_races": int(sub["race_id"].nunique()),
    }


def full_report(frame: pd.DataFrame, prob_col: str = "p_win",
                min_edge: float = 0.05, commission: float = 0.0) -> dict:
    """Everything, in one dict, for printing or saving."""
    baseline = market_baseline(frame)
    roi = flat_stake_roi(frame, prob_col=prob_col, min_edge=min_edge, commission=commission)

    report = {
        "n_races": int(frame.loc[frame["won"].notna(), "race_id"].nunique()),
        "n_runners": int(frame["won"].notna().sum()),
        "log_loss": winner_log_loss(frame, prob_col),
        "brier": brier_score(frame, prob_col),
        "top1_strike_rate": top1_strike_rate(frame, prob_col),
        "top3_strike_rate": topk_strike_rate(frame, 3, prob_col),
        "roi": roi.roi,
        "roi_n_bets": roi.n_bets,
        "roi_p_value": roi.p_value_one_sided,
        "roi_verdict": roi.verdict,
        "baseline": baseline,
    }

    if "p_model" in frame.columns:
        report["log_loss_model_only"] = winner_log_loss(frame, "p_model")
    if "p_market" in frame.columns and frame["p_market"].notna().any():
        report["log_loss_market_only"] = winner_log_loss(frame, "p_market")
        report["beats_market"] = (
            np.isfinite(report["log_loss"])
            and np.isfinite(report["log_loss_market_only"])
            and report["log_loss"] < report["log_loss_market_only"]
        )
    return report
