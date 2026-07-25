"""Race analysis: turn a race plus a model into a betting recommendation.

This is the layer the CLI and the web UI both sit on. It takes a race, a
fitted model and the current market, and produces:

  * a win probability for every runner
  * place probabilities under the correct Australian field-size rules
  * the edge on each available bet
  * suggested stakes under fractional Kelly with hard caps
  * plain-English reasons, so you can judge the recommendation rather than
    obey it

Nothing here invents an opportunity. If no runner clears the edge
threshold, the correct output is "no bet", and that is what it returns.
Most races should produce no bet. A tool that finds a play in every race
is not finding value, it is finding noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .betting.exotics import (
    DEFAULT_LAMBDA_2ND,
    DEFAULT_LAMBDA_3RD,
    place_probabilities,
    places_paid,
    quinella_probability,
    trifecta_probability,
)
from .betting.odds import Method, betfair_commission, devig
from .betting.staking import Stake, StakingPolicy, expected_value, size_bets
from .features import RollingContext, build_features
from .types import Race, Runner


@dataclass
class RunnerAssessment:
    """Everything the tool concluded about one horse."""

    number: int
    name: str
    barrier: Optional[int]
    jockey: Optional[str]
    model_win_prob: float
    market_win_prob: Optional[float]
    model_place_prob: float
    win_odds: Optional[float]
    place_odds: Optional[float]
    win_edge: Optional[float]
    place_edge: Optional[float]
    fair_win_odds: float
    rank: int
    notes: list[str] = field(default_factory=list)

    @property
    def is_value(self) -> bool:
        return self.win_edge is not None and self.win_edge > 0


@dataclass
class RaceAnalysis:
    """The complete assessment of one race."""

    race: Race
    assessments: list[RunnerAssessment]
    stakes: list[Stake]
    n_places: int
    market_overround: Optional[float]
    exotic_suggestions: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def top_pick(self) -> Optional[RunnerAssessment]:
        return self.assessments[0] if self.assessments else None

    @property
    def has_bet(self) -> bool:
        return bool(self.stakes)

    def summary(self) -> str:
        lines = [
            f"{self.race.track.name} R{self.race.race_number} "
            f"{self.race.distance_m}m -- {self.race.name or 'race'}",
        ]
        if self.race.track_condition:
            lines.append(f"Track: {self.race.track_condition} "
                         f"({self.race.field_size} runners, {self.n_places} places paid)")
        if self.market_overround is not None:
            lines.append(f"Market overround: {self.market_overround:.1%}")
        return "\n".join(lines)


def _explain(runner: Runner, race: Race, assessment_rank: int,
             model_prob: float, market_prob: Optional[float]) -> list[str]:
    """Short, human-readable reasons behind a runner's rating."""
    notes: list[str] = []
    history = runner.runs_before(race.date)

    if not history:
        notes.append("First starter -- no form to work from, rating is a prior only")
        return notes

    recent = history[:3]
    placings = [str(r.finish_position) for r in recent if r.finish_position]
    if placings:
        notes.append(f"Last {len(placings)} starts: {'-'.join(placings)}")

    days = (race.date - history[0].date).days
    if days >= 90:
        notes.append(f"First-up after {days} days")
    elif days <= 10:
        notes.append(f"Quick back-up ({days} days)")

    at_distance = [r for r in history if abs(r.distance_m - race.distance_m) <= 100]
    if at_distance:
        wins = sum(1 for r in at_distance if r.won)
        notes.append(f"{wins} win{'s' if wins != 1 else ''} "
                     f"from {len(at_distance)} at the trip")
    else:
        notes.append("First time at this distance")

    if race.track_condition and race.track_condition >= 5:
        wet = [r for r in history if r.track_condition and r.track_condition >= 5]
        if wet:
            wins = sum(1 for r in wet if r.won)
            notes.append(f"Wet track: {wins} from {len(wet)}")
        else:
            notes.append("Unproven on rain-affected ground")

    if runner.barrier and race.field_size >= 8:
        if runner.barrier <= 3:
            notes.append(f"Good draw (barrier {runner.barrier})")
        elif runner.barrier >= race.field_size - 2:
            notes.append(f"Wide draw (barrier {runner.barrier})")

    if market_prob is not None and market_prob > 0:
        ratio = model_prob / market_prob
        if ratio > 1.25:
            notes.append(f"Model rates it {ratio:.0%} of market price -- overpriced")
        elif ratio < 0.8:
            notes.append("Model rates it shorter than the market -- underpriced")

    return notes


def analyse_race(
    race: Race,
    model,
    context: RollingContext,
    bankroll: float = 1000.0,
    policy: Optional[StakingPolicy] = None,
    devig_method: Method = "shin",
    commission: Optional[float] = None,
    lambda_2nd: float = DEFAULT_LAMBDA_2ND,
    lambda_3rd: float = DEFAULT_LAMBDA_3RD,
    include_market_features: bool = False,
) -> RaceAnalysis:
    """Assess a race and recommend bets."""
    policy = policy or StakingPolicy()
    if commission is None:
        commission = betfair_commission(race.track.state)

    runners = race.active_runners
    warnings: list[str] = []
    if len(runners) < 2:
        return RaceAnalysis(race, [], [], 0, None,
                            warnings=["Fewer than two runners; nothing to analyse."])

    feature_set = build_features(race, context,
                                 include_market=include_market_features)

    win_odds = [r.fixed_win_odds or r.tote_win_odds for r in runners]
    has_market = all(o and o > 1.0 for o in win_odds)

    if has_market:
        market_probs = devig(win_odds, devig_method)
        overround = sum(1.0 / o for o in win_odds) - 1.0
    else:
        market_probs = None
        overround = None
        warnings.append(
            "No complete market available. Probabilities come from the "
            "fundamental model alone, which is materially less reliable -- "
            "the market is the strongest single predictor there is.")

    try:
        probabilities = model.predict_proba(
            feature_set.matrix, win_odds if has_market else None)
    except TypeError:
        probabilities = model.predict_proba(feature_set.matrix)
    probabilities = np.asarray(probabilities, dtype=float)

    n_places = places_paid(len(runners))
    place_probs = (place_probabilities(probabilities, len(runners),
                                       lambda_2nd=lambda_2nd, lambda_3rd=lambda_3rd)
                   if n_places else np.zeros(len(runners)))

    order = np.argsort(-probabilities)
    rank_of = {int(idx): position + 1 for position, idx in enumerate(order)}

    assessments: list[RunnerAssessment] = []
    candidates: list[tuple[str, str, float, float]] = []

    for index, runner in enumerate(runners):
        model_p = float(probabilities[index])
        market_p = float(market_probs[index]) if market_probs is not None else None
        odds = win_odds[index]
        place_odds = runner.fixed_place_odds

        win_edge = expected_value(model_p, odds, commission) if odds else None
        place_edge = (expected_value(float(place_probs[index]), place_odds, commission)
                      if place_odds and n_places else None)

        assessments.append(RunnerAssessment(
            number=runner.number,
            name=runner.name,
            barrier=runner.barrier,
            jockey=runner.jockey,
            model_win_prob=model_p,
            market_win_prob=market_p,
            model_place_prob=float(place_probs[index]),
            win_odds=odds,
            place_odds=place_odds,
            win_edge=win_edge,
            place_edge=place_edge,
            fair_win_odds=(1.0 / model_p) if model_p > 0 else float("inf"),
            rank=rank_of[index],
            notes=_explain(runner, race, rank_of[index], model_p, market_p),
        ))

        if odds:
            candidates.append((f"#{runner.number} {runner.name}", "win", model_p, odds))
        if place_odds and n_places:
            candidates.append((f"#{runner.number} {runner.name}", "place",
                               float(place_probs[index]), place_odds))

    stakes = size_bets(candidates, bankroll, policy, commission=commission)
    assessments.sort(key=lambda a: a.rank)

    exotics = _suggest_exotics(probabilities, runners, order,
                               lambda_2nd, lambda_3rd)

    if not stakes:
        warnings.append(
            "No bet. Nothing in this race clears the minimum edge, which is "
            "the normal and correct outcome for most races.")

    # Sanity guard. Genuine value is rare and shows up on one or two
    # runners. If a large share of the field looks like value, the cause is
    # almost never that the whole market is wrong -- it is that the prices
    # being compared against are stale, wrong, or from a different market
    # (a place price compared against a win probability, say). Surfacing
    # this loudly is the difference between a tool that catches its own
    # bad input and one that confidently recommends nonsense.
    n_value = sum(1 for a in assessments if a.is_value)
    if len(assessments) >= 5 and n_value > max(3, len(assessments) * 0.4):
        warnings.append(
            f"SUSPICIOUS: {n_value} of {len(assessments)} runners show positive "
            f"edge. Real value is rare and concentrated. This pattern almost "
            f"always means the odds being used are wrong or stale, not that "
            f"the market has mispriced the whole field. Do not bet on this.")
        stakes = []

    if overround is not None and overround < 0.0:
        warnings.append(
            f"Book totals {1 + overround:.1%}, i.e. under 100%. That is an "
            f"arbitrage or, far more likely, incomplete or stale prices.")

    return RaceAnalysis(
        race=race,
        assessments=assessments,
        stakes=stakes,
        n_places=n_places,
        market_overround=overround,
        exotic_suggestions=exotics,
        warnings=warnings,
    )


def _suggest_exotics(probabilities: np.ndarray, runners: Sequence[Runner],
                     order: np.ndarray, lambda_2nd: float,
                     lambda_3rd: float) -> list[dict]:
    """Probabilities for the most likely exotic combinations.

    These are *probabilities only*, deliberately not expected values.
    Computing EV for a tote exotic needs the pool composition -- how much
    money is already on each combination -- which is not available before
    the jump. Without it, and against a 20-23% takeout, exotic EV is
    guesswork. They are shown so you can see what the model thinks is
    likely, not as recommended bets.
    """
    if len(runners) < 4:
        return []

    top = [int(i) for i in order[:4]]
    suggestions: list[dict] = []

    a, b = top[0], top[1]
    suggestions.append({
        "type": "quinella",
        "selection": f"#{runners[a].number} + #{runners[b].number}",
        "probability": quinella_probability(probabilities, a, b,
                                            lambda_2nd=lambda_2nd,
                                            lambda_3rd=lambda_3rd),
        "note": "Top two on model rating, either order",
    })

    c = top[2]
    suggestions.append({
        "type": "trifecta",
        "selection": (f"#{runners[a].number} > #{runners[b].number} "
                      f"> #{runners[c].number}"),
        "probability": trifecta_probability(probabilities, a, b, c,
                                            lambda_2nd=lambda_2nd,
                                            lambda_3rd=lambda_3rd),
        "note": "Straight trifecta on the top three",
    })
    return suggestions
