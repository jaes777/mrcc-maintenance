"""Blending the fundamental model with the market.

THIS IS THE MOST IMPORTANT MODULE IN THE PROJECT.

Benter's central finding was not that his handicapping model was good. It
was that his model *combined with the public odds* was much better than
either alone. In his 1994 paper the fundamental model scored R^2 = 0.1245;
adding the public's probability estimate lifted it to 0.1396.

The reason is that the betting market aggregates information no feed
carries: stable confidence, trackwork whispers, a jockey's read of the
horse, late gear changes, money from people who know things. No solo model
reproduces that. Your model's job is to be a *correction to* the market,
not a replacement for it.

Tellingly, Benter also found that a professional tipster's opinion added
essentially nothing once the public estimate was known -- but the
fundamental model did. The market is very hard to beat, and the way you
beat it is not by ignoring it.

THE METHOD
----------
Fit a second conditional logit with exactly two features: the log of the
fundamental probability and the log of the devigged market probability.

    V_i = alpha * ln(f_i) + beta * ln(pi_i)
    p_i = exp(V_i) / sum_j exp(V_j)

Because the softmax of a log is a power, this has a closed form:

    p_i = (f_i^alpha * pi_i^beta) / sum_j (f_j^alpha * pi_j^beta)

which is a normalised geometric blend in log-odds space.

WHAT TO EXPECT
--------------
beta should come out large relative to alpha -- the market deserves most of
the weight. If alpha comes out comparable to or larger than beta, the
overwhelmingly likely explanation is leakage in your fundamental features,
not that you have out-thought the entire betting public. Treat a large
alpha as a bug report.

alpha + beta is not constrained to 1. Above 1 sharpens the blend (the
market is under-confident); below 1 flattens it. A beta noticeably above 1
with alpha near 0 is the classic signature of favourite-longshot bias
still present in the input odds.

DISCIPLINE
----------
Stage 1 and stage 2 must be fitted on *disjoint* data, and both must
precede the evaluation period in time. Fitting both on the same races
inflates alpha, because the stage-1 probabilities partly memorised those
races. `TwoStageModel.fit` enforces the split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from ..betting.odds import Method, devig
from .conditional_logit import ConditionalLogit, RaceObservation

log = logging.getLogger(__name__)

_FLOOR = 1e-6  # keeps logs finite for hopeless outsiders


@dataclass
class BlendedRace:
    """One race's inputs to the stage-2 fit."""

    fundamental: np.ndarray   # stage-1 probabilities, sums to 1
    market: np.ndarray        # devigged market probabilities, sums to 1
    winner_index: int


@dataclass
class MarketBlend:
    """Stage 2: fits alpha and beta by maximum likelihood."""

    alpha: float = 0.25
    beta: float = 0.90
    fitted: bool = False
    n_races_fitted: int = 0
    log_likelihood: Optional[float] = None

    def fit(self, races: Sequence[BlendedRace]) -> "MarketBlend":
        usable = [r for r in races
                  if r.fundamental.size >= 2
                  and r.fundamental.size == r.market.size
                  and np.all(np.isfinite(r.market))
                  and 0 <= r.winner_index < r.fundamental.size]
        if len(usable) < 30:
            log.warning(
                "Only %d races available to fit the market blend; keeping "
                "defaults alpha=%.2f beta=%.2f", len(usable), self.alpha, self.beta)
            return self

        log_f = [np.log(np.clip(r.fundamental, _FLOOR, 1.0)) for r in usable]
        log_m = [np.log(np.clip(r.market, _FLOOR, 1.0)) for r in usable]
        winners = [r.winner_index for r in usable]

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            alpha, beta = params
            total = 0.0
            grad = np.zeros(2)
            for lf, lm, winner in zip(log_f, log_m, winners):
                scores = alpha * lf + beta * lm
                peak = scores.max()
                exp_scores = np.exp(scores - peak)
                denominator = exp_scores.sum()
                probs = exp_scores / denominator

                total += scores[winner] - (peak + np.log(denominator))
                grad[0] += lf[winner] - float(probs @ lf)
                grad[1] += lm[winner] - float(probs @ lm)
            return -total, -grad

        result = minimize(objective, np.array([self.alpha, self.beta]),
                          jac=True, method="L-BFGS-B",
                          bounds=[(0.0, 5.0), (0.0, 5.0)],
                          options={"maxiter": 300})

        self.alpha, self.beta = float(result.x[0]), float(result.x[1])
        self.fitted = True
        self.n_races_fitted = len(usable)
        self.log_likelihood = -float(result.fun)

        if self.alpha > self.beta:
            log.warning(
                "Blend fitted alpha=%.3f > beta=%.3f. The fundamental model is "
                "being given more weight than the market, which is very unusual. "
                "Check for leakage in the features before trusting this.",
                self.alpha, self.beta)
        return self

    def combine(self, fundamental: np.ndarray, market: Optional[np.ndarray]
                ) -> np.ndarray:
        """Blend the two probability vectors.

        Falls back to the fundamental model alone when there is no market
        (an unpriced race), and to the market alone when the fundamental
        model has nothing to say.
        """
        fundamental = np.asarray(fundamental, dtype=float)
        if fundamental.size == 0:
            return fundamental
        if market is None or not np.all(np.isfinite(market)):
            return fundamental / fundamental.sum() if fundamental.sum() > 0 else fundamental

        market = np.asarray(market, dtype=float)
        scores = (self.alpha * np.log(np.clip(fundamental, _FLOOR, 1.0))
                  + self.beta * np.log(np.clip(market, _FLOOR, 1.0)))
        exp_scores = np.exp(scores - scores.max())
        return exp_scores / exp_scores.sum()

    def describe(self) -> str:
        state = "fitted" if self.fitted else "default (not fitted)"
        return (f"MarketBlend[{state}] alpha={self.alpha:.3f} (model) "
                f"beta={self.beta:.3f} (market) on {self.n_races_fitted} races")


@dataclass
class TwoStageModel:
    """The full Benter-style pipeline: fundamentals, then market blending."""

    fundamental: ConditionalLogit
    blend: MarketBlend
    devig_method: Method = "shin"

    @classmethod
    def create(cls, l2: float = 1.0, devig_method: Method = "shin") -> "TwoStageModel":
        return cls(fundamental=ConditionalLogit(l2=l2), blend=MarketBlend(),
                   devig_method=devig_method)

    def fit(
        self,
        observations: Sequence[RaceObservation],
        odds_by_race: dict[str, list[Optional[float]]],
        feature_names: Optional[list[str]] = None,
        stage1_fraction: float = 0.6,
        plackett_luce_depth: int = 3,
    ) -> "TwoStageModel":
        """Fit both stages on disjoint, time-ordered slices.

        `observations` must already be in chronological order. The first
        `stage1_fraction` of them trains the fundamental model; the
        remainder fits the blend weights on predictions the fundamental
        model has never seen.
        """
        if not observations:
            raise ValueError("No observations to fit.")

        split = max(1, int(len(observations) * stage1_fraction))
        stage1 = observations[:split]
        stage2 = observations[split:]

        if len(stage2) < 30:
            log.warning(
                "Stage-2 slice has only %d races. Blend weights will stay at "
                "their defaults; give the model more history for a real fit.",
                len(stage2))

        self.fundamental.fit(stage1, feature_names=feature_names,
                             plackett_luce_depth=plackett_luce_depth)

        blended: list[BlendedRace] = []
        for obs in stage2:
            if obs.winner_index is None:
                continue
            odds = odds_by_race.get(obs.race_id)
            if not odds or len(odds) != obs.features.shape[0]:
                continue
            market = devig(odds, self.devig_method)
            if not np.all(np.isfinite(market)):
                continue
            blended.append(BlendedRace(
                fundamental=self.fundamental.predict_proba(obs.features),
                market=market,
                winner_index=obs.winner_index,
            ))

        self.blend.fit(blended)
        return self

    def predict_proba(self, features: np.ndarray,
                      odds: Optional[Sequence[Optional[float]]] = None) -> np.ndarray:
        """Final win probabilities for one race."""
        fundamental = self.fundamental.predict_proba(features)
        if odds is None:
            return fundamental
        market = devig(odds, self.devig_method)
        if not np.all(np.isfinite(market)):
            return fundamental
        return self.blend.combine(fundamental, market)
