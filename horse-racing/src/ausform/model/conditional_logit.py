"""Conditional (multinomial) logit -- the standard race model.

WHY THIS MODEL AND NOT A CLASSIFIER
-----------------------------------
The obvious approach is to train a binary classifier on "did this horse
win?". It is also the most common mistake in amateur racing models, for
reasons worth spelling out:

  * A horse race is a *competition*. What decides it is not how good a
    horse is in absolute terms but how good it is relative to the other
    runners in that specific race. A binary classifier compares runners
    across the whole dataset, so it mostly learns what a winner looks like
    on average, not who wins this race.
  * Predicted probabilities from a binary model do not sum to one within a
    race. Renormalising afterwards is a biased fix that wrecks calibration.
  * Accuracy on the runner-level binary label is meaningless. In a
    12-horse field, always predicting "loses" scores 92%. Published papers
    claiming 95%+ accuracy for race prediction are almost always measuring
    this, and it tells you nothing.

The conditional logit fixes all three by construction. Each runner gets a
strength score V_i = beta . x_i, and the win probability is a softmax over
*the runners in that race only*:

    p_i = exp(V_i) / sum_j exp(V_j)

Variable field sizes are handled automatically -- the denominator runs over
whoever is actually in the race, and probabilities sum to 1 by definition.
The gradient compares runners within a race, which is exactly the
comparison that decides the outcome.

Reference: Bolton & Chapman (1986), "Searching for Positive Returns at the
Track", Management Science 32(8); Benter (1994), "Computer Based Horse Race
Handicapping and Wagering Systems".

IDENTIFICATION NOTE
-------------------
Only *differences* in V within a race are identified. There is no
intercept, and any feature constant across a race (field size, distance,
track condition) contributes nothing on its own -- it cancels in the
softmax. Such features are only useful as interactions with runner-level
ones. The feature builder's `rel_z_*` / `rel_rank_*` columns exist for
precisely this reason.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize

log = logging.getLogger(__name__)


@dataclass
class RaceObservation:
    """One race, ready for fitting."""

    features: np.ndarray        # (n_runners, n_features)
    winner_index: Optional[int]  # row of the winner, or None if unresolved
    finish_order: Optional[list[int]] = None  # rows in finishing order
    race_id: str = ""


@dataclass
class ConditionalLogit:
    """Softmax-over-the-field model fitted by maximum likelihood."""

    l2: float = 1.0
    max_iter: int = 400
    fit_intercept: bool = False  # deliberately unused; see module docstring
    coefficients: Optional[np.ndarray] = None
    feature_names: list[str] = field(default_factory=list)
    # Standardisation, learnt on the training set and reapplied at predict.
    _mean: Optional[np.ndarray] = None
    _scale: Optional[np.ndarray] = None
    n_races_fitted: int = 0
    final_log_likelihood: Optional[float] = None

    # -- preprocessing -----------------------------------------------------

    def _standardise(self, matrix: np.ndarray, learn: bool = False) -> np.ndarray:
        """Centre and scale, and replace missing values with the mean.

        Imputing to the training mean means a missing feature contributes
        zero to the strength score after centring -- i.e. "no information"
        rather than "a low value", which is the honest default.
        """
        if learn:
            # A feature can be entirely missing across a training slice --
            # e.g. sectional times when no runner in the window has them.
            # nanmean warns on such columns; the guard below replaces them
            # with a neutral mean and unit scale, so the warning is noise.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self._mean = np.nanmean(matrix, axis=0)
                self._scale = np.nanstd(matrix, axis=0)
            self._mean = np.where(np.isfinite(self._mean), self._mean, 0.0)
            self._scale = np.where(
                np.isfinite(self._scale) & (self._scale > 1e-9), self._scale, 1.0)

        centred = (matrix - self._mean) / self._scale
        return np.where(np.isfinite(centred), centred, 0.0)

    # -- fitting -----------------------------------------------------------

    def fit(self, observations: Sequence[RaceObservation],
            feature_names: Optional[list[str]] = None,
            plackett_luce_depth: int = 1) -> "ConditionalLogit":
        """Fit by maximising the conditional log-likelihood.

        `plackett_luce_depth` > 1 uses the rank-ordered ("exploded") logit:
        after the winner is chosen from the whole field, second place is
        modelled as a softmax over the remainder, and so on. This extracts
        roughly `depth` times the training signal from the same races,
        which matters a great deal when data is scarce.

        Note the trade-off: the truncated Plackett-Luce likelihood is
        exactly the Harville assumption, which is known to overstate a
        strong favourite's chance of running second or third. It is
        excellent for estimating coefficients efficiently, but final
        probabilities should still be calibrated against win outcomes.
        """
        usable = [o for o in observations
                  if o.winner_index is not None and o.features.shape[0] >= 2]
        if not usable:
            raise ValueError("No resolved races with two or more runners to fit on.")

        self.feature_names = feature_names or self.feature_names
        n_features = usable[0].features.shape[1]

        stacked = np.vstack([o.features for o in usable])
        self._standardise(stacked, learn=True)
        design = [self._standardise(o.features) for o in usable]

        orders: list[list[int]] = []
        for obs in usable:
            if plackett_luce_depth > 1 and obs.finish_order:
                orders.append(obs.finish_order[:plackett_luce_depth])
            else:
                orders.append([obs.winner_index])

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            total = 0.0
            gradient = np.zeros_like(beta)

            for matrix, order in zip(design, orders):
                remaining = list(range(matrix.shape[0]))
                for position in order:
                    if position not in remaining or len(remaining) < 2:
                        break
                    sub = matrix[remaining]
                    scores = sub @ beta
                    peak = scores.max()
                    exp_scores = np.exp(scores - peak)
                    denominator = exp_scores.sum()
                    probs = exp_scores / denominator

                    local = remaining.index(position)
                    total += scores[local] - (peak + np.log(denominator))
                    gradient += sub[local] - probs @ sub

                    remaining.remove(position)

            # L2 penalty. Regularisation is essential here: racing features
            # are heavily collinear (weight, class and prize money all
            # proxy the same thing) and unpenalised coefficients swing
            # wildly between refits.
            penalty = self.l2 * float(beta @ beta)
            total -= penalty
            gradient -= 2.0 * self.l2 * beta

            # scipy minimises, so return the negatives.
            return -total, -gradient

        initial = np.zeros(n_features)
        result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                          options={"maxiter": self.max_iter})

        self.coefficients = result.x
        self.n_races_fitted = len(usable)
        self.final_log_likelihood = -result.fun
        if not result.success:
            log.warning("Conditional logit did not fully converge: %s", result.message)
        return self

    # -- prediction --------------------------------------------------------

    def strengths(self, features: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Model is not fitted.")
        return self._standardise(features) @ self.coefficients

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Win probabilities for one race. Sums to 1 by construction."""
        if features.shape[0] == 0:
            return np.zeros(0)
        if features.shape[0] == 1:
            return np.ones(1)
        scores = self.strengths(features)
        exp_scores = np.exp(scores - scores.max())
        return exp_scores / exp_scores.sum()

    # -- inspection --------------------------------------------------------

    def top_coefficients(self, n: int = 20) -> list[tuple[str, float]]:
        """Largest-magnitude coefficients, for sanity-checking what the
        model has actually latched onto."""
        if self.coefficients is None:
            return []
        names = self.feature_names or [f"f{i}" for i in range(len(self.coefficients))]
        pairs = list(zip(names, self.coefficients))
        return sorted(pairs, key=lambda kv: -abs(kv[1]))[:n]


def observations_from_featuresets(feature_sets, min_runners: int = 4
                                  ) -> list[RaceObservation]:
    """Convert FeatureSet objects into fitting observations.

    Races without a recorded winner are dropped -- they carry no training
    signal for a win model.
    """
    observations: list[RaceObservation] = []
    for fs in feature_sets:
        if fs.n_runners < min_runners:
            continue
        positions = [r.result_position for r in fs.runners]
        if not any(p == 1 for p in positions if p is not None):
            continue

        winner_index = next(i for i, p in enumerate(positions) if p == 1)
        resolved = [(i, p) for i, p in enumerate(positions) if p is not None]
        order = [i for i, _ in sorted(resolved, key=lambda ip: ip[1])]

        observations.append(RaceObservation(
            features=fs.matrix,
            winner_index=winner_index,
            finish_order=order,
            race_id=fs.race.race_id,
        ))
    return observations
