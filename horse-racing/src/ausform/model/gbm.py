"""Gradient-boosted alternative to the linear conditional logit.

The conditional logit is linear in its features, which is a real
limitation: several important racing effects are not linear. Days since
last run is the clearest example -- too fresh is bad, too stale is bad, and
the optimum sits around a two-to-four week back-up. A linear term cannot
express that.

LightGBM's `lambdarank` objective learns within-race comparisons and copes
with such shapes natively. The catch is that it optimises a ranking
objective, not the softmax likelihood, so its raw scores are *not*
calibrated probabilities. We therefore convert scores to probabilities by
softmax within the race and let the temperature be fitted on held-out
data, which restores calibration.

Use this as a second opinion rather than a replacement. The linear model is
more stable on small data, easier to inspect, and less prone to fitting
noise -- and racing data is noisy. Where they disagree sharply, that is
usually worth a look rather than an automatic win for the booster.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .conditional_logit import RaceObservation

log = logging.getLogger(__name__)

try:  # pragma: no cover - exercised by import environment
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    lgb = None
    HAS_LIGHTGBM = False


@dataclass
class GbmRanker:
    """LightGBM ranker with a fitted softmax temperature."""

    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 400
    min_child_samples: int = 40
    subsample: float = 0.85
    colsample_bytree: float = 0.75
    reg_lambda: float = 5.0
    temperature: float = 1.0
    booster: Optional["lgb.Booster"] = None
    feature_names: list[str] = field(default_factory=list)
    n_races_fitted: int = 0

    def fit(self, observations: Sequence[RaceObservation],
            feature_names: Optional[list[str]] = None,
            calibration_fraction: float = 0.2) -> "GbmRanker":
        if not HAS_LIGHTGBM:
            raise RuntimeError(
                "LightGBM is not installed. Install it, or use ConditionalLogit, "
                "which has no such dependency.")

        usable = [o for o in observations
                  if o.winner_index is not None and o.features.shape[0] >= 2]
        if not usable:
            raise ValueError("No resolved races to fit on.")

        self.feature_names = feature_names or self.feature_names

        # Hold out the most recent slice to fit the temperature, so
        # calibration is never measured on rows used to grow the trees.
        split = max(1, int(len(usable) * (1 - calibration_fraction)))
        train, calib = usable[:split], usable[split:]

        matrix = np.vstack([o.features for o in train])
        groups = [o.features.shape[0] for o in train]
        # Graded relevance: winner 3, second 2, third 1, everyone else 0.
        labels = np.concatenate([self._labels(o) for o in train])

        dataset = lgb.Dataset(
            matrix, label=labels, group=groups,
            feature_name=self.feature_names or "auto", free_raw_data=False)

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3],
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_child_samples": self.min_child_samples,
            "bagging_fraction": self.subsample,
            "bagging_freq": 1,
            "feature_fraction": self.colsample_bytree,
            "lambda_l2": self.reg_lambda,
            "verbose": -1,
            "label_gain": [0, 1, 3, 7],
        }
        self.booster = lgb.train(params, dataset, num_boost_round=self.n_estimators)
        self.n_races_fitted = len(train)

        self._fit_temperature(calib or train)
        return self

    @staticmethod
    def _labels(obs: RaceObservation) -> np.ndarray:
        labels = np.zeros(obs.features.shape[0])
        if obs.finish_order:
            for rank, index in enumerate(obs.finish_order[:3]):
                labels[index] = 3 - rank
        elif obs.winner_index is not None:
            labels[obs.winner_index] = 3
        return labels

    def _fit_temperature(self, observations: Sequence[RaceObservation]) -> None:
        """Choose the softmax temperature that minimises log-loss.

        Ranking scores have arbitrary scale; without this step the implied
        probabilities are meaningless even when the ordering is good.
        """
        raw = [(self.booster.predict(o.features), o.winner_index)
               for o in observations if o.winner_index is not None]
        if not raw:
            return

        best_temp, best_loss = 1.0, float("inf")
        for temp in np.linspace(0.15, 6.0, 80):
            total, count = 0.0, 0
            for scores, winner in raw:
                shifted = scores / temp
                exp_scores = np.exp(shifted - shifted.max())
                probs = exp_scores / exp_scores.sum()
                p = max(probs[winner], 1e-12)
                total += -np.log(p)
                count += 1
            loss = total / count
            if loss < best_loss:
                best_loss, best_temp = loss, float(temp)

        self.temperature = best_temp
        log.info("GBM softmax temperature fitted to %.3f (log-loss %.4f)",
                 best_temp, best_loss)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model is not fitted.")
        if features.shape[0] == 0:
            return np.zeros(0)
        if features.shape[0] == 1:
            return np.ones(1)
        scores = np.asarray(self.booster.predict(features), dtype=float) / self.temperature
        exp_scores = np.exp(scores - scores.max())
        return exp_scores / exp_scores.sum()

    def feature_importance(self, n: int = 25) -> list[tuple[str, float]]:
        if self.booster is None:
            return []
        gains = self.booster.feature_importance(importance_type="gain")
        names = self.feature_names or [f"f{i}" for i in range(len(gains))]
        pairs = list(zip(names, (float(g) for g in gains)))
        return sorted(pairs, key=lambda kv: -kv[1])[:n]
