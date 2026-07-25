"""The win-probability model.

Structure, and why:

1. **A gradient-boosted tree** scores every runner independently. Trees handle
   the messy, non-linear, interaction-heavy structure of racing form far better
   than a linear model, and they cope with missing values natively - which
   matters enormously because racing data is full of holes.

2. **Softmax within the race** converts those independent scores into
   probabilities that sum to 1 across the field. This step is what makes it a
   racing model rather than a generic classifier: a horse's chance depends
   entirely on who else is in the race, and normalising within the race is what
   encodes that.

3. **A blend with the betting market** combines the model's opinion with the
   market's. This is the approach Bill Benter described in "Computer Based
   Horse Race Handicapping and Wagering Systems" (1994): a fundamental model,
   however good, is combined with the public odds because the market aggregates
   an enormous amount of information the model cannot see - stable gossip,
   trackwork, late scratchings, money. Benter's central finding was that the
   combined model beat either component, and that the fundamental model alone
   was not profitable.

   The blend is a weighted geometric mean:

       p_final(i) proportional to p_model(i)^alpha * p_market(i)^beta

   with alpha and beta fitted by maximum likelihood on held-out races.

The honest consequence of step 3 is worth stating plainly: this model is
largely *tracking* the market and looking for the places where it disagrees.
That is what a realistic racing model does. Anything claiming to ignore the
market and still pick 60% winners is selling something.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from .features import FEATURE_COLUMNS, feature_matrix

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:  # pragma: no cover - dependency is in requirements.txt
    HAS_LGB = False


DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 200,       # racing data is noisy; big leaves resist overfit
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbose": -1,
    "num_threads": 0,
    "seed": 42,
}


def softmax_by_race(scores: np.ndarray, race_ids: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Normalise scores into probabilities within each race."""
    scores = np.asarray(scores, dtype=float) / max(temperature, 1e-6)
    out = np.zeros_like(scores)
    order = np.argsort(race_ids, kind="mergesort")
    sorted_ids = np.asarray(race_ids)[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1], True])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        idx = order[start:end]
        block = scores[idx]
        block = np.where(np.isfinite(block), block, -30.0)
        out[idx] = np.exp(block - logsumexp(block))
    return out


def blend_probabilities(
    p_model: np.ndarray,
    p_market: np.ndarray,
    race_ids: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Benter-style geometric blend, renormalised within each race.

    Where the market probability is missing the model's own probability is used
    unchanged, so a race with no odds still produces a sensible answer.
    """
    p_model = np.clip(np.asarray(p_model, dtype=float), 1e-9, 1.0)
    p_market = np.asarray(p_market, dtype=float)
    usable = np.isfinite(p_market) & (p_market > 0)

    log_score = alpha * np.log(p_model)
    log_score[usable] += beta * np.log(np.clip(p_market[usable], 1e-9, 1.0))
    return softmax_by_race(log_score, race_ids)


def _blend_nll(params: np.ndarray, p_model, p_market, race_ids, won) -> float:
    alpha, beta = params
    blended = blend_probabilities(p_model, p_market, race_ids, alpha, beta)
    winners = won == 1
    if winners.sum() == 0:
        return 1e9
    return float(-np.mean(np.log(np.clip(blended[winners], 1e-12, 1.0))))


def fit_blend(
    p_model: np.ndarray,
    p_market: np.ndarray,
    race_ids: np.ndarray,
    won: np.ndarray,
) -> tuple[float, float]:
    """Fit the blend weights by maximum likelihood on the winners.

    A beta much larger than alpha is the normal, healthy result and means the
    market knows more than the model. Alpha near zero means the model adds
    nothing beyond the odds.
    """
    best = (1.0, 1.0)
    best_nll = np.inf
    for start in [(1.0, 1.0), (0.5, 1.5), (1.5, 0.5), (0.3, 2.0)]:
        try:
            result = minimize(
                _blend_nll, x0=np.array(start),
                args=(p_model, p_market, race_ids, won),
                method="Nelder-Mead",
                options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-6},
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best = (float(result.x[0]), float(result.x[1]))
        except Exception:
            continue
    # Negative weights are nonsense - they would mean "the more likely the
    # market thinks it is, the less likely we do".
    return max(best[0], 0.0), max(best[1], 0.0)


@dataclass
class RaceModel:
    """A trained win-probability model."""

    booster: Optional[object] = None
    features: list[str] = field(default_factory=list)
    alpha: float = 1.0
    beta: float = 1.0
    use_market: bool = True
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    n_train_races: int = 0
    n_train_runners: int = 0
    trained_through: Optional[str] = None
    metrics: dict = field(default_factory=dict)

    # -- training ---------------------------------------------------------

    def fit(
        self,
        train: pd.DataFrame,
        valid: Optional[pd.DataFrame] = None,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 100,
        feature_columns: Optional[list[str]] = None,
    ) -> "RaceModel":
        """Train on resolved races (those with a known result)."""
        if not HAS_LGB:
            raise RuntimeError("lightgbm is required: pip install lightgbm")

        train = train[train["won"].notna() & ~train["scratched"]].copy()
        if train.empty:
            raise ValueError("No resolved races to train on.")

        X, cols = feature_matrix(train, feature_columns or FEATURE_COLUMNS)
        if not self.use_market:
            cols = [c for c in cols if not c.startswith("mkt_") and c != "field_overround"]
            X = X[cols]
        self.features = cols
        y = train["won"].to_numpy(dtype=float)

        dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
        valid_sets, valid_names, callbacks = [dtrain], ["train"], []

        if valid is not None and not valid.empty:
            valid = valid[valid["won"].notna() & ~valid["scratched"]]
            if not valid.empty:
                Xv = valid[self.features].astype(float)
                dvalid = lgb.Dataset(Xv, label=valid["won"].to_numpy(dtype=float),
                                     reference=dtrain, free_raw_data=False)
                valid_sets.append(dvalid)
                valid_names.append("valid")
                callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        callbacks.append(lgb.log_evaluation(0))

        self.booster = lgb.train(
            self.params, dtrain,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets, valid_names=valid_names,
            callbacks=callbacks,
        )

        self.n_train_races = int(train["race_id"].nunique())
        self.n_train_runners = int(len(train))
        self.trained_through = str(train["race_datetime"].max())

        # Fit the market blend on the validation set where possible; falling
        # back to the training set risks the blend simply memorising the
        # training fit, so we note that in the metrics when it happens.
        blend_source = valid if (valid is not None and not valid.empty) else train
        if self.use_market and "mkt_prob" in blend_source.columns:
            raw = self._raw_model_probs(blend_source)
            market = blend_source["mkt_prob"].to_numpy(dtype=float)
            if np.isfinite(market).sum() > 50:
                self.alpha, self.beta = fit_blend(
                    raw, market,
                    blend_source["race_id"].to_numpy(),
                    blend_source["won"].to_numpy(dtype=float),
                )
                self.metrics["blend_fitted_on"] = "valid" if blend_source is valid else "train"
            else:
                self.alpha, self.beta = 1.0, 0.0
                self.metrics["blend_fitted_on"] = "none (insufficient market data)"
        else:
            self.alpha, self.beta = 1.0, 0.0
            self.metrics["blend_fitted_on"] = "none (market disabled)"

        return self

    # -- prediction -------------------------------------------------------

    def _raw_model_probs(self, frame: pd.DataFrame) -> np.ndarray:
        """Model-only win probabilities, normalised within race."""
        if self.booster is None:
            raise RuntimeError("Model is not trained.")
        X = frame.reindex(columns=self.features).astype(float)
        raw = self.booster.predict(X, raw_score=True)
        return softmax_by_race(np.asarray(raw, dtype=float), frame["race_id"].to_numpy())

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the frame with win probabilities attached.

        Adds:
          p_model  - the model's own opinion
          p_market - the de-vigged market probability
          p_win    - the blended final probability (what you should bet off)
        """
        frame = frame.copy()
        live = ~frame["scratched"].astype(bool)

        frame["p_model"] = np.nan
        frame.loc[live, "p_model"] = self._raw_model_probs(frame[live])

        market = frame["mkt_prob"].to_numpy(dtype=float) if "mkt_prob" in frame.columns \
            else np.full(len(frame), np.nan)
        frame["p_market"] = market

        if self.use_market and self.beta > 0:
            blended = np.full(len(frame), np.nan)
            sub = frame[live]
            blended[live.to_numpy()] = blend_probabilities(
                sub["p_model"].to_numpy(dtype=float),
                sub["p_market"].to_numpy(dtype=float),
                sub["race_id"].to_numpy(),
                self.alpha, self.beta,
            )
            frame["p_win"] = blended
        else:
            frame["p_win"] = frame["p_model"]

        return frame

    # -- introspection ----------------------------------------------------

    def importance(self, top: int = 30) -> pd.DataFrame:
        """Which features the model actually leans on (by information gain)."""
        if self.booster is None:
            raise RuntimeError("Model is not trained.")
        gains = self.booster.feature_importance(importance_type="gain")
        splits = self.booster.feature_importance(importance_type="split")
        table = pd.DataFrame({"feature": self.features, "gain": gains, "splits": splits})
        table["gain_pct"] = 100 * table["gain"] / max(table["gain"].sum(), 1e-9)
        return table.sort_values("gain", ascending=False).head(top).reset_index(drop=True)

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "booster": self.booster.model_to_string() if self.booster else None,
            "features": self.features,
            "alpha": self.alpha,
            "beta": self.beta,
            "use_market": self.use_market,
            "params": self.params,
            "n_train_races": self.n_train_races,
            "n_train_runners": self.n_train_runners,
            "trained_through": self.trained_through,
            "metrics": self.metrics,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        with open(path.with_suffix(".json"), "w") as fh:
            json.dump({k: v for k, v in payload.items() if k != "booster"}, fh, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "RaceModel":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        model = cls(
            features=payload["features"],
            alpha=payload["alpha"],
            beta=payload["beta"],
            use_market=payload["use_market"],
            params=payload["params"],
            n_train_races=payload.get("n_train_races", 0),
            n_train_runners=payload.get("n_train_runners", 0),
            trained_through=payload.get("trained_through"),
            metrics=payload.get("metrics", {}),
        )
        if payload["booster"]:
            model.booster = lgb.Booster(model_str=payload["booster"])
        return model
