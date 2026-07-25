"""Walk-forward backtesting.

The only defensible way to evaluate a racing model. For each evaluation
window, the model is trained exclusively on races that finished before the
window opened, then predicts the window, and is never refitted on it.

WHY NOT ORDINARY CROSS-VALIDATION
---------------------------------
Random k-fold splits leak catastrophically here, in four distinct ways:

  * Same-race leakage. A random row split puts some runners from a race in
    training and the rest in test. The label is a within-race competition,
    so you have literally shown the model the answer.
  * Same-horse leakage. A horse's form in race t+1 is a function of race t.
    Random splits let the model see a horse's future.
  * Regime leakage. Track speed, class structures, prize money and market
    efficiency all drift. Random CV averages over a future you do not have.
  * Feature leakage. Any statistic computed over the whole dataset -- a
    sire's wet-track rate, a track's average winning time -- injects the
    future into every fold. This is why `RollingContext` exists and why it
    refuses out-of-order races.

An embargo gap is also applied between train and test, because features
look back over recent form and a hard boundary alone does not prevent a
training race from informing a test race a day later.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from ..betting.odds import Method, devig
from ..features import RollingContext, build_features
from ..model import (
    ConditionalLogit,
    RaceObservation,
    TwoStageModel,
    observations_from_featuresets,
)
from ..types import Race
from .metrics import RacePrediction, full_report

log = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Backtest schedule and model settings."""

    train_days: int = 365          # history used for each fit
    test_days: int = 30            # window predicted before refitting
    embargo_days: int = 1          # gap between train end and test start
    min_train_races: int = 500
    expanding: bool = True         # grow the training window over time
    l2: float = 1.0
    devig_method: Method = "shin"
    use_market_blend: bool = True
    plackett_luce_depth: int = 3
    include_market_features: bool = False


@dataclass
class BacktestResult:
    predictions: list[RacePrediction]
    windows: int
    config: WalkForwardConfig
    fitted_blends: list[tuple[float, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def report(self, commission: float = 0.0) -> dict:
        result = full_report(self.predictions, commission=commission)
        result["windows"] = self.windows
        if self.fitted_blends:
            alphas = [a for a, _ in self.fitted_blends]
            betas = [b for _, b in self.fitted_blends]
            result["blend"] = {
                "mean_alpha_model_weight": float(np.mean(alphas)),
                "mean_beta_market_weight": float(np.mean(betas)),
            }
        if self.notes:
            result["notes"] = self.notes
        return result


class WalkForwardBacktest:
    """Runs the train/predict cycle across a chronological race list."""

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()

    def run(
        self,
        races: Sequence[Race],
        progress: Optional[Callable[[str], None]] = None,
    ) -> BacktestResult:
        # Sort by scheduled start time where it is known, not just by date.
        # Sorting on date alone leaves same-day races in arbitrary order, so
        # roughly half of them would have their features built after a
        # later-running race had already been folded into the context. The
        # measured effect is tiny, but it quietly breaks the point-in-time
        # guarantee the whole design rests on.
        ordered = sorted(
            races,
            key=lambda r: (r.date,
                           r.start_time.time() if r.start_time else _dt.time.min,
                           r.race_number,
                           r.race_id))
        if not ordered:
            raise ValueError("No races supplied.")

        config = self.config
        notes: list[str] = []

        # Features are built once, in strict date order, against a single
        # rolling context. This is what guarantees every feature value is
        # point-in-time regardless of how the windows are later carved up.
        context = RollingContext()
        feature_sets = []
        for race in ordered:
            feature_sets.append(
                build_features(race, context,
                               include_market=config.include_market_features))
            context.observe(race)

        by_date: dict[_dt.date, list[int]] = {}
        for index, race in enumerate(ordered):
            by_date.setdefault(race.date, []).append(index)

        start_date = ordered[0].date
        end_date = ordered[-1].date

        predictions: list[RacePrediction] = []
        blends: list[tuple[float, float]] = []
        windows = 0
        failed_windows = 0
        unfitted_blends = 0

        test_start = start_date + _dt.timedelta(days=config.train_days)
        while test_start <= end_date:
            test_end = test_start + _dt.timedelta(days=config.test_days)
            train_end = test_start - _dt.timedelta(days=config.embargo_days)
            train_start = (start_date if config.expanding
                           else train_end - _dt.timedelta(days=config.train_days))

            train_idx = [i for i, r in enumerate(ordered)
                         if train_start <= r.date < train_end]
            test_idx = [i for i, r in enumerate(ordered)
                        if test_start <= r.date < test_end]

            if len(train_idx) < config.min_train_races or not test_idx:
                test_start = test_end
                continue

            observations = observations_from_featuresets(
                [feature_sets[i] for i in train_idx])
            if len(observations) < config.min_train_races:
                test_start = test_end
                continue

            model = self._fit(observations, [ordered[i] for i in train_idx],
                              feature_sets[train_idx[0]].names)
            if model is None:
                # A window whose fit failed is not the same as a window that
                # never ran. Record it, or a run where most windows blew up
                # is indistinguishable from a clean one.
                failed_windows += 1
                notes.append(f"Model fit failed for the window starting {test_start}.")
                test_start = test_end
                continue

            if isinstance(model, TwoStageModel):
                blends.append((model.blend.alpha, model.blend.beta))
                if not model.blend.fitted:
                    unfitted_blends += 1

            for i in test_idx:
                prediction = self._predict(model, feature_sets[i], ordered[i])
                if prediction is not None:
                    predictions.append(prediction)

            windows += 1
            if progress:
                progress(f"window {windows}: trained on {len(observations)} races, "
                         f"predicted {test_start} to {test_end}")
            test_start = test_end

        if windows == 0:
            notes.append(
                "No walk-forward window had enough training data. Either supply "
                "more history or lower train_days / min_train_races.")
        if failed_windows:
            notes.append(
                f"{failed_windows} of {windows + failed_windows} windows failed to "
                f"fit and were skipped; the reported metrics cover only the rest.")
        if unfitted_blends:
            notes.append(
                f"{unfitted_blends} of {windows} windows had too little data to fit "
                f"the market blend and fell back to default weights "
                f"(alpha=0.25, beta=0.90). Those blend numbers were never "
                f"estimated from data -- do not read anything into them.")

        return BacktestResult(predictions=predictions, windows=windows,
                              config=config, fitted_blends=blends, notes=notes)

    # -- internals ---------------------------------------------------------

    def _fit(self, observations: list[RaceObservation], races: list[Race],
             feature_names: list[str]):
        config = self.config
        try:
            if config.use_market_blend:
                odds_by_race = {
                    race.race_id: [r.fixed_win_odds or r.tote_win_odds
                                   for r in race.active_runners]
                    for race in races
                }
                model = TwoStageModel.create(l2=config.l2,
                                             devig_method=config.devig_method)
                model.fit(observations, odds_by_race, feature_names=feature_names,
                          plackett_luce_depth=config.plackett_luce_depth)
                return model

            model = ConditionalLogit(l2=config.l2)
            model.fit(observations, feature_names=feature_names,
                      plackett_luce_depth=config.plackett_luce_depth)
            return model
        except (ValueError, RuntimeError) as exc:
            log.warning("Model fit failed for a window: %s", exc)
            return None

    def _predict(self, model, feature_set, race: Race) -> Optional[RacePrediction]:
        runners = feature_set.runners
        if len(runners) < 2:
            return None

        odds = [r.fixed_win_odds or r.tote_win_odds for r in runners]
        has_odds = all(o and o > 1.0 for o in odds)

        if isinstance(model, TwoStageModel):
            probs = model.predict_proba(feature_set.matrix,
                                        odds if has_odds else None)
        else:
            probs = model.predict_proba(feature_set.matrix)

        market = devig(odds, self.config.devig_method) if has_odds else None

        winner_index = None
        positions = [r.result_position for r in runners]
        for index, position in enumerate(positions):
            if position == 1:
                winner_index = index
                break

        resolved = [(i, p) for i, p in enumerate(positions) if p is not None]
        order = [i for i, _ in sorted(resolved, key=lambda ip: ip[1])] or None

        return RacePrediction(
            race_id=race.race_id,
            date=race.date.isoformat(),
            probabilities=np.asarray(probs, dtype=float),
            market_probabilities=market,
            odds=np.array([o if o else np.nan for o in odds], dtype=float),
            winner_index=winner_index,
            runner_names=[r.name for r in runners],
            finish_order=order,
            field_size=len(runners),
        )
