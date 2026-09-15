"""Point-total prediction.

The third leg of the model, alongside the winner and the spread. It answers
a different question from the other two, so it is built differently: a total
does not care which team is better. Two efficient offences and two leaky
defences both push the number up, so every input is a *sum* across the two
sides rather than a difference.

Structure mirrors ``model.GamePredictor`` deliberately:

* **Ridge regression** -- linear, stable, hard to overfit on ~5,000 games.
* **Gradient boosting** -- picks up the interactions the linear half misses,
  notably wind against pass-rate, where the penalty is not additive.

The two are averaged in points, the residual spread is measured on the
training set, and the posted total is blended in afterwards at the points
level -- never as a feature -- so ``use_market=False`` isolates what the
model knows on its own.

A word on what is achievable. Over 2007-2026 the closing total missed the
actual by 10.5 points on average, against a standard deviation of 14.0 in
the totals themselves. Most of a football game's scoring is genuinely
unpredictable; a model that matches the market here is doing well, and one
claiming to beat it by a touchdown is fitting noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import TOTAL_FEATURE_COLUMNS

__all__ = ["TotalsPredictor", "over_probability", "TotalsFitReport"]

SEED = 7

# Weights on the two estimators, in points. The ridge carries more because
# it extrapolates sanely into weather and pace combinations the boosted
# trees have never seen together.
TOTALS_ENSEMBLE_WEIGHTS = {"ridge": 0.60, "gbm": 0.40}


def over_probability(
    predicted_total: float | np.ndarray,
    line: float | np.ndarray,
    sigma: float = config.TOTAL_SIGMA,
) -> np.ndarray:
    """Probability the game goes over ``line`` given a predicted total.

    Scores are integers, so a whole-number line needs a continuity
    correction: to go *over* a posted 44 the game must reach 45, while a
    posted 44.5 only has to reach 45 as well -- but 44 exactly is a push on
    the first and a loss on the second. Half-point lines cannot push and are
    left alone.
    """
    predicted = np.asarray(predicted_total, dtype=float)
    line = np.asarray(line, dtype=float)
    sigma = max(float(sigma), 1e-6)

    is_whole = np.isclose(line % 1.0, 0.0)
    threshold = np.where(is_whole, line + 0.5, line)
    return 1.0 - norm.cdf((threshold - predicted) / sigma)


def push_probability(
    predicted_total: float | np.ndarray,
    line: float | np.ndarray,
    sigma: float = config.TOTAL_SIGMA,
) -> np.ndarray:
    """Probability the game lands exactly on a whole-number total (a push)."""
    predicted = np.asarray(predicted_total, dtype=float)
    line = np.asarray(line, dtype=float)
    sigma = max(float(sigma), 1e-6)

    is_whole = np.isclose(line % 1.0, 0.0)
    upper = norm.cdf((line + 0.5 - predicted) / sigma)
    lower = norm.cdf((line - 0.5 - predicted) / sigma)
    return np.where(is_whole, upper - lower, 0.0)


@dataclass
class TotalsFitReport:
    """What a totals fit actually saw -- surfaced so a backtest can be audited."""

    n_train: int = 0
    seasons: tuple = ()
    sigma: float = config.TOTAL_SIGMA
    train_mae: float = float("nan")
    feature_names: List[str] = field(default_factory=list)
    ridge_coefs: dict = field(default_factory=dict)


class TotalsPredictor:
    """Fit-on-history, predict-forward NFL point-total model."""

    def __init__(
        self,
        *,
        use_market: bool = True,
        market_blend: float = config.DEFAULT_TOTAL_MARKET_BLEND,
        use_gbm: bool = True,
        seed: int = SEED,
    ) -> None:
        self.use_market = use_market
        self.market_blend = float(np.clip(market_blend, 0.0, 1.0))
        self.use_gbm = use_gbm
        self.seed = seed

        self.features: List[str] = list(TOTAL_FEATURE_COLUMNS)
        self._ridge: Pipeline | None = None
        self._gbm: HistGradientBoostingRegressor | None = None
        self.report = TotalsFitReport()
        self.fitted = False

    # -- fitting ----------------------------------------------------------

    @staticmethod
    def actual_total(frame: pd.DataFrame) -> pd.Series:
        """Points actually scored by both teams, or NaN before kickoff."""
        return (
            pd.to_numeric(frame["home_score"], errors="coerce")
            + pd.to_numeric(frame["away_score"], errors="coerce")
        )

    def fit(self, train: pd.DataFrame) -> "TotalsPredictor":
        """Fit on ``train``, which must contain only already-completed games."""
        train = train[train["completed"]].copy()
        train["_total"] = self.actual_total(train)
        train = train[train["_total"].notna()]
        if train.empty:
            raise ValueError("no completed games with a final score to train on")

        X = train[self.features].to_numpy(dtype=float)
        y = train["_total"].to_numpy(dtype=float)

        self._ridge = Pipeline(
            [("scale", StandardScaler()), ("reg", Ridge(alpha=10.0))]
        ).fit(X, y)

        if self.use_gbm and len(train) >= 600:
            self._gbm = HistGradientBoostingRegressor(
                max_depth=3,
                max_iter=220,
                learning_rate=0.045,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=self.seed,
            ).fit(X, y)
        else:
            self._gbm = None

        in_sample = self._ensemble_total(X)
        residuals = y - in_sample
        # Guard against a degenerate fit on a tiny training slice.
        self.report.sigma = float(np.clip(np.std(residuals), 9.0, 20.0))
        self.report.train_mae = float(np.mean(np.abs(residuals)))
        self.report.n_train = len(train)
        self.report.seasons = tuple(sorted(train["season"].unique()))
        self.report.feature_names = list(self.features)
        self.report.ridge_coefs = dict(
            zip(self.features, self._ridge.named_steps["reg"].coef_.round(4))
        )
        self.fitted = True
        return self

    # -- prediction -------------------------------------------------------

    def _ensemble_total(self, X: np.ndarray) -> np.ndarray:
        parts = {"ridge": self._ridge.predict(X)}
        if self._gbm is not None:
            parts["gbm"] = self._gbm.predict(X)
        weight = sum(TOTALS_ENSEMBLE_WEIGHTS[name] for name in parts)
        return sum(
            TOTALS_ENSEMBLE_WEIGHTS[name] * values for name, values in parts.items()
        ) / weight

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return per-game total forecasts for ``frame``.

        Output columns:
          ``model_total``   ensemble prediction, market-free
          ``market_total``  the posted total (NaN when none is up yet)
          ``pred_total``    the number actually used, after any market blend
          ``prob_over``     probability of going over the posted total
          ``prob_push``     probability of landing exactly on a whole number
          ``model_edge``    ``model_total`` minus the posted total
          ``total_edge``    ``pred_total`` minus the posted total
          ``prob_over_model``  over probability from the model's own number
          ``ou_pick``       OVER / UNDER / "-" when there is no line
        """
        if not self.fitted:
            raise RuntimeError("TotalsPredictor.predict called before fit")

        X = frame[self.features].to_numpy(dtype=float)
        model_total = self._ensemble_total(X)

        market = (
            pd.to_numeric(frame["market_total"], errors="coerce").to_numpy(dtype=float)
            if "market_total" in frame.columns
            else np.full(len(frame), np.nan)
        )

        blended = model_total.copy()
        if self.use_market:
            priced = ~np.isnan(market)
            blended[priced] = (
                self.market_blend * model_total[priced]
                + (1.0 - self.market_blend) * market[priced]
            )

        out = pd.DataFrame(index=frame.index)
        out["game_id"] = frame["game_id"].to_numpy()
        out["model_total"] = model_total
        out["market_total"] = market
        out["pred_total"] = blended
        # Two edges, because they answer different questions. ``model_edge``
        # is how far the model's own number sits from the posted one -- the
        # size of the disagreement. ``total_edge`` is what survives the
        # blend, which at the default weight is a tenth of it. Reporting only
        # the second would hide the disagreement; reporting only the first
        # would overstate what the model is actually willing to act on.
        out["model_edge"] = model_total - market
        out["total_edge"] = blended - market
        out["prob_over"] = over_probability(blended, market, self.report.sigma)
        out["prob_over_model"] = over_probability(
            model_total, market, self.report.sigma
        )
        out["prob_push"] = push_probability(blended, market, self.report.sigma)
        out["ou_pick"] = np.where(
            np.isnan(market), "-", np.where(blended > market, "OVER", "UNDER")
        )
        return out
