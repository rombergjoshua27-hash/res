"""Prediction models and market blending.

Three complementary estimators are fitted on the same leak-free feature
matrix and averaged in log-odds space:

* **Logistic regression** -- linear, naturally calibrated, hard to overfit.
* **Ridge margin model** -- predicts the scoring margin, then converts it to
  a probability through a normal CDF with the dispersion measured in
  training. Gives a point spread as a by-product.
* **Gradient boosting** -- captures interactions the linear pair cannot.

The betting market is deliberately kept *out* of the feature matrix and
applied afterwards as a probability-level blend. That keeps ``use_market``
a clean on/off switch, lets the pure model be evaluated on its own merits,
and degrades gracefully for games that have no line posted yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import FEATURE_COLUMNS

__all__ = ["GamePredictor", "spread_to_prob", "prob_to_spread", "devig_moneyline"]

SEED = 7

# Ensemble weights in log-odds space. The linear pair carries most of the
# weight because it is the better-calibrated half; the GBM adds shape.
ENSEMBLE_WEIGHTS = {"logistic": 0.40, "margin": 0.35, "gbm": 0.25}

_PROB_FLOOR = 1e-6


def _safe_logit(prob: np.ndarray | pd.Series) -> np.ndarray:
    return logit(np.clip(np.asarray(prob, dtype=float), _PROB_FLOOR, 1 - _PROB_FLOOR))


def prob_to_spread(prob_home: float | np.ndarray, sigma: float = config.MARGIN_SIGMA):
    """Invert a win probability into the point spread that implies it."""
    return norm.ppf(np.clip(prob_home, _PROB_FLOOR, 1 - _PROB_FLOOR)) * sigma


def spread_to_prob(spread: float | np.ndarray, sigma: float = config.MARGIN_SIGMA):
    """Normal-CDF conversion of a home-team point spread into a win probability."""
    return norm.cdf(np.asarray(spread, dtype=float) / sigma)


def devig_moneyline(home_ml: float, away_ml: float) -> float | None:
    """Vig-free home win probability implied by a two-way moneyline market."""
    if pd.isna(home_ml) or pd.isna(away_ml):
        return None

    def implied(odds: float) -> float:
        odds = float(odds)
        return 100.0 / (odds + 100.0) if odds > 0 else (-odds) / (-odds + 100.0)

    home_raw, away_raw = implied(home_ml), implied(away_ml)
    total = home_raw + away_raw
    if total <= 0:
        return None
    return home_raw / total  # proportional (multiplicative) de-vig


@dataclass
class FitReport:
    """What a fit actually saw -- surfaced so a backtest can be audited."""

    n_train: int = 0
    seasons: tuple = ()
    margin_sigma: float = config.MARGIN_SIGMA
    market_coef: float | None = None
    feature_names: List[str] = field(default_factory=list)
    logistic_coefs: dict = field(default_factory=dict)


class GamePredictor:
    """Fit-on-history, predict-forward NFL game model."""

    def __init__(
        self,
        *,
        use_market: bool = True,
        market_blend: float = config.DEFAULT_MARKET_BLEND,
        use_gbm: bool = True,
        seed: int = SEED,
    ) -> None:
        self.use_market = use_market
        self.market_blend = float(np.clip(market_blend, 0.0, 1.0))
        self.use_gbm = use_gbm
        self.seed = seed

        self.features: List[str] = list(FEATURE_COLUMNS)
        self._logistic: Pipeline | None = None
        self._margin: Pipeline | None = None
        self._gbm: HistGradientBoostingClassifier | None = None
        self._market_model: Pipeline | None = None
        self.report = FitReport()
        self.fitted = False

    # -- fitting ----------------------------------------------------------

    def fit(self, train: pd.DataFrame) -> "GamePredictor":
        """Fit every component on ``train`` (which must contain only past games)."""
        train = train[train["completed"]].copy()
        # Ties carry no directional signal; 15 games in 27 seasons.
        train = train[train["margin"].notna() & (train["margin"] != 0)]
        if train.empty:
            raise ValueError("no completed, non-tied games to train on")

        X = train[self.features].to_numpy(dtype=float)
        y = (train["margin"] > 0).astype(int).to_numpy()
        margin = train["margin"].to_numpy(dtype=float)

        self._logistic = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.5, max_iter=2000, solver="lbfgs", random_state=self.seed
                    ),
                ),
            ]
        ).fit(X, y)

        self._margin = Pipeline(
            [("scale", StandardScaler()), ("reg", Ridge(alpha=10.0))]
        ).fit(X, margin)

        residuals = margin - self._margin.predict(X)
        sigma = float(np.std(residuals))
        # Guard against a degenerate fit on a tiny training slice.
        self.report.margin_sigma = float(np.clip(sigma, 9.0, 18.0))

        if self.use_gbm and len(train) >= 600:
            self._gbm = HistGradientBoostingClassifier(
                max_depth=3,
                max_iter=220,
                learning_rate=0.045,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=self.seed,
            ).fit(X, y)
        else:
            self._gbm = None

        self._fit_market_model(train)

        self.report.n_train = len(train)
        self.report.seasons = tuple(sorted(train["season"].unique()))
        self.report.feature_names = list(self.features)
        self.report.logistic_coefs = dict(
            zip(self.features, self._logistic.named_steps["clf"].coef_[0].round(4))
        )
        self.fitted = True
        return self

    def _fit_market_model(self, train: pd.DataFrame) -> None:
        """Learn the empirical spread -> win-probability curve from history."""
        priced = train[train["market_spread"].notna()]
        if len(priced) < 200:
            self._market_model = None
            self.report.market_coef = None
            return
        Xm = priced[["market_spread"]].to_numpy(dtype=float)
        ym = (priced["margin"] > 0).astype(int).to_numpy()
        self._market_model = Pipeline(
            [("clf", LogisticRegression(C=1e4, max_iter=1000, solver="lbfgs"))]
        ).fit(Xm, ym)
        self.report.market_coef = float(
            self._market_model.named_steps["clf"].coef_[0][0]
        )

    # -- prediction -------------------------------------------------------

    def _component_probabilities(self, X: np.ndarray) -> dict:
        parts = {}
        parts["logistic"] = self._logistic.predict_proba(X)[:, 1]
        pred_margin = self._margin.predict(X)
        parts["margin"] = norm.cdf(pred_margin / self.report.margin_sigma)
        if self._gbm is not None:
            parts["gbm"] = self._gbm.predict_proba(X)[:, 1]
        return parts, pred_margin

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return per-game probabilities and implied spreads for ``frame``.

        Output columns:
          ``model_prob_home``   ensemble probability, market-free
          ``market_prob_home``  market-implied probability (NaN when unpriced)
          ``prob_home``         final probability actually used for picks
          ``model_margin``      ridge point-spread estimate (home perspective)
          ``pred_margin``       spread implied by ``prob_home``
        """
        if not self.fitted:
            raise RuntimeError("GamePredictor.predict called before fit")

        X = frame[self.features].to_numpy(dtype=float)
        parts, model_margin = self._component_probabilities(X)

        total_weight = sum(ENSEMBLE_WEIGHTS[name] for name in parts)
        log_odds = sum(
            ENSEMBLE_WEIGHTS[name] * _safe_logit(prob) for name, prob in parts.items()
        ) / total_weight
        model_prob = expit(log_odds)

        out = pd.DataFrame(index=frame.index)
        out["game_id"] = frame["game_id"].to_numpy()
        out["model_prob_home"] = model_prob
        out["model_margin"] = model_margin

        market_prob = self._market_probabilities(frame)
        out["market_prob_home"] = market_prob

        blended = model_prob.copy()
        if self.use_market:
            usable = ~np.isnan(market_prob)
            if usable.any():
                blended_logodds = self.market_blend * _safe_logit(model_prob[usable]) + (
                    1.0 - self.market_blend
                ) * _safe_logit(market_prob[usable])
                blended[usable] = expit(blended_logodds)

        out["prob_home"] = blended
        out["pred_margin"] = prob_to_spread(blended, self.report.margin_sigma)
        return out

    def _market_probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        """Market-implied home win probability, preferring moneyline over spread."""
        n = len(frame)
        result = np.full(n, np.nan)

        if self._market_model is not None and "market_spread" in frame.columns:
            spreads = frame["market_spread"].to_numpy(dtype=float)
            priced = ~np.isnan(spreads)
            if priced.any():
                result[priced] = self._market_model.predict_proba(
                    spreads[priced].reshape(-1, 1)
                )[:, 1]

        # A two-way moneyline is a more direct price than a spread, so it wins
        # wherever both are present.
        if {"home_moneyline", "away_moneyline"} <= set(frame.columns):
            home_ml = frame["home_moneyline"].to_numpy(dtype=float)
            away_ml = frame["away_moneyline"].to_numpy(dtype=float)
            for i in range(n):
                devigged = devig_moneyline(home_ml[i], away_ml[i])
                if devigged is not None:
                    result[i] = devigged
        return result
