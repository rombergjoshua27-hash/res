"""Derived markets: team totals, first-half lines, and the second half.

Nothing here is a new model. Once a game has a predicted total and a
predicted margin, the individual team totals follow by arithmetic, and the
first half follows by a calibration fitted against those two numbers. Adding
separate models for them would invent independence that does not exist --
a team's total is not free to disagree with the game total and the spread.

**Team totals** are exact::

    home = (total + margin) / 2
    away = (total - margin) / 2

There is no error term to fit here beyond the error already in the two
inputs. The useful thing to report is how far that lands from the truth,
which the backtest measures rather than assumes.

**First-half lines** are a fitted linear map from the full-game pair onto
the first half. Two simpler alternatives were measured over 4,645
walk-forward games and all three land within 0.05 points of each other:

    1H TOTAL   derived 7.147   direct-on-features 7.118   fixed ratio 7.096
    1H MARGIN  derived 8.385   direct-on-features 8.403   fixed ratio 8.436

The calibration is kept because it wins on margin, ties on total, and needs
one mechanism rather than two -- not because it is measurably better. A
first half is about half a game: 50.4% of the points, stable within half a
point across every six-season block since 2006.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

__all__ = ["SplitPredictor", "team_totals", "SplitFitReport"]

# Fallbacks used only when there is too little history to fit a calibration.
# EMPIRICAL over 2006-2026 (5,446 games): the first half carries 50.4% of the
# points and the OLS slope of first-half on full-game margin is 0.554.
FALLBACK_TOTAL_SHARE = 0.504
FALLBACK_MARGIN_SLOPE = 0.554
_MIN_CALIBRATION_GAMES = 200


def team_totals(
    total: float | np.ndarray, margin: float | np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a predicted total and margin into the two team totals.

    ``margin`` is from the home side, so the home team gets the half of the
    total that the margin favours.
    """
    total = np.asarray(total, dtype=float)
    margin = np.asarray(margin, dtype=float)
    return (total + margin) / 2.0, (total - margin) / 2.0


@dataclass
class SplitFitReport:
    """What the first-half calibration learned, surfaced so it can be audited."""

    n_train: int = 0
    total_slope: float = FALLBACK_TOTAL_SHARE
    total_intercept: float = 0.0
    margin_slope: float = FALLBACK_MARGIN_SLOPE
    margin_intercept: float = 0.0
    total_sigma: float = float("nan")
    margin_sigma: float = float("nan")
    fitted_from_history: bool = False
    notes: dict = field(default_factory=dict)


class SplitPredictor:
    """Turn full-game forecasts into team totals and first-half lines."""

    def __init__(self) -> None:
        self.report = SplitFitReport()
        self.fitted = False

    # -- fitting ----------------------------------------------------------

    def fit(self, train: pd.DataFrame) -> "SplitPredictor":
        """Calibrate the first half against full-game forecasts.

        ``train`` must carry ``pred_total`` and ``pred_margin`` (the
        full-game forecasts as they stood for those games) alongside the
        actual ``first_half_total`` and ``first_half_margin``.
        """
        usable = train[
            train["pred_total"].notna()
            & train["pred_margin"].notna()
            & train["first_half_total"].notna()
            & train["first_half_margin"].notna()
        ]
        self.report.n_train = len(usable)

        if len(usable) < _MIN_CALIBRATION_GAMES:
            # Not enough to fit; the measured league-wide shares still apply.
            self.report.fitted_from_history = False
            self.fitted = True
            return self

        for source, target, slope_name, intercept_name, sigma_name in (
            ("pred_total", "first_half_total", "total_slope", "total_intercept", "total_sigma"),
            ("pred_margin", "first_half_margin", "margin_slope", "margin_intercept", "margin_sigma"),
        ):
            X = usable[[source]].to_numpy(dtype=float)
            y = usable[target].to_numpy(dtype=float)
            model = LinearRegression().fit(X, y)
            setattr(self.report, slope_name, float(model.coef_[0]))
            setattr(self.report, intercept_name, float(model.intercept_))
            setattr(self.report, sigma_name, float(np.std(y - model.predict(X))))

        self.report.fitted_from_history = True
        self.fitted = True
        return self

    # -- prediction -------------------------------------------------------

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return team totals and half splits for ``frame``.

        Expects ``pred_total`` and ``pred_margin``; everything else follows.

        Output columns:
          ``home_team_total`` / ``away_team_total``   full-game team totals
          ``first_half_total_pred`` / ``first_half_margin_pred``
          ``fh_home_total`` / ``fh_away_total``       first-half team totals
          ``second_half_total_pred`` / ``second_half_margin_pred``
        """
        if not self.fitted:
            raise RuntimeError("SplitPredictor.predict called before fit")

        total = pd.to_numeric(frame["pred_total"], errors="coerce").to_numpy(dtype=float)
        margin = pd.to_numeric(frame["pred_margin"], errors="coerce").to_numpy(dtype=float)

        out = pd.DataFrame(index=frame.index)
        out["game_id"] = frame["game_id"].to_numpy()

        home, away = team_totals(total, margin)
        out["home_team_total"] = home
        out["away_team_total"] = away

        report = self.report
        if report.fitted_from_history:
            fh_total = report.total_slope * total + report.total_intercept
            fh_margin = report.margin_slope * margin + report.margin_intercept
        else:
            fh_total = FALLBACK_TOTAL_SHARE * total
            fh_margin = FALLBACK_MARGIN_SLOPE * margin

        out["first_half_total_pred"] = fh_total
        out["first_half_margin_pred"] = fh_margin
        fh_home, fh_away = team_totals(fh_total, fh_margin)
        out["fh_home_total"] = fh_home
        out["fh_away_total"] = fh_away

        # The second half is whatever the full game has left over, which keeps
        # the two halves adding up to the game rather than drifting apart.
        out["second_half_total_pred"] = total - fh_total
        out["second_half_margin_pred"] = margin - fh_margin
        return out
