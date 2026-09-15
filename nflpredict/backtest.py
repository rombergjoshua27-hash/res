"""Walk-forward backtesting and evaluation metrics.

The only defensible way to state an accuracy figure for a game model is to
replay history: for each week, fit on everything that had already finished
and predict the week that had not. That is what ``walk_forward`` does. No
fit ever sees a game from its own week or any later week.

Elo and rolling-form features do not need re-simulation per week -- they are
built by a single chronological pass that, by construction, only ever reads
prior games (see ``features``). Only the supervised models are refit.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from . import config
from .model import GamePredictor
from .totals import TotalsPredictor

__all__ = [
    "walk_forward",
    "evaluate",
    "evaluate_totals",
    "calibration_table",
    "ats_record",
    "ou_record",
    "BacktestResult",
]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _clip(prob: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(prob, dtype=float), 1e-9, 1 - 1e-9)


def evaluate(frame: pd.DataFrame, prob_col: str = "prob_home") -> Dict[str, float]:
    """Accuracy, Brier score and log loss for a frame of scored predictions.

    Ties are credited as half a win to both sides rather than discarded.
    """
    scored = frame[frame["home_win"].notna() & frame[prob_col].notna()]
    if scored.empty:
        return {"n": 0}

    prob = _clip(scored[prob_col])
    actual = scored["home_win"].to_numpy(dtype=float)

    picked_home = prob > 0.5
    correct = np.where(
        actual == config.TIE_CREDIT,
        config.TIE_CREDIT,
        np.where(picked_home, actual, 1.0 - actual),
    )

    return {
        "n": int(len(scored)),
        "accuracy": float(np.mean(correct)),
        "brier": float(np.mean((prob - actual) ** 2)),
        "log_loss": float(
            -np.mean(actual * np.log(prob) + (1 - actual) * np.log(1 - prob))
        ),
        "mean_prob": float(np.mean(np.maximum(prob, 1 - prob))),
    }


def calibration_table(
    frame: pd.DataFrame,
    prob_col: str = "prob_home",
    bins: Iterable[float] = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0),
) -> pd.DataFrame:
    """Predicted vs. realised win rate, bucketed by model confidence.

    Probabilities are folded to the favourite's side so both directions of a
    lopsided game land in the same bucket.
    """
    scored = frame[frame["home_win"].notna() & frame[prob_col].notna()].copy()
    if scored.empty:
        return pd.DataFrame()

    prob = _clip(scored[prob_col])
    scored["confidence"] = np.maximum(prob, 1 - prob)
    scored["hit"] = np.where(
        scored["home_win"] == config.TIE_CREDIT,
        config.TIE_CREDIT,
        np.where(prob > 0.5, scored["home_win"], 1.0 - scored["home_win"]),
    )
    scored["bucket"] = pd.cut(scored["confidence"], list(bins), include_lowest=True)

    table = (
        scored.groupby("bucket", observed=True)
        .agg(
            games=("hit", "size"),
            predicted=("confidence", "mean"),
            actual=("hit", "mean"),
        )
        .reset_index()
    )
    table["gap"] = table["actual"] - table["predicted"]
    return table


def ats_record(
    frame: pd.DataFrame,
    *,
    margin_col: str = "pred_margin",
    threshold: float = 0.0,
    odds: int = config.STANDARD_VIG_ODDS,
) -> Dict[str, float]:
    """Against-the-spread record for picks whose edge clears ``threshold``.

    A bet is placed only where the model's predicted margin differs from the
    posted spread by at least ``threshold`` points.
    """
    playable = frame[
        frame["home_win"].notna()
        & frame["market_spread"].notna()
        & frame[margin_col].notna()
    ].copy()
    if playable.empty:
        return {"bets": 0}

    playable["edge"] = playable[margin_col] - playable["market_spread"]
    playable = playable[playable["edge"].abs() >= threshold]
    if playable.empty:
        return {"bets": 0, "threshold": threshold}

    # Positive edge => model likes the home side relative to the price.
    took_home = playable["edge"] > 0
    cover_margin = playable["margin"] - playable["market_spread"]
    result = np.where(took_home, cover_margin, -cover_margin)

    wins = int(np.sum(result > 0))
    losses = int(np.sum(result < 0))
    pushes = int(np.sum(result == 0))
    decided = wins + losses

    payout = 100.0 / abs(odds) if odds < 0 else odds / 100.0
    profit = wins * payout - losses

    return {
        "bets": int(len(playable)),
        "threshold": float(threshold),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": float(wins / decided) if decided else float("nan"),
        "roi": float(profit / decided) if decided else float("nan"),
        "units": float(profit),
        "breakeven": float(1.0 / (1.0 + payout)),
    }


def evaluate_totals(frame: pd.DataFrame, total_col: str = "pred_total") -> Dict[str, float]:
    """Mean absolute error, RMSE and bias for a frame of total forecasts.

    Totals are scored as a regression rather than a classification: the
    useful question is how many points the forecast missed by, not whether
    it landed on the right side of a line.
    """
    scored = frame[frame["actual_total"].notna() & frame[total_col].notna()]
    if scored.empty:
        return {"n": 0}

    error = scored["actual_total"].to_numpy(dtype=float) - scored[total_col].to_numpy(
        dtype=float
    )
    return {
        "n": int(len(scored)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "mean_total": float(np.mean(scored[total_col])),
    }


def ou_record(
    frame: pd.DataFrame,
    *,
    total_col: str = "model_total",
    threshold: float = 0.0,
    odds: int = config.STANDARD_VIG_ODDS,
) -> Dict[str, float]:
    """Over/under record for picks whose disagreement with the line clears
    ``threshold`` points.

    The edge is measured against the model's *own* number rather than the
    blended one, because the blend is mostly the line itself -- measuring
    against it would report the line's disagreement with itself.
    """
    playable = frame[
        frame["actual_total"].notna()
        & frame["market_total"].notna()
        & frame[total_col].notna()
    ].copy()
    if playable.empty:
        return {"bets": 0, "threshold": threshold}

    playable["edge"] = playable[total_col] - playable["market_total"]
    playable = playable[playable["edge"].abs() >= threshold]
    if playable.empty:
        return {"bets": 0, "threshold": threshold}

    took_over = playable["edge"] > 0
    diff = playable["actual_total"] - playable["market_total"]
    result = np.where(took_over, diff, -diff)

    wins = int(np.sum(result > 0))
    losses = int(np.sum(result < 0))
    pushes = int(np.sum(result == 0))
    decided = wins + losses

    payout = 100.0 / abs(odds) if odds < 0 else odds / 100.0
    profit = wins * payout - losses

    return {
        "bets": int(len(playable)),
        "threshold": float(threshold),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": float(wins / decided) if decided else float("nan"),
        "roi": float(profit / decided) if decided else float("nan"),
        "units": float(profit),
        "breakeven": float(1.0 / (1.0 + payout)),
    }


# --------------------------------------------------------------------------
# Walk-forward driver
# --------------------------------------------------------------------------


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    summary: Dict[str, Dict[str, float]]
    calibration: pd.DataFrame
    by_season: pd.DataFrame
    ats: List[Dict[str, float]]
    totals: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ou: List[Dict[str, float]] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - display helper
        model = self.summary.get("model", {})
        return (
            f"<BacktestResult n={model.get('n', 0)} "
            f"acc={model.get('accuracy', float('nan')):.4f}>"
        )


def walk_forward(
    features: pd.DataFrame,
    *,
    start_season: int = config.DEFAULT_BACKTEST_START,
    end_season: int | None = None,
    use_market: bool = True,
    market_blend: float = config.DEFAULT_MARKET_BLEND,
    total_blend: float = config.DEFAULT_TOTAL_MARKET_BLEND,
    use_qb_features: bool = config.USE_QB_FEATURES_DEFAULT,
    refit: str = "week",
    min_train_games: int = 600,
    quiet: bool = False,
) -> BacktestResult:
    """Replay history week by week, fitting only on already-completed games.

    ``refit='week'`` refits before every slate (strictest, slowest);
    ``refit='season'`` refits once per season, which is ~17x cheaper and in
    practice changes accuracy by well under a tenth of a point.
    """
    if refit not in {"week", "season"}:
        raise ValueError("refit must be 'week' or 'season'")

    frame = features.copy()
    frame["period"] = frame["season"] * 100 + frame["week"]

    end_season = end_season or int(frame["season"].max())
    target = frame[
        (frame["season"] >= start_season)
        & (frame["season"] <= end_season)
        & frame["completed"]
    ]
    if target.empty:
        raise ValueError(f"no completed games in {start_season}-{end_season}")

    slates = sorted(target["period"].unique())
    predictions: List[pd.DataFrame] = []
    predictor: GamePredictor | None = None
    totals_predictor: TotalsPredictor | None = None
    fitted_for_season: int | None = None
    fits = 0

    for index, period in enumerate(slates):
        season = period // 100
        history = frame[(frame["period"] < period) & frame["completed"]]
        if len(history) < min_train_games:
            continue

        need_refit = refit == "week" or fitted_for_season != season
        if need_refit or predictor is None:
            predictor = GamePredictor(
                use_market=use_market,
                market_blend=market_blend,
                use_qb_features=use_qb_features,
            ).fit(history)
            totals_predictor = TotalsPredictor(
                use_market=use_market, market_blend=total_blend
            ).fit(history)
            fitted_for_season = season
            fits += 1

        slate = frame[(frame["period"] == period) & frame["completed"]]
        preds = predictor.predict(slate)
        total_preds = totals_predictor.predict(slate)

        carry = [
            "game_id", "season", "week", "home_team", "away_team", "margin",
            "home_win", "market_spread", "elo_prob_home", "elo_diff",
            "home_moneyline", "away_moneyline",
        ]
        merged = slate[[c for c in carry if c in slate.columns]].merge(
            preds.drop(columns=["game_id"]).assign(game_id=preds["game_id"].values),
            on="game_id",
            how="left",
        ).merge(
            total_preds.drop(columns=["game_id"]).assign(
                game_id=total_preds["game_id"].values
            ),
            on="game_id",
            how="left",
        )
        merged["actual_total"] = TotalsPredictor.actual_total(slate).to_numpy()
        merged["n_train"] = predictor.report.n_train
        predictions.append(merged)

        if not quiet and index % 25 == 0:
            print(
                f"  .. {season} wk {period % 100:>2}  "
                f"(train={len(history):,}, fits={fits})",
                file=sys.stderr,
            )

    if not predictions:
        raise ValueError("no slate had enough training history to score")

    scored = pd.concat(predictions, ignore_index=True)

    summary = {
        "model": evaluate(scored, "model_prob_home"),
        "blended": evaluate(scored, "prob_home"),
        "elo_only": evaluate(scored, "elo_prob_home"),
        "market": evaluate(scored, "market_prob_home"),
    }
    summary["home_team_always"] = {
        "n": int(scored["home_win"].notna().sum()),
        "accuracy": float(scored["home_win"].mean()),
    }

    by_season = (
        scored.assign(
            hit=np.where(
                scored["home_win"] == config.TIE_CREDIT,
                config.TIE_CREDIT,
                np.where(
                    scored["prob_home"] > 0.5,
                    scored["home_win"],
                    1.0 - scored["home_win"],
                ),
            )
        )
        .groupby("season")
        .agg(
            games=("hit", "size"),
            accuracy=("hit", "mean"),
            brier=("prob_home", lambda p: float("nan")),
        )
        .reset_index()
    )
    # Brier needs both columns, so compute it separately per season.
    briers = (
        scored.groupby("season")
        .apply(
            lambda g: float(np.mean((_clip(g["prob_home"]) - g["home_win"]) ** 2)),
            include_groups=False,
        )
        .rename("brier")
    )
    by_season = by_season.drop(columns=["brier"]).merge(
        briers.reset_index(), on="season", how="left"
    )

    ats = [
        ats_record(scored, threshold=t) for t in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
    ]

    totals = {
        "blended": evaluate_totals(scored, "pred_total"),
        "model": evaluate_totals(scored, "model_total"),
        "market": evaluate_totals(scored, "market_total"),
    }
    ou = [ou_record(scored, threshold=t) for t in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)]

    return BacktestResult(
        predictions=scored,
        summary=summary,
        calibration=calibration_table(scored, "prob_home"),
        by_season=by_season,
        ats=ats,
        totals=totals,
        ou=ou,
    )
