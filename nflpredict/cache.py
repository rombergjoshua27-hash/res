"""Backtest caching.

Replaying nineteen seasons takes minutes, which is fine once and intolerable
every time somebody wants this week's slate. The backtest is therefore
computed once and cached, and the cache is keyed on everything that could
change its answer:

* how many games have been played (a new result invalidates it),
* the seasons requested and the blend weights, and
* whether the passer terms were in the fit.

A cache that missed any of those would quietly serve last week's accuracy
figures next to this week's picks, which is worse than being slow. If any
key differs the backtest is rerun; nothing is repaired in place.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from . import config
from .backtest import BacktestResult, ats_record, calibration_table, evaluate
from .backtest import evaluate_totals, ou_record
from .backtest import _clip  # noqa: F401 - re-exported for by-season rebuild

__all__ = ["cached_walk_forward", "cache_key", "clear_cache"]

CACHE_VERSION = 1


def _cache_dir() -> Path:
    return config.DATA_DIR / "backtest"


@dataclass(frozen=True)
class _Key:
    """Everything that would change the backtest's answer."""

    version: int
    completed_games: int
    last_game_id: str
    start_season: int
    end_season: int
    use_market: bool
    market_blend: float
    total_blend: float
    use_qb_features: bool
    refit: str

    def digest(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_key(features: pd.DataFrame, **options: Any) -> _Key:
    """Build the cache key for a backtest over ``features``."""
    completed = features[features["completed"]]
    last_game = str(completed["game_id"].iloc[-1]) if len(completed) else ""
    end_season = options.get("end_season") or int(features["season"].max())
    return _Key(
        version=CACHE_VERSION,
        completed_games=int(len(completed)),
        last_game_id=last_game,
        start_season=int(options.get("start_season", config.DEFAULT_BACKTEST_START)),
        end_season=int(end_season),
        use_market=bool(options.get("use_market", True)),
        market_blend=round(float(options.get("market_blend", config.DEFAULT_MARKET_BLEND)), 4),
        total_blend=round(float(options.get("total_blend", config.DEFAULT_TOTAL_MARKET_BLEND)), 4),
        use_qb_features=bool(options.get("use_qb_features", config.USE_QB_FEATURES_DEFAULT)),
        refit=str(options.get("refit", "season")),
    )


def cached_walk_forward(features: pd.DataFrame, *, quiet: bool = True, **options):
    """``walk_forward``, reusing a cached run whose inputs match exactly.

    Only the per-game predictions are stored. Every summary -- accuracy,
    calibration, the ATS and over/under records -- is recomputed from them on
    load, so a change to how a metric is defined can never be masked by a
    stale cached number.
    """
    from .backtest import walk_forward

    key = cache_key(features, **options)
    path = _cache_dir() / f"predictions_{key.digest()}.csv.gz"

    if path.exists():
        try:
            predictions = pd.read_csv(path, low_memory=False)
            if not predictions.empty:
                return _rebuild(predictions), True
        except Exception:  # noqa: BLE001 - a corrupt cache is just a miss
            path.unlink(missing_ok=True)

    result = walk_forward(features, quiet=quiet, **options)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(path, index=False, compression="gzip")
    _prune(path)
    return result, False


def _prune(keep: Path, *, limit: int = 4) -> None:
    """Keep the cache small; a new result each week would otherwise pile up."""
    files = sorted(
        _cache_dir().glob("predictions_*.csv.gz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in files[limit:]:
        if stale != keep:
            stale.unlink(missing_ok=True)


def _rebuild(predictions: pd.DataFrame) -> BacktestResult:
    """Recompute every summary from the cached per-game rows."""
    import numpy as np

    summary: Dict[str, Dict[str, float]] = {
        "model": evaluate(predictions, "model_prob_home"),
        "blended": evaluate(predictions, "prob_home"),
        "elo_only": evaluate(predictions, "elo_prob_home"),
        "market": evaluate(predictions, "market_prob_home"),
    }
    summary["home_team_always"] = {
        "n": int(predictions["home_win"].notna().sum()),
        "accuracy": float(predictions["home_win"].mean()),
    }

    hit = np.where(
        predictions["home_win"] == config.TIE_CREDIT,
        config.TIE_CREDIT,
        np.where(
            predictions["prob_home"] > 0.5,
            predictions["home_win"],
            1.0 - predictions["home_win"],
        ),
    )
    by_season = (
        predictions.assign(hit=hit)
        .groupby("season")
        .agg(games=("hit", "size"), accuracy=("hit", "mean"))
        .reset_index()
    )
    briers = (
        predictions.groupby("season")
        .apply(
            lambda g: float(np.mean((_clip(g["prob_home"]) - g["home_win"]) ** 2)),
            include_groups=False,
        )
        .rename("brier")
        .reset_index()
    )
    by_season = by_season.merge(briers, on="season", how="left")

    return BacktestResult(
        predictions=predictions,
        summary=summary,
        calibration=calibration_table(predictions, "prob_home"),
        by_season=by_season,
        ats=[ats_record(predictions, threshold=t) for t in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)],
        totals={
            "blended": evaluate_totals(predictions, "pred_total"),
            "model": evaluate_totals(predictions, "model_total"),
            "market": evaluate_totals(predictions, "market_total"),
        },
        ou=[ou_record(predictions, threshold=t) for t in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)],
    )


def clear_cache() -> int:
    """Delete every cached backtest; returns how many files went."""
    files = list(_cache_dir().glob("predictions_*.csv.gz"))
    for path in files:
        path.unlink(missing_ok=True)
    return len(files)
