"""Evaluation metrics and the walk-forward driver."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflpredict import config
from nflpredict.backtest import ats_record, calibration_table, evaluate, walk_forward
from nflpredict.features import build_features


def _frame(probs, outcomes, spreads=None, margins=None):
    n = len(probs)
    return pd.DataFrame(
        {
            "prob_home": probs,
            "home_win": outcomes,
            "market_spread": spreads if spreads is not None else [np.nan] * n,
            "margin": margins if margins is not None else [np.nan] * n,
            "pred_margin": [0.0] * n,
        }
    )


def test_perfect_predictions_score_perfectly():
    frame = _frame([0.99, 0.01, 0.99], [1.0, 0.0, 1.0])
    result = evaluate(frame)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["brier"] < 0.001


def test_inverted_predictions_score_zero():
    frame = _frame([0.99, 0.01], [0.0, 1.0])
    assert evaluate(frame)["accuracy"] == pytest.approx(0.0)


def test_coin_flips_give_a_brier_of_one_quarter():
    frame = _frame([0.5] * 4, [1.0, 0.0, 1.0, 0.0])
    assert evaluate(frame)["brier"] == pytest.approx(0.25)


def test_ties_are_credited_as_half_a_win():
    frame = _frame([0.9, 0.9], [config.TIE_CREDIT, config.TIE_CREDIT])
    assert evaluate(frame)["accuracy"] == pytest.approx(config.TIE_CREDIT)


def test_evaluate_handles_an_empty_frame():
    assert evaluate(_frame([], []))["n"] == 0


def test_calibration_folds_both_sides_into_one_bucket():
    """A 10% home call and a 90% home call are both 90% confident picks."""
    frame = _frame([0.9, 0.1, 0.9, 0.1], [1.0, 0.0, 1.0, 0.0])
    table = calibration_table(frame, bins=(0.5, 1.0))
    assert len(table) == 1
    assert table.iloc[0]["games"] == 4
    assert table.iloc[0]["actual"] == pytest.approx(1.0)


def test_ats_counts_pushes_separately_from_wins():
    frame = _frame(
        [0.6, 0.6, 0.6],
        [1.0, 1.0, 1.0],
        spreads=[-3.0, -3.0, -3.0],
        margins=[10.0, -10.0, -3.0],
    )
    frame["pred_margin"] = [5.0, 5.0, 5.0]  # model likes the home side
    record = ats_record(frame)
    assert (record["wins"], record["losses"], record["pushes"]) == (1, 1, 1)
    assert record["win_rate"] == pytest.approx(0.5)


def test_ats_breakeven_is_the_standard_juice():
    frame = _frame([0.6], [1.0], spreads=[-3.0], margins=[10.0])
    assert ats_record(frame)["breakeven"] == pytest.approx(0.5238, abs=1e-4)


def test_ats_threshold_filters_out_small_edges():
    frame = _frame(
        [0.6, 0.6], [1.0, 1.0], spreads=[-3.0, -3.0], margins=[10.0, 10.0]
    )
    frame["pred_margin"] = [-2.9, 6.0]
    assert ats_record(frame, threshold=2.0)["bets"] == 1


@pytest.mark.parametrize("use_market", [True, False])
def test_walk_forward_produces_scored_out_of_sample_predictions(
    games, team_epa, use_market
):
    features = build_features(games, team_epa)
    result = walk_forward(
        features,
        start_season=2021,
        end_season=2022,
        refit="season",
        use_market=use_market,
        quiet=True,
    )
    assert result.summary["model"]["n"] > 400
    # A real NFL model lands in a narrow band. Anything outside it means
    # either leakage (too high) or a broken pipeline (too low).
    assert 0.55 < result.summary["blended"]["accuracy"] < 0.80
    assert 0.15 < result.summary["blended"]["brier"] < 0.30
    assert not result.predictions["prob_home"].isna().any()
    assert result.predictions["prob_home"].between(0, 1).all()


def test_walk_forward_never_trains_on_the_slate_it_predicts(games, team_epa):
    """Training size must be strictly smaller than the full history."""
    features = build_features(games, team_epa)
    result = walk_forward(
        features, start_season=2022, end_season=2022, refit="week", quiet=True
    )
    total_completed = int(features["completed"].sum())
    assert result.predictions["n_train"].max() < total_completed
    # Each successive slate should train on at least as much as the last.
    by_week = result.predictions.groupby("week")["n_train"].first()
    assert list(by_week) == sorted(by_week)


def test_walk_forward_rejects_an_unknown_refit_cadence(games, team_epa):
    features = build_features(games, team_epa)
    with pytest.raises(ValueError):
        walk_forward(features, refit="daily", quiet=True)
