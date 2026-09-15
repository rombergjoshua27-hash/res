"""Team totals and half lines.

These are derived markets, not models: once a game has a projected total and
a projected margin, the team totals follow by arithmetic and the halves by a
fitted linear map. The properties worth testing are therefore mostly
identities -- the pieces must add back up to the game they came from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflpredict.features import build_features
from nflpredict.model import GamePredictor
from nflpredict.splits import (
    FALLBACK_MARGIN_SLOPE,
    FALLBACK_TOTAL_SHARE,
    SplitPredictor,
    team_totals,
)
from nflpredict.totals import TotalsPredictor


# --------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------


def test_team_totals_reconstruct_the_game():
    home, away = team_totals(45.0, 7.0)
    assert home + away == pytest.approx(45.0)
    assert home - away == pytest.approx(7.0)


def test_a_pick_em_splits_the_total_evenly():
    home, away = team_totals(44.0, 0.0)
    assert home == away == pytest.approx(22.0)


def test_the_margin_favours_the_home_side():
    home, away = team_totals(40.0, 10.0)
    assert home > away
    home, away = team_totals(40.0, -10.0)
    assert away > home


def test_team_totals_vectorise():
    home, away = team_totals(np.array([45.0, 50.0]), np.array([7.0, -3.0]))
    np.testing.assert_allclose(home + away, [45.0, 50.0])
    np.testing.assert_allclose(home - away, [7.0, -3.0])


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def calibrated(games, team_epa):
    """A SplitPredictor fitted the way the CLI fits one."""
    half = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "home_first_half": (games["home_score"] * 0.5).round(),
            "away_first_half": (games["away_score"] * 0.5).round(),
        }
    )
    features = build_features(games, team_epa, half_scores=half)
    history = features[features["completed"]].copy()
    winner = GamePredictor().fit(history)
    totals = TotalsPredictor().fit(history)
    history["pred_margin"] = winner.predict(history)["pred_margin"].to_numpy()
    history["pred_total"] = totals.predict(history)["pred_total"].to_numpy()
    return SplitPredictor().fit(history), history


def test_a_fitted_calibration_reports_what_it_learned(calibrated):
    model, _ = calibrated
    assert model.report.fitted_from_history
    assert model.report.n_train > 0
    # This fixture halves every score, so the map must come out near one half.
    assert model.report.total_slope == pytest.approx(0.5, abs=0.15)


def test_predict_before_fit_is_an_error():
    with pytest.raises(RuntimeError):
        SplitPredictor().predict(pd.DataFrame({"game_id": ["x"]}))


def test_too_little_history_falls_back_to_measured_league_shares():
    tiny = pd.DataFrame(
        {
            "game_id": ["a", "b"],
            "pred_total": [44.0, 46.0],
            "pred_margin": [3.0, -3.0],
            "first_half_total": [22.0, 23.0],
            "first_half_margin": [1.0, -1.0],
        }
    )
    model = SplitPredictor().fit(tiny)
    assert not model.report.fitted_from_history

    out = model.predict(tiny)
    assert out["first_half_total_pred"].iloc[0] == pytest.approx(
        44.0 * FALLBACK_TOTAL_SHARE
    )
    assert out["first_half_margin_pred"].iloc[0] == pytest.approx(
        3.0 * FALLBACK_MARGIN_SLOPE
    )


# --------------------------------------------------------------------------
# Identities the output must satisfy
# --------------------------------------------------------------------------


def test_team_totals_add_to_the_game_total(calibrated):
    model, history = calibrated
    out = model.predict(history)
    np.testing.assert_allclose(
        (out["home_team_total"] + out["away_team_total"]).to_numpy(),
        history["pred_total"].to_numpy(),
        atol=1e-9,
    )


def test_team_totals_differ_by_the_spread(calibrated):
    model, history = calibrated
    out = model.predict(history)
    np.testing.assert_allclose(
        (out["home_team_total"] - out["away_team_total"]).to_numpy(),
        history["pred_margin"].to_numpy(),
        atol=1e-9,
    )


def test_the_two_halves_add_back_to_the_full_game(calibrated):
    """A second half defined as the remainder can never drift from the game."""
    model, history = calibrated
    out = model.predict(history)
    np.testing.assert_allclose(
        (out["first_half_total_pred"] + out["second_half_total_pred"]).to_numpy(),
        history["pred_total"].to_numpy(),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        (out["first_half_margin_pred"] + out["second_half_margin_pred"]).to_numpy(),
        history["pred_margin"].to_numpy(),
        atol=1e-9,
    )


def test_first_half_team_totals_add_to_the_first_half_total(calibrated):
    model, history = calibrated
    out = model.predict(history)
    np.testing.assert_allclose(
        (out["fh_home_total"] + out["fh_away_total"]).to_numpy(),
        out["first_half_total_pred"].to_numpy(),
        atol=1e-9,
    )


def test_a_first_half_is_roughly_half_a_game(calibrated):
    model, history = calibrated
    out = model.predict(history)
    share = out["first_half_total_pred"] / history["pred_total"]
    assert 0.4 < share.mean() < 0.6


def test_every_game_gets_one_row(calibrated):
    model, history = calibrated
    out = model.predict(history)
    assert len(out) == len(history)
    assert out["game_id"].is_unique


# --------------------------------------------------------------------------
# Halftime scores from play-by-play
# --------------------------------------------------------------------------


def test_half_scores_never_exceed_the_final_score(games):
    """A cumulative score at halftime cannot be larger than the final one."""
    from nflpredict.data import load_half_scores

    seasons = sorted(games["season"].unique())
    half = load_half_scores(seasons, quiet=True)
    if half.empty:
        pytest.skip("no cached halftime scores")

    merged = games[games["completed"]].merge(half, on="game_id", how="inner")
    assert len(merged) > 0
    assert (merged["home_first_half"] <= merged["home_score"]).all()
    assert (merged["away_first_half"] <= merged["away_score"]).all()
    assert (merged["home_first_half"] >= 0).all()
