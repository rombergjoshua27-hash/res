"""Live season tracking.

The scorecard answers "how has this actually been doing", which is only
worth anything if it grades the same calls the tool would have made at the
time. Two properties carry that: the cutoff is each slate's kickoff, and
nothing is stored between runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflpredict.backtest import _grade, track_season
from nflpredict.features import build_features


@pytest.fixture(scope="module")
def features(games, team_epa):
    return build_features(games, team_epa)


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------


def _frame(**overrides):
    base = {
        "home_team": ["KC"], "away_team": ["BUF"], "home_win": [1.0],
        "prob_home": [0.7], "margin": [7.0], "market_spread": [3.0],
        "pred_margin": [6.0], "actual_total": [45.0], "market_total": [42.0],
        "model_total": [46.0], "pred_total": [44.0],
    }
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


def test_a_correct_pick_is_graded_a_hit():
    assert _grade(_frame())["hit"].iloc[0] == 1.0


def test_an_incorrect_pick_is_graded_a_miss():
    assert _grade(_frame(home_win=0.0, margin=-7.0))["hit"].iloc[0] == 0.0


def test_a_tie_is_graded_as_half():
    assert _grade(_frame(home_win=0.5, margin=0.0))["hit"].iloc[0] == 0.5


def test_the_pick_is_the_side_the_probability_favours():
    assert _grade(_frame())["pick"].iloc[0] == "KC"
    assert _grade(_frame(prob_home=0.3))["pick"].iloc[0] == "BUF"


def test_covering_the_spread_is_an_ats_win():
    # Model likes the home side (6.0 > 3.0); home wins by 7, so it covers.
    assert _grade(_frame())["ats_win"].iloc[0] == 1.0


def test_failing_to_cover_is_an_ats_loss():
    assert _grade(_frame(margin=1.0))["ats_win"].iloc[0] == 0.0


def test_a_spread_push_is_not_counted_either_way():
    """A push is no result, so it must not be graded as a loss."""
    assert np.isnan(_grade(_frame(margin=3.0))["ats_win"].iloc[0])


def test_an_unpriced_game_has_no_ats_result():
    assert np.isnan(_grade(_frame(market_spread=np.nan))["ats_win"].iloc[0])


def test_the_over_is_graded_against_the_models_own_total():
    # model_total 46 > market 42, so OVER; the game landed on 45.
    graded = _grade(_frame())
    assert graded["ou_pick"].iloc[0] == "OVER"
    assert graded["ou_win"].iloc[0] == 1.0


def test_the_under_is_graded_the_other_way():
    graded = _grade(_frame(model_total=38.0, actual_total=40.0))
    assert graded["ou_pick"].iloc[0] == "UNDER"
    assert graded["ou_win"].iloc[0] == 1.0


def test_a_total_push_is_not_counted_either_way():
    assert np.isnan(_grade(_frame(actual_total=42.0))["ou_win"].iloc[0])


def test_total_error_uses_the_blended_projection():
    assert _grade(_frame())["total_error"].iloc[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Walking a season
# --------------------------------------------------------------------------


def test_tracking_a_season_grades_its_settled_games(features):
    season = int(features[features["completed"]]["season"].max())
    card = track_season(features, season)
    assert card.season == season
    assert len(card.games) > 0
    assert 0.0 <= card.summary["accuracy"] <= 1.0
    assert card.summary["games"] == len(card.games)


def test_every_settled_week_appears_once(features):
    season = int(features[features["completed"]]["season"].max())
    card = track_season(features, season)
    assert card.by_week["week"].is_unique
    assert card.by_week["games"].sum() == len(card.games)


def test_the_weekly_totals_reconcile_with_the_summary(features):
    season = int(features[features["completed"]]["season"].max())
    card = track_season(features, season)
    assert card.by_week["correct"].sum() == pytest.approx(card.summary["correct"])


def test_a_season_with_nothing_settled_returns_an_empty_card(features):
    card = track_season(features, 1970)
    assert card.games.empty
    assert card.by_week.empty


def test_no_week_is_graded_on_its_own_results(features):
    """The cutoff is the slate's kickoff, so a week cannot inform itself."""
    season = int(features[features["completed"]]["season"].max())
    card = track_season(features, season)
    per_week = card.games.groupby("week")["n_train"].first().sort_index()
    played = card.games.groupby("week").size().sort_index()

    # The fit grows every week, so no week reuses the previous week's model.
    assert per_week.is_monotonic_increasing

    # And the growth into week N must cover the games played in week N-1:
    # those results were on the board before week N kicked off, so they are
    # in its fit. Anything less would mean a week was scored on stale history.
    growth = per_week.diff().dropna()
    previous = played.shift(1).reindex(growth.index)
    assert (growth >= previous).all()
