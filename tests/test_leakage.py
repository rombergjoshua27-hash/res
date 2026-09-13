"""The load-bearing test: features must not see the future.

Any model that peeks at later results will report a spectacular backtest and
then fail in production. These tests verify the guarantee directly, by
rebuilding features from a dataset with the future erased and checking that
nothing about the past moved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflpredict import features as feat
from nflpredict.elo import EloEngine

CUTOFF_SEASON, CUTOFF_WEEK = 2021, 10


def _redact_future(games: pd.DataFrame, epa: pd.DataFrame):
    """Erase every result at or after the cutoff, keeping the schedule intact."""
    period = games["season"] * 100 + games["week"]
    future = period >= (CUTOFF_SEASON * 100 + CUTOFF_WEEK)

    redacted = games.copy()
    for column in ("result", "margin", "home_win", "home_score", "away_score", "total"):
        if column in redacted.columns:
            redacted.loc[future, column] = np.nan
    redacted.loc[future, "completed"] = False

    future_ids = set(games.loc[future, "game_id"])
    redacted_epa = epa[~epa["game_id"].isin(future_ids)].copy()
    return redacted, redacted_epa


def test_features_do_not_change_when_future_is_erased(games, team_epa):
    """Past rows must be bit-for-bit identical with the future deleted."""
    full = feat.build_features(games, team_epa)
    redacted_games, redacted_epa = _redact_future(games, team_epa)
    redacted = feat.build_features(redacted_games, redacted_epa)

    period = full["season"] * 100 + full["week"]
    past = full[period < (CUTOFF_SEASON * 100 + CUTOFF_WEEK)]
    redacted_past = redacted[redacted["game_id"].isin(past["game_id"])]

    assert len(past) == len(redacted_past) > 0

    left = past.set_index("game_id").sort_index()
    right = redacted_past.set_index("game_id").sort_index()

    for column in feat.FEATURE_COLUMNS:
        np.testing.assert_allclose(
            left[column].to_numpy(dtype=float),
            right[column].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"feature {column!r} changed when future games were erased",
        )


def test_cutoff_week_features_are_unaffected_by_its_own_results(games, team_epa):
    """A slate's own results must not feed the features used to predict it."""
    full = feat.build_features(games, team_epa)
    redacted_games, redacted_epa = _redact_future(games, team_epa)
    redacted = feat.build_features(redacted_games, redacted_epa)

    target = (full["season"] == CUTOFF_SEASON) & (full["week"] == CUTOFF_WEEK)
    ids = full.loc[target, "game_id"]
    assert len(ids) > 0

    left = full[full["game_id"].isin(ids)].set_index("game_id").sort_index()
    right = redacted[redacted["game_id"].isin(ids)].set_index("game_id").sort_index()

    for column in feat.FEATURE_COLUMNS:
        np.testing.assert_allclose(
            left[column].to_numpy(dtype=float),
            right[column].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"feature {column!r} leaked results from its own slate",
        )


def test_elo_pregame_rating_precedes_its_own_update(games):
    """Elo recorded for a game must be the rating from *before* that game."""
    engine = EloEngine()
    frame = engine.run(games)
    merged = games.merge(frame, on="game_id")

    played = merged[merged["completed"]]
    team = played["home_team"].iloc[0]
    team_games = played[
        (played["home_team"] == team) | (played["away_team"] == team)
    ].head(6)

    ratings = [
        row.elo_home_pre if row.home_team == team else row.elo_away_pre
        for row in team_games.itertuples(index=False)
    ]
    # The first appearance must be the untouched initial rating.
    assert ratings[0] == pytest.approx(1500.0)
    # And the rating must actually move afterwards, or nothing is being learned.
    assert any(abs(value - 1500.0) > 1e-6 for value in ratings[1:])


def test_form_features_are_zero_before_any_games_are_played(games, team_epa):
    """Week 1 of the first season has no history, so form must be neutral."""
    full = feat.build_features(games, team_epa)
    first_season = int(full["season"].min())
    opener = full[(full["season"] == first_season) & (full["week"] == 1)]
    assert len(opener) > 0
    assert (opener["form_confidence"] == 0).all()
