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

    for column in feat.FEATURE_COLUMNS + feat.TOTAL_FEATURE_COLUMNS:
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

    for column in feat.FEATURE_COLUMNS + feat.TOTAL_FEATURE_COLUMNS:
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


def test_scoring_form_does_not_contain_the_game_being_predicted(games, team_epa):
    """The sharpest leak risk: ``points_for`` comes from a game's own score.

    A blowout must move the *next* game's scoring feature, never its own. The
    check is direct -- rebuild with that one game's score erased and require
    its own feature value to be unchanged while the team's next game moves.
    """
    full = feat.build_features(games, team_epa)
    played = full[full["completed"]].sort_values(["season", "week", "kickoff"])

    # Skip the first loaded season: its league baseline is centred on itself
    # for want of an earlier one, which is a separate, documented property
    # (see ``_league_baselines``) and is pinned by its own test below.
    scoreable = played[played["season"] > int(played["season"].min())]
    # Pick a genuinely high-scoring game so the downstream effect is visible.
    target = scoreable.loc[
        (scoreable["home_score"] + scoreable["away_score"]).idxmax()
    ]
    team = target["home_team"]

    later = played[
        ((played["home_team"] == team) | (played["away_team"] == team))
        & (played["kickoff"] > target["kickoff"])
    ]
    assert len(later) > 0, "need a subsequent game to observe the update"
    next_game = later.iloc[0]

    erased = games.copy()
    mask = erased["game_id"] == target["game_id"]
    for column in ("home_score", "away_score", "result", "margin", "home_win", "total"):
        if column in erased.columns:
            erased.loc[mask, column] = np.nan
    erased.loc[mask, "completed"] = False
    rebuilt = feat.build_features(erased, team_epa[team_epa["game_id"] != target["game_id"]])

    def value(frame, game_id):
        return float(frame.loc[frame["game_id"] == game_id, "scoring_sum"].iloc[0])

    # Its own feature is untouched: the score never fed the game it came from.
    assert value(rebuilt, target["game_id"]) == pytest.approx(
        value(full, target["game_id"]), abs=1e-9
    )
    # The next game's feature does move, proving the score is used at all.
    assert value(rebuilt, next_game["game_id"]) != pytest.approx(
        value(full, next_game["game_id"]), abs=1e-6
    )


def test_the_market_never_enters_either_feature_matrix(games, team_epa):
    """Lines are blended in afterwards, so no market column may be a feature."""
    market_columns = {
        "market_spread", "market_total", "spread_line", "total_line",
        "home_moneyline", "away_moneyline", "over_odds", "under_odds",
    }
    modelled = set(feat.FEATURE_COLUMNS) | set(feat.TOTAL_FEATURE_COLUMNS)
    assert modelled.isdisjoint(market_columns)


def test_the_self_centred_baseline_is_confined_to_the_first_season(games, team_epa):
    """Bound the one documented exception rather than leaving it implicit.

    The first loaded season centres on its own mean because nothing earlier
    exists. Every season after it must be untouched by its own games -- which
    is what makes that first season a warm-up season rather than a leak in
    the scored range.
    """
    full = feat.build_features(games, team_epa)
    played = full[full["completed"]]
    first_season = int(played["season"].min())

    later = played[played["season"] > first_season]
    target = later.loc[(later["home_score"] + later["away_score"]).idxmax()]

    erased = games.copy()
    mask = erased["game_id"] == target["game_id"]
    for column in ("home_score", "away_score", "result", "margin", "home_win", "total"):
        if column in erased.columns:
            erased.loc[mask, column] = np.nan
    erased.loc[mask, "completed"] = False
    rebuilt = feat.build_features(
        erased, team_epa[team_epa["game_id"] != target["game_id"]]
    )

    left = full[full["game_id"] == target["game_id"]]
    right = rebuilt[rebuilt["game_id"] == target["game_id"]]
    for column in feat.FEATURE_COLUMNS + feat.TOTAL_FEATURE_COLUMNS:
        assert float(right[column].iloc[0]) == pytest.approx(
            float(left[column].iloc[0]), abs=1e-9
        ), f"{column!r} in season {int(target['season'])} saw its own result"
