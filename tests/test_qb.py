"""Passer value and roster availability.

The first quarterback adjustment this project shipped made the model worse,
so the replacement carries a heavier burden of proof. Two properties matter
most and are asserted directly here:

* it is computable **before kickoff** -- the version that identified a
  starter as "whoever threw the most passes" was silently useless for the
  only games anyone needs predicted, and
* it is **leak-free** -- a passer's rating may never contain the game it is
  attached to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflpredict import features as feat
from nflpredict.players import injury_availability
from nflpredict.qb import build_availability_features, build_qb_features


@pytest.fixture(scope="module")
def full(games, team_epa, player_weeks, injuries):
    return feat.build_features(
        games, team_epa, player_weeks=player_weeks, injuries=injuries
    )


# --------------------------------------------------------------------------
# Availability mapping
# --------------------------------------------------------------------------


def test_availability_is_ordered_by_severity():
    assert (
        injury_availability("Out")
        < injury_availability("Doubtful")
        < injury_availability("Questionable")
        < injury_availability(None)
    )


def test_a_player_who_is_out_never_plays():
    assert injury_availability("Out") == 0.0


def test_an_unlisted_player_is_available():
    for status in (None, "", "   ", float("nan"), "Full Participation in Practice"):
        assert injury_availability(status) == 1.0


def test_availability_tolerates_surrounding_whitespace():
    assert injury_availability("  Questionable  ") == injury_availability("Questionable")


# --------------------------------------------------------------------------
# Computable before kickoff
# --------------------------------------------------------------------------


def test_qb_features_exist_for_games_that_have_not_been_played(
    games, team_epa, player_weeks, injuries
):
    """The property the first implementation silently failed.

    Identifying the starter from box-score dropbacks yields nothing for an
    upcoming game, because it has no box score. Using the listed starter --
    which nflverse publishes ahead of kickoff -- is what makes the feature
    usable for the slate being predicted rather than only for history.
    """
    schedule = games.copy()
    # Treat the last week on record as unplayed, keeping the fixtures intact.
    last = schedule[schedule["completed"]]["season"].max()
    final_week = schedule[schedule["season"] == last]["week"].max()
    upcoming = (schedule["season"] == last) & (schedule["week"] == final_week)
    for column in ("home_score", "away_score", "result", "margin", "home_win", "total"):
        if column in schedule.columns:
            schedule.loc[upcoming, column] = np.nan
    schedule.loc[upcoming, "completed"] = False

    # Player rows for those games disappear too, exactly as they would live.
    future_ids = set(schedule.loc[upcoming, "game_id"])
    trimmed = player_weeks[~player_weeks["game_id"].isin(future_ids)]

    built = feat.build_features(
        schedule, team_epa, player_weeks=trimmed, injuries=injuries
    )
    slate = built[built["game_id"].isin(future_ids)]
    assert len(slate) > 0
    assert (slate["qb_edge"] != 0).any(), "no passer rating for an upcoming slate"
    assert (slate["qb_delta"] != 0).any(), "no passer change for an upcoming slate"


def test_features_are_zero_without_player_data(games):
    qb = build_qb_features(games, pd.DataFrame())
    assert (qb["qb_delta"] == 0).all()
    assert (qb["qb_edge"] == 0).all()


def test_availability_is_zero_without_an_injury_report(games, player_weeks):
    out = build_availability_features(games, player_weeks, pd.DataFrame())
    assert (out["availability_edge"] == 0).all()


def test_every_game_gets_exactly_one_row(games, player_weeks, injuries):
    qb = build_qb_features(games, player_weeks)
    availability = build_availability_features(games, player_weeks, injuries)
    assert len(qb) == len(games)
    assert len(availability) == len(games)
    assert qb["game_id"].is_unique


# --------------------------------------------------------------------------
# Shape of the signal
# --------------------------------------------------------------------------


def test_keeping_the_same_starter_is_a_small_change(full):
    """qb_delta measures a *change* of quarterback, so continuity reads small.

    If this drifted large the feature would be restating passer quality --
    which Elo already carries -- and would be double-counting, the exact
    failure the first attempt made.
    """
    played = full[full["completed"]]
    assert played["qb_delta"].abs().median() < played["qb_edge"].abs().median()


def test_a_better_passer_favours_his_side(full):
    played = full[full["completed"] & full["margin"].notna()]
    assert np.corrcoef(played["qb_edge"], played["margin"])[0, 1] > 0.1


def test_missing_players_hurt_the_side_that_is_missing_them(full):
    played = full[full["completed"] & full["margin"].notna()]
    covered = played[played["availability_edge"] != 0]
    assert len(covered) > 0
    assert np.corrcoef(covered["availability_edge"], covered["margin"])[0, 1] > 0


def test_availability_edge_is_antisymmetric(games, player_weeks, injuries):
    """Swapping home and away must flip the sign, never change the magnitude."""
    flipped = games.copy()
    flipped["home_team"], flipped["away_team"] = (
        games["away_team"].to_numpy(), games["home_team"].to_numpy()
    )
    original = build_availability_features(games, player_weeks, injuries)
    swapped = build_availability_features(flipped, player_weeks, injuries)
    np.testing.assert_allclose(
        original["availability_edge"].to_numpy(),
        -swapped["availability_edge"].to_numpy(),
        atol=1e-9,
    )


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------


def test_passer_ratings_do_not_contain_their_own_game(
    games, team_epa, player_weeks, injuries
):
    """Rebuild with the future erased; no past game's passer terms may move."""
    cutoff_season, cutoff_week = 2021, 10
    period = games["season"] * 100 + games["week"]
    future = period >= (cutoff_season * 100 + cutoff_week)

    redacted = games.copy()
    for column in ("result", "margin", "home_win", "home_score", "away_score", "total"):
        if column in redacted.columns:
            redacted.loc[future, column] = np.nan
    redacted.loc[future, "completed"] = False

    future_ids = set(games.loc[future, "game_id"])
    full = feat.build_features(
        games, team_epa, player_weeks=player_weeks, injuries=injuries
    )
    rebuilt = feat.build_features(
        redacted,
        team_epa[~team_epa["game_id"].isin(future_ids)],
        player_weeks=player_weeks[~player_weeks["game_id"].isin(future_ids)],
        injuries=injuries[
            ~(
                (injuries["season"] * 100 + injuries["week"])
                >= (cutoff_season * 100 + cutoff_week)
            )
        ],
    )

    past = full[period < (cutoff_season * 100 + cutoff_week)]
    left = past.set_index("game_id").sort_index()
    right = rebuilt[rebuilt["game_id"].isin(past["game_id"])].set_index(
        "game_id"
    ).sort_index()
    assert len(left) == len(right) > 0

    for column in feat.QB_FEATURE_COLUMNS:
        np.testing.assert_allclose(
            left[column].to_numpy(dtype=float),
            right[column].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"{column!r} changed when later games were erased",
        )


def test_a_passers_own_game_does_not_feed_his_rating_for_it(player_weeks, games):
    """Erase one start; that game's own rating must hold while the next moves."""
    starts = player_weeks[player_weeks["dropbacks"] >= 20]
    busiest = starts["player_id"].value_counts().idxmax()
    his = starts[starts["player_id"] == busiest].sort_values(["season", "week"])
    assert len(his) > 4
    target, following = his.iloc[len(his) // 2], his.iloc[len(his) // 2 + 1]

    full = build_qb_features(games, player_weeks)
    trimmed = build_qb_features(
        games, player_weeks[player_weeks["game_id"] != target["game_id"]]
    )

    def edge(frame, game_id):
        return float(frame.loc[frame["game_id"] == game_id, "qb_edge"].iloc[0])

    assert edge(trimmed, target["game_id"]) == pytest.approx(
        edge(full, target["game_id"]), abs=1e-9
    )
    assert edge(trimmed, following["game_id"]) != pytest.approx(
        edge(full, following["game_id"]), abs=1e-6
    )
