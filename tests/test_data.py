"""Data loading, normalization and chronology."""

from __future__ import annotations

import pandas as pd
import pytest

from nflpredict import config
from nflpredict.data import load_games, normalize_team


def test_relocated_franchises_collapse_to_one_code():
    assert normalize_team("SD") == "LAC"
    assert normalize_team("STL") == "LA"
    assert normalize_team("OAK") == "LV"
    assert normalize_team("GB") == "GB"


def test_normalize_passes_through_non_strings():
    assert normalize_team(None) is None
    assert pd.isna(normalize_team(float("nan")))


def test_exactly_thirty_two_franchises(all_games):
    """Relocations must not inflate the league past 32 teams."""
    teams = set(all_games["home_team"]) | set(all_games["away_team"])
    assert len(teams) == config.N_FRANCHISES
    assert not teams & set(config.FRANCHISE_MAP)


def test_games_are_in_chronological_order(all_games):
    keys = all_games[["season", "week"]].to_numpy()
    assert all(
        (a[0], a[1]) <= (b[0], b[1]) for a, b in zip(keys, keys[1:])
    ), "game log is not sorted by season then week"


def test_every_game_has_a_kickoff_timestamp(all_games):
    assert all_games["kickoff"].notna().all()


def test_completed_flag_matches_score_availability(all_games):
    assert (all_games["completed"] == all_games["result"].notna()).all()
    assert all_games.loc[all_games["completed"], "margin"].notna().all()


def test_home_win_encodes_ties_as_a_half(all_games):
    played = all_games[all_games["completed"]]
    assert set(played["home_win"].unique()) <= {0.0, 0.5, 1.0}
    ties = played[played["margin"] == 0]
    assert (ties["home_win"] == config.TIE_CREDIT).all()


def test_margin_agrees_with_the_box_score(all_games):
    played = all_games[all_games["completed"]].head(500)
    expected = played["home_score"] - played["away_score"]
    pd.testing.assert_series_equal(
        played["margin"].astype(float), expected.astype(float), check_names=False
    )


def test_spread_line_is_quoted_from_the_home_side(all_games):
    """A positive spread_line must mean the home team is favoured."""
    priced = all_games[all_games["completed"] & all_games["spread_line"].notna()]
    home_favoured = priced[priced["spread_line"] > 3]
    away_favoured = priced[priced["spread_line"] < -3]
    assert home_favoured["margin"].mean() > 3
    assert away_favoured["margin"].mean() < -3


def test_no_duplicate_game_ids(all_games):
    assert not all_games["game_id"].duplicated().any()


def test_team_epa_has_both_sides_of_every_game(team_epa):
    counts = team_epa.groupby("game_id").size()
    assert (counts == 2).all(), "every game needs one row per team"


def test_defensive_stats_mirror_the_opponents_offence(team_epa):
    """def_* is by construction the opponent's off_* in the same game."""
    sample = team_epa.head(200)
    merged = sample.merge(
        team_epa[["game_id", "team", "off_epa_play"]].rename(
            columns={"team": "opponent", "off_epa_play": "opp_off_epa"}
        ),
        on=["game_id", "opponent"],
        how="inner",
    )
    assert len(merged) > 100
    pd.testing.assert_series_equal(
        merged["def_epa_play"], merged["opp_off_epa"], check_names=False
    )
