"""Shared fixtures.

Tests run against real cached nflverse data rather than synthetic frames,
because the properties worth testing (leakage, chronology, franchise
continuity) are properties of the real data pipeline.
"""

from __future__ import annotations

import pytest

from nflpredict import data, players

TEST_SEASONS = range(2018, 2024)


@pytest.fixture(scope="session")
def games():
    frame = data.load_games(quiet=True)
    return frame[frame["season"].isin(TEST_SEASONS)].reset_index(drop=True)


@pytest.fixture(scope="session")
def team_epa():
    return data.load_team_game_epa(TEST_SEASONS, quiet=True)


@pytest.fixture(scope="session")
def all_games():
    return data.load_games(quiet=True)


@pytest.fixture(scope="session")
def player_weeks():
    return players.load_player_weeks(TEST_SEASONS, quiet=True)


@pytest.fixture(scope="session")
def injuries():
    return players.load_injuries(TEST_SEASONS, quiet=True)
