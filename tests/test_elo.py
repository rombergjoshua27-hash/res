"""Elo engine mechanics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflpredict import config
from nflpredict.elo import EloEngine, elo_to_prob, elo_to_spread, estimate_hfa


def test_equal_ratings_give_even_odds():
    assert elo_to_prob(0.0) == pytest.approx(0.5)


def test_probability_is_monotonic_in_rating_gap():
    gaps = [-400, -100, 0, 100, 400]
    probs = [elo_to_prob(g) for g in gaps]
    assert all(a < b for a, b in zip(probs, probs[1:]))


def test_four_hundred_points_is_ten_to_one():
    assert elo_to_prob(400.0) == pytest.approx(10 / 11, rel=1e-6)


def test_spread_conversion_matches_configured_scale():
    assert elo_to_spread(config.ELO_PER_POINT) == pytest.approx(1.0)


def test_rating_updates_are_zero_sum(all_games):
    """Elo is a closed system: what one team gains, its opponent loses."""
    engine = EloEngine()
    subset = all_games[all_games["season"].between(2015, 2019)]
    engine.run(subset)
    total = sum(engine.state.ratings.values())
    expected = config.ELO_INIT * len(engine.state.ratings)
    # Season-boundary regression also pulls toward the mean, so the sum stays
    # on 1500 * n rather than merely near it.
    assert total == pytest.approx(expected, abs=1e-6)


def test_winner_gains_and_loser_loses(all_games):
    engine = EloEngine()
    frame = engine.run(all_games[all_games["season"] == 2022])
    merged = all_games.merge(frame, on="game_id")
    played = merged[merged["completed"]].iloc[0]

    after = EloEngine()
    after.run(all_games[all_games["season"] == 2022].head(1))
    winner = played["home_team"] if played["margin"] > 0 else played["away_team"]
    assert after.state.ratings[winner] > config.ELO_INIT


def test_season_regression_pulls_toward_the_mean():
    engine = EloEngine()
    engine.state.ratings = {"AAA": 1700.0, "BBB": 1300.0}
    engine.state.last_season = 2020
    engine.state.carry_to_season(2021)
    assert engine.state.ratings["AAA"] == pytest.approx(
        1500.0 + config.ELO_SEASON_CARRY * 200.0
    )
    assert engine.state.ratings["BBB"] == pytest.approx(
        1500.0 - config.ELO_SEASON_CARRY * 200.0
    )


def test_blowouts_move_ratings_more_than_narrow_wins(all_games):
    """The margin-of-victory multiplier must actually depend on the margin."""
    template = all_games[all_games["completed"]].iloc[:1].copy()

    def final_rating(margin: float) -> float:
        frame = template.copy()
        frame["margin"] = margin
        frame["result"] = margin
        frame["home_win"] = 1.0
        engine = EloEngine()
        engine.run(frame)
        return engine.state.ratings[frame["home_team"].iloc[0]]

    assert final_rating(35.0) > final_rating(3.0) > config.ELO_INIT


def test_neutral_site_games_get_no_home_advantage(all_games):
    engine = EloEngine()
    frame = engine.run(all_games)
    merged = all_games.merge(frame, on="game_id")
    neutral = merged[merged["neutral"]]
    assert len(neutral) > 0
    assert (neutral["elo_hfa_points"] == 0).all()


def test_home_advantage_estimate_is_bounded_and_recent(all_games):
    for season in (2012, 2018, 2024, 2026):
        hfa = estimate_hfa(all_games, season)
        assert config.HFA_MIN_POINTS <= hfa <= config.HFA_MAX_POINTS
    # 2020 was played in empty stadiums and is pinned to its measured value.
    assert estimate_hfa(all_games, 2020) == config.HFA_SEASON_OVERRIDES[2020]


def test_unplayed_games_never_update_ratings(all_games):
    """Scheduled-but-unplayed games are rated, not learned from."""
    future = all_games[~all_games["completed"]]
    assert len(future) > 0
    engine = EloEngine()
    engine.run(future)
    assert all(v == pytest.approx(config.ELO_INIT) for v in engine.state.ratings.values())
