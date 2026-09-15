"""Player prop projections.

Two failures in this module would be silent rather than loud, so both get a
direct test: a rate computed against a vanishing denominator (which once
produced yards-per-carry in the hundreds of millions and destroyed the fit),
and a projection for a slate that has not been played, which is the only
slate anyone needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflpredict.props import (
    PROP_FEATURE_COLUMNS,
    PROP_TARGETS,
    RATE_BOUNDS,
    PropsPredictor,
    build_player_features,
    build_slate_rows,
    projected_roster,
)


@pytest.fixture(scope="module")
def player_features(player_weeks, games, injuries):
    return build_player_features(player_weeks, games, injuries)


# --------------------------------------------------------------------------
# Feature hygiene
# --------------------------------------------------------------------------


def test_every_feature_is_finite(player_features):
    """A single infinity anywhere here takes the whole regression with it."""
    values = player_features[PROP_FEATURE_COLUMNS].to_numpy(dtype=float)
    assert np.isfinite(values).all()


def test_rates_stay_inside_their_bounds(player_features):
    for column, (low, high) in RATE_BOUNDS.items():
        series = player_features[column]
        assert series.min() >= low - 1e-9, f"{column} fell below {low}"
        assert series.max() <= high + 1e-9, f"{column} exceeded {high}"


def test_a_vanishing_denominator_does_not_explode_a_rate():
    """The exact bug: an exponentially-weighted carry count decaying to ~0.

    Guarding only against an exact zero left 1e-9 carries dividing real
    yards, which produced a rate in the hundreds of millions.
    """
    from nflpredict.props import _safe_ratio

    numerator = pd.Series([50.0, 50.0, 50.0])
    denominator = pd.Series([1e-9, 0.0, 10.0])
    rate = _safe_ratio(numerator, denominator, RATE_BOUNDS["form_rush_ypc"])
    assert rate.iloc[0] == 0.0
    assert rate.iloc[1] == 0.0
    assert rate.iloc[2] == pytest.approx(5.0)


def test_one_row_per_player_game(player_features):
    assert not player_features.duplicated(["player_id", "game_id"]).any()


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------


def test_form_excludes_the_game_it_is_attached_to(player_weeks, games, injuries):
    """A player's form must be built from earlier games only."""
    busiest = player_weeks.groupby("player_id")["targets"].sum().idxmax()
    built = build_player_features(player_weeks, games, injuries)
    his = built[built["player_id"] == busiest].sort_values(["season", "week"])
    assert len(his) > 3

    # The first appearance can have no history behind it.
    assert his["form_rec_yards"].iloc[0] == 0.0
    # And each later value must match the running mean of strictly earlier games.
    assert his["form_games"].iloc[0] == 0
    assert (his["form_games"].to_numpy() == np.arange(len(his))).all()


def test_erasing_a_game_leaves_its_own_features_untouched(
    player_weeks, games, injuries
):
    busiest = player_weeks.groupby("player_id")["targets"].sum().idxmax()
    his = player_weeks[player_weeks["player_id"] == busiest].sort_values(
        ["season", "week"]
    )
    target = his.iloc[len(his) // 2]

    full = build_player_features(player_weeks, games, injuries)
    trimmed = build_player_features(
        player_weeks[
            ~(
                (player_weeks["player_id"] == busiest)
                & (player_weeks["game_id"] == target["game_id"])
            )
        ],
        games,
        injuries,
    )

    def form(frame):
        row = frame[
            (frame["player_id"] == busiest) & (frame["game_id"] == target["game_id"])
        ]
        return None if row.empty else float(row["form_rec_yards"].iloc[0])

    assert form(trimmed) is None  # the row itself is gone
    # The following game must move, proving the erased game was being used.
    following = his.iloc[len(his) // 2 + 1]

    def form_at(frame, game_id):
        row = frame[
            (frame["player_id"] == busiest) & (frame["game_id"] == game_id)
        ]
        return float(row["form_rec_yards"].iloc[0])

    assert form_at(trimmed, following["game_id"]) != pytest.approx(
        form_at(full, following["game_id"]), abs=1e-6
    )


# --------------------------------------------------------------------------
# Fitting and projection
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted(player_features):
    history = player_features[player_features["season"] <= 2021]
    return PropsPredictor().fit(history), player_features


def test_projections_are_never_negative(fitted):
    model, features = fitted
    out = model.predict(features[features["season"] >= 2022])
    for target in PROP_TARGETS:
        column = f"proj_{target}"
        if column in out and out[column].notna().any():
            assert (out[column].dropna() >= 0).all()


def test_predict_before_fit_is_an_error(player_features):
    with pytest.raises(RuntimeError):
        PropsPredictor().predict(player_features)


def test_a_player_ruled_out_is_projected_at_zero(fitted):
    model, features = fitted
    frame = features[features["season"] >= 2022].head(50).copy()
    frame["availability"] = 0.0
    out = model.predict(frame)
    for target in PROP_TARGETS:
        column = f"proj_{target}"
        if column in out and out[column].notna().any():
            assert (out[column].dropna() == 0).all()


def test_availability_scales_the_projection(fitted):
    model, features = fitted
    frame = features[features["season"] >= 2022].head(50).copy()
    frame["availability"] = 1.0
    full = model.predict(frame)["proj_receiving_yards"].to_numpy()
    frame["availability"] = 0.5
    halved = model.predict(frame)["proj_receiving_yards"].to_numpy()
    np.testing.assert_allclose(halved, full * 0.5, atol=1e-9)


def test_the_model_beats_a_players_own_recent_average(fitted):
    """The baseline that matters: if it cannot beat this, it is not earning its
    place."""
    model, features = fitted
    test = features[features["season"] >= 2022]
    test = test[test["targets"] >= 3]
    assert len(test) > 200

    out = model.predict(test)
    model_error = (test["receiving_yards"].to_numpy() - out["proj_receiving_yards"].to_numpy())
    form_error = (test["receiving_yards"] - test["form_rec_yards"]).to_numpy()
    assert np.mean(np.abs(model_error)) < np.mean(np.abs(form_error))


# --------------------------------------------------------------------------
# Projecting a slate that has not been played
# --------------------------------------------------------------------------


def test_projected_roster_uses_recently_active_players(player_weeks):
    season = int(player_weeks["season"].max())
    weeks = player_weeks[player_weeks["season"] == season]["week"]
    week = int(weeks.max())
    teams = sorted(player_weeks["team"].dropna().unique())[:4]

    roster = projected_roster(player_weeks, season, week, teams)
    assert not roster.empty
    assert set(roster["team"]) <= set(teams)
    assert roster["appearances"].min() >= 1


def test_slate_rows_exist_for_games_with_no_box_score(
    player_weeks, games, injuries
):
    """The property that makes props usable at all: an upcoming slate."""
    season = int(games["season"].max())
    week = int(games[games["season"] == season]["week"].max())
    slate = games[(games["season"] == season) & (games["week"] == week)]
    history = player_weeks[
        (player_weeks["season"] * 100 + player_weeks["week"])
        < (season * 100 + week)
    ]

    rows = build_slate_rows(history, games, slate, injuries)
    assert not rows.empty, "no projectable players for an upcoming slate"
    assert set(rows["game_id"]) <= set(slate["game_id"])
    # Their form must have come from earlier games, not from nothing.
    assert (rows["form_games"] > 0).any()
    assert np.isfinite(rows[PROP_FEATURE_COLUMNS].to_numpy(dtype=float)).all()


def test_slate_rows_are_empty_without_history(games):
    assert build_slate_rows(pd.DataFrame(), games, games.head(4)).empty
