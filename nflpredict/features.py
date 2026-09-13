"""Leak-free feature construction.

Every feature attached to a game is computed from information available
*before* that game kicks off. The guarantee is enforced structurally: team
form is accumulated by walking each franchise's schedule in order and
recording the running state **before** folding in the current game's result.

`tests/test_leakage.py` asserts this property directly by verifying that
features for a game are unchanged when every later result is deleted.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from . import config
from .elo import EloEngine

__all__ = ["build_features", "FEATURE_COLUMNS", "MARKET_COLUMNS", "TEAM_STATS"]


# Raw per-game team statistics that get rolled forward into "form".
TEAM_STATS: List[str] = [
    "off_epa_play", "off_pass_epa", "off_rush_epa", "off_success",
    "off_explosive", "off_turnover_rate", "off_sack_rate",
    "def_epa_play", "def_pass_epa", "def_rush_epa", "def_success",
    "def_explosive", "def_turnover_rate", "def_sack_rate",
]

# Model inputs that never touch a betting market.
FEATURE_COLUMNS: List[str] = [
    "elo_diff",
    "epa_edge",
    "pass_epa_edge",
    "rush_epa_edge",
    "success_edge",
    "explosive_edge",
    "turnover_edge",
    "sack_edge",
    "rest_diff",
    "div_game",
    "neutral",
    "is_playoff",
    "form_confidence",
]

# Market-derived inputs, kept separate so the model can be run with or
# without them (`--no-market` isolates genuine model edge).
MARKET_COLUMNS: List[str] = ["market_spread"]


# --------------------------------------------------------------------------
# League baselines
# --------------------------------------------------------------------------


def _league_baselines(team_games: pd.DataFrame) -> pd.DataFrame:
    """Mean of each stat over all seasons *strictly before* each season.

    Centering on a prior-seasons-only baseline removes era drift (offensive
    efficiency has risen steadily) without letting the current season inform
    its own normalization.
    """
    seasonal = team_games.groupby("season")[TEAM_STATS].mean()
    counts = team_games.groupby("season").size()

    baselines = {}
    running_sum = pd.Series(0.0, index=TEAM_STATS)
    running_n = 0.0
    for season in sorted(seasonal.index):
        if running_n > 0:
            baselines[season] = running_sum / running_n
        else:
            baselines[season] = seasonal.loc[season]  # first season: self
        running_sum = running_sum + seasonal.loc[season] * counts.loc[season]
        running_n += counts.loc[season]

    return pd.DataFrame(baselines).T.reindex(columns=TEAM_STATS)


# --------------------------------------------------------------------------
# Per-team rolling form
# --------------------------------------------------------------------------


def _roll_one_team(
    frame: pd.DataFrame,
    stats: Sequence[str],
    *,
    alpha: float,
    carry: float,
    blend_full: float,
) -> pd.DataFrame:
    """Roll one franchise's schedule forward, recording pregame form.

    For each row the recorded value blends:
      * the exponentially-weighted mean of *earlier games this season*, and
      * the team's end-of-last-season form, shrunk toward league average,
    with the weight sliding to the current season over ``blend_full`` games.
    """
    n = len(frame)
    out = {stat: np.full(n, np.nan) for stat in stats}
    games_played = np.zeros(n)

    continuous: Dict[str, float] = {s: np.nan for s in stats}
    within: Dict[str, float] = {s: np.nan for s in stats}
    prior_final: Dict[str, float] = {s: np.nan for s in stats}
    current_season: int | None = None
    season_games = 0

    observations = {stat: frame[stat].to_numpy(dtype=float) for stat in stats}
    seasons = frame["season"].to_numpy()

    for i in range(n):
        season = seasons[i]
        if season != current_season:
            if current_season is not None:
                prior_final = dict(continuous)
            within = {s: np.nan for s in stats}
            season_games = 0
            current_season = season

        weight = min(season_games / blend_full, 1.0) if blend_full > 0 else 1.0
        for stat in stats:
            in_season, prior = within[stat], prior_final[stat]
            shrunk_prior = carry * prior if not np.isnan(prior) else np.nan
            if np.isnan(in_season) and np.isnan(shrunk_prior):
                value = np.nan
            elif np.isnan(in_season):
                value = shrunk_prior
            elif np.isnan(shrunk_prior):
                value = in_season
            else:
                value = weight * in_season + (1.0 - weight) * shrunk_prior
            out[stat][i] = value
        games_played[i] = season_games

        # --- fold in this game's result only AFTER recording pregame state
        for stat in stats:
            observed = observations[stat][i]
            if np.isnan(observed):
                continue
            within[stat] = (
                observed if np.isnan(within[stat])
                else within[stat] + alpha * (observed - within[stat])
            )
            continuous[stat] = (
                observed if np.isnan(continuous[stat])
                else continuous[stat] + alpha * (observed - continuous[stat])
            )
        if not np.isnan(observations[stats[0]][i]):
            season_games += 1

    result = pd.DataFrame(out, index=frame.index)
    result["form_games"] = games_played
    return result


def _build_team_form(games: pd.DataFrame, team_epa: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (game_id, team) carrying that team's pregame form."""
    home = games[["game_id", "season", "week", "kickoff", "home_team"]].rename(
        columns={"home_team": "team"}
    )
    away = games[["game_id", "season", "week", "kickoff", "away_team"]].rename(
        columns={"away_team": "team"}
    )
    long = pd.concat([home, away], ignore_index=True)

    stats_available = [s for s in TEAM_STATS if s in team_epa.columns]
    if team_epa.empty or not stats_available:
        long[TEAM_STATS] = np.nan
        long["form_games"] = 0.0
        return long

    long = long.merge(
        team_epa[["game_id", "team"] + stats_available],
        on=["game_id", "team"],
        how="left",
    )
    for missing in set(TEAM_STATS) - set(stats_available):
        long[missing] = np.nan

    # Center on prior-seasons-only league means so 0 == league average.
    observed = long.dropna(subset=["off_epa_play"])
    baselines = _league_baselines(observed)
    centered = long.copy()
    for stat in TEAM_STATS:
        offsets = centered["season"].map(baselines[stat])
        centered[stat] = centered[stat] - offsets.astype(float)

    centered = centered.sort_values(
        ["team", "season", "week", "kickoff", "game_id"], kind="mergesort"
    )
    rolled = centered.groupby("team", group_keys=False, sort=False).apply(
        lambda frame: _roll_one_team(
            frame,
            TEAM_STATS,
            alpha=config.EPA_EWMA_ALPHA,
            carry=config.EPA_PRIOR_SEASON_CARRY,
            blend_full=config.EPA_BLEND_FULL_GAMES,
        ),
        include_groups=False,
    )

    form = centered[["game_id", "team"]].join(rolled)
    return form


# --------------------------------------------------------------------------
# Public builder
# --------------------------------------------------------------------------


def build_features(
    games: pd.DataFrame,
    team_epa: pd.DataFrame | None = None,
    *,
    elo_engine: EloEngine | None = None,
) -> pd.DataFrame:
    """Attach Elo, rolling-form and market columns to every game.

    Returns a frame indexed like ``games`` (one row per game) containing the
    identifier columns, the target, and every column in ``FEATURE_COLUMNS``
    plus ``MARKET_COLUMNS``.
    """
    games = games.sort_values(
        ["season", "week", "kickoff", "game_id"], kind="mergesort"
    ).reset_index(drop=True)
    team_epa = team_epa if team_epa is not None else pd.DataFrame()

    engine = elo_engine or EloEngine()
    elo_frame = engine.run(games, team_epa=team_epa if not team_epa.empty else None)

    form = _build_team_form(games, team_epa)
    stat_columns = [c for c in form.columns if c not in ("game_id", "team")]

    # Join on (game_id, team) -- joining on game_id alone would match both
    # sides of the same game and silently fan the frame out.
    home_form = form.rename(
        columns={"team": "home_team", **{c: f"home_{c}" for c in stat_columns}}
    )
    away_form = form.rename(
        columns={"team": "away_team", **{c: f"away_{c}" for c in stat_columns}}
    )

    out = games.merge(elo_frame, on="game_id", how="left")
    out = out.merge(home_form, on=["game_id", "home_team"], how="left")
    out = out.merge(away_form, on=["game_id", "away_team"], how="left")

    def edge(off_stat: str, def_stat: str) -> pd.Series:
        """Net matchup advantage for the home team, in centered stat units.

        A ``def_*`` column records what that defense *allows*, so a high value
        means a generous defense. The home team's expected production
        therefore rises both with its own offense and with how leaky the away
        defense has been -- the two terms add, they do not cancel.
        """
        home_expected = (
            out[f"home_{off_stat}"].fillna(0.0) + out[f"away_{def_stat}"].fillna(0.0)
        )
        away_expected = (
            out[f"away_{off_stat}"].fillna(0.0) + out[f"home_{def_stat}"].fillna(0.0)
        )
        return home_expected - away_expected

    out["epa_edge"] = edge("off_epa_play", "def_epa_play")
    out["pass_epa_edge"] = edge("off_pass_epa", "def_pass_epa")
    out["rush_epa_edge"] = edge("off_rush_epa", "def_rush_epa")
    out["success_edge"] = edge("off_success", "def_success")
    out["explosive_edge"] = edge("off_explosive", "def_explosive")
    # For turnovers and sacks a higher offensive value is *bad*, so the
    # matchup edge flips sign: fewer giveaways / sacks taken favours the team.
    out["turnover_edge"] = -edge("off_turnover_rate", "def_turnover_rate")
    out["sack_edge"] = -edge("off_sack_rate", "def_sack_rate")

    out["rest_diff"] = (out["home_rest"] - out["away_rest"]).fillna(0.0).clip(
        -config.REST_DIFF_CAP_DAYS, config.REST_DIFF_CAP_DAYS
    )
    out["div_game"] = out["div_game"].fillna(0.0).astype(float)
    out["neutral"] = out["neutral"].astype(float)
    out["is_playoff"] = out["is_playoff"].astype(float)

    # How much history backs the form estimate (0 in week 1, saturating later).
    out["form_confidence"] = np.minimum(
        out[["home_form_games", "away_form_games"]].min(axis=1).fillna(0.0), 8.0
    )

    out["market_spread"] = out["spread_line"].astype(float)
    out["has_market"] = out["market_spread"].notna().astype(float)

    for column in FEATURE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    return out
