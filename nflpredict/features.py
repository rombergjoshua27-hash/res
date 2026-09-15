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

__all__ = [
    "build_features", "FEATURE_COLUMNS", "TOTAL_FEATURE_COLUMNS",
    "MARKET_COLUMNS", "MARKET_TOTAL_COLUMNS", "TEAM_STATS", "TOTALS_STATS",
    "ROLLED_STATS",
]


# Raw per-game team statistics that get rolled forward into "form".
TEAM_STATS: List[str] = [
    "off_epa_play", "off_pass_epa", "off_rush_epa", "off_success",
    "off_explosive", "off_turnover_rate", "off_sack_rate",
    "def_epa_play", "def_pass_epa", "def_rush_epa", "def_success",
    "def_explosive", "def_turnover_rate", "def_sack_rate",
]

# Rolled forward by the same machinery, but consumed only by the totals
# model. Kept in a separate list so the win model's validated feature set is
# untouched by anything added for scoring.
#
# Points come from the game log rather than play-by-play, so they are
# available for every season; possessions and pace come from play-by-play.
TOTALS_STATS: List[str] = [
    "points_for", "points_against",
    "off_plays", "off_drives", "off_plays_per_drive", "off_pass_rate",
    "def_plays", "def_drives", "def_plays_per_drive", "def_pass_rate",
]

# Everything the rolling pass produces, in one list.
ROLLED_STATS: List[str] = TEAM_STATS + TOTALS_STATS

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

# Inputs to the totals model. Like the win model these never touch a
# betting market -- the posted total is blended in afterwards.
TOTAL_FEATURE_COLUMNS: List[str] = [
    "scoring_sum",        # both offences' recent scoring, both defences' leakiness
    "off_epa_sum",        # combined offensive efficiency
    "def_epa_sum",        # combined defensive generosity
    "explosive_sum",      # big plays create points quickly
    "turnover_sum",       # giveaways shorten fields and add possessions
    "plays_sum",          # combined expected snaps
    "drives_sum",         # combined expected possessions
    "plays_per_drive_sum",
    "pass_rate_sum",      # pass-heavy games stop the clock and run longer
    "wind_speed",
    "temperature",
    "is_indoor",
    "div_game",
    "is_playoff",
    "form_confidence",
]

# Market-derived inputs, kept separate so the model can be run with or
# without them (`--no-market` isolates genuine model edge).
MARKET_COLUMNS: List[str] = ["market_spread"]
MARKET_TOTAL_COLUMNS: List[str] = ["market_total"]


# --------------------------------------------------------------------------
# League baselines
# --------------------------------------------------------------------------


def _league_baselines(team_games: pd.DataFrame, stats: Sequence[str]) -> pd.DataFrame:
    """Mean of each stat over all seasons *strictly before* each season.

    Centering on a prior-seasons-only baseline removes era drift -- offensive
    efficiency and scoring have both risen steadily -- without letting the
    current season inform its own normalization.

    Sums and counts are accumulated per statistic rather than per row,
    because availability differs by column: points come from the game log and
    exist for every season, while possessions and efficiency come from
    play-by-play and only exist once it has been loaded.

    ONE DOCUMENTED EXCEPTION. The first season in which a statistic appears
    has no earlier season to centre on, so it is centred on its own mean.
    That single season's baseline therefore contains a trace of its own
    games -- roughly one part in five hundred per game, spread across a
    league-wide average. Every later season is strictly prior-only, which
    ``tests/test_leakage.py`` asserts directly.

    The exception is contained rather than tolerated: the first season of
    loaded data is a warm-up season that the backtest never scores. The
    defaults leave eight seasons between the two (play-by-play from 2006,
    scoring from 2007), and ``cli`` refuses a ``--start`` that would reach
    back into the warm-up season.
    """
    stats = list(stats)
    sums = team_games.groupby("season")[stats].sum(min_count=1)
    counts = team_games.groupby("season")[stats].count().astype(float)

    baselines = {}
    running_sum = pd.Series(0.0, index=stats)
    running_n = pd.Series(0.0, index=stats)
    for season in sorted(sums.index):
        season_sum = sums.loc[season].fillna(0.0)
        season_n = counts.loc[season]
        prior_mean = running_sum / running_n.replace(0.0, np.nan)
        own_mean = season_sum / season_n.replace(0.0, np.nan)
        # No prior history for this stat yet -> centre it on its own season.
        baselines[season] = prior_mean.where(running_n > 0, own_mean)
        running_sum = running_sum + season_sum
        running_n = running_n + season_n

    return pd.DataFrame(baselines).T.reindex(columns=stats)


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
    base = ["game_id", "season", "week", "kickoff"]
    home = games[base + ["home_team", "home_score", "away_score"]].rename(
        columns={
            "home_team": "team",
            "home_score": "points_for",
            "away_score": "points_against",
        }
    )
    away = games[base + ["away_team", "away_score", "home_score"]].rename(
        columns={
            "away_team": "team",
            "away_score": "points_for",
            "home_score": "points_against",
        }
    )
    long = pd.concat([home, away], ignore_index=True)

    # Points come from the game log, so they are present whether or not
    # play-by-play has been loaded. Everything else needs the EPA cache.
    epa_stats = [s for s in ROLLED_STATS if s not in ("points_for", "points_against")]
    stats_available = [s for s in epa_stats if s in team_epa.columns]
    if not team_epa.empty and stats_available:
        long = long.merge(
            team_epa[["game_id", "team"] + stats_available],
            on=["game_id", "team"],
            how="left",
        )
    for missing in set(epa_stats) - set(stats_available):
        long[missing] = np.nan

    # Center on prior-seasons-only league means so 0 == league average.
    baselines = _league_baselines(long, ROLLED_STATS)
    centered = long.copy()
    for stat in ROLLED_STATS:
        offsets = centered["season"].map(baselines[stat])
        centered[stat] = centered[stat] - offsets.astype(float)

    centered = centered.sort_values(
        ["team", "season", "week", "kickoff", "game_id"], kind="mergesort"
    )
    rolled = centered.groupby("team", group_keys=False, sort=False).apply(
        lambda frame: _roll_one_team(
            frame,
            ROLLED_STATS,
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

    _attach_totals_features(out)

    out["market_spread"] = out["spread_line"].astype(float)
    out["has_market"] = out["market_spread"].notna().astype(float)
    out["market_total"] = out["total_line"].astype(float)
    out["has_market_total"] = out["market_total"].notna().astype(float)

    for column in FEATURE_COLUMNS + TOTAL_FEATURE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    return out


# --------------------------------------------------------------------------
# Totals features
# --------------------------------------------------------------------------

# Indoors there is no wind and the thermostat sits in the 60s-70s, so a
# closed roof is encoded as calm and mild rather than as missing weather.
INDOOR_ROOFS = ("dome", "closed")
INDOOR_TEMPERATURE_F = 70.0


def _attach_totals_features(out: pd.DataFrame) -> None:
    """Add the combined-scoring features the totals model consumes, in place.

    Every term is a *sum* across the two teams rather than a difference. A
    total does not care who is better -- two good offences and two bad ones
    both push the number up, so the sides add.
    """

    def pair(stat: str) -> pd.Series:
        """Both teams' rolling value for ``stat``, added together."""
        return out[f"home_{stat}"].fillna(0.0) + out[f"away_{stat}"].fillna(0.0)

    # Recent scoring, counting both what each side puts up and what each side
    # gives up -- averaged so the scale stays in points-per-team-game.
    out["scoring_sum"] = (
        pair("points_for") + pair("points_against")
    ) / 2.0

    out["off_epa_sum"] = pair("off_epa_play")
    # A high def_* value means a generous defence, so this also adds.
    out["def_epa_sum"] = pair("def_epa_play")
    out["explosive_sum"] = pair("off_explosive") + pair("def_explosive")
    out["turnover_sum"] = pair("off_turnover_rate") + pair("def_turnover_rate")
    out["plays_sum"] = pair("off_plays") + pair("def_plays")
    out["drives_sum"] = pair("off_drives") + pair("def_drives")
    out["plays_per_drive_sum"] = pair("off_plays_per_drive") + pair("def_plays_per_drive")
    out["pass_rate_sum"] = pair("off_pass_rate") + pair("def_pass_rate")

    roof = out.get("roof", pd.Series("outdoors", index=out.index)).astype(str)
    indoor = roof.isin(INDOOR_ROOFS)
    out["is_indoor"] = indoor.astype(float)

    wind = pd.to_numeric(out.get("wind"), errors="coerce")
    temp = pd.to_numeric(out.get("temp"), errors="coerce")
    # Indoors the reading is absent because it does not apply, not because it
    # is unknown; outdoors an absent reading falls back to a mild, calm day so
    # a missing value never reads as a storm.
    out["wind_speed"] = wind.where(~indoor, 0.0).fillna(0.0).clip(0.0, 35.0)
    out["temperature"] = (
        temp.where(~indoor, INDOOR_TEMPERATURE_F)
        .fillna(INDOOR_TEMPERATURE_F)
        .clip(-10.0, 110.0)
    )
