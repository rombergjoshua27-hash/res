"""Elo rating engine with margin-of-victory, home field, rest and QB terms.

The engine processes games in strict chronological order and exposes the
pregame rating for every game. Because a rating is always read *before* the
game that updates it, Elo features are leak-free by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from . import config

__all__ = ["EloEngine", "EloState", "estimate_hfa", "elo_to_prob", "elo_to_spread"]


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------


def elo_to_prob(elo_diff: float) -> float:
    """Win probability for the team holding ``elo_diff`` points of advantage."""
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / config.ELO_DIVISOR))


def elo_to_spread(elo_diff: float) -> float:
    """Expected scoring margin implied by an Elo difference."""
    return elo_diff / config.ELO_PER_POINT


def _mov_multiplier(margin: float, winner_elo_diff: float) -> float:
    """FiveThirtyEight margin-of-victory multiplier.

    Scales the rating update by the size of the win, while damping blowouts
    by an already-favoured team so ratings cannot be farmed against weak
    opponents.
    """
    return float(
        np.log(abs(margin) + 1.0)
        * (config.ELO_MOV_B / ((winner_elo_diff * 0.001) + config.ELO_MOV_B))
    )


def estimate_hfa(games: pd.DataFrame, season: int) -> float:
    """Estimate home field advantage in points from recent completed seasons.

    Uses a trailing window strictly *before* ``season`` so the estimate never
    sees the season it is applied to.
    """
    if season in config.HFA_SEASON_OVERRIDES:
        return config.HFA_SEASON_OVERRIDES[season]

    window = games[
        (games["season"] < season)
        & (games["season"] >= season - config.HFA_TRAILING_SEASONS)
        & games["completed"]
        & ~games["neutral"]
        & ~games["is_playoff"]
    ]
    if len(window) < 100:
        return config.HFA_POINTS_DEFAULT

    # Exclude the fanless 2020 season from any trailing estimate.
    window = window[window["season"] != 2020]
    if len(window) < 100:
        return config.HFA_POINTS_DEFAULT

    hfa = float(window["margin"].mean())
    return float(np.clip(hfa, config.HFA_MIN_POINTS, config.HFA_MAX_POINTS))


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


@dataclass
class EloState:
    """Mutable rating state carried through a chronological pass over games."""

    ratings: Dict[str, float] = field(default_factory=dict)
    qb_rating: Dict[str, float] = field(default_factory=dict)
    qb_games: Dict[str, float] = field(default_factory=dict)
    team_qb_baseline: Dict[str, float] = field(default_factory=dict)
    last_season: int | None = None

    def rating(self, team: str) -> float:
        return self.ratings.get(team, config.ELO_INIT)

    def carry_to_season(self, season: int) -> None:
        """Regress every rating toward the mean at a season boundary."""
        if self.last_season is None or season == self.last_season:
            self.last_season = season
            return
        for _ in range(season - self.last_season):
            for team, value in self.ratings.items():
                self.ratings[team] = config.ELO_INIT + config.ELO_SEASON_CARRY * (
                    value - config.ELO_INIT
                )
        self.last_season = season


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


class EloEngine:
    """Chronological Elo pass producing pregame ratings for every game.

    Parameters mirror ``config`` but can be overridden for tuning sweeps.
    """

    def __init__(
        self,
        *,
        k: float = config.ELO_K,
        season_carry: float = config.ELO_SEASON_CARRY,
        use_qb: bool = config.ELO_USE_QB_DEFAULT,
        use_rest: bool = True,
        hfa_override: float | None = None,
    ) -> None:
        self.k = k
        self.season_carry = season_carry
        self.use_qb = use_qb
        self.use_rest = use_rest
        self.hfa_override = hfa_override
        self.state = EloState()

    # -- adjustments ------------------------------------------------------

    def _rest_points(self, home_rest: float, away_rest: float) -> float:
        if not self.use_rest or pd.isna(home_rest) or pd.isna(away_rest):
            return 0.0
        diff = float(
            np.clip(
                home_rest - away_rest,
                -config.REST_DIFF_CAP_DAYS,
                config.REST_DIFF_CAP_DAYS,
            )
        )
        return diff * config.REST_POINTS_PER_DAY

    def _qb_points(self, team: str, qb_id: object) -> float:
        """Points of adjustment for starting ``qb_id`` instead of the usual starter."""
        if not self.use_qb or not isinstance(qb_id, str) or not qb_id:
            return 0.0
        baseline = self.state.team_qb_baseline.get(team)
        if baseline is None:
            return 0.0
        rating = self.state.qb_rating.get(qb_id)
        if rating is None:
            # Unknown starter (rookie, practice-squad callup): shade toward
            # replacement level rather than assuming the usual starter.
            rating = baseline - 0.03
        points = (rating - baseline) * config.QB_EPA_TO_POINTS
        return float(
            np.clip(points, -config.QB_ADJ_CAP_POINTS, config.QB_ADJ_CAP_POINTS)
        )

    def hfa_points(self, games: pd.DataFrame, season: int) -> float:
        if self.hfa_override is not None:
            return self.hfa_override
        return estimate_hfa(games, season)

    # -- main pass --------------------------------------------------------

    def run(
        self,
        games: pd.DataFrame,
        *,
        team_epa: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Walk ``games`` in order, recording pregame ratings and updating after.

        ``team_epa`` (optional) supplies per-game offensive EPA used to keep
        quarterback ratings current. Returns one row per input game with the
        pregame state; games without a final score are rated but never used
        to update.
        """
        games = games.sort_values(
            ["season", "week", "kickoff", "game_id"], kind="mergesort"
        ).reset_index(drop=True)

        qb_epa_lookup = self._build_qb_epa_lookup(team_epa)
        hfa_cache: Dict[int, float] = {}
        records = []

        for row in games.itertuples(index=False):
            season = int(row.season)
            self.state.carry_to_season(season)

            if season not in hfa_cache:
                hfa_cache[season] = self.hfa_points(games, season)
            hfa = 0.0 if row.neutral else hfa_cache[season]

            home_elo = self.state.rating(row.home_team)
            away_elo = self.state.rating(row.away_team)

            rest_points = self._rest_points(row.home_rest, row.away_rest)
            qb_points = self._qb_points(
                row.home_team, getattr(row, "home_qb_id", None)
            ) - self._qb_points(row.away_team, getattr(row, "away_qb_id", None))

            adj_points = hfa + rest_points + qb_points
            elo_diff = (home_elo - away_elo) + adj_points * config.ELO_PER_POINT
            prob_home = elo_to_prob(elo_diff)

            records.append(
                {
                    "game_id": row.game_id,
                    "elo_home_pre": home_elo,
                    "elo_away_pre": away_elo,
                    "elo_diff": elo_diff,
                    "elo_raw_diff": home_elo - away_elo,
                    "elo_hfa_points": hfa,
                    "elo_rest_points": rest_points,
                    "elo_qb_points": qb_points,
                    "elo_prob_home": prob_home,
                    "elo_spread": elo_to_spread(elo_diff),
                }
            )

            if not row.completed:
                continue

            self._update(row, elo_diff, prob_home)
            self._update_qb(row, qb_epa_lookup)

        return pd.DataFrame.from_records(records)

    def _update(self, row, elo_diff: float, prob_home: float) -> None:
        margin = float(row.margin)
        actual = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)

        # Damping uses the *winner's* pregame edge, so flip the sign when the
        # away team wins.
        winner_edge = elo_diff if margin > 0 else -elo_diff
        mult = _mov_multiplier(margin, winner_edge) if margin != 0 else 1.0

        shift = self.k * mult * (actual - prob_home)
        self.state.ratings[row.home_team] = self.state.rating(row.home_team) + shift
        self.state.ratings[row.away_team] = self.state.rating(row.away_team) - shift

    # -- quarterback ratings ---------------------------------------------

    @staticmethod
    def _build_qb_epa_lookup(team_epa: pd.DataFrame | None) -> Dict[tuple, float]:
        if team_epa is None or team_epa.empty:
            return {}
        subset = team_epa[["game_id", "team", "off_epa_play"]].dropna()
        return {
            (game_id, team): float(value)
            for game_id, team, value in subset.itertuples(index=False)
        }

    def _update_qb(self, row, qb_epa_lookup: Dict[tuple, float]) -> None:
        """Attribute the game's offensive EPA to whoever started at QB."""
        if not self.use_qb or not qb_epa_lookup:
            return
        for team, qb_id in (
            (row.home_team, getattr(row, "home_qb_id", None)),
            (row.away_team, getattr(row, "away_qb_id", None)),
        ):
            if not isinstance(qb_id, str) or not qb_id:
                continue
            observed = qb_epa_lookup.get((row.game_id, team))
            if observed is None:
                continue

            games_started = self.state.qb_games.get(qb_id, 0.0)
            current = self.state.qb_rating.get(qb_id, 0.0)
            # Shrink toward league average until a QB has a real sample.
            weight = config.QB_EWMA_ALPHA * (
                games_started / (games_started + config.QB_SHRINK_GAMES)
                if games_started > 0
                else 1.0 / config.QB_SHRINK_GAMES
            )
            self.state.qb_rating[qb_id] = current + max(weight, 0.05) * (
                observed - current
            )
            self.state.qb_games[qb_id] = games_started + 1.0

            baseline = self.state.team_qb_baseline.get(team, self.state.qb_rating[qb_id])
            self.state.team_qb_baseline[team] = baseline + config.QB_EWMA_ALPHA * (
                self.state.qb_rating[qb_id] - baseline
            )

    # -- reporting --------------------------------------------------------

    def current_ratings(self) -> pd.DataFrame:
        rows = [
            {"team": team, "elo": value, "spread_vs_average": elo_to_spread(value - config.ELO_INIT)}
            for team, value in self.state.ratings.items()
        ]
        table = pd.DataFrame(rows).sort_values("elo", ascending=False)
        return table.reset_index(drop=True)
