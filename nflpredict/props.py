"""Player prop projections: passing, rushing and receiving yards.

A prop is a volume question multiplied by an efficiency question. A receiver
gains yards because his offence throws often, because he is the one they
throw to, and because he does something with it -- and the first two move
far more than the third. The features here are built in that shape: recent
workload, recent efficiency, the defence opposite, and how many points the
game model expects that offence to score.

Three things make this harder than the game markets, and all three are
visible in the measured error rather than argued away:

* **Who plays is not known.** For an upcoming game there are no player rows
  at all, so the projected roster is whoever has been playing for that team
  recently, filtered by the injury report. A player the depth chart has
  moved up will be projected on last month's role.
* **The distributions are not symmetric.** A receiver's yardage is long-
  tailed -- a typical game near his median with the occasional 120-yard
  outlier -- so mean absolute error understates how often a projection is
  badly wrong, and no normal-CDF over/under probability would be honest.
  None is offered.
* **Usage moves faster than form.** An injury to a team-mate can double a
  back's carries with no warning in his own history.

Leak-freedom uses the same guarantee as everywhere else: every rolling
figure is shifted a game before it is used, so a player's projection never
contains the game it is projecting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .players import injury_availability

__all__ = [
    "PropsPredictor", "build_player_features", "build_slate_rows",
    "projected_roster", "PROP_TARGETS", "PropFitReport",
]

SEED = 7

# What is projected, and the volume/efficiency pair behind each.
PROP_TARGETS: Dict[str, str] = {
    "passing_yards": "pass",
    "rushing_yards": "rush",
    "receiving_yards": "rec",
}

# Responsiveness of a player's rolling form. Faster than team form, because
# a role change shows up in a couple of games and then persists.
PROP_EWMA_ALPHA = 0.35

# A player needs to have been playing recently to be projected at all.
RECENT_GAMES_WINDOW = 4
MIN_RECENT_APPEARANCES = 1

PLAYER_FORM_COLUMNS: List[str] = [
    "form_pass_yards", "form_attempts", "form_pass_ypa",
    "form_rush_yards", "form_carries", "form_rush_ypc",
    "form_rec_yards", "form_targets", "form_rec_ypt", "form_target_share",
    "form_games",
]

CONTEXT_COLUMNS: List[str] = [
    "is_home", "team_proj_points", "opp_proj_points",
    "opp_pass_yards_allowed", "opp_rush_yards_allowed",
]

# Availability is deliberately NOT a feature, though it is carried alongside
# one. A projection is
#
#     E[yards] = P(he plays) x E[yards | he plays]
#
# and the training rows are already conditional on playing, because nflverse
# only emits a row for a player who appeared. So the model learns the second
# factor and the first is applied afterwards. Feeding availability in as well
# would discount a questionable player twice -- once through a fitted
# coefficient and again through the multiplier.

PROP_FEATURE_COLUMNS: List[str] = PLAYER_FORM_COLUMNS + CONTEXT_COLUMNS


@dataclass
class PropFitReport:
    """What a props fit saw, per target."""

    n_train: Dict[str, int] = field(default_factory=dict)
    train_mae: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Rolling player form
# --------------------------------------------------------------------------


def _rolling(frame: pd.DataFrame, column: str) -> pd.Series:
    """Exponentially-weighted mean of a player's *earlier* games only.

    ``shift(1)`` before the window is what makes this leak-free: the value
    recorded against a game is built from everything up to, and not
    including, that game.
    """
    return frame.groupby("player_id", sort=False)[column].transform(
        lambda s: s.shift(1).ewm(alpha=PROP_EWMA_ALPHA, min_periods=1).mean()
    )


def _defence_allowed(player_weeks: pd.DataFrame) -> pd.DataFrame:
    """Rolling passing and rushing yards each defence has been giving up."""
    per_game = player_weeks.groupby(
        ["opponent_team", "season", "week"], sort=False
    ).agg(
        pass_yards_allowed=("passing_yards", "sum"),
        rush_yards_allowed=("rushing_yards", "sum"),
    ).reset_index().rename(columns={"opponent_team": "team"})

    per_game = per_game.sort_values(["team", "season", "week"], kind="mergesort")
    for column in ("pass_yards_allowed", "rush_yards_allowed"):
        per_game[f"opp_{column}"] = per_game.groupby("team", sort=False)[
            column
        ].transform(
            lambda s: s.shift(1).ewm(alpha=PROP_EWMA_ALPHA, min_periods=1).mean()
        )
    return per_game[
        ["team", "season", "week", "opp_pass_yards_allowed", "opp_rush_yards_allowed"]
    ]


def build_player_features(
    player_weeks: pd.DataFrame,
    games: pd.DataFrame,
    injuries: pd.DataFrame | None = None,
    team_points: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per player-game with pregame form, context, and the outcome.

    ``team_points`` optionally supplies the game model's projected points per
    team (columns ``game_id``, ``team``, ``proj_points``). Without it the
    scoring-environment features fall back to the league average, which
    costs accuracy but keeps the frame usable.
    """
    if player_weeks is None or player_weeks.empty:
        return pd.DataFrame()

    frame = player_weeks.copy()
    for column in (
        "passing_yards", "attempts", "rushing_yards", "carries",
        "receiving_yards", "targets", "target_share",
    ):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce").fillna(0.0)

    frame = frame.sort_values(
        ["player_id", "season", "week"], kind="mergesort"
    ).reset_index(drop=True)

    frame["form_pass_yards"] = _rolling(frame, "passing_yards")
    frame["form_attempts"] = _rolling(frame, "attempts")
    frame["form_rush_yards"] = _rolling(frame, "rushing_yards")
    frame["form_carries"] = _rolling(frame, "carries")
    frame["form_rec_yards"] = _rolling(frame, "receiving_yards")
    frame["form_targets"] = _rolling(frame, "targets")
    frame["form_target_share"] = _rolling(frame, "target_share")

    # Efficiency from the rolled pair, not rolled separately: a per-game
    # average of ratios is dominated by the games with almost no volume.
    frame["form_pass_ypa"] = _safe_ratio(
        frame["form_pass_yards"], frame["form_attempts"], RATE_BOUNDS["form_pass_ypa"]
    )
    frame["form_rush_ypc"] = _safe_ratio(
        frame["form_rush_yards"], frame["form_carries"], RATE_BOUNDS["form_rush_ypc"]
    )
    frame["form_rec_ypt"] = _safe_ratio(
        frame["form_rec_yards"], frame["form_targets"], RATE_BOUNDS["form_rec_ypt"]
    )

    frame["form_games"] = frame.groupby("player_id", sort=False).cumcount()

    defence = _defence_allowed(player_weeks)
    frame = frame.merge(
        defence.rename(columns={"team": "opponent_team"}),
        on=["opponent_team", "season", "week"],
        how="left",
    )

    schedule = games[["game_id", "home_team", "away_team"]]
    frame = frame.merge(schedule, on="game_id", how="left")
    frame["is_home"] = (frame["team"] == frame["home_team"]).astype(float)

    frame = _attach_availability(frame, injuries)
    frame = _attach_team_points(frame, team_points)

    for column in PROP_FEATURE_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


# A rate needs a denominator big enough to mean something. Guarding only
# against exact zero is not enough: an exponentially-weighted carry count
# decays toward zero for a player who has stopped playing, and dividing by
# 1e-9 produced yards-per-carry in the hundreds of millions, which blew up
# the regression outright. Rates are therefore floored on volume and clipped
# to the range football actually produces.
_MIN_RATE_VOLUME = 0.5

# Generous bounds -- wide enough never to bind on a real season, tight enough
# that a degenerate ratio cannot escape.
RATE_BOUNDS = {
    "form_pass_ypa": (0.0, 20.0),
    "form_rush_ypc": (-5.0, 15.0),
    "form_rec_ypt": (0.0, 30.0),
}


def _safe_ratio(
    numerator: pd.Series, denominator: pd.Series, bounds: tuple[float, float]
) -> pd.Series:
    """Rate of ``numerator`` per ``denominator``, or 0 where there is no volume."""
    usable = denominator >= _MIN_RATE_VOLUME
    ratio = (numerator / denominator.where(usable)).fillna(0.0)
    return ratio.clip(*bounds)


def _attach_availability(
    frame: pd.DataFrame, injuries: pd.DataFrame | None
) -> pd.DataFrame:
    """Join the injury report; anyone not listed is treated as available."""
    if injuries is None or injuries.empty:
        frame["availability"] = 1.0
        return frame

    report = injuries[["season", "week", "gsis_id", "report_status"]].dropna(
        subset=["gsis_id"]
    ).rename(columns={"gsis_id": "player_id"}).drop_duplicates(
        ["season", "week", "player_id"]
    )
    frame = frame.merge(report, on=["player_id", "season", "week"], how="left")
    frame["availability"] = frame["report_status"].map(injury_availability).fillna(1.0)
    return frame.drop(columns=["report_status"])


def _attach_team_points(
    frame: pd.DataFrame, team_points: pd.DataFrame | None
) -> pd.DataFrame:
    """Join the game model's projected points for each side of the matchup."""
    league_average = 22.5
    if team_points is None or team_points.empty:
        frame["team_proj_points"] = league_average
        frame["opp_proj_points"] = league_average
        return frame

    points = team_points[["game_id", "team", "proj_points"]]
    frame = frame.merge(
        points.rename(columns={"proj_points": "team_proj_points"}),
        on=["game_id", "team"], how="left",
    )
    frame = frame.merge(
        points.rename(columns={"team": "opponent_team", "proj_points": "opp_proj_points"}),
        on=["game_id", "opponent_team"], how="left",
    )
    frame["team_proj_points"] = frame["team_proj_points"].fillna(league_average)
    frame["opp_proj_points"] = frame["opp_proj_points"].fillna(league_average)
    return frame


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


class PropsPredictor:
    """Fit-on-history, predict-forward player yardage projections."""

    def __init__(self, *, use_gbm: bool = True, seed: int = SEED) -> None:
        self.use_gbm = use_gbm
        self.seed = seed
        self.features = list(PROP_FEATURE_COLUMNS)
        self._models: Dict[str, tuple] = {}
        self.report = PropFitReport()
        self.fitted = False

    def fit(self, train: pd.DataFrame) -> "PropsPredictor":
        """Fit one model per target on already-played player-games."""
        if train.empty:
            raise ValueError("no player-games to train on")

        self.report.feature_names = list(self.features)
        for target in PROP_TARGETS:
            # Only players who were actually involved carry signal for a
            # target; a lineman's zero receiving yards teaches nothing.
            relevant = train[train[target].notna()]
            relevant = relevant[
                (relevant[target] > 0) | (relevant[_volume_for(target)] > 0)
            ]
            if len(relevant) < 500:
                continue

            X = relevant[self.features].to_numpy(dtype=float)
            y = relevant[target].to_numpy(dtype=float)

            ridge = Pipeline(
                [("scale", StandardScaler()), ("reg", Ridge(alpha=10.0))]
            ).fit(X, y)
            gbm = None
            if self.use_gbm and len(relevant) >= 2000:
                gbm = HistGradientBoostingRegressor(
                    max_depth=3, max_iter=220, learning_rate=0.045,
                    min_samples_leaf=40, l2_regularization=1.0,
                    random_state=self.seed,
                ).fit(X, y)

            self._models[target] = (ridge, gbm)
            self.report.n_train[target] = len(relevant)
            self.report.train_mae[target] = float(
                np.mean(np.abs(y - self._blend(target, X)))
            )

        if not self._models:
            raise ValueError("no target had enough player-games to fit")
        self.fitted = True
        return self

    def _blend(self, target: str, X: np.ndarray) -> np.ndarray:
        ridge, gbm = self._models[target]
        if gbm is None:
            return ridge.predict(X)
        return 0.5 * ridge.predict(X) + 0.5 * gbm.predict(X)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Project each target for every player-row in ``frame``.

        Projections are floored at zero -- a linear model will happily
        predict negative receiving yards for a player with no role, and a
        negative projection is not a forecast, it is an artefact.
        """
        if not self.fitted:
            raise RuntimeError("PropsPredictor.predict called before fit")

        X = frame[self.features].to_numpy(dtype=float)
        out = pd.DataFrame(index=frame.index)
        for column in ("player_id", "player_display_name", "position", "team", "game_id"):
            if column in frame.columns:
                out[column] = frame[column].to_numpy()

        for target in PROP_TARGETS:
            name = f"proj_{target}"
            if target not in self._models:
                out[name] = np.nan
                continue
            projection = np.maximum(self._blend(target, X), 0.0)
            # An unavailable player is projected for what he is expected to
            # contribute, which is his production scaled by his chance of
            # taking the field at all.
            out[name] = projection * frame["availability"].to_numpy(dtype=float)
        return out


def _volume_for(target: str) -> str:
    return {
        "passing_yards": "attempts",
        "rushing_yards": "carries",
        "receiving_yards": "targets",
    }[target]


def projected_roster(
    player_weeks: pd.DataFrame,
    season: int,
    week: int,
    teams: List[str],
    *,
    window: int = RECENT_GAMES_WINDOW,
) -> pd.DataFrame:
    """Who to project for an upcoming slate.

    An upcoming game has no player rows, so the roster is whoever has
    appeared for each team in the last ``window`` weeks. This is the honest
    weak point of the whole exercise: a player promoted up the depth chart
    this week is projected on the role he had last month, and a player
    signed since is not projected at all.
    """
    period = season * 100 + week
    history = player_weeks[
        (player_weeks["season"] * 100 + player_weeks["week"] < period)
        & (player_weeks["season"] * 100 + player_weeks["week"] >= period - window)
        & player_weeks["team"].isin(teams)
    ]
    if history.empty:
        return pd.DataFrame()

    appearances = history.groupby(
        ["player_id", "player_display_name", "position", "team"], sort=False
    ).size().reset_index(name="appearances")
    return appearances[appearances["appearances"] >= MIN_RECENT_APPEARANCES]


def build_slate_rows(
    player_weeks: pd.DataFrame,
    games: pd.DataFrame,
    slate: pd.DataFrame,
    injuries: pd.DataFrame | None = None,
    team_points: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature rows for the players expected to appear in an upcoming slate.

    An upcoming game has no player rows at all, so they are created: one per
    projected player per game, with the statistics left empty. The rolling
    form machinery then fills them from that player's earlier games exactly
    as it would for a game already played -- which is the whole point, since
    the form is shifted a game anyway.
    """
    if player_weeks is None or player_weeks.empty or slate.empty:
        return pd.DataFrame()

    teams = sorted(set(slate["home_team"]) | set(slate["away_team"]))
    season = int(slate["season"].iloc[0])
    week = int(slate["week"].iloc[0])
    roster = projected_roster(player_weeks, season, week, teams)
    if roster.empty:
        return pd.DataFrame()

    # Pair each projected player with his team's game on this slate.
    matchups = []
    for side, opponent in (("home_team", "away_team"), ("away_team", "home_team")):
        matchups.append(
            slate[["game_id", "season", "week", side, opponent]].rename(
                columns={side: "team", opponent: "opponent_team"}
            )
        )
    matchups = pd.concat(matchups, ignore_index=True)

    rows = roster.merge(matchups, on="team", how="inner")
    if rows.empty:
        return pd.DataFrame()

    # Empty statistics: this game has not happened, so it contributes nothing
    # to anyone's form. The shift(1) window never reads it in any case.
    for column in player_weeks.columns:
        if column not in rows.columns:
            rows[column] = np.nan

    combined = pd.concat(
        [player_weeks, rows[player_weeks.columns]], ignore_index=True, sort=False
    )
    built = build_player_features(combined, games, injuries, team_points)
    return built[built["game_id"].isin(set(slate["game_id"]))].copy()
