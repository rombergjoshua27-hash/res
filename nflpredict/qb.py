"""Quarterback value and roster availability.

This is the second attempt at a quarterback adjustment. The first one is
documented in ``config.py`` as a failure, and understanding exactly how it
failed is what shapes this one.

**What went wrong before.** A passer was rated by his team's offensive EPA
per play. That is mostly a property of the team, not the passer -- and the
team's strength was already carried by its Elo rating, so the adjustment
re-applied a signal the model already had. It then folded that into Elo at
a hand-picked scale. Accuracy fell about a point at every scale tried.

**What is different here.**

1. *Passer-isolating inputs.* Completion percentage over expected is
   computed per throw against the difficulty of that throw, and EPA is taken
   per dropback rather than per team play, with sacks counted as the failed
   dropbacks they are. Neither is a restatement of team quality.

2. *A difference, not a level.* The feature is not "how good is this
   quarterback" -- Elo already knows roughly that. It is "how far is this
   week's starter from the quarterback this team has been playing", which is
   the part Elo cannot know. When the usual starter plays, it is ~0 by
   construction, so there is nothing to double-count.

3. *A learned weight, not a chosen one.* The previous version hardcoded
   points-per-EPA and a cap. Here the terms enter the feature matrix and the
   model fits their coefficients alongside ``elo_diff``, so it can apportion
   credit between them instead of being told how.

Availability works the same way: the injury report is turned into the share
of a team's recent receiving and rushing production that is expected to be
unavailable, and the model decides what that is worth.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from . import config

__all__ = ["build_qb_features", "build_availability_features", "QB_FEATURES"]

QB_FEATURES: List[str] = ["qb_delta", "qb_edge", "availability_edge"]

# Responsiveness of a passer's own rolling rating, and how many dropbacks of
# league-average prior he carries before it is trusted. A quarterback with
# two starts should not be rated as confidently as one with sixty.
QB_EWMA_ALPHA = 0.25
QB_SHRINK_DROPBACKS = 250.0

# How quickly a team's "who has been playing quarterback here" baseline
# follows a change of starter. Deliberately slower than the passer's own
# rating: the point of the baseline is to represent what Elo has absorbed,
# and Elo absorbs a new starter over several games, not instantly.
TEAM_QB_ALPHA = 0.20

_MIN_DROPBACKS = 5.0

# Dispersion of each centered measure across passer-games, used to put the
# two on one scale before averaging. EMPIRICAL over 1999-2026 (15,773
# passer-games with at least five dropbacks): cpoe 9.93, epa/dropback 0.360.
# Both are stable enough across eras to fix rather than re-measure --
# cpoe ranges 9.57-10.52 and epa 0.343-0.374 across six-season blocks -- and
# fixing them keeps a game's feature value independent of later games.
#
# CPOE itself only exists from 2006, when nflverse's air-yards charting
# begins; before that a passer is rated on EPA per dropback alone.
CPOE_SCALE = 9.93
EPA_SCALE = 0.360


def _season_baselines(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Prior-seasons-only league mean for each column, as in ``features``."""
    sums = frame.groupby("season")[columns].sum(min_count=1)
    counts = frame.groupby("season")[columns].count().astype(float)

    out: Dict[int, pd.Series] = {}
    running_sum = pd.Series(0.0, index=columns)
    running_n = pd.Series(0.0, index=columns)
    for season in sorted(sums.index):
        season_sum = sums.loc[season].fillna(0.0)
        season_n = counts.loc[season]
        prior = running_sum / running_n.replace(0.0, np.nan)
        own = season_sum / season_n.replace(0.0, np.nan)
        out[season] = prior.where(running_n > 0, own)
        running_sum = running_sum + season_sum
        running_n = running_n + season_n
    return pd.DataFrame(out).T.reindex(columns=columns)


def _roll_passer(frame: pd.DataFrame) -> pd.DataFrame:
    """Record each passer's rating *before* the game it is attached to."""
    n = len(frame)
    rating = np.full(n, np.nan)
    experience = np.zeros(n)

    cpoe_state = np.nan
    epa_state = np.nan
    dropbacks_seen = 0.0

    cpoe = frame["cpoe_centered"].to_numpy(dtype=float)
    epa = frame["epa_centered"].to_numpy(dtype=float)
    dropbacks = frame["dropbacks"].to_numpy(dtype=float)

    rating_post = np.full(n, np.nan)

    def current_rating() -> float:
        parts = [v for v in (cpoe_state, epa_state) if not np.isnan(v)]
        if not parts:
            return np.nan
        # Shrink toward league average until enough dropbacks accumulate.
        trust = dropbacks_seen / (dropbacks_seen + QB_SHRINK_DROPBACKS)
        return float(np.mean(parts)) * trust

    for i in range(n):
        rating[i] = current_rating()
        experience[i] = dropbacks_seen

        # --- fold this game in only after recording the pregame state
        if dropbacks[i] >= _MIN_DROPBACKS:
            for value, state_name in ((cpoe[i], "cpoe"), (epa[i], "epa")):
                if np.isnan(value):
                    continue
                current = cpoe_state if state_name == "cpoe" else epa_state
                updated = (
                    value if np.isnan(current)
                    else current + QB_EWMA_ALPHA * (value - current)
                )
                if state_name == "cpoe":
                    cpoe_state = updated
                else:
                    epa_state = updated
            dropbacks_seen += dropbacks[i]

        # State *after* this game, which is what a later week looks up.
        rating_post[i] = current_rating()

    return pd.DataFrame(
        {
            "qb_rating_pre": rating,
            "qb_rating_post": rating_post,
            "qb_dropbacks_pre": experience,
        },
        index=frame.index,
    )


def _passer_ratings(player_weeks: pd.DataFrame) -> pd.DataFrame:
    """Pregame rating for every passer-game, keyed by (player_id, game_id)."""
    passers = player_weeks[
        player_weeks["dropbacks"].fillna(0.0) >= _MIN_DROPBACKS
    ].copy()
    if passers.empty:
        return pd.DataFrame(
            columns=["player_id", "game_id", "qb_rating_pre", "qb_dropbacks_pre"]
        )

    # Both measures are per-dropback rates, so they are comparable once
    # centered on their own prior-seasons league mean and put on one scale.
    passers["cpoe_raw"] = passers["passing_cpoe"]
    passers["epa_raw"] = passers["passing_epa_per_dropback"]

    baselines = _season_baselines(
        passers.rename(columns={"cpoe_raw": "cpoe", "epa_raw": "epa"}), ["cpoe", "epa"]
    )
    passers["cpoe_centered"] = passers["cpoe_raw"] - passers["season"].map(
        baselines["cpoe"]
    ).astype(float)
    passers["epa_centered"] = passers["epa_raw"] - passers["season"].map(
        baselines["epa"]
    ).astype(float)

    # Put the two on a common scale so averaging them is meaningful. The
    # divisors are fixed constants rather than the dispersion of whatever
    # happens to be loaded: measuring them from the frame would let a game's
    # feature value depend on games played after it, which is a small leak
    # but a leak. See CPOE_SCALE / EPA_SCALE for the measurement.
    passers["cpoe_centered"] = passers["cpoe_centered"] / CPOE_SCALE
    passers["epa_centered"] = passers["epa_centered"] / EPA_SCALE

    passers = passers.sort_values(
        ["player_id", "season", "week"], kind="mergesort"
    )
    rolled = passers.groupby("player_id", group_keys=False, sort=False).apply(
        _roll_passer, include_groups=False
    )
    out = passers[["player_id", "game_id", "season", "week", "team"]].join(rolled)
    out["period"] = out["season"] * 100 + out["week"]
    return out


def _team_baselines(starters: pd.DataFrame) -> pd.DataFrame:
    """Each team's running "who has been playing quarterback here" rating.

    Recorded before the game, so a team that just lost its starter still
    carries the departed starter's level -- which is precisely what its Elo
    rating still reflects.
    """
    out = []
    for team, frame in starters.groupby("team", sort=False):
        frame = frame.sort_values(["season", "week"], kind="mergesort")
        state = np.nan
        baseline = np.full(len(frame), np.nan)
        ratings = frame["qb_rating_pre"].to_numpy(dtype=float)
        for i in range(len(frame)):
            baseline[i] = state
            if not np.isnan(ratings[i]):
                state = (
                    ratings[i] if np.isnan(state)
                    else state + TEAM_QB_ALPHA * (ratings[i] - state)
                )
        result = frame[["game_id", "team"]].copy()
        result["team_qb_baseline"] = baseline
        out.append(result)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["game_id", "team", "team_qb_baseline"]
    )


def build_qb_features(games: pd.DataFrame, player_weeks: pd.DataFrame) -> pd.DataFrame:
    """Return per-game ``qb_delta`` and ``qb_edge``, keyed by ``game_id``.

    ``qb_delta`` is the home side's change of quarterback minus the away
    side's -- the part Elo cannot already know. ``qb_edge`` is the raw
    difference in passer quality, which overlaps with Elo and is included so
    the model can decide for itself how much of it is new information.
    """
    empty = pd.DataFrame({"game_id": games["game_id"], "qb_delta": 0.0, "qb_edge": 0.0})
    if player_weeks is None or player_weeks.empty:
        return empty

    ratings = _passer_ratings(player_weeks)
    if ratings.empty:
        return empty

    starters = _listed_starters(games)
    if starters.empty:
        return empty

    # Look each starter's rating up as of the week he is listed for, taking
    # the state after his most recent *earlier* game. Using the game log's
    # listed quarterback rather than whoever ended up throwing most is what
    # makes this usable before kickoff -- the listed starter is published for
    # upcoming games, the dropback leader is only knowable afterwards.
    starters = pd.merge_asof(
        starters.sort_values("period", kind="mergesort"),
        ratings[["player_id", "period", "qb_rating_post"]]
        .sort_values("period", kind="mergesort"),
        on="period",
        by="player_id",
        direction="backward",
        allow_exact_matches=False,
    ).rename(columns={"qb_rating_post": "qb_rating_pre"})

    starters = starters.sort_values(
        ["season", "week", "kickoff"], kind="mergesort"
    )
    baselines = _team_baselines(starters)
    starters = starters.merge(baselines, on=["game_id", "team"], how="left")

    # A team with no quarterback history yet has nothing to have changed from.
    starters["qb_delta_team"] = (
        starters["qb_rating_pre"] - starters["team_qb_baseline"]
    ).fillna(0.0)
    starters["qb_rating_pre"] = starters["qb_rating_pre"].fillna(0.0)

    keep = ["game_id", "team", "qb_delta_team", "qb_rating_pre"]
    home = games[["game_id", "home_team"]].merge(
        starters[keep].rename(columns={"team": "home_team"}),
        on=["game_id", "home_team"], how="left",
    )
    away = games[["game_id", "away_team"]].merge(
        starters[keep].rename(columns={"team": "away_team"}),
        on=["game_id", "away_team"], how="left",
    )

    out = pd.DataFrame({"game_id": games["game_id"].to_numpy()})
    out["qb_delta"] = (
        home["qb_delta_team"].fillna(0.0).to_numpy()
        - away["qb_delta_team"].fillna(0.0).to_numpy()
    )
    out["qb_edge"] = (
        home["qb_rating_pre"].fillna(0.0).to_numpy()
        - away["qb_rating_pre"].fillna(0.0).to_numpy()
    )
    return out


def _listed_starters(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game carrying the quarterback the game log lists.

    nflverse fills ``home_qb_id`` / ``away_qb_id`` for upcoming games from
    the published depth chart, and for finished games with whoever actually
    started. Over 2020-2026 the listed name matches the eventual dropback
    leader 95.7% of the time; the remaining 4% are late changes that nobody
    -- model or market -- knew about beforehand either.
    """
    columns = {"home": "home_qb_id", "away": "away_qb_id"}
    if not set(columns.values()) <= set(games.columns):
        return pd.DataFrame()

    frames = []
    for side, qb_column in columns.items():
        frame = games[["game_id", "season", "week", "kickoff", f"{side}_team", qb_column]]
        frame = frame.rename(columns={f"{side}_team": "team", qb_column: "player_id"})
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    out = out[out["player_id"].notna()].copy()
    out["period"] = out["season"] * 100 + out["week"]
    return out


def build_availability_features(
    games: pd.DataFrame, player_weeks: pd.DataFrame, injuries: pd.DataFrame
) -> pd.DataFrame:
    """Return per-game ``availability_edge``, keyed by ``game_id``.

    A team's cost is the share of its recent skill-position workload --
    targets and carries, the things that actually generate points -- listed
    as unlikely to play. Weighting by workload rather than counting bodies is
    what separates a starting receiver from a third-string guard.

    Every quantity is looked up as of the week *before* the game, so the
    feature is defined for games that have not been played yet. That is not
    only a leakage guard: a slate being predicted has no box scores at all,
    so a version built from the current week's player rows would silently
    read as "nobody is hurt" exactly when it is being asked.

    Returns zeros for seasons before injury reports exist (pre-2009), so the
    feature degrades to "no information" rather than to a wrong number.
    """
    out = pd.DataFrame({"game_id": games["game_id"].to_numpy(), "availability_edge": 0.0})
    if injuries is None or injuries.empty or player_weeks is None or player_weeks.empty:
        return out

    usage = player_weeks[
        ["player_id", "season", "week", "team", "targets", "carries"]
    ].copy()
    usage["touches"] = usage["targets"].fillna(0.0) + usage["carries"].fillna(0.0)
    usage = usage[usage["touches"] > 0]
    if usage.empty:
        return out

    usage["period"] = usage["season"] * 100 + usage["week"]
    usage = usage.sort_values(["player_id", "period"], kind="mergesort")
    # Workload through and including this game -- what a later week reads.
    usage["usage_post"] = usage.groupby("player_id")["touches"].transform(
        lambda s: s.expanding().mean()
    )

    player_timeline = usage[["player_id", "period", "usage_post"]].sort_values(
        "period", kind="mergesort"
    )
    # A team's total workload, so the cost comes out as a share.
    team_timeline = (
        usage.groupby(["team", "period"], sort=False)["usage_post"].sum().reset_index()
        .rename(columns={"usage_post": "team_usage"})
        .sort_values("period", kind="mergesort")
    )

    report = injuries[["season", "week", "team", "gsis_id", "availability"]].dropna(
        subset=["gsis_id"]
    ).copy()
    report["period"] = report["season"] * 100 + report["week"]
    report = report.rename(columns={"gsis_id": "player_id"}).sort_values(
        "period", kind="mergesort"
    )

    report = pd.merge_asof(
        report, player_timeline, on="period", by="player_id",
        direction="backward", allow_exact_matches=False,
    )
    report["usage_post"] = report["usage_post"].fillna(0.0)
    report["missing"] = (1.0 - report["availability"]) * report["usage_post"]

    cost = report.groupby(["season", "week", "team"], sort=False)["missing"].sum(
    ).reset_index()
    cost["period"] = cost["season"] * 100 + cost["week"]
    cost = cost.sort_values("period", kind="mergesort")
    cost = pd.merge_asof(
        cost, team_timeline, on="period", by="team",
        direction="backward", allow_exact_matches=False,
    )
    cost["cost"] = (
        cost["missing"] / cost["team_usage"].where(cost["team_usage"] > 0)
    ).fillna(0.0).clip(0.0, 1.0)

    covered = set(injuries["season"].dropna().astype(int).unique())
    schedule = games[["game_id", "season", "week", "home_team", "away_team"]].copy()
    for side in ("home", "away"):
        schedule = schedule.merge(
            cost[["season", "week", "team", "cost"]].rename(
                columns={"team": f"{side}_team", "cost": f"{side}_cost"}
            ),
            on=["season", "week", f"{side}_team"],
            how="left",
        )

    # A missing away side helps the home side, hence the sign.
    edge = schedule["away_cost"].fillna(0.0) - schedule["home_cost"].fillna(0.0)
    out["availability_edge"] = np.where(
        schedule["season"].isin(covered), edge.to_numpy(), 0.0
    )
    return out
