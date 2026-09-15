"""Player-level data: weekly box scores and injury reports.

Two public entry points, both cached per season the same way ``data.py``
caches play-by-play:

``load_player_weeks``
    One row per player per game since 1999: passing, rushing and receiving
    production, with the efficiency measures that matter downstream --
    ``passing_epa``, ``passing_cpoe``, ``target_share``, ``air_yards_share``.

``load_injuries``
    The weekly injury report since 2009: who was listed, with what, and
    whether they were Out, Doubtful or Questionable.

**Why CPOE matters here.** The previous attempt at a quarterback adjustment
failed because it rated a passer by his team's offensive EPA, which is
mostly a property of the team -- and the team's strength was already in its
Elo rating, so the adjustment double-counted it. Completion percentage over
expected is computed per throw against the difficulty of that throw (depth,
pressure, receiver separation), so it isolates the passer far better. That
is the ingredient the earlier version did not have.

Snap counts are deliberately not loaded. nflverse keys them by Pro Football
Reference player id rather than the gsis id everything else here uses, so
joining them means matching on names -- fragile, and unnecessary when
``target_share`` and ``wopr`` already measure usage on a clean key.

Nothing in this module looks at a game's outcome when building features for
that game; that separation is what keeps the backtest honest.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from . import config
from .data import _download, _normalize_team_column

__all__ = ["load_player_weeks", "load_injuries", "injury_availability"]


# Columns kept from the 150 nflverse publishes. The rest are kicking, punting,
# defensive and fantasy columns that nothing here consumes.
_PLAYER_COLUMNS: List[str] = [
    "player_id", "player_display_name", "position", "position_group",
    "season", "week", "season_type", "game_id", "team", "opponent_team",
    # passing
    "completions", "attempts", "passing_yards", "passing_tds",
    "passing_interceptions", "sacks_suffered", "passing_air_yards",
    "passing_first_downs", "passing_epa", "passing_cpoe",
    # rushing
    "carries", "rushing_yards", "rushing_tds", "rushing_epa",
    # receiving
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "receiving_air_yards", "receiving_epa",
    "target_share", "air_yards_share", "wopr",
]

_INJURY_COLUMNS: List[str] = [
    "season", "week", "season_type", "team", "gsis_id", "position",
    "full_name", "report_primary_injury", "report_status", "practice_status",
]


def _cache_path(kind: str, season: int) -> Path:
    return (
        config.PLAYER_CACHE_DIR
        / f"{kind}_v{config.PLAYER_CACHE_VERSION}_{season}.csv"
    )


def _load_seasons(
    kind: str,
    url_template: str,
    columns: List[str],
    seasons: Iterable[int],
    first_season: int,
    *,
    refresh: bool = False,
    quiet: bool = False,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Download, trim and cache one file per season, then concatenate."""
    seasons = sorted({int(s) for s in seasons if int(s) >= first_season})
    if not seasons:
        return pd.DataFrame(columns=columns)

    missing = [s for s in seasons if refresh or not _cache_path(kind, s).exists()]
    if missing and not quiet:
        print(
            f"Building {kind} cache for {len(missing)} season(s): "
            f"{missing[0]}-{missing[-1]} (one-time download) ..."
        )

    if missing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _fetch_season, kind, url_template, columns, season, refresh
                ): season
                for season in missing
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()  # surface any download/parse error

    frames = []
    for season in seasons:
        path = _cache_path(kind, season)
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def _fetch_season(
    kind: str, url_template: str, columns: List[str], season: int, refresh: bool
) -> Path:
    """Download one season and cache only the columns actually consumed."""
    cache = _cache_path(kind, season)
    if cache.exists() and not refresh:
        return cache

    raw = config.DATA_DIR / kind / f"{kind}_{season}.csv"
    if refresh or not raw.exists():
        _download(url_template.format(season=season), raw, label=f"{kind} {season}")

    frame = pd.read_csv(raw, low_memory=False)
    keep = [c for c in columns if c in frame.columns]
    frame = frame[keep].copy()
    for column in set(columns) - set(keep):
        frame[column] = pd.NA

    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    raw.unlink(missing_ok=True)  # the trimmed cache is 20x smaller
    return cache


def load_player_weeks(
    seasons: Iterable[int], *, refresh: bool = False, quiet: bool = False
) -> pd.DataFrame:
    """Return one row per player per game for ``seasons``."""
    frame = _load_seasons(
        "stats_player", config.PLAYER_STATS_URL_TEMPLATE, _PLAYER_COLUMNS,
        seasons, config.FIRST_PLAYER_STATS_SEASON, refresh=refresh, quiet=quiet,
    )
    if frame.empty:
        return frame

    for column in ("team", "opponent_team"):
        if column in frame.columns:
            frame[column] = _normalize_team_column(frame[column].astype("string"))

    numeric = [c for c in _PLAYER_COLUMNS if c not in {
        "player_id", "player_display_name", "position", "position_group",
        "season_type", "game_id", "team", "opponent_team",
    }]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Dropbacks, not attempts, is the right denominator for a passer: a sack
    # is a failed dropback, and ignoring it flatters quarterbacks who take them.
    frame["dropbacks"] = frame["attempts"].fillna(0.0) + frame["sacks_suffered"].fillna(0.0)
    frame["passing_epa_per_dropback"] = frame["passing_epa"] / frame["dropbacks"].where(
        frame["dropbacks"] > 0
    )
    return frame.sort_values(["season", "week", "team", "player_id"], kind="mergesort")


def load_injuries(
    seasons: Iterable[int], *, refresh: bool = False, quiet: bool = False
) -> pd.DataFrame:
    """Return the weekly injury report for ``seasons`` (2009 onward)."""
    frame = _load_seasons(
        "injuries", config.INJURIES_URL_TEMPLATE, _INJURY_COLUMNS,
        seasons, config.FIRST_INJURY_SEASON, refresh=refresh, quiet=quiet,
    )
    if frame.empty:
        return frame

    if "team" in frame.columns:
        frame["team"] = _normalize_team_column(frame["team"].astype("string"))
    for column in ("season", "week"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["report_status"] = frame["report_status"].astype("string").str.strip()
    frame["availability"] = frame["report_status"].map(injury_availability).fillna(1.0)
    return frame


def injury_availability(status: object) -> float:
    """Probability a player listed with ``status`` actually plays.

    EMPIRICAL, measured over 2009-2026 (73,243 report rows) by joining each
    week's report to the box score that followed:

        status          rows    played   raw    normalised
        Out           16,098         8  0.000        0.001
        Doubtful       3,385        27  0.008        0.011
        Questionable  23,574    11,673  0.495        0.687
        (no status)   30,186    21,754  0.721        1.000

    The raw rates need the normalisation because appearing in a box score is
    an imperfect proxy for playing: a player can dress, take snaps and record
    no countable stat, which is why even players carrying no game status only
    "play" 72% of the time by this measure. Each status is therefore divided
    by that baseline rather than read off directly.

    "Doubtful" really does mean it -- barely one in a hundred plays, close
    enough to "Out" that treating them alike would cost little. "Questionable"
    is the genuinely uncertain one, and the only status where this number
    does real work.

    Anything else -- not on the report, or listed only for a practice
    limitation -- is treated as available.
    """
    if not isinstance(status, str):
        return 1.0
    return {
        "Out": 0.0,
        "Doubtful": 0.01,
        "Questionable": 0.69,
    }.get(status.strip(), 1.0)
