"""Data acquisition and caching.

Two public entry points:

``load_games``
    The nflverse/nfldata game log: every NFL game since 1999 with final
    scores, closing betting lines, rest days, venue, and starting QBs.

``load_team_game_epa``
    Per-team, per-game EPA aggregates distilled from nflverse play-by-play.
    The raw play-by-play is ~19 MB per season; it is aggregated once and
    cached as a ~60 KB CSV per season, so the heavy download happens only
    when a season is first used (or explicitly refreshed).

Nothing here knows anything about modelling. In particular, no function in
this module looks at a game's outcome when building features for that game
-- that separation is what keeps the backtest honest.
"""

from __future__ import annotations

import concurrent.futures
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import requests

from . import config

__all__ = [
    "load_games", "load_team_game_epa", "load_half_scores",
    "normalize_team", "refresh_all",
]


# --------------------------------------------------------------------------
# HTTP with retry
# --------------------------------------------------------------------------


def _download(url: str, dest: Path, *, label: str | None = None) -> Path:
    """Download ``url`` to ``dest`` atomically, retrying with backoff."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(config.DOWNLOAD_RETRIES):
        try:
            with requests.get(
                url, stream=True, timeout=config.DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                response.raise_for_status()
                with open(tmp, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 16):
                        if chunk:
                            handle.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - retry on any transport error
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < config.DOWNLOAD_RETRIES - 1:
                delay = config.DOWNLOAD_BACKOFF_SECONDS * (2**attempt)
                print(
                    f"  ! {label or url} failed ({exc.__class__.__name__}); "
                    f"retrying in {delay:.0f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)

    raise RuntimeError(f"could not download {url}: {last_error}") from last_error


def _is_fresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    return age_hours < max_age_hours


# --------------------------------------------------------------------------
# Team normalization
# --------------------------------------------------------------------------


def normalize_team(team: object) -> object:
    """Map a historical team code onto its current franchise code."""
    if not isinstance(team, str):
        return team
    return config.FRANCHISE_MAP.get(team, team)


def _normalize_team_column(series: pd.Series) -> pd.Series:
    return series.replace(config.FRANCHISE_MAP)


# --------------------------------------------------------------------------
# Game log
# --------------------------------------------------------------------------

_GAMES_CACHE: pd.DataFrame | None = None


def load_games(*, refresh: bool = False, quiet: bool = False) -> pd.DataFrame:
    """Return the full nflverse game log, normalized and chronologically sorted.

    Adds a handful of derived columns used everywhere downstream:

    ``kickoff``    timezone-naive datetime used as the strict ordering key
    ``completed``  whether a final score is on record
    ``margin``     home_score - away_score (NaN until the game is played)
    ``home_win``   1.0 / 0.0 / 0.5 for a tie (NaN until played)
    ``neutral``    True for neutral-site games (no home field advantage)
    """
    global _GAMES_CACHE
    if _GAMES_CACHE is not None and not refresh:
        return _GAMES_CACHE.copy()

    path = config.DATA_DIR / "games.csv"
    if refresh or not _is_fresh(path, config.GAMES_CACHE_HOURS):
        if not quiet:
            print("Downloading nflverse game log ...")
        _download(config.GAMES_URL, path, label="games.csv")

    games = pd.read_csv(path, low_memory=False)

    games["home_team"] = _normalize_team_column(games["home_team"])
    games["away_team"] = _normalize_team_column(games["away_team"])

    numeric = [
        "season", "week", "home_score", "away_score", "result", "total",
        "away_rest", "home_rest", "away_moneyline", "home_moneyline",
        "spread_line", "total_line", "div_game", "temp", "wind",
    ]
    for column in numeric:
        if column in games.columns:
            games[column] = pd.to_numeric(games[column], errors="coerce")

    games["kickoff"] = _build_kickoff(games)
    games["completed"] = games["result"].notna()
    games["margin"] = games["result"]  # nflverse: home_score - away_score
    games["home_win"] = games["margin"].apply(_win_value)
    games["neutral"] = games.get(
        "location", pd.Series("Home", index=games.index)
    ).eq("Neutral")
    games["is_playoff"] = games["game_type"].ne("REG")

    games = games.sort_values(
        ["season", "week", "kickoff", "game_id"], kind="mergesort"
    ).reset_index(drop=True)

    _GAMES_CACHE = games
    return games.copy()


def _build_kickoff(games: pd.DataFrame) -> pd.Series:
    """Best-effort kickoff timestamp; falls back to midnight on the game date."""
    date = pd.to_datetime(games["gameday"], errors="coerce")
    time_str = games.get("gametime")
    if time_str is None:
        return date
    offset = pd.to_timedelta(time_str.astype(str) + ":00", errors="coerce")
    return date + offset.fillna(pd.Timedelta(0))


def _win_value(margin: float) -> float:
    if pd.isna(margin):
        return float("nan")
    if margin > 0:
        return 1.0
    if margin < 0:
        return 0.0
    return config.TIE_CREDIT


# --------------------------------------------------------------------------
# Play-by-play -> team-game EPA
# --------------------------------------------------------------------------

_PBP_COLUMNS = [
    "game_id", "season", "week", "posteam", "defteam", "epa", "success",
    "pass", "rush", "play_type", "qb_epa", "yards_gained", "interception",
    "fumble_lost", "sack", "drive",
]


def _season_epa_path(season: int) -> Path:
    return config.EPA_CACHE_DIR / f"team_game_epa_v{config.EPA_CACHE_VERSION}_{season}.csv"


def _aggregate_season_pbp(season: int, *, refresh: bool = False) -> pd.DataFrame:
    """Download one season of play-by-play and reduce it to team-game rows."""
    cache = _season_epa_path(season)
    if cache.exists() and not refresh:
        return pd.read_csv(cache)

    raw_path = config.DATA_DIR / "pbp" / f"play_by_play_{season}.csv.gz"
    if refresh or not raw_path.exists():
        _download(
            config.PBP_URL_TEMPLATE.format(season=season),
            raw_path,
            label=f"pbp {season}",
        )

    plays = pd.read_csv(
        raw_path, compression="gzip", low_memory=False, usecols=lambda c: c in _PBP_COLUMNS
    )

    # Scrimmage plays only: kneels, spikes, kicks and special teams carry no
    # information about offensive quality.
    plays = plays[
        plays["play_type"].isin(["pass", "run"])
        & plays["epa"].notna()
        & plays["posteam"].notna()
        & plays["defteam"].notna()
    ].copy()

    for column in ("pass", "rush", "success", "interception", "fumble_lost", "sack"):
        if column in plays.columns:
            plays[column] = pd.to_numeric(plays[column], errors="coerce").fillna(0.0)
        else:
            plays[column] = 0.0

    plays["posteam"] = _normalize_team_column(plays["posteam"])
    plays["defteam"] = _normalize_team_column(plays["defteam"])
    plays["explosive"] = (plays["yards_gained"] >= 20).astype(float)
    plays["turnover"] = (
        plays["interception"].clip(0, 1) + plays["fumble_lost"].clip(0, 1)
    ).clip(0, 1)

    grouped = plays.groupby(["game_id", "season", "week", "posteam"], sort=False)
    offense = grouped.apply(_summarise_offense, include_groups=False).reset_index()
    offense = offense.rename(columns={"posteam": "team"})

    # A team's defensive line is simply its opponent's offensive line in the
    # same game, so derive it by pairing the two rows of each game.
    opponents = (
        plays.groupby(["game_id", "posteam"], sort=False)["defteam"].first().reset_index()
    )
    offense = offense.merge(
        opponents.rename(columns={"posteam": "team", "defteam": "opponent"}),
        on=["game_id", "team"],
        how="left",
    )

    offense = offense[offense["off_plays"] >= config.EPA_MIN_PLAYS].copy()

    defense = offense[["game_id", "team"] + _OFFENSE_STATS].copy()
    defense = defense.rename(
        columns={"team": "opponent", **{c: c.replace("off_", "def_") for c in _OFFENSE_STATS}}
    )
    merged = offense.merge(defense, on=["game_id", "opponent"], how="left")

    cache.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache, index=False)
    return merged


_OFFENSE_STATS = [
    "off_epa_play", "off_pass_epa", "off_rush_epa", "off_success",
    "off_explosive", "off_turnover_rate", "off_sack_rate", "off_pass_rate",
    "off_plays", "off_drives", "off_plays_per_drive",
]


def _summarise_offense(frame: pd.DataFrame) -> pd.Series:
    is_pass = frame["pass"] > 0
    is_rush = frame["rush"] > 0
    plays = float(len(frame))
    # Drive count drives the totals model: points scored is roughly
    # (drives) x (points per drive), so possessions are half the equation.
    drives = float(frame["drive"].nunique()) if "drive" in frame.columns else float("nan")
    return pd.Series(
        {
            "off_plays": plays,
            "off_drives": drives,
            "off_plays_per_drive": plays / drives if drives and drives > 0 else float("nan"),
            "off_epa_play": frame["epa"].mean(),
            "off_pass_epa": frame.loc[is_pass, "epa"].mean(),
            "off_rush_epa": frame.loc[is_rush, "epa"].mean(),
            "off_success": frame["success"].mean(),
            "off_explosive": frame["explosive"].mean(),
            "off_turnover_rate": frame["turnover"].mean(),
            "off_sack_rate": frame.loc[is_pass, "sack"].mean() if is_pass.any() else 0.0,
            "off_pass_rate": is_pass.mean(),
        }
    )


def load_team_game_epa(
    seasons: Iterable[int],
    *,
    refresh: bool = False,
    quiet: bool = False,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Return team-game EPA rows for ``seasons``, downloading only what is missing."""
    seasons = sorted({int(s) for s in seasons if int(s) >= config.FIRST_PBP_SEASON})
    if not seasons:
        return pd.DataFrame()

    missing = [s for s in seasons if refresh or not _season_epa_path(s).exists()]
    if missing and not quiet:
        print(
            f"Building EPA cache for {len(missing)} season(s): "
            f"{missing[0]}-{missing[-1]} (one-time download) ..."
        )

    if missing:
        # Downloads dominate the runtime, so overlap them.
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_aggregate_season_pbp, season, refresh=refresh): season
                for season in missing
            }
            for future in concurrent.futures.as_completed(futures):
                season = futures[future]
                future.result()  # surface any download/parse error
                if not quiet:
                    print(f"  . {season} cached")

    frames = [pd.read_csv(_season_epa_path(s)) for s in seasons]
    epa = pd.concat(frames, ignore_index=True)
    epa["team"] = _normalize_team_column(epa["team"])
    epa["opponent"] = _normalize_team_column(epa["opponent"])
    return epa


def refresh_all(seasons: Sequence[int] | None = None) -> None:
    """Force a re-download of the game log and (optionally) EPA seasons."""
    load_games(refresh=True)
    if seasons:
        load_team_game_epa(seasons, refresh=True)


# --------------------------------------------------------------------------
# Halftime scores
# --------------------------------------------------------------------------

_HALF_COLUMNS = ["game_id", "qtr", "total_home_score", "total_away_score"]


def _half_score_path(season: int) -> Path:
    return config.EPA_CACHE_DIR / f"half_scores_v{config.EPA_CACHE_VERSION}_{season}.csv"


def _aggregate_season_halves(season: int, *, refresh: bool = False) -> pd.DataFrame:
    """Reduce one season of play-by-play to the score at halftime."""
    cache = _half_score_path(season)
    if cache.exists() and not refresh:
        return pd.read_csv(cache)

    raw_path = config.DATA_DIR / "pbp" / f"play_by_play_{season}.csv.gz"
    if refresh or not raw_path.exists():
        _download(
            config.PBP_URL_TEMPLATE.format(season=season),
            raw_path,
            label=f"pbp {season}",
        )

    plays = pd.read_csv(
        raw_path, compression="gzip", low_memory=False,
        usecols=lambda c: c in _HALF_COLUMNS,
    )
    # The running score columns are cumulative, so the largest value seen
    # during the first two quarters is the score the teams took to the break.
    first_half = plays[pd.to_numeric(plays["qtr"], errors="coerce") <= 2]
    grouped = first_half.groupby("game_id").agg(
        home_first_half=("total_home_score", "max"),
        away_first_half=("total_away_score", "max"),
    ).reset_index()
    grouped["season"] = season

    cache.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(cache, index=False)
    return grouped


def load_half_scores(
    seasons: Iterable[int],
    *,
    refresh: bool = False,
    quiet: bool = False,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Return the halftime score of every game in ``seasons``.

    Columns: ``game_id``, ``season``, ``home_first_half``, ``away_first_half``.
    Distilled from play-by-play and cached per season, like the EPA summaries.
    """
    seasons = sorted({int(s) for s in seasons if int(s) >= config.FIRST_PBP_SEASON})
    if not seasons:
        return pd.DataFrame(
            columns=["game_id", "season", "home_first_half", "away_first_half"]
        )

    missing = [s for s in seasons if refresh or not _half_score_path(s).exists()]
    if missing and not quiet:
        print(
            f"Building halftime-score cache for {len(missing)} season(s): "
            f"{missing[0]}-{missing[-1]} ..."
        )

    if missing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_aggregate_season_halves, season, refresh=refresh)
                for season in missing
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # surface any download/parse error

    frames = [
        pd.read_csv(_half_score_path(s)) for s in seasons if _half_score_path(s).exists()
    ]
    if not frames:
        return pd.DataFrame(
            columns=["game_id", "season", "home_first_half", "away_first_half"]
        )
    return pd.concat(frames, ignore_index=True)
