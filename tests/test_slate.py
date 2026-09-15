"""Slate selection and the training cutoff.

Two decisions decide what a live run actually predicts, and both are easy to
get subtly wrong: *which* week is the upcoming one, and *how much* of the
season so far the fit is allowed to see.
"""

from __future__ import annotations

from argparse import Namespace

import pandas as pd
import pytest

from nflpredict.cli import _guard_warmup_season, _history_note, _target_slate, _training_history


def _args(**overrides):
    base = dict(season=None, week=None, epa_start=2006, no_epa=False, start=None)
    base.update(overrides)
    return Namespace(**base)


@pytest.fixture
def schedule():
    """Two weeks of a season: week 1 all but finished, week 2 untouched."""
    rows = []
    for week in (1, 2):
        for game in range(4):
            rows.append(
                {
                    "game_id": f"2026_{week:02d}_{game}",
                    "season": 2026,
                    "week": week,
                    "kickoff": pd.Timestamp("2026-09-10") + pd.Timedelta(days=7 * (week - 1) + game),
                    "completed": False,
                }
            )
    frame = pd.DataFrame(rows)
    # Everything in week 1 has been played except its finale.
    frame.loc[(frame["week"] == 1) & (frame["game_id"] != "2026_01_3"), "completed"] = True
    return frame


# --------------------------------------------------------------------------
# Which slate is "upcoming"
# --------------------------------------------------------------------------


def test_a_mostly_played_week_rolls_forward_to_the_next_one(schedule):
    """Three of four played: week 1 is over in every sense that matters."""
    assert _target_slate(schedule, _args()) == (2026, 2)


def test_a_barely_started_week_is_still_the_upcoming_slate(schedule):
    frame = schedule.copy()
    frame.loc[frame["week"] == 1, "completed"] = False
    frame.loc[frame["game_id"] == "2026_01_0", "completed"] = True
    assert _target_slate(frame, _args()) == (2026, 1)


def test_an_untouched_week_is_the_upcoming_slate(schedule):
    frame = schedule.copy()
    frame["completed"] = False
    assert _target_slate(frame, _args()) == (2026, 1)


def test_an_explicit_week_overrides_the_default(schedule):
    """The straggler from a mostly-played week stays reachable."""
    assert _target_slate(schedule, _args(season=2026, week=1)) == (2026, 1)


def test_a_week_on_its_own_is_enough(schedule):
    """Only one season is ever in progress, so --week needs no --season."""
    assert _target_slate(schedule, _args(week=1)) == (2026, 1)


def test_an_unknown_season_is_refused_rather_than_ignored(schedule):
    with pytest.raises(SystemExit, match="no games on record"):
        _target_slate(schedule, _args(season=1998))


def test_a_finished_season_falls_back_to_its_last_week(schedule):
    frame = schedule.copy()
    frame["completed"] = True
    assert _target_slate(frame, _args()) == (2026, 2)


def test_rolling_forward_never_runs_past_the_schedule(schedule):
    """A mostly-played final week has nowhere to roll to, so it stays put."""
    frame = schedule[schedule["week"] == 1].copy()
    assert _target_slate(frame, _args()) == (2026, 1)


# --------------------------------------------------------------------------
# What the fit is allowed to see
# --------------------------------------------------------------------------


def test_training_includes_results_from_earlier_in_the_same_week(schedule):
    """The point of a kickoff cutoff: week 1's finished games train week 2."""
    history = _training_history(schedule, 2026, 2)
    assert len(history) == 3
    assert set(history["week"]) == {1}


def test_training_excludes_the_slate_being_predicted(schedule):
    history = _training_history(schedule, 2026, 2)
    assert history["kickoff"].max() < schedule[schedule["week"] == 2]["kickoff"].min()


def test_every_game_in_a_slate_shares_one_cutoff(schedule):
    """A Sunday game must not be fitted on more than a Thursday game was."""
    history = _training_history(schedule, 2026, 1)
    # Only games kicking off before week 1's *first* game, i.e. none.
    assert len(history) == 0


def test_training_falls_back_to_week_order_without_kickoff_times(schedule):
    frame = schedule.copy()
    frame["kickoff"] = pd.NaT
    history = _training_history(frame, 2026, 2)
    assert len(history) == 3


def test_history_note_reports_current_season_games_in_the_fit(schedule):
    history = _training_history(schedule, 2026, 2)
    assert _history_note(history, 2026) == ", incl. 3 from 2026 wk 1"


def test_history_note_is_silent_when_the_season_has_not_started(schedule):
    assert _history_note(schedule.iloc[:0], 2026) == ""


# --------------------------------------------------------------------------
# Warm-up season guard
# --------------------------------------------------------------------------


def test_scoring_the_first_play_by_play_season_is_refused():
    with pytest.raises(SystemExit, match="centred on themselves"):
        _guard_warmup_season(_args(start=2006, epa_start=2006))


def test_the_default_backtest_range_clears_the_warmup_season():
    from nflpredict import config

    _guard_warmup_season(_args(start=config.DEFAULT_BACKTEST_START, epa_start=2006))


def test_the_guard_does_not_apply_without_play_by_play():
    _guard_warmup_season(_args(start=2006, epa_start=2006, no_epa=True))


# --------------------------------------------------------------------------
# The command line the scheduled job depends on
# --------------------------------------------------------------------------


def test_the_scheduled_refresh_flags_all_parse():
    """The workflow calls these by name; a rename here breaks it silently."""
    from nflpredict.cli import build_parser

    parsed = build_parser().parse_args(
        ["refresh", "--prune-raw", "--quiet", "--season", "2026", "--week", "3"]
    )
    assert parsed.command == "refresh"
    assert parsed.prune_raw is True
    assert parsed.quiet is True
    assert (parsed.season, parsed.week) == (2026, 3)


def test_refresh_accepts_an_output_path():
    from nflpredict.cli import build_parser

    parsed = build_parser().parse_args(["refresh", "-o", "/tmp/book.xlsx"])
    assert parsed.excel == "/tmp/book.xlsx"


def test_every_subcommand_is_still_reachable():
    from nflpredict.cli import build_parser

    parser = build_parser()
    for command in (
        "predict", "refresh", "track", "backtest", "ratings",
        "evaluate", "export", "update",
    ):
        assert parser.parse_args([command]).command == command


def test_pruning_raw_downloads_reports_what_it_removed(tmp_path, monkeypatch):
    """Raw play-by-play is re-downloadable; the summaries built from it are not."""
    from nflpredict import cli, config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    raw = tmp_path / "pbp"
    raw.mkdir()
    for season in (2024, 2025):
        (raw / f"play_by_play_{season}.csv.gz").write_bytes(b"x")
    keep = tmp_path / "epa"
    keep.mkdir()
    (keep / "team_game_epa_v2_2025.csv").write_text("kept")

    assert cli._prune_raw_downloads() == 2
    assert not list(raw.glob("*.csv.gz"))
    assert (keep / "team_game_epa_v2_2025.csv").read_text() == "kept"


def test_pruning_is_safe_when_there_is_nothing_to_prune(tmp_path, monkeypatch):
    from nflpredict import cli, config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert cli._prune_raw_downloads() == 0
