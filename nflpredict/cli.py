"""Command line interface.

    nflpredict predict                 # next unplayed slate
    nflpredict predict --week 5        # a specific week
    nflpredict backtest                # prove the accuracy claim
    nflpredict ratings                 # current Elo power ratings
    nflpredict evaluate --season 2025  # score a finished season
    nflpredict export                  # everything, as an Excel workbook
    nflpredict update                  # refresh cached data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

from . import config, report
from .backtest import walk_forward
from .data import load_games, load_team_game_epa
from .players import load_injuries, load_player_weeks
from .edge import attach_edges
from .elo import EloEngine
from .features import build_features
from .model import GamePredictor
from .totals import TotalsPredictor

__all__ = ["main"]


# --------------------------------------------------------------------------
# Shared pipeline
# --------------------------------------------------------------------------


def _build(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data and build the leak-free feature matrix once."""
    games = load_games(refresh=getattr(args, "refresh", False), quiet=args.quiet)
    seasons = range(args.epa_start, int(games["season"].max()) + 1)
    epa = (
        pd.DataFrame()
        if args.no_epa
        else load_team_game_epa(seasons, quiet=args.quiet)
    )

    player_weeks = injuries = pd.DataFrame()
    if not args.no_players:
        player_weeks = load_player_weeks(seasons, quiet=args.quiet)
        injuries = load_injuries(seasons, quiet=args.quiet)

    engine = EloEngine(use_qb=args.qb_adjustment)
    features = build_features(
        games, epa, elo_engine=engine,
        player_weeks=player_weeks, injuries=injuries,
    )
    # Fitting on passer terms that are structurally zero would teach the model
    # a coefficient it can never use, so the switch follows the data.
    args._have_players = not player_weeks.empty
    return games, features


def _target_slate(features: pd.DataFrame, args) -> Tuple[int, int]:
    """Resolve which (season, week) to predict.

    Defaults to the earliest slate that still has an unplayed game, with one
    refinement: a week whose games have mostly been played is not the
    upcoming slate. On a Monday, week N has a single night game left and week
    N+1 is what actually needs predicting, so the default rolls forward and
    the finished results feed the fit instead. ``--week`` still reaches the
    straggler.
    """
    if args.season and args.week:
        return int(args.season), int(args.week)

    pending = features[~features["completed"]]
    if args.season:
        pending = pending[pending["season"] == int(args.season)]
    if pending.empty:
        played = features[features["completed"]]
        last = played.iloc[-1]
        return int(last["season"]), int(last["week"])

    first = pending.sort_values(["season", "week"]).iloc[0]
    season, week = int(first["season"]), int(first["week"])

    slate = features[(features["season"] == season) & (features["week"] == week)]
    if slate["completed"].mean() >= 0.5:
        later = pending[
            (pending["season"] > season)
            | ((pending["season"] == season) & (pending["week"] > week))
        ]
        if not later.empty:
            nxt = later.sort_values(["season", "week"]).iloc[0]
            season, week = int(nxt["season"]), int(nxt["week"])
    return season, week


def _qb_features_enabled(args) -> bool:
    """Whether the passer and availability terms should be fitted at all."""
    if args.no_qb_features:
        return False
    return bool(getattr(args, "_have_players", False))


def _training_history(features: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Every game that had finished before the target slate kicks off.

    The cutoff is the slate's first kickoff rather than its week number, so a
    midweek run picks up results already on the board -- predicting week 2 on
    a Monday trains on week 1's Sunday games, which a week-number cutoff
    would throw away. All games in the slate share one cutoff, so no game in
    it is fitted on a different information set than its neighbours.
    """
    slate = features[(features["season"] == season) & (features["week"] == week)]
    completed = features[features["completed"]]

    kickoff = slate["kickoff"].min() if not slate.empty else pd.NaT
    if pd.isna(kickoff):
        period = season * 100 + week
        return completed[completed["season"] * 100 + completed["week"] < period]
    return completed[completed["kickoff"] < kickoff]


def _fit_through(
    features: pd.DataFrame, season: int, week: int, args
) -> Tuple[GamePredictor, TotalsPredictor, pd.DataFrame]:
    """Fit the winner/spread model and the totals model on the same history."""
    history = _training_history(features, season, week)
    if len(history) < 200:
        raise SystemExit(
            f"only {len(history)} completed games before {season} week {week}; "
            "not enough history to fit"
        )
    winner = GamePredictor(
        use_market=not args.no_market,
        market_blend=args.market_blend,
        use_qb_features=_qb_features_enabled(args),
    ).fit(history)
    totals = TotalsPredictor(
        use_market=not args.no_market,
        market_blend=getattr(args, "total_blend", config.DEFAULT_TOTAL_MARKET_BLEND),
    ).fit(history)
    return winner, totals, history


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def _predict_slate(features: pd.DataFrame, season: int, week: int, args):
    """Fit both models through the slate's kickoff and score every game in it.

    Returns the enriched slate plus the two fitted predictors, so callers can
    report what the fit actually saw.
    """
    winner, totals, history = _fit_through(features, season, week, args)

    slate = features[
        (features["season"] == season) & (features["week"] == week)
    ].copy()
    if slate.empty:
        raise SystemExit(f"No games scheduled for {season} week {week}.")

    def _attach(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
        """Merge predictions onto the slate, letting them win any name clash.

        A predictor may re-emit a column the feature frame already carries
        (``market_total`` is both an input and an output). Dropping the
        feature-side copy first keeps one authoritative column instead of a
        silently suffixed ``_x``/``_y`` pair.
        """
        payload = predictions.drop(columns=["game_id"]).assign(
            game_id=predictions["game_id"].values
        )
        clashes = [
            c for c in payload.columns if c != "game_id" and c in frame.columns
        ]
        return frame.drop(columns=clashes).merge(payload, on="game_id", how="left")

    merged = _attach(slate, winner.predict(slate))
    merged = _attach(merged, totals.predict(slate))
    merged["actual_total"] = TotalsPredictor.actual_total(merged)

    enriched = attach_edges(merged).sort_values("pick_prob", ascending=False)
    return enriched, winner, totals, history


def cmd_predict(args) -> int:
    _, features = _build(args)
    season, week = _target_slate(features, args)
    enriched, predictor, totals, history = _predict_slate(features, season, week, args)

    priced = int(enriched["market_spread"].notna().sum())
    title = (
        f"{season} WEEK {week}  --  {len(enriched)} games  "
        f"({priced} with a posted line, trained on {predictor.report.n_train:,} games"
        f"{_history_note(history, season)})"
    )
    print(report.format_slate(enriched, title=title))
    print()
    print(report.format_totals(enriched, sigma=totals.report.sigma))

    if not args.no_excel:
        print("\n" + _write_slate_workbook(enriched, features, season, week, args,
                                            predictor, totals, history))

    if args.csv:
        columns = [
            "game_id", "season", "week", "away_team", "home_team", "pick",
            "pick_prob", "confidence", "prob_home", "model_prob_home",
            "market_prob_home", "pred_margin", "market_spread", "spread_edge",
            "ats_pick", "fair_home_ml", "fair_away_ml", "ml_ev", "kelly",
            "pred_total", "model_total", "market_total", "total_edge",
            "prob_over", "ou_pick",
        ]
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = config.OUTPUT_DIR / f"predictions_{season}_wk{week:02d}.csv"
        enriched[[c for c in columns if c in enriched.columns]].to_csv(path, index=False)
        print(f"\nCSV written to {path}")
    return 0


def _write_slate_workbook(
    enriched, features, season: int, week: int, args, predictor, totals, history
) -> str:
    """Write the slate-only workbook and return a line describing where it went.

    Separate from ``export`` because this one skips the backtest entirely: a
    weekly run should cost a second, not the several minutes it takes to
    replay nineteen seasons.
    """
    from datetime import datetime

    from .excel import export_slate_workbook

    engine = EloEngine(use_qb=args.qb_adjustment)
    engine.run(load_games(quiet=True))

    meta = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "slate_season": season,
        "slate_week": week,
        "slate_train": predictor.report.n_train,
        "market_blend": args.market_blend,
        "total_blend": args.total_blend,
        "total_sigma": totals.report.sigma,
        "slate_history_note": _history_note(history, season).lstrip(", "),
    }

    path = Path(args.excel) if args.excel else (
        config.OUTPUT_DIR / f"slate_{season}_wk{week:02d}.xlsx"
    )
    export_slate_workbook(
        path, slate=enriched, ratings=engine.current_ratings(), meta=meta
    )
    return f"Workbook written to {path}"


def _history_note(history: pd.DataFrame, season: int) -> str:
    """Note how much of the current season the fit already contains."""
    if history.empty:
        return ""
    this_season = history[history["season"] == season]
    if this_season.empty:
        return ""
    weeks = sorted(this_season["week"].unique())
    span = f"wk {weeks[0]}" if len(weeks) == 1 else f"wks {weeks[0]}-{weeks[-1]}"
    return f", incl. {len(this_season)} from {season} {span}"


def _guard_warmup_season(args) -> None:
    """Refuse to score the first season of loaded play-by-play.

    That season's league baselines are centred on themselves, because there
    is no earlier season to centre on (see ``features._league_baselines``).
    It is a warm-up season, not a scoreable one, and the defaults already
    leave a gap. Only a hand-picked ``--epa-start`` can close it.
    """
    if args.no_epa or args.start is None:
        return
    if args.start <= args.epa_start:
        raise SystemExit(
            f"--start {args.start} would score {args.epa_start}, the first season "
            f"of play-by-play, whose league baselines are centred on themselves. "
            f"Use --start {args.epa_start + 1} or later, or move --epa-start back."
        )


def cmd_backtest(args) -> int:
    _guard_warmup_season(args)
    _, features = _build(args)
    result = walk_forward(
        features,
        start_season=args.start,
        end_season=args.end,
        use_market=not args.no_market,
        market_blend=args.market_blend,
        total_blend=args.total_blend,
        use_qb_features=_qb_features_enabled(args),
        refit=args.refit,
        quiet=args.quiet,
    )
    label = (
        f"WALK-FORWARD BACKTEST {args.start}-{args.end or 'latest'}  "
        f"(refit per {args.refit}, market {'off' if args.no_market else 'on'})"
    )
    print(report.format_backtest(result, label=label))

    if args.csv:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = config.OUTPUT_DIR / "backtest_predictions.csv"
        result.predictions.to_csv(path, index=False)
        print(f"\nPer-game predictions written to {path}")
    return 0


def cmd_ratings(args) -> int:
    games, _ = _build(args)
    engine = EloEngine(use_qb=args.qb_adjustment)
    engine.run(games)
    print(report.format_ratings(engine.current_ratings(), top=args.top))
    return 0


def cmd_evaluate(args) -> int:
    _, features = _build(args)
    season = int(args.season or features[features["completed"]]["season"].max())
    result = walk_forward(
        features,
        start_season=season,
        end_season=season,
        use_market=not args.no_market,
        market_blend=args.market_blend,
        total_blend=args.total_blend,
        use_qb_features=_qb_features_enabled(args),
        refit=args.refit,
        quiet=args.quiet,
    )
    print(report.format_backtest(result, label=f"SEASON {season} -- OUT-OF-SAMPLE"))
    return 0


def cmd_export(args) -> int:
    """Write every model output to a multi-sheet Excel workbook."""
    from datetime import datetime

    _guard_warmup_season(args)

    from .excel import export_workbook

    _, features = _build(args)

    season, week = _target_slate(features, args)
    slate_out, predictor, totals, history = _predict_slate(features, season, week, args)

    engine = EloEngine(use_qb=args.qb_adjustment)
    engine.run(load_games(quiet=args.quiet))
    ratings = engine.current_ratings()

    if not args.quiet:
        print(f"Backtesting {args.start}-{args.end or 'latest'} for the workbook ...")
    result = walk_forward(
        features,
        start_season=args.start,
        end_season=args.end,
        use_market=not args.no_market,
        market_blend=args.market_blend,
        total_blend=args.total_blend,
        use_qb_features=_qb_features_enabled(args),
        refit=args.refit,
        quiet=args.quiet,
    )

    meta = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "start": args.start,
        "end": args.end or int(result.predictions["season"].max()),
        "slate_season": season,
        "slate_week": week,
        "slate_train": predictor.report.n_train,
        "market_blend": args.market_blend,
        "total_blend": args.total_blend,
        "total_sigma": totals.report.sigma,
        "slate_history_note": _history_note(history, season).lstrip(", "),
    }

    path = Path(args.output) if args.output else (
        config.OUTPUT_DIR / f"nflpredict_{season}_wk{week:02d}.xlsx"
    )
    export_workbook(path, slate=slate_out, ratings=ratings, result=result, meta=meta)
    print(f"Workbook written to {path}")
    print(
        f"  10 sheets | {len(slate_out)} slate games | {len(ratings)} teams | "
        f"{len(result.predictions):,} backtested games"
    )
    return 0


def cmd_update(args) -> int:
    games = load_games(refresh=True, quiet=False)
    seasons = range(args.epa_start, int(games["season"].max()) + 1)
    if not args.no_epa:
        load_team_game_epa(seasons, refresh=args.refresh, quiet=False)
    completed = int(games["completed"].sum())
    print(f"Cache updated: {len(games):,} games ({completed:,} completed).")
    return 0


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def _common_options() -> argparse.ArgumentParser:
    """Options shared by every subcommand.

    Defined on a parent parser rather than the top-level one so they can be
    written after the subcommand (`nflpredict predict --quiet`), which is
    what people actually type.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--epa-start", type=int, default=2006,
        help="first season of play-by-play to load (default: 2006)",
    )
    common.add_argument("--no-epa", action="store_true", help="skip EPA form features")
    common.add_argument(
        "--no-market", action="store_true",
        help="ignore betting lines entirely (pure model)",
    )
    common.add_argument(
        "--market-blend", type=float, default=config.DEFAULT_MARKET_BLEND,
        help=(
            "model weight when blending with the line "
            f"(default: {config.DEFAULT_MARKET_BLEND})"
        ),
    )
    common.add_argument(
        "--total-blend", type=float, default=config.DEFAULT_TOTAL_MARKET_BLEND,
        help=(
            "model weight when blending with the posted total "
            f"(default: {config.DEFAULT_TOTAL_MARKET_BLEND})"
        ),
    )
    common.add_argument(
        "--no-players", action="store_true",
        help="skip player stats and injury reports entirely",
    )
    common.add_argument(
        "--no-qb-features", action="store_true",
        help="load player data but leave the passer/availability terms out of the fit",
    )
    common.add_argument(
        "--qb-adjustment", action="store_true",
        help="enable the Elo QB term (measurably worse; see config.py)",
    )
    common.add_argument("--refresh", action="store_true", help="force re-download")
    common.add_argument("--quiet", action="store_true", help="suppress progress output")
    common.add_argument("--csv", action="store_true", help="also write results to out/")
    return common


def build_parser() -> argparse.ArgumentParser:
    common = _common_options()
    parser = argparse.ArgumentParser(
        prog="nflpredict",
        description="Walk-forward backtested NFL game outcome model.",
        epilog=report.HONESTY_NOTE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser(
        "predict", parents=[common], help="predict a slate of games"
    )
    predict.add_argument("--season", type=int, help="season (default: current)")
    predict.add_argument("--week", type=int, help="week (default: next unplayed)")
    predict.add_argument(
        "--no-excel", action="store_true",
        help="skip the workbook and print to the terminal only",
    )
    predict.add_argument(
        "--excel", metavar="PATH",
        help="workbook path (default: out/slate_<season>_wk<week>.xlsx)",
    )
    predict.set_defaults(func=cmd_predict)

    backtest = subparsers.add_parser(
        "backtest", parents=[common], help="replay history week by week"
    )
    backtest.add_argument("--start", type=int, default=config.DEFAULT_BACKTEST_START)
    backtest.add_argument("--end", type=int, default=None)
    backtest.add_argument("--refit", choices=("week", "season"), default="season")
    backtest.set_defaults(func=cmd_backtest)

    ratings = subparsers.add_parser(
        "ratings", parents=[common], help="current Elo power ratings"
    )
    ratings.add_argument("--top", type=int, default=None)
    ratings.set_defaults(func=cmd_ratings)

    evaluate = subparsers.add_parser(
        "evaluate", parents=[common], help="score one season out-of-sample"
    )
    evaluate.add_argument("--season", type=int, default=None)
    evaluate.add_argument("--refit", choices=("week", "season"), default="week")
    evaluate.set_defaults(func=cmd_evaluate)

    export = subparsers.add_parser(
        "export", parents=[common], help="write all outputs to an Excel workbook"
    )
    export.add_argument("--season", type=int, help="slate season (default: current)")
    export.add_argument("--week", type=int, help="slate week (default: next unplayed)")
    export.add_argument("--start", type=int, default=config.DEFAULT_BACKTEST_START)
    export.add_argument("--end", type=int, default=None)
    export.add_argument("--refit", choices=("week", "season"), default="season")
    export.add_argument("-o", "--output", help="output path (default: out/*.xlsx)")
    export.set_defaults(func=cmd_export)

    update = subparsers.add_parser(
        "update", parents=[common], help="refresh cached data"
    )
    update.set_defaults(func=cmd_update)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
