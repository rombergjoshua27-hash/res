"""Command line interface.

    nflpredict predict                 # next unplayed slate
    nflpredict predict --week 5        # a specific week
    nflpredict backtest                # prove the accuracy claim
    nflpredict ratings                 # current Elo power ratings
    nflpredict evaluate --season 2025  # score a finished season
    nflpredict update                  # refresh cached data
"""

from __future__ import annotations

import argparse
import sys
from typing import Tuple

import pandas as pd

from . import config, report
from .backtest import walk_forward
from .data import load_games, load_team_game_epa
from .edge import attach_edges
from .elo import EloEngine
from .features import build_features
from .model import GamePredictor

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
    engine = EloEngine(use_qb=args.qb_adjustment)
    features = build_features(games, epa, elo_engine=engine)
    return games, features


def _target_slate(features: pd.DataFrame, args) -> Tuple[int, int]:
    """Resolve which (season, week) to predict.

    Defaults to the earliest slate that still has an unplayed game -- i.e.
    the games that actually need predicting.
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
    return int(first["season"]), int(first["week"])


def _fit_through(features: pd.DataFrame, season: int, week: int, args) -> GamePredictor:
    """Fit on every completed game strictly before the target slate."""
    period = season * 100 + week
    history = features[
        (features["season"] * 100 + features["week"] < period) & features["completed"]
    ]
    if len(history) < 200:
        raise SystemExit(
            f"only {len(history)} completed games before {season} week {week}; "
            "not enough history to fit"
        )
    return GamePredictor(
        use_market=not args.no_market, market_blend=args.market_blend
    ).fit(history)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_predict(args) -> int:
    _, features = _build(args)
    season, week = _target_slate(features, args)
    predictor = _fit_through(features, season, week, args)

    slate = features[
        (features["season"] == season) & (features["week"] == week)
    ].copy()
    if slate.empty:
        print(f"No games scheduled for {season} week {week}.", file=sys.stderr)
        return 1

    predictions = predictor.predict(slate)
    merged = slate.merge(
        predictions.drop(columns=["game_id"]).assign(
            game_id=predictions["game_id"].values
        ),
        on="game_id",
        how="left",
    )
    enriched = attach_edges(merged).sort_values("pick_prob", ascending=False)

    priced = int(enriched["market_spread"].notna().sum())
    title = (
        f"{season} WEEK {week}  --  {len(enriched)} games  "
        f"({priced} with a posted line, trained on {predictor.report.n_train:,} games)"
    )
    print(report.format_slate(enriched, title=title))

    if args.csv:
        columns = [
            "game_id", "season", "week", "away_team", "home_team", "pick",
            "pick_prob", "confidence", "prob_home", "model_prob_home",
            "market_prob_home", "pred_margin", "market_spread", "spread_edge",
            "ats_pick", "ml_ev", "kelly",
        ]
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = config.OUTPUT_DIR / f"predictions_{season}_wk{week:02d}.csv"
        enriched[[c for c in columns if c in enriched.columns]].to_csv(path, index=False)
        print(f"\nCSV written to {path}")
    return 0


def cmd_backtest(args) -> int:
    _, features = _build(args)
    result = walk_forward(
        features,
        start_season=args.start,
        end_season=args.end,
        use_market=not args.no_market,
        market_blend=args.market_blend,
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
        refit=args.refit,
        quiet=args.quiet,
    )
    print(report.format_backtest(result, label=f"SEASON {season} -- OUT-OF-SAMPLE"))
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
