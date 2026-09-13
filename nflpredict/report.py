"""Terminal formatting.

Presentation only -- no module here computes a number that is not already
in the frame it was handed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

__all__ = ["format_slate", "format_backtest", "format_ratings", "HONESTY_NOTE"]

HONESTY_NOTE = (
    "These are probabilities, not certainties. Walk-forward backtesting over\n"
    "2008-2025 (4,897 games) puts this model at ~66.6% straight-up -- roughly\n"
    "level with the closing line and ~1 game in 3 wrong. Against the spread it\n"
    "wins 49.6% and loses money at standard -110 juice. No model predicts NFL\n"
    "games without mistakes."
)

_RULE = "-" * 78


def _fmt_pct(value: float, width: int = 6) -> str:
    return "  n/a " if pd.isna(value) else f"{value * 100:{width}.1f}%"


def _fmt_signed(value: float, width: int = 5) -> str:
    return "  -  " if pd.isna(value) else f"{value:+{width}.1f}"


def format_slate(frame: pd.DataFrame, *, title: str = "PREDICTIONS") -> str:
    """Render one slate of games with picks, probabilities and market context."""
    if frame.empty:
        return "No games found for that slate."

    lines = [_RULE, title, _RULE]
    header = (
        f"{'MATCHUP':<20}{'PICK':<6}{'WIN%':>7}{'CONF':>10}"
        f"{'MODEL':>8}{'MKT':>7}{'LINE':>7}{'EDGE':>7}"
    )
    lines += [header, "-" * len(header)]

    any_settled = False
    for row in frame.itertuples(index=False):
        matchup = f"{row.away_team} @ {row.home_team}"
        model_spread = getattr(row, "pred_margin", np.nan)
        market_spread = getattr(row, "market_spread", np.nan)
        picked_home = row.pick == row.home_team

        # Quote the market from the same side the model picked, so WIN% and
        # MKT are directly comparable rather than silently opposed.
        market_prob = getattr(row, "market_prob_home", np.nan)
        if not pd.isna(market_prob) and not picked_home:
            market_prob = 1.0 - market_prob

        suffix = " *" if getattr(row, "disagrees_with_market", False) else ""
        margin = getattr(row, "margin", np.nan)
        if not pd.isna(margin):
            any_settled = True
            home_won = margin > 0
            hit = "PUSH" if margin == 0 else ("HIT" if home_won == picked_home else "MISS")
            suffix = f"{suffix:<2} {hit}"

        lines.append(
            f"{matchup:<20}{row.pick:<6}{_fmt_pct(row.pick_prob)}"
            f"{row.confidence:>10}"
            f"{_fmt_signed(model_spread):>8}"
            f"{_fmt_pct(market_prob, 5):>7}"
            f"{_fmt_signed(market_spread):>7}"
            f"{_fmt_signed(getattr(row, 'spread_edge', np.nan)):>7}{suffix}"
        )

    lines.append(_RULE)
    if any_settled:
        lines.append("HIT/MISS marks games already played (still out-of-sample).")
    lines.append(
        "MODEL/LINE are home-team point spreads (positive = home favoured)."
    )
    if bool(frame.get("disagrees_with_market", pd.Series(dtype=bool)).any()):
        lines.append("* model disagrees with the market on the outright winner.")
    lines += ["", HONESTY_NOTE, _RULE]
    return "\n".join(lines)


def format_backtest(result, *, label: str = "WALK-FORWARD BACKTEST") -> str:
    """Render the full honesty report for a backtest run."""
    lines = [_RULE, label, _RULE]

    order = [
        ("blended", "Model + market blend"),
        ("model", "Model alone (no market)"),
        ("market", "Market (closing line)"),
        ("elo_only", "Elo only"),
        ("home_team_always", "Always pick home"),
    ]
    lines.append(f"{'METHOD':<26}{'GAMES':>7}{'ACC':>9}{'BRIER':>9}{'LOGLOSS':>10}")
    lines.append("-" * 61)
    for key, name in order:
        summary = result.summary.get(key, {})
        if not summary.get("n"):
            continue
        brier = summary.get("brier", np.nan)
        logloss = summary.get("log_loss", np.nan)
        brier_cell = f"{'n/a':>9}" if pd.isna(brier) else f"{brier:9.4f}"
        logloss_cell = f"{'n/a':>10}" if pd.isna(logloss) else f"{logloss:10.4f}"
        lines.append(
            f"{name:<26}{summary['n']:>7,}"
            f"{_fmt_pct(summary['accuracy'], 8)}{brier_cell}{logloss_cell}"
        )

    if not result.calibration.empty:
        lines += ["", "CALIBRATION (is a stated 70% actually 70%?)", "-" * 61]
        lines.append(f"{'CONFIDENCE':<16}{'GAMES':>7}{'SAID':>9}{'ACTUAL':>9}{'GAP':>9}")
        for row in result.calibration.itertuples(index=False):
            lines.append(
                f"{str(row.bucket):<16}{row.games:>7,}"
                f"{_fmt_pct(row.predicted, 8)}{_fmt_pct(row.actual, 8)}"
                f"{row.gap * 100:>+8.1f}%"
            )

    playable = [a for a in result.ats if a.get("bets")]
    if playable:
        breakeven = playable[0].get("breakeven", 0.5238)
        lines += [
            "",
            f"AGAINST THE SPREAD (-110 juice, breakeven {breakeven * 100:.2f}%)",
            "-" * 61,
        ]
        lines.append(f"{'MIN EDGE':<12}{'BETS':>7}{'W-L-P':>14}{'WIN%':>9}{'ROI':>9}")
        for entry in playable:
            record = f"{entry['wins']}-{entry['losses']}-{entry['pushes']}"
            lines.append(
                f"{entry['threshold']:>5.1f} pts   {entry['bets']:>7,}{record:>14}"
                f"{_fmt_pct(entry['win_rate'], 8)}{entry['roi'] * 100:>+8.2f}%"
            )
        lines.append(
            "Small-sample buckets (a few dozen bets) are noise, not an edge."
        )

    if not result.by_season.empty:
        lines += ["", "BY SEASON", "-" * 61]
        chunk = []
        for row in result.by_season.itertuples(index=False):
            chunk.append(f"{int(row.season)}: {row.accuracy * 100:.1f}%")
        for i in range(0, len(chunk), 6):
            lines.append("  " + "   ".join(chunk[i : i + 6]))

    lines += ["", HONESTY_NOTE, _RULE]
    return "\n".join(lines)


def format_ratings(ratings: pd.DataFrame, *, top: int | None = None) -> str:
    """Render the current Elo power ratings."""
    if ratings.empty:
        return "No ratings available."
    table = ratings.head(top) if top else ratings
    lines = [_RULE, "POWER RATINGS (Elo)", _RULE]
    lines.append(f"{'#':>3}  {'TEAM':<6}{'ELO':>9}{'VS AVERAGE':>13}")
    lines.append("-" * 34)
    for i, row in enumerate(table.itertuples(index=False), start=1):
        lines.append(
            f"{i:>3}  {row.team:<6}{row.elo:>9.0f}"
            f"{row.spread_vs_average:>+12.1f} pts"
        )
    lines += [_RULE, f"{config.ELO_PER_POINT:.0f} Elo points = 1 point of spread.", _RULE]
    return "\n".join(lines)
