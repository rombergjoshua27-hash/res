"""Terminal formatting.

Presentation only -- no module here computes a number that is not already
in the frame it was handed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

__all__ = [
    "format_slate", "format_totals", "format_backtest", "format_ratings",
    "HONESTY_NOTE", "TOTALS_NOTE",
]

HONESTY_NOTE = (
    "These are probabilities, not certainties. Walk-forward backtesting over\n"
    "2008-2026 (4,912 games) puts this model at 66.6% straight-up -- level with\n"
    "the closing line and ~1 game in 3 wrong. Against the spread it wins 49.6%\n"
    "and loses money at standard -110 juice. No model predicts NFL games\n"
    "without mistakes."
)

TOTALS_NOTE = (
    "Totals are harder than sides, not easier. Walk-forward over 2008-2026\n"
    "(4,912 games) the model misses the final total by 10.68 points on average;\n"
    "the closing total misses by 10.47. Blending the two lands at 10.46 -- a\n"
    "tie, not an edge. Over/under picks hit 50.8%, under the 52.38% needed to\n"
    "break even at -110."
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


def format_totals(frame: pd.DataFrame, *, sigma: float = config.TOTAL_SIGMA) -> str:
    """Render the point-total forecast for one slate."""
    if frame.empty or "pred_total" not in frame.columns:
        return "No point-total forecast available for this slate."

    ordered = frame.sort_values("pred_total", ascending=False)
    lines = [_RULE, "POINT TOTALS", _RULE]
    header = (
        f"{'MATCHUP':<20}{'PICK':<7}{'MODEL':>7}{'LINE':>7}{'PROJ':>7}"
        f"{'EDGE':>7}{'P(OVER)':>9}{'CONF':>10}"
    )
    lines += [header, "-" * len(header)]

    any_settled = False
    for row in ordered.itertuples(index=False):
        matchup = f"{row.away_team} @ {row.home_team}"
        line = getattr(row, "market_total", np.nan)
        prob_over = getattr(row, "prob_over", np.nan)
        pick = getattr(row, "ou_pick", "-")

        # Quote confidence from the side the model actually took, so the
        # number reads the same way the pick does.
        confidence = (
            np.nan if pd.isna(prob_over)
            else (prob_over if pick == "OVER" else 1.0 - prob_over)
        )

        suffix = ""
        actual = getattr(row, "actual_total", np.nan)
        if not pd.isna(actual) and not pd.isna(line):
            any_settled = True
            if actual == line:
                suffix = "   PUSH"
            else:
                went_over = actual > line
                hit = (went_over and pick == "OVER") or (not went_over and pick == "UNDER")
                suffix = f"   {'HIT' if hit else 'MISS'} ({actual:.0f})"

        model_total = getattr(row, "model_total", np.nan)
        lines.append(
            f"{matchup:<20}{pick:<7}"
            f"{'    -  ' if pd.isna(model_total) else f'{model_total:7.1f}'}"
            f"{'    -  ' if pd.isna(line) else f'{line:7.1f}'}"
            f"{'    -  ' if pd.isna(row.pred_total) else f'{row.pred_total:7.1f}'}"
            f"{_fmt_signed(getattr(row, 'model_edge', np.nan), 6):>7}"
            f"{_fmt_pct(prob_over, 7):>9}"
            f"{_tier_label(confidence):>10}{suffix}"
        )

    lines.append(_RULE)
    if any_settled:
        lines.append("HIT/MISS marks games already played (still out-of-sample).")
    lines += [
        "MODEL is the model's own total, LINE the posted one, PROJ the blend of",
        f"the two that is actually used (model weight "
        f"{config.DEFAULT_TOTAL_MARKET_BLEND:.0%}). EDGE is MODEL minus LINE --",
        "the size of the disagreement, most of which the blend deliberately",
        f"discards. P(OVER) is taken from PROJ, using sigma={sigma:.1f} pts.",
    ]
    lines += ["", TOTALS_NOTE, _RULE]
    return "\n".join(lines)


def _tier_label(confidence: float) -> str:
    """Confidence tier for an over/under probability quoted from the pick side."""
    if pd.isna(confidence):
        return "-"
    for threshold, label in config.CONFIDENCE_TIERS:
        if confidence >= threshold:
            return label
    return "COINFLIP"


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

    totals = getattr(result, "totals", {}) or {}
    if totals.get("model", {}).get("n"):
        lines += ["", "POINT TOTALS (how many points the forecast missed by)", "-" * 61]
        lines.append(f"{'METHOD':<26}{'GAMES':>7}{'MAE':>9}{'RMSE':>9}{'BIAS':>10}")
        for key, name in [
            ("blended", "Model + market blend"),
            ("model", "Model alone (no market)"),
            ("market", "Market (closing total)"),
        ]:
            entry = totals.get(key, {})
            if not entry.get("n"):
                continue
            lines.append(
                f"{name:<26}{entry['n']:>7,}{entry['mae']:>9.3f}"
                f"{entry['rmse']:>9.3f}{entry['bias']:>+10.3f}"
            )

    playable_ou = [o for o in (getattr(result, "ou", []) or []) if o.get("bets")]
    if playable_ou:
        breakeven = playable_ou[0].get("breakeven", 0.5238)
        lines += [
            "",
            f"OVER/UNDER (-110 juice, breakeven {breakeven * 100:.2f}%)",
            "-" * 61,
        ]
        lines.append(f"{'MIN EDGE':<12}{'BETS':>7}{'W-L-P':>14}{'WIN%':>9}{'ROI':>9}")
        for entry in playable_ou:
            record = f"{entry['wins']}-{entry['losses']}-{entry['pushes']}"
            lines.append(
                f"{entry['threshold']:>5.1f} pts   {entry['bets']:>7,}{record:>14}"
                f"{_fmt_pct(entry['win_rate'], 8)}{entry['roi'] * 100:>+8.2f}%"
            )
        lines.append(
            "Edge is measured against the model's own total, not the blend."
        )

    if not result.by_season.empty:
        lines += ["", "BY SEASON", "-" * 61]
        chunk = []
        for row in result.by_season.itertuples(index=False):
            chunk.append(f"{int(row.season)}: {row.accuracy * 100:.1f}%")
        for i in range(0, len(chunk), 6):
            lines.append("  " + "   ".join(chunk[i : i + 6]))

    lines += ["", HONESTY_NOTE, "", TOTALS_NOTE, _RULE]
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
