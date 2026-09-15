"""Market comparison: edges, expected value and stake sizing.

Everything here answers one question -- given the model's probability and
the price on offer, is there a disagreement worth acting on? The backtest
says the answer is almost always *no* against a closing NFL line, so these
functions exist to quantify how small an edge is, not to manufacture one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

__all__ = [
    "american_to_decimal",
    "prob_to_american",
    "american_to_prob",
    "expected_value",
    "kelly_fraction",
    "attach_edges",
    "confidence_tier",
]


def american_to_decimal(odds: float) -> float:
    """Net payout per unit staked (``b`` in the Kelly formula)."""
    odds = float(odds)
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def prob_to_american(prob: float) -> float:
    """The moneyline that a win probability implies, with no vig attached.

    This is the price at which a bet on that side would break even -- the
    model's own moneyline, to be compared against the book's. A favourite
    (prob > 0.5) returns a negative number, an underdog a positive one.
    """
    if pd.isna(prob):
        return float("nan")
    prob = float(np.clip(prob, 1e-6, 1 - 1e-6))
    if prob >= 0.5:
        return -100.0 * prob / (1.0 - prob)
    return 100.0 * (1.0 - prob) / prob


def american_to_prob(odds: float) -> float:
    """Win probability implied by American odds, vig included."""
    if pd.isna(odds):
        return float("nan")
    odds = float(odds)
    return 100.0 / (odds + 100.0) if odds > 0 else (-odds) / (-odds + 100.0)


def expected_value(prob: float, odds: float = config.STANDARD_VIG_ODDS) -> float:
    """Expected profit per unit staked at ``odds`` given true probability ``prob``."""
    if pd.isna(prob) or pd.isna(odds):
        return float("nan")
    b = american_to_decimal(odds)
    return prob * b - (1.0 - prob)


def kelly_fraction(
    prob: float, odds: float = config.STANDARD_VIG_ODDS, *, fraction: float = 0.25
) -> float:
    """Fractional-Kelly stake as a share of bankroll, floored at zero.

    Quarter Kelly by default: full Kelly assumes the probability estimate is
    exactly right, which -- given a Brier score of 0.21 -- it is not.
    """
    if pd.isna(prob) or pd.isna(odds):
        return 0.0
    b = american_to_decimal(odds)
    if b <= 0:
        return 0.0
    edge = prob * b - (1.0 - prob)
    return float(max(0.0, (edge / b) * fraction))


def confidence_tier(prob: float) -> str:
    """Human-readable confidence label for a win probability."""
    if pd.isna(prob):
        return "UNKNOWN"
    confidence = max(prob, 1.0 - prob)
    for threshold, label in config.CONFIDENCE_TIERS:
        if confidence >= threshold:
            return label
    return "COINFLIP"


def attach_edges(frame: pd.DataFrame) -> pd.DataFrame:
    """Add pick, edge, EV and stake columns to a frame of predictions."""
    out = frame.copy()

    prob_home = out["prob_home"].astype(float)
    out["pick"] = np.where(prob_home >= 0.5, out["home_team"], out["away_team"])
    out["pick_prob"] = np.maximum(prob_home, 1.0 - prob_home)
    out["confidence"] = [confidence_tier(p) for p in prob_home]

    # Spread edge: how far the model's margin sits from the posted number.
    if "market_spread" in out.columns:
        out["spread_edge"] = out["pred_margin"] - out["market_spread"]
        out["ats_pick"] = np.where(
            out["spread_edge"].isna(),
            "-",
            np.where(out["spread_edge"] > 0, out["home_team"], out["away_team"]),
        )
    else:
        out["spread_edge"] = np.nan
        out["ats_pick"] = "-"

    # The model's own moneyline for each side: what the price *should* be if
    # the model is right, quoted with no vig so it can be read against a book.
    out["fair_home_ml"] = [prob_to_american(p) for p in prob_home]
    out["fair_away_ml"] = [prob_to_american(1.0 - p) for p in prob_home]

    # Moneyline EV, using the price on the side the model actually likes.
    if {"home_moneyline", "away_moneyline"} <= set(out.columns):
        picked_odds = np.where(
            prob_home >= 0.5, out["home_moneyline"], out["away_moneyline"]
        )
        out["pick_odds"] = picked_odds
        out["ml_ev"] = [
            expected_value(p, o) for p, o in zip(out["pick_prob"], picked_odds)
        ]
        out["kelly"] = [
            kelly_fraction(p, o) for p, o in zip(out["pick_prob"], picked_odds)
        ]
    else:
        out["pick_odds"] = np.nan
        out["ml_ev"] = np.nan
        out["kelly"] = 0.0

    # Where the model and the market disagree on the outright winner at all.
    if "market_prob_home" in out.columns:
        out["disagrees_with_market"] = (
            (prob_home >= 0.5) != (out["market_prob_home"].astype(float) >= 0.5)
        ) & out["market_prob_home"].notna()
    else:
        out["disagrees_with_market"] = False

    return out
