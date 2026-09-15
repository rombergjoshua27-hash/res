"""nflpredict -- a walk-forward backtested NFL game outcome model.

Public surface:

    from nflpredict import load_games, build_features, walk_forward
    from nflpredict import GamePredictor    # winner and spread
    from nflpredict import TotalsPredictor  # point total and over/under
"""

from .backtest import BacktestResult, evaluate, evaluate_totals, walk_forward
from .data import load_games, load_team_game_epa
from .edge import attach_edges, expected_value, kelly_fraction, prob_to_american
from .elo import EloEngine, elo_to_prob, elo_to_spread
from .features import FEATURE_COLUMNS, TOTAL_FEATURE_COLUMNS, build_features
from .model import GamePredictor, devig_moneyline, spread_to_prob
from .totals import TotalsPredictor, over_probability

__version__ = "1.1.0"

__all__ = [
    "BacktestResult", "EloEngine", "FEATURE_COLUMNS", "GamePredictor",
    "TOTAL_FEATURE_COLUMNS", "TotalsPredictor", "attach_edges",
    "build_features", "devig_moneyline", "elo_to_prob", "elo_to_spread",
    "evaluate", "evaluate_totals", "expected_value", "kelly_fraction",
    "load_games", "load_team_game_epa", "over_probability",
    "prob_to_american", "spread_to_prob", "walk_forward",
]
