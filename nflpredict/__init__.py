"""nflpredict -- a walk-forward backtested NFL game outcome model.

Public surface:

    from nflpredict import load_games, build_features, GamePredictor, walk_forward
"""

from .backtest import BacktestResult, evaluate, walk_forward
from .data import load_games, load_team_game_epa
from .edge import attach_edges, expected_value, kelly_fraction
from .elo import EloEngine, elo_to_prob, elo_to_spread
from .features import FEATURE_COLUMNS, build_features
from .model import GamePredictor, devig_moneyline, spread_to_prob

__version__ = "1.0.0"

__all__ = [
    "BacktestResult", "EloEngine", "FEATURE_COLUMNS", "GamePredictor",
    "attach_edges", "build_features", "devig_moneyline", "elo_to_prob",
    "elo_to_spread", "evaluate", "expected_value", "kelly_fraction",
    "load_games", "load_team_game_epa", "spread_to_prob", "walk_forward",
]
