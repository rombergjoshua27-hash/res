"""Central configuration: data sources, cache paths, and model constants.

Constants tagged EMPIRICAL were measured directly from the nflverse game log
(1999-2026 regular season + playoffs, n=7548 completed games). The derivation
for each is reproducible via ``nflpredict calibrate-constants``.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_DIR = Path(os.environ.get("NFLPREDICT_DATA_DIR", PROJECT_ROOT / "data"))
EPA_CACHE_DIR = DATA_DIR / "epa"
PLAYER_CACHE_DIR = DATA_DIR / "players"
OUTPUT_DIR = Path(os.environ.get("NFLPREDICT_OUT_DIR", PROJECT_ROOT / "out"))

# --------------------------------------------------------------------------
# Data sources (nflverse — free, public, no API key)
# --------------------------------------------------------------------------

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/"
    "play_by_play_{season}.csv.gz"
)

PLAYER_STATS_URL_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/stats_player/"
    "stats_player_week_{season}.csv"
)
INJURIES_URL_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/injuries/"
    "injuries_{season}.csv"
)

# Re-download the schedule file if the cached copy is older than this.
# Scores and lines move during the week, so keep this short.
GAMES_CACHE_HOURS = 6.0

# Play-by-play availability. EPA is only modelled from 1999 onward.
FIRST_PBP_SEASON = 1999

# Bumped whenever the team-game aggregate schema changes, so a stale cache is
# rebuilt rather than read back with columns silently missing.
EPA_CACHE_VERSION = 2

# Per-player weekly stats run the full length of the game log. Injury reports
# start later, so any model using them must degrade gracefully before 2009.
FIRST_PLAYER_STATS_SEASON = 1999
FIRST_INJURY_SEASON = 2009
PLAYER_CACHE_VERSION = 1

# Injury report statuses, in order of severity. A player not on the report at
# all is available; "Out" never plays. EMPIRICAL play rates from 2009-2026 are
# measured in `players.py`.
INJURY_STATUS_ORDER = ("Out", "Doubtful", "Questionable")

DOWNLOAD_RETRIES = 4
DOWNLOAD_BACKOFF_SECONDS = 2.0
DOWNLOAD_TIMEOUT_SECONDS = 300

# --------------------------------------------------------------------------
# Elo engine
# --------------------------------------------------------------------------

ELO_INIT = 1500.0
ELO_DIVISOR = 400.0

# K controls how fast ratings move. 20 is the FiveThirtyEight NFL value and
# backtests near-optimally for log loss; see `nflpredict tune-elo`.
ELO_K = 20.0

# Margin-of-victory multiplier damping. Prevents good teams from farming
# rating points via blowouts against bad ones (Silver & Fischer-Baum, 2015).
ELO_MOV_B = 2.2

# Fraction of a team's deviation from 1500 that carries into the next season.
# 0.75 => regress 25% toward the mean each offseason. EMPIRICAL: maximises
# out-of-sample log loss on week 1-4 games across 2006-2025.
ELO_SEASON_CARRY = 0.75

# Elo points per point of point spread. EMPIRICAL: regressing game margin on
# pregame Elo difference gives ~25 Elo == 1 point.
ELO_PER_POINT = 25.0

# --------------------------------------------------------------------------
# Home field advantage
# --------------------------------------------------------------------------

# Fallback only. The Elo engine re-estimates HFA from a trailing window of
# completed games (see elo.estimate_hfa), because HFA has decayed materially:
# 2.62 pts (1999-2007) -> 1.78 (2015-2019) -> 1.95 (2024-2026).
HFA_POINTS_DEFAULT = 1.95
HFA_TRAILING_SEASONS = 3
HFA_MIN_POINTS = 0.0
HFA_MAX_POINTS = 3.5

# 2020 was played with empty or near-empty stadiums; measured HFA was +0.14.
HFA_SEASON_OVERRIDES = {2020: 0.15}

# --------------------------------------------------------------------------
# Situational adjustments (in points, converted to Elo via ELO_PER_POINT)
# --------------------------------------------------------------------------

# EMPIRICAL: mean home margin rises from +1.45 (away rested) to +3.50 (home
# rested) across the rest-differential range -> ~0.10 pts per day, capped.
REST_POINTS_PER_DAY = 0.10
REST_DIFF_CAP_DAYS = 10.0

# Quarterback adjustment. A team's Elo already embeds whoever has been
# starting; when a different QB starts, shift by the gap between his rating
# and that baseline.
#
# DEFAULT OFF. Measured, it makes the model *worse* at every scale tested
# (walk-forward 2008-2025, market off):
#
#     no QB adjustment      65.06%  brier 0.2186
#     QB scale=15 cap=3     64.33%  brier 0.2186
#     QB scale=25 cap=4     64.10%  brier 0.2192
#     QB scale=45 cap=6     64.04%  brier 0.2207
#
# Re-confirmed on 2008-2026 with the current feature set: the default
# scale takes the model from 65.1% / brier 0.2186 / log loss 0.6281 to
# 64.1% / 0.2206 / 0.6325, and drags Elo alone from 65.1% to 63.3%.
#
# The cause is double counting: this rating attributes whole-team offensive
# EPA to the starter, but team strength is already in the Elo rating, so the
# adjustment re-applies a signal the model has. Isolating true QB value needs
# player-level data that separates the passer from his supporting cast.
# Kept behind a flag (`--qb-adjustment`) rather than deleted so the
# experiment stays reproducible.
ELO_USE_QB_DEFAULT = False

# The *second* quarterback attempt, which did work, and is on by default.
#
# `qb.py` rates a passer by completion percentage over expected and EPA per
# dropback -- measures computed per throw, not per team play -- and feeds the
# model the *change* from the quarterback a team has been playing, plus an
# injury-report availability term. The weights are fitted, not chosen.
#
# Walk-forward 2008-2026, 4,912 games:
#
#                        accuracy   brier    log loss
#   market off, base       65.07%   0.2186     0.6281
#   market off, + QB       64.98%   0.2171     0.6247   <- p=0.002 on Brier
#   market on,  base       66.61%   0.2099     0.6087
#   market on,  + QB       66.59%   0.2099     0.6085   <- p=0.064
#
# Unchanged accuracy, significantly better probabilities, market-free: on the
# 280 games where the two disagreed on the pick it was 138-142, a dead heat,
# while the paired Brier improvement is real. With the market on it is a wash
# -- the closing line already prices a backup quarterback and a Friday injury
# report, so only 13 of 4,912 picks moved.
# It is on because the market-free path is the one that matters when no line
# is posted; `--no-qb-features` turns it off.
USE_QB_FEATURES_DEFAULT = True
QB_EWMA_ALPHA = 0.25          # responsiveness of a QB's own rolling rating
QB_SHRINK_GAMES = 8.0         # games of league-average prior before trusting
QB_EPA_TO_POINTS = 45.0       # 0.10 EPA/play gap ~= 4.5 points of spread
QB_ADJ_CAP_POINTS = 6.0       # hard cap so a backup never swings 2 TDs

# --------------------------------------------------------------------------
# Margin -> probability conversion
# --------------------------------------------------------------------------

# EMPIRICAL: std(actual margin - closing spread) = 13.2 over 1999-2026,
# 12.73 over 2015-2026. Used to turn a predicted margin into a win probability.
MARGIN_SIGMA = 13.2

# Ties are rare (15 / 7548 = 0.2%) and are scored as half a win.
TIE_CREDIT = 0.5

# --------------------------------------------------------------------------
# Point totals
# --------------------------------------------------------------------------

# EMPIRICAL: std(actual total - closing total) = 13.43 over 1999-2026,
# 13.31 over 2007-2026, 13.24 over 2015-2026. Used to turn a predicted total
# into an over/under probability. A fitted model re-measures this on its own
# residuals; this is the fallback and the sanity bound.
TOTAL_SIGMA = 13.3

# Weight on the model when blending with the posted total.
#
# Tuned on 2008-2017 (n=2,670) and validated on 2018-2026 (n=2,242). The
# result is the same one the spread gives: the closing number is already
# about as good as this gets. Held-out MAE by weight:
#
#     w=0.00 (market only)  10.4300
#     w=0.10                10.4270   <- held-out optimum
#     w=0.20                10.4304   <- optimum on the tuning years
#     w=0.50                10.4756
#
# The tuning years picked w=0.20, which on held-out data was 0.0004 points
# per game *worse* than simply posting the market number. 0.10 is kept
# because it is the held-out optimum and matches the spread default, but the
# honest reading is that the curve is flat and none of this is a real edge.
# `--total-blend` overrides it; `--no-market` gives the pure model, which is
# what a game with no posted total gets anyway.
DEFAULT_TOTAL_MARKET_BLEND = 0.10

# Roughly half of posted totals are whole numbers, and 2.9% of those land
# exactly on the number for a push. Reported, never silently dropped.
TOTAL_PUSH_RATE_WHOLE_LINES = 0.0286

# --------------------------------------------------------------------------
# Rolling team form (EPA) features
# --------------------------------------------------------------------------

EPA_EWMA_ALPHA = 0.30         # weight on the most recent game
EPA_MIN_PLAYS = 10            # ignore team-games with fewer scrimmage plays
# Carry-over from the prior season when the current one has little data.
EPA_PRIOR_SEASON_CARRY = 0.55
EPA_BLEND_FULL_GAMES = 6.0    # games before current season fully displaces prior

# --------------------------------------------------------------------------
# Model / blending
# --------------------------------------------------------------------------

# Seasons before this are used to warm up Elo and the rolling form but never
# scored in a backtest. With play-by-play loaded from 2006, 2006 centres its
# own league baselines (see features._league_baselines) and 2007 is the first
# season with a full prior season of form behind every team -- so scoring
# starts at 2008, one clear season after that. Every accuracy figure this
# project publishes is measured on this default.
DEFAULT_BACKTEST_START = 2008
MIN_TRAIN_SEASONS = 4

# Weight on the model when blending with the market price.
#
# Tuned on 2008-2017 and validated on 2018-2025. The honest result is that
# the closing line is close to unbeatable: held-out Brier was flat from
# w=0.00 (0.2101) to w=0.10 (0.2104) and degraded monotonically above that.
# 0.10 keeps a small model tilt while staying inside the noise band of the
# market itself; `--market-blend` overrides it, and `--no-market` gives the
# pure model (needed anyway for games with no line posted yet).
DEFAULT_MARKET_BLEND = 0.10

# Standard US odds juice on a point-spread bet.
STANDARD_VIG_ODDS = -110

# Confidence tiers for reporting, by blended win probability.
CONFIDENCE_TIERS = (
    (0.75, "HIGH"),
    (0.65, "MEDIUM"),
    (0.575, "LEAN"),
    (0.0, "COINFLIP"),
)

# --------------------------------------------------------------------------
# What a stated confidence has actually been worth
# --------------------------------------------------------------------------
#
# A confidence rating nobody has checked is decoration. These are the
# realised rates behind each one, measured walk-forward over 2008-2026
# (4,912 games) and reproducible with `nflpredict backtest`. They are what
# the Betting Picks tab quotes next to the model's own number, because the
# gap between the two is the only thing that makes a confidence useful.

# Straight-up picks: (lower bound of stated confidence, games, realised rate).
# The model is well calibrated here -- a stated 70% really does win about 70%.
CALIBRATION_BANDS = (
    (0.90, 32, 0.969),
    (0.80, 475, 0.866),
    (0.75, 522, 0.757),
    (0.70, 703, 0.771),
    (0.65, 738, 0.670),
    (0.60, 877, 0.605),
    (0.55, 866, 0.571),
    (0.50, 699, 0.534),
)

# Against the spread, by how far the model's number sits from the posted one:
# (minimum edge in points, bets, realised win rate).
#
# Every row with a usable sample is below the 52.38% needed to break even at
# -110. The 3.0 row is 60.9% on twenty-three bets, which is not an edge, it is
# a sample of twenty-three -- and is exactly the row a betting system would
# advertise. It is kept so the small sample is visible rather than omitted.
ATS_BY_EDGE = (
    (3.0, 23, 0.609),
    (2.0, 317, 0.478),
    (1.5, 888, 0.508),
    (1.0, 2036, 0.494),
    (0.5, 3488, 0.493),
    (0.0, 4912, 0.496),
)

# Over/under, same shape. The 3-4 point rows creep just past breakeven, but
# on 940-1,612 bets the standard error is about 1.5 points of win rate, so
# they are indistinguishable from it.
OU_BY_EDGE = (
    (5.0, 532, 0.509),
    (4.0, 940, 0.526),
    (3.0, 1612, 0.525),
    (2.0, 2477, 0.518),
    (1.0, 3651, 0.513),
    (0.0, 4912, 0.508),
)

# Win rate needed to break even at standard -110 pricing.
BREAKEVEN_AT_STANDARD_VIG = 0.5238

# --------------------------------------------------------------------------
# Per-game confidence on the spread and the total
# --------------------------------------------------------------------------
#
# The tables above bucket by the *blended* number's edge, which is the wrong
# quantity to rate a bet by: blending pulls the forecast onto the line, so the
# blended edge spans barely four points and every pick looks identical. The
# model's own unblended number disagrees with the market by up to twenty-one
# points, and that is the quantity worth showing.
#
# These are logistic fits of "did this pick win" on the unblended edge in
# points, over the 2008-2026 walk-forward. (intercept, slope).
#
# Both slopes are positive, and both confidence intervals contain zero:
#
#   totals   slope +0.0288  95% CI [-0.0012, +0.0563]  n=4,853
#   spreads  slope +0.0086  95% CI [-0.0186, +0.0369]  n=4,788
#
# So a bigger disagreement with the line is associated with a slightly better
# result, and the association is not distinguishable from chance. The fits are
# used to *differentiate* picks -- which is honest, the edge is a real measured
# quantity -- and not to claim any of them beats the vig. Anything derived from
# them is clamped by CONVICTION_CAP below.
OU_EDGE_FIT = (-0.039792, 0.028793)
ATS_EDGE_FIT = (0.014211, 0.008616)

# Realised win rate by size of the unblended edge: (minimum points, bets, rate).
# The 6+ rows clear breakeven, on 263-277 bets, where the standard error is
# about three points of win rate. They are suggestive and they are not
# significant, and the sample size is printed beside every one of them so the
# reader can see which.
ATS_BY_MODEL_EDGE = (
    (6.0, 263, 0.540),
    (4.0, 608, 0.508),
    (2.0, 1491, 0.518),
    (0.0, 2426, 0.500),
)
OU_BY_MODEL_EDGE = (
    (6.0, 277, 0.538),
    (4.0, 652, 0.521),
    (2.0, 1521, 0.513),
    (0.0, 2403, 0.497),
)

# How far the fitted curve is allowed to run, and the most important line in
# this block.
#
# The curve is a point estimate off a slope whose confidence interval contains
# zero. Left uncapped it quotes 56% on a large edge, which is above the 52.38%
# needed at -110, so the tab would start advertising profitable bets on the
# strength of a relationship it cannot demonstrate. It did exactly that for one
# commit, and the test that forbids a positive expected value caught it.
#
# Capping at breakeven is the honest ceiling: the most this evidence supports
# saying about any spread or total is "a coin flip against the price". Below the
# cap the ranking is fully differentiated, which is the point of the column; at
# the cap it stops, which is the point of the cap. Roughly one total in seven
# reaches it and almost no spread does, so little ordering is lost.
#
# The measured band rate is still printed raw beside each pick, sample size and
# all, so where the realised rate runs past this ceiling the reader can see it
# and see what it rests on.
CONVICTION_CAP = BREAKEVEN_AT_STANDARD_VIG

# Labels for how far the model sits from the posted number. These describe the
# size of the disagreement, not a claim about winning: a BIG edge means the
# model and the market disagree by six points or more, and nothing else.
CONVICTION_TIERS = (
    (6.0, "BIG"),
    (4.0, "MODERATE"),
    (2.0, "SLIGHT"),
    (0.0, "NONE"),
)

# --------------------------------------------------------------------------
# Franchise continuity
# --------------------------------------------------------------------------

# nflverse uses the abbreviation in force at the time, so a relocated club
# appears under two codes. Map historical codes onto the current one so a
# franchise keeps one continuous Elo rating across its move.
FRANCHISE_MAP = {
    "SD": "LAC",   # San Diego -> Los Angeles Chargers (2017)
    "STL": "LA",   # St. Louis -> Los Angeles Rams (2016)
    "OAK": "LV",   # Oakland -> Las Vegas Raiders (2020)
}

N_FRANCHISES = 32
