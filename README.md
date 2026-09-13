# nflpredict

An NFL game outcome model with a walk-forward backtest attached, so every
accuracy number it reports can be checked rather than believed.

```
$ nflpredict predict

------------------------------------------------------------------------------
2026 WEEK 1  --  16 games  (16 with a posted line, trained on 7,261 games)
------------------------------------------------------------------------------
MATCHUP             PICK     WIN%      CONF   MODEL    MKT   LINE   EDGE
------------------------------------------------------------------------
ARI @ LAC           LAC     79.8%      HIGH   +11.3  80.4%   +9.5   +1.8
CLE @ JAX           JAX     78.8%      HIGH   +10.8  78.7%   +8.5   +2.3
NO @ DET            DET     73.6%    MEDIUM    +8.5  74.1%   +7.0   +1.5
...
BUF @ HOU           BUF     50.7%  COINFLIP    -0.2  51.7%   -1.5   +1.3
```

## The headline, stated honestly

**This model is wrong about one game in three, and nothing will fix that.**

You asked for no mistakes. That is not achievable, and it is worth being
precise about why rather than hand-waving. Roughly a third of NFL results
turn on events with no predictable structure: a tipped ball, a 45-yard field
goal that hits the upright, a fumble that bounces one way instead of the
other, a starting quarterback who tweaks a hamstring in warmups. Those are
not modelling failures to be engineered away — they are the variance that
makes the sport worth watching.

The measurable ceiling is visible in the data. Over 1999-2026, the team that
Las Vegas installed as favourite — with a nine-figure incentive to be right,
full injury information, and the ability to move the line up to kickoff —
won **66.4%** of the time. That is the number a good model matches. Any tool
claiming 90%+, "guaranteed locks", or a zero-loss record is either fitting
noise, quietly testing on data it trained on, or lying to sell a pick.

Here is what this one actually does, replayed week by week across 18 seasons:

| Method | Games | Accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|
| **Model + market blend** | 4,897 | **66.61%** | 0.2099 | 0.6087 |
| Market (closing line) | 4,897 | 66.59% | 0.2099 | 0.6085 |
| Elo only | 4,897 | 65.10% | 0.2195 | 0.6305 |
| Model alone, no market | 4,897 | 65.06% | 0.2186 | 0.6280 |
| Always pick the home team | 4,897 | 55.85% | — | — |

*Walk-forward over 2008-2025: for every slate the model is fitted only on
games that had already finished. Reproduce with `nflpredict backtest`.*

Two results in that table deserve to be said out loud rather than buried:

1. **The model does not beat the market — it ties it.** 66.61% against
   66.59% is a coin-flip's worth of difference across 4,897 games. When the
   blend weight was tuned on 2008-2017 and checked on 2018-2025, the
   held-out optimum was *zero model weight*. NFL closing lines are close to
   informationally efficient, and this is what honest evidence of that looks
   like.
2. **It loses money against the spread.** 49.6% ATS over 4,897 games, an ROI
   of -5.3% at standard -110 pricing. Beating a spread requires 52.38% just
   to break even. It does not get there, and neither does almost anything
   else in public.

## Calibration

Accuracy alone can hide a badly behaved model, so the more useful question
is whether a stated 70% actually means 70%. It does, within noise:

| Stated confidence | Games | Said | Actual | Gap |
|---|---:|---:|---:|---:|
| 50-55% | 696 | 52.8% | 53.4% | +0.7% |
| 55-60% | 862 | 57.5% | 57.0% | -0.5% |
| 60-65% | 875 | 62.5% | 60.4% | -2.1% |
| 65-70% | 736 | 67.5% | 67.1% | -0.5% |
| 70-75% | 701 | 72.5% | 77.0% | +4.5% |
| 75-80% | 520 | 77.4% | 75.8% | -1.6% |
| 80-90% | 475 | 84.1% | 86.6% | +2.5% |

A `HIGH` confidence pick really does win around 80-85% of the time. A
`COINFLIP` really is close to a coin flip — and about 12 of a typical
16-game slate land in `LEAN` or `COINFLIP`, which is the honest shape of an
NFL week.

## Install

```bash
git clone <this repo> && cd res
pip install -e .
```

Python 3.9+. The first run downloads ~400 MB of play-by-play and distills it
to ~60 KB per season of cached summaries; subsequent runs are instant. Pass
`--epa-start 2018` to start with a much smaller download.

## Usage

```bash
nflpredict predict                      # next unplayed slate
nflpredict predict --season 2026 --week 5
nflpredict predict --csv                # also write out/predictions_*.csv

nflpredict backtest                     # reproduce the table above
nflpredict backtest --no-market         # model on its own merits
nflpredict backtest --refit week        # refit before every slate (slower)

nflpredict ratings --top 10             # current Elo power ratings
nflpredict evaluate --season 2025       # score one finished season
nflpredict update                       # refresh cached data
```

As a library:

```python
from nflpredict import load_games, load_team_game_epa, build_features, GamePredictor

games = load_games()
epa = load_team_game_epa(range(2006, 2027))
features = build_features(games, epa)

history = features[features.completed & (features.season < 2026)]
model = GamePredictor().fit(history)
model.predict(features[(features.season == 2026) & (features.week == 5)])
```

## How it works

```
nflverse game log ──┐
                    ├─> features ──> Elo + rolling form ──┐
nflverse play-by-play ┘                                   ├─> ensemble ──> blend ──> pick
                                                          │                  ↑
                              closing line / moneyline ────┴──────────────────┘
```

**Elo rating** (`elo.py`) — one continuous rating per franchise, updated
after every game with a margin-of-victory multiplier that damps blowouts so
ratings cannot be farmed against weak opponents. Ratings regress 25% toward
the mean each offseason. Home field advantage is re-estimated from a
trailing three-season window rather than hardcoded, because it has decayed
materially: +2.62 pts (1999-2007) → +1.78 (2015-2019) → +1.95 (2024-2026),
with 2020's empty stadiums measuring +0.14.

**Rolling form** (`features.py`) — offensive and defensive EPA per play,
success rate, explosive-play rate, turnover and sack rates, each an
exponentially-weighted average over prior games, centered on a
prior-seasons-only league baseline so era drift in scoring does not leak in.
Season boundaries blend last year's finish (shrunk toward average) into this
year's evidence over the first six games.

**Ensemble** (`model.py`) — logistic regression, a ridge margin model
converted through a normal CDF, and gradient boosting, averaged in log-odds
space. The linear pair carries most of the weight because it is better
calibrated; the GBM adds interaction shape.

**Market blend** (`model.py`) — the line is deliberately kept *out* of the
feature matrix and applied afterwards at the probability level. That keeps
`--no-market` a clean switch, lets the model be judged on its own, and
degrades gracefully for games with no posted line. Moneylines are de-vigged
proportionally; spreads go through an empirically fitted logistic curve.

### Why the backtest is trustworthy

The failure mode for this kind of project is leakage — letting the model
glimpse information from the future, then reporting a spectacular accuracy
that evaporates in production. Two structural guarantees prevent it here:

1. Elo and rolling form are built by a single chronological pass that
   records state *before* folding in each game's result.
2. `walk_forward` refits using only games with an earlier (season, week).

Both are asserted directly in `tests/test_leakage.py`, which rebuilds the
entire feature matrix from a dataset with every future result erased and
requires that not one past feature value moves.

## What didn't work

Kept here because negative results are the useful half of a model's story.

**A quarterback adjustment made it worse.** The intuition is sound — a
backup starting is worth several points — but this implementation attributed
whole-team offensive EPA to the starter, double-counting team strength that
the Elo rating already carried:

| Setting | Accuracy | Brier |
|---|---:|---:|
| No QB adjustment | **65.06%** | 0.2186 |
| QB term, scale 15 / cap 3 | 64.33% | 0.2186 |
| QB term, scale 25 / cap 4 | 64.10% | 0.2192 |
| QB term, scale 45 / cap 6 | 64.04% | 0.2207 |

It is off by default and reachable via `--qb-adjustment` so the experiment
stays reproducible. Doing it properly needs player-level data separating a
passer from his supporting cast.

**EPA features help, but modestly.** 65.06% with them against 64.77% for Elo
alone — real and consistent across seasons, but a fraction of what the raw
correlations suggest, because Elo already encodes most of it.

**Refitting every week instead of every season changed nothing.** 64.20% vs
64.04%, for 12x the compute. `--refit season` is the default for that reason.

**Large-edge ATS buckets are noise, not signal.** The backtest shows a 60.9%
win rate on picks where the model disagrees with the spread by 3+ points —
on 23 bets. That is a sample of 23. The tool prints the warning next to the
number because this is exactly the trap that sells betting systems.

## Data

Everything comes from [nflverse](https://github.com/nflverse), free and
public, no API key:

- **Game log** — `nflverse/nfldata`: scores, closing spreads and moneylines,
  rest days, venue, weather, starters, back to 1999.
- **Play-by-play** — `nflverse/nflverse-data`: ~50k plays per season with
  EPA, win probability and success flags.

Cached under `data/` (gitignored). `nflpredict update` refreshes.

## Development

```bash
pip install -e . && pip install pytest
pytest -q        # 52 tests
```

Layout: `data.py` fetch/cache · `elo.py` ratings · `features.py` leak-free
matrix · `model.py` ensemble + blend · `backtest.py` walk-forward + metrics ·
`edge.py` EV and Kelly · `report.py` formatting · `cli.py` commands.

## Using this responsibly

It is a forecasting and analysis tool: power ratings, calibrated
probabilities, and an honest measure of how much of the game is actually
predictable. That is what it is good for.

If anyone points it at a sportsbook anyway, the backtest is unambiguous — it
loses at -110, and the `edge.py` Kelly sizing is quarter-Kelly precisely
because a Brier score of 0.21 means the probabilities are good but not
sharp enough to bet aggressively on. A model that ties the closing line is a
genuinely good model and still not a profitable one, because the price
already contains everything it knows.
