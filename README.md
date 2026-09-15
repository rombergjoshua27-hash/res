# nflpredict

An NFL game model that forecasts all three markets — **moneyline, spread and
point total** — with a walk-forward backtest attached, so every accuracy
number it reports can be checked rather than believed.

```
$ nflpredict predict

------------------------------------------------------------------------------
2026 WEEK 2  --  16 games  (16 with a posted line, trained on 7,276 games,
                            incl. 15 from 2026 wk 1)
------------------------------------------------------------------------------
MATCHUP             PICK     WIN%      CONF   MODEL    MKT   LINE   EDGE
------------------------------------------------------------------------
MIA @ SF            SF      86.4%      HIGH   +14.8  87.2%  +13.5   +1.3
CLE @ TB            TB      75.0%      HIGH    +9.1  75.7%   +8.5   +0.6
NO @ BAL            BAL     74.7%    MEDIUM    +9.0  75.1%   +8.5   +0.5
...
CAR @ ATL           CAR     51.6%  COINFLIP    -0.6  53.2%   -1.5   +0.9

------------------------------------------------------------------------------
POINT TOTALS
------------------------------------------------------------------------------
MATCHUP             PICK     MODEL   LINE   PROJ   EDGE  P(OVER)      CONF
--------------------------------------------------------------------------
DET @ BUF           UNDER     49.8   53.5   53.1   -3.7    48.9%  COINFLIP
IND @ KC            UNDER     43.0   48.5   47.9   -5.5    48.4%  COINFLIP
SEA @ ARI           OVER      45.0   41.5   41.8   +3.5    51.0%  COINFLIP
...
```

## The headline, stated honestly

**This model is wrong about one game in three, and nothing will fix that.**

Roughly a third of NFL results turn on events with no predictable structure:
a tipped ball, a 45-yard field goal that hits the upright, a fumble that
bounces one way instead of the other, a starting quarterback who tweaks a
hamstring in warmups. Those are not modelling failures to be engineered
away — they are the variance that makes the sport worth watching.

The measurable ceiling is visible in the data. Over 1999-2026, the team that
Las Vegas installed as favourite — with a nine-figure incentive to be right,
full injury information, and the ability to move the line up to kickoff —
won **66.4%** of the time. That is the number a good model matches. Any tool
claiming 90%+, "guaranteed locks", or a zero-loss record is either fitting
noise, quietly testing on data it trained on, or lying to sell a pick.

Here is what this one actually does, replayed week by week across 19 seasons.

### Picking winners

| Method | Games | Accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|
| **Model + market blend** | 4,912 | **66.61%** | 0.2099 | 0.6087 |
| Market (closing line) | 4,912 | 66.61% | 0.2099 | 0.6086 |
| Elo only | 4,912 | 65.11% | 0.2195 | 0.6305 |
| Model alone, no market | 4,912 | 65.07% | 0.2186 | 0.6281 |
| Always pick the home team | 4,912 | 55.86% | — | — |

### Predicting point totals

Scored as a regression, because the useful question is how many points the
forecast missed by — not whether it landed on the right side of a line.

| Method | Games | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| **Model + market blend** | 4,912 | **10.460** | 13.271 | +0.591 |
| Market (closing total) | 4,912 | 10.466 | 13.279 | +0.587 |
| Model alone, no market | 4,912 | 10.679 | 13.519 | +0.629 |

*Walk-forward over 2008-2026: for every slate both models are fitted only on
games that had already finished. Reproduce with `nflpredict backtest`.*

Three results in those tables deserve to be said out loud rather than buried:

1. **The model does not beat the market — it ties it.** 66.61% against
   66.61% on sides; 10.460 against 10.466 points of error on totals. When
   the blend weights were tuned on 2008-2017 and checked on 2018-2026, the
   held-out optimum was *zero model weight* for sides, and for totals the
   tuned weight came out 0.0004 points per game **worse** than simply
   posting the market number. NFL closing lines are close to
   informationally efficient, and this is what honest evidence of that
   looks like.
2. **It loses money against the spread.** 49.60% ATS over 4,912 games, an
   ROI of -5.30% at standard -110 pricing. Beating a spread requires 52.38%
   just to break even.
3. **Totals are harder than sides, not easier.** Over/under picks hit
   50.77%, ROI -3.07%. The 3- and 4-point edge buckets creep just past
   breakeven (52.47% on 1,612 bets, 52.64% on 940), but the standard error
   on samples that size is around 1.5 points of win rate. That is noise
   wearing the costume of a system.

## Calibration

Accuracy alone can hide a badly behaved model, so the more useful question
is whether a stated 70% actually means 70%. It does, within noise:

| Stated confidence | Games | Said | Actual | Gap |
|---|---:|---:|---:|---:|
| 50-55% | 699 | 52.8% | 53.4% | +0.6% |
| 55-60% | 866 | 57.5% | 57.1% | -0.4% |
| 60-65% | 877 | 62.5% | 60.5% | -2.0% |
| 65-70% | 738 | 67.6% | 67.0% | -0.5% |
| 70-75% | 703 | 72.5% | 77.1% | +4.6% |
| 75-80% | 522 | 77.4% | 75.7% | -1.7% |
| 80-90% | 475 | 84.1% | 86.6% | +2.5% |
| 90-100% | 32 | 91.5% | 96.9% | +5.4% |

A `HIGH` confidence pick really does win around 80-85% of the time. A
`COINFLIP` really is close to a coin flip — and about 12 of a typical
16-game slate land in `LEAN` or `COINFLIP`, which is the honest shape of an
NFL week.

Point totals get no such table, and the reason is worth stating: at the
default blend the model's over/under probabilities sit within a couple of
points of 50% for almost every game. That is not a bug to be tuned away.
It is the blend correctly reporting that it cannot separate itself from the
posted number.

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
nflpredict predict                      # next unplayed slate: sides and totals
nflpredict predict --season 2026 --week 5
nflpredict predict --csv                # also write out/predictions_*.csv

nflpredict export                       # everything, as an Excel workbook
nflpredict export -o ~/week2.xlsx

nflpredict backtest                     # reproduce the tables above
nflpredict backtest --no-market         # both models on their own merits
nflpredict backtest --refit week        # refit before every slate (slower)

nflpredict ratings --top 10             # current Elo power ratings
nflpredict evaluate --season 2025       # score one finished season
nflpredict update                       # refresh cached data
```

As a library:

```python
from nflpredict import load_games, load_team_game_epa, build_features
from nflpredict import GamePredictor, TotalsPredictor

games = load_games()
epa = load_team_game_epa(range(2006, 2027))
features = build_features(games, epa)

history = features[features.completed & (features.season < 2026)]
slate = features[(features.season == 2026) & (features.week == 5)]

GamePredictor().fit(history).predict(slate)    # winner + spread
TotalsPredictor().fit(history).predict(slate)  # point total + over/under
```

## The Excel workbook

`nflpredict export` writes a ten-sheet workbook — the format most people
actually want to read a slate in, and the one that makes the model
auditable by someone who never opens the code.

| Sheet | What it holds |
|---|---|
| Read Me | What everything means, and how accurate it honestly is |
| Predictions | The slate: pick, win %, confidence, model spread vs. the line, **fair moneyline vs. the book's price, and EV per $1** |
| Point Totals | The slate scored for points: model total, posted total, the blend, over/under and P(over) |
| Power Ratings | Every franchise's Elo, in points |
| Backtest Summary | Model vs. market vs. Elo vs. picking the home team |
| Calibration | Whether a stated 70% wins 70% |
| Against the Spread | ATS record at six edge thresholds |
| Point Totals Backtest | MAE, RMSE and bias, plus the over/under record |
| Accuracy by Season | Year-by-year out-of-sample results |
| Game Log | Every backtested game, one row each |

Every summary figure is a **live formula over the Game Log**, not a value
pasted in by Python — so filtering the log re-scores the whole workbook, and
a reader can click any cell to see exactly how the number was reached. Each
formula is written together with its Python-computed result as the cached
value, so the workbook reads correctly before Excel recalculates it and
recalculates on open so a stale cache cannot survive being looked at. The
test suite checks both halves: that the cached values agree with `backtest`,
which computes the same quantities through entirely separate code, and that
no formula cell is left without one.

## Using results as the season goes

Predictions are fitted on **every game that had finished before the slate's
first kickoff** — not merely on prior weeks. Running on a Monday in week 2
therefore trains on week 1's Sunday results, and the header says so:

```
2026 WEEK 2  --  16 games  (16 with a posted line, trained on 7,276 games,
                            incl. 15 from 2026 wk 1)
```

Two consequences worth knowing:

* **The default slate rolls forward once a week is mostly played.** On a
  Monday, week N has a single night game left and week N+1 is what actually
  needs predicting. The straggler stays reachable with `--week N`.
* **Every game in a slate shares one cutoff** — the slate's first kickoff.
  A Sunday game is not fitted on more information than the Thursday game it
  sits beside, which keeps a week internally comparable.

Early in a season the form features are still mostly carrying last year's
evidence, shrunk toward league average, and they slide across to the current
season over the first six games. A week 2 forecast is real, but it leans on
Elo and last season far more than a week 12 forecast does.

## How it works

```
nflverse game log ──┐
                    ├─> features ──> Elo + rolling form ──┬─> ensemble ──> blend ──> pick
nflverse play-by-play ┘                                   │                  ↑
                                                          │   closing line ──┤
                              weather / roof / pace ───────┴─> totals model ─┘
```

**Elo rating** (`elo.py`) — one continuous rating per franchise, updated
after every game with a margin-of-victory multiplier that damps blowouts so
ratings cannot be farmed against weak opponents. Ratings regress 25% toward
the mean each offseason. Home field advantage is re-estimated from a
trailing three-season window rather than hardcoded, because it has decayed
materially: +2.62 pts (1999-2007) → +1.78 (2015-2019) → +1.95 (2024-2026),
with 2020's empty stadiums measuring +0.14.

**Rolling form** (`features.py`) — offensive and defensive EPA per play,
success rate, explosive-play rate, turnover and sack rates, plus possessions
and pace, each an exponentially-weighted average over prior games, centered
on a prior-seasons-only league baseline so era drift in scoring does not
leak in. Season boundaries blend last year's finish (shrunk toward average)
into this year's evidence over the first six games.

**Winner and spread** (`model.py`) — logistic regression, a ridge margin
model converted through a normal CDF, and gradient boosting, averaged in
log-odds space. The linear pair carries most of the weight because it is
better calibrated; the GBM adds interaction shape. A win probability is
also published as a **fair moneyline**, vig-free, to be read against the
book's price.

**Point totals** (`totals.py`) — a different question, so a different model.
A total does not care which team is better: two efficient offences and two
leaky defences both push the number up, so every input is a *sum* across the
two sides rather than a difference. Ridge plus gradient boosting over
combined scoring, efficiency, explosiveness, possessions, pace, pass rate,
and **weather** — wind, temperature, and whether the roof is closed. Indoors
is encoded as calm and mild rather than as missing data. The predicted total
becomes an over/under probability through a normal CDF, with a continuity
correction so a whole-number line is graded as the push it can actually be.

**Market blend** (`model.py`, `totals.py`) — the line is deliberately kept
*out* of both feature matrices and applied afterwards at the probability or
points level. That keeps `--no-market` a clean switch, lets each model be
judged on its own, and degrades gracefully for games with no posted number.
Moneylines are de-vigged proportionally; spreads go through an empirically
fitted logistic curve.

### Why the backtest is trustworthy

The failure mode for this kind of project is leakage — letting the model
glimpse information from the future, then reporting a spectacular accuracy
that evaporates in production. Two structural guarantees prevent it here:

1. Elo and rolling form are built by a single chronological pass that
   records state *before* folding in each game's result.
2. `walk_forward` refits using only games with an earlier (season, week).

Both are asserted directly in `tests/test_leakage.py`, which rebuilds the
entire feature matrix from a dataset with every future result erased and
requires that not one past feature value moves. The sharpest case gets its
own test: a team's `points_for` comes from the final score of a game, so the
suite erases the highest-scoring game on record and requires that game's own
features to be unchanged while the team's *next* game moves.

**One documented exception.** The first season of loaded data has no earlier
season to centre its league baselines on, so it is centred on its own mean —
about one part in five hundred per game, spread across a league-wide
average. That season is a warm-up season the backtest never scores: the
defaults leave eight seasons between the two, and the CLI refuses a
`--start` that would reach back into it. A test pins the exception's scope by
requiring every later season to be exactly untouched by its own games.

## What didn't work

Kept here because negative results are the useful half of a model's story.

**A quarterback adjustment made it worse.** The intuition is sound — a
backup starting is worth several points — but this implementation attributed
whole-team offensive EPA to the starter, double-counting team strength that
the Elo rating already carried:

Measured market-free over 2008-2026, turning it on costs a point of accuracy
and makes the probabilities worse on both proper scoring rules:

| Setting | Games | Accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|
| No QB adjustment (default) | 4,912 | **65.1%** | **0.2186** | **0.6281** |
| QB adjustment on | 4,912 | 64.1% | 0.2206 | 0.6325 |

It degrades the Elo rating itself as much as the ensemble — Elo alone falls
from 65.1% to 63.3% with the term switched on. A sweep across adjustment
scales (15/25/45 Elo per EPA point, caps of 3-6 points) was worse at every
setting; those figures are recorded in `config.py` against the range they
were measured on.

It is off by default and reachable via `--qb-adjustment` so the experiment
stays reproducible. Doing it properly needs player-level data separating a
passer from his supporting cast.

**Recency weighting made the totals model worse.** Scoring has drifted
upward — 41.5 points per game in 2006 against ~45.5 today — so weighting
recent seasons more heavily looks obviously right. It is not. It buys a
smaller bias and pays for it with a larger error:

| Training weight | MAE | Bias |
|---|---:|---:|
| All history, unweighted | **10.679** | +0.630 |
| Half-life 8 seasons | 10.682 | +0.553 |
| Half-life 4 seasons | 10.698 | +0.495 |
| Last 10 seasons only | 10.708 | +0.507 |
| Last 6 seasons only | 10.706 | +0.457 |

Centering the *features* on a prior-seasons-only baseline already absorbs
most of the drift, and past that point more data beats fresher data.

**EPA features buy calibration, not accuracy.** Set against Elo alone, the
full model gets essentially the same games right and is meaningfully more
honest about how sure it is:

| Market-free method | Games | Accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|
| Model alone (Elo + EPA form) | 4,912 | 65.07% | **0.2186** | **0.6281** |
| Elo only | 4,912 | 65.11% | 0.2195 | 0.6305 |

Accuracy is a coin flip apart — Elo is ahead by two games in five thousand,
which is nothing. What the EPA features actually improve is the probability
attached to the pick. That is a smaller claim than the raw correlations
suggest, and it is smaller than this project originally reported, because
Elo already encodes most of what team efficiency has to say.

**Refitting every week instead of every season is not worth the compute.**
`--refit week` refits before each of the ~320 slates rather than once per
season, roughly 19x the fits, and a prior measurement put the accuracy
difference inside a fifth of a point. `--refit season` is the default for
that reason; the stricter path stays available and is worth re-running if
the feature set changes materially.

**Large-edge ATS buckets are noise, not signal.** The backtest shows a 60.9%
win rate on picks where the model disagrees with the spread by 3+ points —
on 23 bets. That is a sample of 23. The tool prints the warning next to the
number because this is exactly the trap that sells betting systems.

## Data

Everything comes from [nflverse](https://github.com/nflverse), free and
public, no API key:

- **Game log** — `nflverse/nfldata`: scores, closing spreads, totals and
  moneylines, rest days, venue, roof, temperature, wind, starters, back to
  1999.
- **Play-by-play** — `nflverse/nflverse-data`: ~50k plays per season with
  EPA, win probability, drive identifiers and success flags.

Cached under `data/` (gitignored). `nflpredict update` refreshes.

## Development

```bash
pip install -e . && pip install pytest
pytest -q        # 101 tests
```

Layout: `data.py` fetch/cache · `elo.py` ratings · `features.py` leak-free
matrix · `model.py` winner/spread ensemble + blend · `totals.py` point
totals · `backtest.py` walk-forward + metrics · `edge.py` EV, fair odds and
Kelly · `report.py` formatting · `excel.py` workbook · `cli.py` commands.

## Using this responsibly

It is a forecasting and analysis tool: power ratings, calibrated
probabilities, point-total projections, and an honest measure of how much of
the game is actually predictable. That is what it is good for.

If anyone points it at a sportsbook anyway, the backtest is unambiguous — it
loses at -110 on sides *and* on totals, and the `edge.py` Kelly sizing is
quarter-Kelly precisely because a Brier score of 0.21 means the
probabilities are good but not sharp enough to bet aggressively on. The
`EV / $1` column in the workbook is negative on every game of a typical
slate, and that is not a bug in the model — it is the vig, shown rather than
hidden. A model that ties the closing line is a genuinely good model and
still not a profitable one, because the price already contains everything it
knows.
