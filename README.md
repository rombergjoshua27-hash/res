# nflpredict

An NFL forecasting model covering **moneylines, spreads, point totals, team
totals, half lines and player props** — with a walk-forward backtest attached
to every one of them, so each accuracy number it reports can be checked
rather than believed.

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

### Team totals and half lines

Derived from the game total and the spread rather than modelled separately —
a team total is not free to disagree with them.

| Projection | Games | Model MAE | Naive league average |
|---|---:|---:|---:|
| Home team total | 4,912 | **7.494** | 8.130 |
| Away team total | 4,912 | **7.301** | 8.130 |
| First-half total | 4,912 | **7.027** | 7.216 |
| First-half margin | 4,912 | 8.350 | — |
| Second-half total | 4,912 | 7.654 | — |

Team totals carry real information. The first-half total barely beats
guessing the league average, and the first-half side is picked correctly
**60.7%** of the time against 66.6% for the full game. First halves are
mostly noise, and the tool says so next to the number rather than in a
footnote.

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

### Player props

Passing, rushing and receiving yards per player. Measured for players with
real involvement, against the two baselines that matter:

| Projection | Games | Model | Player's own average | League average |
|---|---:|---:|---:|---:|
| Passing yards (10+ att) | 7,940 | **61.87** | 68.40 | 64.25 |
| Rushing yards (5+ carries) | 14,599 | **24.28** | 26.06 | 27.01 |
| Receiving yards (3+ targets) | 37,076 | **22.90** | 24.84 | 26.06 |

A clear gain over a player's own recent form on all three, and over the
positional average on rushing and receiving. On passing yards the league
average is nearly as good, because starting quarterbacks cluster tightly —
worth saying rather than burying. And a 60-yard miss on a passing projection
is a wide miss in absolute terms, whatever it beats.

**No over/under probability is offered for props.** Yardage is long-tailed,
and a normal curve over it would understate how often a projection is badly
wrong. A confident-looking percentage is the easiest thing here to fake and
the hardest to justify, so there isn't one.

## Install

```bash
git clone <this repo> && cd res
pip install -e .
```

Python 3.10+, tested on 3.10 and 3.13 in CI. The first run downloads ~400 MB of play-by-play and distills it
to ~60 KB per season of cached summaries; subsequent runs are instant. Pass
`--epa-start 2018` to start with a much smaller download.

## Usage

```bash
nflpredict predict              # the upcoming slate, every market + a workbook
nflpredict predict --season 2026 --week 5
nflpredict predict --no-excel   # terminal only, skip the workbook
nflpredict predict --csv        # also write out/predictions_*.csv

nflpredict track                # how this season's picks have actually landed

nflpredict export               # everything incl. the full backtest, as xlsx
nflpredict export -o ~/week2.xlsx

nflpredict backtest             # reproduce the tables above
nflpredict backtest --no-market # the models on their own merits
nflpredict backtest --refit week

nflpredict ratings --top 10     # current Elo power ratings
nflpredict evaluate --season 2025
nflpredict update               # refresh cached data
```

As a library:

```python
from nflpredict import load_games, load_team_game_epa, build_features
from nflpredict import GamePredictor, TotalsPredictor
from nflpredict.players import load_injuries, load_player_weeks

seasons = range(2006, 2027)
games = load_games()
features = build_features(
    games,
    load_team_game_epa(seasons),
    player_weeks=load_player_weeks(seasons),
    injuries=load_injuries(seasons),
)

history = features[features.completed & (features.season < 2026)]
slate = features[(features.season == 2026) & (features.week == 5)]

GamePredictor().fit(history).predict(slate)    # winner + spread
TotalsPredictor().fit(history).predict(slate)  # point total + over/under
```

`SplitPredictor` derives team totals and half lines from that pair, and
`PropsPredictor` projects player yardage. `track_season(features, 2026)`
returns the live scorecard.

## The Excel workbook

`nflpredict predict` writes **eight tabs and nothing else** — the week, in the
order you would look at it:

| Tab | What it holds |
|---|---|
| **Predictions** | Who wins, how sure, fair odds against the book's, and a column for your own picks |
| **Spread** | The model's number against the posted line, and the gap |
| **Point Totals** | How many points, and which side of the total it leans |
| **Player Projections** | Projected passing, rushing and receiving yards |
| **Matchup Picker** | Two dropdowns — pick any teams, everything recalculates |
| **Betting Picks** | Every pick ranked within its market, with what that confidence has actually been worth |
| **Power Ratings** | Every team, strongest first |
| **Last Week** | How the last slate's picks actually turned out, graded on all three markets |

About 20 KB, a few seconds to build. Each tab carries its own one-line
accuracy note at the foot, so no claim travels without the number it applies
to. **Yellow cells are yours to edit**; everything else is a formula.

### A note on the Betting Picks tab

Straight-up picks carry the model's own probability, because that is the one
number here that has been checked and holds up — a stated 70% has won 76%
and a stated 80% has won 87% across 4,912 games.

Spread and total picks are ranked by **Edge**: how far the model's own
unblended number sits from the posted line. An earlier version rated every
one of them at the global base rate, so sixteen spreads all read 49.6% and
there was nothing to choose between them. That was useless for a fixable
reason — the rating was built off the *blended* forecast, which is
deliberately pulled onto the market and so disagrees with it by at most four
points. The unblended model disagrees by up to twenty-one, and that spread of
disagreement is a real per-game quantity worth ranking on.

What the ranking does not claim is that it beats the price. Fitting win rate
on edge gives a positive slope for both markets and a confidence interval
that contains zero for both:

| Market | Slope per point of edge | 95% CI | n |
|---|---|---|---|
| Totals | +0.0288 | −0.0012 … +0.0563 | 4,853 |
| Spreads | +0.0086 | −0.0186 … +0.0369 | 4,788 |

Twenty cuts of the data — early weeks, late weeks, big totals, small totals,
home side, away side, each era — were checked and **not one clears the 52.38%
needed at -110** with any confidence. The largest edges are suggestive: 6+
points has won 53.8% on totals and 54.0% on spreads, but on 263–277 bets,
where the margin for error is about three points of win rate either way.

So `Win %` is fitted and then **capped at break-even**. Uncapped the curve
quotes 56% on a big edge, which would have the tab advertising profitable
bets on the strength of a relationship it cannot demonstrate — it did exactly
that for one commit, and the test forbidding a positive expected value caught
it. The measured band rate is still printed raw beside each pick, sample size
and all, so where the realised rate runs past that ceiling you can see it and
see what it rests on.

The expected-value column is therefore negative on essentially every row.
That is the vig; it is the measured result rather than a disclaimer, and it
is why a model that ties the closing line is a good forecast and still not a
profitable bet.

`nflpredict export`, or `predict --full`, writes the long version instead:
the same six plus Start Here, Team Totals, Season Scorecard, Team Data,
Backtest Summary, Calibration, Against the Spread, Point Totals Backtest,
Accuracy by Season, Game Log and Read Me. That one is for checking the
model's homework; the short one is for reading the week.

### The Matchup Picker

Choose any two teams from the dropdowns and the whole panel recalculates —
spread, both moneylines, total, both team totals — as Excel formulas. No
regeneration, no Python. In the short workbook the figures it reads live in
hidden columns on the picker itself, because a lookup table is not something
anyone wants to look at and six tabs should not spend one on plumbing.

Where the chosen pair is actually on this week's slate, the sheet also shows
what the full model says, **because the two will not agree**. This week it
has MIA @ SF at 9.5 by the picker and 14.7 by the model: the picker is Elo
plus raw scoring averages, which is all that fits in a spreadsheet formula,
and it cannot see that Miami is starting a backup. The picker is a quick
what-if for matchups nobody has priced; the Predictions tab is the forecast.

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

**The first quarterback adjustment made it worse.** The intuition is sound —
a backup starting is worth several points — but that implementation rated a
passer by his team's offensive EPA, which is mostly a property of the team,
and the team's strength was already in its Elo rating. It double-counted,
then folded the result into Elo at a hand-picked scale:

| Setting | Games | Accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|
| No Elo QB term (default) | 4,912 | **65.1%** | **0.2186** | **0.6281** |
| Elo QB term on | 4,912 | 64.1% | 0.2206 | 0.6325 |

It degrades the Elo rating itself as much as the ensemble — Elo alone falls
from 65.1% to 63.3% with the term switched on. A sweep across scales
(15/25/45 Elo per EPA point, caps of 3-6 points) was worse at every setting.
It stays reachable via `--qb-adjustment` so the experiment is reproducible.

The [second attempt](#quarterbacks-and-injuries) works, and the difference
between them is the whole lesson: rate the passer with measures that isolate
him, feed the model the *change* rather than the level, and let it fit the
weight instead of choosing one.

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

**Recency weighting for the passer rating was not tried, because recency
weighting for totals already failed.** The same reasoning applies: centering
features on a prior-seasons baseline absorbs the drift, and past that point
more data beats fresher data.

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

## Quarterbacks and injuries

The model knows who is starting and who is hurt. Three things make this
version work where the [first attempt](#what-didnt-work) failed:

1. **Passer-isolating inputs.** Completion percentage over expected is
   computed per throw against the difficulty of that throw, and EPA is taken
   per *dropback* — sacks counted as the failed dropbacks they are — rather
   than per team play. Neither is a restatement of team strength.
2. **A difference, not a level.** The feature is not "how good is this
   quarterback", which Elo roughly knows already. It is how far this week's
   starter sits from the one the team has been playing. Same starter reads
   ≈0, so there is nothing to double-count.
3. **A learned weight.** The terms enter the feature matrix and the model
   fits their coefficients alongside `elo_diff`, instead of being told how
   many points a quarterback is worth.

Availability works the same way: the injury report becomes the share of a
team's recent receiving and rushing workload expected to be missing, and the
model decides what that is worth. The play rates behind it are measured, not
assumed — joining each week's report to the box score that followed, over
73,243 report rows from 2009-2026:

| Listed status | Rows | Plays (normalised) |
|---|---:|---:|
| Out | 16,098 | 0.00 |
| Doubtful | 3,385 | 0.01 |
| Questionable | 23,574 | 0.69 |

"Doubtful" really does mean it. "Questionable" is the only status where the
number does real work.

**What it buys:**

| | Accuracy | Brier | Log loss |
|---|---:|---:|---:|
| Market off, base | 65.07% | 0.2186 | 0.6281 |
| Market off, **+ QB/injury** | 64.98% | **0.2171** | **0.6247** |
| Market on, base | 66.61% | 0.2099 | 0.6087 |
| Market on, + QB/injury | 66.59% | 0.2099 | 0.6085 |

Unchanged accuracy, significantly better probabilities market-free
(p=0.002 on a paired test of per-game Brier; on the 280 games where the two
disagreed on the pick it was 138-142, a dead heat).

**With the market on it is a wash** — 13 of 4,912 picks moved. The closing
line already prices a backup quarterback and a Friday injury report. It is
on by default because the market-free path is the one that matters when no
line is posted; `--no-qb-features` turns it off.

## Keeping it current

```bash
nflpredict refresh          # pull the latest data, rebuild the workbook
```

Only the current season is re-downloaded. Scores, closing lines, listed
quarterbacks and the injury report all move during a week; seasons that have
finished do not, and re-fetching twenty of them daily would be twenty minutes
of downloading to discover nothing had happened.

**It also runs itself**, so week to week there is nothing to do.

`.github/workflows/refresh.yml` rebuilds the workbook every morning at 11:00
UTC, plus a second run Sundays at 15:00 UTC (11am Eastern, roughly two hours
before the early slate) to pick up the weekend's line movement and fold in
Thursday's result before kickoff. Note that the injury layer reads the weekly
injury report, which is final on Friday — the Sunday run buys a fresher
market, not gameday inactives. Both publish two ways:

- **[nflpredict-latest.xlsx](../../releases/download/latest/nflpredict-latest.xlsx)**
  on the `latest` release — a fixed filename, so this link always serves the
  newest build and is safe to bookmark.
- An artifact on each run, named for the week it covers, kept 90 days.

To pull a week early, or rebuild a specific one: Actions → *Refresh workbook*
→ *Run workflow*, optionally with a season and week. Locally, `nflpredict
refresh` does the same thing.

### What changes through a week

Nothing needs asking for again, because every part of the week arrives on its
own schedule:

| When | What the rebuild picks up |
|---|---|
| Mon–Tue | Last week's results. The Season Scorecard grades every pick, and the slate rolls forward once a week is mostly played. |
| Wed–Fri | Injury reports publish, so the availability term stops reading "nobody is hurt" and player projections get scaled by who is actually expected to play. |
| Sat–Sun | Lines and listed starters firm up, which is most of what moves a number. |

The backtest grows with the season too: each finished game becomes another
row behind the accuracy figures, and the walk-forward cache notices and
recomputes.

The walk-forward backtest is cached and keyed on everything that could change
its answer: completed games, seasons, blend weights, and whether the passer
terms were fitted. Only the per-game rows are stored; every summary is
recomputed on load, so a change to how a metric is defined can never be
masked by a stale number.

## Tracking the season

```bash
nflpredict track
```

Every week is graded using only the games that had finished before that
slate kicked off — the same call the tool would have made on the morning of,
not a tidied-up version of it. Nothing is stored between runs: the scorecard
is recomputed from the game log each time, so it cannot drift out of step
with a corrected result the way a running tally would.

The output carries its own health warning, because a season is a small
sample and a hot start is not evidence the model improved.

## Data

Everything comes from [nflverse](https://github.com/nflverse), free and
public, no API key:

- **Game log** — `nflverse/nfldata`: scores, closing spreads, totals and
  moneylines, rest days, venue, roof, temperature, wind, starters, back to
  1999.
- **Play-by-play** — `nflverse/nflverse-data`: ~50k plays per season with
  EPA, win probability, drive identifiers and success flags. Also the source
  of halftime scores.
- **Player box scores** — `nflverse-data/stats_player`: per player per game
  since 1999, with CPOE, EPA, target share and air-yards share.
- **Injury reports** — `nflverse-data/injuries`: the weekly report since
  2009, with game status and practice participation.

Snap counts are deliberately *not* used: nflverse keys them by Pro Football
Reference id rather than the gsis id everything else uses, so joining them
means matching on names — fragile, and unnecessary when `target_share` and
`wopr` already measure usage on a clean key.

Cached under `data/` (gitignored). `nflpredict update` refreshes.

## Development

```bash
pip install -e . && pip install pytest
pytest -q        # 177 tests
```

Layout: `data.py` fetch/cache · `players.py` player box scores and injuries ·
`elo.py` ratings · `features.py` leak-free matrix · `model.py` winner/spread
ensemble + blend · `totals.py` point totals · `qb.py` passer value and
availability · `splits.py` team totals and half lines · `props.py` player
projections · `backtest.py` walk-forward, metrics and season tracking ·
`edge.py` EV, fair odds and Kelly · `report.py` formatting · `excel.py`
workbook · `cache.py` backtest cache · `cli.py` commands.

## Using this responsibly

It is a forecasting and analysis tool: power ratings, calibrated
probabilities, point and player projections, and an honest measure of how
much of the game is actually predictable. That is what it is good for.

If anyone points it at a sportsbook anyway, the backtest is unambiguous — it
loses at -110 on sides *and* on totals, the first-half numbers are weaker
still, and the player projections carry no probability at all because the
distributions do not support one. The `edge.py` Kelly sizing is
quarter-Kelly precisely because a Brier score of 0.21 means the
probabilities are good but not sharp enough to bet aggressively on. The
`EV / $1` column in the workbook is negative on every game of a typical
slate, and that is not a bug in the model — it is the vig, shown rather than
hidden. A model that ties the closing line is a genuinely good model and
still not a profitable one, because the price already contains everything it
knows.
