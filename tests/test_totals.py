"""Point-total model: probability conversion, blending and leak-free fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflpredict import config
from nflpredict.backtest import evaluate_totals, ou_record
from nflpredict.edge import american_to_prob, prob_to_american
from nflpredict.features import TOTAL_FEATURE_COLUMNS, build_features
from nflpredict.totals import TotalsPredictor, over_probability, push_probability


# --------------------------------------------------------------------------
# Over / under probability
# --------------------------------------------------------------------------


def test_predicting_the_line_exactly_is_a_coin_flip():
    """A half-point line cannot push, so landing on it splits the field."""
    assert over_probability(44.5, 44.5) == pytest.approx(0.5)


def test_over_probability_rises_with_the_predicted_total():
    low = over_probability(38.0, 44.5)
    mid = over_probability(44.5, 44.5)
    high = over_probability(52.0, 44.5)
    assert low < mid < high
    assert 0.0 < low and high < 1.0


def test_whole_number_lines_get_a_continuity_correction():
    """To beat a posted 44 a game must reach 45, so 44.0 is not a coin flip."""
    assert over_probability(44.0, 44.0) < 0.5
    # The same prediction against a half-point line is exactly even, because
    # 44.5 needs 45 too but cannot be pushed.
    assert over_probability(44.5, 44.5) == pytest.approx(0.5)
    assert over_probability(44.0, 44.0) == pytest.approx(
        over_probability(44.0, 44.5), abs=1e-9
    )


def test_only_whole_number_lines_can_push():
    assert push_probability(45.0, 44.5) == 0.0
    assert push_probability(45.0, 44.0) > 0.0


def test_push_probability_is_near_the_observed_rate():
    """~2.9% of games on a whole-number total land exactly on it."""
    modelled = push_probability(45.0, 45.0, config.TOTAL_SIGMA)
    assert modelled == pytest.approx(config.TOTAL_PUSH_RATE_WHOLE_LINES, abs=0.01)


def test_a_wider_spread_of_outcomes_pulls_probabilities_toward_even():
    tight = over_probability(52.0, 44.5, sigma=8.0)
    loose = over_probability(52.0, 44.5, sigma=20.0)
    assert 0.5 < loose < tight


# --------------------------------------------------------------------------
# Fair moneylines
# --------------------------------------------------------------------------


def test_fair_moneyline_round_trips_through_its_probability():
    for prob in (0.80, 0.65, 0.50, 0.35, 0.20):
        assert american_to_prob(prob_to_american(prob)) == pytest.approx(prob)


def test_favourites_are_priced_negative_and_underdogs_positive():
    assert prob_to_american(0.75) < 0
    assert prob_to_american(0.25) > 0
    assert prob_to_american(0.5) == pytest.approx(-100.0)


def test_fair_moneyline_carries_no_vig():
    """Both sides of a fair price imply probabilities summing to exactly one."""
    for prob in (0.55, 0.7, 0.9):
        pair = american_to_prob(prob_to_american(prob)) + american_to_prob(
            prob_to_american(1 - prob)
        )
        assert pair == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Fitting and prediction
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def features(games, team_epa):
    return build_features(games, team_epa)


@pytest.fixture(scope="module")
def split(features):
    completed = features[features["completed"]]
    train = completed[completed["season"] <= 2021]
    test = completed[completed["season"] >= 2022]
    return train, test


def test_every_totals_feature_is_present_and_finite(features):
    for column in TOTAL_FEATURE_COLUMNS:
        assert column in features.columns
        assert np.isfinite(features[column]).all(), f"{column} has non-finite values"


def test_indoor_games_are_marked_calm_and_mild(features):
    indoor = features[features["is_indoor"] == 1.0]
    assert len(indoor) > 0
    assert (indoor["wind_speed"] == 0.0).all()
    assert (indoor["temperature"] == 70.0).all()


def test_fit_then_predict_produces_plausible_totals(split):
    train, test = split
    model = TotalsPredictor(use_market=False).fit(train)
    out = model.predict(test)

    assert len(out) == len(test)
    # Every NFL total on record sits well inside this band.
    assert out["model_total"].between(20.0, 75.0).all()
    assert 9.0 <= model.report.sigma <= 20.0


def test_predict_before_fit_is_an_error(split):
    train, _ = split
    with pytest.raises(RuntimeError):
        TotalsPredictor().predict(train)


def test_a_zero_weight_blend_returns_the_posted_total(split):
    train, test = split
    out = TotalsPredictor(market_blend=0.0).fit(train).predict(test)
    priced = out[out["market_total"].notna()]
    assert len(priced) > 0
    assert priced["pred_total"].to_numpy() == pytest.approx(
        priced["market_total"].to_numpy()
    )


def test_a_full_weight_blend_returns_the_model_total(split):
    train, test = split
    out = TotalsPredictor(market_blend=1.0).fit(train).predict(test)
    assert out["pred_total"].to_numpy() == pytest.approx(out["model_total"].to_numpy())


def test_turning_the_market_off_ignores_the_posted_total(split):
    train, test = split
    out = TotalsPredictor(use_market=False).fit(train).predict(test)
    assert out["pred_total"].to_numpy() == pytest.approx(out["model_total"].to_numpy())


def test_the_pick_follows_the_projection_not_the_model(split):
    """OVER/UNDER is graded against what the blend actually stands behind."""
    train, test = split
    out = TotalsPredictor().fit(train).predict(test)
    priced = out[out["market_total"].notna()]
    over = priced["ou_pick"] == "OVER"
    assert (priced.loc[over, "pred_total"] > priced.loc[over, "market_total"]).all()
    assert (priced.loc[~over, "pred_total"] <= priced.loc[~over, "market_total"]).all()


def test_games_without_a_posted_total_still_get_a_forecast(split):
    train, test = split
    test = test.copy()
    test["market_total"] = np.nan
    out = TotalsPredictor().fit(train).predict(test)
    assert out["model_total"].notna().all()
    assert out["pred_total"].to_numpy() == pytest.approx(out["model_total"].to_numpy())
    assert (out["ou_pick"] == "-").all()


def test_the_model_beats_predicting_the_league_average(split):
    """A model with no signal would do no better than always guessing the mean."""
    train, test = split
    model = TotalsPredictor(use_market=False).fit(train)
    out = model.predict(test)
    actual = TotalsPredictor.actual_total(test).to_numpy()

    model_mae = np.mean(np.abs(actual - out["model_total"].to_numpy()))
    naive_mae = np.mean(np.abs(actual - TotalsPredictor.actual_total(train).mean()))
    assert model_mae < naive_mae


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_evaluate_totals_reports_error_not_accuracy():
    frame = pd.DataFrame(
        {"actual_total": [40.0, 50.0, 45.0], "pred_total": [42.0, 47.0, 45.0]}
    )
    result = evaluate_totals(frame)
    assert result["n"] == 3
    assert result["mae"] == pytest.approx((2 + 3 + 0) / 3)
    assert result["bias"] == pytest.approx((-2 + 3 + 0) / 3)


def test_evaluate_totals_ignores_games_with_no_result():
    frame = pd.DataFrame(
        {"actual_total": [40.0, np.nan], "pred_total": [42.0, 47.0]}
    )
    assert evaluate_totals(frame)["n"] == 1


def test_ou_record_grades_both_sides_and_counts_pushes():
    frame = pd.DataFrame(
        {
            "actual_total": [50.0, 30.0, 44.0],
            "market_total": [44.0, 44.0, 44.0],
            "model_total": [48.0, 40.0, 47.0],  # OVER, UNDER, OVER
        }
    )
    record = ou_record(frame)
    assert record["bets"] == 3
    assert (record["wins"], record["losses"], record["pushes"]) == (2, 0, 1)


def test_ou_record_threshold_filters_small_disagreements():
    frame = pd.DataFrame(
        {
            "actual_total": [50.0, 30.0],
            "market_total": [44.0, 44.0],
            "model_total": [44.5, 40.0],  # edges of 0.5 and 4.0
        }
    )
    assert ou_record(frame, threshold=0.0)["bets"] == 2
    assert ou_record(frame, threshold=2.0)["bets"] == 1
