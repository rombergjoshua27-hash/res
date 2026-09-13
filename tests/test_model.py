"""Probability conversions, de-vigging and stake sizing."""

from __future__ import annotations

import numpy as np
import pytest

from nflpredict import config
from nflpredict.edge import (
    american_to_decimal,
    confidence_tier,
    expected_value,
    kelly_fraction,
)
from nflpredict.model import devig_moneyline, prob_to_spread, spread_to_prob


def test_pick_em_spread_is_a_coin_flip():
    assert spread_to_prob(0.0) == pytest.approx(0.5)


def test_spread_and_probability_round_trip():
    for spread in (-14.0, -3.5, 0.0, 2.5, 10.0):
        assert prob_to_spread(spread_to_prob(spread)) == pytest.approx(spread, abs=1e-6)


def test_favourites_are_more_likely_to_win_than_underdogs():
    assert spread_to_prob(7.0) > spread_to_prob(0.0) > spread_to_prob(-7.0)


def test_devig_removes_the_house_margin():
    """A de-vigged two-way market sums to exactly one."""
    home = devig_moneyline(-200, 170)
    away = devig_moneyline(170, -200)
    assert home + away == pytest.approx(1.0)
    assert 0.6 < home < 0.68


def test_devig_of_a_symmetric_market_is_even():
    assert devig_moneyline(-110, -110) == pytest.approx(0.5)


def test_devig_returns_none_without_both_prices():
    assert devig_moneyline(np.nan, -110) is None
    assert devig_moneyline(-110, np.nan) is None


def test_american_odds_conversion():
    assert american_to_decimal(100) == pytest.approx(1.0)
    assert american_to_decimal(-110) == pytest.approx(100 / 110)
    assert american_to_decimal(150) == pytest.approx(1.5)


def test_break_even_probability_at_standard_juice():
    """-110 needs 52.38% to break even; that is the bar any edge must clear."""
    breakeven = 1.0 / (1.0 + american_to_decimal(config.STANDARD_VIG_ODDS))
    assert breakeven == pytest.approx(0.5238, abs=1e-4)
    assert expected_value(breakeven, config.STANDARD_VIG_ODDS) == pytest.approx(0, abs=1e-9)


def test_expected_value_sign_follows_the_edge():
    assert expected_value(0.60, -110) > 0
    assert expected_value(0.50, -110) < 0


def test_kelly_never_stakes_on_a_losing_proposition():
    assert kelly_fraction(0.45, -110) == 0.0
    assert kelly_fraction(0.5238, -110) == pytest.approx(0.0, abs=1e-4)


def test_kelly_grows_with_the_edge_and_respects_the_fraction():
    small, large = kelly_fraction(0.56, -110), kelly_fraction(0.70, -110)
    assert 0 < small < large < 1
    quarter = kelly_fraction(0.70, -110, fraction=0.25)
    full = kelly_fraction(0.70, -110, fraction=1.0)
    assert quarter == pytest.approx(full * 0.25)


def test_confidence_tiers_are_ordered_and_symmetric():
    assert confidence_tier(0.95) == "HIGH"
    assert confidence_tier(0.51) == "COINFLIP"
    for prob in (0.2, 0.35, 0.62, 0.88):
        assert confidence_tier(prob) == confidence_tier(1 - prob)
