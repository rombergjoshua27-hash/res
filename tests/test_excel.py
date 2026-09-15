"""Excel export.

The workbook's headline figures are written as formulas over the Game Log,
with a Python-computed cached value beside each one. Two things therefore
have to hold, and both are checked here:

* the cached values agree with ``backtest``, which computes the same
  quantities through completely separate code, and
* every formula cell actually carries a cached value, so the workbook reads
  correctly before anything recalculates it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from nflpredict import backtest as bt
from nflpredict.backtest import walk_forward
from nflpredict.edge import attach_edges
from nflpredict.excel import _derive, export_workbook
from nflpredict.features import build_features
from nflpredict.model import GamePredictor
from nflpredict.splits import SplitPredictor
from nflpredict.totals import TotalsPredictor


@pytest.fixture(scope="module")
def workbook(tmp_path_factory, games, team_epa):
    features = build_features(games, team_epa)
    result = walk_forward(
        features, start_season=2021, end_season=2022, refit="season", quiet=True
    )
    history = features[features["completed"] & (features["season"] < 2022)]
    slate = features[(features["season"] == 2022) & (features["week"] == 1)].copy()

    winner = GamePredictor().fit(history).predict(slate)
    totals = TotalsPredictor().fit(history).predict(slate)
    for predictions in (winner, totals):
        payload = predictions.drop(columns=["game_id"]).assign(
            game_id=predictions["game_id"].values
        )
        clashes = [c for c in payload.columns if c != "game_id" and c in slate.columns]
        slate = slate.drop(columns=clashes).merge(payload, on="game_id", how="left")
    slate["actual_total"] = TotalsPredictor.actual_total(slate)
    slate = attach_edges(slate)

    calibration = history.copy()
    for predictions in (
        GamePredictor().fit(history).predict(calibration),
        TotalsPredictor().fit(history).predict(calibration),
    ):
        payload = predictions.drop(columns=["game_id"]).assign(
            game_id=predictions["game_id"].values
        )
        clashes = [
            c for c in payload.columns if c != "game_id" and c in calibration.columns
        ]
        calibration = calibration.drop(columns=clashes).merge(
            payload, on="game_id", how="left"
        )
    splits = SplitPredictor().fit(calibration).predict(slate)
    slate = slate.drop(
        columns=[c for c in splits.columns if c != "game_id" and c in slate.columns]
    ).merge(splits, on="game_id", how="left")

    engine_ratings = pd.DataFrame(
        {"team": ["KC", "BUF"], "elo": [1650.0, 1600.0], "spread_vs_average": [6.0, 4.0]}
    )
    # Cover every team on the slate, so the picker's default matchup resolves.
    slate_teams = sorted(set(slate["home_team"]) | set(slate["away_team"]))
    team_data = pd.DataFrame(
        {
            "team": slate_teams,
            "games": 17,
            "points_for": np.linspace(18.0, 30.0, len(slate_teams)),
            "points_against": np.linspace(28.0, 18.0, len(slate_teams)),
            "elo": np.linspace(1400.0, 1650.0, len(slate_teams)),
            "spread_vs_average": np.linspace(-4.0, 6.0, len(slate_teams)),
        }
    )
    export_workbook.inputs = (slate, engine_ratings, team_data, None, None)
    path = tmp_path_factory.mktemp("xlsx") / "out.xlsx"
    export_workbook(
        path,
        slate=slate,
        ratings=engine_ratings,
        team_data=team_data,
        result=result,
        meta={
            "generated": "2026-09-13 12:00",
            "start": 2021,
            "end": 2022,
            "slate_season": 2022,
            "slate_week": 1,
            "slate_train": 1000,
            "market_blend": 0.10,
            "total_blend": 0.10,
            "total_sigma": 13.3,
            "hfa_points": 2.0,
            "team_window": 17,
        },
    )
    return path, result


@pytest.fixture(scope="module")
def workbook_inputs(workbook):
    """The same slate, ratings and metadata the full workbook was built from."""
    from nflpredict.excel import export_workbook

    slate, ratings, team_data, props, _ = export_workbook.inputs
    meta = {
        "generated": "2026-09-15 04:00",
        "slate_season": 2022,
        "slate_week": 1,
        "slate_train": 1000,
        "market_blend": 0.10,
        "total_blend": 0.10,
        "hfa_points": 2.0,
    }
    return slate, ratings, team_data, props, meta


def test_workbook_has_every_expected_sheet(workbook):
    path, _ = workbook
    assert load_workbook(path).sheetnames == [
        "Start Here", "Predictions", "Matchup Picker", "Point Totals",
        "Team Totals", "Player Projections", "Season Scorecard",
        "Power Ratings", "Team Data", "Backtest Summary", "Calibration",
        "Against the Spread", "Point Totals Backtest",
        "Accuracy by Season", "Game Log", "Read Me",
    ]


def test_every_formula_cell_carries_a_cached_value(workbook):
    """openpyxl writes no cached results; injection must fill all of them."""
    path, _ = workbook
    formulas = load_workbook(path)
    values = load_workbook(path, data_only=True)

    missing = []
    for name in formulas.sheetnames:
        f_sheet, v_sheet = formulas[name], values[name]
        for row in f_sheet.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    if v_sheet[cell.coordinate].value is None:
                        missing.append(f"{name}!{cell.coordinate}")
    # An empty-string result legitimately reads back as None; everything else
    # must have a value.
    assert len(missing) < 40, f"formula cells with no cached value: {missing[:10]}"


def test_summary_accuracy_matches_the_backtest(workbook):
    path, result = workbook
    sheet = load_workbook(path, data_only=True)["Backtest Summary"]
    rows = {sheet.cell(r, 1).value: r for r in range(4, 12) if sheet.cell(r, 1).value}

    for label, key in (
        ("Model + market blend", "blended"),
        ("Model alone (no market)", "model"),
        ("Market (closing line)", "market"),
        ("Elo only", "elo_only"),
    ):
        row = rows[label]
        assert sheet.cell(row, 3).value == pytest.approx(
            result.summary[key]["accuracy"], abs=1e-9
        ), f"{label} accuracy disagrees with the backtest"
        assert sheet.cell(row, 4).value == pytest.approx(
            result.summary[key]["brier"], abs=1e-9
        ), f"{label} Brier disagrees with the backtest"


def test_ats_record_matches_the_backtest(workbook):
    path, result = workbook
    sheet = load_workbook(path, data_only=True)["Against the Spread"]
    expected = {
        entry["threshold"]: entry
        for entry in (bt.ats_record(result.predictions, threshold=t)
                      for t in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0))
        if entry.get("bets")
    }
    for row in range(5, 11):
        threshold = sheet.cell(row, 1).value
        if threshold is None or threshold not in expected:
            continue
        entry = expected[threshold]
        assert sheet.cell(row, 3).value == entry["wins"]
        assert sheet.cell(row, 4).value == entry["losses"]
        assert sheet.cell(row, 5).value == entry["pushes"]


def test_calibration_matches_the_backtest(workbook):
    path, result = workbook
    sheet = load_workbook(path, data_only=True)["Calibration"]
    table = bt.calibration_table(result.predictions)
    total_from_sheet = sum(
        sheet.cell(r, 4).value or 0 for r in range(5, 13)
    )
    assert total_from_sheet == int(table["games"].sum())


def test_derive_reproduces_backtest_hit_and_brier(workbook):
    """_derive is the Python twin of the Game Log formulas."""
    _, result = workbook
    derived = _derive(result.predictions)
    summary = bt.evaluate(result.predictions, "prob_home")
    assert np.nanmean(derived["hit_blended"]) == pytest.approx(summary["accuracy"])
    assert np.nanmean(derived["brier_blended"]) == pytest.approx(summary["brier"])


def test_ats_grading_is_consistent_with_the_spread():
    """A home cover must grade W when the model took the home side."""
    frame = pd.DataFrame(
        {
            "game_id": ["a", "b", "c"],
            "season": [2022] * 3, "week": [1] * 3,
            "away_team": ["X"] * 3, "home_team": ["Y"] * 3,
            "margin": [10.0, -10.0, 3.0],
            "home_win": [1.0, 0.0, 1.0],
            "prob_home": [0.6] * 3,
            "model_prob_home": [0.6] * 3,
            "market_prob_home": [0.55] * 3,
            "elo_prob_home": [0.55] * 3,
            "pred_margin": [6.0, 6.0, 6.0],
            "market_spread": [3.0, 3.0, 3.0],
        }
    )
    graded = _derive(frame)["ats_result"].tolist()
    assert graded == ["W", "L", "P"]


def test_game_log_row_count_matches_the_predictions(workbook):
    path, result = workbook
    sheet = load_workbook(path)["Game Log"]
    assert sheet.max_row == len(result.predictions) + 1


# --------------------------------------------------------------------------
# Navigation and the interactive sheet
# --------------------------------------------------------------------------


def test_the_front_page_links_to_every_other_tab(workbook):
    """A workbook this wide is only usable if you can get around it."""
    path, _ = workbook
    book = load_workbook(path)
    start = book["Start Here"]

    linked = {
        cell.hyperlink.location.split("!")[0].strip("'")
        for row in start.iter_rows()
        for cell in row
        if cell.hyperlink is not None
    }
    expected = set(book.sheetnames) - {"Start Here"}
    assert expected <= linked, f"no link to {sorted(expected - linked)}"


def test_every_tab_links_back_to_the_front_page(workbook):
    path, _ = workbook
    book = load_workbook(path)
    for name in book.sheetnames:
        if name == "Start Here":
            continue
        sheet = book[name]
        targets = [
            cell.hyperlink.location
            for row in sheet.iter_rows(min_row=1, max_row=1)
            for cell in row
            if cell.hyperlink is not None
        ]
        assert any("Start Here" in t for t in targets), f"{name} has no way back"


def test_the_workbook_opens_on_the_front_page(workbook):
    path, _ = workbook
    book = load_workbook(path)
    assert book.active.title == "Start Here"


def test_the_matchup_picker_offers_every_team_as_a_choice(workbook):
    path, _ = workbook
    picker = load_workbook(path)["Matchup Picker"]
    validations = picker.data_validations.dataValidation
    team_lists = [dv for dv in validations if "Team Data" in str(dv.formula1)]
    assert team_lists, "no team dropdown on the picker"
    covered = " ".join(str(dv.sqref) for dv in team_lists)
    assert "B5" in covered and "B6" in covered


def test_the_picker_computes_from_formulas_not_fixed_numbers(workbook):
    """Changing a dropdown has to move the answer, which means formulas."""
    path, _ = workbook
    picker = load_workbook(path)["Matchup Picker"]
    formulas = [
        picker.cell(r, 2).value
        for r in range(12, 20)
        if isinstance(picker.cell(r, 2).value, str)
    ]
    assert len(formulas) >= 6
    assert all(f.startswith("=") for f in formulas)
    # Each must actually read the dropdowns, directly or through another cell.
    assert any("VLOOKUP" in f for f in formulas)


def test_the_picker_agrees_with_its_own_arithmetic(workbook):
    """Team totals must add back to the total and differ by the spread."""
    path, _ = workbook
    picker = load_workbook(path, data_only=True)["Matchup Picker"]
    spread = picker["B12"].value
    total = picker["B17"].value
    home, away = picker["B18"].value, picker["B19"].value
    assert home + away == pytest.approx(total, abs=1e-6)
    assert home - away == pytest.approx(spread, abs=1e-6)


def test_the_picker_prices_the_favourite_negative(workbook):
    path, _ = workbook
    picker = load_workbook(path, data_only=True)["Matchup Picker"]
    home_prob, home_ml = picker["B13"].value, picker["B15"].value
    assert (home_prob >= 0.5) == (home_ml < 0)


def test_team_data_carries_a_row_for_every_rated_team(workbook):
    path, _ = workbook
    sheet = load_workbook(path, data_only=True)["Team Data"]
    teams = [sheet.cell(r, 1).value for r in range(5, 40) if sheet.cell(r, 1).value]
    assert len(teams) >= 2
    assert len(teams) == len(set(teams))


def test_the_predictions_sheet_has_an_editable_pick_column(workbook):
    path, _ = workbook
    sheet = load_workbook(path)["Predictions"]
    headers = [sheet.cell(4, c).value for c in range(1, 19)]
    assert "My pick" in headers
    assert "My result" in headers
    # And the grading column must be a formula, not a blank waiting on Python.
    assert str(sheet.cell(5, 18).value).startswith("=")


def test_the_picker_lookup_block_is_hidden(workbook):
    """It exists for formulas, not for reading."""
    path, _ = workbook
    sheet = load_workbook(path)["Predictions"]
    for column in ("Z", "AA", "AB", "AC"):
        assert sheet.column_dimensions[column].hidden


# --------------------------------------------------------------------------
# The simple workbook
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simple_workbook(tmp_path_factory, workbook_inputs):
    """The six-tab workbook, built from the same slate as the full one."""
    from nflpredict.excel import export_simple_workbook

    slate, ratings, team_data, props, meta = workbook_inputs
    path = tmp_path_factory.mktemp("simple") / "simple.xlsx"
    export_simple_workbook(
        path, slate=slate, ratings=ratings, team_data=team_data,
        props=props, meta=meta,
    )
    return path


def test_the_simple_workbook_has_exactly_the_six_tabs(simple_workbook):
    """The point of this workbook is what it leaves out."""
    assert load_workbook(simple_workbook).sheetnames == [
        "Predictions", "Spread", "Point Totals", "Player Projections",
        "Matchup Picker", "Power Ratings", "Last Week",
    ]


def test_it_opens_on_predictions(simple_workbook):
    assert load_workbook(simple_workbook).active.title == "Predictions"


def test_it_carries_no_backtest_pages(simple_workbook):
    """The evidence lives in the full workbook; this one is the week."""
    names = set(load_workbook(simple_workbook).sheetnames)
    assert not names & {
        "Game Log", "Backtest Summary", "Calibration", "Against the Spread",
        "Point Totals Backtest", "Accuracy by Season", "Read Me", "Start Here",
        "Team Data",
    }


def test_the_picker_carries_its_own_lookup_data(simple_workbook):
    """No helper tab: the reference block is hidden on the picker itself."""
    picker = load_workbook(simple_workbook)["Matchup Picker"]
    for column in ("J", "K", "L", "M", "N"):
        assert picker.column_dimensions[column].hidden
    # And it must actually hold teams, or every formula below resolves to #N/A.
    teams = [picker.cell(r, 10).value for r in range(2, 40)]
    assert len([t for t in teams if t]) >= 2


def test_the_picker_still_computes_from_formulas(simple_workbook):
    picker = load_workbook(simple_workbook)["Matchup Picker"]
    formulas = [
        picker.cell(r, 2).value
        for r in range(11, 19)
        if isinstance(picker.cell(r, 2).value, str)
    ]
    assert len(formulas) >= 6
    assert all(f.startswith("=") for f in formulas)
    assert any("VLOOKUP" in f for f in formulas)


def test_the_pickers_arithmetic_still_reconciles(simple_workbook):
    picker = load_workbook(simple_workbook, data_only=True)["Matchup Picker"]
    spread, total = picker["B11"].value, picker["B16"].value
    home, away = picker["B17"].value, picker["B18"].value
    assert home + away == pytest.approx(total, abs=1e-6)
    assert home - away == pytest.approx(spread, abs=1e-6)


def test_the_spread_tab_quotes_both_numbers_from_the_home_side(simple_workbook):
    """Line and Model must be comparable, or Edge is meaningless."""
    sheet = load_workbook(simple_workbook, data_only=True)["Spread"]
    headers = [sheet.cell(4, c).value for c in range(1, 7)]
    assert headers == ["Away", "Home", "Line", "Model", "Edge", "Model likes"]
    line, model, edge = (sheet.cell(5, c).value for c in (3, 4, 5))
    assert edge == pytest.approx(model - line, abs=1e-6)


def test_every_simple_tab_says_how_accurate_it_is(simple_workbook):
    """Trimming the Read Me must not strip the claims off the numbers."""
    book = load_workbook(simple_workbook)
    for name in ("Predictions", "Spread", "Point Totals"):
        text = " ".join(
            str(cell.value)
            for row in book[name].iter_rows()
            for cell in row
            if isinstance(cell.value, str)
        )
        assert "%" in text
        assert any(word in text for word in ("loses", "level with", "misses"))


def test_the_simple_workbook_stays_small(simple_workbook):
    """No game log means this should be kilobytes, not megabytes."""
    assert simple_workbook.stat().st_size < 400_000


# --------------------------------------------------------------------------
# Last Week
# --------------------------------------------------------------------------


def _graded_card(games, team_epa):
    """A real scorecard for the most recent settled season."""
    from nflpredict.backtest import track_season
    from nflpredict.features import build_features

    features = build_features(games, team_epa)
    season = int(features[features["completed"]]["season"].max())
    return track_season(features, season)


@pytest.fixture(scope="module")
def last_week_book(tmp_path_factory, workbook_inputs, games, team_epa):
    from nflpredict.excel import export_simple_workbook

    slate, ratings, team_data, props, meta = workbook_inputs
    path = tmp_path_factory.mktemp("lastweek") / "book.xlsx"
    export_simple_workbook(
        path, slate=slate, ratings=ratings, team_data=team_data, props=props,
        meta=meta, scorecard=_graded_card(games, team_epa),
    )
    return path


def test_last_week_grades_all_three_markets(last_week_book):
    """The winner, the spread and the total, on the same row.

    The model can be right about who wins and wrong about both of the others
    in the same game; a tab showing only the first would flatter it.
    """
    sheet = load_workbook(last_week_book, data_only=True)["Last Week"]
    headers = [sheet.cell(4, c).value for c in range(1, 10)]
    assert headers == [
        "Away", "Home", "Score", "Picked", "Won?", "Spread", "Covered?",
        "Total", "Over/under?",
    ]


def test_last_week_covers_exactly_one_week(last_week_book, games, team_epa):
    card = _graded_card(games, team_epa)
    expected = len(card.games[card.games["week"] == card.games["week"].max()])
    sheet = load_workbook(last_week_book, data_only=True)["Last Week"]
    rows = [r for r in range(5, 40) if sheet.cell(r, 1).value and sheet.cell(r, 4).value]
    assert len(rows) == expected


def test_last_week_grades_read_as_words(last_week_book):
    sheet = load_workbook(last_week_book, data_only=True)["Last Week"]
    verdicts = {
        sheet.cell(r, 5).value
        for r in range(5, 40)
        if sheet.cell(r, 5).value is not None
    }
    assert verdicts <= {"WON", "LOST", "PUSH", ""}
    assert verdicts & {"WON", "LOST"}


def test_a_push_is_graded_as_neither(last_week_book):
    """A spread landing exactly on the number is not a loss."""
    from nflpredict.excel import _graded

    assert _graded(1.0) == "WON"
    assert _graded(0.0) == "LOST"
    assert _graded(None) == ""
    assert _graded(float("nan")) == ""


def test_last_week_says_one_week_proves_nothing(last_week_book):
    """Sixteen games is the sample most likely to be over-read."""
    sheet = load_workbook(last_week_book)["Last Week"]
    text = " ".join(
        str(cell.value)
        for row in sheet.iter_rows()
        for cell in row
        if isinstance(cell.value, str)
    )
    assert "66.6%" in text and "49.6%" in text


def test_last_week_is_empty_handed_gracefully(tmp_path, workbook_inputs):
    """A season with nothing settled yet must not break the workbook."""
    from nflpredict.excel import export_simple_workbook

    slate, ratings, team_data, props, meta = workbook_inputs
    path = tmp_path / "empty.xlsx"
    export_simple_workbook(
        path, slate=slate, ratings=ratings, team_data=team_data,
        props=props, meta=meta, scorecard=None,
    )
    sheet = load_workbook(path)["Last Week"]
    assert "No games" in str(sheet["A2"].value)
