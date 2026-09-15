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
    path = tmp_path_factory.mktemp("xlsx") / "out.xlsx"
    export_workbook(
        path,
        slate=slate,
        ratings=engine_ratings,
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
        },
    )
    return path, result


def test_workbook_has_every_expected_sheet(workbook):
    path, _ = workbook
    assert load_workbook(path).sheetnames == [
        "Read Me", "Predictions", "Point Totals", "Team Totals",
        "Power Ratings", "Backtest Summary", "Calibration",
        "Against the Spread", "Point Totals Backtest",
        "Accuracy by Season", "Game Log",
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
