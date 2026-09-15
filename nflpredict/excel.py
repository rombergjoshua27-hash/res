"""Excel export.

Builds a multi-sheet workbook from the model's outputs. Every summary
figure -- accuracy, Brier, calibration, ATS record, per-season splits -- is
written as a live formula over the Game Log sheet rather than a value
computed in Python, so the workbook recalculates if a row is edited or
filtered and a reader can audit exactly how each number was reached.

openpyxl writes formulas with no cached result, which leaves every formula
cell reading back as blank to anything that does not recalculate first
(pandas, most previewers, and Excel's own initial paint). Normally
LibreOffice fills those in, but it cannot run in every environment. So each
formula is written together with its Python-computed value, and
``_inject_cached_values`` writes those values into the sheet XML as the
cached results -- the same ``<f>`` plus ``<v>`` pairing Excel itself stores.
The workbook is therefore correct when opened *and* still live: the
``fullCalcOnLoad`` flag makes Excel recompute every formula on open, so a
stale cached value cannot survive being looked at.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill
from openpyxl.styles.borders import Side
from openpyxl.worksheet.hyperlink import Hyperlink
from openpyxl.utils import column_index_from_string, get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from . import config

__all__ = ["export_workbook", "export_simple_workbook"]

FONT = "Arial"

_HEADER_FILL = PatternFill("solid", fgColor="1F3864")
_HEADER_FONT = Font(name=FONT, size=10, bold=True, color="FFFFFF")
_TITLE_FONT = Font(name=FONT, size=14, bold=True, color="1F3864")
_SECTION_FONT = Font(name=FONT, size=11, bold=True, color="1F3864")
_NOTE_FONT = Font(name=FONT, size=9, italic=True, color="606060")
_BODY_FONT = Font(name=FONT, size=10)
_BOLD = Font(name=FONT, size=10, bold=True)
_INPUT_FONT = Font(name=FONT, size=10, color="0000FF")
_ASSUMPTION_FILL = PatternFill("solid", fgColor="FFFF00")
_BORDER = Border(bottom=Side(style="thin", color="BFBFBF"))

PCT = "0.0%"
PCT2 = "0.00%"
SPREAD = "+0.0;-0.0;0.0"
NUM4 = "0.0000"

_GL = "'Game Log'"

GAME_LOG_HEADERS = [
    # Winner / spread: columns A-Y
    "game_id", "season", "week", "away_team", "home_team", "margin",
    "home_win", "prob_blended", "prob_model", "prob_market", "prob_elo",
    "pred_margin", "market_spread", "hit_blended", "hit_model", "hit_market",
    "hit_elo", "brier_blended", "brier_model", "brier_market", "brier_elo",
    "confidence", "spread_edge", "ats_result", "abs_edge",
    # Point totals: columns Z-AH
    "actual_total", "market_total", "model_total", "pred_total",
    "err_total_blended", "err_total_model", "err_total_market",
    "ou_edge", "ou_result",
    "signed_total_blended", "signed_total_model", "signed_total_market",
]

# Game Log columns the totals sheets point their formulas at.
TOTAL_COLS = {
    "actual": "Z", "market": "AA", "model": "AB", "blend": "AC",
    "err_blend": "AD", "err_model": "AE", "err_market": "AF",
    "edge": "AG", "result": "AH",
    "signed_blend": "AI", "signed_model": "AJ", "signed_market": "AK",
}


# --------------------------------------------------------------------------
# Formula writing with cached values
# --------------------------------------------------------------------------


class Recorder:
    """Collects the Python-computed result of every formula written."""

    def __init__(self) -> None:
        self.values: Dict[str, Dict[str, Any]] = {}

    def formula(
        self, sheet, row: int, col: int, formula: str, value: Any, fmt: str | None = None
    ):
        """Write ``formula`` to a cell and remember ``value`` as its result."""
        cell = sheet.cell(row=row, column=col, value=formula)
        cell.font = _BODY_FONT
        if fmt:
            cell.number_format = fmt
        coord = f"{get_column_letter(col)}{row}"
        self.values.setdefault(sheet.title, {})[coord] = _clean(value)
        return cell


def _clean(value: Any) -> Any:
    """Normalize a computed value for storage as a cached formula result."""
    if value is None:
        return ""
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return ""
    return value


def _num(value) -> float | None:
    """None (a blank cell) rather than NaN, which Excel renders as an error."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return float(value)


_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _sheet_xml_map(archive: zipfile.ZipFile) -> Dict[str, str]:
    """Map each sheet's display name to its XML part inside the package."""
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))

    targets = {
        rel.get("Id"): rel.get("Target")
        for rel in rels.findall(f"{{{_PKG_REL_NS}}}Relationship")
    }

    mapping = {}
    for sheet in workbook.find(f"{{{_NS}}}sheets"):
        target = targets.get(sheet.get(f"{{{_REL_NS}}}id"), "")
        target = target.lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        mapping[sheet.get("name")] = target
    return mapping


def _inject_cached_values(path: Path, recorder: Recorder) -> int:
    """Write each recorded value into its formula cell as the cached result.

    Excel stores a formula cell as ``<c><f>..</f><v>result</v></c>``. openpyxl
    omits the ``<v>``, so this adds it. Returns the number of cells filled.
    """
    path = Path(path)
    temp = path.with_suffix(".tmp.xlsx")
    filled = 0

    with zipfile.ZipFile(path) as source:
        sheet_parts = _sheet_xml_map(source)
        wanted = {
            sheet_parts[name]: values
            for name, values in recorder.values.items()
            if name in sheet_parts
        }
        entries = source.infolist()
        payloads = {entry.filename: source.read(entry.filename) for entry in entries}

    ET.register_namespace("", _NS)
    for part, values in wanted.items():
        root = ET.fromstring(payloads[part])
        for cell in root.iter(f"{{{_NS}}}c"):
            formula = cell.find(f"{{{_NS}}}f")
            if formula is None:
                continue
            if cell.get("r") not in values:
                continue
            value = values[cell.get("r")]

            for existing in cell.findall(f"{{{_NS}}}v"):
                cell.remove(existing)
            node = ET.SubElement(cell, f"{{{_NS}}}v")
            if isinstance(value, str):
                cell.set("t", "str")
                node.text = value
            else:
                cell.attrib.pop("t", None)
                node.text = repr(float(value)) if isinstance(value, float) else str(value)
            filled += 1
        payloads[part] = ET.tostring(root, encoding="UTF-8", xml_declaration=True)

    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as out:
        for entry in entries:
            out.writestr(entry, payloads[entry.filename])
    shutil.move(str(temp), str(path))
    return filled


# --------------------------------------------------------------------------
# Styling helpers
# --------------------------------------------------------------------------


def _write_header(sheet, row: int, headers: Sequence[str]) -> None:
    for index, label in enumerate(headers, start=1):
        cell = sheet.cell(row=row, column=index, value=label)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = _BORDER


def _write_title(sheet, title: str, subtitle: str | None = None) -> int:
    sheet["A1"] = title
    sheet["A1"].font = _TITLE_FONT
    if subtitle:
        sheet["A2"] = subtitle
        sheet["A2"].font = _NOTE_FONT
        return 4
    return 3


def _set_widths(sheet, widths: Iterable[int]) -> None:
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width


def _note(sheet, row: int, text: str) -> None:
    sheet.cell(row=row, column=1, value=text).font = _NOTE_FONT


def _value(sheet, row: int, col: int, value, fmt: str | None = None, font=_BODY_FONT):
    cell = sheet.cell(row=row, column=col, value=value)
    cell.font = font
    if fmt:
        cell.number_format = fmt
    return cell


def _rng(col: str, last_row: int) -> str:
    return f"{_GL}!${col}$2:${col}${last_row}"


# --------------------------------------------------------------------------
# Derived columns -- the Python side of every Game Log formula
# --------------------------------------------------------------------------

_PROB_SOURCES = [
    ("blended", "prob_home"),
    ("model", "model_prob_home"),
    ("market", "market_prob_home"),
    ("elo", "elo_prob_home"),
]


def _derive(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute exactly what the Game Log formulas compute.

    Each expression here mirrors one Excel formula written by
    ``_build_game_log``; keeping them adjacent is what makes the cached
    values trustworthy.
    """
    out = frame.copy()
    for column in ("market_spread", "market_prob_home", "elo_prob_home"):
        if column not in out.columns:
            out[column] = np.nan

    outcome = out["home_win"]
    for name, prob_column in _PROB_SOURCES:
        prob = out[prob_column]
        missing = outcome.isna() | prob.isna()
        out[f"hit_{name}"] = np.where(
            missing,
            np.nan,
            np.where(outcome == 0.5, 0.5, np.where((prob > 0.5) == (outcome == 1), 1.0, 0.0)),
        )
        out[f"brier_{name}"] = np.where(missing, np.nan, (prob - outcome) ** 2)

    out["confidence"] = np.maximum(out["prob_home"], 1.0 - out["prob_home"])
    out["spread_edge"] = out["pred_margin"] - out["market_spread"]

    cover = out["margin"] - out["market_spread"]
    unplayable = out["market_spread"].isna() | out["margin"].isna()
    home_side = out["spread_edge"] > 0
    graded = np.where(
        home_side,
        np.where(cover > 0, "W", np.where(cover == 0, "P", "L")),
        np.where(cover < 0, "W", np.where(cover == 0, "P", "L")),
    )
    out["ats_result"] = np.where(unplayable, "", graded)
    out["abs_edge"] = out["spread_edge"].abs()

    for column in ("actual_total", "market_total", "model_total", "pred_total"):
        if column not in out.columns:
            out[column] = np.nan

    for name, column in (
        ("blended", "pred_total"), ("model", "model_total"), ("market", "market_total"),
    ):
        signed = out["actual_total"] - out[column]
        out[f"err_total_{name}"] = signed.abs()
        out[f"signed_total_{name}"] = signed

    # Over/under is graded against the model's own number, matching
    # ``backtest.ou_record``: the blend is mostly the posted total, so
    # grading against it would be grading the line against itself.
    out["ou_edge"] = out["model_total"] - out["market_total"]
    diff = out["actual_total"] - out["market_total"]
    took_over = out["ou_edge"] > 0
    ou_graded = np.where(
        took_over,
        np.where(diff > 0, "W", np.where(diff == 0, "P", "L")),
        np.where(diff < 0, "W", np.where(diff == 0, "P", "L")),
    )
    out["ou_result"] = np.where(
        out["market_total"].isna() | out["actual_total"].isna() | out["ou_edge"].isna(),
        "",
        ou_graded,
    )
    return out


def _mean(series) -> float:
    values = pd.Series(series).dropna()
    return float(values.mean()) if len(values) else float("nan")


# --------------------------------------------------------------------------
# Game Log
# --------------------------------------------------------------------------


def _build_game_log(sheet, rec: Recorder, frame: pd.DataFrame) -> int:
    derived = _derive(frame)
    _write_header(sheet, 1, GAME_LOG_HEADERS)

    for offset, row in enumerate(derived.itertuples(index=False)):
        r = offset + 2
        _value(sheet, r, 1, row.game_id)
        _value(sheet, r, 2, int(row.season))
        _value(sheet, r, 3, int(row.week))
        _value(sheet, r, 4, row.away_team)
        _value(sheet, r, 5, row.home_team)
        _value(sheet, r, 6, _num(row.margin))
        _value(sheet, r, 7, _num(row.home_win))
        _value(sheet, r, 8, _num(row.prob_home), PCT2)
        _value(sheet, r, 9, _num(row.model_prob_home), PCT2)
        _value(sheet, r, 10, _num(row.market_prob_home), PCT2)
        _value(sheet, r, 11, _num(row.elo_prob_home), PCT2)
        _value(sheet, r, 12, _num(row.pred_margin), "0.0")
        _value(sheet, r, 13, _num(row.market_spread), "0.0")

        for col, prob, name in (
            (14, "H", "blended"), (15, "I", "model"),
            (16, "J", "market"), (17, "K", "elo"),
        ):
            rec.formula(
                sheet, r, col,
                f'=IF(OR($G{r}="",{prob}{r}=""),"",IF($G{r}=0.5,0.5,'
                f'IF(OR(AND({prob}{r}>0.5,$G{r}=1),AND({prob}{r}<=0.5,$G{r}=0)),1,0)))',
                getattr(row, f"hit_{name}"),
            )
        for col, prob, name in (
            (18, "H", "blended"), (19, "I", "model"),
            (20, "J", "market"), (21, "K", "elo"),
        ):
            rec.formula(
                sheet, r, col,
                f'=IF(OR($G{r}="",{prob}{r}=""),"",({prob}{r}-$G{r})^2)',
                getattr(row, f"brier_{name}"), NUM4,
            )

        rec.formula(sheet, r, 22, f"=MAX($H{r},1-$H{r})", row.confidence, PCT2)
        rec.formula(
            sheet, r, 23, f'=IF($M{r}="","",$L{r}-$M{r})', row.spread_edge, "0.0"
        )
        rec.formula(
            sheet, r, 24,
            f'=IF(OR($M{r}="",$F{r}=""),"",'
            f'IF($W{r}>0,IF($F{r}>$M{r},"W",IF($F{r}=$M{r},"P","L")),'
            f'IF($F{r}<$M{r},"W",IF($F{r}=$M{r},"P","L"))))',
            row.ats_result,
        )
        rec.formula(sheet, r, 25, f'=IF($W{r}="","",ABS($W{r}))', row.abs_edge, "0.0")

        # --- point totals, columns Z-AH
        _value(sheet, r, 26, _num(row.actual_total), "0")
        _value(sheet, r, 27, _num(row.market_total), "0.0")
        _value(sheet, r, 28, _num(row.model_total), "0.0")
        _value(sheet, r, 29, _num(row.pred_total), "0.0")
        for col, source, name in (
            (30, "AC", "blended"), (31, "AB", "model"), (32, "AA", "market"),
        ):
            rec.formula(
                sheet, r, col,
                f'=IF(OR($Z{r}="",{source}{r}=""),"",ABS($Z{r}-{source}{r}))',
                getattr(row, f"err_total_{name}"), "0.00",
            )
        rec.formula(
            sheet, r, 33, f'=IF(OR($AA{r}="",$AB{r}=""),"",$AB{r}-$AA{r})',
            row.ou_edge, "0.0",
        )
        rec.formula(
            sheet, r, 34,
            f'=IF(OR($Z{r}="",$AA{r}="",$AG{r}=""),"",'
            f'IF($AG{r}>0,IF($Z{r}>$AA{r},"W",IF($Z{r}=$AA{r},"P","L")),'
            f'IF($Z{r}<$AA{r},"W",IF($Z{r}=$AA{r},"P","L"))))',
            row.ou_result,
        )
        for col, source, name in (
            (35, "AC", "blended"), (36, "AB", "model"), (37, "AA", "market"),
        ):
            rec.formula(
                sheet, r, col,
                f'=IF(OR($Z{r}="",{source}{r}=""),"",$Z{r}-{source}{r})',
                getattr(row, f"signed_total_{name}"), "0.00",
            )

    last_row = len(derived) + 1
    _set_widths(sheet, [18, 8, 6, 10, 10, 8, 10] + [12] * 6 + [11] * 12 + [12] * 12)
    sheet.freeze_panes = "B2"
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(GAME_LOG_HEADERS))}{last_row}"
    return last_row, derived


# --------------------------------------------------------------------------
# Summary sheets
# --------------------------------------------------------------------------

_METHODS = [
    ("Model + market blend", "N", "R", "blended"),
    ("Model alone (no market)", "O", "S", "model"),
    ("Market (closing line)", "P", "T", "market"),
    ("Elo only", "Q", "U", "elo"),
]


def _build_summary(sheet, rec: Recorder, last_row: int, derived: pd.DataFrame, meta: dict) -> None:
    row = _write_title(
        sheet,
        "Backtest Summary",
        f"Walk-forward {meta['start']}-{meta['end']}: each slate was predicted by a "
        f"model fitted only on games that had already finished.",
    )
    _write_header(sheet, row, ["Method", "Games", "Accuracy", "Brier", "Log loss"])

    first = row + 1
    for offset, (name, hit_col, brier_col, key) in enumerate(_METHODS):
        r = first + offset
        _value(sheet, r, 1, name, font=_BOLD if offset == 0 else _BODY_FONT)
        rec.formula(sheet, r, 2, f"=COUNT({_rng(hit_col, last_row)})",
                    int(pd.Series(derived[f"hit_{key}"]).notna().sum()))
        rec.formula(sheet, r, 3, f"=AVERAGE({_rng(hit_col, last_row)})",
                    _mean(derived[f"hit_{key}"]), PCT2)
        rec.formula(sheet, r, 4, f"=AVERAGE({_rng(brier_col, last_row)})",
                    _mean(derived[f"brier_{key}"]), NUM4)
        _value(sheet, r, 5, meta["log_loss"].get(name), NUM4)
        if offset == 0:
            sheet.cell(row=r, column=3).font = _BOLD

    baseline = first + len(_METHODS)
    _value(sheet, baseline, 1, "Always pick the home team")
    rec.formula(sheet, baseline, 2, f"=COUNT({_rng('G', last_row)})",
                int(derived["home_win"].notna().sum()))
    rec.formula(sheet, baseline, 3, f"=AVERAGE({_rng('G', last_row)})",
                _mean(derived["home_win"]), PCT2)

    note = baseline + 2
    _note(sheet, note, "Games, Accuracy and Brier are live formulas over the Game Log sheet.")
    _note(sheet, note + 1,
          "Log loss is carried from the backtest run: it needs per-game logarithms, "
          "not a column average.")
    _note(sheet, note + 3,
          "The blend ties the closing line rather than beating it. Tuning the blend "
          "weight on 2008-2017 and testing on 2018-2025 put the held-out optimum at "
          "zero model weight.")
    _set_widths(sheet, [30, 10, 12, 12, 12])


def _build_calibration(sheet, rec: Recorder, last_row: int, derived: pd.DataFrame) -> None:
    row = _write_title(
        sheet, "Calibration",
        "Does a stated 70% actually win 70% of the time? Probabilities are folded "
        "onto the favourite's side.",
    )
    _write_header(sheet, row, ["Confidence band", "Low", "High", "Games", "Said", "Actual", "Gap"])

    bands = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
             (0.70, 0.75), (0.75, 0.80), (0.80, 0.90), (0.90, 1.00)]
    conf, hit = _rng("V", last_row), _rng("N", last_row)
    confidence, hits = derived["confidence"], derived["hit_blended"]

    first = row + 1
    for offset, (low, high) in enumerate(bands):
        r = first + offset
        _value(sheet, r, 1, f"{low:.0%} - {high:.0%}")
        _value(sheet, r, 2, low, PCT, _INPUT_FONT)
        _value(sheet, r, 3, high, PCT, _INPUT_FONT)
        mask = (confidence > low) & (confidence <= high)
        criteria = f'{conf},">"&$B{r},{conf},"<="&$C{r}'
        rec.formula(sheet, r, 4, f"=COUNTIFS({criteria})", int(mask.sum()))
        said = _mean(confidence[mask])
        actual = _mean(hits[mask])
        rec.formula(sheet, r, 5, f'=IFERROR(AVERAGEIFS({conf},{criteria}),"")', said, PCT)
        rec.formula(sheet, r, 6, f'=IFERROR(AVERAGEIFS({hit},{criteria}),"")', actual, PCT)
        rec.formula(sheet, r, 7, f'=IFERROR($F{r}-$E{r},"")',
                    actual - said if not (np.isnan(said) or np.isnan(actual)) else np.nan, PCT)

    last = first + len(bands)
    _note(sheet, last + 1, "Band edges (blue) are editable; every other cell recalculates from them.")
    _note(sheet, last + 2, "A positive gap means the model was too cautious; negative means overconfident.")
    _set_widths(sheet, [18, 10, 10, 10, 10, 10, 10])


def _build_ats(sheet, rec: Recorder, last_row: int, derived: pd.DataFrame) -> None:
    row = _write_title(
        sheet, "Against the Spread",
        "Bets placed only where the model's predicted margin differs from the posted "
        "number by at least the edge threshold.",
    )

    payout = 100.0 / abs(config.STANDARD_VIG_ODDS)
    _value(sheet, row - 1, 1, "Assumption -- odds on every bet:", font=_BOLD)
    odds_cell = _value(sheet, row - 1, 2, config.STANDARD_VIG_ODDS, "0", _INPUT_FONT)
    odds_cell.fill = _ASSUMPTION_FILL
    _value(sheet, row - 1, 3, "Net payout per unit:", font=_BOLD)
    rec.formula(sheet, row - 1, 4, "=IF($B3>0,$B3/100,100/ABS($B3))", payout, NUM4)

    _write_header(sheet, row, ["Min edge (pts)", "Bets", "Wins", "Losses", "Pushes", "Win %", "ROI", "Units"])

    result_rng, edge_rng = _rng("X", last_row), _rng("Y", last_row)
    results, abs_edge = derived["ats_result"], derived["abs_edge"]

    first = row + 1
    for offset, threshold in enumerate([0.0, 0.5, 1.0, 1.5, 2.0, 3.0]):
        r = first + offset
        _value(sheet, r, 1, threshold, "0.0", _INPUT_FONT)
        gate = f'{edge_rng},">="&$A{r}'
        playable = abs_edge >= threshold
        counts = {k: int(((results == k) & playable).sum()) for k in ("W", "L", "P")}
        rec.formula(sheet, r, 2, f'=COUNTIFS({result_rng},"<>",{gate})',
                    int(((results != "") & playable).sum()))
        rec.formula(sheet, r, 3, f'=COUNTIFS({result_rng},"W",{gate})', counts["W"])
        rec.formula(sheet, r, 4, f'=COUNTIFS({result_rng},"L",{gate})', counts["L"])
        rec.formula(sheet, r, 5, f'=COUNTIFS({result_rng},"P",{gate})', counts["P"])
        decided = counts["W"] + counts["L"]
        rec.formula(sheet, r, 6, f'=IFERROR($C{r}/($C{r}+$D{r}),"")',
                    counts["W"] / decided if decided else np.nan, PCT2)
        rec.formula(sheet, r, 7, f'=IFERROR(($C{r}*$D$3-$D{r})/($C{r}+$D{r}),"")',
                    (counts["W"] * payout - counts["L"]) / decided if decided else np.nan, PCT2)
        rec.formula(sheet, r, 8, f"=$C{r}*$D$3-$D{r}",
                    counts["W"] * payout - counts["L"], "0.0")

    last = first + 6
    _value(sheet, last, 1, "Breakeven win rate", font=_BOLD)
    rec.formula(sheet, last, 6, "=1/(1+$D$3)", 1.0 / (1.0 + payout), PCT2).font = _BOLD
    _note(sheet, last + 2,
          "The model finishes below breakeven at every threshold with a meaningful "
          "sample. A high win rate on a few dozen bets is noise, not an edge.")
    _set_widths(sheet, [16, 10, 10, 10, 10, 10, 10, 10])


def _build_by_season(sheet, rec: Recorder, last_row: int, derived: pd.DataFrame) -> None:
    row = _write_title(sheet, "Accuracy by Season", "Blended model, out-of-sample.")
    _write_header(sheet, row, ["Season", "Games", "Accuracy", "Brier"])

    season_rng = _rng("B", last_row)
    seasons = sorted(derived["season"].unique())
    first = row + 1
    for offset, season in enumerate(seasons):
        r = first + offset
        _value(sheet, r, 1, int(season), "0")
        mask = derived["season"] == season
        match = f"{season_rng},$A{r}"
        rec.formula(sheet, r, 2, f"=COUNTIFS({match})", int(mask.sum()))
        rec.formula(sheet, r, 3, f'=IFERROR(AVERAGEIFS({_rng("N", last_row)},{match}),"")',
                    _mean(derived.loc[mask, "hit_blended"]), PCT2)
        rec.formula(sheet, r, 4, f'=IFERROR(AVERAGEIFS({_rng("R", last_row)},{match}),"")',
                    _mean(derived.loc[mask, "brier_blended"]), NUM4)

    total = first + len(seasons)
    _value(sheet, total, 1, "All", font=_BOLD)
    rec.formula(sheet, total, 2, f"=SUM($B${first}:$B${total - 1})", int(len(derived))).font = _BOLD
    rec.formula(sheet, total, 3, f"=AVERAGE({_rng('N', last_row)})",
                _mean(derived["hit_blended"]), PCT2).font = _BOLD
    rec.formula(sheet, total, 4, f"=AVERAGE({_rng('R', last_row)})",
                _mean(derived["brier_blended"]), NUM4).font = _BOLD
    _set_widths(sheet, [12, 10, 12, 12])


# --------------------------------------------------------------------------
# Slate, ratings and cover sheet
# --------------------------------------------------------------------------


def _tier(confidence: float) -> str:
    if pd.isna(confidence):
        return ""
    for threshold, label in config.CONFIDENCE_TIERS:
        if confidence >= threshold:
            return label
    return "COINFLIP"


def _build_predictions(sheet, rec: Recorder, slate: pd.DataFrame, meta: dict) -> None:
    row = _write_title(
        sheet,
        f"{meta['slate_season']} Week {meta['slate_week']} Predictions",
        f"{len(slate)} games, fitted on {meta['slate_train']:,} completed games. "
        f"Model and Line are home-team point spreads (positive = home favoured).",
    )
    _write_header(
        sheet, row,
        ["Away", "Home", "Pick", "Win %", "Confidence", "Model", "Line", "Edge",
         "Market Win %", "P(home)", "P(home) mkt", "Actual", "Outcome",
         "Fair ML", "Book ML", "EV / $1", "My pick", "My result"],
    )

    first = row + 1
    for offset, game in enumerate(slate.itertuples(index=False)):
        r = first + offset
        prob = float(game.prob_home)
        picked_home = prob >= 0.5
        market_home = _num(getattr(game, "market_prob_home", None))
        margin = _num(getattr(game, "margin", None))
        spread = _num(game.market_spread)

        _value(sheet, r, 1, game.away_team)
        _value(sheet, r, 2, game.home_team)
        rec.formula(sheet, r, 3, f"=IF($J{r}>=0.5,$B{r},$A{r})",
                    game.home_team if picked_home else game.away_team).font = _BOLD
        confidence = max(prob, 1.0 - prob)
        rec.formula(sheet, r, 4, f"=MAX($J{r},1-$J{r})", confidence, PCT)
        rec.formula(
            sheet, r, 5,
            f'=IF($D{r}>=0.75,"HIGH",IF($D{r}>=0.65,"MEDIUM",'
            f'IF($D{r}>=0.575,"LEAN","COINFLIP")))',
            _tier(confidence),
        )
        _value(sheet, r, 6, _num(game.pred_margin), SPREAD)
        _value(sheet, r, 7, spread, SPREAD)
        rec.formula(sheet, r, 8, f'=IF($G{r}="","",$F{r}-$G{r})',
                    None if spread is None else float(game.pred_margin) - spread, SPREAD)
        rec.formula(
            sheet, r, 9, f'=IF($K{r}="","",IF($J{r}>=0.5,$K{r},1-$K{r}))',
            None if market_home is None else (market_home if picked_home else 1 - market_home),
            PCT,
        )
        _value(sheet, r, 10, prob, PCT)
        _value(sheet, r, 11, market_home, PCT)
        _value(sheet, r, 12, margin, "0")
        outcome = ""
        if margin is not None:
            outcome = "PUSH" if margin == 0 else ("HIT" if (margin > 0) == picked_home else "MISS")
        rec.formula(
            sheet, r, 13,
            f'=IF($L{r}="","",IF($L{r}=0,"PUSH",'
            f'IF(OR(AND($J{r}>=0.5,$L{r}>0),AND($J{r}<0.5,$L{r}<0)),"HIT","MISS")))',
            outcome,
        )

        # The model's own price on the side it picked, the book's price on the
        # same side, and what a unit stake is worth at the book's number.
        rec.formula(
            sheet, r, 14, f"=-100*$D{r}/(1-$D{r})",
            -100.0 * confidence / (1.0 - confidence) if confidence < 1.0 else None,
            "+0;-0",
        )
        book_ml = _num(getattr(game, "pick_odds", None))
        _value(sheet, r, 15, book_ml, "+0;-0")
        payout = (
            None if book_ml is None
            else (book_ml / 100.0 if book_ml > 0 else 100.0 / abs(book_ml))
        )
        rec.formula(
            sheet, r, 16,
            f'=IF($O{r}="","",$D{r}*IF($O{r}>0,$O{r}/100,100/ABS($O{r}))-(1-$D{r}))',
            None if payout is None else confidence * payout - (1.0 - confidence),
            "+0.000;-0.000",
        )

        # Yours to fill in: pick a side and the sheet grades it beside the
        # model's, so a disagreement is visible rather than remembered.
        pick_cell = _value(sheet, r, 17, None)
        pick_cell.font = _INPUT_FONT
        pick_cell.fill = _ASSUMPTION_FILL
        own_pick = DataValidation(
            type="list", formula1=f"=$A${r}:$B${r}", allow_blank=True
        )
        sheet.add_data_validation(own_pick)
        own_pick.add(pick_cell)
        rec.formula(
            sheet, r, 18,
            f'=IF(OR($Q{r}="",$L{r}=""),"",IF($L{r}=0,"PUSH",'
            f'IF(OR(AND($Q{r}=$B{r},$L{r}>0),AND($Q{r}=$A{r},$L{r}<0)),"HIT","MISS")))',
            "",
        )

        # Hidden lookup block for the Matchup Picker: away@home -> the numbers.
        _value(sheet, r, 26, f"{game.away_team}@{game.home_team}")
        _value(sheet, r, 27, _num(game.pred_margin), SPREAD)
        _value(sheet, r, 28, prob, PCT)
        _value(sheet, r, 29, _num(getattr(game, "pred_total", None)), "0.0")

    last = first + len(slate)
    _note(sheet, last + 1, "Edge = Model minus Line. A positive edge favours the home side.")
    _note(sheet, last + 2,
          "Outcome fills in for games already played. Those remain out-of-sample: the "
          "model never trained on its own slate.")
    _note(sheet, last + 3,
          "Fair ML is the price the model's own probability implies, with no vig. "
          "Book ML is what is actually posted on that side. EV is per $1 staked at "
          "the book price -- negative almost everywhere, which is the vig working.")
    _note(sheet, last + 4,
          "My pick (yellow) is a dropdown of the two teams. Fill it in and My "
          "result grades it once the game lands, so your record sits next to the "
          "model's rather than in your head.")
    _set_widths(
        sheet,
        [9, 9, 9, 10, 13, 9, 9, 9, 13, 11, 12, 9, 10, 10, 10, 10, 11, 11],
    )
    # Columns Z-AC feed the Matchup Picker and are not meant to be read.
    for column in ("Z", "AA", "AB", "AC"):
        sheet.column_dimensions[column].hidden = True
    sheet.freeze_panes = f"A{first}"


def _build_totals_slate(sheet, rec: Recorder, slate: pd.DataFrame, meta: dict) -> None:
    """Point-total forecast for the slate being predicted."""
    blend = meta.get("total_blend", config.DEFAULT_TOTAL_MARKET_BLEND)
    row = _write_title(
        sheet,
        f"{meta['slate_season']} Week {meta['slate_week']} Point Totals",
        f"Model is the model's own number; Line is the posted total; Projection "
        f"blends them at {blend:.0%} model. Edge is Model minus Line.",
    )
    _write_header(
        sheet, row,
        ["Away", "Home", "Pick", "Model", "Line", "Projection", "Edge",
         "P(over)", "P(push)", "Confidence", "Actual", "Outcome"],
    )

    first = row + 1
    for offset, game in enumerate(slate.itertuples(index=False)):
        r = first + offset
        line = _num(getattr(game, "market_total", None))
        model_total = _num(getattr(game, "model_total", None))
        projection = _num(getattr(game, "pred_total", None))
        prob_over = _num(getattr(game, "prob_over", None))
        actual = _num(getattr(game, "actual_total", None))

        # A slate can reach this sheet with no totals attached at all (a
        # caller that only ran the winner model), so every cell below has to
        # read as blank rather than raise.
        priced = line is not None and projection is not None
        over = priced and projection > line

        _value(sheet, r, 1, game.away_team)
        _value(sheet, r, 2, game.home_team)
        rec.formula(
            sheet, r, 3, f'=IF(OR($E{r}="",$F{r}=""),"-",IF($F{r}>$E{r},"OVER","UNDER"))',
            "OVER" if over else ("UNDER" if priced else "-"),
        ).font = _BOLD
        _value(sheet, r, 4, model_total, "0.0")
        _value(sheet, r, 5, line, "0.0")
        _value(sheet, r, 6, projection, "0.0")
        rec.formula(
            sheet, r, 7, f'=IF(OR($D{r}="",$E{r}=""),"",$D{r}-$E{r})',
            None if (line is None or model_total is None) else model_total - line,
            SPREAD,
        )
        _value(sheet, r, 8, prob_over, PCT)
        _value(sheet, r, 9, _num(getattr(game, "prob_push", None)), PCT)
        confidence = None if prob_over is None else (prob_over if over else 1 - prob_over)
        rec.formula(
            sheet, r, 10,
            f'=IF($H{r}="","-",IF(MAX($H{r},1-$H{r})>=0.575,"LEAN","COINFLIP"))',
            "-" if confidence is None else ("LEAN" if confidence >= 0.575 else "COINFLIP"),
        )
        _value(sheet, r, 11, actual, "0")
        outcome = ""
        if actual is not None and priced:
            if actual == line:
                outcome = "PUSH"
            else:
                outcome = "HIT" if (actual > line) == over else "MISS"
        rec.formula(
            sheet, r, 12,
            f'=IF(OR($K{r}="",$E{r}=""),"",IF($K{r}=$E{r},"PUSH",'
            f'IF(OR(AND($F{r}>$E{r},$K{r}>$E{r}),AND($F{r}<=$E{r},$K{r}<$E{r})),'
            f'"HIT","MISS")))',
            outcome,
        )

    last = first + len(slate)
    _note(sheet, last + 1,
          "Projection is what the model actually stands behind. Because the blend "
          "keeps only a tenth of the model's disagreement, Edge is usually much "
          "larger than Projection minus Line -- that gap is deliberate.")
    _note(sheet, last + 2,
          "Walk-forward over 2008-2026 the model missed the final total by 10.68 "
          "points on average and the closing total by 10.47. Over/under picks hit "
          "50.8%, below the 52.38% needed to break even at -110.")
    _set_widths(sheet, [9, 9, 9, 9, 9, 12, 9, 10, 10, 13, 9, 10])
    sheet.freeze_panes = f"A{first}"


def _build_team_totals(sheet, rec: Recorder, slate: pd.DataFrame, meta: dict) -> None:
    """Per-team totals and the first-half split, derived from the game pair."""
    row = _write_title(
        sheet,
        f"{meta['slate_season']} Week {meta['slate_week']} Team Totals and Halves",
        "Each side's projected points, and the first-half split. Derived from "
        "the game total and the spread, so the two team totals add to the game.",
    )
    _write_header(
        sheet, row,
        ["Away", "Home", "Away pts", "Home pts", "Game total", "Spread",
         "1H total", "1H spread", "1H away", "1H home", "2H total",
         "Actual away", "Actual home"],
    )

    first = row + 1
    for offset, game in enumerate(slate.itertuples(index=False)):
        r = first + offset
        total = _num(getattr(game, "pred_total", None))
        margin = _num(getattr(game, "pred_margin", None))
        fh_total = _num(getattr(game, "first_half_total_pred", None))
        fh_margin = _num(getattr(game, "first_half_margin_pred", None))

        _value(sheet, r, 1, game.away_team)
        _value(sheet, r, 2, game.home_team)
        # The split is arithmetic, so it is written as arithmetic: edit the
        # game total or the spread and both team totals follow.
        rec.formula(
            sheet, r, 3, f'=IF(OR($E{r}="",$F{r}=""),"",($E{r}-$F{r})/2)',
            None if total is None or margin is None else (total - margin) / 2.0, "0.0",
        )
        rec.formula(
            sheet, r, 4, f'=IF(OR($E{r}="",$F{r}=""),"",($E{r}+$F{r})/2)',
            None if total is None or margin is None else (total + margin) / 2.0, "0.0",
        )
        _value(sheet, r, 5, total, "0.0")
        _value(sheet, r, 6, margin, SPREAD)
        _value(sheet, r, 7, fh_total, "0.0")
        _value(sheet, r, 8, fh_margin, SPREAD)
        rec.formula(
            sheet, r, 9, f'=IF(OR($G{r}="",$H{r}=""),"",($G{r}-$H{r})/2)',
            None if fh_total is None or fh_margin is None else (fh_total - fh_margin) / 2.0,
            "0.0",
        )
        rec.formula(
            sheet, r, 10, f'=IF(OR($G{r}="",$H{r}=""),"",($G{r}+$H{r})/2)',
            None if fh_total is None or fh_margin is None else (fh_total + fh_margin) / 2.0,
            "0.0",
        )
        rec.formula(
            sheet, r, 11, f'=IF(OR($E{r}="",$G{r}=""),"",$E{r}-$G{r})',
            None if total is None or fh_total is None else total - fh_total, "0.0",
        )
        _value(sheet, r, 12, _num(getattr(game, "away_score", None)), "0")
        _value(sheet, r, 13, _num(getattr(game, "home_score", None)), "0")

    last = first + len(slate)
    _note(sheet, last + 1,
          "Away pts and Home pts are formulas over Game total and Spread, so "
          "editing either updates both. They add to the game total by construction.")
    _note(sheet, last + 2,
          "Walk-forward 2008-2026: team totals miss by 7.5 and 7.3 points against "
          "8.1 for guessing the league average. The first-half total misses by 7.03 "
          "against 7.22 for that same naive guess -- barely any edge. First halves "
          "are mostly noise; the first-half side is right 60.7% of the time against "
          "66.6% for the full game.")
    _set_widths(sheet, [9, 9, 10, 10, 11, 9, 10, 10, 9, 9, 10, 12, 12])
    sheet.freeze_panes = f"A{first}"


PROP_SECTIONS = (
    ("proj_passing_yards", "Passing yards"),
    ("proj_rushing_yards", "Rushing yards"),
    ("proj_receiving_yards", "Receiving yards"),
)


def _build_props(sheet, props: pd.DataFrame, meta: dict, *, top: int = 30) -> None:
    """Player yardage projections for the slate, one block per category."""
    row = _write_title(
        sheet,
        f"{meta['slate_season']} Week {meta['slate_week']} Player Projections",
        "Projected yards for the players expected to appear. Sorted within "
        "each category; a question mark marks an injury-report listing.",
    )

    if props is None or props.empty:
        _note(sheet, row, "No player projections available for this slate.")
        _set_widths(sheet, [26, 8, 7, 10, 12])
        return

    for column, title in PROP_SECTIONS:
        if column not in props.columns or props[column].isna().all():
            continue
        ranked = props.nlargest(top, column)
        ranked = ranked[ranked[column] > 0]
        if ranked.empty:
            continue

        sheet.cell(row=row, column=1, value=title).font = _SECTION_FONT
        row += 1
        _write_header(sheet, row, ["Player", "Team", "Pos", "Projection", "Status"])
        row += 1
        for game in ranked.itertuples(index=False):
            availability = float(getattr(game, "availability", 1.0) or 1.0)
            _value(sheet, row, 1, str(getattr(game, "player_display_name", "")))
            _value(sheet, row, 2, str(getattr(game, "team", "")))
            _value(sheet, row, 3, str(getattr(game, "position", "")))
            _value(sheet, row, 4, _num(getattr(game, column)), "0.0")
            _value(
                sheet, row, 5,
                "available" if availability >= 0.999 else f"listed ({availability:.0%})",
            )
            row += 1
        row += 1

    _note(sheet, row,
          "Walk-forward 2012-2026, players with real involvement: projections miss "
          "by 62 passing yards, 24 rushing and 23 receiving. That player's own "
          "recent average misses by 68 / 26 / 25, and quoting the league average "
          "for the position misses by 64 / 27 / 26 -- so the gain is clear on "
          "rushing and receiving and slim on passing yards, where starting "
          "quarterbacks cluster tightly. A 60-yard miss is still a wide miss.")
    _note(sheet, row + 1,
          "No over/under probability is offered: yardage is long-tailed, and a "
          "normal curve would understate how often a projection is badly wrong.")
    _note(sheet, row + 2,
          "The roster is whoever played in the last four weeks, so a player "
          "promoted this week is projected on the role he had last month.")
    _set_widths(sheet, [26, 8, 7, 12, 16])


def _build_scorecard(sheet, rec: Recorder, card, meta: dict) -> None:
    """This season's live results, week by week, next to the long-run figures."""
    season = getattr(card, "season", meta.get("slate_season", ""))
    row = _write_title(
        sheet,
        f"{season} Season Scorecard",
        "Every week graded using only games that had finished before that "
        "slate kicked off -- the same call the tool would have made on the day.",
    )

    if card is None or card.games.empty:
        _note(sheet, row, f"No settled games yet for {season}.")
        _set_widths(sheet, [10, 10, 10, 10, 12, 12, 12])
        return

    _write_header(
        sheet, row,
        ["Week", "Games", "Right", "Accuracy", "ATS W-L", "O/U W-L", "Total error"],
    )
    first = row + 1
    for offset, week in enumerate(card.by_week.itertuples(index=False)):
        r = first + offset
        _value(sheet, r, 1, int(week.week))
        _value(sheet, r, 2, int(week.games))
        _value(sheet, r, 3, float(week.correct), "0")
        rec.formula(
            sheet, r, 4, f'=IF($B{r}=0,"",$C{r}/$B{r})', float(week.accuracy), PCT
        )
        _value(
            sheet, r, 5,
            "-" if not week.ats_played
            else f"{int(week.ats_wins)}-{int(week.ats_played - week.ats_wins)}",
        )
        _value(
            sheet, r, 6,
            "-" if not week.ou_played
            else f"{int(week.ou_wins)}-{int(week.ou_played - week.ou_wins)}",
        )
        _value(sheet, r, 7, float(week.total_error), "0.00")

    last = first + len(card.by_week)
    summary = card.summary
    _value(sheet, last, 1, "All", font=_BOLD)
    _value(sheet, last, 2, int(summary["games"]), font=_BOLD)
    _value(sheet, last, 3, float(summary["correct"]), "0", font=_BOLD)
    rec.formula(
        sheet, last, 4, f'=IF($B{last}=0,"",$C{last}/$B{last})',
        float(summary["accuracy"]), PCT,
    ).font = _BOLD
    _value(
        sheet, last, 5,
        "-" if not summary["ats_played"]
        else f"{int(summary['ats_wins'])}-"
             f"{int(summary['ats_played'] - summary['ats_wins'])}",
        font=_BOLD,
    )
    _value(
        sheet, last, 6,
        "-" if not summary["ou_played"]
        else f"{int(summary['ou_wins'])}-"
             f"{int(summary['ou_played'] - summary['ou_wins'])}",
        font=_BOLD,
    )
    _value(sheet, last, 7, float(summary["total_mae"]), "0.00", font=_BOLD)

    row = last + 2
    sheet.cell(row=row, column=1, value="This season against the long run").font = _SECTION_FONT
    row += 1
    _write_header(sheet, row, ["Measure", "This season", "Long run (2008-2026)"])
    row += 1
    for label, value, benchmark, fmt in (
        ("Straight up", summary["accuracy"], 0.6661, PCT),
        ("Against the spread", summary["ats_rate"], 0.4960, PCT),
        ("Over/under", summary["ou_rate"], 0.5077, PCT),
        ("Brier score", summary["brier"], 0.2099, NUM4),
        ("Total error (points)", summary["total_mae"], 10.460, "0.000"),
    ):
        _value(sheet, row, 1, label)
        _value(sheet, row, 2, _num(value), fmt)
        _value(sheet, row, 3, benchmark, fmt)
        row += 1

    _note(sheet, row + 1,
          f"{int(summary['games'])} games is far too few to judge a model on. A "
          "single season swings several points either side of the long-run figure "
          "on noise alone -- a hot start is not evidence the model improved, and a "
          "cold one is not evidence it broke.")
    _note(sheet, row + 2,
          "Nothing here is stored between runs: the scorecard is recomputed from "
          "the game log every time, so it cannot drift out of step with a corrected "
          "result the way a running tally would.")
    _set_widths(sheet, [22, 12, 12, 12, 12, 12, 13])


def _build_totals_backtest(sheet, rec: Recorder, last_row: int, derived: pd.DataFrame) -> None:
    """Totals accuracy and over/under record, as live formulas over the Game Log."""
    row = _write_title(
        sheet,
        "Point Totals -- Backtest",
        "Every figure is a formula over the Game Log, so filtering that sheet "
        "re-scores this one.",
    )

    _write_header(sheet, row, ["Method", "Games", "MAE", "RMSE", "Bias"])
    methods = [
        ("Model + market blend", TOTAL_COLS["err_blend"], TOTAL_COLS["signed_blend"], "blended"),
        ("Model alone (no market)", TOTAL_COLS["err_model"], TOTAL_COLS["signed_model"], "model"),
        ("Market (closing total)", TOTAL_COLS["err_market"], TOTAL_COLS["signed_market"], "market"),
    ]
    first = row + 1
    for offset, (name, err_col, signed_col, key) in enumerate(methods):
        r = first + offset
        err_range = _rng(err_col, last_row)
        signed_range = _rng(signed_col, last_row)
        errors = pd.Series(derived[f"err_total_{key}"]).dropna()
        signed = pd.Series(derived[f"signed_total_{key}"]).dropna()

        _value(sheet, r, 1, name)
        rec.formula(sheet, r, 2, f"=COUNT({err_range})", int(len(errors)), "#,##0")
        rec.formula(sheet, r, 3, f"=AVERAGE({err_range})", _mean(errors), "0.000")
        # SUMSQ skips text and blanks, so the range needs no masking.
        rec.formula(
            sheet, r, 4,
            f"=SQRT(SUMSQ({err_range})/COUNT({err_range}))",
            float(np.sqrt((errors**2).mean())) if len(errors) else float("nan"),
            "0.000",
        )
        rec.formula(sheet, r, 5, f"=AVERAGE({signed_range})", _mean(signed), "+0.000;-0.000")

    row = first + len(methods) + 2
    sheet.cell(row=row, column=1, value="Over/under record").font = _SECTION_FONT
    row += 1
    _write_header(sheet, row, ["Min edge", "Bets", "Won", "Lost", "Push", "Win %", "ROI"])

    edge_range = _rng(TOTAL_COLS["edge"], last_row)
    result_range = _rng(TOTAL_COLS["result"], last_row)
    payout = 100.0 / abs(config.STANDARD_VIG_ODDS)

    graded = derived[derived["ou_result"].isin(["W", "L", "P"])]
    first_bet = row + 1
    for offset, threshold in enumerate((0.0, 1.0, 2.0, 3.0, 4.0, 5.0)):
        r = first_bet + offset
        bucket = graded[graded["ou_edge"].abs() >= threshold]
        wins = int((bucket["ou_result"] == "W").sum())
        losses = int((bucket["ou_result"] == "L").sum())
        pushes = int((bucket["ou_result"] == "P").sum())
        decided = wins + losses

        _value(sheet, r, 1, threshold, "0.0")
        rec.formula(
            sheet, r, 2,
            f'=COUNTIFS({edge_range},">="&$A{r})+COUNTIFS({edge_range},"<="&-$A{r})'
            if threshold else f"=COUNT({edge_range})",
            len(bucket), "#,##0",
        )
        for col, code, value in ((3, "W", wins), (4, "L", losses), (5, "P", pushes)):
            rec.formula(
                sheet, r, col,
                f'=COUNTIFS({result_range},"{code}",{edge_range},">="&$A{r})'
                f'+COUNTIFS({result_range},"{code}",{edge_range},"<="&-$A{r})',
                value, "#,##0",
            )
        rec.formula(
            sheet, r, 6, f'=IF($C{r}+$D{r}=0,"",$C{r}/($C{r}+$D{r}))',
            float(wins / decided) if decided else None, PCT2,
        )
        rec.formula(
            sheet, r, 7,
            f'=IF($C{r}+$D{r}=0,"",($C{r}*{payout:.6f}-$D{r})/($C{r}+$D{r}))',
            float((wins * payout - losses) / decided) if decided else None, PCT2,
        )

    last = first_bet + 6
    _note(sheet, last + 1,
          f"Breakeven at {config.STANDARD_VIG_ODDS} is "
          f"{1 / (1 + payout) * 100:.2f}%. Edge is graded against the model's own "
          "total, not the blend.")
    _note(sheet, last + 2,
          "The 3-4 point buckets edge just past breakeven, but on 940-1,612 bets "
          "the standard error is around 1.5 points of win rate -- that is noise, "
          "not a system.")
    _set_widths(sheet, [26, 10, 10, 10, 10, 11, 11])


def _build_ratings(sheet, rec: Recorder, ratings: pd.DataFrame) -> None:
    row = _write_title(
        sheet, "Power Ratings (Elo)",
        f"{config.ELO_PER_POINT:.0f} Elo points = 1 point of point spread. "
        f"{config.ELO_INIT:.0f} is league average.",
    )
    _write_header(sheet, row, ["Rank", "Team", "Elo", "Pts vs average"])

    first = row + 1
    for offset, team in enumerate(ratings.itertuples(index=False)):
        r = first + offset
        _value(sheet, r, 1, offset + 1)
        _value(sheet, r, 2, team.team)
        _value(sheet, r, 3, round(float(team.elo), 1), "0")
        rec.formula(
            sheet, r, 4, f"=($C{r}-{config.ELO_INIT})/{config.ELO_PER_POINT}",
            (float(team.elo) - config.ELO_INIT) / config.ELO_PER_POINT, SPREAD,
        )
    _set_widths(sheet, [8, 10, 10, 16])
    sheet.freeze_panes = f"A{first}"


_READ_ME = [
    ("title", "nflpredict -- Model Output Workbook"),
    ("blank", ""),
    ("head", "What is in here"),
    ("text", "Predictions -- the upcoming slate: pick, win probability, confidence tier, the model's spread against the posted line, and its fair moneyline against the posted price."),
    ("text", "Point Totals -- the same slate scored for points: the model's own total, the posted total, the blend of the two, and the over/under it implies."),
    ("text", "Team Totals -- each side's projected points and the first-half split, derived from the game total and the spread."),
    ("text", "Player Projections -- projected passing, rushing and receiving yards for the players expected to appear."),
    ("text", "Season Scorecard -- how this season's picks have actually landed, week by week, against the long-run figures."),
    ("text", "Power Ratings -- every franchise's current Elo, and what it is worth in points."),
    ("text", "Backtest Summary -- the model measured against the market, against Elo alone, and against simply picking the home team."),
    ("text", "Calibration -- whether a stated 70% actually wins 70% of the time."),
    ("text", "Against the Spread -- the betting record at several edge thresholds."),
    ("text", "Point Totals Backtest -- how far the totals forecast missed by, and its over/under record at several edge thresholds."),
    ("text", "Accuracy by Season -- year-by-year out-of-sample results."),
    ("text", "Game Log -- every backtested game, one row each. Every summary figure is a live formula over this sheet."),
    ("blank", ""),
    ("head", "How accurate is it, honestly"),
    ("text", "About two games in three. Walk-forward across the backtest range it hits roughly 66.6% straight up, which is level with the Vegas closing line and no better than it."),
    ("text", "Roughly a third of NFL games turn on events with no predictable structure: a tipped pass, a missed field goal, a fumble bounce. No model removes that, and this one does not try to pretend otherwise."),
    ("text", "Against the spread it wins about 49.6% and loses money at standard -110 pricing, where 52.38% is breakeven. Any tool claiming 90%+ accuracy or guaranteed picks is fitting noise or testing on data it already trained on."),
    ("text", "Point totals are harder still. The model misses the final total by about 10.7 points on average; the closing total misses by 10.5. Its over/under picks hit 50.8%, also short of breakeven. Totals are a forecast of how a game will be played, not a soft spot in the market."),
    ("blank", ""),
    ("head", "Why these numbers can be trusted"),
    ("text", "Every figure came from walk-forward testing: for each week, the model was fitted only on games that had already finished. It never saw a result from its own slate or from any later week."),
    ("text", "That guarantee is enforced structurally in the code and asserted by a test that rebuilds the entire feature set with every future result erased, requiring that no past value moves."),
    ("blank", ""),
    ("head", "Working with this workbook"),
    ("text", "The Game Log holds the raw data. Accuracy, Brier, calibration, ATS and per-season figures are formulas over it, so filtering or correcting a row updates them."),
    ("text", "Blue cells are inputs you can edit: calibration band edges, ATS edge thresholds, and the odds assumption (yellow) on the Against the Spread sheet."),
    ("text", "Log loss on the Backtest Summary is the one carried-over value, because it needs per-game logarithms rather than a column average."),
    ("text", "Regenerate at any time with:  nflpredict export"),
]


def _build_read_me(sheet, meta: dict, *, content=None) -> None:
    row = 1
    for kind, text in (content if content is not None else _READ_ME):
        if kind == "blank":
            row += 1
            continue
        cell = sheet.cell(row=row, column=1, value=text)
        if kind == "title":
            cell.font = _TITLE_FONT
        elif kind == "head":
            cell.font = _SECTION_FONT
        else:
            cell.font = _BODY_FONT
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        row += 1

    row += 1
    facts = [("Generated", meta["generated"])]
    if "n_games" in meta:
        facts += [
            ("Backtest range", f"{meta['start']}-{meta['end']}"),
            ("Games backtested", f"{meta['n_games']:,}"),
        ]
    for label, value in facts + [
        ("Slate", f"{meta['slate_season']} week {meta['slate_week']}"),
        ("Games in the fit", f"{meta['slate_train']:,}"),
        ("Market blend weight (sides)", f"{meta['market_blend']:.2f}"),
        ("Market blend weight (totals)", f"{meta.get('total_blend', 0.10):.2f}"),
        ("Totals sigma (points)", f"{meta.get('total_sigma', float('nan')):.2f}"),
        ("Current-season games in fit", meta.get("slate_history_note") or "none yet"),
        ("Data source", "nflverse (nfldata game log + nflverse-data play-by-play)"),
    ]:
        _value(sheet, row, 1, label, font=_BOLD)
        _value(sheet, row, 2, value)
        row += 1

    sheet.column_dimensions["A"].width = 118
    sheet.column_dimensions["B"].width = 52


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


# Every tab, with the one-line answer to "why would I open this".
NAVIGATION = [
    ("Predictions", "Who wins each game, by how much, and at what price"),
    ("Matchup Picker", "Pick any two teams and see the projection update live"),
    ("Point Totals", "How many points each game is projected to produce"),
    ("Team Totals", "Each side's points, plus the first-half split"),
    ("Player Projections", "Passing, rushing and receiving yards per player"),
    ("Season Scorecard", "How this season's picks have actually landed"),
    ("Power Ratings", "Every franchise's rating, in points"),
    ("Team Data", "The per-team figures the Matchup Picker reads"),
    ("Backtest Summary", "The model against the market, Elo, and picking the home team"),
    ("Calibration", "Whether a stated 70% really wins 70% of the time"),
    ("Against the Spread", "The spread record at six edge thresholds"),
    ("Point Totals Backtest", "How far the totals forecast missed, and the over/under record"),
    ("Accuracy by Season", "Year-by-year, out of sample"),
    ("Game Log", "Every backtested game. All the summary tabs are formulas over this"),
    ("Read Me", "What everything means, and how accurate it honestly is"),
]


def _internal_link(cell, target: str) -> None:
    """Point a cell at another sheet in this workbook.

    Assigning a string to ``cell.hyperlink`` makes openpyxl write an
    *external* relationship, which Excel then cannot follow back into its own
    workbook. An internal jump has to set ``location`` instead, and a sheet
    name containing a space has to be quoted inside it.
    """
    quoted = f"'{target}'" if " " in target else target
    cell.hyperlink = Hyperlink(ref=cell.coordinate, location=f"{quoted}!A1")


def _link(sheet, row: int, target: str, label: str, description: str) -> None:
    """A clickable jump to another tab, with a line on why to go there."""
    cell = sheet.cell(row=row, column=1, value=label)
    _internal_link(cell, target)
    cell.font = Font(name=FONT, size=11, bold=True, color="0563C1", underline="single")
    _value(sheet, row, 2, description, font=_NOTE_FONT)


def _build_start_here(sheet, slate: pd.DataFrame, meta: dict, scorecard) -> None:
    """The front page: where you are, what the week looks like, where to go."""
    sheet["A1"] = "nflpredict"
    sheet["A1"].font = Font(name=FONT, size=20, bold=True, color="1F3864")
    sheet["A2"] = (
        f"{meta['slate_season']} Week {meta['slate_week']}  ·  "
        f"{len(slate)} games  ·  generated {meta['generated']}"
    )
    sheet["A2"].font = _NOTE_FONT

    row = 4
    sheet.cell(row=row, column=1, value="This week at a glance").font = _SECTION_FONT
    row += 1

    headline = _headline_rows(slate, meta, scorecard)
    for label, value in headline:
        _value(sheet, row, 1, label, font=_BOLD)
        _value(sheet, row, 2, value)
        row += 1

    row += 1
    sheet.cell(row=row, column=1, value="Where to go").font = _SECTION_FONT
    row += 1
    for target, description in NAVIGATION:
        _link(sheet, row, target, target, description)
        row += 1

    row += 1
    sheet.cell(row=row, column=1, value="Before you read anything else").font = _SECTION_FONT
    row += 1
    for line in (
        "This model is wrong about one game in three, and nothing will fix that.",
        "It hits 66.6% straight up over 4,912 backtested games -- level with the "
        "closing line, not better than it.",
        "Against the spread it wins 49.6% and loses money at standard -110 pricing, "
        "where 52.38% is breakeven.",
        "Yellow cells are yours to edit. Everything else is a formula you can click "
        "to check.",
    ):
        cell = sheet.cell(row=row, column=1, value=line)
        cell.font = _BODY_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        row += 1

    _set_widths(sheet, [34, 78])


def _headline_rows(slate: pd.DataFrame, meta: dict, scorecard) -> list:
    """The handful of numbers worth seeing before opening a tab."""
    rows = []
    if not slate.empty and "pick_prob" in slate.columns:
        best = slate.loc[slate["pick_prob"].idxmax()]
        rows.append(
            (
                "Most confident pick",
                f"{best['pick']} over "
                f"{best['away_team'] if best['pick'] == best['home_team'] else best['home_team']}"
                f" ({float(best['pick_prob']):.0%})",
            )
        )
        coinflips = int((slate["pick_prob"] < 0.575).sum())
        rows.append(("Coin-flip games this week", f"{coinflips} of {len(slate)}"))
    if not slate.empty and "pred_total" in slate.columns:
        highest = slate.loc[slate["pred_total"].idxmax()]
        rows.append(
            (
                "Highest projected total",
                f"{highest['away_team']} @ {highest['home_team']}"
                f"  ({float(highest['pred_total']):.1f} points)",
            )
        )
    if scorecard is not None and not getattr(scorecard, "games", pd.DataFrame()).empty:
        summary = scorecard.summary
        rows.append(
            (
                f"{scorecard.season} record so far",
                f"{int(summary['correct'])} of {summary['games']} "
                f"({summary['accuracy']:.1%}) straight up",
            )
        )
    rows.append(
        ("Games behind the accuracy figures", f"{meta.get('n_games', 0):,}")
    )
    rows.append(
        ("Model fitted on", f"{meta['slate_train']:,} completed games")
    )
    if meta.get("slate_history_note"):
        rows.append(("Includes this season", meta["slate_history_note"]))
    return rows


def _build_team_data(sheet, team_data: pd.DataFrame, meta: dict) -> int:
    """Per-team reference figures. The Matchup Picker reads from here.

    Deliberately plain: a rating, a scoring rate, and a concession rate, in
    the units the picker's formulas use. Anyone can check the arithmetic on
    this sheet against the picker above it.

    Returns the last populated row so the picker can size its lookups.
    """
    row = _write_title(
        sheet,
        "Team Data",
        "What the Matchup Picker looks up. Elo is the model's power rating; "
        "the scoring columns are per-game averages over each team's last "
        f"{meta.get('team_window', 17)} games.",
    )
    _write_header(
        sheet, row,
        ["Team", "Elo", "Pts vs average", "Points for", "Points against", "Games"],
    )

    first = row + 1
    if team_data is None or team_data.empty:
        _note(sheet, first, "No team data available.")
        _set_widths(sheet, [10, 10, 15, 12, 15, 9])
        return first

    ordered = team_data.sort_values("team")
    for offset, team in enumerate(ordered.itertuples(index=False)):
        r = first + offset
        _value(sheet, r, 1, team.team)
        _value(sheet, r, 2, _num(getattr(team, "elo", None)), "0")
        _value(sheet, r, 3, _num(getattr(team, "spread_vs_average", None)), SPREAD)
        _value(sheet, r, 4, _num(getattr(team, "points_for", None)), "0.0")
        _value(sheet, r, 5, _num(getattr(team, "points_against", None)), "0.0")
        _value(sheet, r, 6, _num(getattr(team, "games", None)), "0")

    last = first + len(ordered) - 1
    _note(sheet, last + 2,
          f"{config.ELO_PER_POINT:.0f} Elo points = 1 point of spread. Points for "
          "and against are raw per-game averages, not opponent-adjusted -- the "
          "full model adjusts for schedule, the picker's quick estimate does not.")
    _set_widths(sheet, [10, 10, 15, 12, 15, 9])
    sheet.freeze_panes = f"A{first}"
    return last


def _build_matchup_picker(
    sheet, rec: Recorder, team_data: pd.DataFrame, slate: pd.DataFrame,
    meta: dict, data_last_row: int,
) -> None:
    """Pick any two teams and get a live projection.

    Every number below is an Excel formula over the Team Data sheet, so
    changing either dropdown recalculates the whole panel instantly -- no
    Python, no regeneration. Matchups that happen to be on this week's slate
    also show what the full model says, because the two will not agree and
    pretending otherwise would be the dishonest part of this sheet.
    """
    row = _write_title(
        sheet,
        "Matchup Picker",
        "Choose any two teams. Everything recalculates from the Team Data tab.",
    )

    if team_data is None or team_data.empty:
        _note(sheet, row, "No team data available.")
        return

    teams = sorted(team_data["team"].astype(str))
    default_away, default_home = (teams + ["", ""])[:2]
    if not slate.empty:
        first_game = slate.iloc[0]
        default_away = str(first_game["away_team"])
        default_home = str(first_game["home_team"])

    lookup = f"'Team Data'!$A${5}:$F${data_last_row}"
    away_cell, home_cell, neutral_cell = "B5", "B6", "B7"

    # --- inputs
    sheet.cell(row=4, column=1, value="Choose the matchup").font = _SECTION_FONT
    for r, label, value in (
        (5, "Away team", default_away),
        (6, "Home team", default_home),
        (7, "Neutral site?", "No"),
    ):
        _value(sheet, r, 1, label, font=_BOLD)
        cell = _value(sheet, r, 2, value, font=_INPUT_FONT)
        cell.fill = _ASSUMPTION_FILL

    validation = DataValidation(
        type="list", formula1=f"='Team Data'!$A$5:$A${data_last_row}", allow_blank=False
    )
    validation.error = "Pick a team from the list."
    validation.prompt = "Choose a team"
    sheet.add_data_validation(validation)
    validation.add(sheet[away_cell])
    validation.add(sheet[home_cell])

    neutral_validation = DataValidation(type="list", formula1='"No,Yes"', allow_blank=False)
    sheet.add_data_validation(neutral_validation)
    neutral_validation.add(sheet[neutral_cell])

    _note(sheet, 8, "Yellow cells are yours to change. Everything else is a formula.")

    # --- the projection
    row = 10
    sheet.cell(row=row, column=1, value="Projection").font = _SECTION_FONT
    row += 1
    _write_header(sheet, row, ["Measure", "Value", "How it is worked out"])
    row += 1

    hfa = float(meta.get("hfa_points", config.HFA_POINTS_DEFAULT))
    elo_per_point = config.ELO_PER_POINT
    sigma = config.MARGIN_SIGMA

    away_elo = f'VLOOKUP(${away_cell},{lookup},2,FALSE)'
    home_elo = f'VLOOKUP(${home_cell},{lookup},2,FALSE)'
    away_pf = f'VLOOKUP(${away_cell},{lookup},4,FALSE)'
    home_pf = f'VLOOKUP(${home_cell},{lookup},4,FALSE)'
    away_pa = f'VLOOKUP(${away_cell},{lookup},5,FALSE)'
    home_pa = f'VLOOKUP(${home_cell},{lookup},5,FALSE)'

    hfa_term = f'IF(${neutral_cell}="Yes",0,{hfa})'
    spread_formula = f"=({home_elo}-{away_elo})/{elo_per_point}+{hfa_term}"
    total_formula = f"=(({home_pf}+{away_pa})/2)+(({away_pf}+{home_pa})/2)"

    rows_spec = [
        ("Home spread", spread_formula, SPREAD,
         f"Elo gap / {elo_per_point:.0f}, plus {hfa:.2f} home field"),
        ("Home win probability", f"=NORM.DIST($B{row},0,{sigma},TRUE)", PCT,
         f"Normal curve on the spread, sigma {sigma}"),
        ("Away win probability", f"=1-$B{row + 1}", PCT, "The other side of it"),
        ("Home fair moneyline",
         f'=IF($B{row + 1}>=0.5,-100*$B{row + 1}/(1-$B{row + 1}),'
         f'100*(1-$B{row + 1})/$B{row + 1})', "+0;-0", "The price that probability implies"),
        ("Away fair moneyline",
         f'=IF($B{row + 2}>=0.5,-100*$B{row + 2}/(1-$B{row + 2}),'
         f'100*(1-$B{row + 2})/$B{row + 2})', "+0;-0", "Same, from the away side"),
        ("Game total", total_formula, "0.0",
         "Each side's scoring averaged with what the other allows"),
        ("Home team total", f"=($B{row + 5}+$B{row})/2", "0.0", "Total and spread split"),
        ("Away team total", f"=($B{row + 5}-$B{row})/2", "0.0", "The remainder"),
    ]

    values = _matchup_values(team_data, default_away, default_home, hfa)
    for offset, (label, formula, fmt, explanation) in enumerate(rows_spec):
        r = row + offset
        _value(sheet, r, 1, label)
        rec.formula(sheet, r, 2, formula, values[offset], fmt)
        _value(sheet, r, 3, explanation, font=_NOTE_FONT)

    last = row + len(rows_spec)

    # --- what the full model says, where the two teams actually meet
    last += 1
    sheet.cell(row=last, column=1, value="On this week's slate").font = _SECTION_FONT
    last += 1
    key_range = _slate_key_range(slate)
    if key_range:
        _write_header(sheet, last, ["Measure", "Full model", "Picker estimate"])
        last += 1
        matched = (
            f'IFERROR(VLOOKUP(${away_cell}&"@"&${home_cell},{key_range},'
        )
        # The cached value has to be the real lookup result, not a blank: a
        # reader who never recalculates would otherwise see an empty column
        # exactly where the comparison lives.
        on_slate = slate[
            (slate["away_team"] == default_away) & (slate["home_team"] == default_home)
        ]
        model_values = (
            [
                _num(on_slate["pred_margin"].iloc[0]),
                _num(on_slate["prob_home"].iloc[0]),
                _num(on_slate.get("pred_total", pd.Series([None])).iloc[0]),
            ]
            if not on_slate.empty
            else ["not on this slate"] * 3
        )
        for offset, (label, column, fmt, picker_row) in enumerate(
            (("Home spread", 2, SPREAD, row),
             ("Home win probability", 3, PCT, row + 1),
             ("Game total", 4, "0.0", row + 5)),
        ):
            r = last + offset
            _value(sheet, r, 1, label)
            rec.formula(
                sheet, r, 2, f'={matched}{column},FALSE),"not on this slate")',
                model_values[offset], fmt,
            )
            rec.formula(sheet, r, 3, f"=$B{picker_row}", values[picker_row - row], fmt)
        last += 3
    else:
        _note(sheet, last, "No slate loaded, so there is nothing to compare against.")
        last += 1

    _note(sheet, last + 1,
          "The picker and the full model will not agree, and that is expected. "
          "The picker is Elo plus raw scoring averages, which is all that fits in "
          "a spreadsheet formula. The full model adds EPA form, pace, weather, "
          "rest, the passer, injuries -- and then blends toward the market.")
    _note(sheet, last + 2,
          "Treat the picker as a quick what-if for matchups nobody has priced, "
          "and the Predictions tab as the actual forecast.")
    _set_widths(sheet, [24, 16, 46])


def _matchup_values(team_data: pd.DataFrame, away: str, home: str, hfa: float) -> list:
    """The picker's formulas, computed in Python for the cached values."""
    import numpy as np
    from scipy.stats import norm

    indexed = team_data.set_index("team")
    if away not in indexed.index or home not in indexed.index:
        return [None] * 8

    def field(team: str, column: str) -> float:
        value = indexed.loc[team].get(column)
        return float(value) if value is not None and not pd.isna(value) else float("nan")

    spread = (
        field(home, "elo") - field(away, "elo")
    ) / config.ELO_PER_POINT + hfa
    home_prob = float(norm.cdf(spread / config.MARGIN_SIGMA))
    away_prob = 1.0 - home_prob
    total = (
        (field(home, "points_for") + field(away, "points_against")) / 2.0
        + (field(away, "points_for") + field(home, "points_against")) / 2.0
    )

    def american(prob: float) -> float:
        if prob >= 0.5:
            return -100.0 * prob / (1.0 - prob)
        return 100.0 * (1.0 - prob) / prob

    return [
        spread, home_prob, away_prob, american(home_prob), american(away_prob),
        total, (total + spread) / 2.0, (total - spread) / 2.0,
    ]


def _slate_key_range(slate: pd.DataFrame) -> str:
    """The Predictions sheet's hidden lookup block, or '' when there is none."""
    if slate is None or slate.empty:
        return ""
    first = 5  # Predictions data starts here
    last = first + len(slate) - 1
    return f"Predictions!$Z${first}:$AC${last}"


def _add_back_link(sheet) -> None:
    """A way home from every tab, so the workbook is navigable in both directions."""
    cell = sheet.cell(row=1, column=column_index_from_string("H"), value="\u2190 Start Here")
    _internal_link(cell, "Start Here")
    cell.font = Font(name=FONT, size=9, color="0563C1", underline="single")


# --------------------------------------------------------------------------
# The simple workbook
# --------------------------------------------------------------------------
#
# Six tabs, in the order you would actually look at them, and nothing else.
# The full version keeps the backtest evidence, the calibration table and the
# game log; this one assumes you have read those once and now just want the
# week. Every honesty note is compressed to a line or two at the foot of the
# tab it applies to, rather than given a page of its own -- the claims still
# have to travel with the numbers, but they do not need to be the first thing
# you see every time.

SIMPLE_SHEETS = (
    "Predictions", "Spread", "Point Totals", "Player Projections",
    "Matchup Picker", "Power Ratings",
)


def _simple_title(sheet, title: str, meta: dict, scorecard) -> int:
    """Shared header: what week this is, and how the season is actually going."""
    sheet["A1"] = title
    sheet["A1"].font = _TITLE_FONT

    parts = [f"{meta['slate_season']} Week {meta['slate_week']}"]
    if scorecard is not None and not getattr(scorecard, "games", pd.DataFrame()).empty:
        summary = scorecard.summary
        parts.append(
            f"this season {int(summary['correct'])} of {summary['games']} "
            f"({summary['accuracy']:.0%}) straight up"
        )
    parts.append(f"rebuilt {meta['generated']}")
    sheet["A2"] = "  ·  ".join(parts)
    sheet["A2"].font = _NOTE_FONT
    return 4


def _build_simple_predictions(sheet, rec: Recorder, slate, meta, scorecard) -> None:
    """Who wins, how sure, and at what price."""
    row = _simple_title(sheet, "Predictions", meta, scorecard)
    _write_header(
        sheet, row,
        ["Away", "Home", "Pick", "Win %", "Confidence", "Fair odds",
         "Book odds", "My pick", "Result"],
    )

    first = row + 1
    for offset, game in enumerate(slate.itertuples(index=False)):
        r = first + offset
        prob = float(game.prob_home)
        picked_home = prob >= 0.5
        confidence = max(prob, 1.0 - prob)

        _value(sheet, r, 1, game.away_team)
        _value(sheet, r, 2, game.home_team)
        _value(sheet, r, 3, game.home_team if picked_home else game.away_team, font=_BOLD)
        _value(sheet, r, 4, confidence, PCT)
        rec.formula(
            sheet, r, 5,
            f'=IF($D{r}>=0.75,"HIGH",IF($D{r}>=0.65,"MEDIUM",'
            f'IF($D{r}>=0.575,"LEAN","COIN FLIP")))',
            _tier(confidence),
        )
        rec.formula(
            sheet, r, 6, f"=-100*$D{r}/(1-$D{r})",
            -100.0 * confidence / (1.0 - confidence) if confidence < 1.0 else None,
            "+0;-0",
        )
        _value(sheet, r, 7, _num(getattr(game, "pick_odds", None)), "+0;-0")

        own = _value(sheet, r, 8, None)
        own.font = _INPUT_FONT
        own.fill = _ASSUMPTION_FILL
        validation = DataValidation(
            type="list", formula1=f"=$A${r}:$B${r}", allow_blank=True
        )
        sheet.add_data_validation(validation)
        validation.add(own)

        margin = _num(getattr(game, "margin", None))
        outcome = ""
        if margin is not None:
            outcome = "PUSH" if margin == 0 else (
                "HIT" if (margin > 0) == picked_home else "MISS"
            )
        _value(sheet, r, 9, outcome)

    last = first + len(slate)
    _note(sheet, last + 1,
          "Right about two games in three (66.6% over 4,912 backtested games) -- "
          "level with the betting line, not better than it. Fair odds are what "
          "the model's probability is worth; Book odds are what is posted.")
    _note(sheet, last + 2,
          "Yellow cells are yours. Pick a side and Result grades it once the "
          "game lands.")
    _set_widths(sheet, [9, 9, 9, 9, 13, 11, 11, 11, 10])
    sheet.freeze_panes = f"A{first}"


def _build_simple_spread(sheet, rec: Recorder, slate, meta, scorecard) -> None:
    """The model's number against the posted one."""
    row = _simple_title(sheet, "Spread", meta, scorecard)
    _write_header(
        sheet, row,
        ["Away", "Home", "Line", "Model", "Edge", "Model likes"],
    )

    ordered = slate.sort_values("pred_margin", ascending=False)
    first = row + 1
    for offset, game in enumerate(ordered.itertuples(index=False)):
        r = first + offset
        line = _num(getattr(game, "market_spread", None))
        model = _num(getattr(game, "pred_margin", None))

        _value(sheet, r, 1, game.away_team)
        _value(sheet, r, 2, game.home_team)
        _value(sheet, r, 3, line, SPREAD)
        _value(sheet, r, 4, model, SPREAD)
        rec.formula(
            sheet, r, 5, f'=IF(OR($C{r}="",$D{r}=""),"",$D{r}-$C{r})',
            None if line is None or model is None else model - line, SPREAD,
        )
        rec.formula(
            sheet, r, 6, f'=IF($E{r}="","-",IF($E{r}>0,$B{r},$A{r}))',
            "-" if line is None or model is None else (
                game.home_team if model > line else game.away_team
            ),
        )

    last = first + len(ordered)
    _note(sheet, last + 1,
          "Both numbers are the home team's spread: positive means the home side "
          "is favoured. Edge is how far the model sits from the posted line.")
    _note(sheet, last + 2,
          "Against the spread the model wins 49.6% and loses money at standard "
          "-110 pricing, where 52.38% breaks even. A large edge is usually the "
          "model missing something the line already knows.")
    _set_widths(sheet, [9, 9, 10, 10, 10, 14])
    sheet.freeze_panes = f"A{first}"


def _build_simple_totals(sheet, rec: Recorder, slate, meta, scorecard) -> None:
    """How many points the game produces."""
    row = _simple_title(sheet, "Point Totals", meta, scorecard)
    _write_header(
        sheet, row,
        ["Away", "Home", "Line", "Model", "Projection", "Edge", "Lean", "P(over)"],
    )

    ordered = slate.sort_values("pred_total", ascending=False)
    first = row + 1
    for offset, game in enumerate(ordered.itertuples(index=False)):
        r = first + offset
        line = _num(getattr(game, "market_total", None))
        model = _num(getattr(game, "model_total", None))
        projection = _num(getattr(game, "pred_total", None))

        _value(sheet, r, 1, game.away_team)
        _value(sheet, r, 2, game.home_team)
        _value(sheet, r, 3, line, "0.0")
        _value(sheet, r, 4, model, "0.0")
        _value(sheet, r, 5, projection, "0.0")
        rec.formula(
            sheet, r, 6, f'=IF(OR($C{r}="",$D{r}=""),"",$D{r}-$C{r})',
            None if line is None or model is None else model - line, SPREAD,
        )
        rec.formula(
            sheet, r, 7, f'=IF(OR($C{r}="",$E{r}=""),"-",IF($E{r}>$C{r},"OVER","UNDER"))',
            "-" if line is None or projection is None else (
                "OVER" if projection > line else "UNDER"
            ),
        )
        _value(sheet, r, 8, _num(getattr(game, "prob_over", None)), PCT)

    last = first + len(ordered)
    _note(sheet, last + 1,
          "Model is the model's own total; Projection blends it with the posted "
          "line and is what it stands behind. Edge is Model minus Line -- most of "
          "it is deliberately discarded by that blend.")
    _note(sheet, last + 2,
          "The model misses the final total by 10.7 points on average, the posted "
          "total by 10.5. Over/under picks hit 50.8%, under the 52.38% needed to "
          "break even. P(over) sitting near 50% is the honest answer, not a bug.")
    _set_widths(sheet, [9, 9, 10, 10, 12, 10, 10, 10])
    sheet.freeze_panes = f"A{first}"


def _build_simple_props(sheet, props, meta, scorecard, *, top: int = 20) -> None:
    """Projected yards, one block per category."""
    row = _simple_title(sheet, "Player Projections", meta, scorecard)

    if props is None or props.empty:
        _note(sheet, row, "No player projections available for this slate.")
        _set_widths(sheet, [24, 8, 7, 12, 14])
        return

    for column, title in PROP_SECTIONS:
        if column not in props.columns or props[column].isna().all():
            continue
        ranked = props.nlargest(top, column)
        ranked = ranked[ranked[column] > 0]
        if ranked.empty:
            continue

        sheet.cell(row=row, column=1, value=title).font = _SECTION_FONT
        row += 1
        _write_header(sheet, row, ["Player", "Team", "Pos", "Yards", "Status"])
        row += 1
        for game in ranked.itertuples(index=False):
            availability = float(getattr(game, "availability", 1.0) or 1.0)
            _value(sheet, row, 1, str(getattr(game, "player_display_name", "")))
            _value(sheet, row, 2, str(getattr(game, "team", "")))
            _value(sheet, row, 3, str(getattr(game, "position", "")))
            _value(sheet, row, 4, _num(getattr(game, column)), "0.0")
            _value(
                sheet, row, 5,
                "" if availability >= 0.999 else f"listed ({availability:.0%})",
            )
            row += 1
        row += 1

    _note(sheet, row,
          "Projections miss by about 62 passing yards, 24 rushing and 23 "
          "receiving -- better than the player's own recent average, still a wide "
          "miss on any single game. No over/under is offered: yardage is "
          "long-tailed and a tidy percentage would overstate the confidence.")
    _note(sheet, row + 1,
          "Status marks a player on the injury report; his projection is already "
          "scaled by how often that status actually plays.")
    _set_widths(sheet, [24, 8, 7, 12, 14])


def _build_simple_ratings(sheet, ratings, meta, scorecard) -> None:
    """Every team, strongest first."""
    row = _simple_title(sheet, "Power Ratings", meta, scorecard)
    _write_header(sheet, row, ["#", "Team", "Rating", "Worth"])

    first = row + 1
    for offset, team in enumerate(ratings.itertuples(index=False)):
        r = first + offset
        _value(sheet, r, 1, offset + 1)
        _value(sheet, r, 2, team.team, font=_BOLD)
        _value(sheet, r, 3, _num(team.elo), "0")
        _value(sheet, r, 4, _num(team.spread_vs_average), SPREAD)

    last = first + len(ratings)
    _note(sheet, last + 1,
          f"Worth is what the rating is worth in points against an average team. "
          f"{config.ELO_PER_POINT:.0f} rating points = 1 point of spread.")
    _set_widths(sheet, [5, 10, 10, 10])
    sheet.freeze_panes = f"A{first}"


def _build_simple_picker(
    sheet, rec: Recorder, team_data, slate, meta, scorecard
) -> None:
    """Pick any two teams; everything recalculates.

    The team figures the formulas read live in hidden columns on this same
    sheet rather than a tab of their own. A lookup table is not something
    anyone wants to look at, and a workbook trimmed to six tabs should not
    spend one of them on plumbing.
    """
    row = _simple_title(sheet, "Matchup Picker", meta, scorecard)

    if team_data is None or team_data.empty:
        _note(sheet, row, "No team data available.")
        return

    ordered = team_data.sort_values("team")
    # --- hidden reference block, columns J-N
    data_first = 2
    for offset, team in enumerate(ordered.itertuples(index=False)):
        r = data_first + offset
        _value(sheet, r, 10, team.team)
        _value(sheet, r, 11, _num(getattr(team, "elo", None)))
        _value(sheet, r, 12, _num(getattr(team, "spread_vs_average", None)))
        _value(sheet, r, 13, _num(getattr(team, "points_for", None)))
        _value(sheet, r, 14, _num(getattr(team, "points_against", None)))
    data_last = data_first + len(ordered) - 1
    for column in ("J", "K", "L", "M", "N"):
        sheet.column_dimensions[column].hidden = True

    default_away, default_home = (sorted(ordered["team"].astype(str)) + ["", ""])[:2]
    if not slate.empty:
        first_game = slate.iloc[0]
        default_away = str(first_game["away_team"])
        default_home = str(first_game["home_team"])

    lookup = f"$J${data_first}:$N${data_last}"
    away_cell, home_cell, neutral_cell = "B5", "B6", "B7"

    for r, label, value in (
        (5, "Away team", default_away),
        (6, "Home team", default_home),
        (7, "Neutral site", "No"),
    ):
        _value(sheet, r, 1, label, font=_BOLD)
        cell = _value(sheet, r, 2, value, font=_INPUT_FONT)
        cell.fill = _ASSUMPTION_FILL

    teams_validation = DataValidation(
        type="list", formula1=f"=$J${data_first}:$J${data_last}", allow_blank=False
    )
    teams_validation.prompt = "Choose a team"
    sheet.add_data_validation(teams_validation)
    teams_validation.add(sheet[away_cell])
    teams_validation.add(sheet[home_cell])

    neutral_validation = DataValidation(type="list", formula1='"No,Yes"', allow_blank=False)
    sheet.add_data_validation(neutral_validation)
    neutral_validation.add(sheet[neutral_cell])

    _note(sheet, 8, "Change either yellow cell. Everything below updates.")

    hfa = float(meta.get("hfa_points", config.HFA_POINTS_DEFAULT))
    away_elo = f"VLOOKUP(${away_cell},{lookup},2,FALSE)"
    home_elo = f"VLOOKUP(${home_cell},{lookup},2,FALSE)"
    away_pf = f"VLOOKUP(${away_cell},{lookup},4,FALSE)"
    home_pf = f"VLOOKUP(${home_cell},{lookup},4,FALSE)"
    away_pa = f"VLOOKUP(${away_cell},{lookup},5,FALSE)"
    home_pa = f"VLOOKUP(${home_cell},{lookup},5,FALSE)"

    row = 10
    _write_header(sheet, row, ["", "Projection"])
    row += 1
    start = row

    rows_spec = [
        ("Home spread",
         f'=({home_elo}-{away_elo})/{config.ELO_PER_POINT}'
         f'+IF(${neutral_cell}="Yes",0,{hfa})', SPREAD),
        ("Home win probability", f"=NORM.DIST($B{start},0,{config.MARGIN_SIGMA},TRUE)", PCT),
        ("Away win probability", f"=1-$B{start + 1}", PCT),
        ("Home fair odds",
         f'=IF($B{start + 1}>=0.5,-100*$B{start + 1}/(1-$B{start + 1}),'
         f'100*(1-$B{start + 1})/$B{start + 1})', "+0;-0"),
        ("Away fair odds",
         f'=IF($B{start + 2}>=0.5,-100*$B{start + 2}/(1-$B{start + 2}),'
         f'100*(1-$B{start + 2})/$B{start + 2})', "+0;-0"),
        ("Game total",
         f"=(({home_pf}+{away_pa})/2)+(({away_pf}+{home_pa})/2)", "0.0"),
        ("Home team total", f"=($B{start + 5}+$B{start})/2", "0.0"),
        ("Away team total", f"=($B{start + 5}-$B{start})/2", "0.0"),
    ]

    values = _matchup_values(ordered, default_away, default_home, hfa)
    for offset, (label, formula, fmt) in enumerate(rows_spec):
        r = start + offset
        _value(sheet, r, 1, label)
        rec.formula(sheet, r, 2, formula, values[offset], fmt)

    last = start + len(rows_spec)
    _note(sheet, last + 1,
          "A quick estimate from the power ratings and each team's scoring, which "
          "is all a spreadsheet formula can hold. For games on this week's slate "
          "the Predictions tab is the real forecast -- it also weighs recent form, "
          "pace, weather, rest, the quarterback and injuries, then leans on the "
          "betting line. The two will not agree.")
    _set_widths(sheet, [22, 14])


def export_simple_workbook(
    path, *, slate: pd.DataFrame, ratings: pd.DataFrame, meta: dict,
    props: pd.DataFrame | None = None, scorecard=None,
    team_data: pd.DataFrame | None = None,
) -> Path:
    """Write the short workbook: six tabs, no index, no backtest pages."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    workbook.remove(workbook.active)
    workbook.calculation.fullCalcOnLoad = True

    rec = Recorder()
    sheets = {name: workbook.create_sheet(name) for name in SIMPLE_SHEETS}

    _build_simple_predictions(sheets["Predictions"], rec, slate, meta, scorecard)
    _build_simple_spread(sheets["Spread"], rec, slate, meta, scorecard)
    _build_simple_totals(sheets["Point Totals"], rec, slate, meta, scorecard)
    _build_simple_props(sheets["Player Projections"], props, meta, scorecard)
    _build_simple_picker(
        sheets["Matchup Picker"], rec, team_data, slate, meta, scorecard
    )
    _build_simple_ratings(sheets["Power Ratings"], ratings, meta, scorecard)

    workbook.active = 0
    workbook.save(path)
    _inject_cached_values(path, rec)
    export_simple_workbook.sheet_count = len(sheets)
    return path


def export_workbook(
    path, *, slate: pd.DataFrame, ratings: pd.DataFrame, result, meta: dict,
    props: pd.DataFrame | None = None, scorecard=None,
    team_data: pd.DataFrame | None = None,
) -> Path:
    """Write the full multi-sheet workbook to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    workbook.remove(workbook.active)
    # Excel recomputes every formula on open, so an injected cached value can
    # never go stale without the reader seeing the corrected number.
    workbook.calculation.fullCalcOnLoad = True

    rec = Recorder()
    sheets = {
        name: workbook.create_sheet(name)
        for name in (
            "Start Here", "Predictions", "Matchup Picker", "Point Totals",
            "Team Totals", "Player Projections", "Season Scorecard",
            "Power Ratings", "Team Data", "Backtest Summary", "Calibration",
            "Against the Spread", "Point Totals Backtest",
            "Accuracy by Season", "Game Log", "Read Me",
        )
    }

    last_row, derived = _build_game_log(sheets["Game Log"], rec, result.predictions)

    meta = dict(meta)
    meta["n_games"] = last_row - 1
    meta["log_loss"] = {
        "Model + market blend": result.summary["blended"].get("log_loss"),
        "Model alone (no market)": result.summary["model"].get("log_loss"),
        "Market (closing line)": result.summary["market"].get("log_loss"),
        "Elo only": result.summary["elo_only"].get("log_loss"),
    }

    _build_read_me(sheets["Read Me"], meta)
    _build_start_here(sheets["Start Here"], slate, meta, scorecard)
    _build_predictions(sheets["Predictions"], rec, slate, meta)
    data_last_row = _build_team_data(sheets["Team Data"], team_data, meta)
    _build_matchup_picker(
        sheets["Matchup Picker"], rec, team_data, slate, meta, data_last_row
    )
    _build_totals_slate(sheets["Point Totals"], rec, slate, meta)
    _build_team_totals(sheets["Team Totals"], rec, slate, meta)
    _build_props(sheets["Player Projections"], props, meta)
    _build_scorecard(sheets["Season Scorecard"], rec, scorecard, meta)
    _build_ratings(sheets["Power Ratings"], rec, ratings)
    _build_summary(sheets["Backtest Summary"], rec, last_row, derived, meta)
    _build_calibration(sheets["Calibration"], rec, last_row, derived)
    _build_ats(sheets["Against the Spread"], rec, last_row, derived)
    _build_totals_backtest(sheets["Point Totals Backtest"], rec, last_row, derived)
    _build_by_season(sheets["Accuracy by Season"], rec, last_row, derived)

    for name, sheet in sheets.items():
        if name != "Start Here":
            _add_back_link(sheet)
    workbook.active = workbook.index(sheets["Start Here"])

    workbook.save(path)
    _inject_cached_values(path, rec)
    export_workbook.sheet_count = len(sheets)
    return path
