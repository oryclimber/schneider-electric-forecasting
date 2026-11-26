"""Supplier counterparty-risk scorecard from the PitchBook balance-sheet files.

Scores each of the nine suppliers on five simple indicators and gives a
low / medium / high rating. Counterparty risk is kept separate from the price
forecast: it feeds offer terms (validity, indexation, extra buffer), not prices.

Indicators (latest fiscal year with data; each scores 0 = low, 1 = medium,
2 = high risk; the rating is the mean of the indicators that are available):
- Leverage: total liabilities / equity (<2 low, 2-4 medium, >4 high). Liabilities
  rather than reported debt, because several files report debt as 0 despite large
  liabilities (Green Transfo).
- Liquidity: current ratio (>=1.5 low, 1.0-1.5 medium, <1.0 high).
- Profit level: EBITDA / total assets (>=8% low, 2-8% medium, <2% high).
- Profit trend: latest EBITDA vs the mean of up to 3 prior years
  (>=-10% low, -10% to -40% medium, <-40% or a loss high).
- Interest cover: EBITDA / interest expense (>=4 low, 1.5-4 medium, <1.5 high).
Rating: mean score < 0.67 low, < 1.33 medium, otherwise high.

The workbooks do not say what each supplier sells to Schneider, so this script
does NOT map suppliers to materials; see data/submission/supplier_material_map.csv.

Usage:
    uv run --locked python scripts/supplier_risk.py
"""

from __future__ import annotations

import glob

import numpy as np
import openpyxl
import pandas as pd

from schneider_electric_forecasting import config

SUPPLIER_DIR = config.DATA_DIR / "Supplier_balancesheet"
LABELS = {
    "ebitda": "EBITDA",
    "net_income": "Net Income Available to Common Stockholders",
    "interest": "Interest Expense Net of Capitalized Interest",
    "cur_assets": "Total Current Assets",
    "cur_liab": "Total Current Liabilities",
    "assets": "Total Assets",
    "liab": "Total Liabilities",
    "equity": "Total Equity",
    "debt": "Total Debt",
}


def _load(path) -> dict[str, dict[int, float]]:
    """Return {row label: {fiscal year: value}} across the statement sheets."""
    out: dict[str, dict[int, float]] = {}
    wb = openpyxl.load_workbook(path, data_only=True)
    for sheet in wb.sheetnames:
        if sheet in ("Overview", "Disclaimer"):
            continue
        rows = list(wb[sheet].iter_rows(values_only=True))
        header = rows[5]
        cols = {j: int(str(c)[-4:]) for j, c in enumerate(header) if j >= 12 and c}
        for row in rows[10:]:
            if not isinstance(row[2], str):
                continue
            label = row[2].strip()
            for j, year in cols.items():
                if isinstance(row[j], int | float):
                    out.setdefault(label, {}).setdefault(year, float(row[j]))
    return out


def _series(data, key):
    return pd.Series(data.get(LABELS[key], {}), dtype=float).sort_index()


def _score(value, low, high, higher_is_worse):
    if pd.isna(value):
        return np.nan
    if not higher_is_worse:
        value, low, high = -value, -high, -low
    return 0 if value < low else (1 if value <= high else 2)


def score_supplier(path) -> dict:
    data = _load(path)
    s = {k: _series(data, k) for k in LABELS}
    bs = pd.concat(
        [s["cur_assets"], s["cur_liab"], s["assets"], s["liab"], s["equity"]], axis=1
    )
    bs.columns = ["ca", "cl", "assets", "liab", "equity"]
    bs = bs.dropna(subset=["assets", "equity"]).sort_index()
    row: dict = {
        "supplier": path.name.split("_-_")[1].split("_Company")[0].replace("_", " ")
    }
    if bs.empty:
        return row
    year = int(bs.index[-1])
    last = bs.loc[year]
    liab = last.liab if pd.notna(last.liab) else last.assets - last.equity
    row["bs_year"] = year
    row["leverage_liab_to_equity"] = liab / last.equity if last.equity > 0 else np.inf
    row["current_ratio"] = (
        last.ca / last.cl if pd.notna(last.ca) and last.cl else np.nan
    )
    row["equity_ratio"] = last.equity / last.assets
    ebitda = s["ebitda"].dropna()
    if len(ebitda):
        e_year = int(ebitda.index[-1])
        prior = ebitda[ebitda.index < e_year].tail(3)
        row["pl_year"] = e_year
        assets_then = bs.assets.get(e_year, last.assets)
        row["ebitda_to_assets"] = ebitda.iloc[-1] / assets_then
        interest = s["interest"].get(e_year, np.nan)
        row["ebitda_interest_cover"] = (
            ebitda.iloc[-1] / interest if interest else np.nan
        )
        if len(prior) and prior.mean() > 0:
            row["ebitda_trend_vs_prior3"] = ebitda.iloc[-1] / prior.mean() - 1
        else:
            row["ebitda_trend_vs_prior3"] = np.nan
        row["ebitda_negative"] = bool(ebitda.iloc[-1] < 0)
    return row


def rate(table: pd.DataFrame) -> pd.DataFrame:
    scores = pd.DataFrame(index=table.index)
    # ASSUMPTION: all cut-offs below (and the equal indicator weights) are judgment
    # values, not calibrated to defaults; see the module docstring.
    scores["leverage"] = table.leverage_liab_to_equity.map(
        lambda v: _score(v, 2, 4, True)
    )
    scores["liquidity"] = table.current_ratio.map(lambda v: _score(v, 1.0, 1.5, False))
    scores["profit_level"] = table.ebitda_to_assets.map(
        lambda v: _score(v, 0.02, 0.08, False)
    )
    trend = table.ebitda_trend_vs_prior3.map(lambda v: _score(v, -0.40, -0.10, False))
    scores["profit_trend"] = trend.where(
        ~table.ebitda_negative.fillna(False).astype(bool), 2
    )
    scores["interest_cover"] = table.ebitda_interest_cover.map(
        lambda v: _score(v, 1.5, 4, False)
    )
    table = table.join(scores.add_prefix("score_"))
    table["risk_score"] = scores.mean(axis=1)
    table["indicators_used"] = scores.notna().sum(axis=1)
    # ASSUMPTION: rating bands at 0.67 / 1.33 of the mean 0-2 score.
    table["rating"] = pd.cut(
        table.risk_score, [-1, 0.67, 1.33, 3], labels=["low", "medium", "high"]
    ).astype(str)
    return table


def build() -> pd.DataFrame:
    files = sorted(glob.glob(str(SUPPLIER_DIR / "*.xlsx")))
    from pathlib import Path

    table = pd.DataFrame([score_supplier(Path(f)) for f in files]).set_index("supplier")
    return rate(table)


def main():
    out = config.DATA_DIR / "submission"
    out.mkdir(parents=True, exist_ok=True)
    table = build()
    table.to_csv(out / "supplier_risk_scorecard.csv")
    with pd.option_context(
        "display.width",
        250,
        "display.max_columns",
        30,
        "display.float_format",
        "{:.2f}".format,
    ):
        print(
            table.drop(columns=[c for c in table if c.startswith("score_")]).to_string()
        )
        print(table.filter(like="score_").assign(rating=table.rating).to_string())


if __name__ == "__main__":
    main()
