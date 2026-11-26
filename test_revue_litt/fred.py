"""US capacity utilisation in primary metals, as published at each date."""

from urllib.request import urlopen

import pandas as pd

SERIES = "CAPUTLG331S"
URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"


def download_capacity(start, end):
    """Download the series as it was published at each month end.

    The result has one row per month and one column per publication date.
    """
    dates = pd.date_range(start, end, freq="ME").strftime("%Y-%m-%d")
    tables = []
    for first in range(0, len(dates), 12):
        chunk = dates[first : first + 12]
        ids = ",".join([SERIES] * len(chunk))
        url = f"{URL}?id={ids}&vintage_date={','.join(chunk)}"
        with urlopen(url, timeout=90) as response:
            tables.append(pd.read_csv(response, index_col=0, na_values="."))
    vintages = pd.concat(tables, axis=1)
    vintages.columns = vintages.columns.str[-8:]
    return vintages


def load_capacity(path, start, end):
    """Read the local file, or download it the first time."""
    if not path.exists():
        download_capacity(start, end).to_csv(path)
    vintages = pd.read_csv(path, index_col=0, parse_dates=True)
    vintages.columns = pd.to_datetime(vintages.columns, format="%Y%m%d")
    return vintages


def capacity_features(vintages):
    """Return the last published level and its 3-month change for each month."""
    rows = {}
    for date, values in vintages.items():
        published = values.dropna()
        rows[date.replace(day=1)] = {
            "capacity": published.iloc[-1],
            "capacity_change": published.iloc[-1] - published.iloc[-4],
        }
    return pd.DataFrame.from_dict(rows, orient="index")
