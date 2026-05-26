"""Data loaders + canonical merge.

Loaders accept either a Config object or explicit paths. They normalize
dtypes (parse dates, coerce booleans) and validate the expected schema.
The merge is left-join transactions onto calendar + stores so every row
keeps its temporal/operational context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Config

_TRANSACTION_DTYPES = {
    "store_id": "string",
    "category": "string",
    "total_transactions": "Int64",
    "cash_transactions": "Int64",
    "card_transactions": "Int64",
    "amount_total": "float64",
    "amount_cash": "float64",
    "amount_card": "float64",
    "units_sold": "float64",
    "avg_ticket": "float64",
    "has_promotion": "Int8",
    "replenishment_signal": "float64",
}

_STORE_BOOL_COLS = ("has_pharmacy", "has_fuel_station")
_CAL_BOOL_COLS = (
    "is_holiday",
    "is_payday",
    "is_weekend",
    "is_navidad_season",
    "is_buen_fin",
    "is_semana_santa",
)


def _coerce_bools(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns and df[c].dtype != bool:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
                .astype("boolean")
            )
    return df


def load_transactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    for col, dtype in _TRANSACTION_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df.sort_values(["store_id", "category", "date"]).reset_index(drop=True)


def load_stores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["store_id"] = df["store_id"].astype("string")
    df["store_format"] = df["store_format"].astype("category")
    df["region"] = df["region"].astype("category")
    df["socioeconomic_level"] = df["socioeconomic_level"].astype("category")
    df = _coerce_bools(df, _STORE_BOOL_COLS)
    return df


def load_calendar(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = _coerce_bools(df, _CAL_BOOL_COLS)
    if "season" in df.columns:
        df["season"] = df["season"].astype("category")
    if "day_name" in df.columns:
        df["day_name"] = df["day_name"].astype("category")
    df["is_event"] = (
        df["is_holiday"].fillna(False).astype(bool)
        | df["is_buen_fin"].fillna(False).astype(bool)
        | df["is_semana_santa"].fillna(False).astype(bool)
        | df["is_navidad_season"].fillna(False).astype(bool)
    )
    return df.sort_values("date").reset_index(drop=True)


def build_merged(cfg: Config) -> pd.DataFrame:
    """Return transactions enriched with store metadata and calendar flags."""
    tx = load_transactions(cfg.paths.raw_dir / cfg.data.transactions_file)
    st = load_stores(cfg.paths.raw_dir / cfg.data.stores_file)
    cal = load_calendar(cfg.paths.raw_dir / cfg.data.calendar_file)
    df = tx.merge(st, on="store_id", how="left")
    df = df.merge(cal, on="date", how="left")
    return df


def coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts + percentages."""
    n = len(df)
    miss = df.isna().sum()
    return pd.DataFrame(
        {
            "column": miss.index,
            "n_missing": miss.values,
            "pct_missing": (miss.values / n * 100).round(3),
        }
    ).sort_values("n_missing", ascending=False, ignore_index=True)


def cash_store_day(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the merged frame to store-day for the cash forecasting task.

    Only sums columns that exist; preserves calendar/store metadata via first().
    """
    sum_cols = [
        c
        for c in [
            "total_transactions",
            "cash_transactions",
            "card_transactions",
            "amount_total",
            "amount_cash",
            "amount_card",
            "units_sold",
        ]
        if c in df.columns
    ]
    keep_first = [
        c
        for c in [
            "store_format",
            "region",
            "size_sqm",
            "num_checkouts",
            "opening_year",
            "socioeconomic_level",
            "has_pharmacy",
            "has_fuel_station",
            "day_of_week",
            "day_name",
            "week_of_year",
            "month",
            "year",
            "quarter",
            "season",
            "is_holiday",
            "holiday_name",
            "is_payday",
            "is_weekend",
            "is_navidad_season",
            "is_buen_fin",
            "is_semana_santa",
            "is_event",
        ]
        if c in df.columns
    ]
    grouped = df.groupby(["store_id", "date"], as_index=False).agg(
        {**{c: "sum" for c in sum_cols}, **{c: "first" for c in keep_first}}
    )
    # Cash columns containing NaNs must remain NaN after sum if ALL inputs are NaN
    # pandas default sum() returns 0 for all-NaN; restore NaN where appropriate.
    for cash_col in ("amount_cash", "cash_transactions"):
        if cash_col in grouped.columns:
            all_nan = (
                df.groupby(["store_id", "date"])[cash_col]
                .apply(lambda s: s.isna().all())
                .reset_index(name="_all_nan")
            )
            grouped = grouped.merge(all_nan, on=["store_id", "date"], how="left")
            grouped.loc[grouped["_all_nan"], cash_col] = np.nan
            grouped = grouped.drop(columns="_all_nan")
    return grouped.sort_values(["store_id", "date"]).reset_index(drop=True)
