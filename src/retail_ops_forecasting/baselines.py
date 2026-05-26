"""Simple baselines: naive lag-7, historical day-of-week mean, rolling mean."""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_lag(df: pd.DataFrame, target: str, lag: int, entity_cols: list[str]) -> pd.Series:
    g = df[entity_cols].astype(str).agg("|".join, axis=1)
    return df[target].groupby(g).shift(lag)


def rolling_mean_lag(
    df: pd.DataFrame, target: str, window: int, entity_cols: list[str]
) -> pd.Series:
    g = df[entity_cols].astype(str).agg("|".join, axis=1)
    return (
        df[target]
        .groupby(g)
        .shift(1)
        .groupby(g)
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )


def seasonal_naive_dow(df: pd.DataFrame, target: str, entity_cols: list[str]) -> pd.Series:
    """Last-week same DOW — identical to lag-7 but named for reporting clarity."""
    return naive_lag(df, target, lag=7, entity_cols=entity_cols)


def historical_dow_mean(
    train_df: pd.DataFrame, target: str, entity_cols: list[str]
) -> pd.DataFrame:
    """Returns a lookup table: (entity..., day_of_week) -> mean(target)."""
    keys = list(entity_cols) + ["day_of_week"]
    return train_df.groupby(keys, as_index=False)[target].mean().rename(columns={target: f"hist_dow_{target}"})


def apply_historical_dow_mean(
    df: pd.DataFrame, lookup: pd.DataFrame, target: str, entity_cols: list[str]
) -> pd.Series:
    keys = list(entity_cols) + ["day_of_week"]
    merged = df.merge(lookup, on=keys, how="left")
    return merged[f"hist_dow_{target}"]
