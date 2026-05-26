"""Feature engineering for demand and cash forecasting.

The contract: features are built per (entity × date) where entity is
(store, category) for demand and store for cash. *No future information*
ever crosses the row boundary. Lag/rolling features are computed with
`shift(>=1)` and groupwise to avoid bleeding across stores.

The `leakage_blocklist` from config is applied at the end as a safety net.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

CAT_COL_DEFAULT = ("store_format", "region", "socioeconomic_level", "category", "season", "day_name")


def add_calendar_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap event/calendar derivations that don't require history."""
    out = df.copy()
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12)
    out["is_payday_x_weekend"] = (
        out["is_payday"].fillna(False).astype(int) * out["is_weekend"].fillna(False).astype(int)
    )
    out["store_age_years"] = (out["year"] - out["opening_year"]).clip(lower=0)
    return out


def _group_lag(s: pd.Series, group: pd.Series, lag: int) -> pd.Series:
    return s.groupby(group).shift(lag)


def _group_rolling_mean(s: pd.Series, group: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    return (
        s.groupby(group)
        .shift(1)  # critical: shift before rolling — no same-day leak
        .groupby(group)
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )


def _group_rolling_std(s: pd.Series, group: pd.Series, window: int, min_periods: int = 2) -> pd.Series:
    return (
        s.groupby(group)
        .shift(1)
        .groupby(group)
        .rolling(window=window, min_periods=min_periods)
        .std()
        .reset_index(level=0, drop=True)
    )


def build_demand_features(
    df: pd.DataFrame, target_col: str = "total_transactions", entity_cols: Iterable[str] = ("store_id", "category")
) -> pd.DataFrame:
    """Build features for demand forecasting at store-category-day granularity.

    Adds lag (1,7,14,28), rolling mean (7,28), rolling std (28). Returns a
    new dataframe; the caller is responsible for dropping rows with NA in
    feature columns before training.
    """
    out = add_calendar_interactions(df)
    out = out.sort_values(list(entity_cols) + ["date"]).reset_index(drop=True)
    group = out[list(entity_cols)].astype(str).agg("|".join, axis=1)
    for lag in (1, 7, 14, 28):
        out[f"lag_{lag}"] = _group_lag(out[target_col], group, lag)
    out["rmean_7"] = _group_rolling_mean(out[target_col], group, 7)
    out["rmean_28"] = _group_rolling_mean(out[target_col], group, 28)
    out["rstd_28"] = _group_rolling_std(out[target_col], group, 28)
    # Last-week same-DOW value (lag_7 is the same — kept for clarity downstream)
    out["same_dow_lag_7"] = out["lag_7"]
    # Trend proxy: this week's mean minus prior week's mean (both shifted >=1)
    out["trend_7v28"] = out["rmean_7"] - out["rmean_28"]
    return out


def build_cash_features(
    df_store_day: pd.DataFrame,
    target_col: str = "amount_cash",
    aux_col: str | None = "total_transactions",
) -> pd.DataFrame:
    """Build features for cash forecasting at store-day granularity.

    Includes lag (1,7,14,28) and rolling stats of cash; if `aux_col` provided,
    its lagged values are also added (e.g., yesterday's total_transactions).
    """
    out = add_calendar_interactions(df_store_day)
    out = out.sort_values(["store_id", "date"]).reset_index(drop=True)
    group = out["store_id"].astype(str)
    for lag in (1, 7, 14, 28):
        out[f"cash_lag_{lag}"] = _group_lag(out[target_col], group, lag)
    out["cash_rmean_7"] = _group_rolling_mean(out[target_col], group, 7)
    out["cash_rmean_28"] = _group_rolling_mean(out[target_col], group, 28)
    out["cash_rstd_28"] = _group_rolling_std(out[target_col], group, 28)
    if aux_col and aux_col in out.columns:
        for lag in (1, 7):
            out[f"{aux_col}_lag_{lag}"] = _group_lag(out[aux_col], group, lag)
        out[f"{aux_col}_rmean_7"] = _group_rolling_mean(out[aux_col], group, 7)
    # Cash share lagged: how much of yesterday's amount was cash?
    if "amount_total" in out.columns:
        amt_lag1 = _group_lag(out["amount_total"], group, 1)
        cash_lag1 = out["cash_lag_1"]
        out["cash_share_lag_1"] = cash_lag1 / amt_lag1.replace(0, np.nan)
    return out


def select_feature_columns(
    df: pd.DataFrame, target_col: str, leakage_blocklist: Iterable[str], extra_drop: Iterable[str] = ()
) -> List[str]:
    """Return predictor columns, excluding identifiers, the target, and leakage cols."""
    base_drop = {
        target_col,
        "date",
        "store_id",
        "holiday_name",
        "_all_nan",
        "category_name",  # string companion of the encoded `category` column
    }
    base_drop.update(leakage_blocklist)
    base_drop.update(extra_drop)
    cols = [c for c in df.columns if c not in base_drop]
    return cols


def encode_categoricals(df: pd.DataFrame, cat_cols: Iterable[str] = CAT_COL_DEFAULT) -> pd.DataFrame:
    """Map known categorical columns to integer codes, leaving unknowns as -1.

    Encode string categoricals here. Encoding is fit on the union of unique values
    seen in the input frame; that is fine for our static categoricals
    (formats/regions/categories don't change in test).

    Also creates a `store_code` column (integer encoding of `store_id`)
    that downstream pipelines can use as a feature; we keep `store_id`
    untouched so the dataframe remains joinable.
    """
    out = df.copy()
    # Preserve original string for `category` (used in reports and breakdowns).
    if "category" in out.columns and "category_name" not in out.columns:
        out["category_name"] = out["category"].astype(str)
    for c in cat_cols:
        if c in out.columns:
            cat = out[c].astype("category")
            out[c] = cat.cat.codes.astype("int32")
    if "store_id" in out.columns and "store_code" not in out.columns:
        out["store_code"] = out["store_id"].astype("category").cat.codes.astype("int32")
    return out
