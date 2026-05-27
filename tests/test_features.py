"""Anti-leakage tests for feature engineering.

These are the *critical* tests: a regression here would silently inflate
validation metrics. They assert two invariants:
  1. Lag and rolling features never use information from the current row.
  2. select_feature_columns excludes everything in the leakage blocklist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retail_ops_forecasting.features import (
    add_calendar_interactions,
    build_cash_features,
    build_demand_features,
    select_feature_columns,
)


def _toy(toy_transactions, toy_calendar, toy_stores):
    df = toy_transactions.merge(toy_stores, on="store_id", how="left")
    df = df.merge(toy_calendar, on="date", how="left")
    return df


def test_lag_features_do_not_use_same_day_target(toy_transactions, toy_calendar, toy_stores):
    df = _toy(toy_transactions, toy_calendar, toy_stores)
    feats = build_demand_features(df, target_col="total_transactions")
    # For each (store, category) the first 28 rows of lag_28 must be NaN
    sample = (
        feats.sort_values(["store_id", "category", "date"])
        .groupby(["store_id", "category"])
        .head(1)
    )
    assert sample["lag_1"].isna().all()
    assert sample["lag_7"].isna().all()
    assert sample["lag_28"].isna().all()
    # And no exact equality with the same-day target (would imply leakage)
    valid = feats.dropna(subset=["lag_1"])
    assert not (valid["lag_1"] == valid["total_transactions"]).all()


def test_rolling_mean_uses_only_past(toy_transactions, toy_calendar, toy_stores):
    df = _toy(toy_transactions, toy_calendar, toy_stores)
    feats = build_demand_features(df, target_col="total_transactions")
    # Manual check on a single series
    sub = feats[(feats["store_id"] == "STR_001") & (feats["category"] == "Abarrotes")].sort_values(
        "date"
    )
    target = sub["total_transactions"].values
    rmean_7 = sub["rmean_7"].values
    # rmean_7[i] should equal mean(target[max(0, i-7):i])  (note: i excluded)
    for i in range(8, len(sub)):
        expected = float(np.nanmean(target[i - 7 : i]))
        if not np.isnan(rmean_7[i]):
            assert abs(rmean_7[i] - expected) < 1e-6, f"row {i}: {rmean_7[i]} vs {expected}"


def test_select_feature_columns_drops_blocklist(toy_transactions, toy_calendar, toy_stores):
    df = _toy(toy_transactions, toy_calendar, toy_stores)
    feats = build_demand_features(df, target_col="total_transactions")
    blocklist = [
        "replenishment_signal",
        "cash_transactions",
        "amount_cash",
        "amount_card",
        "card_transactions",
        "amount_total",
        "units_sold",
        "avg_ticket",
    ]
    cols = select_feature_columns(
        feats, target_col="total_transactions", leakage_blocklist=blocklist
    )
    for b in blocklist:
        assert b not in cols, f"{b} leaked into feature columns"
    assert "total_transactions" not in cols
    assert "date" not in cols
    assert "store_id" not in cols


def test_cash_lag_no_same_day(toy_transactions, toy_calendar, toy_stores):
    df = _toy(toy_transactions, toy_calendar, toy_stores)
    # Aggregate to store-day for cash
    sd = df.groupby(["store_id", "date"], as_index=False).agg(
        {
            "amount_cash": "sum",
            "amount_total": "sum",
            "total_transactions": "sum",
            "day_of_week": "first",
            "month": "first",
            "year": "first",
            "opening_year": "first",
            "is_payday": "first",
            "is_weekend": "first",
        }
    )
    feats = build_cash_features(sd, target_col="amount_cash")
    first_rows = feats.sort_values(["store_id", "date"]).groupby("store_id").head(1)
    assert first_rows["cash_lag_1"].isna().all()
    assert first_rows["cash_lag_28"].isna().all()


def test_calendar_interactions_deterministic(toy_transactions, toy_calendar, toy_stores):
    df = _toy(toy_transactions, toy_calendar, toy_stores)
    a = add_calendar_interactions(df)
    b = add_calendar_interactions(df)
    pd.testing.assert_frame_equal(a, b)
