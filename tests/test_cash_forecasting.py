"""Cash forecasting: buffer computation and coverage semantics."""

from __future__ import annotations

import pandas as pd

from retail_ops_forecasting.cash_forecasting import compute_buffers, recommend_cash


def test_buffers_per_store():
    df = pd.DataFrame(
        {
            "store_id": ["A"] * 10 + ["B"] * 10,
            "amount_cash": [100.0] * 20,
            "y_pred": [90.0] * 10 + [80.0] * 10,
        }
    )
    buf = compute_buffers(df, "amount_cash", "y_pred", quantile=0.9)
    assert set(buf.index) == {"A", "B"}
    # Store A absolute error is 10, B is 20 — both constant, so the P90 equals each
    assert abs(buf["A"] - 10.0) < 1e-9
    assert abs(buf["B"] - 20.0) < 1e-9


def test_recommend_cash_adds_buffer():
    preds = pd.DataFrame(
        {
            "store_id": ["A", "B"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "amount_cash": [100.0, 200.0],
            "y_pred": [95.0, 195.0],
        }
    )
    buf = pd.Series({"A": 10.0, "B": 20.0})
    out = recommend_cash(preds, buf)
    assert (out.loc[out["store_id"] == "A", "recommended_cash"].iloc[0]) == 105.0
    assert (out.loc[out["store_id"] == "B", "recommended_cash"].iloc[0]) == 215.0
