"""Cash forecasting pipeline + safety buffer recommendation.

Two-stage workflow:
  1. Forecast amount_cash at store-day granularity (HGB).
  2. Compute a per-store P-quantile safety buffer from validation residuals.
     recommended_cash = max(0, predicted) + buffer_q(store)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class CashFit:
    model: HistGradientBoostingRegressor
    feature_cols: list[str]
    buffers: pd.Series  # indexed by store_id, scalar per store


def fit_cash_model(
    train_df: pd.DataFrame, feature_cols: Iterable[str], target_col: str, params: dict
) -> HistGradientBoostingRegressor:
    feature_cols = list(feature_cols)
    X = train_df[feature_cols].astype(np.float32).values
    y = train_df[target_col].astype(np.float32).values
    model = HistGradientBoostingRegressor(**params)
    model.fit(X, y)
    return model


def predict_cash(
    model: HistGradientBoostingRegressor, df: pd.DataFrame, feature_cols: list[str]
) -> np.ndarray:
    return np.clip(model.predict(df[feature_cols].astype(np.float32).values), a_min=0, a_max=None)


def compute_buffers(
    val_df: pd.DataFrame, target_col: str, pred_col: str, quantile: float = 0.90
) -> pd.Series:
    """Per-store quantile of absolute residuals on validation."""
    res = (val_df[target_col] - val_df[pred_col]).abs()
    out = (
        pd.DataFrame({"store_id": val_df["store_id"].values, "abs_err": res.values})
        .dropna()
        .groupby("store_id")["abs_err"]
        .quantile(quantile)
    )
    return out


def recommend_cash(
    predictions: pd.DataFrame, buffers: pd.Series, pred_col: str = "y_pred"
) -> pd.DataFrame:
    """Add a `recommended_cash` column = clipped prediction + per-store buffer."""
    out = predictions.copy()
    out["buffer"] = (
        out["store_id"].map(buffers).fillna(buffers.median() if not buffers.empty else 0.0)
    )
    out["recommended_cash"] = out[pred_col].clip(lower=0) + out["buffer"]
    return out
