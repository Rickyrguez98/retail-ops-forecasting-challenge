"""Demand forecasting pipeline.

Uses HistGradientBoostingRegressor from sklearn. Categoricals are encoded
to integer codes upstream (see features.encode_categoricals). The same
function trains, predicts, and returns the model artifact so MLflow logging
stays in the caller.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class FitResult:
    model: HistGradientBoostingRegressor
    feature_cols: list[str]


def fit_hgb(
    train_df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
    params: dict,
) -> FitResult:
    feature_cols = list(feature_cols)
    X = (
        train_df[feature_cols].to_numpy(dtype=np.float32, copy=False, na_value=np.nan)
        if hasattr(pd.DataFrame, "to_numpy")
        else train_df[feature_cols].values
    )
    # to_numpy doesn't support na_value before pandas 2.0; do it manually
    X = train_df[feature_cols].astype(np.float32).values
    y = train_df[target_col].astype(np.float32).values
    model = HistGradientBoostingRegressor(**params)
    model.fit(X, y)
    return FitResult(model=model, feature_cols=feature_cols)


def predict(fit: FitResult, df: pd.DataFrame) -> np.ndarray:
    X = df[fit.feature_cols].astype(np.float32).values
    return fit.model.predict(X)


def assemble_predictions(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred: np.ndarray,
    id_cols: tuple[str, ...] = ("store_id", "category", "date"),
) -> pd.DataFrame:
    out = df[list(id_cols) + [y_true_col]].copy()
    out["y_pred"] = y_pred
    out["y_pred"] = out["y_pred"].clip(lower=0)  # demand cannot be negative
    return out
