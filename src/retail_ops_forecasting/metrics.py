"""Evaluation metrics for demand and cash forecasting.

Primary: WAPE (Weighted Absolute Percentage Error) — robust to zeros,
interpretable, and aligned with retail planning conventions.
Secondary: MAE, RMSE, sMAPE, Bias.

All functions accept arrays/Series and ignore NaN pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def _align(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return y_true[mask], y_pred[mask]


def wape(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    denom = np.abs(yt).sum()
    if denom < EPS:
        return float("nan")
    return float(np.abs(yt - yp).sum() / denom)


def mae(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def smape(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    denom = np.abs(yt) + np.abs(yp)
    mask = denom > EPS
    if not mask.any():
        return float("nan")
    return float(np.mean(2 * np.abs(yt[mask] - yp[mask]) / denom[mask]))


def bias(y_true, y_pred) -> float:
    """Mean signed error (positive = under-forecast)."""
    yt, yp = _align(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(yt - yp))


def coverage_at_threshold(y_true, recommended, threshold: float = 1.0) -> float:
    """Service-level proxy: fraction of days where recommended >= threshold * actual."""
    yt, yr = _align(y_true, recommended)
    if yt.size == 0:
        return float("nan")
    return float((yr >= threshold * yt).mean())


def summarize(y_true, y_pred) -> dict:
    return {
        "wape": wape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "n": int(
            np.sum(~(np.isnan(np.asarray(y_true, float)) | np.isnan(np.asarray(y_pred, float))))
        ),
    }


def grouped_metrics(
    df: pd.DataFrame, y_true_col: str, y_pred_col: str, group_cols: list[str]
) -> pd.DataFrame:
    rows = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, sub in df.groupby(grouper, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        m = summarize(sub[y_true_col], sub[y_pred_col])
        row = dict(zip(group_cols, keys))
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
