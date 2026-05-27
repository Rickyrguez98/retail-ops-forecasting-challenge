"""Time-based train/val/test splits.

The function returns three pandas Index objects (not slices) so callers can
align with arbitrary frame orderings. A separate `walk_forward_folds` helper
builds expanding-window folds inside the train window for hyperparameter
selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import SplitsCfg


@dataclass(frozen=True)
class SplitIndex:
    train: pd.Index
    val: pd.Index
    test: pd.Index


def time_based_split(df: pd.DataFrame, splits: SplitsCfg, date_col: str = "date") -> SplitIndex:
    if date_col not in df.columns:
        raise KeyError(f"{date_col} not in dataframe")
    dates = pd.to_datetime(df[date_col])
    train_mask = (dates >= splits.train_start) & (dates <= splits.train_end)
    val_mask = (dates >= splits.val_start) & (dates <= splits.val_end)
    test_mask = (dates >= splits.test_start) & (dates <= splits.test_end)
    if (
        (train_mask & val_mask).any()
        or (val_mask & test_mask).any()
        or (train_mask & test_mask).any()
    ):
        raise ValueError("train/val/test windows overlap in config")
    return SplitIndex(
        train=df.index[train_mask],
        val=df.index[val_mask],
        test=df.index[test_mask],
    )


def walk_forward_folds(
    df: pd.DataFrame,
    train_end: str,
    n_folds: int = 3,
    fold_days: int = 28,
    gap_days: int = 0,
    date_col: str = "date",
) -> list[tuple[pd.Index, pd.Index]]:
    """Expanding-window CV folds inside the train window.

    Each fold: train = [start, fold_val_start - gap), val = [fold_val_start, fold_val_start + fold_days].
    Folds are taken from the latest portion of the train window.
    """
    dates = pd.to_datetime(df[date_col])
    train_end_ts = pd.Timestamp(train_end)
    folds: list[tuple[pd.Index, pd.Index]] = []
    for i in range(n_folds):
        val_end = train_end_ts - pd.Timedelta(days=i * fold_days)
        val_start = val_end - pd.Timedelta(days=fold_days - 1)
        cutoff = val_start - pd.Timedelta(days=gap_days + 1)
        train_idx = df.index[dates <= cutoff]
        val_idx = df.index[(dates >= val_start) & (dates <= val_end)]
        folds.append((train_idx, val_idx))
    return list(reversed(folds))  # chronological order
