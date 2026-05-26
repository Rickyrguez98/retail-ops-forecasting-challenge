"""Splits cannot overlap and must respect time order."""

from __future__ import annotations

import pandas as pd

from retail_ops_forecasting.config import SplitsCfg
from retail_ops_forecasting.splits import time_based_split, walk_forward_folds


def _frame():
    dates = pd.date_range("2023-01-01", "2024-02-29", freq="D")
    return pd.DataFrame({"date": dates, "x": range(len(dates))})


def test_split_partitions_are_disjoint_and_ordered():
    df = _frame()
    cfg = SplitsCfg(
        train_start="2023-01-01",
        train_end="2023-09-30",
        val_start="2023-10-01",
        val_end="2023-11-30",
        test_start="2023-12-01",
        test_end="2024-02-29",
    )
    s = time_based_split(df, cfg)
    train_dates = df.loc[s.train, "date"]
    val_dates = df.loc[s.val, "date"]
    test_dates = df.loc[s.test, "date"]
    assert train_dates.max() < val_dates.min()
    assert val_dates.max() < test_dates.min()
    assert set(s.train).isdisjoint(s.val)
    assert set(s.val).isdisjoint(s.test)
    assert set(s.train).isdisjoint(s.test)


def test_walk_forward_folds_chronological():
    df = _frame()
    folds = walk_forward_folds(df, train_end="2023-09-30", n_folds=3, fold_days=28)
    last_train_end = None
    for train_idx, val_idx in folds:
        train_max = df.loc[train_idx, "date"].max()
        val_min = df.loc[val_idx, "date"].min()
        val_max = df.loc[val_idx, "date"].max()
        assert train_max < val_min  # no leak
        if last_train_end is not None:
            assert val_max >= last_train_end  # monotonic
        last_train_end = val_max
