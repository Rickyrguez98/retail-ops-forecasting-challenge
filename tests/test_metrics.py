"""Metric correctness on tiny hand-checked arrays."""

from __future__ import annotations

import numpy as np

from retail_ops_forecasting.metrics import (
    bias,
    coverage_at_threshold,
    mae,
    rmse,
    smape,
    summarize,
    wape,
)


def test_wape_basic():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    expected = (2 + 2 + 3) / (10 + 20 + 30)
    assert abs(wape(y_true, y_pred) - expected) < 1e-9


def test_wape_handles_zero_truth():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert np.isnan(wape(y_true, y_pred))


def test_mae_rmse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])
    assert mae(y_true, y_pred) == 2.0 / 3.0
    assert abs(rmse(y_true, y_pred) - np.sqrt(4.0 / 3.0)) < 1e-9


def test_bias_sign_convention():
    # Under-forecast → positive bias
    assert bias([10, 10], [8, 8]) == 2.0
    assert bias([10, 10], [12, 12]) == -2.0


def test_smape_symmetry():
    a = smape([100], [80])
    b = smape([80], [100])
    assert abs(a - b) < 1e-9


def test_coverage_simple():
    actual = [10, 10, 10]
    rec = [11, 9, 12]
    # rec >= 1.0 * actual on 2 of 3 days
    assert abs(coverage_at_threshold(actual, rec, 1.0) - 2 / 3) < 1e-9


def test_summarize_keys():
    s = summarize([1, 2, 3], [1, 2, 3])
    for k in ("wape", "mae", "rmse", "smape", "bias", "n"):
        assert k in s
