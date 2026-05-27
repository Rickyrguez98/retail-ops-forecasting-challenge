"""Data quality + schema contract tests on the toy fixtures."""

from __future__ import annotations

from retail_ops_forecasting.validation import (
    EXPECTED_TX_COLS,
    check_schema,
    check_unique_keys,
    check_value_ranges,
)


def test_schema_passes_on_toy(toy_transactions):
    status, _ = check_schema(toy_transactions, EXPECTED_TX_COLS, "transactions")
    assert status == "OK"


def test_unique_keys(toy_transactions):
    status, _ = check_unique_keys(
        toy_transactions, ["date", "store_id", "category"], "transactions"
    )
    assert status == "OK"


def test_value_ranges(toy_transactions):
    status, _ = check_value_ranges(toy_transactions)
    assert status == "OK"
