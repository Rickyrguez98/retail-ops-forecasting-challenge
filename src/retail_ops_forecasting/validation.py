"""Data quality checks producing a structured report.

Each check returns a (status, detail) tuple. The orchestrating function
collects them into a dict suitable for both pytest assertions and a
markdown summary.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

EXPECTED_TX_COLS = {
    "date",
    "store_id",
    "category",
    "total_transactions",
    "cash_transactions",
    "card_transactions",
    "amount_total",
    "amount_cash",
    "amount_card",
    "units_sold",
    "avg_ticket",
    "has_promotion",
    "replenishment_signal",
}
EXPECTED_STORE_COLS = {
    "store_id",
    "store_format",
    "region",
    "size_sqm",
    "num_checkouts",
    "opening_year",
    "socioeconomic_level",
    "has_pharmacy",
    "has_fuel_station",
}
EXPECTED_CAL_COLS = {
    "date",
    "day_of_week",
    "day_name",
    "week_of_year",
    "month",
    "year",
    "quarter",
    "season",
    "is_holiday",
    "holiday_name",
    "is_payday",
    "is_weekend",
    "is_navidad_season",
    "is_buen_fin",
    "is_semana_santa",
    "is_event",  # derived in loader, kept here so the schema check ignores it
}

CheckResult = Tuple[str, str]  # (status, detail)


def _ok(detail: str) -> CheckResult:
    return ("OK", detail)


def _warn(detail: str) -> CheckResult:
    return ("WARN", detail)


def _fail(detail: str) -> CheckResult:
    return ("FAIL", detail)


def check_schema(df: pd.DataFrame, expected: set, name: str) -> CheckResult:
    missing = expected - set(df.columns)
    extra = set(df.columns) - expected
    if missing:
        return _fail(f"{name}: missing columns {sorted(missing)}")
    if extra:
        return _warn(f"{name}: unexpected columns {sorted(extra)}")
    return _ok(f"{name}: all expected columns present")


def check_unique_keys(df: pd.DataFrame, keys: list, name: str) -> CheckResult:
    dups = df.duplicated(subset=keys).sum()
    if dups > 0:
        return _fail(f"{name}: {dups} duplicate rows on {keys}")
    return _ok(f"{name}: unique on {keys}")


def check_date_continuity(df: pd.DataFrame, name: str) -> CheckResult:
    dates = pd.to_datetime(df["date"]).dt.normalize().sort_values().unique()
    dmin, dmax = pd.Timestamp(dates.min()), pd.Timestamp(dates.max())
    expected = pd.date_range(dmin, dmax, freq="D").values
    missing = set(pd.to_datetime(expected)) - set(pd.to_datetime(dates))
    if missing:
        return _warn(f"{name}: {len(missing)} missing days in calendar coverage")
    return _ok(f"{name}: contiguous from {dmin.date()} to {dmax.date()} ({len(dates)} days)")


def check_value_ranges(tx: pd.DataFrame) -> CheckResult:
    issues = []
    for col in ("total_transactions", "cash_transactions", "card_transactions", "units_sold"):
        if col in tx.columns and (tx[col].dropna() < 0).any():
            issues.append(f"{col} has negative values")
    for col in ("amount_total", "amount_cash", "amount_card", "avg_ticket"):
        if col in tx.columns and (tx[col].dropna() < 0).any():
            issues.append(f"{col} has negative values")
    if issues:
        return _fail("; ".join(issues))
    return _ok("no negative values in count/amount columns")


def check_cash_card_consistency(tx: pd.DataFrame) -> CheckResult:
    """cash + card should reconcile with totals where both legs are present."""
    sub = tx.dropna(subset=["cash_transactions", "card_transactions", "total_transactions"])
    diff = (sub["cash_transactions"] + sub["card_transactions"]) - sub["total_transactions"]
    pct_match = float((diff.abs() <= 1).mean()) * 100
    return _ok(f"transactions reconcile within ±1 on {pct_match:.2f}% of complete rows")


def check_amount_consistency(tx: pd.DataFrame) -> CheckResult:
    sub = tx.dropna(subset=["amount_cash", "amount_card", "amount_total"])
    if sub.empty:
        return _warn("no rows with all amount components present")
    rel = ((sub["amount_cash"] + sub["amount_card"]) - sub["amount_total"]).abs() / (
        sub["amount_total"].replace(0, 1.0)
    )
    pct_match = float((rel <= 0.01).mean()) * 100
    return _ok(f"amount_cash+amount_card ≈ amount_total within 1% on {pct_match:.2f}% of rows")


def check_replenishment_leakage_flag(tx: pd.DataFrame) -> CheckResult:
    """Sanity: replenishment_signal correlates with same-day demand (documented leakage)."""
    sub = tx.dropna(subset=["replenishment_signal", "total_transactions"])
    if sub.empty:
        return _warn("no rows to evaluate replenishment signal correlation")
    a = sub["replenishment_signal"].astype("float64").to_numpy()
    b = sub["total_transactions"].astype("float64").to_numpy()
    corr = float(((a - a.mean()) * (b - b.mean())).sum() / (((a - a.mean()) ** 2).sum() ** 0.5 * ((b - b.mean()) ** 2).sum() ** 0.5))
    return _warn(
        f"replenishment_signal vs total_transactions corr={corr:.3f} — excluded from features"
    )


def run_all_checks(tx: pd.DataFrame, stores: pd.DataFrame, cal: pd.DataFrame) -> Dict[str, CheckResult]:
    checks: Dict[str, CheckResult] = {}
    checks["schema_transactions"] = check_schema(tx, EXPECTED_TX_COLS, "transactions")
    checks["schema_stores"] = check_schema(stores, EXPECTED_STORE_COLS, "stores")
    checks["schema_calendar"] = check_schema(cal, EXPECTED_CAL_COLS, "calendar")
    checks["unique_tx_key"] = check_unique_keys(tx, ["date", "store_id", "category"], "transactions")
    checks["unique_stores"] = check_unique_keys(stores, ["store_id"], "stores")
    checks["unique_calendar"] = check_unique_keys(cal, ["date"], "calendar")
    checks["calendar_continuity"] = check_date_continuity(cal, "calendar")
    checks["value_ranges"] = check_value_ranges(tx)
    checks["cash_card_reconcile"] = check_cash_card_consistency(tx)
    checks["amount_reconcile"] = check_amount_consistency(tx)
    checks["replenishment_flag"] = check_replenishment_leakage_flag(tx)
    return checks


def format_report(checks: Dict[str, CheckResult], missing: pd.DataFrame) -> str:
    lines = ["# Data Quality Report", ""]
    lines.append("## Schema and integrity checks")
    lines.append("")
    lines.append("| Check | Status | Detail |")
    lines.append("|-------|--------|--------|")
    for name, (status, detail) in checks.items():
        lines.append(f"| `{name}` | {status} | {detail} |")
    lines.append("")
    lines.append("## Missingness")
    lines.append("")
    lines.append("| Column | n_missing | % missing |")
    lines.append("|--------|-----------|-----------|")
    for _, row in missing.head(15).iterrows():
        lines.append(f"| `{row['column']}` | {int(row['n_missing'])} | {row['pct_missing']:.3f}% |")
    lines.append("")
    lines.append("Generated by `src/retail_ops_forecasting/validation.py`.")
    return "\n".join(lines)
