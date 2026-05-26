"""CLI: run data quality checks and write a markdown report."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_ops_forecasting.config import load_config  # noqa: E402
from retail_ops_forecasting.data import (  # noqa: E402
    coverage_summary,
    load_calendar,
    load_stores,
    load_transactions,
)
from retail_ops_forecasting.utils import ensure_dir, get_logger, setup_logging  # noqa: E402
from retail_ops_forecasting.validation import format_report, run_all_checks  # noqa: E402


def main() -> int:
    setup_logging()
    log = get_logger("quality")
    cfg = load_config()
    tx = load_transactions(cfg.paths.raw_dir / cfg.data.transactions_file)
    stores = load_stores(cfg.paths.raw_dir / cfg.data.stores_file)
    cal = load_calendar(cfg.paths.raw_dir / cfg.data.calendar_file)
    checks = run_all_checks(tx, stores, cal)
    miss = coverage_summary(tx)
    ensure_dir(cfg.paths.reports_dir)
    report_path = cfg.paths.reports_dir / "data_quality_report.md"
    report_path.write_text(format_report(checks, miss))
    log.info("Wrote %s", report_path)
    # Hard fails should be visible
    fails = [n for n, (s, _) in checks.items() if s == "FAIL"]
    if fails:
        log.error("Failing checks: %s", fails)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
