"""CLI: build interim and processed datasets.

Outputs:
  data/interim/merged.parquet              — joined transactions + stores + calendar
  data/interim/cash_store_day.parquet      — store-day aggregation for cash task
  data/processed/demand_features.parquet   — feature matrix for demand model
  data/processed/cash_features.parquet     — feature matrix for cash model
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_ops_forecasting.config import load_config  # noqa: E402
from retail_ops_forecasting.data import build_merged, cash_store_day  # noqa: E402
from retail_ops_forecasting.features import (  # noqa: E402
    build_cash_features,
    build_demand_features,
    encode_categoricals,
)
from retail_ops_forecasting.utils import ensure_dir, get_logger, setup_logging  # noqa: E402


def main() -> int:
    setup_logging()
    log = get_logger("prepare")
    cfg = load_config()

    ensure_dir(cfg.paths.interim_dir)
    ensure_dir(cfg.paths.processed_dir)

    log.info("Loading and merging raw data...")
    merged = build_merged(cfg)
    log.info("Merged shape: %s", merged.shape)
    merged_pq = cfg.paths.interim_dir / "merged.parquet"
    try:
        merged.to_parquet(merged_pq, index=False)
    except Exception:
        # Fallback if pyarrow/fastparquet missing
        merged_pq = cfg.paths.interim_dir / "merged.csv"
        merged.to_csv(merged_pq, index=False)
    log.info("Wrote %s", merged_pq)

    log.info("Aggregating cash store-day...")
    cs = cash_store_day(merged)
    cs_pq = cfg.paths.interim_dir / (
        "cash_store_day.parquet" if merged_pq.suffix == ".parquet" else "cash_store_day.csv"
    )
    if cs_pq.suffix == ".parquet":
        cs.to_parquet(cs_pq, index=False)
    else:
        cs.to_csv(cs_pq, index=False)
    log.info("Wrote %s", cs_pq)

    log.info("Building demand features...")
    demand = build_demand_features(merged, target_col=cfg.targets.demand_primary)
    demand = encode_categoricals(demand)
    out = cfg.paths.processed_dir / (
        "demand_features.parquet" if merged_pq.suffix == ".parquet" else "demand_features.csv"
    )
    if out.suffix == ".parquet":
        demand.to_parquet(out, index=False)
    else:
        demand.to_csv(out, index=False)
    log.info("Wrote %s (%s)", out, demand.shape)

    log.info("Building cash features...")
    cash_feat = build_cash_features(cs, target_col=cfg.targets.cash_primary)
    cash_feat = encode_categoricals(cash_feat)
    out_cash = cfg.paths.processed_dir / (
        "cash_features.parquet" if merged_pq.suffix == ".parquet" else "cash_features.csv"
    )
    if out_cash.suffix == ".parquet":
        cash_feat.to_parquet(out_cash, index=False)
    else:
        cash_feat.to_csv(out_cash, index=False)
    log.info("Wrote %s (%s)", out_cash, cash_feat.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
