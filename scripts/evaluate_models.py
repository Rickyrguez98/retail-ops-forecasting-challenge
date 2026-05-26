"""CLI: evaluate the held-out test set and produce segment + event metric tables."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_ops_forecasting import tracking  # noqa: E402
from retail_ops_forecasting.config import load_config  # noqa: E402
from retail_ops_forecasting.data import load_calendar, load_stores  # noqa: E402
from retail_ops_forecasting.reporting import (  # noqa: E402
    plot_actual_vs_pred,
    plot_residual_hist,
    write_breakdowns,
    write_overall,
)
from retail_ops_forecasting.utils import ensure_dir, get_logger, setup_logging  # noqa: E402


def _attach_event_flag(df: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    use = cal[["date", "is_buen_fin", "is_semana_santa", "is_navidad_season", "is_payday", "is_holiday", "is_weekend"]]
    df = df.merge(use, on="date", how="left")
    df["event_label"] = "regular"
    for col, label in [
        ("is_buen_fin", "buen_fin"),
        ("is_semana_santa", "semana_santa"),
        ("is_navidad_season", "navidad"),
        ("is_payday", "payday"),
        ("is_holiday", "holiday"),
        ("is_weekend", "weekend"),
    ]:
        df.loc[df[col].fillna(False).astype(bool), "event_label"] = label
    return df


def main() -> int:
    setup_logging()
    log = get_logger("evaluate")
    cfg = load_config()
    ensure_dir(cfg.paths.figures_dir)

    cal = load_calendar(cfg.paths.raw_dir / cfg.data.calendar_file)
    stores = load_stores(cfg.paths.raw_dir / cfg.data.stores_file)

    tracking.configure(cfg.paths.mlflow_uri, "evaluation")

    # ---- Demand evaluation ----
    demand_test = pd.read_csv(cfg.paths.processed_dir / "demand_predictions_test.csv", parse_dates=["date"])
    target_d = cfg.targets.demand_primary
    demand_test = demand_test.rename(columns={target_d: "y_true"})
    demand_test = demand_test.merge(stores[["store_id", "store_format", "region", "socioeconomic_level"]], on="store_id", how="left")
    demand_test = _attach_event_flag(demand_test, cal)

    with tracking.start_run("evaluate_demand", tags={"task": "demand", "stage": "evaluation"}):
        s = write_overall(demand_test, cfg.paths.reports_dir / "demand_overall_test_metrics.csv")
        log.info("Demand overall test: %s", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in s.items()})
        tracking.log_metrics({f"test_{k}": v for k, v in s.items() if isinstance(v, (int, float))})
        write_breakdowns(
            demand_test,
            cfg.paths.figures_dir,
            cfg.paths.reports_dir,
            groups={
                "category": ["category"],
                "region": ["region"],
                "store_format": ["store_format"],
                "event": ["event_label"],
            },
            prefix="demand",
        )
        plot_actual_vs_pred(demand_test, cfg.paths.figures_dir / "demand_actual_vs_pred.png", "Demand forecast — test")
        plot_residual_hist(demand_test, cfg.paths.figures_dir / "demand_residuals.png")

        # Log figures + breakdown CSVs to MLflow
        for fname in ("demand_actual_vs_pred.png", "demand_residuals.png", "error_demand_by_category.png"):
            tracking.log_artifact(cfg.paths.figures_dir / fname, artifact_path="figures")
        for csv in cfg.paths.reports_dir.glob("demand_metrics_by_*.csv"):
            tracking.log_artifact(csv, artifact_path="breakdowns")
        tracking.log_artifact(cfg.paths.reports_dir / "demand_overall_test_metrics.csv", artifact_path="breakdowns")

    # ---- Cash evaluation ----
    cash_test = pd.read_csv(cfg.paths.processed_dir / "cash_predictions_test.csv", parse_dates=["date"])
    target_c = cfg.targets.cash_primary
    cash_test = cash_test.rename(columns={target_c: "y_true"})
    cash_test = cash_test.merge(stores[["store_id", "store_format", "region", "socioeconomic_level"]], on="store_id", how="left")
    cash_test = _attach_event_flag(cash_test, cal)

    with tracking.start_run("evaluate_cash", tags={"task": "cash", "stage": "evaluation"}):
        s = write_overall(cash_test, cfg.paths.reports_dir / "cash_overall_test_metrics.csv")
        log.info("Cash overall test: %s", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in s.items()})
        tracking.log_metrics({f"test_{k}": v for k, v in s.items() if isinstance(v, (int, float))})
        write_breakdowns(
            cash_test,
            cfg.paths.figures_dir,
            cfg.paths.reports_dir,
            groups={
                "region": ["region"],
                "store_format": ["store_format"],
                "event": ["event_label"],
            },
            prefix="cash",
        )
        plot_actual_vs_pred(cash_test, cfg.paths.figures_dir / "cash_actual_vs_pred.png", "Cash forecast — test")
        plot_residual_hist(cash_test, cfg.paths.figures_dir / "cash_residuals.png")

        for fname in ("cash_actual_vs_pred.png", "cash_residuals.png", "cash_coverage_per_store.png"):
            tracking.log_artifact(cfg.paths.figures_dir / fname, artifact_path="figures")
        for csv in cfg.paths.reports_dir.glob("cash_metrics_by_*.csv"):
            tracking.log_artifact(csv, artifact_path="breakdowns")
        tracking.log_artifact(cfg.paths.reports_dir / "cash_overall_test_metrics.csv", artifact_path="breakdowns")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
