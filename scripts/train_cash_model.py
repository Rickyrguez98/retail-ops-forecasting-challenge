"""CLI: train cash forecast (HGB) + per-store P-quantile safety buffer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib  # type: ignore
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_ops_forecasting import tracking  # noqa: E402
from retail_ops_forecasting.cash_forecasting import (  # noqa: E402
    compute_buffers,
    fit_cash_model,
    predict_cash,
    recommend_cash,
)
from retail_ops_forecasting.config import load_config, load_model_config  # noqa: E402
from retail_ops_forecasting.features import select_feature_columns  # noqa: E402
from retail_ops_forecasting.metrics import coverage_at_threshold, summarize  # noqa: E402
from retail_ops_forecasting.splits import time_based_split  # noqa: E402
from retail_ops_forecasting.utils import ensure_dir, get_logger, set_seed, setup_logging  # noqa: E402


def _read_processed(p: Path) -> pd.DataFrame:
    if p.with_suffix(".parquet").exists():
        return pd.read_parquet(p.with_suffix(".parquet"))
    if p.with_suffix(".csv").exists():
        return pd.read_csv(p.with_suffix(".csv"), parse_dates=["date"])
    raise FileNotFoundError(p)


def main() -> int:
    setup_logging()
    log = get_logger("train-cash")
    cfg = load_config()
    mcfg = load_model_config()
    set_seed(cfg.seed)
    tracking.configure(cfg.paths.mlflow_uri, "cash_forecasting")
    ensure_dir(cfg.paths.models_dir)
    ensure_dir(cfg.paths.reports_dir)

    df = _read_processed(cfg.paths.processed_dir / "cash_features")
    target = cfg.targets.cash_primary
    log.info("Loaded cash features: %s (target=%s)", df.shape, target)

    # Cash rows where target is NaN cannot be used for training/eval.
    df = df.dropna(subset=[target]).reset_index(drop=True)
    splits = time_based_split(df, cfg.splits)
    train_df = df.loc[splits.train]
    val_df = df.loc[splits.val]
    test_df = df.loc[splits.test]
    log.info("Cash splits: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    feature_cols = select_feature_columns(
        df,
        target_col=target,
        leakage_blocklist=cfg.leakage_blocklist,
    )
    train_fit = train_df.dropna(subset=["cash_lag_28", "cash_rmean_28"])
    log.info("HGB cash training rows: %d (features=%d)", len(train_fit), len(feature_cols))

    metrics_summary = {}
    with tracking.start_run("hgb_cash_v1", tags={"family": "ml", "task": "cash"}):
        params = dict(mcfg["cash"]["hgb"])
        model = fit_cash_model(train_fit, feature_cols, target, params)
        val_pred = predict_cash(model, val_df, feature_cols)
        test_pred = predict_cash(model, test_df, feature_cols)
        val_df = val_df.assign(y_pred=val_pred)
        test_df = test_df.assign(y_pred=test_pred)

        m_val = summarize(val_df[target], val_df["y_pred"])
        m_test = summarize(test_df[target], test_df["y_pred"])
        tracking.log_params({"n_features": len(feature_cols), **{f"hgb_{k}": v for k, v in params.items()}})
        tracking.log_metrics({f"val_{k}": v for k, v in m_val.items() if isinstance(v, float)})
        tracking.log_metrics({f"test_{k}": v for k, v in m_test.items() if isinstance(v, float)})
        log.info("cash val: %s", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in m_val.items()})
        log.info("cash test: %s", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in m_test.items()})
        metrics_summary["hgb_cash_v1"] = {"val": m_val, "test": m_test}

        buffers = compute_buffers(val_df, target_col=target, pred_col="y_pred", quantile=cfg.cash.buffer_quantile)
        recs_val = recommend_cash(val_df[["store_id", "date", target, "y_pred"]], buffers)
        recs_test = recommend_cash(test_df[["store_id", "date", target, "y_pred"]], buffers)

        cov_val = coverage_at_threshold(recs_val[target], recs_val["recommended_cash"], threshold=1.0)
        cov_test = coverage_at_threshold(recs_test[target], recs_test["recommended_cash"], threshold=1.0)
        tracking.log_metrics({"val_coverage": cov_val, "test_coverage": cov_test, "buffer_quantile": cfg.cash.buffer_quantile})
        log.info("Coverage P%d val=%.3f test=%.3f", int(cfg.cash.buffer_quantile * 100), cov_val, cov_test)

        joblib.dump({"model": model, "feature_cols": feature_cols, "buffers": buffers}, cfg.paths.models_dir / "cash_hgb.joblib")
        recs_test.to_csv(cfg.paths.reports_dir / "cash_forecast_recommendations.csv", index=False)
        # Predictions for the unified evaluator
        out_val = cfg.paths.processed_dir / "cash_predictions_val.csv"
        out_test = cfg.paths.processed_dir / "cash_predictions_test.csv"
        val_df[["store_id", "date", target, "y_pred"]].to_csv(out_val, index=False)
        test_df[["store_id", "date", target, "y_pred"]].to_csv(out_test, index=False)

    cfg.paths.reports_dir.joinpath("cash_experiment_summary.json").write_text(
        json.dumps(metrics_summary, indent=2, default=float)
    )
    tracking.dump_sidelog(cfg.paths.reports_dir / "cash_tracking_sidelog.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
