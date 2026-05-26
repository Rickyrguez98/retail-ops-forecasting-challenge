"""CLI: train demand baselines + HGB, log to MLflow, persist predictions."""

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
from retail_ops_forecasting.baselines import (  # noqa: E402
    apply_historical_dow_mean,
    historical_dow_mean,
    naive_lag,
    rolling_mean_lag,
)
from retail_ops_forecasting.config import load_config, load_model_config  # noqa: E402
from retail_ops_forecasting.features import select_feature_columns  # noqa: E402
from retail_ops_forecasting.metrics import summarize  # noqa: E402
from retail_ops_forecasting.modeling import assemble_predictions, fit_hgb, predict  # noqa: E402
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
    log = get_logger("train-demand")
    cfg = load_config()
    mcfg = load_model_config()
    set_seed(cfg.seed)

    tracking.configure(cfg.paths.mlflow_uri, "demand_forecasting")
    ensure_dir(cfg.paths.models_dir)
    ensure_dir(cfg.paths.reports_dir)

    df = _read_processed(cfg.paths.processed_dir / "demand_features")
    log.info("Loaded demand features: %s", df.shape)

    target = cfg.targets.demand_primary
    entity = ["store_id", "category"]
    df["__baseline_lag7"] = naive_lag(df, target, lag=7, entity_cols=entity)
    df["__baseline_rmean28"] = rolling_mean_lag(df, target, window=28, entity_cols=entity)

    splits = time_based_split(df, cfg.splits)
    log.info("Split sizes: train=%d, val=%d, test=%d", len(splits.train), len(splits.val), len(splits.test))

    train_df = df.loc[splits.train].dropna(subset=[target])
    val_df = df.loc[splits.val].dropna(subset=[target])
    test_df = df.loc[splits.test].dropna(subset=[target])

    # ---------- Baselines ----------
    metrics_summary = {}

    # 1. seasonal naive lag-7
    with tracking.start_run("baseline_naive_lag7", tags={"family": "baseline", "model": "lag7"}):
        b_val = val_df.dropna(subset=["__baseline_lag7"])
        b_test = test_df.dropna(subset=["__baseline_lag7"])
        m_val = summarize(b_val[target], b_val["__baseline_lag7"])
        m_test = summarize(b_test[target], b_test["__baseline_lag7"])
        tracking.log_params({"model": "seasonal_naive_lag7"})
        tracking.log_metrics({f"val_{k}": v for k, v in m_val.items() if isinstance(v, float)})
        tracking.log_metrics({f"test_{k}": v for k, v in m_test.items() if isinstance(v, float)})
        log.info("baseline_lag7 val: %s", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in m_val.items()})
        log.info("baseline_lag7 test: %s", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in m_test.items()})
        metrics_summary["baseline_naive_lag7"] = {"val": m_val, "test": m_test}

    # 2. rolling mean 28
    with tracking.start_run("baseline_rolling28", tags={"family": "baseline", "model": "rmean28"}):
        b_val = val_df.dropna(subset=["__baseline_rmean28"])
        b_test = test_df.dropna(subset=["__baseline_rmean28"])
        m_val = summarize(b_val[target], b_val["__baseline_rmean28"])
        m_test = summarize(b_test[target], b_test["__baseline_rmean28"])
        tracking.log_params({"model": "rolling_mean_28"})
        tracking.log_metrics({f"val_{k}": v for k, v in m_val.items() if isinstance(v, float)})
        tracking.log_metrics({f"test_{k}": v for k, v in m_test.items() if isinstance(v, float)})
        metrics_summary["baseline_rolling28"] = {"val": m_val, "test": m_test}

    # 3. historical DOW mean
    with tracking.start_run("baseline_dow_mean", tags={"family": "baseline", "model": "dow_mean"}):
        lookup = historical_dow_mean(train_df, target, entity)
        val_pred = apply_historical_dow_mean(val_df, lookup, target, entity)
        test_pred = apply_historical_dow_mean(test_df, lookup, target, entity)
        m_val = summarize(val_df[target], val_pred)
        m_test = summarize(test_df[target], test_pred)
        tracking.log_params({"model": "historical_dow_mean"})
        tracking.log_metrics({f"val_{k}": v for k, v in m_val.items() if isinstance(v, float)})
        tracking.log_metrics({f"test_{k}": v for k, v in m_test.items() if isinstance(v, float)})
        metrics_summary["baseline_dow_mean"] = {"val": m_val, "test": m_test}

    # ---------- HGB model ----------
    feature_cols = select_feature_columns(
        df, target_col=target, leakage_blocklist=cfg.leakage_blocklist, extra_drop=("__baseline_lag7", "__baseline_rmean28")
    )
    # HGB cannot accept NaN target; lag features carry many NaN early in series — those rows are dropped.
    train_fit = train_df.dropna(subset=["lag_28", "rmean_28"])
    log.info("HGB training rows: %d (features=%d)", len(train_fit), len(feature_cols))

    with tracking.start_run("hgb_demand_v1", tags={"family": "ml", "model": "HistGradientBoosting"}):
        params = dict(mcfg["demand"]["hgb"])
        fit = fit_hgb(train_fit, feature_cols=feature_cols, target_col=target, params=params)
        val_pred = predict(fit, val_df)
        test_pred = predict(fit, test_df)
        m_val = summarize(val_df[target], np.clip(val_pred, 0, None))
        m_test = summarize(test_df[target], np.clip(test_pred, 0, None))
        tracking.log_params({"n_features": len(feature_cols), **{f"hgb_{k}": v for k, v in params.items()}})
        tracking.log_metrics({f"val_{k}": v for k, v in m_val.items() if isinstance(v, float)})
        tracking.log_metrics({f"test_{k}": v for k, v in m_test.items() if isinstance(v, float)})
        log.info("hgb val: %s", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in m_val.items()})
        log.info("hgb test: %s", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in m_test.items()})
        metrics_summary["hgb_demand_v1"] = {"val": m_val, "test": m_test}

        # Persist artefacts
        model_path = cfg.paths.models_dir / "demand_hgb.joblib"
        joblib.dump({"model": fit.model, "feature_cols": fit.feature_cols}, model_path)
        feat_path = cfg.paths.models_dir / "demand_feature_cols.json"
        feat_path.write_text(json.dumps(fit.feature_cols, indent=2))

        # Persist predictions for downstream evaluation. Use the readable
        # category_name when present so reports don't show integer codes.
        cat_col = "category_name" if "category_name" in val_df.columns else "category"
        preds_val = val_df[["store_id", cat_col, "date", target]].copy()
        preds_val["y_pred"] = np.clip(val_pred, 0, None)
        preds_test = test_df[["store_id", cat_col, "date", target]].copy()
        preds_test["y_pred"] = np.clip(test_pred, 0, None)
        if cat_col == "category_name":
            preds_val = preds_val.rename(columns={"category_name": "category"})
            preds_test = preds_test.rename(columns={"category_name": "category"})
        preds_val.to_csv(cfg.paths.processed_dir / "demand_predictions_val.csv", index=False)
        preds_test.to_csv(cfg.paths.processed_dir / "demand_predictions_test.csv", index=False)

    # Summary JSON for the report
    out_summary = cfg.paths.reports_dir / "demand_experiment_summary.json"
    out_summary.write_text(json.dumps(metrics_summary, indent=2, default=float))
    tracking.dump_sidelog(cfg.paths.reports_dir / "demand_tracking_sidelog.json")
    log.info("Done. Summary at %s", out_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
