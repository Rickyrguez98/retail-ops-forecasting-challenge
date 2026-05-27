"""CLI: LightGBM demand benchmark with Optuna Bayesian hyperparameter tuning.

This is the second-stage model: HGB establishes the baseline + uplift
pipeline; LightGBM is added as a tuned benchmark.  Optuna runs N trials
of TPE Bayesian search over a 9-D hyperparameter space, picking the
configuration that minimizes WAPE on the validation set.  The winning
model is then evaluated on the held-out test set and persisted alongside
the event-uplift post-processor used by the HGB pipeline.

Usage
-----
    python scripts/train_demand_lightgbm.py [--n-trials N]

Default n_trials = 30 (≈ 4-8 min on a laptop).  Set lower for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_ops_forecasting import tracking  # noqa: E402
from retail_ops_forecasting.config import load_config  # noqa: E402
from retail_ops_forecasting.data import load_calendar  # noqa: E402
from retail_ops_forecasting.event_uplift import apply_uplift, build_uplift_map  # noqa: E402
from retail_ops_forecasting.features import select_feature_columns  # noqa: E402
from retail_ops_forecasting.lightgbm_model import predict_lgb, tune_lightgbm_bayes  # noqa: E402
from retail_ops_forecasting.metrics import summarize  # noqa: E402
from retail_ops_forecasting.splits import time_based_split  # noqa: E402
from retail_ops_forecasting.utils import (  # noqa: E402
    ensure_dir,
    get_logger,
    set_seed,
    setup_logging,
)


def _read_processed(p: Path) -> pd.DataFrame:
    if p.with_suffix(".parquet").exists():
        return pd.read_parquet(p.with_suffix(".parquet"))
    if p.with_suffix(".csv").exists():
        return pd.read_csv(p.with_suffix(".csv"), parse_dates=["date"])
    raise FileNotFoundError(p)


def main() -> int:
    parser = argparse.ArgumentParser(description="LightGBM demand benchmark + Optuna tuning")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials (default 30)")
    parser.add_argument("--timeout", type=int, default=None, help="Optuna wall-clock cap (s)")
    args = parser.parse_args()

    setup_logging()
    log = get_logger("train-demand-lgb")
    cfg = load_config()
    set_seed(cfg.seed)

    tracking.configure(cfg.paths.mlflow_uri, "demand_forecasting")
    ensure_dir(cfg.paths.models_dir)
    ensure_dir(cfg.paths.reports_dir)

    df = _read_processed(cfg.paths.processed_dir / "demand_features")
    log.info("Loaded demand features: %s", df.shape)

    target = cfg.targets.demand_primary
    splits = time_based_split(df, cfg.splits)

    train_df = df.loc[splits.train].dropna(subset=[target])
    val_df = df.loc[splits.val].dropna(subset=[target])
    test_df = df.loc[splits.test].dropna(subset=[target])

    feature_cols = select_feature_columns(
        df,
        target_col=target,
        leakage_blocklist=cfg.leakage_blocklist,
        extra_drop=("__baseline_lag7", "__baseline_rmean28"),
    )
    # Drop rows with NaN in the long-lag features (same convention as HGB)
    train_fit = train_df.dropna(subset=["lag_28", "rmean_28"])
    val_fit = val_df.dropna(subset=["lag_28", "rmean_28"])

    log.info(
        "LightGBM training rows=%d, val rows=%d, features=%d",
        len(train_fit),
        len(val_fit),
        len(feature_cols),
    )

    # Calendar for event uplift
    calendar_df = load_calendar(cfg.paths.raw_dir / cfg.data.calendar_file)

    with tracking.start_run(
        "lgb_demand_tuned",
        tags={"family": "ml", "model": "LightGBM", "tuner": "optuna-tpe"},
    ):
        log.info("Launching Optuna Bayesian search with %d trials…", args.n_trials)
        fit = tune_lightgbm_bayes(
            train_fit,
            val_fit,
            feature_cols=feature_cols,
            target_col=target,
            n_trials=args.n_trials,
            seed=cfg.seed,
            timeout=args.timeout,
        )

        val_pred_raw = predict_lgb(fit, val_df)
        test_pred_raw = predict_lgb(fit, test_df)

        m_val_base = summarize(val_df[target], val_pred_raw)
        m_test_base = summarize(test_df[target], test_pred_raw)

        # ── Event-window uplift (same calibrated rule as HGB) ────────────────
        cat_col = "category_name" if "category_name" in val_df.columns else "category"
        preds_val = val_df[["store_id", cat_col, "date", target]].copy()
        preds_val["y_pred"] = val_pred_raw
        preds_test = test_df[["store_id", cat_col, "date", target]].copy()
        preds_test["y_pred"] = test_pred_raw

        uplift_map = build_uplift_map(
            calendar_df=calendar_df,
            cfg=cfg,
            val_df=preds_val.rename(columns={cat_col: "category"}),
            target_col=target,
            pred_col="y_pred",
        )
        preds_val = apply_uplift(preds_val, "y_pred", uplift_map)
        preds_test = apply_uplift(preds_test, "y_pred", uplift_map)

        m_val = summarize(preds_val[target], preds_val["y_pred"])
        m_test = summarize(preds_test[target], preds_test["y_pred"])

        # ── MLflow logging ───────────────────────────────────────────────────
        tracking.log_params(
            {
                "n_features": len(feature_cols),
                "tuner": "optuna_tpe",
                "n_trials": fit.n_trials,
                "best_iteration": fit.best_iteration,
                **{
                    f"lgb_{k}": v
                    for k, v in fit.best_params.items()
                    if k not in ("objective", "metric", "verbosity", "n_jobs", "force_col_wise")
                },
            }
        )
        tracking.log_metrics({f"val_{k}": v for k, v in m_val.items() if isinstance(v, float)})
        tracking.log_metrics({f"test_{k}": v for k, v in m_test.items() if isinstance(v, float)})
        tracking.log_metrics(
            {f"val_base_{k}": v for k, v in m_val_base.items() if isinstance(v, float)}
        )
        tracking.log_metrics(
            {f"test_base_{k}": v for k, v in m_test_base.items() if isinstance(v, float)}
        )

        log.info(
            "lgb base   val: %s",
            {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in m_val_base.items()},
        )
        log.info(
            "lgb uplift val: %s",
            {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in m_val.items()},
        )
        log.info(
            "lgb base   test: %s",
            {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in m_test_base.items()},
        )
        log.info(
            "lgb uplift test: %s",
            {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in m_test.items()},
        )

        # ── Persist artefacts ────────────────────────────────────────────────
        booster_path = cfg.paths.models_dir / "demand_lgb.txt"
        fit.model.save_model(str(booster_path), num_iteration=fit.best_iteration)
        meta_path = cfg.paths.models_dir / "demand_lgb_meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "feature_cols": fit.feature_cols,
                    "best_params": {
                        k: (v if isinstance(v, int | float | str | bool) else str(v))
                        for k, v in fit.best_params.items()
                    },
                    "best_iteration": fit.best_iteration,
                    "n_trials": fit.n_trials,
                    "val_wape_optuna": fit.val_wape,
                },
                indent=2,
            )
        )

        if cat_col == "category_name":
            preds_val = preds_val.rename(columns={"category_name": "category"})
            preds_test = preds_test.rename(columns={"category_name": "category"})
        val_preds_path = cfg.paths.processed_dir / "demand_lgb_predictions_val.csv"
        test_preds_path = cfg.paths.processed_dir / "demand_lgb_predictions_test.csv"
        preds_val.to_csv(val_preds_path, index=False)
        preds_test.to_csv(test_preds_path, index=False)

        # ── MLflow: Model Registry + Artifacts + Dataset lineage ─────────────
        try:
            from mlflow.models.signature import infer_signature

            example = train_fit[fit.feature_cols].head(3)
            signature = infer_signature(example, predict_lgb(fit, train_fit.head(3)))
        except Exception:
            example, signature = None, None

        tracking.log_lightgbm_model(
            fit.model,
            name="demand_lgb_model",
            input_example=example,
            signature=signature,
            registered_model_name="demand_lgb",
        )
        tracking.log_artifact(booster_path, artifact_path="model_files")
        tracking.log_artifact(meta_path, artifact_path="model_files")
        tracking.log_artifact(val_preds_path, artifact_path="predictions")
        tracking.log_artifact(test_preds_path, artifact_path="predictions")

        raw_src = str(cfg.paths.processed_dir / "demand_features.parquet")
        tracking.log_dataset(
            train_fit, name="demand_train", source=raw_src, context="training", targets=target
        )
        tracking.log_dataset(
            val_fit, name="demand_val", source=raw_src, context="validation", targets=target
        )
        tracking.log_dataset(
            test_df, name="demand_test", source=raw_src, context="test", targets=target
        )

    # ── Append to demand experiment summary ──────────────────────────────────
    summary_path = cfg.paths.reports_dir / "demand_experiment_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["lgb_demand_tuned"] = {"val": m_val, "test": m_test}
    summary["lgb_demand_tuned_base"] = {"val": m_val_base, "test": m_test_base}
    summary_path.write_text(json.dumps(summary, indent=2, default=float))

    tracking.dump_sidelog(cfg.paths.reports_dir / "demand_tracking_sidelog.json")
    log.info("Done. Summary updated at %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
