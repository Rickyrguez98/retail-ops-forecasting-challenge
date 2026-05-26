"""LightGBM benchmark with Optuna Bayesian hyperparameter optimization.

LightGBM frequently outperforms HistGradientBoosting on tabular tasks with
careful tuning.  We use Optuna's TPE sampler (Bayesian optimization) to
search a 9-dimensional hyperparameter space, minimizing WAPE on the
validation set.  The best configuration is refit and exposed as `LGBFit`.

Key design choices
------------------
* Search space includes the most impactful LGB knobs: ``learning_rate``,
  ``num_leaves``, ``max_depth``, ``min_child_samples``, regularization
  (alpha + lambda), subsampling, and feature subsampling.
* Early stopping on val WAPE (50 rounds patience) inside each trial keeps
  individual fits fast.
* TPE seed is pinned for reproducibility.
* Optuna log level is WARNING so the pipeline stays readable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from .metrics import wape

logger = logging.getLogger(__name__)


@dataclass
class LGBFit:
    """Trained LightGBM model + provenance for downstream prediction."""
    model: lgb.Booster
    feature_cols: List[str]
    best_params: Dict = field(default_factory=dict)
    best_iteration: int = 0
    val_wape: float = 0.0
    n_trials: int = 0


# ── helpers ───────────────────────────────────────────────────────────────────

def _base_params(seed: int = 42) -> Dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "seed": seed,
        "n_jobs": -1,
        "force_col_wise": True,
        "feature_pre_filter": False,  # allow dynamic min_child_samples across trials
    }


def _train_one(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    params: Dict,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
) -> tuple[lgb.Booster, float]:
    """Fit one LightGBM model and return (booster, val_wape)."""
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(float)
    X_val = val_df[feature_cols]
    y_val = val_df[target_col].astype(float)

    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, free_raw_data=False)

    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=num_boost_round,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    val_pred = np.clip(booster.predict(X_val, num_iteration=booster.best_iteration), 0, None)
    return booster, float(wape(y_val.values, val_pred))


# ── public API ────────────────────────────────────────────────────────────────

def fit_lightgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    params: Optional[Dict] = None,
    seed: int = 42,
) -> LGBFit:
    """Fit a LightGBM model with given (or default) params + early stopping."""
    merged = {**_base_params(seed)}
    if params:
        merged.update(params)
    booster, val_w = _train_one(train_df, val_df, feature_cols, target_col, merged)
    return LGBFit(
        model=booster,
        feature_cols=feature_cols,
        best_params=merged,
        best_iteration=booster.best_iteration,
        val_wape=val_w,
    )


def predict_lgb(fit: LGBFit, df: pd.DataFrame) -> np.ndarray:
    """Predict using the best iteration from training."""
    preds = fit.model.predict(df[fit.feature_cols], num_iteration=fit.best_iteration)
    return np.clip(preds, 0, None)


def tune_lightgbm_bayes(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_trials: int = 30,
    seed: int = 42,
    timeout: Optional[int] = None,
) -> LGBFit:
    """Bayesian hyperparameter search via Optuna TPE sampler.

    Optimizes WAPE on validation set across `n_trials` (or `timeout` seconds).
    Returns the refit best model as an LGBFit.

    Parameters
    ----------
    train_df, val_df : feature/target frames already filtered (no NaNs in target).
    feature_cols     : explicit feature list (anti-leakage controls applied upstream).
    target_col       : name of the target column.
    n_trials         : Optuna trials. 30 is a good balance for ~10 min runtime.
    seed             : seed for TPE sampler and LightGBM.
    timeout          : optional wall-clock cap in seconds.
    """
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(float)
    X_val = val_df[feature_cols]
    y_val = val_df[target_col].astype(float)

    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, free_raw_data=False)

    def objective(trial: optuna.Trial) -> float:
        params = {
            **_base_params(seed),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 255),
            "max_depth":         trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq":    1,
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        booster = lgb.train(
            params,
            lgb_train,
            num_boost_round=2000,
            valid_sets=[lgb_val],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        val_pred = np.clip(booster.predict(X_val, num_iteration=booster.best_iteration), 0, None)
        return float(wape(y_val.values, val_pred))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name="lightgbm_demand")

    logger.info("Starting Optuna Bayesian search: n_trials=%d, timeout=%s", n_trials, timeout)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    logger.info("Optuna best WAPE on val: %.4f", study.best_value)
    logger.info("Optuna best params: %s", study.best_params)

    # Refit with the winning hyperparameters
    best_params = {**_base_params(seed), **study.best_params, "subsample_freq": 1}
    booster, val_w = _train_one(train_df, val_df, feature_cols, target_col, best_params)

    return LGBFit(
        model=booster,
        feature_cols=feature_cols,
        best_params=best_params,
        best_iteration=booster.best_iteration,
        val_wape=val_w,
        n_trials=len(study.trials),
    )
