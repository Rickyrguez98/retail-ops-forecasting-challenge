"""MLflow helpers with a graceful no-op fallback.

If MLflow is not importable, tracking calls become no-ops and a JSON
sidecar is written to `reports/` so reviewers still see what was run.

Exposes 5 logging primitives (all no-op when MLflow is missing):
  * log_params       — flat key/value params
  * log_metrics      — float metrics, NaN-safe
  * log_artifact     — any file (figures, CSVs, JSON summaries)
  * log_sklearn_model — registers a sklearn model with optional signature
  * log_lightgbm_model — registers a LightGBM Booster
  * log_dataset      — tracks dataset lineage (source, hash, rows, schema)
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import mlflow
    import mlflow.data  # noqa: F401  (loads the data subpackage)

    _MLFLOW_OK = True
except Exception:  # pragma: no cover - environment dependent
    _MLFLOW_OK = False

logger = logging.getLogger(__name__)

_SIDE_LOG: list[Dict[str, Any]] = []


def configure(uri: str, experiment: str) -> None:
    if _MLFLOW_OK:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)


@contextmanager
def start_run(run_name: str, tags: Optional[Dict[str, str]] = None):
    if _MLFLOW_OK:
        with mlflow.start_run(run_name=run_name, tags=tags or {}):
            yield
    else:
        _SIDE_LOG.append(
            {
                "run_name": run_name,
                "tags": tags or {},
                "params": {},
                "metrics": {},
                "ts": datetime.utcnow().isoformat(),
            }
        )
        yield


def log_params(params: Dict[str, Any]) -> None:
    if _MLFLOW_OK:
        mlflow.log_params({k: str(v) for k, v in params.items()})
    elif _SIDE_LOG:
        _SIDE_LOG[-1]["params"].update(params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    clean = {k: float(v) for k, v in metrics.items() if v is not None and v == v}  # drop NaN
    if _MLFLOW_OK:
        mlflow.log_metrics(clean, step=step)
    elif _SIDE_LOG:
        _SIDE_LOG[-1]["metrics"].update(clean)


def log_artifact(local_path: str | Path, artifact_path: Optional[str] = None) -> None:
    """Attach any file (figure, CSV, JSON) to the active run."""
    if _MLFLOW_OK and Path(local_path).exists():
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)


def log_sklearn_model(
    model,
    name: str,
    input_example=None,
    signature=None,
    registered_model_name: Optional[str] = None,
) -> None:
    """Log a sklearn-compatible model with optional registry registration.

    The model is serialised in MLflow's sklearn flavor (joblib under the hood)
    and stored as an artifact of the active run.  Pass `registered_model_name`
    to also register it in the Model Registry (visible in the "Models" tab).
    """
    if not _MLFLOW_OK:
        return
    try:
        kwargs = {"sk_model": model, "input_example": input_example, "signature": signature}
        # MLflow 3.x uses `name=`, MLflow 2.x uses `artifact_path=` — try both.
        try:
            mlflow.sklearn.log_model(name=name, registered_model_name=registered_model_name, **kwargs)
        except TypeError:
            mlflow.sklearn.log_model(artifact_path=name, registered_model_name=registered_model_name, **kwargs)
        logger.info("Logged sklearn model '%s' to MLflow", name)
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to log sklearn model '%s': %s", name, e)


def log_lightgbm_model(
    booster,
    name: str,
    input_example=None,
    signature=None,
    registered_model_name: Optional[str] = None,
) -> None:
    """Log a LightGBM Booster and optionally register it."""
    if not _MLFLOW_OK:
        return
    try:
        import mlflow.lightgbm  # local import: lightgbm flavor is optional

        kwargs = {"lgb_model": booster, "input_example": input_example, "signature": signature}
        try:
            mlflow.lightgbm.log_model(name=name, registered_model_name=registered_model_name, **kwargs)
        except TypeError:
            mlflow.lightgbm.log_model(artifact_path=name, registered_model_name=registered_model_name, **kwargs)
        logger.info("Logged LightGBM model '%s' to MLflow", name)
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to log LightGBM model '%s': %s", name, e)


def log_dataset(
    df,
    name: str,
    source: str,
    context: str = "training",
    targets: Optional[str] = None,
) -> None:
    """Track dataset lineage for the active run.

    Records: source path, schema, profile (row count, columns), hash.
    Visible in the run's "Datasets used" panel and in the "Datasets" tab.

    Parameters
    ----------
    df       : pandas DataFrame.
    name     : human-friendly dataset name (e.g. "demand_train").
    source   : path or URI describing where the data came from.
    context  : one of "training", "validation", "test", "evaluation".
    targets  : optional column name flagged as the prediction target.
    """
    if not _MLFLOW_OK:
        return
    try:
        ds = mlflow.data.from_pandas(df, source=source, name=name, targets=targets)
        mlflow.log_input(ds, context=context)
        logger.info("Logged dataset '%s' (%s, n=%d) to MLflow", name, context, len(df))
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to log dataset '%s': %s", name, e)


def dump_sidelog(path: str | Path) -> None:
    if not _MLFLOW_OK:
        Path(path).write_text(json.dumps(_SIDE_LOG, indent=2))


def backend_active() -> bool:
    return _MLFLOW_OK
