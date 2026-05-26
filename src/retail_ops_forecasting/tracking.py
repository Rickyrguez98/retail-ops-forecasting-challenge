"""MLflow helpers with a graceful no-op fallback.

If MLflow is not importable, tracking calls become no-ops and a JSON
sidecar is written to `reports/` so reviewers still see what was run.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import mlflow

    _MLFLOW_OK = True
except Exception:  # pragma: no cover - environment dependent
    _MLFLOW_OK = False

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
        _SIDE_LOG.append({"run_name": run_name, "tags": tags or {}, "params": {}, "metrics": {}, "ts": datetime.utcnow().isoformat()})
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


def log_artifact(local_path: str | Path) -> None:
    if _MLFLOW_OK:
        mlflow.log_artifact(str(local_path))


def dump_sidelog(path: str | Path) -> None:
    if not _MLFLOW_OK:
        Path(path).write_text(json.dumps(_SIDE_LOG, indent=2))


def backend_active() -> bool:
    return _MLFLOW_OK
