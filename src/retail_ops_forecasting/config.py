"""Typed configuration loader.

The YAML config is the single source of truth for paths, splits, targets,
and the leakage blocklist. Tests rely on these dataclasses, so changes to
the YAML structure must update the dataclasses too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path
    figures_dir: Path
    models_dir: Path
    mlflow_uri: str


@dataclass(frozen=True)
class DataCfg:
    transactions_file: str
    stores_file: str
    calendar_file: str
    categories: List[str]


@dataclass(frozen=True)
class SplitsCfg:
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class TargetsCfg:
    demand_primary: str
    cash_primary: str


@dataclass(frozen=True)
class CashCfg:
    buffer_quantile: float
    aggregation: str


@dataclass(frozen=True)
class Config:
    seed: int
    paths: Paths
    data: DataCfg
    splits: SplitsCfg
    targets: TargetsCfg
    leakage_blocklist: List[str]
    cash: CashCfg
    _raw: dict = field(default_factory=dict)


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | Path = "configs/config.yaml") -> Config:
    path = _resolve(str(path))
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    paths = Paths(
        raw_dir=_resolve(raw["paths"]["raw_dir"]),
        interim_dir=_resolve(raw["paths"]["interim_dir"]),
        processed_dir=_resolve(raw["paths"]["processed_dir"]),
        reports_dir=_resolve(raw["paths"]["reports_dir"]),
        figures_dir=_resolve(raw["paths"]["figures_dir"]),
        models_dir=_resolve(raw["paths"]["models_dir"]),
        mlflow_uri=raw["paths"]["mlflow_uri"],
    )
    return Config(
        seed=int(raw["seed"]),
        paths=paths,
        data=DataCfg(**raw["data"]),
        splits=SplitsCfg(**raw["splits"]),
        targets=TargetsCfg(**raw["targets"]),
        leakage_blocklist=list(raw["leakage_blocklist"]),
        cash=CashCfg(**raw["cash"]),
        _raw=raw,
    )


def load_model_config(path: str | Path = "configs/model_config.yaml") -> dict:
    path = _resolve(str(path))
    with open(path, "r") as f:
        return yaml.safe_load(f)
