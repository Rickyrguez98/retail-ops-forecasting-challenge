"""Typed configuration loader.

The YAML config is the single source of truth for paths, splits, targets,
and the leakage blocklist. Tests rely on these dataclasses, so changes to
the YAML structure must update the dataclasses too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
    categories: list[str]


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
class EventUpliftCfg:
    enabled: bool = True
    # Buen Fin
    buen_fin_peak_multiplier: float | None = None  # None = learn from val
    buen_fin_pre_multiplier: float = 1.10  # ~10% short on pre-event days
    buen_fin_post_multiplier: float = 1.0  # model recovers after event
    buen_fin_window_pre: int = 3
    buen_fin_window_post: int = 0
    # Dec 24-25
    dec_24_25_anchor_month: int = 12
    dec_24_25_anchor_days: list[int] = field(default_factory=lambda: [24, 25])
    dec_24_25_peak_multiplier: float = 2.8
    dec_24_25_pre_multiplier: float = 1.13  # ~13% short on Dec 22-23
    dec_24_25_post_multiplier: float = 1.0  # model accurate Dec 26+
    dec_24_25_window_pre: int = 2
    dec_24_25_window_post: int = 0
    # Dec 31
    dec_31_anchor_month: int = 12
    dec_31_anchor_days: list[int] = field(default_factory=lambda: [31])
    dec_31_peak_multiplier: float = 1.8
    dec_31_pre_multiplier: float = 1.03  # Dec 29-30 nearly accurate
    dec_31_post_multiplier: float = 1.0  # Jan 1+ model over-predicts
    dec_31_window_pre: int = 2
    dec_31_window_post: int = 0


@dataclass(frozen=True)
class Config:
    seed: int
    paths: Paths
    data: DataCfg
    splits: SplitsCfg
    targets: TargetsCfg
    leakage_blocklist: list[str]
    cash: CashCfg
    event_uplift: EventUpliftCfg = field(default_factory=EventUpliftCfg)
    _raw: dict = field(default_factory=dict)


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | Path = "configs/config.yaml") -> Config:
    path = _resolve(str(path))
    with open(path) as f:
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
    # Parse event_uplift section (optional — falls back to dataclass defaults)
    eu_raw = raw.get("event_uplift", {})
    bf = eu_raw.get("buen_fin", {})
    d25 = eu_raw.get("dec_24_25", {})
    d31 = eu_raw.get("dec_31", {})
    event_uplift = EventUpliftCfg(
        enabled=eu_raw.get("enabled", True),
        buen_fin_peak_multiplier=bf.get("peak_multiplier"),
        buen_fin_pre_multiplier=bf.get("pre_multiplier", 1.10),
        buen_fin_post_multiplier=bf.get("post_multiplier", 1.0),
        buen_fin_window_pre=bf.get("window_pre", 3),
        buen_fin_window_post=bf.get("window_post", 0),
        dec_24_25_anchor_month=d25.get("anchor_month", 12),
        dec_24_25_anchor_days=d25.get("anchor_days", [24, 25]),
        dec_24_25_peak_multiplier=d25.get("peak_multiplier", 2.8),
        dec_24_25_pre_multiplier=d25.get("pre_multiplier", 1.13),
        dec_24_25_post_multiplier=d25.get("post_multiplier", 1.0),
        dec_24_25_window_pre=d25.get("window_pre", 2),
        dec_24_25_window_post=d25.get("window_post", 0),
        dec_31_anchor_month=d31.get("anchor_month", 12),
        dec_31_anchor_days=d31.get("anchor_days", [31]),
        dec_31_peak_multiplier=d31.get("peak_multiplier", 1.8),
        dec_31_pre_multiplier=d31.get("pre_multiplier", 1.03),
        dec_31_post_multiplier=d31.get("post_multiplier", 1.0),
        dec_31_window_pre=d31.get("window_pre", 2),
        dec_31_window_post=d31.get("window_post", 0),
    )
    return Config(
        seed=int(raw["seed"]),
        paths=paths,
        data=DataCfg(**raw["data"]),
        splits=SplitsCfg(**raw["splits"]),
        targets=TargetsCfg(**raw["targets"]),
        leakage_blocklist=list(raw["leakage_blocklist"]),
        cash=CashCfg(**raw["cash"]),
        event_uplift=event_uplift,
        _raw=raw,
    )


def load_model_config(path: str | Path = "configs/model_config.yaml") -> dict:
    path = _resolve(str(path))
    with open(path) as f:
        return yaml.safe_load(f)
