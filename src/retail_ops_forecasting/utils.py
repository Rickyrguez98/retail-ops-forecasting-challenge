"""Shared helpers: logging setup, deterministic seeding, IO."""

from __future__ import annotations

import logging
import logging.config
import os
import random
from pathlib import Path

import numpy as np
import yaml


def setup_logging(path: str | Path = "configs/logging.yaml") -> None:
    p = Path(path)
    if not p.is_absolute():
        from .config import REPO_ROOT

        p = REPO_ROOT / p
    if p.exists():
        with open(p, "r") as f:
            logging.config.dictConfig(yaml.safe_load(f))
    else:
        logging.basicConfig(level=logging.INFO)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
