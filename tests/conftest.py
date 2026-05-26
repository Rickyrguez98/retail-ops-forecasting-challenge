"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def toy_transactions() -> pd.DataFrame:
    """Small synthetic frame matching the transactions schema."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    stores = ["STR_001", "STR_002"]
    cats = ["Abarrotes", "Bebidas"]
    rows = []
    for s in stores:
        for c in cats:
            base = rng.integers(100, 300)
            for d in dates:
                tx = int(base + rng.normal(0, 20))
                cash = int(tx * 0.5)
                card = tx - cash
                rows.append(
                    {
                        "date": d,
                        "store_id": s,
                        "category": c,
                        "total_transactions": tx,
                        "cash_transactions": cash,
                        "card_transactions": card,
                        "amount_total": tx * 50.0,
                        "amount_cash": cash * 50.0,
                        "amount_card": card * 50.0,
                        "units_sold": tx * 1.2,
                        "avg_ticket": 50.0,
                        "has_promotion": 0,
                        "replenishment_signal": tx * 1.1,
                    }
                )
    df = pd.DataFrame(rows)
    return df.sort_values(["store_id", "category", "date"]).reset_index(drop=True)


@pytest.fixture
def toy_calendar() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "day_of_week": dates.dayofweek,
            "day_name": dates.day_name(),
            "week_of_year": dates.isocalendar().week.values,
            "month": dates.month,
            "year": dates.year,
            "quarter": dates.quarter,
            "season": ["Invierno"] * 60,
            "is_holiday": [False] * 60,
            "holiday_name": [None] * 60,
            "is_payday": (dates.day.isin([15, 30, 31])).tolist(),
            "is_weekend": (dates.dayofweek >= 5).tolist(),
            "is_navidad_season": [False] * 60,
            "is_buen_fin": [False] * 60,
            "is_semana_santa": [False] * 60,
        }
    )


@pytest.fixture
def toy_stores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "store_id": "STR_001",
                "store_format": "Supercenter",
                "region": "Centro",
                "size_sqm": 8000,
                "num_checkouts": 20,
                "opening_year": 2010,
                "socioeconomic_level": "B",
                "has_pharmacy": True,
                "has_fuel_station": False,
            },
            {
                "store_id": "STR_002",
                "store_format": "Express",
                "region": "Norte",
                "size_sqm": 500,
                "num_checkouts": 4,
                "opening_year": 2018,
                "socioeconomic_level": "C",
                "has_pharmacy": False,
                "has_fuel_station": False,
            },
        ]
    )
