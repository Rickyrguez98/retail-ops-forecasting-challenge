"""Build reporting tables and figures.

These helpers are intentionally simple — they accept a long-format
predictions dataframe and write CSVs + figures to `reports/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from . import metrics as mt


def _grouped(df: pd.DataFrame, group_cols: list[str], y_true: str, y_pred: str) -> pd.DataFrame:
    rows = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, sub in df.groupby(grouper, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(mt.summarize(sub[y_true], sub[y_pred]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def write_overall(df: pd.DataFrame, out_path: Path, y_true: str = "y_true", y_pred: str = "y_pred") -> dict:
    summary = mt.summarize(df[y_true], df[y_pred])
    pd.DataFrame([summary]).to_csv(out_path, index=False)
    return summary


def write_breakdowns(
    df: pd.DataFrame,
    figures_dir: Path,
    reports_dir: Path,
    groups: Dict[str, list[str]],
    y_true: str = "y_true",
    y_pred: str = "y_pred",
    prefix: str = "demand",
) -> Dict[str, Path]:
    paths = {}
    for name, cols in groups.items():
        tbl = _grouped(df, cols, y_true, y_pred)
        out = reports_dir / f"{prefix}_metrics_by_{name}.csv"
        tbl.to_csv(out, index=False)
        paths[name] = out
    return paths


def plot_actual_vs_pred(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    y_true: str = "y_true",
    y_pred: str = "y_pred",
    date_col: str = "date",
    max_points: int = 5000,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    agg = df.groupby(date_col)[[y_true, y_pred]].sum().reset_index()
    ax.plot(agg[date_col], agg[y_true], label="actual", linewidth=1.4)
    ax.plot(agg[date_col], agg[y_pred], label="predicted", linewidth=1.4, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("value (sum across entities)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_residual_hist(df: pd.DataFrame, out_path: Path, y_true="y_true", y_pred="y_pred") -> Path:
    res = df[y_true] - df[y_pred]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(res.dropna(), bins=80, color="steelblue", alpha=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Residuals (actual - predicted)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
