"""Build reporting tables and figures.

These helpers are intentionally simple — they accept a long-format
predictions dataframe and write CSVs + figures to `reports/`.
"""

from __future__ import annotations

from pathlib import Path

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


def write_overall(
    df: pd.DataFrame, out_path: Path, y_true: str = "y_true", y_pred: str = "y_pred"
) -> dict:
    summary = mt.summarize(df[y_true], df[y_pred])
    pd.DataFrame([summary]).to_csv(out_path, index=False)
    return summary


def write_breakdowns(
    df: pd.DataFrame,
    figures_dir: Path,
    reports_dir: Path,
    groups: dict[str, list[str]],
    y_true: str = "y_true",
    y_pred: str = "y_pred",
    prefix: str = "demand",
) -> dict[str, Path]:
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


def plot_error_by_category(
    df: pd.DataFrame,
    out_path: Path,
    y_true: str = "y_true",
    y_pred: str = "y_pred",
    category_col: str = "category",
) -> Path:
    """Boxplot of signed residuals (actual - predicted) per category.

    Used to spot categories where the model systematically over- or under-forecasts.
    """
    residuals = (df[y_true] - df[y_pred]).rename("residual")
    plot_df = pd.concat([df[category_col].rename("category"), residuals], axis=1).dropna()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    categories = sorted(plot_df["category"].unique())
    data = [plot_df.loc[plot_df["category"] == c, "residual"].values for c in categories]
    ax.boxplot(data, labels=categories, showfliers=False, patch_artist=True)
    ax.axhline(0, color="k", lw=0.8, linestyle="--")
    ax.set_title("Demand residuals by category (test set)")
    ax.set_ylabel("residual (actual - predicted)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_cash_coverage_per_store(
    df: pd.DataFrame,
    out_path: Path,
    y_true: str = "y_true",
    rec_col: str = "recommended_cash",
    store_col: str = "store_id",
    target_coverage: float = 0.90,
) -> Path:
    """Histogram of per-store coverage (fraction of days where recommended >= actual).

    Used to verify the cash buffer rule holds per store, not just in aggregate.
    """
    if rec_col not in df.columns:
        # Backstop: derive from y_pred + a per-store residual P90 if recommended column is absent.
        residuals = (df[y_true] - df["y_pred"]).abs()
        p90 = residuals.groupby(df[store_col]).quantile(0.90)
        df = df.copy()
        df[rec_col] = df["y_pred"].clip(lower=0) + df[store_col].map(p90)
    covered = (df[rec_col] >= df[y_true]).astype(int)
    coverage = covered.groupby(df[store_col]).mean()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(coverage.values, bins=30, color="seagreen", alpha=0.8, edgecolor="white")
    ax.axvline(
        target_coverage, color="red", lw=1.2, linestyle="--", label=f"target {target_coverage:.0%}"
    )
    ax.set_title(f"Cash coverage per store (mean = {coverage.mean():.3f})")
    ax.set_xlabel("fraction of days where recommended ≥ actual")
    ax.set_ylabel("number of stores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
