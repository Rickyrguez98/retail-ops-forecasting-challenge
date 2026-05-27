"""CLI: assemble final report markdown sections from generated CSVs.

Touches only `reports/experiment_summary.md`. Other narrative files
(`final_report.md`, `model_card.md`) are hand-written; this script
provides the data-driven counterpart and is safe to re-run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_ops_forecasting.config import load_config  # noqa: E402
from retail_ops_forecasting.utils import get_logger, setup_logging  # noqa: E402


def _df_to_md(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_(no rows)_"
    df = df.copy()
    for c in df.select_dtypes(include="float").columns:
        df[c] = df[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    return df.head(max_rows).to_markdown(index=False)


def _read_if_exists(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def main() -> int:
    setup_logging()
    log = get_logger("report")
    cfg = load_config()
    r = cfg.paths.reports_dir

    sections: list[str] = ["# Experiment Summary", ""]

    # Pull demand summary JSON
    p = r / "demand_experiment_summary.json"
    if p.exists():
        summary = json.loads(p.read_text())
        rows = []
        for run, splits in summary.items():
            for split, m in splits.items():
                row = {"run": run, "split": split}
                row.update({k: v for k, v in m.items() if isinstance(v, int | float)})
                rows.append(row)
        sections.append("## Demand — runs")
        sections.append(_df_to_md(pd.DataFrame(rows)))
        sections.append("")

    p = r / "cash_experiment_summary.json"
    if p.exists():
        summary = json.loads(p.read_text())
        rows = []
        for run, splits in summary.items():
            for split, m in splits.items():
                row = {"run": run, "split": split}
                row.update({k: v for k, v in m.items() if isinstance(v, int | float)})
                rows.append(row)
        sections.append("## Cash — runs")
        sections.append(_df_to_md(pd.DataFrame(rows)))
        sections.append("")

    for label, fname in [
        ("Demand by category", "demand_metrics_by_category.csv"),
        ("Demand by region", "demand_metrics_by_region.csv"),
        ("Demand by store format", "demand_metrics_by_store_format.csv"),
        ("Demand by event", "demand_metrics_by_event.csv"),
        ("Cash by region", "cash_metrics_by_region.csv"),
        ("Cash by store format", "cash_metrics_by_store_format.csv"),
        ("Cash by event", "cash_metrics_by_event.csv"),
    ]:
        df = _read_if_exists(r / fname)
        if not df.empty:
            sections.append(f"## {label}")
            sections.append(_df_to_md(df))
            sections.append("")

    (r / "experiment_summary.md").write_text("\n".join(sections))
    log.info("Wrote %s", r / "experiment_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
