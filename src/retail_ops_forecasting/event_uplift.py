"""
Event-window uplift post-processor for demand forecasts.

The HGB base model systematically under-forecasts peak event days because
it was trained on only one occurrence of each event.  This module applies
a transparent, calibrated multiplier in a window around each known peak:

    adjusted_pred = base_pred * uplift(date)

Three zones per event, each with its own flat multiplier:
  · Pre-event  (window_pre days before anchor): small uplift (~1.10-1.13x).
               Data shows only ~10-13% underestimation in this zone.
  · Anchor days (peak): large uplift (2.8-2.9x).
               Model is 60-70% short; multiplier is learned or configured.
  · Post-event (window_post days after anchor): usually 1.0 (no uplift).
               Model recovers quickly or even over-predicts after the event.

Events handled
--------------
* Buen Fin      — anchor dates from calendar `is_buen_fin` flag; peak
                  multiplier learned automatically from val residuals.
* Dec 24-25     — fixed annual dates; configurable multipliers (default 2.8x peak).
* Dec 31        — fixed annual date; configurable (default 1.8x peak);
                  no post-event window (model over-predicts Jan 1+).
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _expand_window(
    anchor_dates: list[pd.Timestamp],
    peak_mult: float,
    pre_mult: float,
    post_mult: float,
    window_pre: int,
    window_post: int,
) -> Dict[pd.Timestamp, float]:
    """Build {date: multiplier} using separate flat rates for pre/peak/post.

    For dates inside two overlapping windows, the max multiplier wins
    (conservative: never reduce a predicted peak).

    Parameters
    ----------
    anchor_dates : core event dates (e.g. all Buen Fin days, Dec 24 and 25).
    peak_mult    : multiplier applied to anchor dates themselves.
    pre_mult     : flat multiplier for the `window_pre` days before the first anchor.
    post_mult    : flat multiplier for the `window_post` days after the last anchor.
    """
    if not anchor_dates:
        return {}

    result: Dict[pd.Timestamp, float] = {}
    anchor_set = set(anchor_dates)
    first_anchor = min(anchor_dates)
    last_anchor  = max(anchor_dates)

    # Anchor days
    for anchor in anchor_dates:
        result[anchor] = max(result.get(anchor, 1.0), peak_mult)

    # Pre-event window (relative to first anchor day)
    if pre_mult > 1.0 and window_pre > 0:
        for d in range(1, window_pre + 1):
            day = first_anchor - timedelta(days=d)
            if day not in anchor_set:
                result[day] = max(result.get(day, 1.0), pre_mult)

    # Post-event window (relative to last anchor day)
    if post_mult > 1.0 and window_post > 0:
        for d in range(1, window_post + 1):
            day = last_anchor + timedelta(days=d)
            if day not in anchor_set:
                result[day] = max(result.get(day, 1.0), post_mult)

    return result


# ── public API ────────────────────────────────────────────────────────────────

def learn_buen_fin_multiplier(
    val_df: pd.DataFrame,
    target_col: str,
    pred_col: str,
    buen_fin_dates: list[pd.Timestamp],
) -> float:
    """Compute peak multiplier for Buen Fin from validation residuals.

    Uses the daily aggregate actual/predicted ratio on the core Buen Fin days
    (not the pre/post window).  Clamps to [1.0, 5.0] for safety.
    """
    if not buen_fin_dates or target_col not in val_df.columns:
        logger.warning("Cannot learn Buen Fin multiplier — returning 1.0")
        return 1.0

    bf_set = set(buen_fin_dates)
    subset = val_df[val_df["date"].isin(bf_set)].copy()
    if subset.empty:
        logger.warning("No Buen Fin rows in val_df — returning 1.0")
        return 1.0

    daily = subset.groupby("date").agg({target_col: "sum", pred_col: "sum"})
    # ratio per day, then average
    ratios = (daily[target_col] / daily[pred_col].clip(lower=1.0)).values
    mult = float(np.mean(ratios))
    mult = float(np.clip(mult, 1.0, 5.0))
    logger.info("Learned Buen Fin peak multiplier from val: %.3fx", mult)
    return mult


def build_uplift_map(
    calendar_df: pd.DataFrame,
    cfg,
    val_df: Optional[pd.DataFrame] = None,
    target_col: Optional[str] = None,
    pred_col: str = "y_pred",
) -> Dict[pd.Timestamp, float]:
    """Build a {date → multiplier} map for all configured event windows.

    Parameters
    ----------
    calendar_df : full calendar DataFrame (must have `date`, `is_buen_fin`).
    cfg         : project Config (loaded via load_config()).
    val_df      : validation predictions (needed to learn Buen Fin multiplier).
    target_col  : actual target column name in val_df.
    pred_col    : prediction column name in val_df.

    Returns
    -------
    dict mapping pd.Timestamp → float uplift multiplier (1.0 = no change).
    """
    eu = cfg.event_uplift  # EventUpliftCfg dataclass

    if not eu.enabled:
        logger.info("Event uplift disabled — returning empty map")
        return {}

    uplift_map: Dict[pd.Timestamp, float] = {}

    # ── 1. Buen Fin ──────────────────────────────────────────────────────────
    bf_dates = sorted(
        calendar_df[calendar_df["is_buen_fin"].fillna(False)]["date"].tolist()
    )
    if bf_dates:
        if eu.buen_fin_peak_multiplier is None and val_df is not None and target_col:
            bf_mult = learn_buen_fin_multiplier(val_df, target_col, pred_col, bf_dates)
        else:
            bf_mult = eu.buen_fin_peak_multiplier or 2.9
        window = _expand_window(
            bf_dates, bf_mult,
            pre_mult=eu.buen_fin_pre_multiplier,
            post_mult=eu.buen_fin_post_multiplier,
            window_pre=eu.buen_fin_window_pre,
            window_post=eu.buen_fin_window_post,
        )
        for d, m in window.items():
            uplift_map[d] = max(uplift_map.get(d, 1.0), m)
        logger.info(
            "Buen Fin uplift: peak=%.2fx pre=%.2fx over %d dates (pre=%d post=%d days)",
            bf_mult, eu.buen_fin_pre_multiplier, len(window),
            eu.buen_fin_window_pre, eu.buen_fin_window_post,
        )

    # ── 2. Dec 24-25 ─────────────────────────────────────────────────────────
    years = calendar_df["date"].dt.year.unique()
    dec_anchors = [
        pd.Timestamp(int(y), eu.dec_24_25_anchor_month, day)
        for y in years
        for day in eu.dec_24_25_anchor_days
        if pd.Timestamp(int(y), eu.dec_24_25_anchor_month, day) in set(calendar_df["date"])
    ]
    if dec_anchors:
        window = _expand_window(
            dec_anchors, eu.dec_24_25_peak_multiplier,
            pre_mult=eu.dec_24_25_pre_multiplier,
            post_mult=eu.dec_24_25_post_multiplier,
            window_pre=eu.dec_24_25_window_pre,
            window_post=eu.dec_24_25_window_post,
        )
        for d, m in window.items():
            uplift_map[d] = max(uplift_map.get(d, 1.0), m)
        logger.info(
            "Dec 24-25 uplift: peak=%.2fx pre=%.2fx over %d dates",
            eu.dec_24_25_peak_multiplier, eu.dec_24_25_pre_multiplier, len(window),
        )

    # ── 3. Dec 31 ────────────────────────────────────────────────────────────
    dec31_anchors = [
        pd.Timestamp(int(y), eu.dec_31_anchor_month, day)
        for y in years
        for day in eu.dec_31_anchor_days
        if pd.Timestamp(int(y), eu.dec_31_anchor_month, day) in set(calendar_df["date"])
    ]
    if dec31_anchors:
        window = _expand_window(
            dec31_anchors, eu.dec_31_peak_multiplier,
            pre_mult=eu.dec_31_pre_multiplier,
            post_mult=eu.dec_31_post_multiplier,
            window_pre=eu.dec_31_window_pre,
            window_post=eu.dec_31_window_post,
        )
        for d, m in window.items():
            uplift_map[d] = max(uplift_map.get(d, 1.0), m)
        logger.info(
            "Dec 31 uplift: peak=%.2fx pre=%.2fx over %d dates",
            eu.dec_31_peak_multiplier, eu.dec_31_pre_multiplier, len(window),
        )

    logger.info("Total event uplift dates in map: %d", len(uplift_map))
    return uplift_map


def apply_uplift(
    preds_df: pd.DataFrame,
    pred_col: str,
    uplift_map: Dict[pd.Timestamp, float],
) -> pd.DataFrame:
    """Apply event-window uplift to a predictions DataFrame.

    Multiplies `pred_col` by the date's uplift factor (1.0 if not in map).
    Clips result to ≥ 0.  Returns a copy.
    """
    if not uplift_map:
        return preds_df.copy()

    df = preds_df.copy()
    mult_series = df["date"].map(lambda d: uplift_map.get(pd.Timestamp(d), 1.0))
    adjusted = df[pred_col] * mult_series
    n_affected = int((mult_series > 1.0).sum())
    logger.info(
        "Event uplift applied to %d rows (pred_col='%s')", n_affected, pred_col
    )
    df[pred_col] = adjusted.clip(lower=0.0)
    return df
