"""Shared data-wrangling helpers for statistical analysis workflows."""

import numpy as np
import pandas as pd

from config import DEFAULT_EXCLUDE_COLUMNS


def split_eye_condition(features_df):
    """Split feature table by eyes-open and eyes-closed conditions.

    Parameters
    ----------
    features_df : pandas.DataFrame
        Feature table containing ``cop_type`` and ``eye_condition`` columns.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``(df_eo, df_ec)`` filtered to non-combined COP channels.
    """
    df_eyes = features_df[features_df["cop_type"] != "combined"].copy()
    df_eo = df_eyes[df_eyes["eye_condition"] == "eyes_open"].copy()
    df_ec = df_eyes[df_eyes["eye_condition"] == "eyes_closed"].copy()
    return df_eo, df_ec


def subject_level_mean(df, group_cols):
    """Aggregate numeric columns to subject-level means.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    group_cols : list[str]
        Grouping keys for averaging.

    Returns
    -------
    pandas.DataFrame
        Grouped table with mean of numeric columns.
    """
    return df.groupby(group_cols, as_index=False).mean(numeric_only=True)


def feature_columns(df, extra_excludes=None):
    """Return feature columns excluding metadata and user-specified fields.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    extra_excludes : set[str] or None, default None
        Additional columns to exclude.

    Returns
    -------
    list[str]
        Feature column names.
    """
    excludes = set(DEFAULT_EXCLUDE_COLUMNS)
    if extra_excludes:
        excludes.update(extra_excludes)
    return [c for c in df.columns if c not in excludes]


def derive_limb_status(row):
    """Derive limb-status class label for one row.

    Parameters
    ----------
    row : pandas.Series
        Record containing ``category``, ``affected_side``, and ``cop_type``.

    Returns
    -------
    str or float
        ``"healthy"``, ``"affected"``, ``"unaffected"``, or ``nan``.
    """
    if row["category"] != "stroke":
        return "healthy"

    # Map plate side to affected/unaffected class using each subject's affected side.
    if row["affected_side"] == "left_affected":
        return "affected" if row["cop_type"] == "left" else "unaffected"

    if row["affected_side"] == "right_affected":
        return "affected" if row["cop_type"] == "right" else "unaffected"

    return np.nan


def build_limb_paired_table(stroke_df, feature_cols):
    """Construct wide paired table for affected vs unaffected comparisons.

    Parameters
    ----------
    stroke_df : pandas.DataFrame
        Stroke-only feature table.
    feature_cols : list[str]
        Feature names to include.

    Returns
    -------
    pandas.DataFrame
        Wide table indexed by subject with limb-status columns.
    """
    stroke_df = stroke_df.copy()
    stroke_df["limb_status"] = stroke_df.apply(derive_limb_status, axis=1)
    stroke_subj = subject_level_mean(stroke_df, ["subject_id", "limb_status"])
    # Pivot to wide format so paired tests align affected/unaffected values per subject.
    return stroke_subj.pivot(index="subject_id", columns="limb_status", values=feature_cols)

def get_order(ttest_df):
    """Get feature order by absolute t-statistic (ascending).

    Parameters
    ----------
    ttest_df : pandas.DataFrame
        T-test result table.

    Returns
    -------
    list[str]
        Ordered feature names.
    """
    return (
        ttest_df.set_index("feature")["t_stat"]
        .abs()
        .sort_values(ascending=True)
        .index.tolist()
    )

def top_features_by_abs_t(results_df, top_n=10):
    """Select top features by absolute t-statistic.

    Parameters
    ----------
    results_df : pandas.DataFrame
        T-test result table.
    top_n : int, default 10
        Number of top-ranked features to return.

    Returns
    -------
    pandas.DataFrame
        Subset table with top-ranked features.
    """
    if results_df.empty:
        return results_df.copy()

    out = results_df[np.isfinite(results_df["t_stat"])].copy()
    out["abs_t"] = np.abs(out["t_stat"])
    return out.nlargest(10, "abs_t")[["feature", "t_stat", "p_value"]]
    out = out.sort_values("abs_t", ascending=False).head(top_n)
    return out.drop(columns=["abs_t"]).reset_index(drop=True)

