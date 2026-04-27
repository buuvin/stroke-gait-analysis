"""Parametric t-test helpers for feature-level hypothesis testing."""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel

from stats.cohens_d import cohens_d, cohens_d_paired


def run_welch_ttests(subject_df, feature_cols, group_col="category", a_label="healthy", b_label="stroke"):
    """Run Welch two-sample t-tests for each feature.

    Parameters
    ----------
    subject_df : pandas.DataFrame
        Subject-level feature table.
    feature_cols : list[str]
        Feature names to test.
    group_col : str, default "category"
        Column containing group labels.
    a_label : str, default "healthy"
        First group label.
    b_label : str, default "stroke"
        Second group label.

    Returns
    -------
    pandas.DataFrame
        Per-feature Welch t-test statistics and effect sizes.
    """
    rows = []
    for feat in feature_cols:
        a_vals = subject_df[subject_df[group_col] == a_label][feat].dropna()
        b_vals = subject_df[subject_df[group_col] == b_label][feat].dropna()

        if len(a_vals) < 2 or len(b_vals) < 2:
            continue

        t_stat, p_val = ttest_ind(a_vals, b_vals, equal_var=False)
        rows.append(
            {
                "feature": feat,
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                f"mean_{a_label}": float(a_vals.mean()),
                f"mean_{b_label}": float(b_vals.mean()),
                f"std_{a_label}": float(a_vals.std(ddof=1)),
                f"std_{b_label}": float(b_vals.std(ddof=1)),
                "cohens_d": cohens_d(a_vals.values, b_vals.values),
            }
        )

    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True) if rows else pd.DataFrame()


def run_paired_ttests(wide_df, feature_cols, a_col="affected", b_col="unaffected"):
    """Run paired t-tests for affected vs unaffected limbs.

    Parameters
    ----------
    wide_df : pandas.DataFrame
        Wide table with MultiIndex columns ``(feature, limb_status)``.
    feature_cols : list[str]
        Feature names to test.
    a_col : str, default "affected"
        First paired limb-status column.
    b_col : str, default "unaffected"
        Second paired limb-status column.

    Returns
    -------
    pandas.DataFrame
        Per-feature paired t-test statistics and paired effect sizes.
    """
    rows = []
    for feat in feature_cols:
        if (feat, a_col) not in wide_df.columns or (feat, b_col) not in wide_df.columns:
            continue

        # Inner join enforces matched within-subject pairs before paired t-test.
        paired = wide_df[(feat, a_col)].to_frame(a_col).join(
            wide_df[(feat, b_col)].to_frame(b_col),
            how="inner",
        ).dropna()

        if len(paired) < 2:
            continue

        t_stat, p_val = ttest_rel(paired[a_col], paired[b_col])
        rows.append(
            {
                "feature": feat,
                "n_subjects": int(len(paired)),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                f"mean_{a_col}": float(paired[a_col].mean()),
                f"mean_{b_col}": float(paired[b_col].mean()),
                "cohens_d_paired": cohens_d_paired(paired[a_col].values, paired[b_col].values),
            }
        )

    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True) if rows else pd.DataFrame()
