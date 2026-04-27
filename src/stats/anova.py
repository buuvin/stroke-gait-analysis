"""ANOVA and Tukey-HSD utilities for limb-group feature comparisons."""

from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd

def _pair_pvalue(tukey_df, a, b):
    """Extract one pairwise Tukey adjusted p-value from summary table.

    Parameters
    ----------
    tukey_df : pandas.DataFrame
        Pairwise-comparison table from ``pairwise_tukeyhsd`` summary output.
    a : str
        First group label.
    b : str
        Second group label.

    Returns
    -------
    float
        Pairwise adjusted p-value when available, else ``nan``.
    """
    mask = (
        ((tukey_df["group1"] == a) & (tukey_df["group2"] == b))
        | ((tukey_df["group1"] == b) & (tukey_df["group2"] == a))
    )
    if not mask.any():
        return np.nan
    return float(tukey_df.loc[mask, "p-adj"].iloc[0])


def compute_feature_anova_tukey(limb_features, limb_subject):
    """Compute per-feature one-way ANOVA and Tukey-HSD contrasts.

    Parameters
    ----------
    limb_features : list[str]
        Feature names to test.
    limb_subject : pandas.DataFrame
        Subject-level table containing ``limb_status`` and feature columns.

    Returns
    -------
    pandas.DataFrame
        Per-feature ANOVA statistics, group means, and Tukey contrast p-values.

    Notes
    -----
    Features with insufficient sample size in any limb-status group are skipped.
    """
    results = []
    for feat in limb_features:
        # Convert each feature to long form so ANOVA/Tukey can operate by group label.
        long_df = limb_subject[["limb_status", feat]].rename(columns={feat: "value"}).dropna()

        healthy_vals = long_df.loc[long_df["limb_status"] == "healthy", "value"].values
        affected_vals = long_df.loc[long_df["limb_status"] == "affected", "value"].values
        unaffected_vals = long_df.loc[long_df["limb_status"] == "unaffected", "value"].values

        # Require at least two observations per class for stable ANOVA estimation.
        if min(len(healthy_vals), len(affected_vals), len(unaffected_vals)) < 2:
            continue

        try:
            f_stat, p_anova = f_oneway(healthy_vals, affected_vals, unaffected_vals)
        except Exception:
            continue

        row = {
            "feature": feat,
            "f_stat": float(f_stat),
            "p_anova": float(p_anova),
            "mean_healthy": float(np.mean(healthy_vals)),
            "mean_affected": float(np.mean(affected_vals)),
            "mean_unaffected": float(np.mean(unaffected_vals)),
        }

        try:
            tukey = pairwise_tukeyhsd(endog=long_df["value"], groups=long_df["limb_status"], alpha=0.05)
            # Convert Tukey summary table to numeric p-values for programmatic extraction.
            tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
            tukey_df["p-adj"] = pd.to_numeric(tukey_df["p-adj"], errors="coerce")

            row["p_affected_vs_healthy_tukey"] = _pair_pvalue(tukey_df, "affected", "healthy")
            row["p_unaffected_vs_healthy_tukey"] = _pair_pvalue(tukey_df, "unaffected", "healthy")
            row["p_affected_vs_unaffected_tukey"] = _pair_pvalue(tukey_df, "affected", "unaffected")
        except Exception:
            row["p_affected_vs_healthy_tukey"] = np.nan
            row["p_unaffected_vs_healthy_tukey"] = np.nan
            row["p_affected_vs_unaffected_tukey"] = np.nan

        results.append(row)
    
    anova_tukey = pd.DataFrame(results)
    return anova_tukey

def contrast_wise_fdr(anova_tukey):
    """Apply FDR correction separately for each Tukey contrast column.

    Parameters
    ----------
    anova_tukey : pandas.DataFrame
        Output table from ``compute_feature_anova_tukey`` with Tukey p-columns.

    Returns
    -------
    pandas.DataFrame
        Input table with added FDR-adjusted contrast columns and significance
        flags.
    """
    tukey_cols = [
        "p_affected_vs_healthy_tukey",
        "p_unaffected_vs_healthy_tukey",
        "p_affected_vs_unaffected_tukey",
    ]
    for col in tukey_cols:
        # Correct each contrast family across features (not across different contrasts).
        valid = anova_tukey[col].notna()
        p_col_fdr = np.full(len(anova_tukey), np.nan)
        sig_col = np.full(len(anova_tukey), False)
        if valid.sum() > 0:
            rej_col, p_adj_col, _, _ = multipletests(anova_tukey.loc[valid, col].values, alpha=0.05, method="fdr_bh")
            p_col_fdr[valid.to_numpy()] = p_adj_col
            sig_col[valid.to_numpy()] = rej_col
        anova_tukey[f"{col}_fdr"] = p_col_fdr
        anova_tukey[f"{col}_significant_fdr"] = sig_col

    anova_tukey = anova_tukey.sort_values("p_fdr", ascending=True).reset_index(drop=True)
    return anova_tukey
