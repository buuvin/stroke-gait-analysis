"""Multiple-testing correction helpers for feature-level p-values."""

from statsmodels.stats.multitest import multipletests


def add_fdr_columns(results_df, p_col="p_value", alpha=0.05, method="fdr_bh"):
    """Append FDR-adjusted p-values and significance flags.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Table containing raw p-values.
    p_col : str, default "p_value"
        Column name with raw p-values.
    alpha : float, default 0.05
        Family-wise significance threshold used for rejection flags.
    method : str, default "fdr_bh"
        Method passed to ``statsmodels.stats.multitest.multipletests``.

    Returns
    -------
    pandas.DataFrame
        Copy of input table with ``p_fdr`` and ``significant_fdr`` columns.
    """
    out = results_df.copy()
    if len(out) == 0:
        out["p_fdr"] = []
        out["significant_fdr"] = []
        return out

    reject, p_fdr, _, _ = multipletests(out[p_col].values, alpha=alpha, method=method)
    out["p_fdr"] = p_fdr
    out["significant_fdr"] = reject
    return out
