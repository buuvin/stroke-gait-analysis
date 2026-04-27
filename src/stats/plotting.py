"""Plotting utilities for t-test and ANOVA/Tukey statistical summaries."""

import pandas as pd
import numpy as np

from stats.stats_utils import get_order, top_features_by_abs_t



def plot_eyes_ttest(ttest_df, title, filename):
    """Create and save top-feature bar plot for eyes-condition t-tests.

    Parameters
    ----------
    ttest_df : pandas.DataFrame
        Per-feature t-test result table including ``feature``, ``t_stat``, and
        ``p_value`` columns.
    title : str
        Figure title.
    filename : str or pathlib.Path
        Output image path.

    Returns
    -------
    None
        Figure is saved to ``filename``.

    Notes
    -----
    A plotnine implementation is intentionally retained as commented reference;
    active execution currently uses matplotlib.
    """
    plot_df = top_features_by_abs_t(ttest_df, top_n=10)
    # Keep plotting order tied to effect size so bars are visually interpretable.
    order = get_order(plot_df)
    plot_df["feature"] = pd.Categorical(plot_df["feature"], categories=order, ordered=True)
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 5))
    order = plot_df["feature"].cat.categories.tolist()
    x = np.arange(len(order))
    vals = plot_df.set_index("feature").loc[order]
    pvals = vals["p_value"].replace(0, 1e-10)
    # Clamp color range to avoid near-zero p-values collapsing the full color scale.
    vmin, vmax = max(1e-10, pvals.min()), min(0.1, pvals.max())
    if vmin >= vmax:
        vmax = vmin + 0.01
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis_r

    ax.bar(x, vals["t_stat"].values, width=0.75, color=cmap(norm(pvals.values)))
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("t-statistic")
    ax.set_xlabel("Feature")
    ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="p-value")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

def p_stars(p):
    """Map p-values to significance-star annotation.

    Parameters
    ----------
    p : float
        P-value.

    Returns
    -------
    str
        ``"***"``, ``"**"``, ``"*"``, or empty string.
    """
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

def plot_anova_tukey(anova_tukey, title, filename):
    """Create and save ANOVA/Tukey heatmap for top-ranked limb features.

    Parameters
    ----------
    anova_tukey : pandas.DataFrame
        Table with ANOVA and contrast-wise FDR-adjusted Tukey p-values.
    title : str
        Figure title.
    filename : str or pathlib.Path
        Output image path.

    Returns
    -------
    None
        Figure is saved to ``filename``.
    """
    plot_df = anova_tukey.dropna(subset=["p_fdr"]).sort_values("p_fdr").head(10).copy()


    if not plot_df.empty:
        contrast_cols = [
            ("p_affected_vs_healthy_tukey_fdr", "affected_vs_healthy"),
            ("p_unaffected_vs_healthy_tukey_fdr", "unaffected_vs_healthy"),
            ("p_affected_vs_unaffected_tukey_fdr", "affected_vs_unaffected"),
        ]

        long_plot = []
        for _, row in plot_df.iterrows():
            for pcol, contrast in contrast_cols:
                pval = row.get(pcol)
                if pd.notna(pval):
                    long_plot.append({
                        "feature": row["feature"],
                        "contrast": contrast,
                        "neg_log10_p": -np.log10(pval + 1e-300),
                        "p_stars": p_stars(pval),
                        "p_label": f"{p_stars(pval)} {pval:.2e}".strip(),
                    })

        plot_long_df = pd.DataFrame(long_plot)
        order = (
            plot_df.set_index("feature")["p_fdr"]
            .sort_values(ascending=True)
            .index.tolist()
        )
        plot_long_df["feature"] = pd.Categorical(plot_long_df["feature"], categories=order, ordered=True)
        plot_long_df["contrast"] = pd.Categorical(
            plot_long_df["contrast"],
            categories=["affected_vs_healthy", "unaffected_vs_healthy", "affected_vs_unaffected"],
            ordered=True,
        )

        # Use a percentile cap so a single extreme value does not dominate heatmap contrast.
        vmax = float(np.nanpercentile(plot_long_df["neg_log10_p"], 95))
        vmax = max(1.0, min(vmax, 3.0))

        import matplotlib.pyplot as plt
        import seaborn as sns

        heat = plot_long_df.pivot(index="feature", columns="contrast", values="neg_log10_p")
        annot = plot_long_df.pivot(index="feature", columns="contrast", values="p_label")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            heat,
            annot=annot,
            fmt="",
            cmap="Blues",
            ax=ax,
            vmin=0,
            vmax=vmax,
            cbar_kws={"label": "-log10(FDR-adjusted p-value)"},
        )
        ax.set_title(title)
        ax.set_xlabel("Tukey HSD Contrast")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()