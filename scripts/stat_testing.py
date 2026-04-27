from pathlib import Path

import pandas as pd

# Allow running as a script from repo root without requiring prior package install.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paths import EXTRACTED, ML_FEATURES_FILE, TTEST_EO_FILE, TTEST_EC_FILE, TTEST_LIMB_FILE, TTEST_EO_PLOT, TTEST_EC_PLOT, TTEST_LIMB_PLOT
from stats.multiple_testing import add_fdr_columns
from stats.stats_utils import derive_limb_status, feature_columns, split_eye_condition, subject_level_mean
from stats.ttests import run_paired_ttests, run_welch_ttests
from stats.anova import compute_feature_anova_tukey, contrast_wise_fdr
from stats.plotting import plot_eyes_ttest, plot_anova_tukey


# def notebook_feature_cols(df, include_limb_status=False):
#     exclude_cols = set(DEFAULT_EXCLUDE_COLUMNS)
#     if include_limb_status:
#         exclude_cols.add("limb_status")
#     return [c for c in df.columns if c not in exclude_cols]


def run_eyes_tests(features_raw, extra_excludes=None):
    df_eo, df_ec = split_eye_condition(features_raw)

    eo_subject = subject_level_mean(df_eo, ["subject_id", "category"])
    ec_subject = subject_level_mean(df_ec, ["subject_id", "category"])
        
    eo_features = feature_columns(eo_subject, extra_excludes=extra_excludes)
    ec_features = feature_columns(ec_subject, extra_excludes=extra_excludes)

    ttest_eo = run_welch_ttests(eo_subject, eo_features)
    ttest_ec = run_welch_ttests(ec_subject, ec_features)

    ttest_eo = add_fdr_columns(ttest_eo)
    ttest_ec = add_fdr_columns(ttest_ec)
    return ttest_eo, ttest_ec


def run_limb_tests(features_raw, extra_excludes=None):
    df_limb = features_raw.copy()
    df_limb["limb_status"] = df_limb.apply(derive_limb_status, axis=1)
    print(len(df_limb), df_limb["cop_type"].value_counts())
    limb_subject = subject_level_mean(df_limb, ["subject_id", "limb_status"])

    limb_features = feature_columns(limb_subject, extra_excludes=extra_excludes)

    wide = limb_subject.pivot(index="subject_id", columns="limb_status", values=limb_features)

    anova_df = compute_feature_anova_tukey(limb_features, limb_subject)
    print(anova_df.columns)
    anova_df = add_fdr_columns(anova_df, p_col="p_anova")
    anova_df = contrast_wise_fdr(anova_df)

    return anova_df


def main():
    features_raw = pd.read_csv(ML_FEATURES_FILE)

    ttest_eo, ttest_ec = run_eyes_tests(features_raw)
    anova_tukey_limbs = run_limb_tests(features_raw, extra_excludes={"limb_status"})

    ttest_eo.to_csv(TTEST_EO_FILE, index=False)
    ttest_ec.to_csv(TTEST_EC_FILE, index=False)
    anova_tukey_limbs.to_csv(TTEST_LIMB_FILE, index=False)

    print(f"Saved eyes-open results: {TTEST_EO_FILE}")
    print(f"Saved eyes-closed results: {TTEST_EC_FILE}")
    print(f"Saved limbs-paired results: {TTEST_LIMB_FILE}")

    print("\nTop 10 eyes-open features by absolute t-statistic:")

    plot_eyes_ttest(ttest_eo, "Top Eyes Open Features", TTEST_EO_PLOT)
    plot_eyes_ttest(ttest_ec, "Top Eyes Closed Features", TTEST_EC_PLOT)
    plot_anova_tukey(anova_tukey_limbs, "Top Limb Features by ANOVA Significance", TTEST_LIMB_PLOT)
    ## fix paths to grab from paths.py and ensure everything runs smoothly
    print(f"Saved eyes-open results: {TTEST_EO_PLOT}")
    print(f"Saved eyes-closed results: {TTEST_EC_PLOT}")


if __name__ == "__main__":
    main()
