"""Public exports for statistical analysis utilities."""

from stats.cohens_d import cohens_d, cohens_d_paired
from stats.multiple_testing import add_fdr_columns
from stats.stats_utils import (
    build_limb_paired_table,
    derive_limb_status,
    split_eye_condition,
    subject_level_mean,
    top_features_by_abs_t
)
from stats.ttests import run_paired_ttests, run_welch_ttests

__all__ = [
    "add_fdr_columns",
    "build_limb_paired_table",
    "cohens_d",
    "cohens_d_paired",
    "derive_limb_status",
    "run_paired_ttests",
    "run_welch_ttests",
    "split_eye_condition",
    "subject_level_mean",
    "top_features_by_abs_t",
]
