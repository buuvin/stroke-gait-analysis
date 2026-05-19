"""Feature-table preprocessing helpers for the classical ML workflows."""

import numpy as np
import pandas as pd


def clean_data(df):
    """Add basic quality flags and drop rows that are not used downstream.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw feature table before task-specific filtering.

    Returns
    -------
    pandas.DataFrame
        Copy of the input table with the quality flag added and combined COP
        rows removed.
    """
    df['avg_diag_line_nan'] = df['avg_diag_line'].isna().astype(int)
    df = df[df["cop_type"] != "combined"]
    return df

def preprocess_limb_data(features_split):
    """Derive limb-status labels and keep the axes used for limb modeling.

    Parameters
    ----------
    features_split : pandas.DataFrame
        Feature table after the subject split has been applied.

    Returns
    -------
    pandas.DataFrame
        Limb-ready table with a ``limb_status`` column and x/y axes only.
    """
    df = features_split.copy()
    df['affected_side'] = df['affected_side'].replace('none', 'healthy')

    df["limb_status"] = "healthy"

    is_stroke = df['affected_side'].ne("healthy")

    affected = df["affected_side"].str.replace("_affected", "", regex=False)

    is_left_plate = df["cop_type"].eq("left")
    is_right_plate = df["cop_type"].eq("right")

    conditions = [
        is_stroke & (df["cop_type"] == affected),
        is_stroke & (df["cop_type"] != affected),
    ]

    choices = [
        "affected",
        "less_affected",
    ]

    df["limb_status"] = np.select(conditions, choices, default="healthy")

    df = df[
    df["axis"].isin(["x", "y"])
    ].copy()

    return df

def preprocess_eyes_data(df_clean):
    """Split the cleaned table into eyes-open and eyes-closed subsets.

    Parameters
    ----------
    df_clean : pandas.DataFrame
        Cleaned feature table containing an ``eye_condition`` column.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``(df_eyes_open, df_eyes_closed)`` subsets for the two conditions.
    """
    df_eyes_open = df_clean[(df_clean["eye_condition"] == "eyes_open")].copy()
    df_eyes_closed = df_clean[(df_clean["eye_condition"] == "eyes_closed")].copy()

    return df_eyes_open, df_eyes_closed
