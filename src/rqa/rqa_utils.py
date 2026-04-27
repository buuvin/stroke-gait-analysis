"""Shared RQA utilities for group parsing and neighborhood optimization."""

from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric

from pyrqa.computation import RPComputation

import pandas as pd
import numpy as np
import time
import random
import csv

import os
from pathlib import Path


def load_optimal_params(optimal_params_file):
    """Load subgroup optimal parameters from CSV.

    Parameters
    ----------
    optimal_params_file : str or pathlib.Path
        Path to ``optimal_params.csv``.

    Returns
    -------
    dict
        Mapping from subgroup key
        ``(category, affected_side, cop_type, eye_condition, axis)``
        to parameter tuple ``(tau, n, neighborhood)``.
    """
    df = pd.read_csv(optimal_params_file)
    params_dict = {}
    
    for _, row in df.iterrows():
        key = (
            row['category'],
            str(row['affected_side']) if pd.notna(row['affected_side']) else '',
            row['cop_type'],
            row['eye'],
            row['axis']
        )
        params_dict[key] = (int(row['tau']), int(row['n']), float(row['neighborhood']))
    
    return params_dict

def determine_group_from_path(file_path):
    """Infer subgroup metadata from a sorted-data file path.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to one sorted COP signal file.

    Returns
    -------
    tuple or None
        ``(category, affected_side, cop_type, eye_condition, axis)`` if the
        expected folder structure is present, else ``None``.
    """
    # Parse directory hierarchy to recover the subgroup key used throughout
    # parameter estimation and batch RQA computation.
    parts = file_path.parts
    
    if 'healthy' in parts:
        category = 'healthy'
        affected_side = ''
        healthy_idx = parts.index('healthy')
        cop_type = parts[healthy_idx + 1]
        eye_condition = parts[healthy_idx + 2]
        axis = parts[healthy_idx + 3]
    elif 'stroke' in parts:
        category = 'stroke'
        stroke_idx = parts.index('stroke')
        affected_side = parts[stroke_idx + 1]
        cop_type = parts[stroke_idx + 2]
        eye_condition = parts[stroke_idx + 3]
        axis = parts[stroke_idx + 4]
    else:
        return None
    
    return (category, affected_side, cop_type, eye_condition, axis)

def compute_rr(data_points, n, tau, eps, theiler_corrector = 1):
    """Compute recurrence rate for one candidate neighborhood radius.

    Parameters
    ----------
    data_points : array-like
        One-dimensional COP signal.
    n : int
        Embedding dimension.
    tau : int
        Embedding delay.
    eps : float
        Candidate neighborhood radius.
    theiler_corrector : int, default 1
        Theiler window used by pyrqa.

    Returns
    -------
    float
        Recurrence rate for the configured embedding and radius.
    """
    time_series = TimeSeries(data_points,
                            embedding_dimension = n,
                            time_delay = tau)
    settings = Settings(time_series,
                        analysis_type = Classic,
                        neighbourhood = FixedRadius(eps),
                        similarity_measure = EuclideanMetric,
                        theiler_corrector = theiler_corrector)
    computation = RPComputation.create(settings)
    result = computation.run()

    R = result.recurrence_matrix

    RR = R.sum() / R.size
    return RR

# Binary search over epsilon to match a target recurrence rate.
def find_opt_neighborhood(data_points, n, tau, target_rr=0.05):
    """Find neighborhood radius that approximates a target recurrence rate.

    Parameters
    ----------
    data_points : array-like
        One-dimensional COP signal.
    n : int
        Embedding dimension.
    tau : int
        Embedding delay.
    target_rr : float, default 0.05
        Target recurrence rate used in binary search.

    Returns
    -------
    float
        Estimated neighborhood radius.

    Notes
    -----
    Uses a fixed 10-iteration binary search over ``[0, 1]``.
    """
    # Search interval chosen for normalized COP signals.
    low, high = 0, 1.0
    for i in range(10):
        mid = (low + high) / 2
        rr = compute_rr(data_points, n, tau, mid)

        if rr < target_rr:
            low = mid
        else:
            high = mid
    print("RR found: ", rr)
    return mid