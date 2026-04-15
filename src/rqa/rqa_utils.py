from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric

from pyrqa.computation import RPComputation

import numpy as np
import time
import random
import csv

import os
from pathlib import Path

def determine_group_from_path(file_path):
    """
    Determine the group (category, affected_side, cop_type, eye_condition, axis) from file path.
    
    Args:
        file_path: Path object to the file
    
    Returns:
        tuple: (category, affected_side, cop_type, eye_condition, axis)
    """
    parts = file_path.parts
    
    # Find indices in path
    if 'healthy' in parts:
        category = 'healthy'
        affected_side = ''
        # Path structure: .../healthy/COP_type/eye_condition/axis/file.txt
        healthy_idx = parts.index('healthy')
        cop_type = parts[healthy_idx + 1]  # COP1, COP2, or COP
        eye_condition = parts[healthy_idx + 2]
        axis = parts[healthy_idx + 3]
    elif 'stroke' in parts:
        category = 'stroke'
        # Path structure: .../stroke/affected_side/COP_type/eye_condition/axis/file.txt
        stroke_idx = parts.index('stroke')
        affected_side = parts[stroke_idx + 1]  # left_affected or right_affected
        cop_type = parts[stroke_idx + 2]  # COP1, COP2, or COP
        eye_condition = parts[stroke_idx + 3]
        axis = parts[stroke_idx + 4]
    else:
        return None
    
    return (category, affected_side, cop_type, eye_condition, axis)

def compute_rr(data_points, n, tau, eps, theiler_corrector = 1):
    # create RP generator
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

    # recurrence rate
    RR = R.sum() / R.size
    return RR

# Fixed version of find_opt_neighborhood - corrects parameter mismatch
def find_opt_neighborhood(data_points, n, tau, target_rr=0.05):
    """
    Find optimal neighborhood radius using binary search to achieve target recurrence rate.
    
    Args:
        data_points: numpy array of time series data
        n: embedding dimension
        tau: time delay
        target_rr: target recurrence rate (default 0.05)
    
    Returns:
        Optimal neighborhood radius
    """
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