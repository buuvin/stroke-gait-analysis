"""Time-domain feature extraction for COP time-series signals."""

import numpy as np
from scipy.stats import skew, kurtosis

from features.features_utils import rms, sample_entropy, hurst_rs, dfa_alpha

def extract_time_domain_features(data):
    """Compute descriptive and nonlinear time-domain features.

    Parameters
    ----------
    data : array-like
        One-dimensional COP signal.

    Returns
    -------
    dict
        Time-domain feature values for downstream ML/statistical analysis.

    Notes
    -----
    Includes central tendency, dispersion, distribution-shape, and nonlinear
    complexity metrics used in this project.
    """
    data = np.asarray(data, dtype=np.float64)
    return {
            'mean': data.mean(),
            'std': data.std(),
            'var': data.var(),
            'min': data.min(),
            'max': data.max(),
            'range': data.max() - data.min(),
            'median': np.median(data),
            'skew': skew(data),
            'kurt': kurtosis(data),
            'rms': rms(data),
            'quantile_25': np.percentile(data, 25),
            'quantile_50': np.percentile(data, 50),
            'quantile_75': np.percentile(data, 75),
            # Nonlinear metrics capture complexity beyond simple moments.
            'samp_ent': sample_entropy(data),
            'hurst': hurst_rs(data),
            'dfa': dfa_alpha(data)
        }