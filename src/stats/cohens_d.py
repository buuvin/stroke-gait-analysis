"""Effect-size estimators used by hypothesis-testing utilities."""

import numpy as np


def cohens_d(x, y):
    """Compute Cohen's d for two independent samples.

    Parameters
    ----------
    x : array-like
        Values for sample A.
    y : array-like
        Values for sample B.

    Returns
    -------
    float
        Cohen's d using pooled sample variance, or ``nan`` if undefined.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan

    sx2 = np.var(x, ddof=1)
    sy2 = np.var(y, ddof=1)
    pooled_var = ((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2)
    if pooled_var <= 0:
        return np.nan

    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled_var))


def cohens_d_paired(x, y):
    """Compute Cohen's d for paired samples.

    Parameters
    ----------
    x : array-like
        Values for paired condition A.
    y : array-like
        Values for paired condition B.

    Returns
    -------
    float
        Mean difference divided by standard deviation of paired differences,
        or ``nan`` if undefined.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    diff = x[mask] - y[mask]
    if len(diff) < 2:
        return np.nan
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return np.nan
    return float(np.mean(diff) / sd)
