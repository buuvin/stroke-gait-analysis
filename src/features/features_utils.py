"""Utility functions for metadata parsing and nonlinear COP signal features."""

from pathlib import Path
import numpy as np

patient_side = {
    "CK01":	"left",
    "CK02":	"left",
    "CK03":	"right",
    "CK05":	"left",
    "CK06":	"right",
    "CK07":	"right",
    "CK08":	"right",
    "CK09":	"left",
    "CK10":	"right",
    "CK11":	"left",
    "CK12":	"right",
    "CK13":	"right",
    "CK14":	"left",
    "CK15":	"left",
}

def parse_metadata_from_filename(file_path, patient_side):
    """Parse subject/condition metadata from a COP filename.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Name or path of one COP text file.
    patient_side : dict
        Mapping from stroke subject ID to affected limb side.

    Returns
    -------
    dict
        Metadata fields used for feature records and merge keys.

    Notes
    -----
    Healthy subjects are encoded with an empty ``affected_side`` string.
    """
    name = Path(file_path).name

    tokens = name.split("_")
    subject_id = tokens[0]  # CK09 or SUP07


    # ---- Main label ----
    if subject_id.startswith("CK"):
        label = 1  # stroke
        condition = "stroke"
        affected_side = patient_side[subject_id] + "_affected"
    elif subject_id.startswith("SUP"):
        label = 0  # healthy
        condition = "healthy"
        # Keep empty string for healthy to match existing metadata conventions.
        affected_side = ""
    else:
        raise ValueError(f"Unknown subject type: {subject_id}")


    # ---- Eyes open / closed ----
    if "PSEO" in name:
        eyes = "eyes_open"
    elif "PSEC" in name:
        eyes = "eyes_closed"
    else:
        eyes = "unknown"

    # ---- COP plate ----
    if "COP1" in name:
        cop_plate = "left"
    elif "COP2" in name:
        cop_plate = "right"
    else:
        cop_plate = "combined"

    if name.endswith("_X.txt"):
        axis = "x"
    elif name.endswith("_Y.txt"):
        axis = "y"
    else:
        axis = "resultant"

    return {
        "filename": name,
        "subject_id": subject_id,
        "label": label,
        "category": condition,
        "eye_condition": eyes,
        "cop_type": cop_plate,
        "affected_side": affected_side,
        "axis": axis
    }

def band_power(freq, psd, fmin, fmax):
    """Integrate spectral power within a frequency band.

    Parameters
    ----------
    freq : numpy.ndarray
        Frequency bins.
    psd : numpy.ndarray
        Power spectral density values aligned to ``freq``.
    fmin : float
        Lower frequency bound.
    fmax : float
        Upper frequency bound.

    Returns
    -------
    float
        Integrated band power between ``fmin`` and ``fmax``.
    """
    mask = (freq >= fmin) & (freq <= fmax)
    if np.count_nonzero(mask) == 0:
        return 0.0
    if np.count_nonzero(mask) == 1:
        # single-bin band: approximate area = PSD * df
        df = freq[1] - freq[0] if len(freq) > 1 else 0.0
        return float(psd[mask][0] * df)
    return float(np.trapezoid(psd[mask], freq[mask]))

def rms(data):
    """Compute root-mean-square amplitude.

    Parameters
    ----------
    data : array-like
        Signal values.

    Returns
    -------
    float
        Root-mean-square value of the input signal.
    """
    return np.sqrt(np.mean(data**2))

def ApEn(U, m, r):
    """Compute approximate entropy (ApEn).

    Parameters
    ----------
    U : array-like
        Input signal.
    m : int
        Embedding dimension for matching templates.
    r : float
        Tolerance threshold.

    Returns
    -------
    float
        Approximate entropy estimate.
    """
    N = len(U)

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [U[i:i + m] for i in range(N - m + 1)]
        C = []
        for x_i in x:
            C.append(sum(_maxdist(x_i, x_j) <= r for x_j in x) / (N - m + 1))
        return np.log(np.mean(C))

    return abs(_phi(m) - _phi(m + 1))

def sample_entropy(data, m=2, r_ratio=0.2):
    """Compute sample entropy for a one-dimensional signal.

    Parameters
    ----------
    data : array-like
        Input signal.
    m : int, default 2
        Embedding dimension.
    r_ratio : float, default 0.2
        Tolerance ratio relative to signal standard deviation.

    Returns
    -------
    float
        Sample entropy estimate, or ``nan`` when counts are insufficient.
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.size
    # Need at least m+2 points to compare templates of length m+1.
    if n <= m + 1:
        return np.nan

    r = r_ratio * np.std(data)
    # Constant signals have zero tolerance and entropy is treated as zero.
    if r <= 0:
        return 0.0

    def _phi(mm):
        templates = np.array([data[i:i + mm] for i in range(n - mm + 1)])
        count = 0
        total = 0
        for i in range(len(templates) - 1):
            d = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            c = np.sum(d <= r)
            count += c
            total += len(d)
        return count, total

    b_count, b_total = _phi(m)
    a_count, a_total = _phi(m + 1)

    if b_count == 0 or a_count == 0:
        return np.nan

    b_prob = b_count / max(b_total, 1)
    a_prob = a_count / max(a_total, 1)
    return float(-np.log(a_prob / b_prob))


def hurst_rs(data):
    """Estimate Hurst exponent via rescaled-range method.

    Parameters
    ----------
    data : array-like
        Input signal.

    Returns
    -------
    float
        Hurst exponent estimate, or ``nan`` when not computable.
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.size
    # Short records produce unstable R/S slopes.
    if n < 16:
        return np.nan

    mean = np.mean(data)
    centered = data - mean
    cumulative = np.cumsum(centered)
    r = np.max(cumulative) - np.min(cumulative)
    s = np.std(data)
    if s == 0 or r == 0:
        return np.nan

    return float(np.log(r / s) / np.log(n))


def dfa_alpha(data):
    """Estimate DFA scaling exponent (alpha).

    Parameters
    ----------
    data : array-like
        Input signal.

    Returns
    -------
    float
        DFA alpha estimate, or ``nan`` when scale fitting is not feasible.
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.size
    if n < 32:
        return np.nan

    integrated = np.cumsum(data - np.mean(data))
    max_scale = n // 4
    if max_scale < 4:
        return np.nan

    # Log-spaced scales avoid over-weighting either short or long windows.
    scales = np.unique(np.logspace(np.log10(4), np.log10(max_scale), num=8).astype(int))
    flucts = []
    valid_scales = []

    for s in scales:
        if s < 4:
            continue
        n_windows = n // s
        if n_windows < 2:
            continue

        rms_vals = []
        for i in range(n_windows):
            segment = integrated[i * s:(i + 1) * s]
            x = np.arange(s)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            rms_vals.append(np.sqrt(np.mean((segment - trend) ** 2)))

        f_s = np.sqrt(np.mean(np.square(rms_vals)))
        if f_s > 0:
            flucts.append(f_s)
            valid_scales.append(s)

    if len(valid_scales) < 2:
        return np.nan

    slope, _ = np.polyfit(np.log(valid_scales), np.log(flucts), 1)
    return float(slope)