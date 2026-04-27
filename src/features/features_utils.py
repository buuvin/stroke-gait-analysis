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
    """
    Extract labels and metadata from filename.
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
    mask = (freq >= fmin) & (freq <= fmax)
    if np.count_nonzero(mask) == 0:
        return 0.0
    if np.count_nonzero(mask) == 1:
        # single-bin band: approximate area = PSD * df
        df = freq[1] - freq[0] if len(freq) > 1 else 0.0
        return float(psd[mask][0] * df)
    return float(np.trapezoid(psd[mask], freq[mask]))

def rms(data):
    return np.sqrt(np.mean(data**2))

def ApEn(U, m, r):
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
    data = np.asarray(data, dtype=np.float64)
    n = data.size
    if n <= m + 1:
        return np.nan

    r = r_ratio * np.std(data)
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
    data = np.asarray(data, dtype=np.float64)
    n = data.size
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
    data = np.asarray(data, dtype=np.float64)
    n = data.size
    if n < 32:
        return np.nan

    integrated = np.cumsum(data - np.mean(data))
    max_scale = n // 4
    if max_scale < 4:
        return np.nan

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