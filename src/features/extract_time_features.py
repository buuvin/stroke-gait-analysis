import numpy as np
from scipy.stats import skew, kurtosis

from features.features_utils import rms, sample_entropy, hurst_rs, dfa_alpha

def extract_time_domain_features(data):
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
            'samp_ent': sample_entropy(data),
            'hurst': hurst_rs(data),
            'dfa': dfa_alpha(data),
            # 'apEn_Re': ApEn(data, 2, 0.2 * data.std())
        }