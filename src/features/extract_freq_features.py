import numpy as np
from scipy.signal import welch

from features.features_utils import band_power

def extract_frequency_domain_features(data):
    
    fs = 100
    freq, psd = welch(
        data,
        fs = fs,
        nperseg = min(1024, len(data)),
        detrend = 'constant'
    )

    valid = freq <= 5
    freq = freq[valid]
    psd = psd[valid]

    total_power = np.trapezoid(psd, freq)
    i0 = 1  # drop freq==0 bin
    dom_freq = freq[i0:][np.argmax(psd[i0:])]
    mean_freq = np.sum(freq * psd) / np.sum(psd)
    cum_power = np.cumsum(psd)
    median_freq = freq[np.where(cum_power >= cum_power[-1] / 2)[0][0]]

    band_0p1_0p5 = band_power(freq, psd, 0.1, 0.5)
    band_0p5_1p5 = band_power(freq, psd, 0.5, 1.5)
    band_1p5_3 = band_power(freq, psd, 1.5, 3.0)

    df = freq[1] - freq[0]
    total_power = np.trapezoid(psd, freq)
    
    print("band_power_0.1–0.5 =", band_0p1_0p5)
    print("band_power_0.5–1.5 =", band_0p5_1p5)
    print("band_power_1.5-3.0 =", band_1p5_3)

    if band_0p1_0p5 + band_0p5_1p5 + band_1p5_3 > total_power * 1.05:
        print("WARN: bands exceed total power (check bands / freq range)")

    low_high_ratio = np.log1p(band_0p1_0p5 / (band_1p5_3 + 1e-12))

    psd_norm = psd / np.sum(psd)
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))

    freq_variance = np.sum(((freq - mean_freq)**2) * psd) / np.sum(psd)

    return {
        'total_power': total_power,
        'dominant_frequency': dom_freq,
        'mean_frequency': mean_freq,
        'median_frequency': median_freq,
        'band_power_0p1_0p5': band_0p1_0p5,
        'band_power_0p5_1p5': band_0p5_1p5,
        'band_power_1p5_3': band_1p5_3,
        'low_to_high_power_ratio': low_high_ratio,
        'spectral_entropy': spectral_entropy,
        'frequency_variance': freq_variance
    }