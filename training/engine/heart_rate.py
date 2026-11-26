"""Heart-rate postprocessing for waveform and differential predictions."""
import numpy as np
from scipy import signal

def spectral_bpm(trace, fs, low, high, integrate=True):
    # DiffNormalized 标签对应差分波形；先积分还原，再计算窗口内的主要频率。
    trace = trace.astype(np.float64)
    reconstructed = signal.detrend(np.cumsum(trace) if integrate else trace)
    sos = signal.butter(3, [low, high], btype="bandpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, reconstructed)
    frequencies, power = signal.welch(filtered, fs=fs, nperseg=min(len(filtered), round(10 * fs)), nfft=4096)
    mask = (frequencies >= low) & (frequencies <= high)
    if not np.isfinite(power).all() or power[mask].sum() <= 1e-20:
        return None
    return float(frequencies[mask][np.argmax(power[mask])] * 60)
