import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
import seaborn as sns
import matplotlib.pyplot as plt

def rms(x: pd.Series) -> float:
    x = x.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(x**2)))

def estimate_fs_from_time(t: pd.Series) -> float:
    '''
    Estimate sampling frequency (Hz) from time column.
    If estimation fails, returns a safe default.
    '''
    t = pd.to_numeric(t, errors="coerce").dropna().to_numpy(dtype=float)
    if len(t) < 3:
        return 100.0
    dt = np.diff(np.sort(t))
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 100.0
    fs = 1.0 / np.median(dt)
    if not np.isfinite(fs) or fs <= 0:
        return 100.0
    return float(fs)

def dominant_frequency(angle: pd.Series, time: pd.Series) -> float:
    """
    Dominant frequency (Hz) via Fourier transform on de-meaned signal.
    Uses sampling frequency estimated from time.
    """
    y = pd.to_numeric(angle, errors="coerce").dropna().to_numpy(dtype=float)
    if len(y) < 8:
        return np.nan
    fs = estimate_fs_from_time(time)
    y = y - np.mean(y)

    yf = np.abs(rfft(y))
    xf = rfftfreq(len(y), d=1/fs)

    # Ignore 0 Hz component for dominant frequency
    if len(yf) > 1:
        idx = np.argmax(yf[1:]) + 1
    else:
        idx = 0
    return float(xf[idx])


def plot_corr_heatmap(corr: pd.DataFrame, title: str, filename: str, annot=True):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=annot,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1, vmax=1,
        square=True,
        cbar_kws={"label": "Pearson r"}
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved: {filename}")