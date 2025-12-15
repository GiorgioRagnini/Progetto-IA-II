import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from ucimlrepo import fetch_ucirepo

# Load dataset

multivariate_gait_data = fetch_ucirepo(id=760) 
data = multivariate_gait_data.data.features

# Separate features and target

y = data['condition']
X = data.drop('condition', axis=1)


print("="*70)
print("EDA (EXPLORATORY DATA ANALYSIS)")
print("="*70)

# 1. Visualize some temporal patterns

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for cond in [1, 2, 3]:
    sample = data[(data['condition'] == cond) & 
                  (data['subject'] == 1) & 
                  (data['replication'] == 1) &
                  (data['leg'] == 1) &
                  (data['joint'] == 1)]
    axes[cond-1].plot(sample['time'], sample['angle'])
    axes[cond-1].set_title(f'Condition {cond}')
    axes[cond-1].set_xlabel('Time')
    axes[cond-1].set_ylabel('Angle (degrees)')
plt.savefig('temporal_patterns.png')
print("\n Saved: temporal_patterns.png")

# 2. Angle distribution by condition

plt.figure(figsize=(10, 6))
for cond in [1, 2, 3]:
    subset = data[data['condition'] == cond]['angle']
    plt.hist(subset, alpha=0.5, bins=50, label=f'Condition {cond}')
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.legend()
plt.title('Angle Distribution by Condition')
plt.savefig('angle_distribution.png')
print("\n Saved: angle_distribution.png")

# 3. Correlation heatmap

plt.figure(figsize=(8, 6))
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
print("\n Saved: correlation_matrix.png")

print("\n" + "="*70)
print("SUMMARY:")
print(f"- Dataset shape: {data.shape}")
print(f"- Target: 'condition' (3 balanced classes)")
print(f"- Features: {list(X.columns)}")
print(f'- Sum of missing values:\n{data.isna().sum()}')
print(f"---> No missing values")
print("="*70)

# 4. Detailed temporal patterns for each joint and condition

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for joint in [1, 2, 3]:
    for cond in [1, 2, 3]:
        sample = data[(data['condition'] == cond) & 
                      (data['subject'] == 1) & 
                      (data['replication'] == 1) &
                      (data['leg'] == 1) &
                      (data['joint'] == joint)]
        
        ax = axes[joint-1, cond-1]
        ax.plot(sample['time'], sample['angle'], linewidth=2)
        ax.set_title(f'Joint {joint}, Condition {cond}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Angle')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('detailed_temporal_patterns.png', dpi=300)
print(" Saved: detailed_temporal_patterns.png")

# 5. Analysis of Subjects' Variability
# - Plot temporal patterns for each subject under the same condition and joint, to see if normalization is needed.

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

axes = axes.flatten()

for i, subject in enumerate(range(1, 11)):
    sample = data[(data['subject'] == subject) & 
                  (data['condition'] == 1) & 
                  (data['replication'] == 1) &
                  (data['leg'] == 1) &
                  (data['joint'] == 1)]
    
    axes[i].plot(sample['time'], sample['angle'], color = 'red')
    axes[i].set_title(f'Subject {subject}')
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Angle')

plt.tight_layout()
plt.savefig("Subjects' variability.png", dpi=300)
print("\n Saved: Subjects' variability.png")

# 6. Analysis to see the differences between Conditions

# Test ANOVA for every joint
print("\n" + "="*80)
print("TEST ANOVA - Differences between Conditions")
print("="*80)

for joint in [1, 2, 3]:
    cond1 = data[(data['joint'] == joint) & (data['condition'] == 1)]['angle']
    cond2 = data[(data['joint'] == joint) & (data['condition'] == 2)]['angle']
    cond3 = data[(data['joint'] == joint) & (data['condition'] == 3)]['angle']
    
    f_stat, p_value = stats.f_oneway(cond1, cond2, cond3)
    
    print(f"\nJoint {joint}:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  P-value: {p_value:.4e}")
    print(f"  Significant differences: {'Yes' if p_value < 0.05 else 'No'}")






# 7. Additional EDA: Feature extraction and correlation heatmaps

from scipy.fft import rfft, rfftfreq

# Helper: feature functions

def rms(x: pd.Series) -> float:
    x = x.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(x**2)))

def estimate_fs_from_time(t: pd.Series) -> float:
    """
    Estimate sampling frequency (Hz) from time column.
    Works even if time is not perfectly regular.
    If estimation fails, returns a safe default.
    """
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
    Dominant frequency (Hz) via FFT on de-meaned signal.
    Uses sampling frequency estimated from time.
    """
    y = pd.to_numeric(angle, errors="coerce").dropna().to_numpy(dtype=float)
    if len(y) < 8:
        return np.nan
    fs = estimate_fs_from_time(time)
    y = y - np.mean(y)

    yf = np.abs(rfft(y))
    xf = rfftfreq(len(y), d=1/fs)

    # ignore 0 Hz (DC) if possible
    if len(yf) > 1:
        idx = np.argmax(yf[1:]) + 1
    else:
        idx = 0
    return float(xf[idx])


# 7.1) Build feature tables

grp = data.groupby(["subject", "condition", "joint"], dropna=False)

# Basic stats + ROM + RMS
features_basic = grp["angle"].agg(
    mean="mean",
    std="std",
    min="min",
    max="max",
    RMS=rms,
)
features_basic["ROM"] = features_basic["max"] - features_basic["min"]
features_basic = features_basic.reset_index()

# Dominant frequency needs both angle and time
features_freq = grp.apply(lambda g: dominant_frequency(g["angle"], g["time"]))
features_freq = features_freq.reset_index(name="dominant_freq")

# Merge all features
features_all = pd.merge(
    features_basic,
    features_freq,
    on=["subject", "condition", "joint"],
    how="left"
)


# 7.2) Correlation matrices

corr_stats = features_all[["mean", "std", "min", "max"]].corr(numeric_only=True)
corr_rom   = features_all[["ROM"]].corr(numeric_only=True)  # (1x1, not shown)
corr_rms   = features_all[["RMS"]].corr(numeric_only=True)  # (1x1, not shown)
corr_freq  = features_all[["dominant_freq"]].corr(numeric_only=True)  # (1x1, not shown)
corr_dyn = features_all[["ROM", "RMS", "dominant_freq"]].corr(numeric_only=True)
corr_all   = features_all[["mean", "std", "min", "max", "ROM", "RMS", "dominant_freq"]].corr(numeric_only=True)



# 7.3) Heatmap plotter

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


# 7.4) Plot and save heatmaps

plot_corr_heatmap(corr_stats, "Correlation (Stats: mean/std/min/max)", "corr_stats.png", annot=True)
plot_corr_heatmap(corr_dyn,   "Correlation (Dynamic features: ROM, RMS, Dominant Freq)", "corr_dynamic_features.png", annot=True)
plot_corr_heatmap(corr_all, "Correlation (All extracted features)", "corr_all_features.png", annot=True)

# 7.5) Print tables and correlations

print("\n=== Feature table preview ===")
print(features_all.head(12))

print("\n=== Correlation (all features) ===")
print(corr_all)



# Statistics for every combination of condition and joint

print("\n" + "="*80)
print("\n Average statistics for condition and joint:")
print("="*80)
stats = data.groupby(['condition', 'joint'])['angle'].agg(['mean', 'std', 'min', 'max'])
print(stats)