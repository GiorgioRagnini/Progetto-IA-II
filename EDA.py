import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

# Statistics for every combination of condition and joint

print("\n Average statistics for condition and joint:")
stats = data.groupby(['condition', 'joint'])['angle'].agg(['mean', 'std', 'min', 'max'])
print(stats)


