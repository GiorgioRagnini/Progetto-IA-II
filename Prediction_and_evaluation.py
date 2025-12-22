import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.base import clone
from src.models import train_random_forest, train_logistic_regression, train_ridge_classifier, train_gaussian_nb
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.evaluation import cv_confusion_matrix, cv_classification_reports



# Loading datasets

data_raw = pd.read_csv('data/raw/gait_raw.csv')
y_r = data_raw['condition']
X_r = data_raw.drop('condition', axis=1)
print(data_raw.head())
print(data_raw.shape)

data_processed  = pd.read_csv('data/processed/gait_features_v1.csv')
X_p = data_processed.drop('condition', axis=1)
y_p = data_processed['condition']
print(data_processed.head())
print(data_processed.shape)

print("="*70)
print("MODEL TRAINING AND EVALUATION FOR RAW AND PROCESSED DATA")
print('might take a while...')
print("="*70)

# Random Forest on raw data
rf_result_raw = train_random_forest(X_r, y_r, test_size = 0.2, random_state = 50)
print("\nRandom Forest on Raw Data")
print("Metrics:", rf_result_raw.metrics)
print("\n" + "="*80)

# Random Forest on processed data
rf_result_processed = train_random_forest(X_p, y_p, test_size = 0.2, random_state = 50)
print("\nRandom Forest on Processed Data")
print("Metrics:", rf_result_processed.metrics)
print("\n" + "="*80)



# Logistic Regression on raw data
lr_result_raw = train_logistic_regression(X_r, y_r, test_size = 0.2, random_state = 50)
print("\nLogistic Regression on Raw Data")
print("Metrics:", lr_result_raw.metrics)
print("\n" + "="*80)

# Logistic Regression on processed data
lr_result_processed = train_logistic_regression(X_p, y_p, test_size = 0.2, random_state = 50)
print("\nLogistic Regression on Processed Data")
print("Metrics:", lr_result_processed.metrics)
print("\n" + "="*80)


# Ridge Classifier on raw data
rc_result_raw = train_ridge_classifier(X_r, y_r, test_size = 0.2, random_state = 50)
print("\nRidge Classifier on Raw Data")
print("Metrics:", rc_result_raw.metrics)
print("\n" + "="*80)

# Ridge Classifier on processed data
rc_result_processed = train_ridge_classifier(X_p, y_p, test_size = 0.2, random_state = 50)
print("\nRidge Classifier on Processed Data")
print("Metrics:", rc_result_processed.metrics)
print("\n" + "="*80)


# Gaussian Naive Bayes on raw data
gnb_result_raw = train_gaussian_nb(X_r, y_r, test_size = 0.2, random_state = 50)
print("\nGaussian Naive Bayes on Raw Data")
print("Metrics:", gnb_result_raw.metrics)
print("\n" + "="*80)

# Gaussian Naive Bayes on processed data
gnb_result_processed = train_gaussian_nb(X_p, y_p, test_size = 0.2, random_state = 50)
print("\nGaussian Naive Bayes on Processed Data")
print("Metrics:", gnb_result_processed.metrics)
print("\n" + "="*80)


print("=== CROSS-VALIDATION AND EVALUATION WITH GROUPKFOLD ON WIDE DATA ===")

# Reshape processed data to wide format for model training
df = data_processed.copy()

df_wide = df.set_index(
    ["subject", "condition", "joint"]
)
df_wide = df_wide.unstack("joint")
df_wide.columns = [
    f"{feat}_joint{joint}"
    for feat, joint in df_wide.columns
]

data_wide = df_wide.reset_index()
print(data_wide.head())
print(data_wide.shape)
X_w = data_wide.drop(['subject', 'condition'], axis=1)
y_w = data_wide['condition']
groups = data_wide['subject']
    
# Check for missing values
print("\nSum of missing values in wide data: " + str(df_wide.isna().sum().sum()))

# GroupKFold cross-validation setup

gkf = GroupKFold(n_splits=5)

# Check 1 to ensure no subject appears in X columns
print('Check 1: Features columns in X_w:')
print(X_w.columns)

# Check 2 to ensure no data leakage between folds
print('\nCheck 2: Overlap of groups between train and test indices in each fold (should be empty sets):')
for train_idx, test_idx in gkf.split(X_w, y_w, groups):
    print(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))


# Cross-validation scores and repeated shuffle tests
models = {
    "Logistic": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=50
    ),
    "GaussianNB": GaussianNB(),
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RidgeClassifier())
    ])
}

print("\n" + "="*80)
print("=== CROSS-VALIDATION SCORES ===")
print('\n')
for name, model in models.items():
    scores = cross_val_score(
        model,
        X_w,
        y_w,
        cv=gkf,
        groups=groups,
        scoring="accuracy"
    )
    print(f"{name}: mean={scores.mean():.3f}, std={scores.std():.3f}")

print("\n" + "="*80)
print("=== REPEATED SHUFFLE TEST ===")
print('this might take a while as well...')
print("\n")

for name, model in models.items():
    means = []
    for seed in range(50):
        y_shuf = shuffle(y_w, random_state=seed)
        scores = cross_val_score(
            model,
            X_w,
            y_shuf,
            cv=gkf,
            groups=groups,
            scoring="accuracy"
        )
        means.append(scores.mean())

    print(f"{name}: shuffle mean={np.mean(means):.3f}, std={np.std(means):.3f}")

print("\n" + "="*80)


# Fold-wise classification reports
print("=== FOLD-WISE CLASSIFICATION REPORTS ===")
print('\n')

report_long, report_mean = cv_classification_reports(models, X_w, y_w, groups, gkf)

print("=== DETAILED FOLD-WISE REPORTS ===")
print('report_long:')
print(report_long)
print('report_mean:')
print(report_mean)
print("\n" + "="*80)

# Fold-wise confusion matrices
print("=== FOLD-WISE CONFUSION MATRICES ===")


print('\nGenerating and saving confusion matrix plots...')
cms = cv_confusion_matrix(models, X_w, y_w, groups, gkf, normalize=None, dir = 'png_confusion_matrices')

cms_norm = cv_confusion_matrix(models, X_w, y_w, groups, gkf, normalize="true", dir = 'png_confusion_matrices')

