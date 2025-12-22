
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def cv_confusion_matrix(models, X, y, groups, gkf, labels=None, normalize=None, dir=None):
    
    if labels is None:
        labels = sorted(y.unique())

    cms = {}

    for name, model in models.items():
        y_true_all = []
        y_pred_all = []

        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            m = clone(model)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)

            y_true_all.append(y_test.to_numpy())
            y_pred_all.append(np.asarray(y_pred))

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)

        cm = confusion_matrix(y_true_all, y_pred_all, labels=labels, normalize=normalize)
        cms[name] = cm

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(values_format=".2f" if normalize else "d")
        plt.title(f"{name} — Confusion Matrix (GroupKFold, aggregated)")
        plt.tight_layout()
        norm_tag = f"_norm-{normalize}" if normalize else ""
        if dir is not None:
            plt.savefig(dir +'/' f"{name}_confusion_matrix{norm_tag}.png", dpi=300)
            print(f"\n Saved: '{name}_confusion_matrix{norm_tag}.png")
        plt.show()

    return cms



def cv_classification_reports(models, X, y, groups, gkf: GroupKFold, zero_division=0):
    rows = []

    for model_name, model in models.items():
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            m = clone(model)     #cloning to avoid data leakage
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)

            # Generate classification report
            report = classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=zero_division
            )

            rows.append({
                "model": model_name,
                "fold": fold,
                "accuracy": report["accuracy"],
                "macro_precision": report["macro avg"]["precision"],
                "macro_recall": report["macro avg"]["recall"],
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_precision": report["weighted avg"]["precision"],
                "weighted_recall": report["weighted avg"]["recall"],
                "weighted_f1": report["weighted avg"]["f1-score"],
            })

    report_long = pd.DataFrame(rows)

    # Compute mean and std of metrics for each model
    metrics_cols = [
        "accuracy",
        "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1"
    ]

    report_mean = ( report_long.groupby("model")[metrics_cols].agg(["mean", "std"]).reset_index())

    return report_long, report_mean