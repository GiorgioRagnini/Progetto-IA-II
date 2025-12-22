from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB


@dataclass
class TrainResult:
    model: Any
    metrics: Dict[str, Any]
    cm: np.ndarray


def _split_data(
    X, y,
    test_size: float,
    random_state: int,
    groups: Optional[np.ndarray] = None) -> Tuple:
    if groups is None:
        return train_test_split(X, y, test_size = test_size, random_state = random_state, stratify=y)
    gss = GroupShuffleSplit(n_splits = 1, test_size = test_size, random_state = random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def train_and_evaluate(model, X, y, *, test_size = 0.2, random_state = 50, groups=None) -> TrainResult:
    X_train, X_test, y_train, y_test = _split_data(X, y, test_size, random_state, groups=groups)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    confm = confusion_matrix(y_test, y_pred)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, digits=3)
    }
    return TrainResult(model = model, metrics = metrics, cm = confm)


# ---- Building models ----

# Random Forest Classifier
def train_random_forest(X, y, *, test_size = 0.2, random_state = 50, groups=None, **kwargs) -> TrainResult:
    model = RandomForestClassifier(
        n_estimators = kwargs.get("n_estimators", 300),
        max_depth=kwargs.get("max_depth", None),
        random_state = random_state
    )
    return train_and_evaluate(model, X, y, test_size = test_size, random_state = random_state, groups = groups)

# Logistic Regression
def train_logistic_regression(X, y, *, test_size = 0.2, random_state = 50, groups = None, **kwargs) -> TrainResult:
    model = LogisticRegression(
        max_iter = kwargs.get("max_iter", 2000),
        C = kwargs.get("C", 1.0),
        solver =kwargs.get("solver", "lbfgs"),
        random_state = random_state
    )
    return train_and_evaluate(model, X, y, test_size = test_size, random_state = random_state, groups = groups)

# Ridge Classifier for multi-class classification
def train_ridge_classifier(X, y, *, test_size = 0.2, random_state = 50, groups=None, **kwargs) -> TrainResult:
    model = RidgeClassifier(alpha=kwargs.get("alpha", 1.0), random_state=random_state)
    return train_and_evaluate(model, X, y, test_size = test_size, random_state = random_state, groups = groups)

# Gaussian Naive Bayes
def train_gaussian_nb(X, y, *, test_size = 0.2, random_state = 50, groups  = None, **kwargs) -> TrainResult:
    model = GaussianNB()
    return train_and_evaluate(model, X, y, test_size = test_size, random_state = random_state, groups = groups)
