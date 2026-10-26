"""Chronological return-direction baseline to prevent future-to-past leakage."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss


def chronological_indices(n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_rows < 15:
        raise ValueError("at least 15 chronologically ordered observations are required")
    train_end = int(n_rows * 0.6)
    validation_end = int(n_rows * 0.8)
    return (
        np.arange(0, train_end),
        np.arange(train_end, validation_end),
        np.arange(validation_end, n_rows),
    )


def evaluate_return_direction(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X and y must be aligned 2D features and 1D labels")
    train_idx, validation_idx, test_idx = chronological_indices(len(y))
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X[train_idx], y[train_idx])
    train_loss = log_loss(y[train_idx], model.predict_proba(X[train_idx]), labels=model.classes_)
    validation_loss = log_loss(
        y[validation_idx], model.predict_proba(X[validation_idx]), labels=model.classes_
    )
    validation_f1 = f1_score(
        y[validation_idx], model.predict(X[validation_idx]), average="weighted", zero_division=0
    )
    final_idx = np.concatenate([train_idx, validation_idx])
    model.fit(X[final_idx], y[final_idx])
    predictions = model.predict(X[test_idx])
    return {
        "train_loss": float(train_loss),
        "validation_loss": float(validation_loss),
        "validation_f1_weighted": float(validation_f1),
        "test_accuracy": float(accuracy_score(y[test_idx], predictions)),
        "test_f1_weighted": float(
            f1_score(y[test_idx], predictions, average="weighted", zero_division=0)
        ),
    }
