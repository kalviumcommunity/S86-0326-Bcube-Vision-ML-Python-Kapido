"""
Tests for baseline model utilities.
"""
import numpy as np

from src.baselines import evaluate_classification_baseline


def test_majority_class_baseline_returns_metrics():
    X_train = np.array([[1], [2], [3], [4]])
    y_train = np.array([0, 0, 1, 0])
    X_test = np.array([[5], [6]])
    y_test = np.array([0, 1])

    metrics = evaluate_classification_baseline(X_train, y_train, X_test, y_test)

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert metrics["accuracy"] == 0.5
    assert metrics["recall"] == 0.0
    assert metrics["precision"] == 0.0
