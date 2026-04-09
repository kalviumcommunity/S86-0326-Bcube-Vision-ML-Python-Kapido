"""
Baseline model utilities for the ride-sharing classification pipeline.

Baselines provide a minimum-performance reference point for model
evaluation. For this project, the canonical baseline is a majority-class
DummyClassifier trained only on the training split.

The helper in this module keeps baseline logic isolated from the core model
training code while still making it easy to compare a learned model against a
trivial predictor.
"""
import logging
from typing import Any, Dict, Optional

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


logger = logging.getLogger(__name__)


def evaluate_classification_baseline(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    strategy: str = "most_frequent",
    random_state: Optional[int] = 42,
) -> Dict[str, float]:
    """
    Fit and evaluate a simple classification baseline.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        strategy: DummyClassifier strategy to use.
        random_state: Seed used by stochastic strategies.

    Returns:
        Dictionary with accuracy, precision, recall, f1, and roc_auc.
    """
    if X_train is None or y_train is None:
        raise ValueError("X_train and y_train cannot be None")
    if X_test is None or y_test is None:
        raise ValueError("X_test and y_test cannot be None")
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("X_train and y_train cannot be empty")
    if len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("X_test and y_test cannot be empty")
    if len(X_train) != len(y_train):
        raise ValueError(f"X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}")
    if len(X_test) != len(y_test):
        raise ValueError(f"X_test and y_test length mismatch: {len(X_test)} vs {len(y_test)}")

    logger.info("Training baseline model using DummyClassifier(strategy=%s)", strategy)
    baseline = DummyClassifier(strategy=strategy, random_state=random_state)
    baseline.fit(X_train, y_train)

    predictions = baseline.predict(X_test)
    probabilities = None
    if hasattr(baseline, "predict_proba"):
        probabilities = baseline.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }

    if probabilities is not None and probabilities.shape[1] > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, probabilities[:, 1])
        except ValueError:
            metrics["roc_auc"] = float("nan")
            logger.warning("ROC-AUC could not be computed for the baseline; using NaN")
    else:
        metrics["roc_auc"] = float("nan")

    logger.info(
        "Baseline metrics - Accuracy: %.3f, F1: %.3f, Precision: %.3f, Recall: %.3f",
        metrics["accuracy"],
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
    )

    return metrics