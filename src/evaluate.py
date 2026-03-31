"""
Evaluation function for ride-sharing demand/supply prediction model.
"""
from typing import Any, Dict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def evaluate_model(model: Any, X_test, y_test) -> Dict[str, float]:
    """
    Compute evaluation metrics on test data.
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test target.
    Returns:
        Dictionary with precision, recall, f1, roc_auc.
    """
    predictions = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = np.zeros_like(predictions, dtype=float)
    return {
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1': f1_score(y_test, predictions),
        'roc_auc': roc_auc_score(y_test, probabilities) if probabilities.any() else float('nan')
    }
