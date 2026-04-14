"""
Model evaluation module for ride-sharing demand/supply prediction.

This module handles the evaluation stage of the ML pipeline: computing
performance metrics on held-out test data.

This is a BINARY CLASSIFICATION problem, so we use classification metrics:
- Accuracy: Overall correctness (can be misleading if imbalanced)
- Precision: Of predicted completions, what fraction are actually completed?
- Recall: Of actual completions, what fraction did we correctly identify?
- F1 Score: Harmonic mean of precision and recall
- ROC-AUC: Threshold-independent measure of discrimination ability

Key principle: Evaluation is completely separate from training. A model
can be loaded from disk and evaluated without running training code.
Changes to evaluation metrics do not affect training or prediction logic.

This separation ensures:
- Reproducible evaluation of multiple models
- Easy integration with experiment tracking systems
- Clear audit trail of how metrics are computed
"""
import logging
from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, confusion_matrix, classification_report
)

# Create logger for this module
logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics on test data.
    
    Metrics computed:
    - Accuracy: Overall correctness
    - Precision: True positives / all positive predictions
    - Recall: True positives / all actual positives
    - F1 Score: Harmonic mean of precision and recall
    - ROC-AUC: Area under receiver operating characteristic curve
    
    Args:
        model: Trained model (must have predict and predict_proba methods).
        X_test: Test features (shape: (n_samples, n_features)).
        y_test: Test target variable (shape: (n_samples,)).
        
    Returns:
        Dictionary containing all computed metrics.
        
    Raises:
        ValueError: If inputs are invalid or None.
        AttributeError: If model lacks required methods.
    """
    # Input validation
    if model is None:
        raise ValueError("Model cannot be None")
    if X_test is None or len(X_test) == 0:
        raise ValueError("X_test cannot be None or empty")
    if y_test is None or len(y_test) == 0:
        raise ValueError("y_test cannot be None or empty")
    if len(X_test) != len(y_test):
        raise ValueError(f"X_test and y_test length mismatch: {len(X_test)} vs {len(y_test)}")
    
    logger.info(f"Starting model evaluation on {len(X_test)} test samples")
    
    try:
        # Generate predictions
        predictions = model.predict(X_test)
        
        # Get probability estimates if available (for ROC-AUC)
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1': f1_score(y_test, predictions, zero_division=0),
        }
        
        # Add ROC-AUC if probabilities available
        if probabilities is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, probabilities)
            except ValueError:
                metrics['roc_auc'] = float('nan')
                logger.warning("ROC-AUC could not be computed; using NaN")
        else:
            metrics['roc_auc'] = float('nan')
            logger.warning("Model does not have predict_proba; ROC-AUC set to NaN")
        
        # Log metrics
        logger.info(f"Evaluation metrics - Accuracy: {metrics['accuracy']:.3f}, "
                   f"F1: {metrics['f1']:.3f}, Precision: {metrics['precision']:.3f}, "
                   f"Recall: {metrics['recall']:.3f}")
        
        # Log confusion matrix
        cm = confusion_matrix(y_test, predictions)
        logger.debug(f"Confusion matrix:\n{cm}")
        
        # Log classification report
        report = classification_report(y_test, predictions, output_dict=False)
        logger.debug(f"Classification report:\n{report}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

