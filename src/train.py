"""
Model training module for ride-sharing demand/supply prediction.

This module handles the training stage of the ML pipeline: fitting a
Random Forest classifier on preprocessed training data.

Key principle: This module ONLY trains. It does not perform preprocessing
(that's feature_engineering.py), does not evaluate (that's evaluate.py),
and does not save (that's persistence.py). Each responsibility is separate
and encapsulated.

Training and prediction are completely isolated from each other. Prediction
code will never import from this module, ensuring that prediction logic
cannot accidentally refit the model on new data (data leakage prevention).
"""
import logging
from typing import Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Create logger for this module
logger = logging.getLogger(__name__)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on preprocessed training data.
    
    Args:
        X_train: Training features (preprocessed, shape: (n_samples, n_features)).
        y_train: Training target variable (shape: (n_samples,)).
        random_state: Seed for reproducibility (default 42).
        n_estimators: Number of trees in the forest (default 100).
        max_depth: Maximum depth of trees (default 10).
        min_samples_split: Min samples to split a node (default 5).
        min_samples_leaf: Min samples required at a leaf (default 2).
        
    Returns:
        Fitted RandomForestClassifier instance.
        
    Raises:
        ValueError: If inputs are invalid or empty.
        Exception: If training fails unexpectedly.
    """
    # Input validation
    if X_train is None or len(X_train) == 0:
        raise ValueError("X_train cannot be None or empty")
    if y_train is None or len(y_train) == 0:
        raise ValueError("y_train cannot be None or empty")
    if len(X_train) != len(y_train):
        raise ValueError(f"X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}")
    
    logger.info(
        f"Starting model training: {len(X_train)} samples, "
        f"{X_train.shape[1]} features, n_estimators={n_estimators}"
    )
    
    try:
        # Create and train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        
        model.fit(X_train, y_train)
        
        # Log feature importances
        importances = model.feature_importances_
        top_features = np.argsort(importances)[-5:][::-1]
        logger.info(f"Top 5 important feature indices: {top_features}")
        logger.info(f"Training accuracy: {model.score(X_train, y_train):.3f}")
        
        logger.info("Model training completed successfully")
        return model
    
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

