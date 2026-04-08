"""
Model training module for ride-sharing demand/supply prediction.

This module handles the training stage of the ML pipeline: loading data,
splitting, preprocessing, and fitting a Random Forest classifier.

This is a BINARY CLASSIFICATION problem:
- Target: ride_completed (0 = not completed, 1 = completed)
- Algorithm: Random Forest Classifier (handles non-linear relationships)
- Output: Probability of completion + class prediction

Key principle: This module ONLY trains. It does not evaluate (that's evaluate.py),
and does not save (that's persistence.py). Each responsibility is separate
and encapsulated.

Training and prediction are completely isolated from each other. Prediction
code will never import from this module, ensuring that prediction logic
cannot accidentally refit the model on new data (data leakage prevention).
"""
import logging
from typing import Tuple, Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .data_loader import load_data
from .preprocessing import build_preprocessing_pipeline, verify_scaling, log_scaling_verification

# Create logger for this module
logger = logging.getLogger(__name__)


def train_model(
    data_path: str,
    target_column: str,
    categorical_cols: list,
    numerical_cols: list,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    verify_scaling_enabled: bool = True,
    scaling_mean_tolerance: float = 0.1,
    scaling_std_tolerance: float = 0.2
) -> Tuple[RandomForestClassifier, object, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Train a Random Forest classifier on ride-sharing data.
    
    This function handles the complete training workflow: loading, splitting,
    preprocessing, and model fitting. Optionally verifies that numerical
    features are correctly scaled after preprocessing.
    
    Args:
        data_path: Path to the CSV data file.
        target_column: Name of the target column.
        categorical_cols: List of categorical column names.
        numerical_cols: List of numerical column names.
        test_size: Proportion of data for test set (default 0.2).
        random_state: Seed for reproducibility (default 42).
        n_estimators: Number of trees in the forest (default 100).
        max_depth: Maximum depth of trees (default 10).
        min_samples_split: Min samples to split a node (default 5).
        min_samples_leaf: Min samples required at a leaf (default 2).
        verify_scaling_enabled: Whether to verify StandardScaler results (default True).
        scaling_mean_tolerance: Acceptable deviation from mean=0 (default 0.1).
        scaling_std_tolerance: Acceptable deviation from std=1 (default 0.2).
        
    Returns:
        Tuple of:
        - fitted_model: Trained RandomForestClassifier
        - fitted_pipeline: Fitted preprocessing pipeline
        - X_test: Unprocessed test features (for later evaluation)
        - y_test: Test target values
        - verification_stats: Dict with 'scaling_valid' and 'scaling_results' keys
        
    Raises:
        ValueError: If inputs are invalid.
        Exception: If training fails.
    """
    # Initialize verification stats
    verification_stats = {
        'scaling_valid': None,
        'scaling_results': {}
    }
    
    # Load data
    df = load_data(data_path)
    
    # Split data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
    
    # Build and fit preprocessing pipeline
    pipeline = build_preprocessing_pipeline(categorical_cols, numerical_cols)
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    logger.info(f"Preprocessing complete: {X_train_processed.shape[1]} features")
    
    # Verify scaling (if enabled)
    if verify_scaling_enabled and numerical_cols:
        try:
            # Get the starting index of numerical features in the processed data
            # (categorical features are one-hot encoded first in ColumnTransformer)
            num_categorical_features = 0
            cat_transformer = pipeline.named_transformers_.get('cat')
            if cat_transformer is not None:
                # One-hot encoder output size = n_categories - 1 (drop='first')
                # This is an approximation; exact count depends on cat data
                num_categorical_features = X_train_processed.shape[1] - len(numerical_cols)
            
            # Numerical features should be at the end of the processed array
            numerical_indices = list(range(num_categorical_features, X_train_processed.shape[1]))
            
            if numerical_indices:
                # Verify scaling on training data
                scaling_results = verify_scaling(
                    X_train_processed,
                    numerical_indices,
                    mean_tolerance=scaling_mean_tolerance,
                    std_tolerance=scaling_std_tolerance
                )
                
                # Log verification results
                scaling_valid = log_scaling_verification(scaling_results, numerical_cols)
                
                verification_stats['scaling_valid'] = scaling_valid
                verification_stats['scaling_results'] = scaling_results
        except Exception as e:
            logger.warning(f"Scaling verification failed: {e}")
            verification_stats['scaling_valid'] = None
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train_processed, y_train)
    
    logger.info(f"Model trained. Training accuracy: {model.score(X_train_processed, y_train):.3f}")
    
    return model, pipeline, X_test, y_test, verification_stats

