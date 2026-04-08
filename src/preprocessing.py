"""
Feature engineering module for ride-sharing dataset.

This module handles the second stage of the ML pipeline: transforming
cleaned data into model-ready features through encoding and scaling.

Functions here:
- Define preprocessing strategies (encoding categorical, scaling numerical)
- Build and return a transformer pipeline
- Verify that numerical features are correctly scaled or normalized

Key principle: Preprocessing pipelines are built but NOT fit here. Fitting
happens during training. This ensures the pipeline can be fit on training
data and applied to test/prediction data without data leakage.

Scaling Methods:
- StandardScaler: Centers to mean=0, std=1 (default, general-purpose)
- MinMaxScaler: Bounds to [0,1] range (useful for neural nets, distance models)

Note: Feature engineering is completely separate from model training
(train.py) and prediction (predict.py).
"""
import logging
from typing import List, Dict, Tuple
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Create logger for this module
logger = logging.getLogger(__name__)


def build_preprocessing_pipeline(
    categorical_cols: List[str],
    numerical_cols: List[str]
) -> ColumnTransformer:
    """
    Construct a preprocessing pipeline for categorical and numerical features.
    
    The pipeline includes:
    - Categorical: imputation (mode) + one-hot encoding
    - Numerical: imputation (median) + standardization
    
    This pipeline is unfitted and ready to be fit on training data.
    
    Args:
        categorical_cols: List of categorical column names.
        numerical_cols: List of numerical column names.
        
    Returns:
        ColumnTransformer object (unfitted, ready for fit_transform).
        
    Raises:
        ValueError: If column lists are empty.
    """
    if not categorical_cols and not numerical_cols:
        raise ValueError("Must provide at least one categorical or numerical column")
    
    logger.info(
        f"Building preprocessing pipeline with "
        f"{len(categorical_cols)} categorical and {len(numerical_cols)} numerical columns"
    )
    
    transformers = []
    
    # Categorical transformation: impute mode + one-hot encode
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_cols))
        logger.debug(f"Added categorical transformer for columns: {categorical_cols}")
    
    # Numerical transformation: impute median + standardize
    if numerical_cols:
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numerical_transformer, numerical_cols))
        logger.debug(f"Added numerical transformer for columns: {numerical_cols}")
    
    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(transformers=transformers)
    
    logger.info("Preprocessing pipeline created successfully")
    return preprocessor


def verify_scaling(
    X_scaled: np.ndarray,
    numerical_feature_indices: List[int],
    mean_tolerance: float = 0.1,
    std_tolerance: float = 0.2
) -> Dict[str, Tuple[np.ndarray, np.ndarray, bool]]:
    """
    Verify that numerical features are correctly scaled (mean ≈ 0, std ≈ 1).
    
    After StandardScaler fits on training data, numerical features should have:
    - Mean ≈ 0 (within tolerance)
    - Standard deviation ≈ 1 (within tolerance)
    
    This function extracts the numerical feature columns from a scaled dataset,
    computes their statistics, and verifies they are within expected ranges.
    
    Args:
        X_scaled: Scaled feature matrix (output from pipeline.fit_transform).
        numerical_feature_indices: Column indices of numerical features in X_scaled.
        mean_tolerance: Acceptable deviation from mean=0 (default 0.1).
        std_tolerance: Acceptable deviation from std=1 (default 0.2).
        
    Returns:
        Dictionary with structure:
        {
            'feature_idx_0': (mean, std, is_valid),
            'feature_idx_1': (mean, std, is_valid),
            ...
        }
        where is_valid = True if mean in [-tol, +tol] and std in [1-tol, 1+tol].
        
    Raises:
        ValueError: If numerical_feature_indices is empty.
        TypeError: If X_scaled is not a numpy array.
    """
    if not isinstance(X_scaled, np.ndarray):
        raise TypeError(f"X_scaled must be numpy array, got {type(X_scaled)}")
    
    if not numerical_feature_indices:
        raise ValueError("numerical_feature_indices cannot be empty")
    
    verification_results = {}
    
    for idx in numerical_feature_indices:
        if idx >= X_scaled.shape[1]:
            logger.warning(f"Feature index {idx} out of range for scaled data shape {X_scaled.shape}")
            continue
        
        feature_data = X_scaled[:, idx]
        feature_mean = np.mean(feature_data)
        feature_std = np.std(feature_data)
        
        # Check if within tolerance
        mean_valid = abs(feature_mean) <= mean_tolerance
        std_valid = abs(feature_std - 1.0) <= std_tolerance
        is_valid = mean_valid and std_valid
        
        verification_results[f'feature_{idx}'] = (feature_mean, feature_std, is_valid)
    
    return verification_results


def log_scaling_verification(
    verification_results: Dict[str, Tuple[float, float, bool]],
    numerical_cols: List[str]
) -> bool:
    """
    Log scaling verification results for human readability.
    
    Displays scaling statistics as a formatted table showing mean, std, and
    validity status for each numerical feature. Returns overall pass/fail status.
    
    Args:
        verification_results: Dictionary from verify_scaling().
        numerical_cols: List of numerical column names (for display).
        
    Returns:
        True if all numerical features are validly scaled, False otherwise.
    """
    if not verification_results:
        logger.warning("No features to verify")
        return False
    
    logger.info("-" * 80)
    logger.info("SCALING VERIFICATION (StandardScaler)")
    logger.info("-" * 80)
    
    all_valid = True
    for idx, (feature_name, (mean, std, is_valid)) in enumerate(verification_results.items()):
        status = "[PASS]" if is_valid else "[FAIL]"
        feature_label = numerical_cols[idx] if idx < len(numerical_cols) else f"num_feature_{idx}"
        
        logger.info(
            f"  {feature_label:20s}  Mean: {mean:8.6f}  Std: {std:8.6f}  {status}"
        )
        
        if not is_valid:
            all_valid = False
    
    logger.info("-" * 80)
    overall_status = "[PASS] ALL FEATURES PROPERLY SCALED" if all_valid else "[FAIL] SOME FEATURES OUT OF TOLERANCE"
    logger.info(f"OVERALL: {overall_status}")
    logger.info("-" * 80)
    
    return all_valid


def build_preprocessing_pipeline_with_normalization(
    categorical_cols: List[str],
    numerical_cols: List[str]
) -> ColumnTransformer:
    """
    Construct a preprocessing pipeline using MinMaxScaler for normalization.
    
    NORMALIZATION (MinMaxScaler) differs from STANDARDIZATION (StandardScaler):
    - StandardScaler: Centers to mean=0, scales to std=1 (unbounded output)
    - MinMaxScaler: Scales to [0, 1] range (bounded output)
    
    MinMaxScaler is beneficial for:
    - Neural networks (bounded activations prevent saturation)
    - Distance-based algorithms (kNN, k-Means, SVM with RBF)
    - Algorithms sensitive to feature magnitude (PCA, anomaly detection)
    
    MinMaxScaler is NOT recommended for:
    - Tree-based models (scale-invariant)
    - Datasets with extreme outliers (single outlier compresses majority of data)
    
    The pipeline includes:
    - Categorical: imputation (mode) + one-hot encoding
    - Numerical: imputation (median) + normalization (MinMaxScaler)
    
    This pipeline is unfitted and ready to be fit on training data.
    
    Args:
        categorical_cols: List of categorical column names.
        numerical_cols: List of numerical column names.
        
    Returns:
        ColumnTransformer object (unfitted, ready for fit_transform).
        
    Raises:
        ValueError: If column lists are empty.
    """
    if not categorical_cols and not numerical_cols:
        raise ValueError("Must provide at least one categorical or numerical column")
    
    logger.info(
        f"Building normalization pipeline (MinMaxScaler) with "
        f"{len(categorical_cols)} categorical and {len(numerical_cols)} numerical columns"
    )
    
    transformers = []
    
    # Categorical transformation: impute mode + one-hot encode
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_cols))
        logger.debug(f"Added categorical transformer for columns: {categorical_cols}")
    
    # Numerical transformation: impute median + normalize to [0,1]
    if numerical_cols:
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('normalizer', MinMaxScaler(feature_range=(0, 1)))
        ])
        transformers.append(('num', numerical_transformer, numerical_cols))
        logger.debug(f"Added normalization transformer for columns: {numerical_cols}")
    
    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(transformers=transformers)
    
    logger.info("Normalization pipeline created successfully")
    return preprocessor


def verify_normalization(
    X_normalized: np.ndarray,
    numerical_feature_indices: List[int],
    min_tolerance: float = 0.01,
    max_tolerance: float = 0.01
) -> Dict[str, Tuple[float, float, bool]]:
    """
    Verify that numerical features are correctly normalized (bounded to [0, 1]).
    
    After MinMaxScaler fits on training data, numerical features should have:
    - Minimum value ≈ 0 (or slightly above, within tolerance)
    - Maximum value ≈ 1 (or slightly below, within tolerance)
    - All values fall within [0, 1] range
    
    Note: Test values may slightly exceed [0, 1] if they exceed the training range.
    This is expected and correct — it indicates the test sample is outside the
    observed training range, not a failure of normalization.
    
    Args:
        X_normalized: Normalized feature matrix (output from pipeline.fit_transform).
        numerical_feature_indices: Column indices of numerical features in X_normalized.
        min_tolerance: Acceptable deviation of minimum from 0 (default 0.01).
        max_tolerance: Acceptable deviation of maximum from 1.0 (default 0.01).
        
    Returns:
        Dictionary with structure:
        {
            'feature_idx_0': (min_val, max_val, is_valid),
            'feature_idx_1': (min_val, max_val, is_valid),
            ...
        }
        where is_valid = True if min is close to 0 and max is close to 1.
        
    Raises:
        ValueError: If numerical_feature_indices is empty.
        TypeError: If X_normalized is not a numpy array.
    """
    if not isinstance(X_normalized, np.ndarray):
        raise TypeError(f"X_normalized must be numpy array, got {type(X_normalized)}")
    
    if not numerical_feature_indices:
        raise ValueError("numerical_feature_indices cannot be empty")
    
    verification_results = {}
    
    for idx in numerical_feature_indices:
        if idx >= X_normalized.shape[1]:
            logger.warning(f"Feature index {idx} out of range for normalized data shape {X_normalized.shape}")
            continue
        
        feature_data = X_normalized[:, idx]
        feature_min = np.min(feature_data)
        feature_max = np.max(feature_data)
        
        # Check if within tolerance (should be close to 0 and 1)
        min_valid = feature_min >= -min_tolerance and feature_min <= min_tolerance
        max_valid = feature_max >= (1.0 - max_tolerance) and feature_max <= (1.0 + max_tolerance)
        is_valid = min_valid and max_valid
        
        verification_results[f'feature_{idx}'] = (feature_min, feature_max, is_valid)
    
    return verification_results


def log_normalization_verification(
    verification_results: Dict[str, Tuple[float, float, bool]],
    numerical_cols: List[str]
) -> bool:
    """
    Log normalization verification results for human readability.
    
    Displays normalization statistics as a formatted table showing min, max, and
    validity status for each numerical feature. Returns overall pass/fail status.
    
    Args:
        verification_results: Dictionary from verify_normalization().
        numerical_cols: List of numerical column names (for display).
        
    Returns:
        True if all numerical features are properly normalized within bounds, False otherwise.
    """
    if not verification_results:
        logger.warning("No features to verify")
        return False
    
    logger.info("-" * 80)
    logger.info("NORMALIZATION VERIFICATION (MinMaxScaler)")
    logger.info("-" * 80)
    
    all_valid = True
    for idx, (feature_name, (min_val, max_val, is_valid)) in enumerate(verification_results.items()):
        status = "[PASS]" if is_valid else "[FAIL]"
        feature_label = numerical_cols[idx] if idx < len(numerical_cols) else f"num_feature_{idx}"
        
        logger.info(
            f"  {feature_label:20s}  Min: {min_val:8.6f}  Max: {max_val:8.6f}  {status}"
        )
        
        if not is_valid:
            all_valid = False
    
    logger.info("-" * 80)
    overall_status = "[PASS] ALL FEATURES PROPERLY NORMALIZED TO [0,1]" if all_valid else "[FAIL] SOME FEATURES OUT OF BOUNDS"
    logger.info(f"OVERALL: {overall_status}")
    logger.info("-" * 80)
    
    return all_valid

