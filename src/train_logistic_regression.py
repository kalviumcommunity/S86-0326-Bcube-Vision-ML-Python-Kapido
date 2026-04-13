"""
Logistic Regression Training Module for Binary Classification

This module implements proper Logistic Regression training following best practices:

PROBLEM: Binary classification — predict whether a sample belongs to class 1 or 0
         (e.g., Spam vs Not Spam, Fraud vs Legitimate, Churn vs No Churn)

KEY PRINCIPLES IMPLEMENTED:
1. Train/test split BEFORE preprocessing (prevents data leakage)
2. Stratified split to preserve class proportions in both train and test
3. Feature scaling via Pipeline (fitted only on training data)
4. Proper regularization with L2 and optional L1
5. Baseline comparison using DummyClassifier (majority class strategy)
6. Cross-validation for stability assessment
7. Both class predictions and probabilities for proper evaluation
8. Coefficient interpretation as odds ratios
9. Separation of concerns: training handles fitting only, not evaluation

WORKFLOW:
    Load data with binary target (0/1)
         │
    X_train, X_test, y_train, y_test ← [SPLIT with stratify=y]
                      │
    Pipeline builds: StandardScaler → LogisticRegression
                      │
    Fit scaler on X_train only
    Fit LogisticRegression on scaled X_train
                      │
    Transform X_test with fitted scaler
    Generate predictions and probabilities
                      │
                  [EVALUATE on test set]

This workflow prevents:
- Data leakage (scaling statistics not contaminated by test data)
- Train/test contamination (scaler fitted only on training data)
- Stratification error (class imbalance misrepresented in splits)
- Overly optimistic evaluation (testing on fresh, held-out data)
"""

import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# Create logger for this module
logger = logging.getLogger(__name__)


def train_logistic_regression_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
    C: float = 1.0,
    penalty: str = "l2",
    class_weight: Optional[str] = None,
) -> Tuple[Pipeline, Pipeline, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Train a Logistic Regression model with complete workflow including baseline.

    This function implements the full Logistic Regression training pipeline:
    1. Split data with stratification (preserve class proportions)
    2. Create preprocessing pipeline (scaling only)
    3. Train both baseline (DummyClassifier) and Logistic Regression
    4. Return pipelines and data for later evaluation

    CRITICAL: Train/test split happens FIRST with stratification.
    This prevents data leakage and ensures representative splits.

    Args:
        X: Feature matrix (shape: (n_samples, n_features))
        y: Binary target variable (shape: (n_samples,)) with values 0/1
        test_size: Proportion of data for test set (default 0.2)
        random_state: Seed for reproducibility (default 42)
        max_iter: Maximum iterations for solver (default 1000)
        C: Inverse of regularization strength (default 1.0)
           Lower C = stronger regularization
        penalty: Type of regularization - "l2" (default) or "l1"
        class_weight: If "balanced", adjust weights inversely to class frequency
                     Useful for imbalanced datasets

    Returns:
        Tuple of:
        - model_pipeline: Fitted Pipeline (StandardScaler → LogisticRegression)
        - baseline_pipeline: Fitted Pipeline (StandardScaler → DummyClassifier)
        - X_test: Test features (unscaled)
        - y_test: Test target values
        - training_data: Dict containing metrics and training information

    Raises:
        ValueError: If inputs are invalid or target is not binary
        Exception: If training fails
    """
    # Input validation
    if X is None or len(X) == 0:
        raise ValueError("X cannot be None or empty")
    if y is None or len(y) == 0:
        raise ValueError("y cannot be None or empty")
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    
    # Check binary classification
    unique_classes = np.unique(y)
    if len(unique_classes) != 2:
        raise ValueError(f"Target must be binary. Found {len(unique_classes)} classes: {unique_classes}")
    
    logger.info("="*70)
    logger.info("LOGISTIC REGRESSION TRAINING")
    logger.info("="*70)
    logger.info(f"\nDataset Info:")
    logger.info(f"  Total samples: {len(y)}")
    logger.info(f"  Total features: {X.shape[1]}")
    
    # Check class distribution
    class_counts = pd.Series(y).value_counts()
    logger.info(f"  Class distribution:")
    for class_label in sorted(unique_classes):
        count = (y == class_label).sum()
        pct = count / len(y) * 100
        logger.info(f"    Class {class_label}: {count} ({pct:.1f}%)")
    
    # 1. SPLIT FIRST with stratification
    logger.info(f"\nSplitting data with stratification (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # CRITICAL: Preserve class distribution in both splits
    )
    logger.info(f"  Train set: {len(X_train)} samples")
    logger.info(f"  Test set:  {len(X_test)} samples")
    
    # Verify stratification worked
    train_dist = pd.Series(y_train).value_counts()
    test_dist = pd.Series(y_test).value_counts()
    logger.info(f"  Train class distribution: {dict(train_dist)}")
    logger.info(f"  Test class distribution:  {dict(test_dist)}")
    
    # 2. BUILD PIPELINES
    logger.info(f"\nBuilding preprocessing + model pipelines...")
    
    # Baseline: DummyClassifier (always predicts majority class)
    baseline_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", DummyClassifier(strategy="most_frequent"))
    ])
    
    # Model: LogisticRegression with regularization
    model_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            C=C,
            penalty=penalty,
            class_weight=class_weight,
            solver="lbfgs" if penalty == "l2" else "liblinear"
        ))
    ])
    
    # 3. TRAIN MODELS
    logger.info(f"\nTraining models...")
    
    try:
        baseline_pipeline.fit(X_train, y_train)
        logger.info("  ✓ Baseline model trained")
    except Exception as e:
        logger.error(f"Baseline training failed: {str(e)}")
        raise
    
    try:
        model_pipeline.fit(X_train, y_train)
        logger.info("  ✓ Logistic Regression model trained")
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise
    
    # 4. GET PREDICTIONS FOR EVALUATION
    logger.info(f"\nGenerating predictions on test set...")
    baseline_pred = baseline_pipeline.predict(X_test)
    baseline_prob = baseline_pipeline.predict_proba(X_test)[:, 1]
    
    model_pred = model_pipeline.predict(X_test)
    model_prob = model_pipeline.predict_proba(X_test)[:, 1]
    
    # 5. EXTRACT COEFFICIENTS
    model_coef = model_pipeline.named_steps["model"].coef_[0]
    model_intercept = model_pipeline.named_steps["model"].intercept_[0]
    
    # 6. PREPARE TRAINING DATA FOR RETURN
    training_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "baseline_pred": baseline_pred,
        "baseline_prob": baseline_prob,
        "model_pred": model_pred,
        "model_prob": model_prob,
        "coefficients": model_coef,
        "intercept": model_intercept,
        "feature_names": X.columns if isinstance(X, pd.DataFrame) else None,
        "class_distribution_train": dict(pd.Series(y_train).value_counts()),
        "class_distribution_test": dict(pd.Series(y_test).value_counts()),
    }
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70 + "\n")
    
    return model_pipeline, baseline_pipeline, X_test, y_test, training_data


def extract_coefficient_interpretation(
    model_pipeline: Pipeline,
    feature_names: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Extract and interpret Logistic Regression coefficients as odds ratios.

    Args:
        model_pipeline: Fitted Pipeline containing LogisticRegression
        feature_names: Optional array of feature names (same order as features used in training)

    Returns:
        DataFrame with columns:
        - Feature: Feature name or index
        - Coefficient: Raw coefficient (in log-odds space)
        - Odds Ratio: exp(coefficient) - multiplicative change in odds per unit increase
        - Interpretation: Human-readable meaning
    """
    model = model_pipeline.named_steps["model"]
    coef = model.coef_[0]
    
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(coef))]
    
    # Compute odds ratios
    odds_ratios = np.exp(coef)
    
    # Create interpretation
    interpretations = []
    for or_val in odds_ratios:
        if or_val > 1.0:
            pct_change = (or_val - 1.0) * 100
            interp = f"+{pct_change:.1f}% odds of class 1"
        elif or_val < 1.0:
            pct_change = (1.0 - or_val) * 100
            interp = f"−{pct_change:.1f}% odds of class 1"
        else:
            interp = "No effect"
        interpretations.append(interp)
    
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coef,
        "Odds Ratio": odds_ratios,
        "Interpretation": interpretations
    }).sort_values("Coefficient", key=abs, ascending=False)
    
    logger.info(f"Intercept (log-odds baseline): {model.intercept_[0]:.4f}")
    logger.info(f"\nCoefficient Interpretation (sorted by magnitude):")
    logger.info(coef_df.to_string(index=False))
    
    return coef_df
