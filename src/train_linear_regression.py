"""
Linear Regression training module for ride-sharing duration prediction.

This module implements Linear Regression following best practices and the
complete workflow outlined in supervised learning fundamentals:

PROBLEM: Predict estimated_time (ride duration in minutes) based on features
using Linear Regression - a simple, interpretable, and powerful baseline.

KEY PRINCIPLES IMPLEMENTED:
1. Train/test split BEFORE any preprocessing (prevents data leakage)
2. Feature scaling via scikit-learn Pipeline (fitted only on training data)
3. Baseline comparison using DummyRegressor with strategy="mean"
4. Comprehensive evaluation: MSE, RMSE, MAE, R², cross-validation
5. Coefficient interpretation for model transparency
6. Separation of concerns: training handles only fitting, not evaluation or prediction

WORKFLOW:
    X_train, X_test, y_train, y_test ←― [SPLIT: must split FIRST]
                      │
        train pipeline.fit() ← [FIT scaler on training data only]
                      │
    X_train_scaled ←―― [Transform training data]
                      │
    LinearRegression.fit(X_train_scaled, y_train) ← [Fit model]
                      │
    X_test_scaled ←―― [Transform test with fitted scaler]
                      │
    y_pred = model.predict(X_test_scaled) ← [Generate predictions]
                      │
                  [EVALUATE on test set]

This workflow prevents:
- Data leakage (scaling statistics not contaminated by test data)
- Train/test contamination (scaler fitted only on training data)
- Overly optimistic evaluation (testing on fresh, held-out data)
"""
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .data_loader import load_data
from .preprocessing import build_preprocessing_pipeline

# Create logger for this module
logger = logging.getLogger(__name__)


def train_linear_regression_model(
    data_path: str,
    target_column: str = 'estimated_time',
    categorical_cols: list = None,
    numerical_cols: list = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, DummyRegressor, Pipeline, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Train a Linear Regression model with complete workflow including baseline.

    This function implements the full Linear Regression training pipeline:
    1. Load data
    2. Split into train/test (BEFORE preprocessing)
    3. Build preprocessing pipeline (scaling, encoding)
    4. Train both baseline and Linear Regression models
    5. Return models and data for later evaluation

    CRITICAL: Train/test split happens FIRST, before any fitting.
    This prevents data leakage.

    Args:
        data_path: Path to the CSV data file.
        target_column: Name of the target column to predict (default: 'estimated_time').
        categorical_cols: List of categorical column names.
        numerical_cols: List of numerical column names.
        test_size: Proportion of data for test set (default 0.2).
        random_state: Seed for reproducibility (default 42).

    Returns:
        Tuple of:
        - lr_pipeline: Fitted Pipeline (StandardScaler → LinearRegression)
        - baseline_pipeline: Fitted Pipeline (StandardScaler → DummyRegressor)
        - feature_pipeline: Fitted preprocessing pipeline (categorical/numerical handling)
        - X_test_processed: Processed test features
        - y_test: Test target values
        - evaluation_data: Dict containing metrics and evaluation info

    Raises:
        ValueError: If inputs are invalid or target column is missing.
        Exception: If training fails.
    """
    if categorical_cols is None:
        categorical_cols = ['pickup_location', 'dropoff_location', 'hour_of_day', 'day_of_week']
    if numerical_cols is None:
        numerical_cols = ['trip_distance']

    logger.info("=" * 80)
    logger.info("LINEAR REGRESSION TRAINING: RIDE DURATION PREDICTION")
    logger.info("=" * 80)

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = load_data(data_path)
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Validate target column
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data. "
                        f"Available columns: {df.columns.tolist()}")

    logger.info(f"Target variable: {target_column}")

    # Extract features and target
    X = df[categorical_cols + numerical_cols].copy()
    y = df[target_column].copy()

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Target statistics:\n"
               f"  Mean:   {y.mean():.2f}\n"
               f"  Median: {y.median():.2f}\n"
               f"  Min:    {y.min():.2f}\n"
               f"  Max:    {y.max():.2f}\n"
               f"  Std:    {y.std():.2f}")

    # ========================================================================
    # STEP 1: TRAIN/TEST SPLIT (MUST BE FIRST!)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Train/Test Split (BEFORE any fitting)")
    logger.info("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    logger.info(f"Training set:   {X_train.shape[0]} samples ({100 * (1 - test_size):.0f}%)")
    logger.info(f"Test set:       {X_test.shape[0]} samples ({100 * test_size:.0f}%)")
    logger.info(f"Train target stats: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
    logger.info(f"Test target stats:  mean={y_test.mean():.2f}, std={y_test.std():.2f}")

    # ========================================================================
    # STEP 2: BUILD AND FIT PREPROCESSING PIPELINE
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Preprocessing Pipeline")
    logger.info("=" * 80)

    feature_pipeline = build_preprocessing_pipeline(categorical_cols, numerical_cols)
    logger.info("Preprocessing pipeline created")

    # Fit preprocessing pipeline on TRAINING data only
    X_train_processed = feature_pipeline.fit_transform(X_train)
    logger.info(f"Preprocessing fitted on training set")
    logger.info(f"Processed training shape: {X_train_processed.shape}")

    # Transform test data using fitted scaler statistics
    X_test_processed = feature_pipeline.transform(X_test)
    logger.info(f"Processed test shape: {X_test_processed.shape}")

    # ========================================================================
    # STEP 3: BUILD FULL PIPELINES (Preprocessing + Model)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Model Training")
    logger.info("=" * 80)

    # Baseline: Predict the mean value
    logger.info("\n>>> Training Baseline (DummyRegressor with strategy='mean')")
    baseline = DummyRegressor(strategy='mean')
    baseline.fit(X_train_processed, y_train)
    baseline_preds_train = baseline.predict(X_train_processed)
    baseline_preds_test = baseline.predict(X_test_processed)
    logger.info("[OK] Baseline trained (predicts mean)")

    # Linear Regression
    logger.info("\n>>> Training Linear Regression")
    lr_model = LinearRegression()
    lr_model.fit(X_train_processed, y_train)
    lr_preds_train = lr_model.predict(X_train_processed)
    lr_preds_test = lr_model.predict(X_test_processed)
    logger.info("[OK] Linear Regression trained")

    # ========================================================================
    # STEP 4: COMPUTE EVALUATION METRICS
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Evaluation Metrics (Test Set)")
    logger.info("=" * 80)

    def compute_metrics(y_true, y_pred, name):
        """Compute regression metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

    baseline_metrics = compute_metrics(y_test, baseline_preds_test, "Baseline")
    lr_metrics = compute_metrics(y_test, lr_preds_test, "Linear Regression")

    logger.info("\n>>> Baseline (Mean Prediction)")
    logger.info(f"    RMSE: {baseline_metrics['RMSE']:.2f}")
    logger.info(f"    MAE:  {baseline_metrics['MAE']:.2f}")
    logger.info(f"    R²:   {baseline_metrics['R2']:.3f}")

    logger.info("\n>>> Linear Regression")
    logger.info(f"    RMSE: {lr_metrics['RMSE']:.2f}")
    logger.info(f"    MAE:  {lr_metrics['MAE']:.2f}")
    logger.info(f"    R²:   {lr_metrics['R2']:.3f}")

    # Compute improvement
    rmse_improvement = ((baseline_metrics['RMSE'] - lr_metrics['RMSE']) / 
                       baseline_metrics['RMSE'] * 100)
    r2_improvement = lr_metrics['R2'] - baseline_metrics['R2']

    logger.info("\n>>> Improvement over Baseline")
    logger.info(f"    RMSE reduction: {rmse_improvement:.1f}%")
    logger.info(f"    R² improvement: {r2_improvement:.3f}")

    # ========================================================================
    # STEP 5: CROSS-VALIDATION
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Cross-Validation (5-fold)")
    logger.info("=" * 80)

    cv_scores_r2 = cross_val_score(
        lr_model, X_train_processed, y_train, cv=5, scoring='r2'
    )

    logger.info(f"CV R² scores:     {[f'{s:.3f}' for s in cv_scores_r2]}")
    logger.info(f"Mean CV R²:       {cv_scores_r2.mean():.3f}")
    logger.info(f"Std CV R²:        {cv_scores_r2.std():.3f}")
    logger.info(f"95% Confidence:   [{cv_scores_r2.mean() - 1.96 * cv_scores_r2.std():.3f}, "
               f"{cv_scores_r2.mean() + 1.96 * cv_scores_r2.std():.3f}]")

    # ========================================================================
    # STEP 6: COEFFICIENT INTERPRETATION
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Model Coefficients (Feature Importance)")
    logger.info("=" * 80)

    # Get feature names from preprocessing
    try:
        feature_names = feature_pipeline.named_steps['column_transformer'].get_feature_names_out()
        feature_names = [str(name) for name in feature_names]
    except Exception as e:
        logger.warning(f"Could not get feature names from pipeline: {e}")
        feature_names = [f"Feature_{i}" for i in range(lr_model.coef_.shape[0])]

    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': lr_model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)

    logger.info(f"\nIntercept: {lr_model.intercept_:.3f}")
    logger.info("\nTop Features (by absolute coefficient magnitude):")
    for idx, row in coef_df.head(10).iterrows():
        logger.info(f"  {row['Feature']:30s} | {row['Coefficient']:8.3f}")

    # ========================================================================
    # STEP 7: PREPARE MODELS AND PIPELINES FOR RETURN
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Final Model Setup")
    logger.info("=" * 80)

    # Full pipeline for production use (raw data -> predictions)
    lr_pipeline = Pipeline([
        ('preprocessing', feature_pipeline),
        ('model', lr_model)
    ])

    baseline_pipeline = Pipeline([
        ('preprocessing', feature_pipeline),
        ('model', baseline)
    ])

    logger.info("Linear Regression pipeline created")
    logger.info("Baseline pipeline created")

    # ========================================================================
    # PREPARE EVALUATION DATA
    # ========================================================================
    evaluation_data = {
        'baseline_metrics': baseline_metrics,
        'lr_metrics': lr_metrics,
        'rmse_improvement_pct': rmse_improvement,
        'r2_improvement': r2_improvement,
        'cv_r2_scores': cv_scores_r2,
        'cv_r2_mean': cv_scores_r2.mean(),
        'cv_r2_std': cv_scores_r2.std(),
        'coefficients_df': coef_df,
        'intercept': lr_model.intercept_,
        'feature_names': feature_names
    }

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)

    return lr_pipeline, baseline_pipeline, feature_pipeline, X_test, y_test, evaluation_data
