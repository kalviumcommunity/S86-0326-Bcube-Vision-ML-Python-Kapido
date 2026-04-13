"""
Demonstration: MSE and R² Evaluation Best Practices

This script shows how to use the RegressionMetricsEvaluator class with
concrete examples from the regression evaluation lesson.

EXAMPLES DEMONSTRATED:
1. Basic test set evaluation with all metrics
2. Baseline comparison (mean predictor)
3. Cross-validation for stability assessment
4. Interpretation when MSE and R² tell different stories
5. Full evaluation report generation
"""

import logging
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .evaluate_regression_metrics import RegressionMetricsEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

logger = logging.getLogger(__name__)


def example_1_basic_evaluation():
    """
    EXAMPLE 1: Basic Test Set Evaluation
    
    Demonstrates computing MSE, RMSE, MAE, and R² on a held-out test set.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Test Set Evaluation")
    print("="*70)
    
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        noise=20,
        random_state=42
    )
    
    # Split data BEFORE any preprocessing (prevents data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and fit model with preprocessing pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    evaluator = RegressionMetricsEvaluator()
    metrics = evaluator.evaluate_on_test_set(y_test, y_pred, "Linear Regression")
    
    print(f"\n✓ Test Set Metrics Computed:")
    print(f"  MSE:  {metrics['mse']:.4f} (squared units)")
    print(f"  RMSE: {metrics['rmse']:.4f} (same units as target) ← Use for reporting")
    print(f"  MAE:  {metrics['mae']:.4f} (average absolute error)")
    print(f"  R²:   {metrics['r2']:.4f} (fraction of variance explained)")


def example_2_baseline_comparison():
    """
    EXAMPLE 2: Baseline Comparison
    
    Demonstrates the critical principle: ALWAYS compare against a baseline.
    R² has no meaning without a baseline reference.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Baseline Comparison (Mean Predictor)")
    print("="*70)
    print("\nPrinciple: R²=0 is defined as always predicting the mean.")
    print("Without baseline comparison, R² values are uninterpretable.\n")
    
    # Generate data
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        noise=30,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Compare with baseline
    evaluator = RegressionMetricsEvaluator()
    comparison = evaluator.compare_with_baseline(
        X_train, y_train, X_test, y_test, y_pred, "Linear Regression"
    )
    
    # Interpretation
    baseline_r2 = comparison['baseline']['r2']
    model_r2 = comparison['model']['r2']
    improvement = (model_r2 - baseline_r2) * 100
    
    print(f"\n✓ Comparison Summary:")
    print(f"  Baseline R²:        {baseline_r2:.4f} (≈0, as expected)")
    print(f"  Model R²:           {model_r2:.4f}")
    print(f"  Improvement:        {improvement:.1f} percentage points")
    
    if improvement > 0:
        print(f"\n  → Model explains {model_r2*100:.1f}% of target variance")
    else:
        print(f"\n  → ⚠ Model is WORSE than baseline (red flag!)")


def example_3_cross_validation():
    """
    EXAMPLE 3: Cross-Validation for Stability
    
    Demonstrates that a single test split can be misleading.
    Cross-validation shows performance consistency across data subsets.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Cross-Validation for Stability Assessment")
    print("="*70)
    print("\nA high mean R² with high std dev means: unstable model, possible overfitting\n")
    
    # Generate data
    X, y = make_regression(
        n_samples=150,
        n_features=5,
        noise=20,
        random_state=42
    )
    
    # Create model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    
    # Cross-validate
    evaluator = RegressionMetricsEvaluator()
    
    cv_r2 = evaluator.cross_validate_r2(pipeline, X, y, cv=5)
    cv_rmse = evaluator.cross_validate_rmse(pipeline, X, y, cv=5)
    
    # Interpretation
    print(f"\n✓ Cross-Validation Results (5 folds):")
    print(f"\n  R² Scores:")
    print(f"    Individual: {cv_r2['scores'].round(3)}")
    print(f"    Mean: {cv_r2['mean']:.4f} ± {cv_r2['std']:.4f}")
    
    print(f"\n  RMSE Scores:")
    print(f"    Individual: {cv_rmse['scores'].round(3)}")
    print(f"    Mean: {cv_rmse['mean']:.4f} ± {cv_rmse['std']:.4f}")
    
    # Interpretation guide
    if cv_r2['std'] < 0.05:
        print(f"\n  → ✓ STABLE model (low std dev: {cv_r2['std']:.3f})")
    elif cv_r2['std'] < 0.15:
        print(f"\n  → ~ MODERATE stability (std dev: {cv_r2['std']:.3f})")
    else:
        print(f"\n  → ⚠ UNSTABLE model (high std dev: {cv_r2['std']:.3f}) — possible overfitting")


def example_4_contrasting_stories():
    """
    EXAMPLE 4: When MSE and R² Tell Different Stories
    
    Demonstrates the principle: MSE (absolute) and R² (relative) can move
    in opposite directions depending on target variance.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: When MSE and R² Tell Different Stories")
    print("="*70)
    print("\nMSE is absolute (in squared target units)")
    print("R² is relative (compared to mean baseline)\n")
    
    # Scenario A: Low MSE, Low R² (target has low variance)
    print("SCENARIO A: Low MSE, Low R² ← Target variance is small")
    print("-" * 70)
    
    y_test_a = np.array([10.1, 10.2, 10.0, 10.3, 10.1, 9.9, 10.2, 10.0])
    y_pred_a = np.array([10.0, 10.3, 10.1, 10.2, 10.0, 10.0, 10.1, 9.9])
    
    evaluator = RegressionMetricsEvaluator()
    metrics_a = evaluator.evaluate_on_test_set(y_test_a, y_pred_a, "Model A")
    
    print(f"  Target values:   {y_test_a}")
    print(f"  Predictions:     {y_pred_a}")
    print(f"\n  MSE:  {metrics_a['mse']:.6f}  ← Looks good (absolute error is tiny)")
    print(f"  R²:   {metrics_a['r2']:.4f}       ← Looks bad (barely explains variance)")
    print(f"\n  Interpretation: Model's errors are small in absolute terms, but the")
    print(f"  target variable itself has very low variance. The mean baseline also")
    print(f"  makes tiny errors, so relative improvement (R²) is poor.")
    
    # Scenario B: High MSE, High R² (target has high variance)
    print("\n\nSCENARIO B: High MSE, High R² ← Target variance is large")
    print("-" * 70)
    
    y_test_b = np.array([10, 50, 100, 25, 75, 120, 30, 80])
    y_pred_b = np.array([15, 45, 95, 30, 70, 125, 25, 85])
    
    metrics_b = evaluator.evaluate_on_test_set(y_test_b, y_pred_b, "Model B")
    
    print(f"  Target values:   {y_test_b}")
    print(f"  Predictions:     {y_pred_b}")
    print(f"\n  MSE:  {metrics_b['mse']:.4f}  ← Looks bad (absolute error seems large)")
    print(f"  R²:   {metrics_b['r2']:.4f}     ← Looks good (explains most variance)")
    print(f"\n  Interpretation: The target values span a wide range. Absolute errors")
    print(f"  appear large, but the baseline's errors are even larger. R² correctly")
    print(f"  identifies strong relative performance. Check if RMSE={np.sqrt(metrics_b['mse']):.2f}")
    print(f"  is acceptable for your business use case.")


def example_5_full_report():
    """
    EXAMPLE 5: Full Evaluation Report
    
    Demonstrates the complete evaluation workflow with a formatted report.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Complete Evaluation Report")
    print("="*70)
    
    # Generate data
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        noise=25,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    from sklearn.dummy import DummyRegressor
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Baseline
    baseline = DummyRegressor(strategy='mean')
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)
    
    # Generate report
    evaluator = RegressionMetricsEvaluator()
    evaluator.print_evaluation_report(
        y_test, y_pred, y_pred_baseline, "Linear Regression"
    )


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MSE AND R² EVALUATION: PRACTICAL EXAMPLES")
    print("="*70)
    print("\nKey Principle from Lesson:")
    print("  Neither MSE nor R² alone tells the full story.")
    print("  MSE: Absolute squared error (sensitive to outliers)")
    print("  R²:  Relative improvement over mean baseline (0-1 scale)")
    print("  → ALWAYS interpret both metrics together\n")
    
    example_1_basic_evaluation()
    example_2_baseline_comparison()
    example_3_cross_validation()
    example_4_contrasting_stories()
    example_5_full_report()
    
    print("\n" + "="*70)
    print("END OF EXAMPLES")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. Use MSE internally for optimization; use RMSE for reporting")
    print("  2. Compare against baseline (mean predictor) to contextualize R²")
    print("  3. Use cross-validation to assess stability across data subsets")
    print("  4. Interpret MSE and R² together — they measure different things")
    print("  5. Negative R² is a red flag: model is worse than baseline\n")


if __name__ == '__main__':
    main()
