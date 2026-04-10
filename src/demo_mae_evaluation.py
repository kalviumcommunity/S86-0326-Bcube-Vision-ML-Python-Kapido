"""
Complete Mean Absolute Error (MAE) evaluation demonstration.

This script demonstrates the full MAE evaluation workflow as covered in the lesson:

1. Train baseline and model
2. Compute comprehensive metrics (MAE, MSE, RMSE, R², MAPE)
3. Compare MAE vs RMSE vs MSE
4. Interpret MAE with proper context
5. Use cross-validation with MAE
6. Create diagnostic visualizations
7. Highlight common mistakes to avoid

KEY INSIGHT:
MAE is most powerful when compared against a baseline, contextualized by
target scale, and understood in business terms. Isolated, it's just a number.

EXECUTION:
    From project root:
    python -m src.demo_mae_evaluation
"""
import logging
import sys
import os
from typing import Dict, Any

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/mae_evaluation.log')
    ]
)

logger = logging.getLogger(__name__)

# Safe to import after logging configured
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

from src.config import DATA_PATH, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.data_loader import load_data
from src.preprocessing import build_preprocessing_pipeline
from src.evaluate_mae import (
    compute_regression_metrics,
    compare_mae_vs_rmse_vs_mse,
    interpret_mae_with_context,
    plot_mae_comparison,
    explain_mae_mistakes
)


def create_directories():
    """Ensure required directories exist."""
    dirs = ['logs', 'reports', 'models']
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def train_models(
    X_train, X_test, y_train, y_test,
    feature_pipeline
) -> tuple:
    """Train baseline and Linear Regression models."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PHASE")
    logger.info("=" * 80)

    # Preprocess
    logger.info("\nPreprocessing training data...")
    X_train_processed = feature_pipeline.fit_transform(X_train)
    X_test_processed = feature_pipeline.transform(X_test)
    logger.info(f"Training shape: {X_train_processed.shape}")
    logger.info(f"Test shape: {X_test_processed.shape}")

    # Baseline
    logger.info("\nTraining baseline (mean predictor)...")
    baseline = DummyRegressor(strategy='mean')
    baseline.fit(X_train_processed, y_train)
    baseline_preds_train = baseline.predict(X_train_processed)
    baseline_preds_test = baseline.predict(X_test_processed)
    logger.info("[OK] Baseline trained")

    # Model
    logger.info("\nTraining Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_processed, y_train)
    lr_preds_train = lr_model.predict(X_train_processed)
    lr_preds_test = lr_model.predict(X_test_processed)
    logger.info("[OK] Linear Regression trained")

    return (baseline, lr_model, baseline_preds_test, lr_preds_test,
            X_train_processed, X_test_processed)


def evaluate_models(
    y_test: np.ndarray,
    baseline_preds: np.ndarray,
    lr_preds: np.ndarray
) -> Dict[str, Any]:
    """Evaluate both models comprehensively."""
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION PHASE: COMPREHENSIVE METRICS")
    logger.info("=" * 80)

    # Baseline metrics
    logger.info("\n>>> BASELINE METRICS (Mean Predictor)")
    baseline_metrics = compute_regression_metrics(
        y_test, baseline_preds, model_name="Baseline"
    )

    # Model metrics with baseline comparison
    logger.info("\n>>> MODEL METRICS (Linear Regression)")
    model_metrics = compute_regression_metrics(
        y_test, lr_preds, baseline_preds, model_name="Linear Regression"
    )

    return {
        'baseline_metrics': baseline_metrics,
        'model_metrics': model_metrics,
        'y_test': y_test,
        'baseline_preds': baseline_preds,
        'lr_preds': lr_preds
    }


def compare_metrics(results: Dict[str, Any]) -> None:
    """Compare MAE vs RMSE vs MSE."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: COMPARING MAE vs RMSE vs MSE")
    logger.info("=" * 80)

    y_test = results['y_test']
    lr_preds = results['lr_preds']

    comparison = compare_mae_vs_rmse_vs_mse(y_test, lr_preds, verbose=True)

    logger.info("\nKEY TAKEAWAYS:")
    logger.info("1. MAE and RMSE are often similar for small datasets")
    logger.info("2. RMSE emphasizes large errors (outlier sensitivity)")
    logger.info("3. MAE is linearly proportional to error magnitude")
    logger.info(f"4. In this case: RMSE={comparison['rmse']:.4f}, MAE={results['model_metrics']['mae']:.4f}")

    if comparison['rmse'] > results['model_metrics']['mae'] * 1.1:
        logger.info("5. RMSE > MAE suggests presence of outliers or skewed errors")
    else:
        logger.info("5. RMSE approx equal to MAE suggests relatively uniform errors")


def interpret_with_context(results: Dict[str, Any]) -> None:
    """Interpret MAE with proper context."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: INTERPRETING MAE WITH CONTEXT")
    logger.info("=" * 80)

    y_test = results['y_test']
    model_mae = results['model_metrics']['mae']
    baseline_mae = results['baseline_metrics']['mae']

    interpretation = interpret_mae_with_context(
        model_mae, baseline_mae, y_test, target_name="Ride Duration (minutes)"
    )

    logger.info("\nQUALITY JUDGMENT:")
    if interpretation['mae_pct_of_mean'] < 5:
        logger.info("This model meets or exceeds typical business requirements.")
        logger.info("Recommendations:")
        logger.info("  1. Consider for production deployment")
        logger.info("  2. May still benefit from feature engineering")
        logger.info("  3. Monitor performance on new data")
    elif interpretation['mae_pct_of_mean'] < 10:
        logger.info("This model shows promise but has room for improvement.")
        logger.info("Recommendations:")
        logger.info("  1. Try adding polynomial features")
        logger.info("  2. Engineer domain-specific features")
        logger.info("  3. Try more complex models (Tree, Forest, Boosting)")
    else:
        logger.info("This model needs significant improvement before production use.")
        logger.info("Recommendations:")
        logger.info("  1. Conduct thorough feature engineering")
        logger.info("  2. Try tree-based ensemble methods")
        logger.info("  3. Investigate data quality issues")
        logger.info("  4. Confirm target variable is predictable")


def cross_validate_with_mae(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lr_model
) -> None:
    """Use cross-validation to assess MAE stability."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: CROSS-VALIDATION WITH MAE")
    logger.info("=" * 80)

    logger.info("\nPerforming 5-fold cross-validation...")
    logger.info("(Note: scikit-learn returns negative MAE, we negate it)")

    # CV scores
    cv_scores = cross_val_score(
        lr_model, X_train, y_train,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    mae_scores = -cv_scores  # Flip sign to get actual MAE

    logger.info("\nCROSS-VALIDATION RESULTS:")
    logger.info(f"  Fold 1 MAE: {mae_scores[0]:.4f}")
    logger.info(f"  Fold 2 MAE: {mae_scores[1]:.4f}")
    logger.info(f"  Fold 3 MAE: {mae_scores[2]:.4f}")
    logger.info(f"  Fold 4 MAE: {mae_scores[3]:.4f}")
    logger.info(f"  Fold 5 MAE: {mae_scores[4]:.4f}")
    logger.info(f"\n  Mean CV MAE:  {mae_scores.mean():.4f}")
    logger.info(f"  Std CV MAE:   {mae_scores.std():.4f}")

    # Interpretation
    logger.info("\nINTERPRETATION:")
    if mae_scores.std() < 0.5:
        logger.info("  [Excellent] Low variance across folds")
        logger.info("  Model performs consistently regardless of train/test split")
        logger.info("  Confidence in generalization is high")
    elif mae_scores.std() < 1.0:
        logger.info("  [Good] Moderate variance across folds")
        logger.info("  Model is reasonably stable")
    else:
        logger.info("  [Caution] High variance across folds")
        logger.info("  Model performance is sensitive to which data it trains on")
        logger.info("  Consider: regularization, more features, or more data")

    ci_lower = mae_scores.mean() - 1.96 * mae_scores.std()
    ci_upper = mae_scores.mean() + 1.96 * mae_scores.std()
    logger.info(f"\n95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    logger.info("New data will likely have MAE in this range")


def model_selection_example(results: Dict[str, Any]) -> None:
    """Show how to use MAE for model selection."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4: USING MAE FOR MODEL SELECTION")
    logger.info("=" * 80)

    model_mae = results['model_metrics']['mae']
    baseline_mae = results['baseline_metrics']['mae']

    logger.info("\nMODEL COMPARISON (hypothetical ensemble):")
    logger.info("=" * 80)
    logger.info("Model                    | Mean CV MAE | Std CV MAE | Complexity | Interpret.")
    logger.info("=" * 80)
    logger.info(f"Baseline (mean)          | {baseline_mae:10.4f}   | 0.001      | Very Low   | Excellent")
    logger.info(f"Linear Regression        | {model_mae:10.4f}   | 0.050      | Low        | Excellent")
    logger.info("Ridge Regression (alpha)  | 8.5000     | 0.060      | Low        | Good")
    logger.info("Random Forest (100 trees) | 6.8000     | 0.075      | Medium     | Poor")
    logger.info("=" * 80)

    logger.info("\nDECISION PROCESS:")
    logger.info("\n1. PERFORMANCE vs COMPLEXITY TRADE-OFF:")
    logger.info("   Random Forest has lowest MAE but:")
    logger.info("   - Less interpretable (can't explain feature importance easily)")
    logger.info("   - Slower inference (100 trees vs. 1 linear model)")
    logger.info("   - Higher maintenance cost in production")

    logger.info("\n2. BUSINESS REQUIREMENTS:")
    logger.info("   If MAE < 5 is acceptable: Use Linear Regression")
    logger.info("   If need MAE < 7: Use Random Forest")
    logger.info("   If need interpretability: Use Ridge Regression")

    logger.info("\n3. RECOMMENDATION:")
    improvement_pct = ((baseline_mae - model_mae) / baseline_mae * 100)
    if improvement_pct > 50:
        logger.info("   Use Linear Regression - simple, interpretable, sufficient")
    else:
        logger.info("   Try Tree/Forest models - adequate gain from added complexity")


def visualize_mae(results: Dict[str, Any]) -> None:
    """Create visualization of MAE comparison."""
    logger.info("\n" + "=" * 80)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("=" * 80)

    y_test = results['y_test']
    baseline_preds = results['baseline_preds']
    lr_preds = results['lr_preds']
    model_mae = results['model_metrics']['mae']
    baseline_mae = results['baseline_metrics']['mae']

    logger.info("Generating 4-panel MAE comparison plot...")
    plot_mae_comparison(
        y_test, lr_preds, baseline_preds,
        model_mae, baseline_mae,
        save_path='reports/mae_comparison.png'
    )


def common_mistakes_check(results: Dict[str, Any]) -> None:
    """Check for and explain common MAE mistakes."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 5: COMMON MISTAKES TO AVOID")
    logger.info("=" * 80)

    y_test = results['y_test']
    lr_preds = results['lr_preds']

    explain_mae_mistakes(y_test, lr_preds)


def main() -> Dict[str, Any]:
    """Execute complete MAE evaluation workflow."""
    try:
        create_directories()

        logger.info("\n" + "=" * 80)
        logger.info("MEAN ABSOLUTE ERROR (MAE) EVALUATION DEMONSTRATION")
        logger.info("=" * 80)
        logger.info("\nLesson: Evaluating Regression Models Using MAE")
        logger.info("Problem: Predict ride duration from ride-sharing features")
        logger.info("Baseline: DummyRegressor(strategy='mean')")
        logger.info("Model: LinearRegression with StandardScaler")

        # Load and prepare data
        logger.info("\n" + "=" * 80)
        logger.info("DATA PREPARATION")
        logger.info("=" * 80)

        df = load_data(DATA_PATH)
        logger.info(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")

        X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES].copy()
        y = df['estimated_time'].copy()
        logger.info(f"Target: estimated_time (ride duration in minutes)")
        logger.info(f"  Mean: {y.mean():.2f}, Std: {y.std():.2f}")
        logger.info(f"  Range: {y.min():.2f} to {y.max():.2f}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info(f"Train: {X_train.shape[0]} samples (80%)")
        logger.info(f"Test:  {X_test.shape[0]} samples (20%)")

        # Build preprocessing pipeline
        feature_pipeline = build_preprocessing_pipeline(
            CATEGORICAL_FEATURES, NUMERICAL_FEATURES
        )

        # Train models
        baseline, lr_model, baseline_preds, lr_preds, X_train_proc, X_test_proc = \
            train_models(X_train, X_test, y_train, y_test, feature_pipeline)

        # Comprehensive evaluation
        results = evaluate_models(y_test, baseline_preds, lr_preds)

        # MAE vs RMSE comparison
        compare_metrics(results)

        # Interpretation with context
        interpret_with_context(results)

        # Cross-validation
        cross_validate_with_mae(X_train_proc, y_train, lr_model)

        # Model selection example
        model_selection_example(results)

        # Visualizations
        visualize_mae(results)

        # Common mistakes
        common_mistakes_check(results)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY & KEY TAKEAWAYS")
        logger.info("=" * 80)

        logger.info("\n1. MAE INTERPRETATION:")
        logger.info(f"   Model MAE: {results['model_metrics']['mae']:.2f} minutes")
        logger.info(f"   Baseline MAE: {results['baseline_metrics']['mae']:.2f} minutes")
        logger.info(f"   Improvement: {results['model_metrics'].get('mae_improvement', 0):.2f} "
                   f"({results['model_metrics'].get('mae_pct_improvement', 0):.1f}%)")

        logger.info("\n2. CONTEXTUAL MEANING:")
        mean_target = y_test.mean()
        mae_pct = (results['model_metrics']['mae'] / mean_target * 100)
        logger.info(f"   MAE = {mae_pct:.1f}% of mean target value")
        if mae_pct < 5:
            logger.info("   Assessment: High-quality predictions")
        elif mae_pct < 10:
            logger.info("   Assessment: Acceptable predictions")
        else:
            logger.info("   Assessment: Predictions need improvement")

        logger.info("\n3. WHAT MAE TELLS US:")
        logger.info("   - Average magnitude of prediction error")
        logger.info("   - NOT directional bias (use residual plots)")
        logger.info("   - NOT outlier sensitivity (compare with RMSE)")
        logger.info("   - More interpretable than MSE or R²")

        logger.info("\n4. WHAT MAE DOESN'T TELL US:")
        logger.info("   - Whether errors are normally distributed")
        logger.info("   - Whether model under/over-predicts systematically")
        logger.info("   - Whether large outliers dominate (see RMSE)")
        logger.info("   - Whether observed relationships are still strong elsewhere")

        logger.info("\n5. BEST PRACTICES:")
        logger.info("   [OK] Always compare against a baseline")
        logger.info("   [OK] Contextualize MAE as % of mean target")
        logger.info("   [OK] Use cross-validation to assess stability")
        logger.info("   [OK] Report MAE alongside RMSE and R²")
        logger.info("   [OK] Confirm baseline isn't secretly doing well")
        logger.info("   [OK] Inspect residuals for systematic bias")
        logger.info("   [OK] Connect to business tolerance thresholds")

        logger.info("\n" + "=" * 80)
        logger.info("DEMO COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nOutput files:")
        logger.info(f"  - Logs: logs/mae_evaluation.log")
        logger.info(f"  - Plots: reports/mae_comparison.png")
        logger.info("\n")

        return results

    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    results = main()
