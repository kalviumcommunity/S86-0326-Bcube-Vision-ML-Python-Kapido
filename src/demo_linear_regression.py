"""
Complete Linear Regression demonstration script.

This script demonstrates the full Linear Regression workflow as covered in
the supervised learning fundamentals lesson:

1. Data loading
2. Train/test split (BEFORE preprocessing)
3. Feature preprocessing (scaling, encoding)
4. Baseline model training (mean predictor)
5. Linear Regression model training
6. Comprehensive evaluation (MSE, RMSE, MAE, R², CV)
7. Coefficient interpretation
8. Model comparison and insights

EXECUTION:
    From project root:
    python -m src.demo_linear_regression

    or

    python src/demo_linear_regression.py
"""
import logging
import sys
import os

# Configure logging FIRST (before importing modules)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/linear_regression.log')
    ]
)

logger = logging.getLogger(__name__)

# Now safe to import
import numpy as np
import pandas as pd
from src.train_linear_regression import train_linear_regression_model
from src.evaluate_linear_regression import evaluate_linear_regression, print_evaluation_summary
from src.config import DATA_PATH, CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def create_directories():
    """Ensure required directories exist."""
    dirs = ['logs', 'reports', 'models']
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def main():
    """Execute complete Linear Regression workflow."""
    try:
        create_directories()

        logger.info("\n" + "=" * 80)
        logger.info("LINEAR REGRESSION TRAINING DEMONSTRATION")
        logger.info("=" * 80)
        logger.info("\nProblem: Predict ride duration (estimated_time) from features")
        logger.info("Method:  Linear Regression with StandardScaler preprocessing")
        logger.info("Baseline: DummyRegressor predicting mean duration")
        logger.info("Evaluation: RMSE, MAE, R², cross-validation")

        # ====================================================================
        # TRAIN MODELS
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING PHASE")
        logger.info("=" * 80)

        lr_pipeline, baseline_pipeline, feature_pipeline, X_test, y_test, eval_data = (
            train_linear_regression_model(
                data_path=DATA_PATH,
                target_column='estimated_time',
                categorical_cols=CATEGORICAL_FEATURES,
                numerical_cols=NUMERICAL_FEATURES,
                test_size=0.2,
                random_state=42
            )
        )

        # ====================================================================
        # EVALUATE MODELS
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION PHASE")
        logger.info("=" * 80)

        # Get predictions (X_test is raw/unprocessed, pipelines handle preprocessing)
        lr_predictions = lr_pipeline.predict(X_test)
        baseline_predictions = baseline_pipeline.predict(X_test)

        # Evaluate
        metrics = evaluate_linear_regression(
            y_test=y_test,
            lr_predictions=lr_predictions,
            baseline_predictions=baseline_predictions,
            save_plot_path='reports/linear_regression_residuals.png'
        )

        # Print summary
        print_evaluation_summary(metrics)

        # ====================================================================
        # INTERPRETATION
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("MODEL INTERPRETATION")
        logger.info("=" * 80)

        logger.info("\nIntercept (baseline prediction): {:.3f} minutes".format(
            eval_data['intercept']
        ))

        logger.info("\nTop 10 Most Important Features (by absolute coefficient):")
        logger.info("-" * 80)
        for idx, (_, row) in enumerate(eval_data['coefficients_df'].head(10).iterrows(), 1):
            sign = "+" if row['Coefficient'] > 0 else ""
            logger.info(
                f"{idx:2d}. {row['Feature']:30s} {sign}{row['Coefficient']:8.4f} "
                f"(change: {row['Coefficient']:+.2f} min)"
            )

        # ====================================================================
        # INSIGHTS
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("KEY INSIGHTS")
        logger.info("=" * 80)

        r2_improvement = eval_data['r2_improvement']
        rmse_improvement = eval_data['rmse_improvement_pct']

        logger.info("\n1. BASELINE COMPARISON:")
        logger.info(f"   • Linear Regression RMSE:  {eval_data['lr_metrics']['RMSE']:.2f} minutes")
        logger.info(f"   • Baseline RMSE:           {eval_data['baseline_metrics']['RMSE']:.2f} minutes")
        logger.info(f"   • RMSE Improvement:        {rmse_improvement:.1f}%")
        logger.info(f"   • R² Improvement:          {r2_improvement:+.3f}")

        if rmse_improvement > 0:
            logger.info("\n   [OK] Model significantly outperforms the mean baseline")
        else:
            logger.info("\n   [ERROR] Model performs worse than baseline - reconsider features or try other algorithms")

        logger.info("\n2. MODEL QUALITY:")
        r2 = eval_data['lr_metrics']['R2']
        if r2 > 0.7:
            logger.info(f"   • Strong model: R²={r2:.3f} (explains {100*r2:.1f}% of variance)")
        elif r2 > 0.4:
            logger.info(f"   • Moderate model: R²={r2:.3f} (explains {100*r2:.1f}% of variance)")
        else:
            logger.info(f"   • Weak model: R²={r2:.3f} (explains {100*r2:.1f}% of variance)")
            logger.info("\n   → Consider: more features, feature engineering, non-linear models")

        logger.info("\n3. CROSS-VALIDATION STABILITY:")
        cv_mean = eval_data['cv_r2_mean']
        cv_std = eval_data['cv_r2_std']
        logger.info(f"   • Mean CV R²:     {cv_mean:.3f}")
        logger.info(f"   • Std CV R²:      {cv_std:.3f}")

        if cv_std < 0.05:
            logger.info("   • Model is stable across folds (consistent performance)")
        else:
            logger.info("   • Model shows variance across folds (may overfit on training data)")

        logger.info("\n4. RESIDUAL ANALYSIS:")
        residuals_mean = metrics['residuals_mean']
        residuals_std = metrics['residuals_std']
        logger.info(f"   • Mean residual: {residuals_mean:.4f} (should be near zero)")
        logger.info(f"   • Std residual:  {residuals_std:.2f} minutes")

        if abs(residuals_mean) < 0.1:
            logger.info("   [OK] Residuals centered at zero (good)")
        else:
            logger.info("   [WARN] Residuals biased - model systematically over/under-predicts")

        # ====================================================================
        # RECOMMENDATIONS
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("NEXT STEPS")
        logger.info("=" * 80)

        logger.info("\n1. FEATURE ENGINEERING:")
        logger.info("   • Add interaction terms (e.g., location x time)")
        logger.info("   • Try polynomial features for non-linear relationships")
        logger.info("   • Add domain-specific features (rush hour indicator, weather, etc.)")

        logger.info("\n2. REGULARIZATION (if overfitting):")
        logger.info("   • Try Ridge Regression (L2 penalty) for correlated features")
        logger.info("   • Try Lasso Regression (L1 penalty) for automatic feature selection")

        logger.info("\n3. CHECK ASSUMPTIONS:")
        logger.info("   • Linear relationship: Plot residuals vs fitted values")
        logger.info("   • Homoscedasticity: Check if error variance is constant")
        logger.info("   • Multicollinearity: Compute VIF for each feature")
        logger.info("   • Normality: Q-Q plot of residuals (for inference, not prediction)")

        logger.info("\n4. COMPARE WITH OTHER MODELS:")
        logger.info("   • Decision Trees (non-linear)")
        logger.info("   • Random Forest (ensemble, robust to outliers)")
        logger.info("   • Gradient Boosting (powerful but complex)")

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80 + "\n")

        return {
            'lr_pipeline': lr_pipeline,
            'baseline_pipeline': baseline_pipeline,
            'metrics': metrics,
            'evaluation_data': eval_data
        }

    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    result = main()
