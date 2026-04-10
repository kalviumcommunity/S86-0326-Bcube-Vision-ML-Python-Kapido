"""
Linear Regression evaluation module for ride-sharing duration prediction.

This module computes comprehensive regression evaluation metrics on held-out
test data. It is completely separate from training to allow:
- Reproducible evaluation of multiple models
- Easy integration with experiment tracking systems
- Clear audit trail of how metrics are computed

METRICS COMPUTED:
- MSE (Mean Squared Error): Training objective - minimized by Linear Regression
- RMSE (Root Mean Squared Error): Same units as target (minutes) - interpretable
- MAE (Mean Absolute Error): Average absolute error - robust to outliers
- R² Score: Proportion of variance explained (0 = bad, 1 = perfect)

KEY PRINCIPLE:
R² should ALWAYS be compared against a baseline (e.g., R²=0 for mean predictor).
An R² of 0.45 might be excellent or terrible depending on domain and baseline.
"""
import logging
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create logger for this module
logger = logging.getLogger(__name__)


def evaluate_linear_regression(
    y_test: np.ndarray,
    lr_predictions: np.ndarray,
    baseline_predictions: np.ndarray = None,
    save_plot_path: str = None
) -> Dict[str, float]:
    """
    Compute comprehensive regression evaluation metrics.

    Args:
        y_test: True target values (shape: (n_samples,))
        lr_predictions: Linear Regression predictions (shape: (n_samples,))
        baseline_predictions: Optional baseline predictions for comparison
        save_plot_path: Optional path to save residual plots

    Returns:
        Dictionary containing:
        - mse: Mean Squared Error
        - rmse: Root Mean Squared Error
        - mae: Mean Absolute Error
        - r2: R² Score
        - residuals: Prediction errors (y_test - lr_predictions)
        - baseline_metrics: Optional baseline comparison metrics

    Raises:
        ValueError: If inputs are invalid.
    """
    # Input validation
    if lr_predictions is None or len(lr_predictions) == 0:
        raise ValueError("lr_predictions cannot be None or empty")
    if y_test is None or len(y_test) == 0:
        raise ValueError("y_test cannot be None or empty")
    if len(lr_predictions) != len(y_test):
        raise ValueError(f"Length mismatch: {len(lr_predictions)} vs {len(y_test)}")

    logger.info(f"Evaluating Linear Regression on {len(y_test)} test samples")

    # Compute metrics
    mse = mean_squared_error(y_test, lr_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, lr_predictions)
    r2 = r2_score(y_test, lr_predictions)
    residuals = y_test - lr_predictions

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'residuals': residuals,
        'residuals_mean': np.mean(residuals),
        'residuals_std': np.std(residuals)
    }

    # Log results
    logger.info(f"MSE:  {mse:.3f}")
    logger.info(f"RMSE: {rmse:.3f} minutes")
    logger.info(f"MAE:  {mae:.3f} minutes")
    logger.info(f"R²:   {r2:.3f}")

    # Optional: baseline comparison
    if baseline_predictions is not None:
        baseline_mse = mean_squared_error(y_test, baseline_predictions)
        baseline_rmse = np.sqrt(baseline_mse)
        baseline_mae = mean_absolute_error(y_test, baseline_predictions)
        baseline_r2 = r2_score(y_test, baseline_predictions)

        metrics['baseline_metrics'] = {
            'mse': baseline_mse,
            'rmse': baseline_rmse,
            'mae': baseline_mae,
            'r2': baseline_r2
        }

        logger.info("\nBaseline Comparison:")
        logger.info(f"  Baseline RMSE: {baseline_rmse:.3f} minutes")
        logger.info(f"  Model RMSE: {rmse:.3f} minutes")
        logger.info(f"  RMSE Improvement: {100 * (baseline_rmse - rmse) / baseline_rmse:.1f}%")
        logger.info(f"  Baseline R²: {baseline_r2:.3f}")
        logger.info(f"  Model R²: {r2:.3f}")
        logger.info(f"  R² Improvement: {r2 - baseline_r2:.3f}")

    # Optional: residual plot
    if save_plot_path:
        try:
            _plot_residuals(y_test, lr_predictions, residuals, save_plot_path)
        except Exception as e:
            logger.warning(f"Could not save residual plot: {e}")

    return metrics


def _plot_residuals(y_true, y_pred, residuals, save_path):
    """Create and save residual diagnostic plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Actual vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
    axes[0, 0].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Duration (minutes)')
    axes[0, 0].set_ylabel('Predicted Duration (minutes)')
    axes[0, 0].set_title('Actual vs Predicted Values')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Residuals vs Fitted Values
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Fitted Values (minutes)')
    axes[0, 1].set_ylabel('Residuals (minutes)')
    axes[0, 1].set_title('Residuals vs Fitted Values')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Distribution of Residuals
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Residuals (minutes)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Q-Q Plot (approximate)
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = np.sort(np.random.normal(0, residuals.std(), len(residuals)))
    axes[1, 1].scatter(theoretical_quantiles, sorted_residuals, alpha=0.5)
    axes[1, 1].plot([sorted_residuals.min(), sorted_residuals.max()], 
                    [sorted_residuals.min(), sorted_residuals.max()], 'r--', lw=2)
    axes[1, 1].set_xlabel('Theoretical Quantiles')
    axes[1, 1].set_ylabel('Sample Quantiles')
    axes[1, 1].set_title('Q-Q Plot')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    logger.info(f"Residual plots saved to {save_path}")
    plt.close()


def print_evaluation_summary(metrics: Dict[str, Any]) -> None:
    """Print a formatted summary of evaluation metrics."""
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)

    logger.info("\nREGRESSION METRICS (Test Set):")
    logger.info(f"  Mean Squared Error (MSE):      {metrics['mse']:.3f}")
    logger.info(f"  Root Mean Squared Error (RMSE): {metrics['rmse']:.3f} minutes")
    logger.info(f"  Mean Absolute Error (MAE):     {metrics['mae']:.3f} minutes")
    logger.info(f"  R² Score:                       {metrics['r2']:.3f}")

    logger.info("\nRESIDUAL STATISTICS:")
    logger.info(f"  Mean of Residuals:             {metrics['residuals_mean']:.4f}")
    logger.info(f"  Std of Residuals:              {metrics['residuals_std']:.3f}")

    if 'baseline_metrics' in metrics:
        baseline = metrics['baseline_metrics']
        logger.info("\nBASELINE COMPARISON:")
        logger.info(f"  Baseline RMSE:                 {baseline['rmse']:.3f} minutes")
        logger.info(f"  Model RMSE:                    {metrics['rmse']:.3f} minutes")
        logger.info(f"  RMSE Improvement:              "
                   f"{100 * (baseline['rmse'] - metrics['rmse']) / baseline['rmse']:.1f}%")
        logger.info(f"  Baseline R²:                   {baseline['r2']:.3f}")
        logger.info(f"  Model R²:                      {metrics['r2']:.3f}")

    logger.info("\nINTERPRETATION:")
    logger.info(f"  RMSE of {metrics['rmse']:.2f} minutes means predictions are off by")
    logger.info(f"  ~{metrics['rmse']:.0f} minutes on average (±1 std dev).")
    logger.info(f"  R² of {metrics['r2']:.3f} means the model explains {100*metrics['r2']:.1f}%")
    logger.info(f"  of variance in ride duration.")
    logger.info("=" * 80 + "\n")
