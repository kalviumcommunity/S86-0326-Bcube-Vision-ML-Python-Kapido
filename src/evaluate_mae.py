"""
Mean Absolute Error (MAE) evaluation module for regression models.

This module implements comprehensive MAE evaluation following best practices:

WHAT IS MAE?
Mean Absolute Error = average absolute difference between predicted and actual values
- Linear penalty: errors contribute proportionally (no squaring)
- Interpretable units: same as target variable
- Robust to outliers: large errors don't dominate excessively

KEY PRINCIPLE:
MAE in isolation is meaningless. Always compare:
1. Against a baseline (e.g., mean predictor with MAE=0 improvement)
2. In target units (% of mean, absolute units)
3. With context (business tolerance, application domain)
4. Across multiple folds (cross-validation)

WHEN TO USE MAE:
✓ Interpretability critical (stakeholders need concrete units)
✓ Outliers exist but shouldn't dominate metric
✓ Errors have roughly uniform cost (not exponential)
✓ Reporting to non-technical audiences
✓ Applications: sales forecasting, delivery time, demand prediction

WHEN MAE IS NOT IDEAL:
✗ Large errors are catastrophically costly → use RMSE
✗ Gradient-based training → MSE is differentiable
✗ Want to penalize inconsistency → use RMSE
✗ Errors systematically biased → inspect residuals
"""
import logging
from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)

# Create logger
logger = logging.getLogger(__name__)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: Optional[np.ndarray] = None,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics: MAE, MSE, RMSE, R², MAPE.

    Args:
        y_true: Actual values (shape: (n_samples,))
        y_pred: Predicted values (shape: (n_samples,))
        y_baseline: Optional baseline predictions for comparison
        model_name: Name of model for logging

    Returns:
        Dictionary containing:
        - mae: Mean Absolute Error
        - mse: Mean Squared Error
        - rmse: Root Mean Squared Error
        - r2: R² Score
        - mape: Mean Absolute Percentage Error
        - baseline_mae: Optional baseline MAE
        - mae_improvement: Optional improvement over baseline
        - mae_pct_improvement: Optional % improvement over baseline

    Raises:
        ValueError: If inputs invalid or lengths don't match
    """
    # Validation
    if y_pred is None or len(y_pred) == 0:
        raise ValueError("y_pred cannot be None or empty")
    if y_true is None or len(y_true) == 0:
        raise ValueError("y_true cannot be None or empty")
    if len(y_pred) != len(y_true):
        raise ValueError(f"Length mismatch: {len(y_pred)} vs {len(y_true)}")

    logger.info(f"Computing metrics for {model_name} ({len(y_true)} samples)")

    # Core metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'n_samples': len(y_true)
    }

    # Baseline comparison
    if y_baseline is not None:
        baseline_mae = mean_absolute_error(y_true, y_baseline)
        mae_improvement = baseline_mae - mae
        mae_pct_improvement = (mae_improvement / baseline_mae * 100) if baseline_mae > 0 else 0

        metrics.update({
            'baseline_mae': baseline_mae,
            'mae_improvement': mae_improvement,
            'mae_pct_improvement': mae_pct_improvement
        })

        logger.info(f"\n{model_name} Metrics:")
        logger.info(f"  MAE:  {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE % of baseline: {(mae/baseline_mae*100):.1f}%")
        logger.info(f"  MAE Improvement:   {mae_improvement:.4f} ({mae_pct_improvement:.1f}%)")
    else:
        logger.info(f"\n{model_name} Metrics:")
        logger.info(f"  MAE:  {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  R²:   {r2:.4f}")

    return metrics


def compare_mae_vs_rmse_vs_mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare MAE, MSE, and RMSE to illustrate differences in error penalties.

    KEY INSIGHT:
    - MAE: Linear penalty (error of 20 is 10× worse than error of 2)
    - MSE/RMSE: Quadratic penalty (error of 20 is 100× worse than error of 2)
    - RMSE: Same units as target (interpretable like MAE)
    - MSE: Squared units (not directly interpretable)

    Args:
        y_true: Actual values
        y_pred: Predicted values
        verbose: Whether to log comparison

    Returns:
        Dictionary with all three metrics and their properties
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Compute error array to show individual impact
    errors = np.abs(y_pred - y_true)
    squared_errors = errors ** 2

    # Identify outliers (top 5% of errors)
    error_threshold = np.percentile(errors, 95)
    outlier_mask = errors >= error_threshold
    n_outliers = np.sum(outlier_mask)

    result = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'errors': errors,
        'n_outliers': n_outliers,
        'outlier_pct': (n_outliers / len(errors) * 100) if len(errors) > 0 else 0,
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'median_error': np.median(errors)
    }

    if verbose:
        logger.info("\n" + "=" * 80)
        logger.info("MAE vs RMSE vs MSE: Understanding Error Penalties")
        logger.info("=" * 80)

        logger.info("\nMETRIC VALUES:")
        logger.info(f"  MAE (linear penalty):     {mae:.4f}")
        logger.info(f"  RMSE (quadratic penalty): {rmse:.4f}")
        logger.info(f"  MSE (squared units):      {mse:.4f}")

        logger.info("\nINTERPRETATION:")
        logger.info(f"  MAE tells us: Predictions off by {mae:.2f} on average")
        logger.info(f"  RMSE tells us: Penalizes outliers harder (value: {rmse:.2f})")
        logger.info(f"  MSE is in squared units (not directly interpretable)")

        logger.info("\nOUTLIER SENSITIVITY:")
        logger.info(f"  Number of outliers (top 5%):  {n_outliers}")
        logger.info(f"  Outlier percentage:            {result['outlier_pct']:.1f}%")
        logger.info(f"  Max error:                     {result['max_error']:.4f}")
        logger.info(f"  Median error:                  {result['median_error']:.4f}")

        if result['max_error'] / (mae + 1e-10) > 5:
            logger.info("\n  [INSIGHT] Large outliers detected!")
            logger.info(f"  Max error is {result['max_error']/mae:.1f}x the MAE")
            logger.info("  RMSE will be significantly higher than MAE")
        else:
            logger.info("\n  [INSIGHT] Error distribution is relatively uniform")
            logger.info("  RMSE and MAE should be similar")

        # Impact calculation
        if n_outliers > 0:
            mse_without_outliers = mean_squared_error(
                y_true[~outlier_mask], y_pred[~outlier_mask]
            )
            mae_without_outliers = mean_absolute_error(
                y_true[~outlier_mask], y_pred[~outlier_mask]
            )
            logger.info("\nIMPACT OF OUTLIERS (removing top 5%):")
            logger.info(f"  MAE without outliers:  {mae_without_outliers:.4f} "
                       f"(was {mae:.4f}, change: {(mae-mae_without_outliers):.4f})")
            logger.info(f"  MSE without outliers: {mse_without_outliers:.4f} "
                       f"(was {mse:.4f}, change: {(mse-mse_without_outliers):.4f})")
            logger.info(f"  Outliers impact MSE far more than MAE")

        logger.info("=" * 80 + "\n")

    return result


def interpret_mae_with_context(
    model_mae: float,
    baseline_mae: float,
    y_test: np.ndarray,
    target_name: str = "Target"
) -> Dict[str, Any]:
    """
    Interpret MAE with proper context: target scale, baseline, and business tolerance.

    PROPER INTERPRETATION REQUIRES 3 ANCHORS:
    1. Target scale - Is MAE large relative to typical target values?
    2. Baseline performance - Is model actually learning?
    3. Business tolerance - What error is acceptable?

    Args:
        model_mae: Model's Mean Absolute Error
        baseline_mae: Baseline (mean predictor) MAE
        y_test: Test target values
        target_name: Name of target variable

    Returns:
        Dictionary with interpretation metrics and guidance
    """
    mean_target = np.mean(y_test)
    std_target = np.std(y_test)
    range_target = np.max(y_test) - np.min(y_test)

    # Interpretation metrics
    mae_pct_of_mean = (model_mae / mean_target * 100) if mean_target > 0 else 0
    mae_pct_of_std = (model_mae / std_target * 100) if std_target > 0 else 0
    mae_pct_of_range = (model_mae / range_target * 100) if range_target > 0 else 0
    
    improvement = baseline_mae - model_mae
    improvement_pct = (improvement / baseline_mae * 100) if baseline_mae > 0 else 0

    # Quality assessment
    if mae_pct_of_mean < 2:
        quality = "EXCELLENT"
        quality_desc = "Less than 2% of mean - exceptional model"
    elif mae_pct_of_mean < 5:
        quality = "GOOD"
        quality_desc = "Less than 5% of mean - strong model"
    elif mae_pct_of_mean < 10:
        quality = "MODERATE"
        quality_desc = "Less than 10% of mean - acceptable"
    elif mae_pct_of_mean < 25:
        quality = "WEAK"
        quality_desc = "Less than 25% of mean - limited practical value"
    else:
        quality = "POOR"
        quality_desc = "More than 25% of mean - not recommendable"

    result = {
        'mean_target': mean_target,
        'std_target': std_target,
        'range_target': range_target,
        'mae_pct_of_mean': mae_pct_of_mean,
        'mae_pct_of_std': mae_pct_of_std,
        'mae_pct_of_range': mae_pct_of_range,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'quality': quality,
        'quality_desc': quality_desc
    }

    logger.info("\n" + "=" * 80)
    logger.info("INTERPRETING MAE WITH CONTEXT")
    logger.info("=" * 80)

    logger.info(f"\n1. TARGET SCALE ('{target_name}'):")
    logger.info(f"   Mean:        {mean_target:.2f}")
    logger.info(f"   Std Dev:     {std_target:.2f}")
    logger.info(f"   Range:       {np.min(y_test):.2f} to {np.max(y_test):.2f}")

    logger.info(f"\n2. MAE RELATIVE TO TARGET SCALE:")
    logger.info(f"   Model MAE:               {model_mae:.4f}")
    logger.info(f"   As % of mean:            {mae_pct_of_mean:.1f}%")
    logger.info(f"   As % of std dev:         {mae_pct_of_std:.1f}%")
    logger.info(f"   As % of range:           {mae_pct_of_range:.1f}%")

    logger.info(f"\n3. BASELINE COMPARISON:")
    logger.info(f"   Baseline MAE:            {baseline_mae:.4f}")
    logger.info(f"   Model MAE:               {model_mae:.4f}")
    logger.info(f"   Absolute improvement:    {improvement:.4f}")
    logger.info(f"   Percent improvement:     {improvement_pct:.1f}%")

    logger.info(f"\n4. MODEL QUALITY ASSESSMENT:")
    logger.info(f"   Rating:      {quality}")
    logger.info(f"   Description: {quality_desc}")

    logger.info("\n5. BUSINESS INTERPRETATION:")
    if improvement_pct > 50:
        logger.info(f"   Model is {improvement_pct:.0f}% better than baseline")
        logger.info("   Strong recommendation to use model vs. mean prediction")
    elif improvement_pct > 20:
        logger.info(f"   Model is {improvement_pct:.0f}% better than baseline")
        logger.info("   Meaningful improvement - consider using model if it meets other requirements")
    else:
        logger.info(f"   Model is {improvement_pct:.0f}% better than baseline")
        logger.info("   Marginal improvement - evaluate if performance gain justifies complexity")

    logger.info("=" * 80 + "\n")

    return result


def plot_mae_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: np.ndarray,
    model_mae: float,
    baseline_mae: float,
    save_path: Optional[str] = None
) -> None:
    """
    Create diagnostic plots comparing model predictions with baseline.

    Plots:
    1. MAE distribution by sample (sorted)
    2. Prediction accuracy (predicted vs actual)
    3. Residuals distribution
    4. MAE vs baseline bar chart
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Error magnitude by sample (sorted)
    model_errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    baseline_errors = np.abs(np.asarray(y_true) - np.asarray(y_baseline))
    sorted_idx = np.argsort(model_errors)

    axes[0, 0].plot(np.arange(len(model_errors)), 
                    model_errors[sorted_idx], 'b-', label='Model', linewidth=2)
    axes[0, 0].plot(np.arange(len(baseline_errors)), 
                    baseline_errors[sorted_idx], 'r--', label='Baseline', linewidth=2)
    axes[0, 0].axhline(y=model_mae, color='b', linestyle=':', 
                       label=f'Model MAE: {model_mae:.2f}', linewidth=2)
    axes[0, 0].axhline(y=baseline_mae, color='r', linestyle=':', 
                       label=f'Baseline MAE: {baseline_mae:.2f}', linewidth=2)
    axes[0, 0].set_xlabel('Sample (sorted by model error)')
    axes[0, 0].set_ylabel('Absolute Error')
    axes[0, 0].set_title('ERROR MAGNITUDE BY SAMPLE (Sorted)')
    axes[0, 0].legend(loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Predictions vs Actual
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    y_baseline_np = np.asarray(y_baseline)
    
    axes[0, 1].scatter(y_true_np, y_pred_np, alpha=0.6, s=50, label='Model')
    axes[0, 1].scatter(y_true_np, y_baseline_np, alpha=0.3, s=30, 
                       marker='^', label='Baseline')
    axes[0, 1].plot([y_true_np.min(), y_true_np.max()], 
                    [y_true_np.min(), y_true_np.max()], 'k--', lw=2, label='Perfect')
    axes[0, 1].set_xlabel('Actual Value')
    axes[0, 1].set_ylabel('Predicted Value')
    axes[0, 1].set_title('PREDICTIONS vs ACTUAL')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Residuals distribution
    residuals = y_pred_np - y_true_np
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero')
    axes[1, 0].axvline(x=np.mean(residuals), color='g', linestyle='-', 
                       linewidth=2, label=f'Mean: {np.mean(residuals):.2f}')
    axes[1, 0].set_xlabel('Residuals (Prediction - Actual)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('RESIDUALS DISTRIBUTION')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: MAE comparison bar chart
    metrics = ['Model', 'Baseline']
    mae_values = [model_mae, baseline_mae]
    colors = ['green' if v < baseline_mae else 'red' for v in mae_values]
    
    bars = axes[1, 1].bar(metrics, mae_values, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('Mean Absolute Error')
    axes[1, 1].set_title('MAE COMPARISON')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, mae_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"MAE comparison plots saved to {save_path}")
    
    plt.close()


def explain_mae_mistakes(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Check for common MAE interpretation mistakes and provide guidance.
    """
    mae = mean_absolute_error(y_test, y_pred)
    residuals = y_pred - y_test
    
    logger.info("\n" + "=" * 80)
    logger.info("COMMON MAE MISTAKES TO AVOID")
    logger.info("=" * 80)

    # Mistake 1: Reporting without baseline
    logger.info("\n1. REPORTING MAE WITHOUT BASELINE:")
    logger.info("   ✗ WRONG: 'Our model achieves MAE of 5.2'")
    logger.info("   ✓ RIGHT: 'Our model achieves MAE of 5.2, vs baseline of 12.4'")
    logger.info("            'This represents a 58% improvement'")

    # Mistake 2: Not considering target scale
    logger.info("\n2. NOT INTERPRETING RELATIVE TO TARGET SCALE:")
    mean_target = np.mean(y_test)
    mae_pct = (mae / mean_target * 100) if mean_target > 0 else 0
    logger.info(f"   ✗ WRONG: 'Model has MAE of {mae:.2f}'")
    logger.info(f"   ✓ RIGHT: 'Model has MAE of {mae:.2f}, only {mae_pct:.1f}% of mean target'")

    # Mistake 3: Ignoring residual bias
    residual_mean = np.mean(residuals)
    logger.info("\n3. IGNORING DIRECTIONAL BIAS IN RESIDUALS:")
    logger.info(f"   Residual mean: {residual_mean:.4f}")
    if abs(residual_mean) > 0.1 * mae:
        logger.info("   [ISSUE] Residuals are biased (not centered at zero)")
        logger.info("   Model is systematically over-predicting or under-predicting")
        logger.info("   Always plot residuals vs fitted values to detect this!")
    else:
        logger.info("   [OK] Residuals appear unbiased")

    # Mistake 4: Comparing different metrics
    logger.info("\n4. COMPARING DIFFERENT METRICS BETWEEN MODELS:")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    logger.info("   ✗ WRONG: 'Model A has MAE of 5, Model B has RMSE of 6'")
    logger.info("            'Therefore Model A is better'")
    logger.info(f"   ✓ RIGHT: Compare on same metric (e.g., MAE={mae:.2f} vs MAE={mae:.2f})")

    # Mistake 5: Forgetting to back-transform
    logger.info("\n5. FORGETTING TO BACK-TRANSFORM (if target was scaled):")
    logger.info("   If target was log-transformed or scaled during preprocessing,")
    logger.info("   compute MAE on ORIGINAL scale for interpretability")
    logger.info("   Example: mae_original = back_transform(mae_scaled)")

    logger.info("=" * 80 + "\n")
