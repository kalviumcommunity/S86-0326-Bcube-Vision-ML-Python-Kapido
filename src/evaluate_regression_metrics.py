"""
Comprehensive Regression Evaluation Module: MSE and R² Best Practices

This module implements best practices for evaluating regression models using MSE 
and R² metrics as outlined in the regression fundamentals lesson.

KEY PRINCIPLES IMPLEMENTED:
1. MSE measures absolute squared error magnitude (sensitive to outliers)
2. R² measures relative improvement over the mean baseline
3. Neither metric tells the full story alone — always interpret both together
4. Cross-validation provides stability assessment across folds
5. Baseline comparison is mandatory — R² has no meaning without a reference point
6. RMSE (root of MSE) is more interpretable as it shares units with the target

METRICS PROVIDED:
- MSE: Mean Squared Error (squared units, sensitive to outliers)
- RMSE: Root Mean Squared Error (original units, interpretable)
- MAE: Mean Absolute Error (original units, robust to outliers)
- R²: Coefficient of Determination (proportion of variance explained)

INTERPRETATION GUIDE:
- R² = 1.0: Perfect prediction
- R² = 0.75: Model explains 75% of variance
- R² = 0.0: Model performs identically to mean baseline
- R² < 0.0: Model is worse than always predicting the mean (red flag)

WHEN TO USE WHICH:
- MSE internally: For optimization and model comparison
- RMSE for stakeholders: It has interpretable units and outlier sensitivity
- R² with baseline: Never report R² without baseline comparison
- Cross-validation: For stability assessment and detecting overfitting
"""

import logging
from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyRegressor

# Create logger for this module
logger = logging.getLogger(__name__)


class RegressionMetricsEvaluator:
    """
    Complete regression evaluation with MSE, RMSE, R², MAE, and baseline comparison.
    
    This class provides production-ready evaluation of regression models including:
    - Point estimates on test data
    - Cross-validation assessment for stability
    - Baseline (mean predictor) comparison
    - Comprehensive interpretation framework
    - Easy integration with model selection pipelines
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the evaluator.
        
        Args:
            random_state: Random seed for reproducible cross-validation splits
        """
        self.random_state = random_state
    
    def evaluate_on_test_set(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Compute all regression metrics on a held-out test set.
        
        Args:
            y_test: True target values (shape: (n_samples,))
            y_pred: Model's predicted values (shape: (n_samples,))
            model_name: Name of the model for logging
            
        Returns:
            Dictionary containing:
                mse: Mean Squared Error
                rmse: Root Mean Squared Error (sqrt of MSE)
                mae: Mean Absolute Error
                r2: R² Score (Coefficient of Determination)
                
        Raises:
            ValueError: If inputs are invalid
        """
        # Input validation
        self._validate_inputs(y_test, y_pred)
        
        # Compute metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"{model_name} Test Set Metrics:")
        logger.info(f"  MSE:  {mse:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE:  {mae:.4f}")
        logger.info(f"  R²:   {r2:.4f}")
        
        return metrics
    
    def compare_with_baseline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred_model: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare model against the mean baseline predictor.
        
        The mean baseline is the definition of R²=0. This comparison is MANDATORY:
        - Shows absolute improvement in MSE/RMSE
        - Contextualizes R² (no R² without baseline reference)
        - Identifies if model is worse than "always predict mean"
        
        Args:
            X_train: Training features (used to fit baseline)
            y_train: Training target values (used to fit baseline)
            X_test: Test features
            y_test: Test target values
            y_pred_model: Model's predictions on test set
            model_name: Name of the model for logging
            
        Returns:
            Dictionary with 'baseline' and 'model' sub-dictionaries, each containing
            mse, rmse, mae, r2
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Fit baseline (mean predictor)
        baseline = DummyRegressor(strategy='mean')
        baseline.fit(X_train, y_train)
        y_pred_baseline = baseline.predict(X_test)
        
        # Evaluate both
        baseline_metrics = self.evaluate_on_test_set(y_test, y_pred_baseline, "Baseline (Mean)")
        model_metrics = self.evaluate_on_test_set(y_test, y_pred_model, model_name)
        
        # Create comparison
        comparison = {
            'baseline': baseline_metrics,
            'model': model_metrics
        }
        
        # Log comparison with interpretation
        self._log_comparison(baseline_metrics, model_metrics, model_name)
        
        return comparison
    
    def cross_validate_r2(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform 5-fold cross-validation on R² metric.
        
        Cross-validation assesses stability: Does the model generalize consistently
        across different data subsets, or is performance highly variable?
        
        Output interpretation:
        - Low std dev: Stable, reliable model
        - High std dev: Model is sensitive to training data — possible overfitting
        - Negative R² in any fold: Model fails entirely on that subset (red flag)
        
        Args:
            model: Fitted regression model (must have fit, predict methods)
            X: Feature matrix
            y: Target values
            cv: Number of cross-validation folds (default: 5)
            
        Returns:
            Dictionary containing:
                scores: Array of R² scores for each fold
                mean: Mean R² across folds
                std: Standard deviation of R² across folds
        """
        self._validate_inputs(X, y)
        
        # sklearn's cross_val_score returns R² scores (higher is better)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        cv_results = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
        
        logger.info(f"Cross-Validation R² (cv={cv}):")
        logger.info(f"  Fold scores: {cv_scores.round(3)}")
        logger.info(f"  Mean R²:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Warning if any fold has negative R²
        if np.any(cv_scores < 0):
            logger.warning(f"  ⚠ Negative R² in {np.sum(cv_scores < 0)} fold(s) — "
                         f"model fails on some data subsets")
        
        return cv_results
    
    def cross_validate_rmse(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform 5-fold cross-validation on RMSE metric.
        
        RMSE cross-validation shows how much the error magnitude varies across folds.
        Unlike R², lower is better for RMSE.
        
        Args:
            model: Fitted regression model
            X: Feature matrix
            y: Target values
            cv: Number of cross-validation folds (default: 5)
            
        Returns:
            Dictionary containing:
                scores: Array of RMSE scores for each fold
                mean: Mean RMSE across folds
                std: Standard deviation of RMSE across folds
        """
        self._validate_inputs(X, y)
        
        # sklearn returns MSE scores (we negate to use scoring convention)
        # Convert MSE to RMSE
        mse_scores = -cross_val_score(
            model, X, y, cv=cv, scoring='neg_mean_squared_error'
        )
        rmse_scores = np.sqrt(mse_scores)
        
        cv_results = {
            'scores': rmse_scores,
            'mean': rmse_scores.mean(),
            'std': rmse_scores.std()
        }
        
        logger.info(f"Cross-Validation RMSE (cv={cv}):")
        logger.info(f"  Fold scores: {rmse_scores.round(3)}")
        logger.info(f"  Mean RMSE:   {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
        
        return cv_results
    
    def interpret_metrics(
        self,
        mse: float,
        rmse: float,
        r2: float,
        baseline_r2: float = 0.0,
        target_scale: Optional[float] = None
    ) -> Dict[str, str]:
        """
        Provide human-readable interpretation of MSE/RMSE and R² results.
        
        This function implements the fundamental principle: MSE and R² tell
        different stories and must be interpreted together.
        
        Args:
            mse: Mean Squared Error (in squared target units)
            rmse: Root Mean Squared Error (in target units — preferred for reporting)
            r2: R² score (proportion of variance explained)
            baseline_r2: Baseline R² for comparison (default: 0.0 for mean predictor)
            target_scale: Typical magnitude of target values (for context)
            
        Returns:
            Dictionary with interpretation keys like 'mse_interpretation', 'r2_status'
        """
        interpretations = {}
        
        # MSE interpretation
        if mse < 10:
            interpretations['mse_magnitude'] = "Small (< 10)"
        elif mse < 100:
            interpretations['mse_magnitude'] = "Moderate (10-100)"
        else:
            interpretations['mse_magnitude'] = "Large (> 100)"
        
        # R² interpretation: level of explanation
        if r2 >= 0.9:
            interpretations['r2_level'] = "Excellent (90%+ variance explained)"
        elif r2 >= 0.7:
            interpretations['r2_level'] = "Good (70-90% variance explained)"
        elif r2 >= 0.5:
            interpretations['r2_level'] = "Moderate (50-70% variance explained)"
        elif r2 >= 0.0:
            interpretations['r2_level'] = "Poor (<50% variance explained)"
        else:
            interpretations['r2_level'] = "WORSE than baseline (negative R² — red flag)"
        
        # Improvement over baseline
        if r2 > baseline_r2:
            improvement = (r2 - baseline_r2) * 100
            interpretations['baseline_improvement'] = f"Improvement: {improvement:.1f} percentage points"
        else:
            interpretations['baseline_improvement'] = "No improvement vs baseline"
        
        # Combined interpretation: when MSE and R² tell different stories
        if mse < 50 and r2 < 0.3:
            interpretations['combined_story'] = (
                "Low MSE but low R²: Absolute errors are small, but target variance is low. "
                "Model isn't adding much beyond the baseline."
            )
        elif mse > 200 and r2 > 0.8:
            interpretations['combined_story'] = (
                "High MSE but high R²: Absolute errors appear large because target spans "
                "wide range, but model dramatically outperforms baseline. Check if absolute "
                "RMSE is acceptable for your use case."
            )
        elif r2 > 0.0:
            interpretations['combined_story'] = (
                f"Model explains {r2*100:.1f}% of variance with RMSE={rmse:.2f}. "
                "Both metrics indicate positive performance."
            )
        else:
            interpretations['combined_story'] = (
                "CRITICAL: Model is performing worse than simply predicting the mean. "
                "Investigate for data leakage, train/test distribution shift, or bugs."
            )
        
        return interpretations
    
    def print_evaluation_report(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_pred_baseline: Optional[np.ndarray] = None,
        model_name: str = "Model"
    ) -> None:
        """
        Print a formatted evaluation report to console/logs.
        
        Args:
            y_test: True test values
            y_pred: Model predictions
            y_pred_baseline: Optional baseline predictions
            model_name: Name of model for report
        """
        # Validate inputs
        self._validate_inputs(y_test, y_pred)
        
        # Compute metrics
        model_metrics = self.evaluate_on_test_set(y_test, y_pred, model_name)
        
        # Format report
        print("\n" + "="*70)
        print(f"REGRESSION EVALUATION REPORT: {model_name}")
        print("="*70)
        print(f"\nTest Set Size: {len(y_test)} samples")
        print(f"\n{'Metric':<15} {'Value':<15} {'Interpretation':<40}")
        print("-"*70)
        
        # MSE row
        print(f"{'MSE':<15} {model_metrics['mse']:<15.4f} {'Squared units (less interpretable)':<40}")
        
        # RMSE row (preferred)
        print(f"{'RMSE (★)':<15} {model_metrics['rmse']:<15.4f} {'Same units as target (preferred)':<40}")
        
        # MAE row
        print(f"{'MAE':<15} {model_metrics['mae']:<15.4f} {'Average absolute error':<40}")
        
        # R² row
        r2_status = "Fraction of variance explained"
        if model_metrics['r2'] >= 0.7:
            r2_status += " (GOOD)"
        elif model_metrics['r2'] >= 0.0:
            r2_status += " (needs improvement)"
        else:
            r2_status += " (RED FLAG)"
        print(f"{'R²':<15} {model_metrics['r2']:<15.4f} {r2_status:<40}")
        
        # Baseline comparison if provided
        if y_pred_baseline is not None:
            baseline_metrics = self.evaluate_on_test_set(y_test, y_pred_baseline, "Baseline")
            print("\n" + "-"*70)
            print("BASELINE COMPARISON (Mean Predictor)")
            print("-"*70)
            print(f"{'Metric':<15} {'Baseline':<15} {'Model':<15} {'Improvement':<25}")
            print("-"*70)
            
            mse_improvement = baseline_metrics['mse'] - model_metrics['mse']
            rmse_improvement = baseline_metrics['rmse'] - model_metrics['rmse']
            r2_improvement = model_metrics['r2'] - baseline_metrics['r2']
            
            print(f"{'MSE':<15} {baseline_metrics['mse']:<15.4f} {model_metrics['mse']:<15.4f} "
                  f"{mse_improvement:+.4f}")
            print(f"{'RMSE':<15} {baseline_metrics['rmse']:<15.4f} {model_metrics['rmse']:<15.4f} "
                  f"{rmse_improvement:+.4f}")
            print(f"{'R²':<15} {baseline_metrics['r2']:<15.4f} {model_metrics['r2']:<15.4f} "
                  f"{r2_improvement:+.4f}")
        
        print("\n" + "="*70 + "\n")
    
    @staticmethod
    def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
        """Validate that inputs are valid arrays."""
        if X is None or len(X) == 0:
            raise ValueError("X cannot be None or empty")
        if y is None or len(y) == 0:
            raise ValueError("y cannot be None or empty")
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: {len(X)} vs {len(y)}")
    
    @staticmethod
    def _log_comparison(
        baseline_metrics: Dict[str, float],
        model_metrics: Dict[str, float],
        model_name: str
    ) -> None:
        """Log comparison between baseline and model."""
        logger.info(f"\nBASELINE vs {model_name} COMPARISON:")
        logger.info(f"  MSE:  {baseline_metrics['mse']:.4f} → {model_metrics['mse']:.4f} "
                   f"({model_metrics['mse'] - baseline_metrics['mse']:+.4f})")
        logger.info(f"  RMSE: {baseline_metrics['rmse']:.4f} → {model_metrics['rmse']:.4f} "
                   f"({model_metrics['rmse'] - baseline_metrics['rmse']:+.4f})")
        logger.info(f"  R²:   {baseline_metrics['r2']:.4f} → {model_metrics['r2']:.4f} "
                   f"({model_metrics['r2'] - baseline_metrics['r2']:+.4f})")
        
        if model_metrics['r2'] < baseline_metrics['r2']:
            logger.warning(f"⚠ Model's R² is worse than baseline!")


def create_evaluation_summary(
    models_comparison: Dict[str, Dict[str, float]],
    target_name: str = "Target"
) -> pd.DataFrame:
    """
    Create a pandas DataFrame summarizing evaluation metrics for multiple models.
    
    Args:
        models_comparison: Dictionary where keys are model names and values are
                          metric dictionaries (mse, rmse, mae, r2)
        target_name: Name of the target variable (for column description)
        
    Returns:
        DataFrame with models as rows and metrics as columns
    """
    data = []
    for model_name, metrics in models_comparison.items():
        data.append({
            'Model': model_name,
            'MSE': metrics.get('mse', np.nan),
            'RMSE': metrics.get('rmse', np.nan),
            'MAE': metrics.get('mae', np.nan),
            'R²': metrics.get('r2', np.nan)
        })
    
    df = pd.DataFrame(data)
    df = df.set_index('Model')
    return df
