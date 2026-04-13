"""
Integration Example: Using MSE and R² Evaluation with Your Linear Regression Pipeline

This module shows how to integrate the RegressionMetricsEvaluator with your
existing linear regression training and prediction workflow.

WORKFLOW:
1. Train linear regression model (from train_linear_regression.py)
2. Make predictions on test set
3. Evaluate using proper MSE and R² methodology
4. Generate comprehensive report
5. Save metrics to file for tracking
"""

import logging
import json
from typing import Dict, Any
import numpy as np
from pathlib import Path

from .train_linear_regression import train_linear_regression_model
from .evaluate_regression_metrics import RegressionMetricsEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_linear_regression_comprehensive(
    data_path: str = 'data/raw/ride_data.csv',
    metrics_output_path: str = 'reports/regression_metrics.json'
) -> Dict[str, Any]:
    """
    Complete workflow: Train → Evaluate (Baseline + Cross-Val) → Report → Save Results
    
    This function demonstrates best practices for regression evaluation:
    1. Train the model with baseline from training step
    2. Evaluate on test set with all metrics
    3. Compare against mean baseline predictor
    4. Cross-validate for stability assessment
    5. Generate comprehensive report
    6. Save metrics for tracking/comparison
    
    Args:
        data_path: Path to the CSV data file
        metrics_output_path: Path to save metrics JSON
        
    Returns:
        Dictionary containing all evaluation results:
        {
            'test_metrics': {mse, rmse, mae, r2},
            'baseline_metrics': {mse, rmse, mae, r2},
            'cross_validation': {r2_mean, r2_std, rmse_mean, rmse_std},
            'improvement': {mse_pct, rmse_pct, r2_improvement}
        }
    """
    logger.info("="*70)
    logger.info("COMPREHENSIVE LINEAR REGRESSION EVALUATION")
    logger.info("="*70)
    
    try:
        # STEP 1: Train model and get baseline
        logger.info("\nSTEP 1: Training models...")
        (lr_pipeline, baseline_pipeline, feature_pipeline, 
         X_test_processed, y_test, training_data) = train_linear_regression_model(
            data_path=data_path,
            target_column='estimated_time',
            test_size=0.2,
            random_state=42
        )
        logger.info("✓ Models trained successfully")
        
        # Extract training data
        X_train_processed = training_data['X_train_processed']
        y_train = training_data['y_train']
        
        # STEP 2: Make predictions
        logger.info("\nSTEP 2: Generating predictions...")
        y_pred_lr = lr_pipeline.predict(X_test_processed)
        y_pred_baseline = baseline_pipeline.predict(X_test_processed)
        logger.info(f"✓ Generated {len(y_pred_lr)} predictions")
        
        # STEP 3: Initialize evaluator
        evaluator = RegressionMetricsEvaluator(random_state=42)
        
        # STEP 4: Test set evaluation
        logger.info("\nSTEP 3: Evaluating on test set...")
        comparison = evaluator.compare_with_baseline(
            X_train_processed, y_train,
            X_test_processed, y_test,
            y_pred_lr,
            model_name="Linear Regression"
        )
        
        test_metrics = comparison['model']
        baseline_metrics = comparison['baseline']
        logger.info("✓ Test set evaluation complete")
        
        # STEP 5: Cross-validation for stability
        logger.info("\nSTEP 4: Cross-validating for stability...")
        cv_r2 = evaluator.cross_validate_r2(
            lr_pipeline, X_train_processed, y_train, cv=5
        )
        cv_rmse = evaluator.cross_validate_rmse(
            lr_pipeline, X_train_processed, y_train, cv=5
        )
        logger.info("✓ Cross-validation complete")
        
        # STEP 6: Compute improvement metrics
        logger.info("\nSTEP 5: Computing improvement metrics...")
        mse_improvement_pct = (
            (baseline_metrics['mse'] - test_metrics['mse']) 
            / baseline_metrics['mse'] * 100
        ) if baseline_metrics['mse'] != 0 else 0
        
        rmse_improvement_pct = (
            (baseline_metrics['rmse'] - test_metrics['rmse']) 
            / baseline_metrics['rmse'] * 100
        ) if baseline_metrics['rmse'] != 0 else 0
        
        r2_improvement = test_metrics['r2'] - baseline_metrics['r2']
        
        improvement_metrics = {
            'mse_improvement_percent': round(mse_improvement_pct, 2),
            'rmse_improvement_percent': round(rmse_improvement_pct, 2),
            'r2_improvement_points': round(r2_improvement, 4),
            'model_explains_percent': round(test_metrics['r2'] * 100, 2)
        }
        logger.info("✓ Improvement metrics computed")
        
        # STEP 7: Generate formatted report
        logger.info("\nSTEP 6: Generating evaluation report...")
        evaluator.print_evaluation_report(
            y_test, y_pred_lr, y_pred_baseline, "Linear Regression"
        )
        
        # STEP 8: Interpretation
        logger.info("\nSTEP 7: Providing interpretation...")
        interpretation = evaluator.interpret_metrics(
            mse=test_metrics['mse'],
            rmse=test_metrics['rmse'],
            r2=test_metrics['r2'],
            baseline_r2=baseline_metrics['r2'],
            target_scale=np.mean(np.abs(test_metrics['mae']))
        )
        
        logger.info("\nMODEL INTERPRETATION:")
        logger.info(f"  MSE Magnitude:       {interpretation['mse_magnitude']}")
        logger.info(f"  R² Level:            {interpretation['r2_level']}")
        logger.info(f"  Baseline Improvement: {interpretation['baseline_improvement']}")
        logger.info(f"  Combined Story:      {interpretation['combined_story']}")
        
        # STEP 9: Compile results
        results = {
            'test_metrics': {
                'mse': round(test_metrics['mse'], 4),
                'rmse': round(test_metrics['rmse'], 4),
                'mae': round(test_metrics['mae'], 4),
                'r2': round(test_metrics['r2'], 4),
                'test_samples': len(y_test)
            },
            'baseline_metrics': {
                'mse': round(baseline_metrics['mse'], 4),
                'rmse': round(baseline_metrics['rmse'], 4),
                'mae': round(baseline_metrics['mae'], 4),
                'r2': round(baseline_metrics['r2'], 4)
            },
            'cross_validation': {
                'r2_scores': cv_r2['scores'].round(4).tolist(),
                'r2_mean': round(cv_r2['mean'], 4),
                'r2_std': round(cv_r2['std'], 4),
                'rmse_scores': cv_rmse['scores'].round(4).tolist(),
                'rmse_mean': round(cv_rmse['mean'], 4),
                'rmse_std': round(cv_rmse['std'], 4),
                'folds': 5
            },
            'improvement': improvement_metrics,
            'interpretation': interpretation
        }
        
        # STEP 10: Save results
        logger.info(f"\nSTEP 8: Saving metrics to {metrics_output_path}...")
        Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"✓ Metrics saved to {metrics_output_path}")
        
        logger.info("\n" + "="*70)
        logger.info("EVALUATION COMPLETE")
        logger.info("="*70)
        
        return results
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise


def main():
    """Run comprehensive evaluation."""
    results = evaluate_linear_regression_comprehensive(
        data_path='data/raw/ride_data.csv',
        metrics_output_path='reports/regression_metrics.json'
    )
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print("\nKEY FINDINGS:")
    print(f"  • Model R²: {results['test_metrics']['r2']:.4f}")
    print(f"  • Model explains {results['improvement']['model_explains_percent']:.1f}% "
          f"of target variance")
    print(f"  • RMSE: {results['test_metrics']['rmse']:.2f} minutes (target units)")
    print(f"  • MSE improvement over baseline: {results['improvement']['mse_improvement_percent']:.1f}%")
    print(f"\nSTABILITY (Cross-Validation):")
    print(f"  • Mean CV R²: {results['cross_validation']['r2_mean']:.4f} "
          f"± {results['cross_validation']['r2_std']:.4f}")
    if results['cross_validation']['r2_std'] < 0.05:
        print(f"  • Status: ✓ STABLE (low variance across folds)")
    elif results['cross_validation']['r2_std'] < 0.15:
        print(f"  • Status: ~ MODERATE (moderate variance across folds)")
    else:
        print(f"  • Status: ⚠ UNSTABLE (high variance — check for overfitting)")
    
    print("\nINTERPRETATION:")
    print(f"  {results['interpretation']['combined_story']}")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
