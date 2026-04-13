"""
Integration Example: Logistic Regression with Classification Evaluation

This module shows how to integrate Logistic Regression training and evaluation
with your ride-sharing classification problem: Predict ride completion (0/1).

PROBLEM: Binary classification - predict whether a ride will be completed (1) 
         or not (0)

WORKFLOW:
1. Load and preprocess data
2. Train Logistic Regression with proper stratification
3. Evaluate with proper classification metrics
4. Compare against majority-class baseline
5. Cross-validate for stability
6. Interpret coefficients as odds ratios
7. Save results for tracking
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

from .train_logistic_regression import train_logistic_regression_model, extract_coefficient_interpretation
from .data_loader import load_data
from .preprocessing import build_preprocessing_pipeline
from .evaluate_classification_metrics import ClassificationMetricsEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(
    data_path: str = 'data/raw/ride_data.csv',
    target_column: str = 'ride_completed'
) -> tuple:
    """
    Load and prepare data for classification.
    
    Args:
        data_path: Path to CSV file
        target_column: Name of binary target column
        
    Returns:
        Tuple of (X, y) where y is binary (0/1)
    """
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"✓ Loaded {len(df)} samples")
    
    # Separate features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = X.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        logger.info(f"  Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    logger.info(f"Features: {X_encoded.shape[1]} ({list(X_encoded.columns)})")
    logger.info(f"Target: {y.name} (binary: {sorted(y.unique())})")
    logger.info(f"Class distribution: {dict(y.value_counts())}")
    
    return X_encoded, y


def train_and_evaluate_logistic_regression(
    data_path: str = 'data/raw/ride_data.csv',
    target_column: str = 'ride_completed',
    test_size: float = 0.2,
    results_output_path: str = 'reports/logistic_regression_results.json'
) -> Dict[str, Any]:
    """
    Complete workflow: Load → Train → Evaluate → Report → Save
    
    Args:
        data_path: Path to CSV file
        target_column: Name of binary target column
        test_size: Proportion for test set
        results_output_path: Path to save results JSON
        
    Returns:
        Dictionary containing all evaluation results
    """
    logger.info("="*70)
    logger.info("LOGISTIC REGRESSION CLASSIFICATION WORKFLOW")
    logger.info("="*70)
    
    try:
        # STEP 1: Load and prepare data
        logger.info("\nSTEP 1: Loading and preparing data...")
        X, y = load_and_prepare_data(data_path, target_column)
        
        # STEP 2: Train model
        logger.info("\nSTEP 2: Training Logistic Regression model...")
        model_pipeline, baseline_pipeline, X_test, y_test, training_data = \
            train_logistic_regression_model(
                X, y,
                test_size=test_size,
                random_state=42,
                max_iter=1000,
                C=1.0,
                penalty="l2"
            )
        
        # Extract training data
        X_train = training_data['X_train']
        y_train = training_data['y_train']
        y_pred_model = training_data['model_pred']
        y_prob_model = training_data['model_prob']
        y_pred_baseline = training_data['baseline_pred']
        y_prob_baseline = training_data['baseline_prob']
        
        # STEP 3: Evaluate on test set
        logger.info("\nSTEP 3: Evaluating on test set...")
        evaluator = ClassificationMetricsEvaluator()
        
        # Test set evaluation
        model_metrics = evaluator.evaluate_on_test_set(
            y_test, y_pred_model, y_prob_model, "Logistic Regression"
        )
        
        baseline_metrics = evaluator.evaluate_on_test_set(
            y_test, y_pred_baseline, y_prob_baseline, "Baseline (Majority Class)"
        )
        
        # STEP 4: Compute improvements
        logger.info("\nSTEP 4: Computing improvement metrics...")
        accuracy_improvement = model_metrics['accuracy'] - baseline_metrics['accuracy']
        f1_improvement = model_metrics['f1'] - baseline_metrics['f1']
        auc_improvement = model_metrics['roc_auc'] - baseline_metrics['roc_auc']
        
        improvement_metrics = {
            'accuracy_improvement': round(accuracy_improvement, 4),
            'f1_improvement': round(f1_improvement, 4),
            'auc_improvement': round(auc_improvement, 4),
            'auc_improvement_percent': round(auc_improvement * 100, 2)
        }
        
        # STEP 5: Cross-validation
        logger.info("\nSTEP 5: Cross-validating for stability...")
        cv_f1 = evaluator.cross_validate_f1(model_pipeline, X_train, y_train, cv=5)
        cv_auc = evaluator.cross_validate_roc_auc(model_pipeline, X_train, y_train, cv=5)
        
        # STEP 6: Coefficient interpretation
        logger.info("\nSTEP 6: Interpreting coefficients as odds ratios...")
        coef_df = extract_coefficient_interpretation(
            model_pipeline,
            feature_names=np.array(X.columns)
        )
        
        # STEP 7: Generate report
        logger.info("\nSTEP 7: Generating evaluation report...")
        evaluator.print_evaluation_report(
            y_test, y_pred_model, y_prob_model,
            y_pred_baseline, y_prob_baseline,
            "Logistic Regression"
        )
        
        # STEP 8: Compile and save results
        logger.info(f"\nSTEP 8: Saving results to {results_output_path}...")
        
        results = {
            'dataset': {
                'total_samples': len(X),
                'total_features': X.shape[1],
                'feature_names': list(X.columns),
                'test_samples': len(y_test),
                'class_distribution': dict(y.value_counts().to_dict()),
                'class_distribution_test': dict(y_test.value_counts().to_dict())
            },
            'model_hyperparameters': {
                'max_iter': 1000,
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'lbfgs',
                'random_state': 42
            },
            'test_metrics': {
                'accuracy': round(model_metrics['accuracy'], 4),
                'precision': round(model_metrics['precision'], 4),
                'recall': round(model_metrics['recall'], 4),
                'f1': round(model_metrics['f1'], 4),
                'roc_auc': round(model_metrics['roc_auc'], 4)
            },
            'baseline_metrics': {
                'accuracy': round(baseline_metrics['accuracy'], 4),
                'f1': round(baseline_metrics['f1'], 4),
                'roc_auc': round(baseline_metrics['roc_auc'], 4)
            },
            'improvement': improvement_metrics,
            'cross_validation': {
                'f1': {
                    'scores': cv_f1['scores'].round(4).tolist(),
                    'mean': round(cv_f1['mean'], 4),
                    'std': round(cv_f1['std'], 4)
                },
                'roc_auc': {
                    'scores': cv_auc['scores'].round(4).tolist(),
                    'mean': round(cv_auc['mean'], 4),
                    'std': round(cv_auc['std'], 4)
                },
                'folds': 5
            },
            'coefficients': {
                'intercept': round(training_data['intercept'], 4),
                'features': {
                    row['Feature']: {
                        'coefficient': round(row['Coefficient'], 4),
                        'odds_ratio': round(row['Odds Ratio'], 4)
                    }
                    for _, row in coef_df.iterrows()
                }
            }
        }
        
        # Save results
        Path(results_output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(results_output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"✓ Results saved to {results_output_path}")
        
        logger.info("\n" + "="*70)
        logger.info("WORKFLOW COMPLETE")
        logger.info("="*70)
        
        return results
        
    except Exception as e:
        logger.error(f"Error during workflow: {str(e)}", exc_info=True)
        raise


def main():
    """Run workflow and print summary."""
    results = train_and_evaluate_logistic_regression(
        data_path='data/raw/ride_data.csv',
        target_column='ride_completed',
        test_size=0.2,
        results_output_path='reports/logistic_regression_results.json'
    )
    
    # Print summary
    print("\n" + "="*70)
    print("CLASSIFICATION RESULTS SUMMARY")
    print("="*70)
    print("\nMODEL PERFORMANCE:")
    test_metrics = results['test_metrics']
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f} ← Primary metric for imbalanced data")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f} ← Best for ranking quality")
    
    print("\nIMPROVEMENT OVER BASELINE:")
    improvement = results['improvement']
    print(f"  Accuracy improvement:  {improvement['accuracy_improvement']:+.4f}")
    print(f"  F1 improvement:        {improvement['f1_improvement']:+.4f}")
    print(f"  ROC-AUC improvement:   {improvement['auc_improvement_percent']:+.2f}%")
    
    print("\nSTABILITY (Cross-Validation):")
    cv = results['cross_validation']
    print(f"  Mean CV F1:      {cv['f1']['mean']:.4f} ± {cv['f1']['std']:.4f}")
    print(f"  Mean CV ROC-AUC: {cv['roc_auc']['mean']:.4f} ± {cv['roc_auc']['std']:.4f}")
    if cv['roc_auc']['std'] < 0.05:
        print(f"  Status: ✓ STABLE model")
    elif cv['roc_auc']['std'] < 0.15:
        print(f"  Status: ~ MODERATE stability")
    else:
        print(f"  Status: ⚠ UNSTABLE model")
    
    print("\nTOP FEATURE COEFFICIENTS (Odds Ratios):")
    coef = results['coefficients']
    # Sort by absolute coefficient value
    features_sorted = sorted(
        coef['features'].items(),
        key=lambda x: abs(x[1]['coefficient']),
        reverse=True
    )
    for feature, values in features_sorted[:5]:
        or_val = values['odds_ratio']
        if or_val > 1.0:
            pct = (or_val - 1.0) * 100
            print(f"  • {feature}: +{pct:.1f}% odds per unit")
        else:
            pct = (1.0 - or_val) * 100
            print(f"  • {feature}: −{pct:.1f}% odds per unit")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
