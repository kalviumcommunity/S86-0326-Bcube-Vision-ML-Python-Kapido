"""
═══════════════════════════════════════════════════════════════════════════════
ACCURACY EVALUATION - PROJECT DATA INTEGRATION
═══════════════════════════════════════════════════════════════════════════════

Evaluates a classification model on real project data with accuracy analysis.

Workflow:
  1. Load data from CSV
  2. Train Logistic Regression model
  3. Evaluate accuracy with baseline comparison
  4. Analyze confusion matrix
  5. Cross-validate stability
  6. Generate comprehensive report
  7. Save results to JSON

═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

from src.evaluate_accuracy import AccuracyEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(data_path: str, target_column: str = "ride_completed"):
    """
    Load data from CSV and prepare for classification.
    
    Args:
        data_path: Path to CSV file
        target_column: Name of binary target column
    
    Returns:
        Tuple of (X_features, y_target, feature_names)
    """
    logger.info(f"Loading data from {data_path}...")
    
    df = pd.read_csv(data_path)
    logger.info(f"✓ Loaded {len(df)} samples")
    
    # Separate features and target
    y = df[target_column].values
    X = df.drop(columns=[target_column])
    
    # Encode categorical columns
    X = X.copy()
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        unique_vals = X[col].unique()
        encoding = {val: idx for idx, val in enumerate(unique_vals)}
        X[col] = X[col].map(encoding)
        logger.info(f"  Encoded {col}: {encoding}")
    
    X = X.values.astype(float)
    feature_names = df.drop(columns=[target_column]).columns.tolist()
    
    logger.info(f"Features: {len(feature_names)} ({feature_names})")
    logger.info(f"Target: {target_column} (binary: {np.unique(y).tolist()})")
    
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Class distribution: {dict(zip(unique, counts))}")
    
    return X, y, feature_names


def train_and_evaluate_accuracy(
    data_path: str,
    target_column: str = "ride_completed",
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Complete accuracy evaluation workflow on project data.
    
    Workflow:
      1. Load and prepare data
      2. Stratified train/test split
      3. Create model with preprocessing pipeline
      4. Train model and baseline
      5. Evaluate accuracy
      6. Compare with baseline
      7. Cross-validate
      8. Generate confusion matrix breakdown
      9. Generate classification report
      10. Create results dictionary
    
    Args:
        data_path: Path to CSV file
        target_column: Target column name
        test_size: Fraction for test set
        random_state: Random seed
    
    Returns:
        Dictionary with complete evaluation results
    """
    
    print("\n" + "="*80)
    print("ACCURACY EVALUATION - PROJECT DATA INTEGRATION")
    print("="*80)
    
    # Step 1: Load and prepare
    logger.info("\nSTEP 1: Loading and preparing data...")
    X, y, feature_names = load_and_prepare_data(data_path, target_column)
    
    # Step 2: Stratified split
    logger.info("\nSTEP 2: Stratified train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"  Train: {len(X_train)} samples")
    logger.info(f"  Test:  {len(X_test)} samples")
    
    # Step 3: Create pipeline
    logger.info("\nSTEP 3: Building preprocessing + model pipeline...")
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=random_state))
    ])
    
    # Baseline
    baseline_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', DummyClassifier(strategy='most_frequent'))
    ])
    
    # Step 4: Train
    logger.info("\nSTEP 4: Training models...")
    logger.info("  Training model...")
    model_pipeline.fit(X_train, y_train)
    
    logger.info("  Training baseline...")
    baseline_pipeline.fit(X_train, y_train)
    
    # Step 5: Generate predictions
    logger.info("\nSTEP 5: Generating predictions...")
    y_pred_model = model_pipeline.predict(X_test)
    y_pred_baseline = baseline_pipeline.predict(X_test)
    
    # Step 6: Evaluate accuracy
    logger.info("\nSTEP 6: Evaluating accuracy...")
    evaluator = AccuracyEvaluator()
    
    model_acc = evaluator.compute_accuracy(y_test, y_pred_model)
    model_bal_acc = evaluator.compute_balanced_accuracy(y_test, y_pred_model)
    
    baseline_acc = evaluator.compute_accuracy(y_test, y_pred_baseline)
    baseline_bal_acc = evaluator.compute_balanced_accuracy(y_test, y_pred_baseline)
    
    logger.info(f"  Model Accuracy:           {model_acc:.4f}")
    logger.info(f"  Model Balanced Accuracy:  {model_bal_acc:.4f}")
    logger.info(f"  Baseline Accuracy:        {baseline_acc:.4f}")
    logger.info(f"  Baseline Balanced Accuracy: {baseline_bal_acc:.4f}")
    
    # Step 7: Compare against baseline
    logger.info("\nSTEP 7: Baseline comparison...")
    improvement = model_acc - baseline_acc
    improvement_pct = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0
    logger.info(f"  Improvement: +{improvement:.4f} ({improvement_pct:.1f}% relative)")
    
    # Step 8: Confusion matrix
    logger.info("\nSTEP 8: Analyzing confusion matrix...")
    cm_analysis = evaluator.analyze_confusion_matrix(y_test, y_pred_model)
    logger.info(f"  TP: {cm_analysis['TP']}, TN: {cm_analysis['TN']}")
    logger.info(f"  FP: {cm_analysis['FP']}, FN: {cm_analysis['FN']}")
    
    # Step 9: Cross-validation
    logger.info("\nSTEP 9: Cross-validation for stability...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(
        model_pipeline, X_train, y_train,
        cv=skf, scoring='accuracy'
    )
    logger.info(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Step 10: Generate report
    logger.info("\nSTEP 10: Generating accuracy report...")
    report_str = evaluator.evaluate_classification_report(y_test, y_pred_model)
    
    # Step 11: Imbalance check
    logger.info("\nSTEP 11: Checking for class imbalance...")
    is_imbalanced = evaluator.is_imbalanced(y)
    class_dist = evaluator.get_class_distribution(y)
    logger.info(f"  Imbalanced: {is_imbalanced}")
    logger.info(f"  Class distribution: {class_dist}")
    
    # Compile results
    results = {
        "dataset": {
            "total_samples": len(X),
            "test_samples": len(X_test),
            "total_features": X.shape[1],
            "feature_names": feature_names,
            "class_distribution": {str(k): int(v) for k, v in dict(zip(*np.unique(y, return_counts=True))).items()},
            "is_imbalanced": bool(is_imbalanced),
            "class_distribution_fraction": {str(k): float(v) for k, v in class_dist.items()},
        },
        "model_accuracy": {
            "standard_accuracy": float(model_acc),
            "balanced_accuracy": float(model_bal_acc),
        },
        "baseline_accuracy": {
            "standard_accuracy": float(baseline_acc),
            "balanced_accuracy": float(baseline_bal_acc),
        },
        "improvement": {
            "absolute": float(improvement),
            "relative_percent": float(improvement_pct),
        },
        "confusion_matrix": {
            "TP": cm_analysis['TP'],
            "TN": cm_analysis['TN'],
            "FP": cm_analysis['FP'],
            "FN": cm_analysis['FN'],
            "sensitivity_recall": float(cm_analysis['sensitivity_recall_tpr']),
            "specificity": float(cm_analysis['specificity_tnr']),
            "false_positive_rate": float(cm_analysis['false_positive_rate']),
            "false_negative_rate": float(cm_analysis['false_negative_rate']),
            "precision": float(cm_analysis['precision']),
        },
        "cross_validation": {
            "fold_scores": [float(s) for s in cv_scores],
            "mean": float(cv_scores.mean()),
            "std": float(cv_scores.std()),
            "min": float(cv_scores.min()),
            "max": float(cv_scores.max()),
        },
        "warnings": []
    }
    
    # Add warnings
    if is_imbalanced:
        results["warnings"].append(
            "Dataset is imbalanced. Standard accuracy may be misleading. "
            "Use Balanced Accuracy, F1, or ROC-AUC instead."
        )
    
    if improvement < 0.01:
        results["warnings"].append(
            "Model improvement over baseline is minimal (< 1%). "
            "Check if model is learning meaningful patterns."
        )
    
    if cv_scores.std() > 0.1:
        results["warnings"].append(
            "Cross-validation std dev is high (> 0.1). "
            "Model performance is unstable across folds."
        )
    
    return results


def main():
    """Execute complete accuracy evaluation workflow."""
    
    data_path = Path("data/raw/ride_data.csv")
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return
    
    # Run evaluation
    results = train_and_evaluate_accuracy(
        str(data_path),
        target_column="ride_completed",
        test_size=0.2,
        random_state=42
    )
    
    # Print summary
    print("\n" + "="*80)
    print("ACCURACY EVALUATION RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nDATASET:")
    print(f"  Total samples:     {results['dataset']['total_samples']}")
    print(f"  Test samples:      {results['dataset']['test_samples']}")
    print(f"  Features:          {results['dataset']['total_features']}")
    print(f"  Imbalanced:        {results['dataset']['is_imbalanced']}")
    
    print(f"\nMODEL PERFORMANCE:")
    print(f"  Accuracy:          {results['model_accuracy']['standard_accuracy']:.4f}")
    print(f"  Balanced Accuracy: {results['model_accuracy']['balanced_accuracy']:.4f}")
    
    print(f"\nBASELINE COMPARISON:")
    print(f"  Baseline Accuracy: {results['baseline_accuracy']['standard_accuracy']:.4f}")
    print(f"  Improvement:       +{results['improvement']['absolute']:.4f} ({results['improvement']['relative_percent']:.1f}%)")
    
    print(f"\nCONFUSION MATRIX:")
    cm = results['confusion_matrix']
    print(f"  TP: {cm['TP']:<4} TN: {cm['TN']}")
    print(f"  FP: {cm['FP']:<4} FN: {cm['FN']}")
    
    print(f"\nCROSS-VALIDATION (5-fold):")
    print(f"  Mean:  {results['cross_validation']['mean']:.4f}")
    print(f"  Std:   {results['cross_validation']['std']:.4f}")
    
    if results['warnings']:
        print(f"\n⚠️ WARNINGS:")
        for warning in results['warnings']:
            print(f"  • {warning}")
    
    # Save results
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "accuracy_evaluation_results.json"
    
    logger.info(f"\nSaving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"✓ Results saved")
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
