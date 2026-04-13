"""
Demonstration: Logistic Regression Classification Best Practices

This script shows how to use Logistic Regression with proper evaluation,
baseline comparison, and cross-validation.

EXAMPLES DEMONSTRATED:
1. Balanced binary classification
2. Imbalanced binary classification
3. Baseline comparison (majority class predictor)
4. Cross-validation for stability assessment
5. Complete workflow with coefficient interpretation
"""

import logging
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

from .train_logistic_regression import train_logistic_regression_model, extract_coefficient_interpretation
from .evaluate_classification_metrics import ClassificationMetricsEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

logger = logging.getLogger(__name__)


def example_1_balanced_classification():
    """
    EXAMPLE 1: Balanced Binary Classification
    
    Demonstrates training and evaluating on balanced data where
    accuracy is a meaningful metric.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Balanced Binary Classification")
    print("="*70)
    
    # Generate balanced data
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        weights=[0.5, 0.5],  # 50-50 class distribution
        random_state=42
    )
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.sum(y == 0)} negatives, {np.sum(y == 1)} positives")
    print(f"Balance ratio: {np.sum(y == 1) / np.sum(y == 0):.2f}")
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluate
    evaluator = ClassificationMetricsEvaluator()
    metrics = evaluator.evaluate_on_test_set(y_test, y_pred, y_prob, "Logistic Regression")
    
    print(f"\n✓ Balanced Data Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} (meaningful on balanced data)")
    print(f"  Precision: {metrics['precision']:.4f} (of predictions, % correct)")
    print(f"  Recall:    {metrics['recall']:.4f} (of actuals, % found)")
    print(f"  F1:        {metrics['f1']:.4f} (harmonic mean)")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f} (ranking quality)")


def example_2_imbalanced_classification():
    """
    EXAMPLE 2: Imbalanced Binary Classification
    
    Demonstrates why accuracy is misleading on imbalanced data and why
    we use F1 and ROC-AUC instead.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Imbalanced Binary Classification (Why Accuracy Lies)")
    print("="*70)
    
    # Generate imbalanced data (90/10 split)
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        weights=[0.9, 0.1],  # 90% negative, 10% positive (imbalanced)
        random_state=42
    )
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.sum(y == 0)} negatives, {np.sum(y == 1)} positives")
    print(f"Balance ratio: {np.sum(y == 1) / np.sum(y == 0):.3f} (IMBALANCED!)")
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Naive model: always predict majority class
    print(f"\n--- Naive Baseline (always predict majority class = 0) ---")
    y_pred_naive = np.zeros_like(y_test)
    y_prob_naive = np.ones_like(y_test, dtype=float) * 0.0
    
    acc_naive = accuracy_score(y_test, y_pred_naive)
    f1_naive = f1_score(y_test, y_pred_naive)
    auc_naive = 0.5  # By definition, predicting constant gives AUC=0.5
    
    print(f"  Accuracy: {acc_naive:.4f} ← VERY HIGH (but model learns nothing!)")
    print(f"  F1:       {f1_naive:.4f} ← ZERO (can't fool F1)")
    print(f"  ROC-AUC:  {auc_naive:.4f} ← 0.5 (can't fool AUC)")
    
    # Logistic Regression model
    print(f"\n--- Logistic Regression Model ---")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    acc_model = accuracy_score(y_test, y_pred)
    f1_model = f1_score(y_test, y_pred)
    auc_model = roc_auc_score(y_test, y_prob)
    
    print(f"  Accuracy: {acc_model:.4f} (might be high even if mediocre)")
    print(f"  F1:       {f1_model:.4f} (true judgment of minority class performance)")
    print(f"  ROC-AUC:  {auc_model:.4f} (best metric for imbalanced data)")
    
    # Key insight
    print(f"\n✓ KEY INSIGHT:")
    print(f"  Accuracy improved vs naive: {acc_model - acc_naive:+.4f}")
    print(f"  F1 improved vs naive:       {f1_model - f1_naive:+.4f} ← MASSIVE improvement!")
    print(f"  ROC-AUC improved vs naive:  {auc_model - auc_naive:+.4f} ← Clear advantage")
    print(f"\n  On imbalanced data:")
    print(f"  - DON'T trust accuracy")
    print(f"  - DO trust F1 and ROC-AUC")


def example_3_baseline_comparison():
    """
    EXAMPLE 3: Baseline Comparison
    
    Demonstrates comparing model against majority-class baseline using
    proper DummyClassifier.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Baseline Comparison (Majority Class Predictor)")
    print("="*70)
    print("\nPrinciple: Always compare against a baseline.")
    print("For classification, baseline = always predict majority class\n")
    
    # Generate data
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        weights=[0.7, 0.3],  # 70-30 imbalance
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Baseline: DummyClassifier with majority class strategy
    baseline_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', DummyClassifier(strategy='most_frequent'))
    ])
    baseline_pipeline.fit(X_train, y_train)
    baseline_pred = baseline_pipeline.predict(X_test)
    baseline_prob = baseline_pipeline.predict_proba(X_test)[:, 1]
    
    # Model: Logistic Regression
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    model_pipeline.fit(X_train, y_train)
    model_pred = model_pipeline.predict(X_test)
    model_prob = model_pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluate both
    evaluator = ClassificationMetricsEvaluator()
    comparison = evaluator.compare_with_baseline(
        y_test,
        model_pred, model_prob,
        baseline_pred, baseline_prob,
        "Logistic Regression"
    )
    
    # Analysis
    baseline_auc = comparison['baseline']['roc_auc']
    model_auc = comparison['model']['roc_auc']
    
    print(f"\n✓ Comparison Summary:")
    print(f"  Baseline ROC-AUC:      {baseline_auc:.4f} (by definition)")
    print(f"  Model ROC-AUC:         {model_auc:.4f}")
    print(f"  Improvement:           {model_auc - baseline_auc:+.4f}")
    
    if model_auc > baseline_auc:
        print(f"\n  → Model beats baseline by {(model_auc - baseline_auc)*100:.1f} percentage points")
    else:
        print(f"\n  → ⚠ Model is worse than baseline (red flag!)")


def example_4_cross_validation():
    """
    EXAMPLE 4: Cross-Validation for Stability
    
    Demonstrates that a single test split can be misleading.
    Cross-validation shows performance consistency across data subsets.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Cross-Validation for Stability Assessment")
    print("="*70)
    print("\nA high mean F1 with high std dev means: unstable model\n")
    
    # Generate data
    X, y = make_classification(
        n_samples=150,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        weights=[0.6, 0.4],
        random_state=42
    )
    
    # Create model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Cross-validate
    evaluator = ClassificationMetricsEvaluator()
    
    cv_f1 = evaluator.cross_validate_f1(pipeline, X, y, cv=5)
    cv_auc = evaluator.cross_validate_roc_auc(pipeline, X, y, cv=5)
    
    # Interpretation
    print(f"\n✓ Cross-Validation Results (5 folds):")
    print(f"\n  F1 Scores:")
    print(f"    Individual: {cv_f1['scores'].round(3)}")
    print(f"    Mean: {cv_f1['mean']:.4f} ± {cv_f1['std']:.4f}")
    
    print(f"\n  ROC-AUC Scores:")
    print(f"    Individual: {cv_auc['scores'].round(3)}")
    print(f"    Mean: {cv_auc['mean']:.4f} ± {cv_auc['std']:.4f}")
    
    # Stability interpretation
    if cv_auc['std'] < 0.05:
        print(f"\n  → ✓ STABLE model (low std dev: {cv_auc['std']:.3f})")
    elif cv_auc['std'] < 0.15:
        print(f"\n  → ~ MODERATE stability (std dev: {cv_auc['std']:.3f})")
    else:
        print(f"\n  → ⚠ UNSTABLE model (high std dev: {cv_auc['std']:.3f})")


def example_5_full_workflow():
    """
    EXAMPLE 5: Complete Workflow with Coefficient Interpretation
    
    Demonstrates the complete end-to-end process including
    coefficient interpretation as odds ratios.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Complete Workflow with Coefficient Interpretation")
    print("="*70)
    
    # Generate data with meaningful features
    np.random.seed(42)
    X_arr = np.random.randn(200, 3)
    # Create binary target based on features
    y_arr = (2*X_arr[:, 0] - X_arr[:, 1] + 0.5*X_arr[:, 2] + np.random.randn(200)*0.5 > 0).astype(int)
    X = pd.DataFrame(X_arr, columns=['Age', 'Income', 'Engagement'])
    y = pd.Series(y_arr, name='Conversion')
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.sum(y == 0)} negatives, {np.sum(y == 1)} positives")
    
    # Train
    print(f"\nTraining Logistic Regression...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluate
    print(f"\nEvaluation:")
    evaluator = ClassificationMetricsEvaluator()
    metrics = evaluator.evaluate_on_test_set(y_test, y_pred, y_prob, "Logistic Regression")
    
    # Coefficients
    print(f"\nCoefficient Interpretation (Odds Ratios):")
    print(f"─" * 70)
    
    model = pipeline.named_steps['model']
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0],
        'Odds Ratio': np.exp(model.coef_[0])
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print(f"Intercept: {model.intercept_[0]:.4f} (log-odds baseline)")
    print(coef_df.to_string(index=False))
    
    print(f"\nInterpretation:")
    for _, row in coef_df.iterrows():
        feature = row['Feature']
        or_val = row['Odds Ratio']
        if or_val > 1.0:
            pct = (or_val - 1.0) * 100
            print(f"  • {feature}: +{pct:.1f}% odds of conversion per unit increase")
        elif or_val < 1.0:
            pct = (1.0 - or_val) * 100
            print(f"  • {feature}: −{pct:.1f}% odds of conversion per unit increase")
    
    print(f"\n✓ Complete workflow demonstrates:")
    print(f"  ✓ Stratified train/test split")
    print(f"  ✓ Feature scaling in pipeline")
    print(f"  ✓ Model training")
    print(f"  ✓ Proper evaluation metrics")
    print(f"  ✓ Coefficient interpretation")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION: PRACTICAL EXAMPLES")
    print("="*70)
    print("\nKey Principles from Lesson:")
    print("  • Logistic Regression models P(class 1 | features)")
    print("  • Uses sigmoid to squash outputs to [0, 1]")
    print("  • Trained with log loss (not MSE)")
    print("  • Always use stratified train/test split")
    print("  • Never trust accuracy on imbalanced data")
    print("  • Use F1 and ROC-AUC instead")
    print("  • Always compare against majority-class baseline")
    print("  • Coefficients are odds ratios (exp(coef))\n")
    
    example_1_balanced_classification()
    example_2_imbalanced_classification()
    example_3_baseline_comparison()
    example_4_cross_validation()
    example_5_full_workflow()
    
    print("\n" + "="*70)
    print("END OF EXAMPLES")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. Accuracy is misleading on imbalanced data")
    print("  2. Use F1 and ROC-AUC instead")
    print("  3. Always stratify train/test splits")
    print("  4. Compare against majority-class baseline")
    print("  5. Cross-validate for stability")
    print("  6. Interpret coefficients as odds ratios\n")


if __name__ == '__main__':
    main()
