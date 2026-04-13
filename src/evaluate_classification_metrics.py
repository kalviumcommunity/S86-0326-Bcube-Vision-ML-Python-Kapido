"""
Comprehensive Classification Evaluation Module: Logistic Regression Best Practices

This module implements best practices for evaluating binary classification models
using accuracy, precision, recall, F1, and ROC-AUC metrics.

KEY PRINCIPLES IMPLEMENTED:
1. Never use accuracy alone on imbalanced data
2. ROC-AUC for evaluating ranking quality (probability calibration)
3. Precision/Recall/F1 for per-class performance breakdown
4. Always compare against majority-class baseline
5. Cross-validation for stability assessment
6. Confusion matrix for understanding error types
7. Comprehensive classification report with per-class metrics

METRICS PROVIDED:
- Accuracy: Overall correctness (only meaningful on balanced data)
- Precision: Of positive predictions, what % are actually positive
- Recall: Of actual positives, what % did we correctly identify
- F1: Harmonic mean of precision and recall (primary imbalanced metric)
- ROC-AUC: Ranking quality across all thresholds
- Confusion Matrix: Breakdown of TP, TN, FP, FN

INTERPRETATION GUIDE:
- ROC-AUC = 1.0: Perfect ranking of samples
- ROC-AUC = 0.7–0.9: Good model
- ROC-AUC = 0.5: No better than random guessing
- ROC-AUC < 0.5: Worse than random (check labels)
"""

import logging
from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.model_selection import cross_val_score

# Create logger for this module
logger = logging.getLogger(__name__)


class ClassificationMetricsEvaluator:
    """
    Complete binary classification evaluation with proper metrics.
    
    This class provides production-ready evaluation of classification models including:
    - Point estimates on test data
    - Cross-validation assessment for stability
    - Baseline (majority class) comparison
    - Comprehensive interpretation framework
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
        y_prob: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Compute all classification metrics on a held-out test set.
        
        Args:
            y_test: True target values (0 or 1)
            y_pred: Model's predicted labels (0 or 1)
            y_prob: Model's predicted probabilities of class 1 (0.0 to 1.0)
            model_name: Name of the model for logging
            
        Returns:
            Dictionary containing:
                accuracy: Overall correctness
                precision: Of predicted positives, what % are truly positive
                recall: Of actual positives, what % did we correctly identify
                f1: Harmonic mean of precision and recall
                roc_auc: Area under receiver operating characteristic curve
                
        Raises:
            ValueError: If inputs are invalid
        """
        # Input validation
        self._validate_inputs(y_test, y_pred)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
        logger.info(f"{model_name} Test Set Metrics:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1:        {f1:.4f}")
        logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
        
        return metrics
    
    def compare_with_baseline(
        self,
        y_test: np.ndarray,
        y_pred_model: np.ndarray,
        y_prob_model: np.ndarray,
        y_pred_baseline: np.ndarray,
        y_prob_baseline: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare model against the majority-class baseline predictor.
        
        The majority-class baseline always predicts the most frequent class.
        This comparison is MANDATORY:
        - Shows whether model learns anything beyond class imbalance
        - Contextualizes performance metrics
        - Identifies if model is worse than "always guess majority"
        
        Args:
            y_test: True test target values
            y_pred_model: Model's predicted labels
            y_prob_model: Model's predicted probabilities
            y_pred_baseline: Baseline predicted labels
            y_prob_baseline: Baseline predicted probabilities
            model_name: Name of the model for logging
            
        Returns:
            Dictionary with 'baseline' and 'model' sub-dictionaries, each containing
            accuracy, precision, recall, f1, roc_auc
        """
        # Evaluate both
        baseline_metrics = self.evaluate_on_test_set(
            y_test, y_pred_baseline, y_prob_baseline, "Baseline (Majority Class)"
        )
        model_metrics = self.evaluate_on_test_set(
            y_test, y_pred_model, y_prob_model, model_name
        )
        
        # Create comparison
        comparison = {
            'baseline': baseline_metrics,
            'model': model_metrics
        }
        
        # Log comparison with interpretation
        self._log_comparison(baseline_metrics, model_metrics, model_name)
        
        return comparison
    
    def cross_validate_f1(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation on F1 metric.
        
        F1 is the primary metric for imbalanced classification since it
        can't be gamed by predicting the majority class.
        
        Args:
            model: Fitted classification model
            X: Feature matrix
            y: Target values
            cv: Number of cross-validation folds (default: 5)
            
        Returns:
            Dictionary containing:
                scores: Array of F1 scores for each fold
                mean: Mean F1 across folds
                std: Standard deviation of F1 across folds
        """
        self._validate_inputs(X, y)
        
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        
        cv_results = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
        
        logger.info(f"Cross-Validation F1 (cv={cv}):")
        logger.info(f"  Fold scores: {cv_scores.round(3)}")
        logger.info(f"  Mean F1:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return cv_results
    
    def cross_validate_roc_auc(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation on ROC-AUC metric.
        
        ROC-AUC measures ranking quality, unaffected by class imbalance.
        
        Args:
            model: Fitted classification model
            X: Feature matrix
            y: Target values
            cv: Number of cross-validation folds (default: 5)
            
        Returns:
            Dictionary containing:
                scores: Array of ROC-AUC scores for each fold
                mean: Mean ROC-AUC across folds
                std: Standard deviation of ROC-AUC across folds
        """
        self._validate_inputs(X, y)
        
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        
        cv_results = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
        
        logger.info(f"Cross-Validation ROC-AUC (cv={cv}):")
        logger.info(f"  Fold scores: {cv_scores.round(3)}")
        logger.info(f"  Mean AUC:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return cv_results
    
    def get_confusion_matrix_breakdown(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, int]:
        """
        Compute and interpret confusion matrix components.
        
        Returns breakdown of:
        - TP (True Positive): Correctly predicted positive
        - TN (True Negative): Correctly predicted negative
        - FP (False Positive): Incorrectly predicted positive
        - FN (False Negative): Incorrectly predicted negative
        """
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        breakdown = {
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        }
        
        logger.info(f"Confusion Matrix Breakdown:")
        logger.info(f"  True Positives:  {tp}")
        logger.info(f"  True Negatives:  {tn}")
        logger.info(f"  False Positives: {fp}")
        logger.info(f"  False Negatives: {fn}")
        
        return breakdown
    
    def interpret_metrics(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        roc_auc: float,
        baseline_roc_auc: float = 0.5
    ) -> Dict[str, str]:
        """
        Provide human-readable interpretation of classification metrics.
        
        Args:
            accuracy: Overall correctness
            precision: Of positive predictions, % that are true positives
            recall: Of actual positives, % that we correctly identified
            f1: Harmonic mean of precision and recall
            roc_auc: Area under receiver operating characteristic curve
            baseline_roc_auc: Baseline ROC-AUC (default 0.5 for random)
            
        Returns:
            Dictionary with interpretation keys
        """
        interpretations = {}
        
        # ROC-AUC interpretation (most important for imbalanced data)
        if roc_auc >= 0.9:
            interpretations['roc_auc_level'] = "Excellent (90%+ AUC)"
        elif roc_auc >= 0.7:
            interpretations['roc_auc_level'] = "Good (70-90% AUC)"
        elif roc_auc >= 0.6:
            interpretations['roc_auc_level'] = "Fair (60-70% AUC)"
        elif roc_auc > 0.5:
            interpretations['roc_auc_level'] = "Poor (50-60% AUC)"
        elif roc_auc == 0.5:
            interpretations['roc_auc_level'] = "Random guessing (AUC = 0.5)"
        else:
            interpretations['roc_auc_level'] = "WORSE than random (AUC < 0.5) — RED FLAG"
        
        # F1 interpretation (primary for imbalanced)
        if f1 >= 0.8:
            interpretations['f1_level'] = "Excellent F1"
        elif f1 >= 0.7:
            interpretations['f1_level'] = "Good F1"
        elif f1 >= 0.5:
            interpretations['f1_level'] = "Fair F1"
        else:
            interpretations['f1_level'] = "Poor F1 (needs improvement)"
        
        # Precision vs Recall tradeoff
        if precision > recall:
            interpretations['tradeoff'] = (
                f"Model is conservative (precision {precision:.2f} > recall {recall:.2f}). "
                f"Few false positives, but misses some positives."
            )
        elif recall > precision:
            interpretations['tradeoff'] = (
                f"Model is sensitive (recall {recall:.2f} > precision {precision:.2f}). "
                f"Catches most positives, but has some false alarms."
            )
        else:
            interpretations['tradeoff'] = "Balanced precision and recall"
        
        # Improvement interpretation
        auc_improvement = (roc_auc - baseline_roc_auc) * 100
        interpretations['auc_improvement'] = f"ROC-AUC improvement: {auc_improvement:+.1f} percentage points"
        
        return interpretations
    
    def print_evaluation_report(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        y_pred_baseline: Optional[np.ndarray] = None,
        y_prob_baseline: Optional[np.ndarray] = None,
        model_name: str = "Model"
    ) -> None:
        """
        Print a formatted evaluation report to console/logs.
        
        Args:
            y_test: True test values
            y_pred: Model's predicted labels
            y_prob: Model's predicted probabilities
            y_pred_baseline: Optional baseline predictions
            y_prob_baseline: Optional baseline probabilities
            model_name: Name of model for report
        """
        # Validate inputs
        self._validate_inputs(y_test, y_pred)
        
        # Compute metrics
        model_metrics = self.evaluate_on_test_set(y_test, y_pred, y_prob, model_name)
        
        # Format report
        print("\n" + "="*80)
        print(f"CLASSIFICATION EVALUATION REPORT: {model_name}")
        print("="*80)
        print(f"\nTest Set Size: {len(y_test)} samples")
        print(f"Positive samples (class 1): {(y_test == 1).sum()}")
        print(f"Negative samples (class 0): {(y_test == 0).sum()}")
        
        print(f"\n{'Metric':<15} {'Value':<15} {'Interpretation':<50}")
        print("-"*80)
        
        # Accuracy (with caveat about imbalance)
        print(f"{'Accuracy':<15} {model_metrics['accuracy']:<15.4f} "
              f"{'(warning: see precision/recall below)':<50}")
        
        # Precision (positive predictive value)
        print(f"{'Precision':<15} {model_metrics['precision']:<15.4f} "
              f"{'Of predicted positives, % are truly positive':<50}")
        
        # Recall (sensitivity/true positive rate)
        print(f"{'Recall':<15} {model_metrics['recall']:<15.4f} "
              f"{'Of actual positives, % we correctly found':<50}")
        
        # F1 (primary metric for imbalanced data)
        print(f"{'F1 (★)':<15} {model_metrics['f1']:<15.4f} "
              f"{'Primary metric: harmonic mean of precision/recall':<50}")
        
        # ROC-AUC (ranking quality)
        print(f"{'ROC-AUC':<15} {model_metrics['roc_auc']:<15.4f} "
              f"{'Ranking quality of probabilities':<50}")
        
        # Baseline comparison if provided
        if y_pred_baseline is not None and y_prob_baseline is not None:
            baseline_metrics = self.evaluate_on_test_set(
                y_test, y_pred_baseline, y_prob_baseline, "Baseline"
            )
            print("\n" + "-"*80)
            print("BASELINE COMPARISON (Majority Class Predictor)")
            print("-"*80)
            print(f"{'Metric':<15} {'Baseline':<15} {'Model':<15} {'Improvement':<35}")
            print("-"*80)
            
            acc_improvement = model_metrics['accuracy'] - baseline_metrics['accuracy']
            f1_improvement = model_metrics['f1'] - baseline_metrics['f1']
            auc_improvement = model_metrics['roc_auc'] - baseline_metrics['roc_auc']
            
            print(f"{'Accuracy':<15} {baseline_metrics['accuracy']:<15.4f} "
                  f"{model_metrics['accuracy']:<15.4f} {acc_improvement:+.4f}")
            print(f"{'F1':<15} {baseline_metrics['f1']:<15.4f} "
                  f"{model_metrics['f1']:<15.4f} {f1_improvement:+.4f}")
            print(f"{'ROC-AUC':<15} {baseline_metrics['roc_auc']:<15.4f} "
                  f"{model_metrics['roc_auc']:<15.4f} {auc_improvement:+.4f}")
        
        # Classification report
        print("\n" + "-"*80)
        print("DETAILED CLASSIFICATION REPORT")
        print("-"*80)
        print(classification_report(y_test, y_pred))
        
        print("="*80 + "\n")
    
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
        logger.info(f"  Accuracy: {baseline_metrics['accuracy']:.4f} → {model_metrics['accuracy']:.4f} "
                   f"({model_metrics['accuracy'] - baseline_metrics['accuracy']:+.4f})")
        logger.info(f"  F1:       {baseline_metrics['f1']:.4f} → {model_metrics['f1']:.4f} "
                   f"({model_metrics['f1'] - baseline_metrics['f1']:+.4f})")
        logger.info(f"  ROC-AUC:  {baseline_metrics['roc_auc']:.4f} → {model_metrics['roc_auc']:.4f} "
                   f"({model_metrics['roc_auc'] - baseline_metrics['roc_auc']:+.4f})")
        
        if model_metrics['roc_auc'] < baseline_metrics['roc_auc']:
            logger.warning(f"⚠ Model's ROC-AUC is worse than baseline!")


def create_classification_summary(
    models_comparison: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Create a pandas DataFrame summarizing evaluation metrics for multiple models.
    
    Args:
        models_comparison: Dictionary where keys are model names and values are
                          metric dictionaries (accuracy, precision, recall, f1, roc_auc)
        
    Returns:
        DataFrame with models as rows and metrics as columns
    """
    data = []
    for model_name, metrics in models_comparison.items():
        data.append({
            'Model': model_name,
            'Accuracy': metrics.get('accuracy', np.nan),
            'Precision': metrics.get('precision', np.nan),
            'Recall': metrics.get('recall', np.nan),
            'F1': metrics.get('f1', np.nan),
            'ROC-AUC': metrics.get('roc_auc', np.nan)
        })
    
    df = pd.DataFrame(data)
    df = df.set_index('Model')
    return df
