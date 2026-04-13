"""
═══════════════════════════════════════════════════════════════════════════════
ACCURACY EVALUATION MODULE - CLASSIFICATION METRICS
═══════════════════════════════════════════════════════════════════════════════

Accuracy: The most intuitive but easily misused classification metric.

Accuracy = (TP + TN) / (TP + TN + FP + FN)

When it works: Balanced datasets with similar error costs
When it fails: Imbalanced datasets where majority class dominates

Lesson: Always compare against a baseline and inspect the confusion matrix.
        On imbalanced data, use Balanced Accuracy, F1, or ROC-AUC instead.

═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.dummy import DummyClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


class AccuracyEvaluator:
    """
    Comprehensive accuracy evaluation framework with baselines, cross-validation,
    and confusion matrix analysis.
    
    WARNING: Standard accuracy is misleading on imbalanced datasets.
    Always use Balanced Accuracy, F1, or ROC-AUC instead.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.logger = logger
    
    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute standard accuracy.
        
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        Args:
            y_true: True binary labels (0 or 1)
            y_pred: Predicted binary labels (0 or 1)
        
        Returns:
            Accuracy score (0 to 1)
        """
        return accuracy_score(y_true, y_pred)
    
    def compute_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute balanced accuracy (better for imbalanced data).
        
        Balanced Accuracy = (Recall_class_0 + Recall_class_1) / 2
        
        This ensures minority class contributes equally regardless of frequency.
        Random guessing achieves BA = 0.5 (natural intuitive baseline).
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
        
        Returns:
            Balanced accuracy score (0 to 1)
        """
        return balanced_accuracy_score(y_true, y_pred)
    
    def compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Compute confusion matrix and extract TP, TN, FP, FN.
        
        Confusion Matrix Structure:
                            Predicted 0    Predicted 1
        Actual 0        TN                 FP
        Actual 1        FN                 TP
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
        
        Returns:
            Tuple of:
            - confusion_matrix: 2x2 numpy array
            - components: Dict with TP, TN, FP, FN values
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Extract components from confusion matrix
        # cm[0, 0] = TN, cm[0, 1] = FP, cm[1, 0] = FN, cm[1, 1] = TP
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        
        components = {
            'TP': int(tp),  # True Positives
            'TN': int(tn),  # True Negatives
            'FP': int(fp),  # False Positives
            'FN': int(fn),  # False Negatives
        }
        
        return cm, components
    
    def analyze_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Analyze confusion matrix in detail with ratios and error rates.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
        
        Returns:
            Dictionary with detailed confusion matrix breakdown
        """
        cm, comp = self.compute_confusion_matrix(y_true, y_pred)
        tp, tn, fp, fn = comp['TP'], comp['TN'], comp['FP'], comp['FN']
        
        # Compute derived metrics
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall / Sensitivity
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
        fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0  # False Positive Rate
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0  # False Negative Rate
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        return {
            'confusion_matrix': cm,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'total': total,
            'accuracy': accuracy,
            'sensitivity_recall_tpr': tpr,
            'specificity_tnr': tnr,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'precision': precision,
        }
    
    def compare_accuracy_with_baseline(
        self,
        y_true: np.ndarray,
        y_pred_model: np.ndarray,
        y_pred_baseline: Optional[np.ndarray] = None,
        baseline_strategy: str = "most_frequent"
    ) -> Dict[str, float]:
        """
        Compare model accuracy against baseline predictor.
        
        CRITICAL: This comparison is non-negotiable. A model that beats the
        baseline is learning something real. A model that only matches the
        baseline is not useful (or is learning trivial patterns).
        
        Args:
            y_true: True labels
            y_pred_model: Model predictions
            y_pred_baseline: Baseline predictions (if None, computed automatically)
            baseline_strategy: Strategy for DummyClassifier if baseline not provided
        
        Returns:
            Dictionary with baseline, model, and improvement metrics
        """
        model_acc = accuracy_score(y_true, y_pred_model)
        
        # Compute baseline if not provided
        if y_pred_baseline is None:
            # DummyClassifier with "most_frequent" strategy predicts majority class
            clf = DummyClassifier(strategy=baseline_strategy)
            # Dummy fit requires 2D training data, use dummy X
            X_dummy = np.zeros((len(y_true), 1))
            clf.fit(X_dummy, y_true)
            y_pred_baseline = clf.predict(X_dummy)
        
        baseline_acc = accuracy_score(y_true, y_pred_baseline)
        improvement = model_acc - baseline_acc
        improvement_pct = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0.0
        
        self.logger.info(f"Baseline Accuracy: {baseline_acc:.4f}")
        self.logger.info(f"Model Accuracy:    {model_acc:.4f}")
        self.logger.info(f"Improvement:       +{improvement:.4f} ({improvement_pct:.1f}% relative gain)")
        
        return {
            'baseline_accuracy': baseline_acc,
            'model_accuracy': model_acc,
            'absolute_improvement': improvement,
            'relative_improvement_pct': improvement_pct,
        }
    
    def cross_validate_accuracy(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        stratified: bool = True
    ) -> Dict[str, float]:
        """
        Cross-validate accuracy with multiple folds.
        
        A high mean accuracy with high std dev indicates instability or overfitting.
        Look for: high mean (good performance), low std (consistent performance).
        
        Args:
            model: Fitted estimator with fit/predict methods
            X: Feature matrix
            y: Target labels
            cv: Number of cross-validation folds
            stratified: Use StratifiedKFold to preserve class distribution
        
        Returns:
            Dictionary with CV scores, mean, and std
        """
        if stratified and len(np.unique(y)) > 1:
            split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            split = cv
        
        scores = cross_val_score(
            model, X, y,
            cv=split,
            scoring="accuracy"
        )
        
        self.logger.info(f"Cross-Validation Accuracy (cv={cv}):")
        self.logger.info(f"  Fold scores: {scores.round(4)}")
        self.logger.info(f"  Mean Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
        
        return {
            'fold_scores': scores.tolist(),
            'mean_accuracy': float(scores.mean()),
            'std_accuracy': float(scores.std()),
            'min_accuracy': float(scores.min()),
            'max_accuracy': float(scores.max()),
        }
    
    def cross_validate_balanced_accuracy(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        stratified: bool = True
    ) -> Dict[str, float]:
        """
        Cross-validate balanced accuracy (better for imbalanced data).
        
        Args:
            model: Fitted estimator
            X: Feature matrix
            y: Target labels
            cv: Number of CV folds
            stratified: Use StratifiedKFold
        
        Returns:
            Dictionary with CV scores, mean, and std
        """
        if stratified and len(np.unique(y)) > 1:
            split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            split = cv
        
        scores = cross_val_score(
            model, X, y,
            cv=split,
            scoring="balanced_accuracy"
        )
        
        self.logger.info(f"Cross-Validation Balanced Accuracy (cv={cv}):")
        self.logger.info(f"  Fold scores: {scores.round(4)}")
        self.logger.info(f"  Mean Balanced Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
        
        return {
            'fold_scores': scores.tolist(),
            'mean_balanced_accuracy': float(scores.mean()),
            'std_balanced_accuracy': float(scores.std()),
            'min_balanced_accuracy': float(scores.min()),
            'max_balanced_accuracy': float(scores.max()),
        }
    
    def evaluate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate comprehensive scikit-learn classification report.
        
        This reveals per-class precision, recall, and F1 — exposing if accuracy
        is hiding poor performance on the minority class.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Optional class names (e.g., ['Negative', 'Positive'])
        
        Returns:
            Classification report as formatted string
        """
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def is_imbalanced(self, y: np.ndarray, threshold: float = 0.4) -> bool:
        """
        Check if dataset is imbalanced.
        
        A dataset is considered imbalanced if the minority class comprises
        less than threshold fraction of the data (default 40%).
        
        Args:
            y: Target labels
            threshold: Minority class threshold (default 0.4 = 40%)
        
        Returns:
            True if imbalanced, False if balanced
        """
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            return False
        
        min_fraction = counts.min() / len(y)
        return min_fraction < threshold
    
    def get_class_distribution(self, y: np.ndarray) -> Dict[int, float]:
        """
        Get class distribution as fractions.
        
        Args:
            y: Target labels
        
        Returns:
            Dictionary mapping class to fraction of dataset
        """
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        dist = {}
        for cls, count in zip(unique, counts):
            dist[int(cls)] = float(count / total)
        
        return dist
    
    def print_accuracy_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Classification Model",
        y_prob: Optional[np.ndarray] = None
    ) -> None:
        """
        Generate and print a comprehensive accuracy report.
        
        This includes:
        - Standard and balanced accuracy
        - Confusion matrix with components
        - Classification report
        - Class distribution
        - Imbalance warning if applicable
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of model for reporting
            y_prob: Optional predicted probabilities (for ROC-AUC)
        """
        print("\n" + "="*80)
        print(f"ACCURACY EVALUATION REPORT: {model_name}")
        print("="*80)
        
        # Basic metrics
        std_acc = self.compute_accuracy(y_true, y_pred)
        bal_acc = self.compute_balanced_accuracy(y_true, y_pred)
        
        print(f"\nAccuracy (Standard):  {std_acc:.4f}")
        print(f"Accuracy (Balanced):  {bal_acc:.4f}")
        
        # Class distribution
        dist = self.get_class_distribution(y_true)
        imbalanced = self.is_imbalanced(y_true)
        
        print(f"\nClass Distribution:")
        for cls, frac in sorted(dist.items()):
            print(f"  Class {cls}: {frac:.1%}")
        
        if imbalanced:
            print("\n⚠️  WARNING: Dataset is IMBALANCED")
            print("   → Do NOT trust standard accuracy")
            print("   → Use Balanced Accuracy, F1, or ROC-AUC instead")
        
        # Confusion Matrix
        cm_analysis = self.analyze_confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted 0    Predicted 1")
        print(f"Actual 0 (TN):   {cm_analysis['TN']:>6}            FP: {cm_analysis['FP']:<6}")
        print(f"Actual 1 (FN):   {cm_analysis['FN']:>6}            TP: {cm_analysis['TP']:<6}")
        
        print(f"\nDetailed Breakdown:")
        print(f"  True Positives (TP):   {cm_analysis['TP']}")
        print(f"  True Negatives (TN):   {cm_analysis['TN']}")
        print(f"  False Positives (FP):  {cm_analysis['FP']}")
        print(f"  False Negatives (FN):  {cm_analysis['FN']}")
        print(f"  Total:                 {cm_analysis['total']}")
        
        print(f"\nPer-Class Rates:")
        print(f"  Sensitivity/Recall (TPR): {cm_analysis['sensitivity_recall_tpr']:.4f}")
        print(f"  Specificity (TNR):        {cm_analysis['specificity_tnr']:.4f}")
        print(f"  False Positive Rate:      {cm_analysis['false_positive_rate']:.4f}")
        print(f"  False Negative Rate:      {cm_analysis['false_negative_rate']:.4f}")
        
        # Classification Report
        print(f"\nDetailed Classification Report:")
        print(self.evaluate_classification_report(y_true, y_pred))
        
        # Optional ROC-AUC
        if y_prob is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_prob)
                print(f"ROC-AUC Score: {roc_auc:.4f}")
            except Exception as e:
                self.logger.warning(f"Could not compute ROC-AUC: {e}")
        
        print("="*80 + "\n")


def create_accuracy_comparison_table(evaluations: List[Dict]) -> pd.DataFrame:
    """
    Create a summary table comparing accuracy across multiple models.
    
    Args:
        evaluations: List of dicts with 'model_name', 'accuracy', 'balanced_accuracy'
    
    Returns:
        pandas DataFrame with comparison
    """
    df = pd.DataFrame(evaluations)
    df = df.sort_values('accuracy', ascending=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def demonstrate_accuracy_on_balanced_data():
    """
    Example 1: Accuracy on balanced data (where it works well).
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    
    # Create balanced dataset
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        weights=[0.5, 0.5],  # Balanced!
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    evaluator = AccuracyEvaluator()
    accuracy = evaluator.compute_accuracy(y_test, y_pred)
    balanced_acc = evaluator.compute_balanced_accuracy(y_test, y_pred)
    
    return {
        'dataset': 'Balanced (50-50)',
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'y_test': y_test,
        'y_pred': y_pred,
    }


def demonstrate_accuracy_on_imbalanced_data():
    """
    Example 2: Accuracy on imbalanced data (where it fails).
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    
    # Create imbalanced dataset (90% class 0, 10% class 1)
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        weights=[0.9, 0.1],  # Imbalanced!
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    evaluator = AccuracyEvaluator()
    accuracy = evaluator.compute_accuracy(y_test, y_pred)
    balanced_acc = evaluator.compute_balanced_accuracy(y_test, y_pred)
    
    return {
        'dataset': 'Imbalanced (90-10)',
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'y_test': y_test,
        'y_pred': y_pred,
    }
