"""
═══════════════════════════════════════════════════════════════════════════════
ACCURACY EVALUATION - PRACTICAL EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

These 6 examples demonstrate:
1. Accuracy on balanced data (where it works)
2. Accuracy on imbalanced data (where it fails)
3. Baseline comparison (always required)
4. Confusion matrix interpretation
5. Cross-validation stability
6. Common pitfalls and how to avoid them

═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from src.evaluate_accuracy import AccuracyEvaluator

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


print("\n" + "="*80)
print("ACCURACY EVALUATION: PRACTICAL EXAMPLES")
print("="*80)

print("\nKey Principles from Lesson:")
print("  • Accuracy = (TP + TN) / Total")
print("  • Works well on BALANCED data")
print("  • FAILS on IMBALANCED data")
print("  • Always compare against majority-class baseline")
print("  • Use Balanced Accuracy on imbalanced data instead")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: Accuracy on Balanced Data (Accuracy Works Well Here)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("EXAMPLE 1: Balanced Binary Classification")
print("="*80)

# Create balanced dataset (50% each class)
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

print(f"\nDataset: 200 samples, 5 features")
print(f"Class distribution: {np.unique(y_test, return_counts=True)[1][0]} negatives, {np.unique(y_test, return_counts=True)[1][1]} positives")
print(f"Balance ratio: {np.unique(y_test, return_counts=True)[1][0] / np.unique(y_test, return_counts=True)[1][1]:.2f}")

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
evaluator = AccuracyEvaluator()
accuracy = evaluator.compute_accuracy(y_test, y_pred)
balanced_acc = evaluator.compute_balanced_accuracy(y_test, y_pred)

# Confusion matrix
cm_analysis = evaluator.analyze_confusion_matrix(y_test, y_pred)

print(f"\n✓ Balanced Data Metrics:")
print(f"  Accuracy (Standard): {accuracy:.4f}")
print(f"  Accuracy (Balanced): {balanced_acc:.4f}")
print(f"  → They're similar, which is good!")
print(f"\nConfusion Matrix:")
print(f"  TP: {cm_analysis['TP']:<4} TN: {cm_analysis['TN']}")
print(f"  FP: {cm_analysis['FP']:<4} FN: {cm_analysis['FN']}")
print(f"\n✓ KEY INSIGHT:")
print(f"  On balanced data, accuracy is meaningful")
print(f"  Standard accuracy = {accuracy:.1%} is a fair summary")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: Accuracy on Imbalanced Data (Accuracy Fails Here)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("EXAMPLE 2: Imbalanced Binary Classification (Why Accuracy Lies)")
print("="*80)

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

print(f"\nDataset: 200 samples, 5 features")
print(f"Class distribution: {np.unique(y_test, return_counts=True)[1][0]} negatives (90%), {np.unique(y_test, return_counts=True)[1][1]} positives (10%)")
print(f"Balance ratio: {np.unique(y_test, return_counts=True)[1][1] / np.unique(y_test, return_counts=True)[1][0]:.2f} (IMBALANCED!)")

# Compare: Naive baseline vs Logistic Regression
print(f"\n--- Naive Baseline (always predict class 0 = majority) ---")
y_pred_baseline = np.zeros_like(y_test)  # Predict class 0 for everything
baseline_acc = accuracy_score(y_test, y_pred_baseline)
baseline_balanced = balanced_accuracy_score(y_test, y_pred_baseline)
cm_baseline = evaluator.analyze_confusion_matrix(y_test, y_pred_baseline)

print(f"  Accuracy (Standard): {baseline_acc:.4f} ← LOOKS GOOD!")
print(f"  Accuracy (Balanced): {baseline_balanced:.4f}")
print(f"  Recall for class 1:  {cm_baseline['sensitivity_recall_tpr']:.4f} ← ZERO! (catches NO positives)")

print(f"\n--- Logistic Regression Model ---")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

model_acc = accuracy_score(y_test, y_pred)
model_balanced = balanced_accuracy_score(y_test, y_pred)
cm_model = evaluator.analyze_confusion_matrix(y_test, y_pred)

print(f"  Accuracy (Standard): {model_acc:.4f}")
print(f"  Accuracy (Balanced): {model_balanced:.4f}")
print(f"  Recall for class 1:  {cm_model['sensitivity_recall_tpr']:.4f} ← Better!")

print(f"\n✓ CRITICAL COMPARISON:")
print(f"  Baseline Accuracy:  {baseline_acc:.4f}")
print(f"  Model Accuracy:     {model_acc:.4f}")
print(f"  Improvement:        {model_acc - baseline_acc:+.4f} ← Might be ZERO!")
print(f"\n  But look at Balanced Accuracy:")
print(f"  Baseline Balanced:  {baseline_balanced:.4f}")
print(f"  Model Balanced:     {model_balanced:.4f}")
print(f"  Improvement:        {model_balanced - baseline_balanced:+.4f} ← MUCH BIGGER!")

print(f"\n✓ KEY INSIGHT:")
print(f"  On imbalanced data (this case 90-10):")
print(f"  - Standard accuracy might improve 0-5%")
print(f"  - Balanced accuracy improves 20-40%+")
print(f"  - Accuracy hides the real improvement!")
print(f"  - DO NOT trust standard accuracy")
print(f"  - Use Balanced Accuracy, F1, or ROC-AUC instead")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 3: Baseline Comparison (Mandatory Principle)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("EXAMPLE 3: Baseline Comparison (Always Required)")
print("="*80)

# Use fresh balanced data
X, y = make_classification(
    n_samples=200,
    n_features=5,
    n_informative=5,
    n_redundant=0,
    weights=[0.7, 0.3],  # Slightly imbalanced
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDataset: 200 samples (70% class 0, 30% class 1)")

# Baseline: Majority class predictor
print(f"\n--- Strategy: Always predict majority class ---")
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
y_pred_baseline = baseline.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred_baseline)
print(f"Baseline Accuracy: {baseline_acc:.4f}")

# Model: Logistic Regression
print(f"\n--- Logistic Regression Model ---")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model_acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy:    {model_acc:.4f}")

improvement = model_acc - baseline_acc
improvement_pct = improvement / baseline_acc * 100 if baseline_acc > 0 else 0

print(f"\n✓ BASELINE COMPARISON:")
print(f"  Improvement:       +{improvement:+.4f}")
print(f"  Relative gain:     +{improvement_pct:.1f}%")
print(f"\n  Interpretation:")
if improvement > 0.05:
    print(f"  → Model is learning meaningful patterns")
elif improvement > 0:
    print(f"  → Model beats baseline but by small margin (examine closely)")
else:
    print(f"  → Model does NOT beat baseline (not learning anything useful)")

print(f"\n✓ KEY INSIGHT:")
print(f"  Accuracy is ONLY meaningful with a baseline")
print(f"  Always compare against DummyClassifier(strategy='most_frequent')")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 4: Confusion Matrix Interpretation
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("EXAMPLE 4: Confusion Matrix - The Full Picture")
print("="*80)

# Use imbalanced data to show FP vs FN trade-off
X, y = make_classification(
    n_samples=200,
    n_features=5,
    n_informative=5,
    n_redundant=0,
    weights=[0.8, 0.2],  # 80-20 imbalanced
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix (80-20 imbalanced data):")
print(f"\n                        Predicted 0    Predicted 1")
print(f"Actual 0 (Negative)     TN: {tn:<4}        FP: {fp}")
print(f"Actual 1 (Positive)     FN: {fn:<4}        TP: {tp}")

print(f"\nDetailed Breakdown:")
print(f"  True Positives (TP):   {tp} - Correctly predicted positive")
print(f"  True Negatives (TN):   {tn} - Correctly predicted negative")
print(f"  False Positives (FP):  {fp} - Incorrectly predicted positive (Type I error)")
print(f"  False Negatives (FN):  {fn} - Incorrectly predicted negative (Type II error)")
print(f"  Total:                 {tn + fp + fn + tp}")

print(f"\nAccuracy Calculation:")
total = tn + fp + fn + tp
accuracy = (tn + tp) / total
print(f"  Accuracy = (TP + TN) / Total = ({tp} + {tn}) / {total} = {accuracy:.4f}")

print(f"\nPer-Class Analysis:")
print(f"  Recall (catch positive): TP / (TP + FN) = {tp} / {tp + fn} = {tp / (tp + fn):.4f}")
print(f"  Specificity (avoid FP):  TN / (TN + FP) = {tn} / {tn + fp} = {tn / (tn + fp):.4f}")

print(f"\n✓ KEY INSIGHT:")
print(f"  Accuracy = {accuracy:.1%} ← Looks at diagonal (TP + TN) only")
print(f"  But off-diagonal errors (FP and FN) matter differently!")
print(f"  Example:")
print(f"    Medical diagnosis: FN (miss disease) is far worse than FP (unnecessary follow-up)")
print(f"    Fraud detection: FN (miss fraud) is worse than FP (flag legitimate txn)")
print(f"  Always inspect the confusion matrix, not just accuracy")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 5: Cross-Validation Stability
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("EXAMPLE 5: Cross-Validation - Stability Assessment")
print("="*80)

# Use imbalanced data
X, y = make_classification(
    n_samples=200,
    n_features=5,
    n_informative=5,
    n_redundant=0,
    weights=[0.75, 0.25],  # 75-25
    random_state=42
)

print(f"\nDataset: 200 samples (75% class 0, 25% class 1)")

model = LogisticRegression(max_iter=1000, random_state=42)

# Single train/test split
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

print(f"\nCross-Validation Accuracy (5 folds):")
print(f"  Fold scores: {cv_scores.round(4)}")
print(f"  Mean:        {cv_scores.mean():.4f}")
print(f"  Std Dev:     {cv_scores.std():.4f}")

print(f"\nInterpretation:")
if cv_scores.std() < 0.02:
    print(f"  ✓ Very stable (std {cv_scores.std():.4f} < 0.02)")
    print(f"  → Model performs consistently across different data splits")
elif cv_scores.std() < 0.05:
    print(f"  ✓ Moderate stability (std {cv_scores.std():.4f} < 0.05)")
    print(f"  → Some variance, but acceptable")
else:
    print(f"  ⚠️ High variance (std {cv_scores.std():.4f} >= 0.05)")
    print(f"  → Model performance varies significantly across folds")
    print(f"  → May indicate overfitting or high data sensitivity")

print(f"\n✓ KEY INSIGHT:")
print(f"  A single train/test split can be luck (good or bad)")
print(f"  Cross-validation shows whether performance is stable")
print(f"  Look for: high mean (good), low std (consistent)")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 6: Common Pitfalls and How to Avoid Them
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("EXAMPLE 6: Common Pitfalls (And How to Avoid Them)")
print("="*80)

# Create a realistic imbalanced scenario
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    weights=[0.95, 0.05],  # Highly imbalanced: 95% vs 5%
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nScenario: Highly imbalanced classification (95% negative, 5% positive)")
print(f"Test set: {len(y_test)} samples")

# Pitfall 1: Reporting only accuracy
print(f"\n⚠️ PITFALL 1: Reporting accuracy without baseline")
print(f"   {'='*60}")

baseline_always_0 = np.zeros_like(y_test)
baseline_acc = accuracy_score(y_test, baseline_always_0)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model_acc = accuracy_score(y_test, y_pred)

print(f"   Model Accuracy: {model_acc:.1%}")
print(f"   → Sounds good! But is it?")
print(f"\n   Baseline (always predict majority): {baseline_acc:.1%}")
print(f"   → Baseline already achieves 95% without learning anything!")
print(f"\n   Model improvement: {model_acc - baseline_acc:+.1%}")
print(f"   → Improvement is tiny!")
print(f"\n   FIX: Always report baseline AND model accuracy together")


# Pitfall 2: Ignoring minority class performance
print(f"\n⚠️ PITFALL 2: Ignoring minority class performance")
print(f"   {'='*60}")

from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

print(f"   Class 0 (negative, 95%): Recall = {recall[0]:.1%}, F1 = {f1[0]:.3f}")
print(f"   Class 1 (positive, 5%):  Recall = {recall[1]:.1%}, F1 = {f1[1]:.3f}")
print(f"   → Class 1 has TERRIBLE F1 score!")
print(f"   → Accuracy hides this poor performance")
print(f"\n   FIX: Always report per-class precision/recall (classification report)")


# Pitfall 3: Using training accuracy instead of test accuracy
print(f"\n⚠️ PITFALL 3: Training accuracy vs Test accuracy")
print(f"   {'='*60}")

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"   Training Accuracy: {train_acc:.4f}")
print(f"   Test Accuracy:     {test_acc:.4f}")
print(f"   Difference:        {train_acc - test_acc:+.4f}")

if train_acc > test_acc + 0.03:
    print(f"   → Training accuracy significantly higher (OVERFITTING)")
else:
    print(f"   → Similar (good generalization)")

print(f"\n   FIX: Only report test accuracy or cross-validated accuracy")


print("\n" + "="*80)
print("END OF EXAMPLES")
print("="*80)

print("""
Golden Rules for Using Accuracy:
─────────────────────────────────

1. ✓ Always report accuracy WITH a baseline
   → Use DummyClassifier(strategy='most_frequent')

2. ✓ Check class distribution FIRST
   → If imbalanced, don't trust standard accuracy

3. ✓ Use Balanced Accuracy on imbalanced data
   → Better than standard accuracy for minority class

4. ✓ Inspect confusion matrix
   → Accuracy only shows the diagonal

5. ✓ Report per-class metrics
   → Use classification_report() from scikit-learn

6. ✓ Use cross-validation
   → Single split can be luck (good or bad)

7. ✓ Compare multiple metrics
   → Accuracy + Precision + Recall + F1 + ROC-AUC

8. ✓ Report test accuracy, NOT training accuracy
   → Training accuracy is inflated
""")
