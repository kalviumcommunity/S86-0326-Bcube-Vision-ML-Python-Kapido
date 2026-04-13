# Accuracy Evaluation - Complete Guide

## Table of Contents

1. [What is Accuracy?](#what-is-accuracy)
2. [When Accuracy Works Well](#when-accuracy-works-well)
3. [When Accuracy Fails](#when-accuracy-fails)
4. [Accuracy and Confusion Matrix](#accuracy-and-confusion-matrix)
5. [Computing Accuracy in Scikit-Learn](#computing-accuracy-in-scikit-learn)
6. [Baseline Comparison](#baseline-comparison)
7. [Balanced Accuracy for Imbalanced Data](#balanced-accuracy-for-imbalanced-data)
8. [Cross-Validation with Accuracy](#cross-validation-with-accuracy)
9. [Common Mistakes](#common-mistakes)

---

## What is Accuracy?

**Accuracy** is the fraction of predictions that are correct:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Where:
- **TP (True Positives)**: Model predicted class 1, actual class was 1 ✓
- **TN (True Negatives)**: Model predicted class 0, actual class was 0 ✓
- **FP (False Positives)**: Model predicted class 1, actual class was 0 ✗
- **FN (False Negatives)**: Model predicted class 0, actual class was 1 ✗

### Simple Example

100 test samples, roughly equal class distribution:

| Component | Count |
|-----------|-------|
| True Positives (TP) | 40 |
| True Negatives (TN) | 45 |
| False Positives (FP) | 10 |
| False Negatives (FN) | 5 |

**Accuracy = (40 + 45) / 100 = 85%**

With balanced classes, this 85% is meaningful — the model correctly identifies the majority of both positives and negatives.

---

## When Accuracy Works Well

Accuracy is an **appropriate primary metric** when:

✓ Classes are roughly balanced — no single class dominates
✓ All errors carry similar costs — false positive ≈ false negative cost
✓ Overall correctness is the goal
✓ The problem is not safety-critical

### Good Use Cases

- Handwritten digit recognition (10 roughly balanced classes)
- General sentiment classification (balanced positive/negative)
- Multi-class product categorization (even distribution)
- Image classification benchmarks (MNIST, CIFAR-10)

---

## When Accuracy Fails

Accuracy becomes **misleading — sometimes dangerously** — when:

✗ The dataset is imbalanced — one class significantly outnumbers others
✗ The minority class is more important — the rare outcome usually matters most
✗ Error costs are asymmetric — one error type is far worse than the other

### Critical Example: Imbalanced Data (95% vs 5%)

Suppose we have 100 test samples:
- 95 genuine customers (no churn)
- 5 customers who churn

A model that predicts **"No Churn" for every customer** achieves:

| Metric | Value | Problem |
|--------|-------|---------|
| Accuracy | 95% | ✓ Looks excellent! |
| Recall (churn) | 0% | ✗ Misses every churner |
| Precision (churn) | N/A | ✗ Never predicted churn |
| F1-score (churn) | 0 | ✗ Completely useless |

**95% accuracy while catching zero churners. The number flatters the model because the majority class does the work.**

### High-Stakes Examples

**Fraud Detection** (fraudulent < 1% of transactions):
- A model saying "not fraud" for everything: 99%+ accuracy, 0% fraud caught
- The model is useless but accuracy hides it

**Medical Diagnosis** (rare disease affects 2% of patients):
- A model saying "healthy" for everyone: 98% accuracy, misses every patient
- Patients don't get treatment because the metric lies

**Churn Prediction** (95% don't churn, 5% do):
- A model saying "no churn" for everyone: 95% accuracy, zero actionable signal
- Accuracy hides complete failure

**Intrusion Detection** (attacks are rare events):
- A model ignoring attacks looks accurate while systems are exposed
- Accuracy misleads about real security

---

## Accuracy and Confusion Matrix

The **confusion matrix** is a 2×2 table for binary classification:

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | TN | FP |
| **Actual 1** | FN | TP |

**Accuracy only sums the diagonal (TN + TP)** — it completely ignores the off-diagonal trade-off between FP and FN.

### Why This Matters

FP and FN have **different costs** in almost every real problem:

| Problem | FP Cost | FN Cost |
|---------|---------|---------|
| **Fraud Detection** | Inconvenience customer | Lose money to fraud |
| **Medical Diagnosis** | Unnecessary anxiety/expense | Patient doesn't get treatment |
| **Spam Detection** | Legitimate email hidden | User sees spam |
| **Intrusion Detection** | False alarm | System compromised |

**Accuracy treats them identically.** The confusion matrix reveals them separately. Always inspect the full confusion matrix — the diagonal alone is never enough.

---

## Computing Accuracy in Scikit-Learn

### Standard Accuracy

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```

### Complete Picture: Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

Output shows:
```
              precision    recall  f1-score   support

           0       0.92      0.85      0.88       100
           1       0.78      0.88      0.83        50

    accuracy                           0.86       150
   macro avg       0.85      0.87      0.86       150
weighted avg       0.87      0.86      0.87       150
```

This immediately reveals if high accuracy hides poor performance on one class.

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
# [[TN, FP],
#  [FN, TP]]
```

### Visualization

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()
```

---

## Baseline Comparison

**Accuracy without a reference point is meaningless.**

The most important reference is the **majority-class baseline** — the accuracy achievable by predicting the most common class for every sample.

### Implementation

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# Train baseline (majority class predictor)
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

# Compare
baseline_acc = accuracy_score(y_test, baseline_pred)
model_acc    = accuracy_score(y_test, y_pred)
improvement  = model_acc - baseline_acc

print(f"Baseline Accuracy: {baseline_acc:.3f}")
print(f"Model Accuracy:    {model_acc:.3f}")
print(f"Improvement:       +{improvement:.3f}")
```

### Interpretation Example

| Baseline | Model | Improvement | Interpretation |
|----------|-------|-------------|-----------------|
| 50% | 52% | +2% | On balanced task: minimal improvement |
| 90% | 92% | +2% | On imbalanced task: might be huge! |

**The same improvement (2%) means different things depending on context.**

A 2% improvement from 50% → 52% (balanced) is minimal.
A 2% improvement from 90% → 92% (imbalanced) could represent much better minority-class detection — or it could mean the model learned almost nothing. **You need the confusion matrix and per-class metrics to know which.**

---

## Balanced Accuracy for Imbalanced Data

When classes are imbalanced, **Balanced Accuracy** is a more honest metric than standard accuracy.

**Balanced Accuracy** computes recall separately for each class and averages them:

$$\text{Balanced Accuracy} = \frac{\text{Recall}_{\text{class 0}} + \text{Recall}_{\text{class 1}}}{2}$$

For binary classification:

$$BA = \frac{TN/(TN+FP) + TP/(TP+FN)}{2}$$

### Key Properties

✓ Unaffected by class imbalance
✓ Random classifier always achieves BA = 0.5 (intuitive baseline)
✓ Ensures minority class contributes equally regardless of frequency

### Example: 90-10 Imbalanced Data

Suppose class A = 90% of data, class B = 10%, and model predicts only class A:

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Accuracy | 0.90 | Looks strong |
| Balanced Accuracy | 0.50 | Exposes failure — equivalent to random guessing |

A random classifier always achieves BA = 0.5 — providing a natural, intuitive baseline.

### Code

```python
from sklearn.metrics import balanced_accuracy_score

balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Accuracy:          {accuracy_score(y_test, y_pred):.3f}")
print(f"Balanced Accuracy: {balanced_acc:.3f}")
```

---

## Cross-Validation with Accuracy

A single train/test split can give a misleading accuracy — especially if the split was unlucky. Cross-validation provides a more stable estimate.

### Basic Cross-Validation

```python
from sklearn.model_selection import cross_val_score
import numpy as np

cv_scores = cross_val_score(
    model, X_train, y_train,
    cv=5,
    scoring="accuracy"
)

print(f"CV Accuracy scores: {cv_scores.round(3)}")
print(f"Mean CV Accuracy:   {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

### What to Look For

1. **High mean** (the model performs well)
2. **Low standard deviation** (the model performs consistently well)

A model with **88% ± 1%** is far more trustworthy than one with **88% ± 9%**.

### Stratified Cross-Validation (for Imbalanced Data)

On imbalanced datasets, use stratified CV to ensure each fold preserves class proportions:

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    model, X_train, y_train,
    cv=skf,
    scoring="accuracy"
)
print(f"Mean CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

### Interpretation

| Mean | Std Dev | Interpretation |
|------|---------|-----------------|
| 0.88 | 0.01 | Excellent — stable, consistent performance |
| 0.88 | 0.05 | Good — acceptable variance |
| 0.88 | 0.15 | Poor — high variance, unstable |

High variance indicates overfitting or high sensitivity to data splits.

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Reporting Accuracy Without a Baseline

**Bad**: "Our model achieved 92% accuracy"

**Better**: "Our model achieved 92% accuracy (baseline: 91%)"

An accuracy of 92% sounds impressive until you learn the baseline is 91%. **Always establish what the naive predictor achieves first.**

### ❌ Mistake 2: Using Standard Accuracy on Imbalanced Datasets

**Bad**: Reporting accuracy = 94% as primary metric on 96% vs 4% imbalanced data

**Better**: Reporting Balanced Accuracy = 72%, F1 = 0.65, ROC-AUC = 0.82

If your dataset is imbalanced, switch to **Balanced Accuracy, F1-score, or ROC-AUC** as your primary metric. Use accuracy as a secondary reference at most.

### ❌ Mistake 3: Ignoring the Confusion Matrix

**Bad**: "Accuracy is 88%"

**Better**: Showing:
```
              precision    recall  f1-score
           0       0.95      0.90      0.92
           1       0.65      0.72      0.68
```

Accuracy collapses four numbers into one — and that collapsing discards crucial information. Always display the full confusion matrix — especially on binary problems where the FP/FN trade-off is the heart of the evaluation.

### ❌ Mistake 4: Reporting Training Accuracy

**Bad**: "Training accuracy: 96%, Test accuracy: 88%"

**Better**: "Cross-validated accuracy: 87% ± 2%"

Training accuracy is almost always inflated relative to test accuracy — especially for complex models. **Only test-set or cross-validated accuracy reflects true generalization.**

### ❌ Mistake 5: Ignoring Per-Class Recall

A model can achieve high accuracy while having near-zero recall on the minority class. The classification report exposes this; accuracy alone hides it.

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

If class 1 has recall = 0.05, your model is missing 95% of positive cases — accuracy hides this disaster.

### ❌ Mistake 6: Treating All Accuracy Improvements as Equal

A +2% improvement from 50% → 52% (on a balanced task) is **very different** from +2% from 90% → 92% (on an imbalanced task where the baseline is already doing the heavy lifting).

Always look at **where the improvement is coming from** — balanced? Or concentrated in one class?

---

## Summary

### Quick Decision Tree

**Does my dataset have balanced classes (roughly 40-60% split)?**

→ **YES**: Use Accuracy as primary metric. Still use baseline and confusion matrix for context.

→ **NO** (Imbalanced): Use **Balanced Accuracy, F1, or ROC-AUC** as primary. Accuracy is supplementary only.

### Golden Rules

1. ✓ **Always report accuracy WITH a baseline**
   - Use `DummyClassifier(strategy="most_frequent")`

2. ✓ **Check class distribution FIRST**
   - If imbalanced, don't trust standard accuracy

3. ✓ **Use Balanced Accuracy on imbalanced data**
   - Better than standard accuracy for fairness across classes

4. ✓ **Inspect confusion matrix**
   - Accuracy only shows the diagonal

5. ✓ **Report per-class metrics**
   - Use `classification_report()` from scikit-learn

6. ✓ **Use cross-validation**
   - Single split can be luck (good or bad)

7. ✓ **Compare multiple metrics**
   - Accuracy + Balanced Accuracy + Precision + Recall + F1 + ROC-AUC

8. ✓ **Report test accuracy, NOT training accuracy**
   - Training accuracy is inflated

---

## References

- Scikit-learn documentation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Confusion matrix: https://en.wikipedia.org/wiki/Confusion_matrix
- Balanced Accuracy: https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification
