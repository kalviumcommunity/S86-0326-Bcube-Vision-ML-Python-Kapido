# Accuracy Evaluation - Quick Reference

## 30-Second Essence

**Accuracy** = (TP + TN) / Total — simplest classification metric

**When it works**: Balanced datasets (40%-60% split)
**When it fails**: Imbalanced datasets — majority class dominates
**Solution**: Use **Balanced Accuracy, F1, or ROC-AUC** on imbalanced data

**Always compare to baseline**: `DummyClassifier(strategy='most_frequent')`

---

## Formula & Components

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

| Component | Meaning |
|-----------|---------|
| **TP** | True Positives — correctly predicted positive |
| **TN** | True Negatives — correctly predicted negative |
| **FP** | False Positives — wrongly predicted positive |
| **FN** | False Negatives — wrongly predicted negative |

---

## Quick Code: Computing Accuracy

### Standard Accuracy
```python
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f}")
```

### Balanced Accuracy (Better for Imbalanced)
```python
from sklearn.metrics import balanced_accuracy_score
bal_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy: {bal_acc:.3f}")
```

### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Output: [[TN, FP], [FN, TP]]
```

### Classification Report (Full Picture)
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

### Baseline Comparison
```python
from sklearn.dummy import DummyClassifier
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
baseline_acc = accuracy_score(y_test, baseline_pred)
model_acc = accuracy_score(y_test, y_pred)
print(f"Model: {model_acc:.3f}, Baseline: {baseline_acc:.3f}")
```

### Cross-Validation (Stratified for Imbalance)
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
print(f"Mean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

---

## Decision Tree: When to Use Accuracy

```
Is your dataset balanced? (40%-60% class split)
│
├─ YES → Use Accuracy as PRIMARY metric
│        (Still report baseline + confusion matrix for context)
│
└─ NO → Use Balanced Accuracy, F1, or ROC-AUC as PRIMARY
        (Accuracy is supplementary only)
```

---

## Class Distribution: Check First

| Class Split | Imbalance Level | Use Accuracy? | Better Alternative |
|------------|-----------------|---------------|--------------------|
| 50%-50% | Balanced | ✓ Yes | - |
| 60%-40% | Balanced | ✓ Yes | - |
| 70%-30% | Mildly | ⚠️ Caution | Balanced Accuracy |
| 80%-20% | Moderate | ✗ No | Balanced Accuracy + F1 |
| 90%-10% | Severe | ✗ No | Balanced Accuracy + F1 + ROC-AUC |
| 95%-5% | Extreme | ✗ No | ROC-AUC (primary) + F1 + BA |

---

## Confusion Matrix at a Glance

```
                Predicted 0    Predicted 1
Actual 0        ✓ TN           ✗ FP
Actual 1        ✗ FN           ✓ TP

Accuracy = (TP + TN) / Total  ← Only diagonal!
```

### Per-Class Rates

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Sensitivity / Recall (TPR)** | TP / (TP + FN) | "Of actual positives, % found" |
| **Specificity (TNR)** | TN / (TN + FP) | "Of actual negatives, % correct" |
| **Precision** | TP / (TP + FP) | "Of predicted positives, % correct" |
| **False Positive Rate** | FP / (TN + FP) | "Of actual negatives, % missed" |
| **False Negative Rate** | FN / (TP + FN) | "Of actual positives, % missed" |

---

## Balanced Accuracy (Better for Imbalance)

$$\text{Balanced Accuracy} = \frac{\text{Recall}_{\text{class 0}} + \text{Recall}_{\text{class 1}}}{2}$$

**Code**:
```python
from sklearn.metrics import balanced_accuracy_score
bal_acc = balanced_accuracy_score(y_test, y_pred)
```

**Key Property**: 
- Random classifier always achieves BA = 0.5 (intuitive baseline)
- Unaffected by class frequency
- Ensures minority class contributes equally

**Example (90%-10% split)**:
- Model predicts only majority: Accuracy = 90%, Balanced Accuracy = 50%
- BA = 0.5 immediately exposes the failure

---

## Cross-Validation: Stability Interpretation

```python
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(f"Mean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

| Pattern | Mean | Std Dev | Interpretation |
|---------|------|---------|-----------------|
| Excellent | 0.88 | 0.01 | Stable, consistent |
| Good | 0.88 | 0.03 | Acceptable variance |
| Fair | 0.88 | 0.07 | Noticeable variance |
| Poor | 0.88 | 0.15+ | Unstable (overfitting?) |

Look for: **high mean + low std**

---

## Common Pitfalls Checklist

| Pitfall | Bad | Good |
|---------|-----|------|
| No baseline | "92% accuracy" | "92% model vs 91% baseline" |
| Imbalanced data | "Accuracy 94%" | "Balanced Acc 71%, F1 0.65" |
| Ignore CM | "88% accuracy" | Show full confusion matrix |
| Training acc | Report train/test acc difference | Report test or CV accuracy only |
| Per-class | Ignore class 1 recall | Show classification_report() |
| All improvements equal | "2% improvement" | "2% improvement: where is it from?" |

---

## Imbalanced Data: Red Flags

🚩 Class split is 80%-20% or worse
🚩 Accuracy improvement < 1% over baseline
🚩 Model recall for minority class < 0.5
🚩 High accuracy but low F1-score
🚩 CV std dev > 0.1

**Solution**: Use Balanced Accuracy, F1, or ROC-AUC as primary metric

---

## What to Report: Template

### Minimum Report
```
Model:        Logistic Regression
Accuracy:     0.850
Baseline:     0.700
Improvement:  +0.150 (21.4% relative)
```

### Good Report (Balanced Data)
```
Model:             Logistic Regression
Accuracy:          0.850
Baseline Accuracy: 0.700
Improvement:       +0.150 (21.4% relative)

Confusion Matrix:     TP: 34  TN: 36
                      FP: 8   FN: 6

Cross-Validation:     0.847 ± 0.023
```

### Excellent Report (Imbalanced Data)
```
Model:                Logistic Regression
Dataset:              150 samples (70% class 0, 30% class 1)

Accuracy:             0.867
Balanced Accuracy:    0.823 ← Primary metric
F1-score:             0.812
Baseline Accuracy:    0.700
Improvement:          +0.167 absolute, +23.8% relative

Confusion Matrix:     TP: 22  TN: 26
                      FP: 3   FN: 5

Per-Class Performance:
  Class 0: Precision 0.90, Recall 0.90
  Class 1: Precision 0.88, Recall 0.81 ← Check minority!

Cross-Validation:     BA=0.820 ± 0.035 (stable)
```

---

## Golden Rules

1. ✓ **Always report accuracy WITH a baseline**
2. ✓ **Check class distribution FIRST**
3. ✓ **Use Balanced Accuracy on imbalanced data**
4. ✓ **Inspect confusion matrix always**
5. ✓ **Report per-class metrics**
6. ✓ **Use cross-validation for stability**
7. ✓ **Compare multiple metrics together**
8. ✓ **Report test accuracy, NOT training accuracy**

---

## Accuracy vs Other Metrics Comparison

| Metric | Balanced | Imbalanced | Per-Class | Threshold |
|--------|----------|-----------|-----------|-----------|
| **Accuracy** | ✓ Use | ✗ Skip | ✗ No | Fixed 0.5 |
| **Balanced Acc** | ✓ Yes | ✓ Primary | ✗ No | Fixed 0.5 |
| **Precision** | ✓ Yes | ✓ Yes | ✓ Yes | Fixed 0.5 |
| **Recall** | ✓ Yes | ✓ Yes | ✓ Yes | Fixed 0.5 |
| **F1** | ✓ Yes | ✓ Primary | ✓ Yes | Fixed 0.5 |
| **ROC-AUC** | ✓ Yes | ✓ Primary | ✗ No | All thresholds |

**Recommendation**:
- Balanced data: Accuracy + F1
- Imbalanced data: Balanced Accuracy + F1 + ROC-AUC

---

## Accuracy Does NOT Tell You

❌ How well each class is predicted (need per-class metrics)
❌ Whether false positives or false negatives are worse (need confusion matrix)
❌ How the model performs on data different from test set (need CV)
❌ Whether your model beat the baseline (need baseline comparison)
❌ Whether minority class is predicted well (need class-specific recall)

Always ask more questions than accuracy can answer.

---

## Quick Formulas Reference

| Name | Formula | Bounds |
|------|---------|--------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | [0, 1] |
| Balanced Accuracy | (Recall₀ + Recall₁)/2 | [0, 1] |
| Sensitivity/Recall | TP/(TP+FN) | [0, 1] |
| Specificity | TN/(TN+FP) | [0, 1] |
| Precision | TP/(TP+FP) | [0, 1] |
| False Positive Rate | FP/(TN+FP) | [0, 1] |
| F1 | 2×Precision×Recall/(Prec+Rec) | [0, 1] |

---

## Integration Example

```python
from src.evaluate_accuracy import AccuracyEvaluator

evaluator = AccuracyEvaluator()

# Check if imbalanced
if evaluator.is_imbalanced(y):
    print("⚠️ Dataset is imbalanced")
    
# Compute both standard and balanced
std_acc = evaluator.compute_accuracy(y_test, y_pred)
bal_acc = evaluator.compute_balanced_accuracy(y_test, y_pred)

# Analyze confusion matrix
cm_analysis = evaluator.analyze_confusion_matrix(y_test, y_pred)

# Compare with baseline
comparison = evaluator.compare_accuracy_with_baseline(y_test, y_pred)

# Cross-validate
cv_results = evaluator.cross_validate_accuracy(model, X_train, y_train)

# Print full report
evaluator.print_accuracy_report(y_test, y_pred, "My Model")
```

---

## When Accuracy Actually Works

| Scenario | Classes | Use Accuracy? | Why Works |
|----------|---------|---------------|-----------| 
| MNIST digit recognition | 10 classes (~10% each) | ✓ YES | Balanced by design |
| Sentiment (pos/neg) | 50%-50% split | ✓ YES | Truly balanced |
| Product categorization | 5 categories (20% each) | ✓ YES | Even distribution |
| Medical: common disease | 40% disease, 60% healthy | ✓ YES | Reasonable balance |

---

## When Accuracy Fails

| Scenario | Classes | Use Accuracy? | Why Fails |
|----------|---------|---------------|-----------| 
| Fraud detection | 99.5% legit, 0.5% fraud | ✗ NO | Baseline = 99.5%! |
| Disease (rare) | 98% healthy, 2% diseased | ✗ NO | Baseline = 98% |
| Anomaly detection | 97% normal, 3% anomaly | ✗ NO | Baseline = 97% |
| Customer churn | 95% stay, 5% churn | ✗ NO | Baseline = 95% |

---

## Performance Interpretation Table

| Accuracy | Baseline | Improvement | Assessment |
|----------|----------|------------|------------|
| 0.90 | 0.50 | +0.40 | Strong (40 pt improvement) |
| 0.90 | 0.85 | +0.05 | Weak (5 pt improvement) |
| 0.95 | 0.90 | +0.05 | Weak (only 5 pt improvement) |
| 0.92 | 0.90 | +0.02 | Minimal (2 pt improvement) |

**Key insight**: The same accuracy improvement means different things depending on baseline.

---

## References

- AccuracyEvaluator class: see `src/evaluate_accuracy.py`
- Demo examples: `src/demo_accuracy_evaluation.py`
- Full guide: `ACCURACY_EVALUATION_GUIDE.md`
- Project integration: `src/integrate_accuracy_evaluation.py`
