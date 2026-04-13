"""
═══════════════════════════════════════════════════════════════════════════════
LOGISTIC REGRESSION QUICK REFERENCE CARD
═══════════════════════════════════════════════════════════════════════════════
Keep this handy while training and evaluating classification models.

═══════════════════════════════════════════════════════════════════════════════
THE ESSENCE IN 30 SECONDS
═══════════════════════════════════════════════════════════════════════════════

Logistic Regression:
  Input: Features
  → Linear: z = w·X + b
  → Sigmoid: P = 1/(1+e^-z)
  → Output: Probability of class 1 ∈ [0,1]

Key: Always use stratified train/test split.
     Never trust accuracy on imbalanced data.
     Use F1 and ROC-AUC instead.
     Always compare to majority-class baseline.


═══════════════════════════════════════════════════════════════════════════════
THE FIVE ESSENTIAL METRICS
═══════════════════════════════════════════════════════════════════════════════

┌──────────┬────────────────┬──────────┬────────────────────┬─────────────────┐
│ Metric   │ Formula/Scale  │ When Use │ Good on Imbalanced? │ Baseline Value  │
├──────────┼────────────────┼──────────┼────────────────────┼─────────────────┤
│Accuracy  │ (TP+TN)/Total  │ When     │ ✗ NO! Misleading   │ % Majority      │
│          │ 0-1 scale      │ balanced │ on imbalanced data  │ class           │
├──────────┼────────────────┼──────────┼────────────────────┼─────────────────┤
│Precision │ TP/(TP+FP)     │ FP cost  │ ✓ Yes, if high     │ ≈ % Majority    │
│          │ 0-1 scale      │ is high  │ precision needed   │ class           │
├──────────┼────────────────┼──────────┼────────────────────┼─────────────────┤
│Recall    │ TP/(TP+FN)     │ FN cost  │ ✓ Yes, if high     │ ≈ % Majority    │
│(Sensitivity)│ 0-1 scale   │ is high  │ recall needed      │ class           │
├──────────┼────────────────┼──────────┼────────────────────┼─────────────────┤
│F1 (★)    │ 2×(P×R)/(P+R)  │ PRIMARY  │ ✓ YES! (Perfect)   │ ≈ 0 (baseline   │
│          │ 0-1 scale      │ default  │ Can't be gamed     │ can't fool F1)  │
├──────────┼────────────────┼──────────┼────────────────────┼─────────────────┤
│ROC-AUC   │ Ranking prob.  │ Ranking  │ ✓ YES! (Perfect)   │ 0.5 (random)    │
│          │ 0-1 scale      │ quality  │ Unaffected by      │                 │
│          │                │          │ imbalance          │                 │
└──────────┴────────────────┴──────────┴────────────────────┴─────────────────┘

BEST PRACTICES:
  ✓ Balanced data: Report accuracy + F1 + ROC-AUC
  ✓ Imbalanced data: Report F1 + ROC-AUC (ignore accuracy)
  ✓ ALWAYS: Compare against baseline


═══════════════════════════════════════════════════════════════════════════════
QUICK CODE TEMPLATE
═══════════════════════════════════════════════════════════════════════════════

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, classification_report

# 1. Stratified split (ALWAYS with stratify=y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Pipeline = scale + model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])
pipeline.fit(X_train, y_train)

# 3. Predictions (both labels AND probabilities)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# 4. Evaluation
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
print(f"F1: {f1:.3f}, ROC-AUC: {auc:.3f}")
print(classification_report(y_test, y_pred))

# 5. Cross-validation
cv_auc = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
print(f"CV ROC-AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")


═══════════════════════════════════════════════════════════════════════════════
F1 vs ROC-AUC: WHICH METRIC TO OPTIMIZE?
═══════════════════════════════════════════════════════════════════════════════

F1 SCORE: Harmonic mean of precision and recall
───────────────────────────────────────────────
✓ Respects both false positives AND false negatives
✓ Good when you care about per-class performance
✓ Can't be gamed by predicting majority class
✓ PRIMARY metric for imbalanced classification

Use when:
  - You need a single number for model comparison
  - False positives and false negatives both matter
  - Class imbalance is severe


ROC-AUC: Area under receiver operating characteristic
──────────────────────────────────────────────────
✓ Measures ranking quality across ALL possible thresholds
✓ Threshold-independent (not at 0.5)
✓ Good for probability calibration
✓ Unaffected by class imbalance

Use when:
  - You can adjust prediction threshold
  - Ranking quality matters more than calibration
  - You want to explore precision-recall tradeoff


BOTTOM LINE: Use BOTH F1 and ROC-AUC to get complete picture
──────────────────────────────────────────────────────────────


═══════════════════════════════════════════════════════════════════════════════
SIGMOID FUNCTION BEHAVIOR
═══════════════════════════════════════════════════════════════════════════════

Input (z)     Sigmoid(z)    Log-Odds      Meaning
────────────────────────────────────────────────────────────────────
z = −5        0.007         −5            Very confident: class 0
z = −2        0.119         −2            Confident: class 0
z = −1        0.269         −1            Lean: class 0
z = 0         0.500         0             Maximum uncertainty (50-50)
z = +1        0.731         +1            Lean: class 1
z = +2        0.881         +2            Confident: class 1
z = +5        0.993         +5            Very confident: class 1

Key insight: As |z| grows, probability saturates near 0 or 1 → high confidence


═══════════════════════════════════════════════════════════════════════════════
IMBALANCED DATA: RED FLAGS AND SOLUTIONS
═══════════════════════════════════════════════════════════════════════════════

RED FLAG: 90% Class 0, 10% Class 1
──────────────────────────────────

Naive Baseline (always predict 0):
  Accuracy: 90% (looks great but learns nothing!)
  F1: 0.0 (can't fool F1)
  ROC-AUC: 0.5 (random)

Solution 1: Use class_weight="balanced"
  from sklearn.linear_model import LogisticRegression
  model = LogisticRegression(class_weight="balanced")

Solution 2: Adjust decision threshold
  y_pred_adjusted = (y_prob > 0.3)  # Lower threshold for minority class

Solution 3: Resampling (advanced)
  - Oversample minority class
  - Undersample majority class
  - Use SMOTE (synthetic oversampling)

Solution 4: Use F1 and ROC-AUC (not accuracy)
  - F1 always reflects true minority class performance
  - ROC-AUC unaffected by class distribution


═══════════════════════════════════════════════════════════════════════════════
COEFFICIENT INTERPRETATION (ODDS RATIOS)
═══════════════════════════════════════════════════════════════════════════════

Raw Coefficients (log-odds space):
──────────────────────────────────
model = pipeline.named_steps["model"]
coef = model.coef_[0]

"Feature X increases log-odds by coef[i]"
Not intuitive for business stakeholders.


Odds Ratio (more intuitive):
───────────────────────────
odds_ratio = exp(coef)

Coefficient    Odds Ratio    Meaning
────────────────────────────────────────────────────────────────
+0.69          2.0           "2x more likely / doubles the odds"
+0.41          1.5           "50% more likely / 1.5x odds"
+0.10          1.11          "11% more likely"
0.00           1.0           "No effect"
−0.10          0.90          "10% less likely"
−0.41          0.67          "33% less likely / 0.67x odds"
−0.69          0.5           "Half as likely / halves the odds"


Example Interpretation:
───────────────────────
Feature: Age_standardized, Coefficient: 0.50, Odds Ratio: 1.67

"One standard deviation increase in age increases odds of  
 conversion by 67%, or roughly 1.67x higher odds."


CRITICAL: Features must be STANDARDIZED!
─────────────────────────────────────────
Without scaling:
  - Feature in thousands → tiny coefficient (misleading!)
  - Feature in single digits → larger coefficient (misleading!)

With StandardScaler in pipeline:
  - All coefficients on same scale
  - Magnitude reflects true importance
  - Odds ratios are comparable


═══════════════════════════════════════════════════════════════════════════════
CONFUSION MATRIX INTERPRETATION
═══════════════════════════════════════════════════════════════════════════════

                Predicted Negative    Predicted Positive
Actual Negative      TN (TP)              FP (False Positive)
Actual Positive      FN (False Negative)  TP (True Positive)

TN (True Negative): Correctly predicted class 0
FP (False Positive): Incorrectly predicted class 1 (type I error)
FN (False Negative): Incorrectly predicted class 0 (type II error)
TP (True Positive): Correctly predicted class 1

Metrics from confusion matrix:
  Precision = TP / (TP + FP)     [Of predicted 1s, % correct]
  Recall = TP / (TP + FN)        [Of actual 1s, % found]
  Accuracy = (TP + TN) / Total   [Overall correctness]
  F1 = 2 × (Precision × Recall) / (Precision + Recall)

When FP is costly (e.g., spam): Optimize for high precision
When FN is costly (e.g., cancer): Optimize for high recall


═══════════════════════════════════════════════════════════════════════════════
REGULARIZATION: L2 VS L1
═══════════════════════════════════════════════════════════════════════════════

L2 Regularization (default, penalty="l2"):
───────────────────────────────────────
  Adds penalty proportional to magnitude of coefficients
  Effect: Shrinks all coefficients, none to exactly zero
  Use: When you want to use all features but prevent overfitting
  Control: C parameter (lower C = more regularization)

L1 Regularization (penalty="l1", solver="liblinear"):
──────────────────────────────────────────────────
  Adds penalty proportional to absolute value of coefficients
  Effect: Drives some coefficients to exactly zero (feature selection)
  Use: When you want automatic feature selection
  Control: C parameter (lower C = more coefficients zeroed)

Comparison:
  L2: All features used with smaller coefficients
  L1: Some features removed entirely (coef = 0)

Which to use?
  - L2 (default): Start here, safe choice
  - L1: If feature selection is important


═══════════════════════════════════════════════════════════════════════════════
STRATIFICATION: WHY IT MATTERS
═══════════════════════════════════════════════════════════════════════════════

WITHOUT stratification:
  Original: 90% class 0, 10% class 1
  Random split might give:
    Train: 85% class 0, 15% class 1
    Test: 95% class 0, 5% class 1
  
  Result: Test set is unrepresentative! CV results unreliable.


WITH stratification (stratify=y):
  Original: 90% class 0, 10% class 1
  Guaranteed:
    Train: ~90% class 0, ~10% class 1
    Test: ~90% class 0, ~10% class 1
  
  Result: Best practice for classification!


Code:
  ✗ X_train, X_test = train_test_split(X, ...)  # No stratification!
  ✓ X_train, X_test = train_test_split(X, stratify=y)  # Correct!


═══════════════════════════════════════════════════════════════════════════════
CROSS-VALIDATION: CHECKING STABILITY
═══════════════════════════════════════════════════════════════════════════════

from sklearn.model_selection import cross_val_score

cv_auc = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
print(f"CV ROC-AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

Output example 1 (STABLE):
  CV ROC-AUC: 0.87 ± 0.02
  → Reliable model, consistent performance

Output example 2 (UNSTABLE):
  CV ROC-AUC: 0.87 ± 0.25
  → Erratic performance, depends on which data seen
  → Red flag for overfitting or outliers


Interpretation:
  Std Dev < 0.05:      ✓ Very stable
  Std Dev 0.05-0.10:   ✓ Stable
  Std Dev 0.10-0.20:   ~ Moderate (acceptable)
  Std Dev > 0.20:      ⚠ Unstable (investigate)


═══════════════════════════════════════════════════════════════════════════════
COMMON PITFALLS
═══════════════════════════════════════════════════════════════════════════════

PITFALL 1: Not stratifying train/test split
──────────────────────────────────────────
✗ Wrong: X_train, X_test = train_test_split(X, y, ...)
✓ Right: X_train, X_test = train_test_split(X, y, stratify=y)


PITFALL 2: Trusting accuracy on imbalanced data
────────────────────────────────────────────────
✗ Wrong: "Our model achieves 92% accuracy!"
         (Baseline might also achieve 90% by predicting majority)
✓ Right: "Our model achieves F1=0.78 and ROC-AUC=0.88"
         (These reflect true minority class performance)


PITFALL 3: Scaling BEFORE splitting
────────────────────────────────────
✗ Wrong: X_scaled = StandardScaler().fit_transform(X)
         X_train, X_test = train_test_split(X_scaled, ...)
         (Test statistics leaked into scaler!)
✓ Right: X_train, X_test = train_test_split(X, stratify=y)
         Insert StandardScaler in Pipeline (fitted on train only)


PITFALL 4: Not comparing against baseline
──────────────────────────────────────────
✗ Wrong: "F1 = 0.65" (Is this good?)
✓ Right: "F1 = 0.65 vs baseline F1 = 0.08" (Clearly better!)


PITFALL 5: Using wrong evaluation metric on imbalanced data
────────────────────────────────────────────────────────────
✗ Wrong: "Accuracy improved from 89% to 91%"
         (Both might be dominated by majority class)
✓ Right: "F1 improved from 0.15 to 0.72"
         (True minority class improvement)


PITFALL 6: Convergence warnings
────────────────────────────────
✗ Warning: "ConvergenceWarning: lbfgs failed to converge"
✓ Solution: Increase max_iter parameter
  LogisticRegression(max_iter=5000)


═══════════════════════════════════════════════════════════════════════════════
KEY FILES REFERENCE
═══════════════════════════════════════════════════════════════════════════════

Core Implementation:
  src/train_logistic_regression.py          — Training module
  src/evaluate_classification_metrics.py    — Evaluation module

Examples & Integration:
  src/demo_logistic_regression.py           — 5 worked examples (RUN THIS!)
  src/integrate_logistic_regression.py      — Integration example

Guides & References:
  LOGISTIC_REGRESSION_GUIDE.md              — Comprehensive guide
  LOGISTIC_REGRESSION_QUICK_REFERENCE.md    — This document


═══════════════════════════════════════════════════════════════════════════════
THE GOLDEN RULES
═══════════════════════════════════════════════════════════════════════════════

1. Always stratify train/test split
   stratify=y prevents unrepresentative splits

2. Never trust accuracy on imbalanced data
   Use F1 and ROC-AUC instead

3. Always compare against majority-class baseline
   Without baseline, metrics are uncontextualized

4. Always make both class predictions and probabilities
   y_pred for accuracy, y_prob for ROC-AUC

5. Always cross-validate
   Single split can be misleading

6. Interpret coefficients as odds ratios
   exp(coef) is more intuitive than raw coefficient

7. Features must be standardized
   Use StandardScaler in pipeline for comparable coefficients

8. Scale features BEFORE and AFTER split
   Use pipeline to prevent leakage
"""
