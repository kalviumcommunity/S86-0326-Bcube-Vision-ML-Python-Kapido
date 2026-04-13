"""
═══════════════════════════════════════════════════════════════════════════════
LOGISTIC REGRESSION CLASSIFICATION: COMPREHENSIVE GUIDE
═══════════════════════════════════════════════════════════════════════════════

This guide covers training and evaluating Logistic Regression classification 
models following best practices from the lesson.

What's covered:
- Part 1: Understanding Logistic Regression conceptually
- Part 2: Why not use Linear Regression for classification
- Part 3: The sigmoid function and decision boundary
- Part 4: Training with log loss
- Part 5: Implementation in scikit-learn
- Part 6: Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Part 7: Baseline comparison
- Part 8: Coefficient interpretation
- Part 9: Common pitfalls

═══════════════════════════════════════════════════════════════════════════════
PART 1: WHAT IS LOGISTIC REGRESSION?
═══════════════════════════════════════════════════════════════════════════════

Despite its name, Logistic Regression is a CLASSIFICATION algorithm, not regression.

The "regression" refers to the underlying linear model. The "logistic" refers to
the logistic (sigmoid) function that transforms linear outputs to probabilities.

INPUT: Features (X)
  ↓
LINEAR: z = w·X + b (like linear regression)
  ↓
SIGMOID: P(class 1) = sigmoid(z) = 1 / (1 + e^-z)
  ↓
OUTPUT: Probability ∈ [0, 1]

Decision rule (at threshold 0.5):
  If P(class 1) ≥ 0.5 → predict class 1
  If P(class 1) < 0.5 → predict class 0


═══════════════════════════════════════════════════════════════════════════════
PART 2: WHY NOT USE LINEAR REGRESSION FOR CLASSIFICATION?
═══════════════════════════════════════════════════════════════════════════════

Problem 1: Unbounded Outputs
─────────────────────────────
Linear Regression can predict -0.5 or 1.7 for a binary target.
These values are invalid probabilities (must be in [0, 1]).
Example: A prediction of 1.7 means "170% chance of class 1" — nonsense.

Problem 2: Wrong Relationship Shape
────────────────────────────────────
The actual relationship between features and class membership is non-linear,
especially near the decision boundary.
Linear Regression forces a straight line through inherently S-shaped data,
leading to systematic misclassification near the boundary.

Problem 3: Wrong Loss Function
──────────────────────────────
Linear Regression minimizes MSE, which assumes normally distributed errors.
Binary targets (0/1) violate this assumption severely.
Prediction errors for classification are uniform (off by 0.5 is bad; off by 1.0 is bad).
Linear Regression doesn't penalize binary errors appropriately.

SOLUTION: Logistic Regression
──────────────────────────────
✓ Sigmoid ensures outputs are always in [0, 1]
✓ S-shaped sigmoid matches actual class probability patterns
✓ Uses log loss, designed specifically for probability estimation


═══════════════════════════════════════════════════════════════════════════════
PART 3: THE SIGMOID FUNCTION AND DECISION BOUNDARY
═══════════════════════════════════════════════════════════════════════════════

The Sigmoid Function
────────────────────
sigmoid(z) = 1 / (1 + e^-z)

Behavior:
  z → −∞        sigmoid(z) → 0      (strong prediction of class 0)
  z = −5        sigmoid(z) ≈ 0.007
  z = 0         sigmoid(z) = 0.5    (maximum uncertainty)
  z = +5        sigmoid(z) ≈ 0.993
  z → +∞        sigmoid(z) → 1      (strong prediction of class 1)

Key property: The curve is smooth and S-shaped (logistic curve).
As z grows in either direction, the probability saturates near 0 or 1.


The Decision Boundary
─────────────────────
Decision boundary occurs where sigmoid(z) = 0.5
This happens when z = 0, i.e., when w·X + b = 0

This is a LINEAR decision boundary:
  - In 2D: a line separating class 0 and class 1
  - In 3D: a plane
  - In nD: a hyperplane

This is both the strength and limitation of Logistic Regression:
  ✓ Strength: Interpretable, efficient, works well for linearly separable data
  ✗ Limitation: Can't represent curved boundaries (use tree-based models instead)


Threshold Adjustment
────────────────────
The default threshold is 0.5, but you can adjust it:

  Higher threshold (e.g., 0.7): More conservative predictions
    - Fewer false positives
    - More false negatives
    - Use when predicting positives is costly (e.g., loan defaults)

  Lower threshold (e.g., 0.3): More sensitive predictions  
    - More false positives
    - Fewer false negatives
    - Use when missing positives is costly (e.g., disease diagnosis)


═══════════════════════════════════════════════════════════════════════════════
PART 4: TRAINING WITH LOG LOSS
═══════════════════════════════════════════════════════════════════════════════

Log Loss (Cross-Entropy Loss)
────────────────────────────
Log Loss = −(1/n) Σ [y·log(p) + (1−y)·log(1−p)]

Where:
  y = true label (0 or 1)
  p = predicted probability of class 1
  n = number of samples

Unpacking the Intuition
───────────────────────
When y = 1 (true positive):
  Loss = −log(p)
  If p = 0.99: loss ≈ 0.01 (good)
  If p = 0.50: loss ≈ 0.69 (bad)
  If p = 0.01: loss ≈ 4.60 (terrible)

When y = 0 (true negative):
  Loss = −log(1−p)
  If p = 0.01: loss ≈ 0.01 (good)
  If p = 0.50: loss ≈ 0.69 (bad)
  If p = 0.99: loss ≈ 4.60 (terrible)

KEY INSIGHT: Being confidently wrong incurs enormous loss.
The model severely penalizes predictions that are both:
  1. Wrong (incorrect class)
  2. Confident (high probability)


Why Log Loss Over MSE?
──────────────────────
1. Heavy penalty for confident wrong predictions
   You can't afford to be certain and wrong.

2. Encourages well-calibrated probabilities
   Predicted probabilities reflect true likelihoods.

3. Convex optimization landscape
   Single global minimum → gradient descent guaranteed to find best solution.

4. Designed for probability estimation
   Inherently treats classification as probability prediction.


═══════════════════════════════════════════════════════════════════════════════
PART 5: IMPLEMENTATION IN SCIKIT-LEARN
═══════════════════════════════════════════════════════════════════════════════

Step 1: Imports
───────────────
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report


Step 2: Stratified Train-Test Split
────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # ← CRITICAL: Preserve class proportions
)

Why stratify?
Without stratification, random split might put 90% of minority class in train
and 10% in test, making evaluation unrepresentative.

stratify=y guarantees both train and test reflect original class distribution.


Step 3: Build Pipeline
──────────────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),           # Scale features (fitted on train only)
    ("model", LogisticRegression(
        max_iter=1000,                      # Increase from default 100 if needed
        random_state=42,
        C=1.0,                              # Inverse regularization strength
        penalty="l2"                        # L2 regularization (default)
    ))
])

Why Pipeline?
- Scaler fitted ONLY on training data
- Same scaling applied to test data
- Prevents data leakage


Important Parameters:
  max_iter: Increase if you see ConvergenceWarning
  C: Inverse regularization (lower = more regularization)
  penalty: "l2" or "l1" (l1 for feature selection)
  class_weight: "balanced" for imbalanced data


Step 4: Fit Model
─────────────────
pipeline.fit(X_train, y_train)


Step 5: Make Predictions
─────────────────────────
y_pred = pipeline.predict(X_test)           # Class labels (0 or 1)
y_prob  = pipeline.predict_proba(X_test)[:, 1]  # Probabilities of class 1

ALWAYS get both:
  - y_pred for accuracy, precision, recall, F1
  - y_prob for ROC-AUC and threshold tuning


═══════════════════════════════════════════════════════════════════════════════
PART 6: EVALUATION METRICS FOR CLASSIFICATION
═══════════════════════════════════════════════════════════════════════════════

Four Essential Metrics
──────────────────────

1. ACCURACY
───────────
accuracy = (TP + TN) / (TP + TN + FP + FN)
         = Correct predictions / Total predictions

Interpretation:
  ✓ Good for balanced data
  ✗ MISLEADING on imbalanced data

Example (90% negative class):
  Naive baseline (always predict 0): 90% accuracy but learns nothing!
  Your model (balanced predictions): 85% accuracy but actually learns.

RULE: Never report accuracy alone on imbalanced data. Always include F1/ROC-AUC.


2. PRECISION
────────────
precision = TP / (TP + FP)
          = Of predicted positives, what % are truly positive

Interpretation:
  High precision = few false positives
  Use when false positives are costly
  Example: Spam detection. False alarm = user annoyed.

When to optimize:
  - Medical treatments (avoid false positives → unnecessary treatment)
  - Loan approvals (avoid false positives → bad debt)


3. RECALL (Sensitivity)
───────────────────────
recall = TP / (TP + FN)
       = Of actual positives, what % did we correctly identify

Interpretation:
  High recall = few false negatives
  Use when false negatives are costly
  Example: Disease diagnosis. Missing a disease = patient dies.

When to optimize:
  - Disease detection (avoid false negatives → missed diagnosis)
  - Security threats (avoid false negatives → breach)
  - Fraud detection (avoid false negatives → transaction goes through)


4. F1-SCORE (PRIMARY FOR IMBALANCED DATA)
──────────────────────────────────────────
F1 = 2 × (precision × recall) / (precision + recall)
   = Harmonic mean of precision and recall

Interpretation:
  ✓ Can't be gamed by predicting majority class
  ✓ Considers both false positives AND false negatives
  ✓ PRIMARY metric for imbalanced classification

Example (90% negative class):
  Naive baseline: F1 ≈ 0 (can't fool F1!)
  Your model: F1 ≈ 0.8 (clear advantage)


5. ROC-AUC (Ranking Quality)
──────────────────────────────
ROC-AUC = Area under Receiver Operating Characteristic curve

Interpretation:
  "If I pick one random positive and one random negative,
   what's the probability my model ranks the positive higher?"

AUC = 1.0: Perfect ranking (positive always scores higher than negative)
AUC = 0.9+: Excellent ranking
AUC = 0.7–0.9: Good ranking
AUC = 0.6–0.7: Fair ranking
AUC = 0.5: Random guessing (same as majority class baseline)
AUC < 0.5: Worse than random (check label inversion!)

Why ROC-AUC for imbalanced data?
  - Unaffected by class imbalance
  - Evaluates ranking (not absolute predictions)
  - Baseline AUC = 0.5 (not skewed by class proportions)


═══════════════════════════════════════════════════════════════════════════════
PART 7: BASELINE COMPARISON (MANDATORY)
═══════════════════════════════════════════════════════════════════════════════

CRITICAL PRINCIPLE: Always compare against a baseline.

For classification, the baseline is DummyClassifier with "most_frequent" strategy:

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
baseline_prob = baseline.predict_proba(X_test)[:, 1]


Baseline Behavior
─────────────────
The baseline always predicts the majority class for every sample.

On balanced data (50-50):
  Baseline accuracy = 50%
  Baseline F1 ≈ 0.33
  Baseline ROC-AUC = 0.5

On imbalanced data (90-10):
  Baseline accuracy = 90% (!) (but learns nothing)
  Baseline F1 ≈ 0 (can't fool F1)
  Baseline ROC-AUC = 0.5


Why Compare?
────────────
1. Shows if your model learns anything beyond class imbalance
2. Contextualizes your metrics (65% accuracy is bad if baseline is 60%)
3. Ensures your model isn't worse than naive approach
4. Provides meaningful improvement measurement


Example Comparison
──────────────────
                Baseline    Your Model    Improvement
Accuracy        90%         91%           +1%
F1              0.0         0.75          +0.75 ← Massive!
ROC-AUC         0.50        0.85          +0.35 ← Significant!

On imbalanced data, F1 and ROC-AUC improvements matter; accuracy doesn't.


═══════════════════════════════════════════════════════════════════════════════
PART 8: COEFFICIENT INTERPRETATION (ODDS RATIOS)
═══════════════════════════════════════════════════════════════════════════════

Logistic Regression coefficients live in log-odds space.

Model: log(odds) = w·X + b
Where: odds = P(class 1) / P(class 0)


Raw Coefficient Interpretation
────────────────────────────────
Coefficient w_i means:
  "A one-unit increase in feature X_i changes log-odds by w_i"

Example:
  w_age = 0.05
  → One year older increases log-odds by 0.05

But log-odds isn't intuitive to business stakeholders.


Odds Ratio Interpretation (More Intuitive)
─────────────────────────────────────────
Odds Ratio = exp(coefficient)

Example:
  w_age = 0.69
  Odds Ratio = exp(0.69) = 2.0
  → One year older DOUBLES the odds of class 1

Examples:
  Coefficient = +0.69  →  Odds Ratio = 2.0   →  "2x more likely"
  Coefficient = −0.69  →  Odds Ratio = 0.5   →  "50% as likely" or "−50%"
  Coefficient = 0.00   →  Odds Ratio = 1.0   →  "No effect"
  Coefficient = 0.10   →  Odds Ratio = 1.11  →  "11% more likely"


How to Extract and Interpret
──────────────────────────────
model = pipeline.named_steps["model"]
coef = model.coef_[0]
odds_ratios = np.exp(coef)

# Create interpretable DataFrame
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": coef,
    "Odds Ratio": odds_ratios
}).sort_values("Coefficient", key=abs, ascending=False)

# Interpret top features
for _, row in coef_df.iterrows():
    feature = row["Feature"]
    or_val = row["Odds Ratio"]
    if or_val > 1:
        pct = (or_val - 1) * 100
        print(f"{feature}: +{pct:.1f}% odds per unit increase")
    else:
        pct = (1 - or_val) * 100
        print(f"{feature}: −{pct:.1f}% odds per unit increase")


IMPORTANT: Features must be standardized!
────────────────────────────────────────
Coefficient magnitudes are only comparable across features if SCALED.

Without scaling:
  - Feature in thousands → small coefficient (even if important)
  - Feature in single digits → large coefficient (even if unimportant)

Solution: Use StandardScaler in pipeline (already implemented).
With scaling, coefficient magnitudes reflect true feature importance.


═══════════════════════════════════════════════════════════════════════════════
PART 9: REGULARIZATION
═══════════════════════════════════════════════════════════════════════════════

Scikit-learn's LogisticRegression includes L2 regularization by default.

Why Regularization?
───────────────────
1. Prevents overfitting on small/noisy datasets
2. Handles multicollinearity (correlated features)
3. Improves generalization


Regularization Parameters
─────────────────────────
LogisticRegression(C=1.0, penalty="l2")

C = Inverse regularization strength:
  C = 0.001   → Very strong regularization (shrink coefficients aggressively)
  C = 1.0     → Moderate regularization (default)
  C = 100     → Weak regularization (close to no regularization)

penalty = Type of regularization:
  "l2" (default): Even shrinkage across all features
  "l1": Drives some coefficients to exactly 0 (feature selection)


Finding Optimal C (Grid Search)
──────────────────────────────
from sklearn.model_selection import GridSearchCV

param_grid = {"model__C": [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc")
grid_search.fit(X_train, y_train)
print(f"Best C: {grid_search.best_params_['model__C']}")


When to Use Imbalanced Classes
──────────────────────────────
from sklearn.linear_model import LogisticRegression

LogisticRegression(class_weight="balanced")

With class_weight="balanced":
  - Minority class is weighted higher during training
  - Model doesn't default to majority class
  - Particularly important if minority class is rare
  - First response to severe imbalance before resampling

═══════════════════════════════════════════════════════════════════════════════
QUICK START TEMPLATE
═══════════════════════════════════════════════════════════════════════════════

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# 1. Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

# 3. Model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])
pipeline.fit(X_train, y_train)

# 4. Evaluate
baseline_pred = baseline.predict(X_test)
baseline_prob = baseline.predict_proba(X_test)[:, 1]

model_pred = pipeline.predict(X_test)
model_prob  = pipeline.predict_proba(X_test)[:, 1]

# 5. Report
print(f"Baseline ROC-AUC: {roc_auc_score(y_test, baseline_prob):.3f}")
print(f"Model ROC-AUC:    {roc_auc_score(y_test, model_prob):.3f}")
print(classification_report(y_test, model_pred))

# 6. Cross-validate
cv_auc = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
print(f"CV ROC-AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

═══════════════════════════════════════════════════════════════════════════════
"""
