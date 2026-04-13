"""
MSE AND R² EVALUATION: COMPREHENSIVE USAGE GUIDE

This guide explains how to use the RegressionMetricsEvaluator class to
properly evaluate regression models using MSE and R² metrics.

WORKSHOP STRUCTURE:
├─ PART 1: Understanding MSE vs R²
├─ PART 2: Implementation Walkthrough
├─ PART 3: Integration with Your Pipeline
├─ PART 4: Common Pitfalls & Solutions
└─ PART 5: Quick Reference

═══════════════════════════════════════════════════════════════════════════════
PART 1: UNDERSTANDING MSE vs R² (THE FUNDAMENTAL DIFFERENCE)
═══════════════════════════════════════════════════════════════════════════════

MSE (Mean Squared Error)
─────────────────────────
Formula: MSE = (1/n) × Σ(y_true - y_pred)²

Characteristics:
  • ABSOLUTE quantity (in squared target units)
  • Sensitive to outliers (errors squared mean large errors amplified)
  • No inherent interpretation (depends on target scale)
  • Directly optimized by Linear Regression

Example:
  Target values:    [10, 20, 30]
  Predictions:      [12, 18, 35]
  Errors:           [2, -2, 5]
  Squared errors:   [4, 4, 25]
  MSE:              11

R² (Coefficient of Determination)
──────────────────────────────────
Formula: R² = 1 - (SS_res / SS_tot)
  Where:
    SS_res = Σ(y_true - y_pred)²    (your model's error sum)
    SS_tot = Σ(y_true - mean)²      (mean baseline's error sum)

Characteristics:
  • RELATIVE quantity (compared to mean baseline)
  • Represents proportion of variance explained
  • Scale: from -∞ to 1 (though typically -1 to 1)
  • Interpretation: R² = 0.75 means "75% of variance explained"

Example (same data):
  SS_tot = 400 (total variance around mean of 20)
  SS_res = 45  (your model's squared errors)
  R² = 1 - (45/400) = 0.8875 ← Model explains 88.75% of variance

KEY INSIGHT FROM LESSON:
────────────────────────
"MSE without R² gives you no sense of how much you've improved over doing nothing.
R² without MSE gives you no sense of how large the actual errors are."

SOLUTION: Always report both metrics together, with baseline comparison.

═══════════════════════════════════════════════════════════════════════════════
PART 2: IMPLEMENTATION WALKTHROUGH
═══════════════════════════════════════════════════════════════════════════════

Step 1: Create Evaluator
────────────────────────
from evaluate_regression_metrics import RegressionMetricsEvaluator

evaluator = RegressionMetricsEvaluator(random_state=42)


Step 2: Basic Test Set Evaluation
──────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Split data BEFORE preprocessing (prevents leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
metrics = evaluator.evaluate_on_test_set(y_test, y_pred, "My Model")

print(f"MSE:  {metrics['mse']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")  # Use this for reporting
print(f"MAE:  {metrics['mae']:.4f}")
print(f"R²:   {metrics['r2']:.4f}")


Step 3: Baseline Comparison (MANDATORY)
────────────────────────────────────────
# This is the CRITICAL step that makes R² interpretable

comparison = evaluator.compare_with_baseline(
    X_train, y_train,
    X_test, y_test,
    y_pred,
    model_name="My Model"
)

baseline_metrics = comparison['baseline']    # MSE=400, R²≈0
model_metrics = comparison['model']          # MSE=100, R²=0.75

# Interpretation:
# - Baseline R² will be ≈ 0 (by definition)
# - If model R² > baseline R², model is better
# - If model R² < baseline R², something is wrong


Step 4: Cross-Validation (For Stability)
─────────────────────────────────────────
# Single test split can be misleading. Cross-validation shows consistency.

cv_r2 = evaluator.cross_validate_r2(model, X_train, y_train, cv=5)
cv_rmse = evaluator.cross_validate_rmse(model, X_train, y_train, cv=5)

print(f"R² scores: {cv_r2['scores']}")
print(f"Mean R²:   {cv_r2['mean']:.4f} ± {cv_r2['std']:.4f}")

# Stability interpretation:
# - std < 0.05: ✓ Stable, reliable
# - std 0.05-0.15: ~ Moderate stability
# - std > 0.15: ⚠ Unstable, check for overfitting


Step 5: Interpretation
──────────────────────
# Get human-readable interpretation
interpretation = evaluator.interpret_metrics(
    mse=metrics['mse'],
    rmse=metrics['rmse'],
    r2=metrics['r2'],
    baseline_r2=0.0  # Mean predictor has R²=0
)

print(interpretation['mse_magnitude'])      # "Small", "Moderate", "Large"
print(interpretation['r2_level'])            # "Excellent", "Good", etc.
print(interpretation['combined_story'])      # Full interpretation


Step 6: Formatted Report
────────────────────────
# Generate a nice-looking report
evaluator.print_evaluation_report(
    y_test, y_pred, 
    y_pred_baseline,          # Optional
    model_name="My Model"
)

Output:
  ======================================================================
  REGRESSION EVALUATION REPORT: My Model
  ======================================================================
  
  Test Set Size: 40 samples
  
  Metric              Value           Interpretation
  ──────────────────────────────────────────────────────────────────────
  MSE                 100.1234        Squared units (less interpretable)
  RMSE (★)            10.0062         Same units as target (preferred)
  MAE                 7.5432          Average absolute error
  R²                  0.7500          Fraction of variance explained (GOOD)
  
  ======================================================================

═══════════════════════════════════════════════════════════════════════════════
PART 3: INTEGRATION WITH YOUR PIPELINE
═══════════════════════════════════════════════════════════════════════════════

Example: Complete ML Pipeline with Proper Evaluation
─────────────────────────────────────────────────────

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from evaluate_regression_metrics import RegressionMetricsEvaluator

def train_and_evaluate():
    '''Complete ML pipeline with proper evaluation.'''
    
    # 1. Load data
    X, y = load_data('data/ride_data.csv')
    
    # 2. SPLIT FIRST (before any preprocessing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Create preprocessing + model pipeline
    # (scaler is fitted ONLY on training data)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    
    # 4. Train
    pipeline.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = pipeline.predict(X_test)
    
    evaluator = RegressionMetricsEvaluator()
    
    # 5a. Basic metrics
    metrics = evaluator.evaluate_on_test_set(y_test, y_pred, "Linear Regression")
    
    # 5b. Baseline comparison
    comparison = evaluator.compare_with_baseline(
        X_train, y_train, X_test, y_test, y_pred, "Linear Regression"
    )
    
    # 5c. Cross-validation
    cv_results = evaluator.cross_validate_r2(pipeline, X_train, y_train, cv=5)
    
    # 5d. Report
    y_pred_baseline = comparison['baseline_predictions']
    evaluator.print_evaluation_report(y_test, y_pred, y_pred_baseline)
    
    return pipeline, metrics, cv_results


═══════════════════════════════════════════════════════════════════════════════
PART 4: COMMON PITFALLS & SOLUTIONS
═══════════════════════════════════════════════════════════════════════════════

PITFALL 1: Reporting MSE Without RMSE
──────────────────────────────────────
Problem: MSE = 25 lakhs² is meaningless to stakeholders
Solution: Always report RMSE = √25 = 5 lakhs (same units as target)

Code:
  mse = mean_squared_error(y_test, y_pred)
  rmse = np.sqrt(mse)  # ← THIS converts to interpretable units
  print(f"RMSE: {rmse:.2f}")  # Use this with stakeholders


PITFALL 2: Evaluating Without Baseline
────────────────────────────────────────
Problem: R² = 0.45 tells you nothing without context
Solution: Always compute R² for mean baseline (will be ≈0 on test set)

Code:
  baseline = DummyRegressor(strategy='mean')
  baseline.fit(X_train, y_train)
  y_pred_baseline = baseline.predict(X_test)
  r2_baseline = r2_score(y_test, y_pred_baseline)  # Should be ≈0
  r2_model = r2_score(y_test, y_pred)              # Should be > 0


PITFALL 3: Train/Test Data Leakage in Preprocessing
─────────────────────────────────────────────────────
Problem: Fitting scaler on BOTH train and test data
  ✗ X_scaled = scaler.fit_transform(X)  # Uses test data to compute mean!
  X_train, X_test = train_test_split(X_scaled, ...)

Solution: Split BEFORE preprocessing
  X_train, X_test = train_test_split(X, ...)  # Raw data
  scaler.fit(X_train)                          # Learn on train only
  X_train_scaled = scaler.transform(X_train)  # Transform train
  X_test_scaled = scaler.transform(X_test)    # Transform test


PITFALL 4: High R² on Training, Low on Testing
───────────────────────────────────────────────
Problem: This indicates overfitting
Solution: Use cross-validation to detect it early

Code:
  from sklearn.model_selection import cross_val_score
  train_r2 = evaluator.evaluate_on_test_set(y_train, y_pred_train)
  test_r2 = evaluator.evaluate_on_test_set(y_test, y_pred_test)
  
  if train_r2['r2'] > test_r2['r2'] + 0.1:
      print("⚠ Possible overfitting detected")


PITFALL 5: Ignoring Negative R²
────────────────────────────────
Problem: Negative R² means model is worse than mean baseline (red flag!)
Solution: Investigate immediately

Code:
  if r2 < 0:
      logger.critical("Model R² is negative! Investigate:")
      logger.critical("  1. Check for data leakage")
      logger.critical("  2. Verify train/test split")
      logger.critical("  3. Look for distribution shift")
      logger.critical("  4. Check for bugs in pipeline")


═══════════════════════════════════════════════════════════════════════════════
PART 5: QUICK REFERENCE
═══════════════════════════════════════════════════════════════════════════════

When to Use Each Metric:
────────────────────────

MSE:
  ✓ Internal optimization (that's what Linear Regression minimizes)
  ✓ Model comparison (higher MSE = better for loss functions)
  ✗ Reporting to stakeholders (units are squared and confusing)

RMSE:
  ✓ Reporting to stakeholders (same units as target)
  ✓ Making error magnitude intuitive
  ✓ Comparing models when scale matters
  
MAE:
  ✓ When outliers shouldn't be heavily penalized
  ✓ For more interpretable "average error"
  
R²:
  ✓ Understanding relative model quality
  ✓ MUST compare against baseline
  ✓ For answering "how much variance does this explain?"
  ✗ As standalone metric (always needs baseline context)


Metric Interpretation Quick Table:
──────────────────────────────────

R² Value    Meaning
───────────────────────────────────────────────────────────────
1.0         Perfect prediction (unlikely in practice)
0.90-0.99   Excellent — model explains >90% of variance
0.70-0.90   Good — model explains 70-90% of variance
0.50-0.70   Moderate — half to 3/4 of variance explained
0.30-0.50   Fair — model learning something but limited
0.0-0.30    Poor — explains <30% of variance
0.0         Model equals mean predictor (baseline)
< 0.0       CRITICAL: Model worse than guessing mean (red flag!)


Evaluation Checklist:
─────────────────────
□ Split data BEFORE preprocessing (prevents leakage)
□ Fit scaler only on training data
□ Evaluate on held-out test set
□ Compute: MSE, RMSE, MAE, R²
□ Compare against baseline (mean predictor)
□ Run cross-validation (cv=5) for stability
□ Check for negative R² (red flag if present)
□ Check for overfitting (train >> test R²)
□ Report RMSE, not MSE, to stakeholders
□ Provide interpretation alongside metrics


═══════════════════════════════════════════════════════════════════════════════

For more details, see:
  - evaluate_regression_metrics.py (implementation)
  - demo_mse_r2_evaluation.py (examples)
  - Original lesson (MSE and R² best practices)
"""

def quick_evaluation_template():
    """
    Copy-paste template for quick evaluation.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from evaluate_regression_metrics import RegressionMetricsEvaluator
    
    # Your data
    X, y = ...  # Load your data
    
    # 1. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 2. Train with preprocessing pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    
    # 3. Evaluate
    y_pred = pipeline.predict(X_test)
    evaluator = RegressionMetricsEvaluator()
    
    # 3a. Baseline comparison
    comparison = evaluator.compare_with_baseline(
        X_train, y_train, X_test, y_test, y_pred
    )
    
    # 3b. Cross-validation
    cv_r2 = evaluator.cross_validate_r2(pipeline, X_train, y_train, cv=5)
    
    # 3c. Report
    evaluator.print_evaluation_report(y_test, y_pred)
    
    # 3d. Return results
    return {
        'test_metrics': comparison['model'],
        'baseline_metrics': comparison['baseline'],
        'cv_r2': cv_r2
    }
