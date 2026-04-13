"""
═══════════════════════════════════════════════════════════════════════════════
MSE & R² QUICK REFERENCE CARD
═══════════════════════════════════════════════════════════════════════════════
Keep this handy while evaluating regression models.

═══════════════════════════════════════════════════════════════════════════════
ONE-LINER SUMMARY
═══════════════════════════════════════════════════════════════════════════════

MSE measures absolute error. R² measures relative improvement.
Report RMSE (not MSE) to stakeholders. Always compare against baseline.
If R² < 0, something is seriously wrong.


═══════════════════════════════════════════════════════════════════════════════
THE FOUR ESSENTIAL METRICS
═══════════════════════════════════════════════════════════════════════════════

┌─────────┬──────────────┬──────────────┬────────────────┬──────────────────┐
│ Metric  │ Formula      │ Units        │ Interpretation │ Use When         │
├─────────┼──────────────┼──────────────┼────────────────┼──────────────────┤
│ MSE     │ mean(error²) │ Squared      │ Lower = better │ Internal         │
│         │              │ target units │ (sensitive to  │ optimization     │
│         │              │              │ outliers)      │                  │
├─────────┼──────────────┼──────────────┼────────────────┼──────────────────┤
│ RMSE ★  │ √MSE         │ Target units │ Lower = better │ Reporting to     │
│         │              │ (interpret.) │ (same units as │ stakeholders     │
│         │              │              │ target!)       │ (PREFERRED)      │
├─────────┼──────────────┼──────────────┼────────────────┼──────────────────┤
│ MAE     │ mean(|error|)│ Target units │ Lower = better │ Outlier-robust   │
│         │              │              │ (outlier safe) │ assessment       │
├─────────┼──────────────┼──────────────┼────────────────┼──────────────────┤
│ R²      │ 1-(SS_res/   │ Proportion   │ 1.0 = perfect  │ Relative model   │
│         │ SS_tot)      │ (-∞ to 1)    │ 0.0 = baseline │ quality (ALWAYS  │
│         │              │              │ <0 = worse!    │ use with         │
│         │              │              │                │ baseline)        │
└─────────┴──────────────┴──────────────┴────────────────┴──────────────────┘


═══════════════════════════════════════════════════════════════════════════════
R² QUICK INTERPRETATION
═══════════════════════════════════════════════════════════════════════════════

R² > 0.7         Excellent     85%+ variance explained      ✓ Deploy
R² 0.5-0.7       Good           Model worth using           ✓ Consider
R² 0.3-0.5       Fair           Better than nothing         ⚠ Investigate
R² 0.0-0.3       Poor           Barely better than baseline ⚠ Improve
R² = 0.0         Baseline       Same as mean predictor      (definition)
R² < 0.0         WORSE!         Model beats mean baseline   🚨 RED FLAG!


═══════════════════════════════════════════════════════════════════════════════
QUICK EVALUATION CODE TEMPLATE
═══════════════════════════════════════════════════════════════════════════════

from src.evaluate_regression_metrics import RegressionMetricsEvaluator

evaluator = RegressionMetricsEvaluator()

# 1. Basic evaluation
metrics = evaluator.evaluate_on_test_set(y_test, y_pred, "Model")

# 2. Baseline comparison (MANDATORY)
comparison = evaluator.compare_with_baseline(
    X_train, y_train, X_test, y_test, y_pred
)

# 3. Cross-validation (for stability)
cv_r2 = evaluator.cross_validate_r2(model, X_train, y_train, cv=5)

# 4. Report
evaluator.print_evaluation_report(y_test, y_pred)

# Output: All metrics computed and compared


═══════════════════════════════════════════════════════════════════════════════
WHAT DOES EACH RESULT MEAN?
═══════════════════════════════════════════════════════════════════════════════

SCENARIO: Your Code Output
─────────────────────────────

Test Set Metrics:
  MSE:  100.0
  RMSE: 10.0        ← Use this to report to stakeholders
  MAE:  8.0
  R²:   0.75

Baseline Metrics:
  MSE:  400.0
  RMSE: 20.0
  MAE:  15.0
  R²:   0.0         ← Should always be ~0 (by definition)

Comparison:
  MSE improved by 75%       → Good improvement in absolute terms
  RMSE improved by 50%      → Typical error cut in half
  R² improved by 0.75       → Model explains 75% of variance

Cross-Validation R² (cv=5):
  Fold scores: [0.74, 0.76, 0.73, 0.75, 0.77]
  Mean: 0.75 ± 0.015      ← Low std dev = STABLE model ✓

INTERPRETATION:
  "Model explains 75% of variance (R²=0.75) with typical error of 10 
   units (RMSE=10). Cross-validation shows stable performance. Ready 
   for deployment."


═══════════════════════════════════════════════════════════════════════════════
DIAGNOSIS TABLE: When Something Seems Wrong
═══════════════════════════════════════════════════════════════════════════════

Observation                     Likely Cause            Solution
─────────────────────────────────────────────────────────────────────────────

R² is negative                  Model worse than mean   🚨 Stop. Investigate:
                                baseline                 → Data leakage?
                                                         → Train/test shift?
                                                         → Bug in pipeline?

High train R², low test R²      Overfitting             ⚠ Simplify model or
                                                        get more data

High CV std dev                 Unstable model or       ⚠ Check for
(e.g. 0.75 ± 0.35)            outliers                overfitting or
                                                        strange subsets

Some CV folds have R² < 0       Model fails on some     ⚠ Check data quality
                                data subsets            in those subsets

RMSE much larger than MAE       Outliers in data        Consider MAE instead
                                                        or investigate outliers

R² changes significantly        Different data          ⚠ Check for
between folds                   distributions in       important features
                                different folds        missing in some

Model worse than baseline       (see "R² negative")     🚨 Stop. Investigate.


═══════════════════════════════════════════════════════════════════════════════
COMMON QUESTIONS & ANSWERS
═══════════════════════════════════════════════════════════════════════════════

Q: Should I report MSE or RMSE?
A: RMSE. MSE has squared units that mean nothing to stakeholders.
   Use MSE internally, RMSE in reports.

Q: What's a "good" R²?
A: Depends on your domain. 0.7+ is usually good. 0.5+ is fair.
   Always compare against baseline.

Q: What if baseline R² is not 0?
A: Something is wrong. Baseline R² should always be ≈0 on test set.
   Check your train/test split.

Q: How many CV folds should I use?
A: Minimum 5. 10 is better. 5-fold is the default here.

Q: What if R² is negative?
A: RED FLAG. Model is worse than guessing the mean.
   Stop and investigate: data leakage? train/test shift? bugs?

Q: How do I know if RMSE is acceptable?
A: Compare to baseline RMSE and business tolerance.
   If baseline RMSE=100 and yours is 20, that's 80% improvement.

Q: My model has high RMSE but high R². Is that good?
A: Possibly. High RMSE indicates large absolute errors, but high R²
   means you've dramatically improved relative to baseline.
   Check if the absolute RMSE is acceptable for your use case.

Q: What does "explain 75% of variance" really mean?
A: Your model accounts for 75% of the ups and downs in the target.
   The remaining 25% is either noise or unexplained by your features.


═══════════════════════════════════════════════════════════════════════════════
EVALUATION WORKFLOW (STEP BY STEP)
═══════════════════════════════════════════════════════════════════════════════

Step 1: Train/Test Split
   → Split RAW DATA first (before any preprocessing)
   → Don't fit scalers on full dataset

Step 2: Train Model
   → Fit on training data only
   → Never peek at test data

Step 3: Generate Predictions
   → y_pred = model.predict(X_test)

Step 4: Compute Test Metrics
   → MSE, RMSE, MAE, R² on test set

Step 5: Compare with Baseline
   → Train DummyRegressor(strategy="mean")
   → Compare baseline vs model metrics
   → Baseline R² should be ≈ 0

Step 6: Cross-Validate
   → cv_r2 = cross_val_score(model, X_train, y_train, cv=5)
   → Check mean and std dev
   → Investigate high std dev or negative R²

Step 7: Interpret Results
   → Use interpret_metrics() for guided interpretation
   → Report RMSE to stakeholders
   → Provide context (baseline comparison, CV results)

Step 8: Diagnose Issues (if any)
   → Negative R²? → Check for data leakage
   → High std dev? → Check for overfitting
   → Suspicious metrics? → Verify preprocessing


═══════════════════════════════════════════════════════════════════════════════
KEY FILES & WHERE TO START
═══════════════════════════════════════════════════════════════════════════════

1. See Examples First (5 min):
   src/demo_mse_r2_evaluation.py
   Run with: python -m src.demo_mse_r2_evaluation

2. Understand Theory (1 hour):
   REGRESSION_EVALUATION_GUIDE.md
   Read carefully for deep understanding

3. Use in Your Code (ongoing):
   from src.evaluate_regression_metrics import RegressionMetricsEvaluator

4. Full Integration Example:
   src/integrate_mse_r2_evaluation.py
   Shows complete end-to-end workflow


═══════════════════════════════════════════════════════════════════════════════
THE GOLDEN RULE
═══════════════════════════════════════════════════════════════════════════════

Never report a single metric without context.

WRONG:  "R² = 0.75"
RIGHT:  "R² = 0.75 (baseline R² ≈ 0, RMSE = 10 units)"

WRONG:  "MSE = 100"
RIGHT:  "RMSE = 10 units (4x improvement over baseline)"

WRONG:  "Model is better"
RIGHT:  "Model explains 75% of variance, with CV R² = 0.75 ± 0.03"

Always include:
  ✓ Baseline comparison
  ✓ Cross-validation results
  ✓ Business context
"""
