# Architecture & Design Patterns

## Project Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: Raw Data                             │
│                    data/raw/ride_data.csv                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: DATA LOADING & CLEANING                                   │
│  data_preprocessing.py                                              │
│  ├── load_data()                                                    │
│  ├── clean_data()     ← Handles missing values                      │
│  └── split_data()     ← Train/test split                            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
          X_train, y_train         X_test, y_test
                │                         │
                ▼                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: FEATURE ENGINEERING                                       │
│  feature_engineering.py                                             │
│  ├── build_preprocessing_pipeline()                                 │
│  │   ├── Categorical: Imputation → OneHotEncoding                  │
│  │   └── Numerical:   Imputation → StandardScaling                 │
│  └── [Returns UNFITTED pipeline for safe fitting]                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
  FIT on X_train          TRANSFORM X_test
  (learn statistics)      (apply statistics)
                │                         │
                ▼                         ▼
    X_train_processed              X_test_processed
                │                         │
                ▼                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: MODEL TRAINING                                            │
│  train.py                                                           │
│  └── train_model()                                                  │
│      └── RandomForestClassifier.fit(X_train_processed, y_train)    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
                       Fitted Model
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         │
┌──────────────────────────┐              │
│  STAGE 4: EVALUATION    │              │
│  evaluate.py            │              │
│  └── evaluate_model()   │              │
│      └── Compute:       │              │
│          • Accuracy     │              │
│          • Precision    │              │
│          • Recall       │              │
│          • F1 Score     │              │
│          • ROC-AUC      │              │
└──────────────────────────┘              │
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 5: PERSISTENCE                                               │
│  persistence.py                                                     │
│  ├── save_artifacts()                                              │
│  │   ├── model.pkl      ← Fitted RandomForest                      │
│  │   └── pipeline.pkl   ← Fitted ColumnTransformer                │
│  └── load_artifacts()                                              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 6: PREDICTION (Isolated from Training)                      │
│  predict.py                                                         │
│  └── predict()                                                      │
│      ├── Load from persistence                                      │
│      ├── pipeline.transform() [NEVER fit_transform]                │
│      ├── model.predict()                                            │
│      └── Return predictions                                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OUTPUT: Predictions                              │
│              {'prediction': [...], 'probability': [...]}           │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Dependency Graph

```
              ┌─────────────────┐
              │   config.py     │  ← All constants and paths
              │  (no deps)      │
              └────────┬────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │  data    │  │ feature  │  │  train   │
   │  preproc │  │  eng     │  │ (trains) │
   └────┬─────┘  └─────┬────┘  └─────┬────┘
        │              │             │
        │              │             ▼
        │              │        ┌──────────┐
        │              │        │ evaluate │
        │              │        │(test eval)
        │              │        └─────┬────┘
        │              │              │
        │              ▼              │
        │         ┌──────────┐        │
        └────────►│persist  │◄───────┘
                  │ (save)  │
                  └────┬─────┘
                       │
                       ▼
                  ┌──────────┐
                  │ predict  │
                  │(load &   │
                  │ apply)   │
                  └──────────┘

KEY PRINCIPLE:
- Arrows point to dependencies
- No circular dependencies (DAG structure)
- Training → Persistence, but Prediction ≠ Training
```

## Data Flow: Training Pipeline

```
Raw CSV
   │
   ▼
load_data()
   │
   ├── Shape: (120, 7)
   ├── Columns: ['pickup_location', 'dropoff_location', ...]
   └── Target: 'ride_completed'
   
   ▼
clean_data()
   │
   ├── Handle missing values
   │   ├── Numerical: median imputation
   │   └── Categorical: mode imputation
   └── Result: No nulls remain
   
   ▼
split_data()
   │
   ├── X_train (80% = 96 rows, 6 features)
   ├── X_test  (20% = 24 rows, 6 features)
   ├── y_train (96 binary labels)
   └── y_test  (24 binary labels)
   
   ▼
build_preprocessing_pipeline()  [UNFITTED]
   │
   ├── Categorical Variables:
   │   ├── SimpleImputer(strategy='most_frequent')
   │   └── OneHotEncoder(drop='first', handle_unknown='ignore')
   │
   └── Numerical Variables:
       ├── SimpleImputer(strategy='median')
       └── StandardScaler()
   
   ▼
pipeline.fit_transform(X_train)  ← FIT ON TRAINING DATA
   │
   ├── Input:  (96, 6) → after preprocessing → (96, n_features)
   ├── Step 1: Impute
   ├── Step 2: Encode categorical (creates dummies)
   ├── Step 3: Scale numerical (zero mean, unit variance)
   └── Output: X_train_processed (96, n_features) - numerical only
   
   ▼
pipeline.transform(X_test)  ← APPLY FITTED TRANSFORMATION
   │
   ├── Input:  (24, 6)
   ├── Apply fitted imputation
   ├── Apply fitted encoding (using X_train categories)
   ├── Apply fitted scaling (using X_train statistics)
   └── Output: X_test_processed (24, n_features)
   
   ▼
train_model(X_train_processed, y_train)
   │
   ├── Create: RandomForestClassifier(n_estimators=100, ...)
   ├── Fit on: X_train_processed, y_train
   ├── Learn: Tree structures and feature importance
   └── Return: Fitted model
   
   ▼
evaluate_model(model, X_test_processed, y_test)
   │
   ├── Predict: model.predict(X_test_processed)
   ├── Compute metrics:
   │   ├── Accuracy = correct / total
   │   ├── Precision = TP / (TP + FP)
   │   ├── Recall = TP / (TP + FN)
   │   ├── F1 = 2 * (Precision * Recall) / (Precision + Recall)
   │   └── ROC-AUC = area under ROC curve
   └── Return: {metric_name: value}
   
   ▼
save_artifacts(model, pipeline, model_path, pipeline_path)
   │
   ├── Save model → models/model.pkl (joblib format)
   └── Save pipeline → models/preprocessing_pipeline.pkl
```

## Data Flow: Prediction Pipeline

```
New Data (Not Seen During Training)
   │
   ▼
predict(new_data, model, pipeline)
   │
   ├── validate_input(new_data)
   │   ├── Check: not None
   │   ├── Check: is DataFrame
   │   └── Check: not empty
   
   ▼
pipeline.transform(new_data)  ← CRITICAL: Never fit_transform()
   │
   ├── Input:  (n, 6)
   ├── Apply preprocessing learned from TRAINING data:
   │   ├── Impute using X_train statistics
   │   ├── Encode using X_train categories
   │   └── Scale using X_train mean/std
   └── Output: new_data_processed (n, n_features)
   
   ▼
model.predict(new_data_processed)
   │
   ├── Input: Features in same space as training
   ├── Apply: Learned tree logic
   └── Output: Binary predictions (0 or 1)
   
   ▼
model.predict_proba(new_data_processed)  [Optional]
   │
   ├── Input: Features
   ├── Apply: Tree probabilities
   └── Output: Probability estimates (0.0 to 1.0)
   
   ▼
Return DataFrame
   │
   ├── Column 'prediction': [0, 1, 1, 0, ...]
   └── Column 'probability': [0.23, 0.89, 0.76, 0.15, ...]
```

## Dependency Hierarchy

This ensures no circular imports and clear module relationships:

```
Level 0 (No dependencies):
└── config.py

Level 1 (Depends only on Level 0):
├── data_preprocessing.py → config
├── feature_engineering.py → config
└── persistence.py → config

Level 2 (Depends on Levels 0-1):
├── train.py → config, feature_engineering
└── evaluate.py → config

Level 3 (Depends on Levels 0-2):
├── main.py → all modules above

Level 4 (Isolated):
└── predict.py → config, persistence, (NOT train or evaluate)

Test Layer:
└── test_pipeline.py → all modules (but mocks data)
```

## Data Leakage Prevention Architecture

### ❌ Bad Design (Data Leakage Risk)
```
predict.py
  ├── import train.py
  ├── load_raw_data()
  ├── fit_preprocessing()  ← REFITTING ON NEW DATA!
  ├── train_model()        ← TRAINING ON NEW DATA!
  └── predict()            ← Using info from NEW DATA in model training

Problem: New data is used to train the model, making predictions invalid
```

### ✅ Good Design (Our Implementation)
```
Training Flow:
  train.py
    ├── Load raw data
    ├── fit_transform() on X_train
    └── Save pipeline with X_train statistics

Prediction Flow:
  predict.py
    ├── Load saved pipeline (fitted on X_train)
    ├── transform() on new data (applies X_train statistics)
    └── predict() (no new training)

Result: New data only influences predictions, never the model
```

## Testing Architecture

```
test_pipeline.py Structure:

┌────────────────────────────────────┐
│         Fixtures                   │
├────────────────────────────────────┤
│ @pytest.fixture                    │
│ def sample_data():                 │
│   └── mock data (no file I/O)     │
│                                    │
│ @pytest.fixture                    │
│ def sample_data_with_nulls():      │
│   └── mock data with missing values│
└────────────────────────────────────┘
         │
         ├─────────────────────────────────────────┐
         ▼                                         ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│ TestDataPreprocessing    │    │ TestFeatureEngineering   │
├──────────────────────────┤    ├──────────────────────────┤
│ test_clean_removes_nulls │    │ test_builds_transformer  │
│ test_split_correct_sizes │    │ test_transforms_data     │
│ test_invalid_target      │    └──────────────────────────┘
└──────────────────────────┘
         │
         ├─────────────────────────────────────────┤
         │                                         │
         ▼                                         ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│ TestModelTraining        │    │ TestModelEvaluation      │
├──────────────────────────┤    ├──────────────────────────┤
│ test_returns_classifier  │    │ test_returns_metrics     │
│ test_invalid_inputs      │    │ test_metrics_in_range    │
│ test_length_mismatch     │    └──────────────────────────┘
└──────────────────────────┘
         │
         └─────────────────────────────────────────┐
                                                   │
                                                   ▼
                                      ┌──────────────────────────┐
                                      │ TestPipelineIntegration  │
                                      ├──────────────────────────┤
                                      │ test_complete_pipeline() │
                                      │ (end-to-end test)        │
                                      └──────────────────────────┘

Key Design: Tests use FIXTURES (mock data), never load actual files
```

## Configuration Management

```
config.py: Single Source of Truth

┌─────────────────────────────────────┐
│           CONFIG.PY                 │
├─────────────────────────────────────┤
│                                     │
│  FILE PATHS                         │
│  ├── DATA_PATH                      │
│  ├── MODEL_PATH                     │
│  ├── PIPELINE_PATH                  │
│  └── LOG_PATH                       │
│                                     │
│  DATA CONFIGURATION                 │
│  ├── TARGET_COLUMN                  │
│  ├── CATEGORICAL_COLS               │
│  ├── NUMERICAL_COLS                 │
│                                     │
│  ML HYPERPARAMETERS                 │
│  ├── RANDOM_STATE                   │
│  ├── TEST_SIZE                      │
│  ├── N_ESTIMATORS                   │
│  ├── MAX_DEPTH                      │
│  └── MIN_SAMPLES_SPLIT              │
│                                     │
│  LOGGING CONFIGURATION              │
│  ├── LOG_LEVEL                      │
│  └── LOG_FORMAT                     │
│                                     │
└─────────────────────────────────────┘
         │
    ┌────┼────┬────┬────┬────┐
    │    │    │    │    │    │
    ▼    ▼    ▼    ▼    ▼    ▼
  train evaluate persist data  feature  main
  .py   .py    .py   .py   eng.py    .py

Key Principle: Change in one place affects all modules
Example: Change RANDOM_STATE=123 → all modules use 123
```

## Error Handling Strategy

```
Function Input Validation:
┌──────────────────────────────┐
│  Function(param1, param2)    │
├──────────────────────────────┤
│ if param1 is None:           │
│   raise ValueError(...)      │
│ if not is_valid(param2):     │
│   raise ValueError(...)      │
│ try:                         │
│   result = compute(...)      │
│ except Exception as e:       │
│   log.error(...)             │
│   raise                      │
└──────────────────────────────┘
    │
    ├─ Explicit validation (fail fast)
    ├─ Clear error messages (include context)
    ├─ Logging (for debugging)
    └─ Re-raise (don't silently fail)
```

## Logging Strategy

```
Entry Point → Main → Stages → Modules → Functions

Each Level Logs:

main.py (high level):
  logger.info("STAGE 1: Loading data")
  logger.info("STAGE 2: Training model")
  logger.info("STAGE 3: Saving artifacts")

Module (medium level):
  logger.info(f"Training {len(X_train)} samples")
  logger.debug(f"Feature importances: {importances}")
  logger.warning("Found {n} missing values")

Function (detailed level):
  try:
    result = operation()
    logger.info("Operation successful")
  except Exception as e:
    logger.error(f"Operation failed: {e}")
    raise

Result: logs/pipeline.log contains full execution trace
```

---

## Summary

This architecture ensures:
✅ **Modularity** - Each module is independent and testable
✅ **Reproducibility** - Configuration is centralized and logged
✅ **No Data Leakage** - Training and prediction are isolated
✅ **Error Handling** - Failures are caught and logged clearly
✅ **Maintainability** - Changes are localized to relevant modules
✅ **Extensibility** - New features fit into existing structure

The structure is the foundation. Everything else builds on it.
