# Ride-Sharing Demand/Supply Prediction ML Project

A professionally structured machine learning pipeline for predicting ride-sharing demand completion. This project demonstrates production-ready code organization with separation of concerns, reproducibility, and maintainability at its core.

## Project Philosophy

This project is built on the principle that **structure is foundational**. In real-world ML systems, clarity of structure is not cosmetic — it is the difference between a model that can be deployed and extended versus one that becomes abandoned.

Key principles:
- **Separation of Concerns**: Each module has one clear responsibility
- **Reproducibility**: Fixed random seeds and configuration management
- **Testability**: All components can be tested in isolation
- **Reusability**: Preprocessing logic shared across training and prediction
- **No Data Leakage**: Training and prediction pipelines are completely isolated

## Problem Type Identification

This project solves a **binary classification** problem in supervised learning.

**What are we predicting?**
- Target variable: `ride_completed` (1 = ride completed successfully, 0 = ride not completed)
- Output type: Discrete category (completed vs not completed)
- Number of classes: 2 (binary classification)

**Business context:**
- Positive class (1): Ride completed - successful match between rider and driver
- Negative class (0): Ride not completed - failed to find driver or other issues
- Success metric: High recall (catch as many failed rides as possible) with acceptable precision

**Why classification, not regression?**
- The outcome is categorical: a ride is either completed or not
- There is no meaningful "degree" of completion - it's binary
- Regression would be inappropriate as it might predict values like 0.7 (nonsensical)

**Evaluation approach:**
- Primary metrics: Precision, Recall, F1-Score, ROC-AUC
- Accuracy alone is misleading due to potential class imbalance
- Focus on recall to minimize missed failed rides (false negatives are costly)

## Directory Structure

```
S86-0326-Bcube-Vision-ML-Python-Kapido/
│
├── data/                          # Data storage
│   ├── raw/                       # Original, immutable data
│   └── processed/                 # Cleaned, transformed data
│
├── models/                        # Saved model and pipeline artifacts
│   ├── model.pkl                  # Trained Random Forest classifier
│   └── preprocessing_pipeline.pkl # Fitted preprocessing pipeline
│
├── reports/                       # Evaluation reports and metrics
│   └── metrics.json               # Test set metrics
│
├── logs/                          # Experiment and pipeline logs
│   └── pipeline.log               # Pipeline execution log
│
├── src/                           # Source code package
│   ├── __init__.py                # Package initialization
│   ├── config.py                  # Centralized configuration & constants
│   ├── data_preprocessing.py      # Data loading, cleaning, splitting
│   ├── feature_engineering.py     # Encoding, scaling, pipelines
│   ├── train.py                   # Model training logic
│   ├── evaluate.py                # Model evaluation metrics
│   ├── predict.py                 # Prediction on new data (isolated)
│   ├── persistence.py             # Save/load artifacts
│   ├── main.py                    # Pipeline orchestration & entry point
│   └── test_pipeline.py           # Unit tests with pytest
│
├── requirements.txt               # Python dependencies (pinned versions)
├── README.md                      # This file
└── .gitignore                     # Git ignore patterns
```

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run the pipeline: `python src/main.py`

## Module Responsibilities

### `config.py`
**Purpose**: Centralized configuration source of truth.

Contains all constants, paths, hyperparameters, and column definitions. This ensures:
- Changes propagate automatically to all modules
- No hardcoded values scattered through the codebase
- Easy reproducibility across machines
- Single point of modification for experiments

### `data_preprocessing.py`
**Purpose**: Data loading, cleaning, and splitting.

Functions:
- `load_data()`: Load raw data from CSV
- `clean_data()`: Handle missing values (median for numerical, mode for categorical)
- `split_data()`: Split into train/test with reproducible random state

**Separation**: Only handles raw data preparation. Feature transformations are in `feature_engineering.py`.

### `feature_engineering.py`
**Purpose**: Build preprocessing transformations (encoding, scaling).

Functions:
- `build_preprocessing_pipeline()`: Create unfitted ColumnTransformer

**Key Design**: Returns an *unfitted* pipeline. Fitting happens during training on training data only. This prevents data leakage.

### `train.py`
**Purpose**: Model training isolated from all other concerns.

Functions:
- `train_model()`: Fit Random Forest on preprocessed training data

**Isolation**: 
- Does NOT import from `evaluate.py` or `predict.py`
- Does NOT handle data loading or feature engineering
- Returns only the fitted model
- Training code is never executed during prediction

### `evaluate.py`
**Purpose**: Model evaluation and metrics computation.

Functions:
- `evaluate_model()`: Compute precision, recall, F1, ROC-AUC on test data

**Separation**: Can evaluate any model, even one loaded from disk. Does not depend on training logic.

### `predict.py`
**Purpose**: Generate predictions on new data without retraining.

Functions:
- `predict()`: Apply fitted pipeline and model to new data
- `main()`: Example entry point for batch prediction

**Critical Design**: 
- Uses `.transform()` only (never fits)
- Loads artifacts from persistence
- Never imports training code
- Architecturally impossible to cause data leakage

### `persistence.py`
**Purpose**: Save and load artifact objects.

Functions:
- `save_artifacts()`: Serialize model and pipeline to disk
- `load_artifacts()`: Load saved artifacts

**Usage**: Training saves artifacts. Prediction loads them.

### `main.py`
**Purpose**: Orchestration entry point that coordinates all stages.

Execution flow:
1. Load and clean data
2. Split into train/test
3. Build preprocessing pipeline
4. Fit pipeline on training data
5. Transform both datasets
6. Train model
7. Evaluate on test data
8. Save artifacts

**Key**: Errors at any stage stop the entire pipeline with clear logging.

### `test_pipeline.py`
**Purpose**: Unit tests for all modules.

Uses pytest with fixtures for mock data. Tests:
- Data preprocessing (null handling, splitting)
- Feature engineering (pipeline construction, transformation)
- Model training (classifier creation, input validation)
- Model evaluation (metrics computation)
- Integration (complete pipeline)

## Installation & Setup

Python required: 3.11.9

### 1. Clone Repository
```bash
git clone <repository_url>
cd S86-0326-Bcube-Vision-ML-Python-Kapido
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

Activate environment:

- Windows (PowerShell/CMD):
```bash
venv\Scripts\activate
```

- macOS/Linux:
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Optional exact lockfile install (includes transitive dependencies):

```bash
pip install -r requirements-frozen.txt
```

### 4. Prepare Data
Place your ride-sharing data CSV at: `data/raw/ride_data.csv`

File must contain columns:
- Categorical: `pickup_location`, `dropoff_location`, `hour_of_day`, `day_of_week`
- Numerical: `trip_distance`, `estimated_time`
- Target: `ride_completed` (binary: 0 or 1)

Or update these in `src/config.py` to match your data.

## Dependency Management and Reproducibility

This project uses strict version pinning in `requirements.txt` with the `==` operator for reproducible ML behavior across machines.

Why this matters:
- Prevents silent behavior changes in preprocessing and model defaults
- Keeps model serialization/loading compatible across environments
- Reduces metric drift caused by dependency upgrades

Recommended workflow:
1. Create and activate a virtual environment
2. Install dependencies from `requirements.txt`
3. Run the pipeline and tests from a clean environment
4. If you add a new library, pin it in `requirements.txt` immediately

Reproducibility check before submission:
1. Delete your local virtual environment
2. Recreate it from scratch
3. Install with `pip install -r requirements.txt`
4. Run full pipeline: `python -m src.main`
5. Run tests: `pytest src/test_pipeline.py -v`

If all steps pass on a clean environment, your dependency management is correct.

## Usage

### Run Complete Pipeline
```bash
python -m src.main
```

This will:
- Load and clean data
- Train model
- Evaluate on test set
- Save artifacts to `models/`
- Log all operations to `logs/pipeline.log`

Output:
```
================================================================================
STARTING ML PIPELINE
================================================================================
STAGE 1: Loading and cleaning data
...
EVALUATION METRICS
--------------------------------------------------------------------------------
ACCURACY       : 0.8234
PRECISION      : 0.7891
RECALL         : 0.8456
F1             : 0.8167
ROC_AUC        : 0.8912
================================================================================
ML PIPELINE COMPLETED SUCCESSFULLY
================================================================================
```

### Generate Predictions on New Data
```bash
python -m src.predict
```

Or in Python:
```python
from src.persistence import load_artifacts
from src.predict import predict
import pandas as pd

# Load saved artifacts
model, pipeline = load_artifacts('models/model.pkl', 'models/preprocessing_pipeline.pkl')

# Create new data
new_rides = pd.DataFrame({
    'pickup_location': ['A', 'B'],
    'dropoff_location': ['X', 'Y'],
    'hour_of_day': [8, 14],
    'day_of_week': ['Mon', 'Wed'],
    'trip_distance': [2.5, 3.2],
    'estimated_time': [10, 15]
})

# Get predictions
predictions = predict(new_rides, model, pipeline)
print(predictions)
```

### Run Tests
```bash
# Run all tests
pytest src/test_pipeline.py -v

# Run with coverage report
pytest src/test_pipeline.py -v --cov=src

# Run specific test class
pytest src/test_pipeline.py::TestDataPreprocessing -v
```

## Key Design Patterns

### 1. Separation of Concerns
**Anti-pattern**:
```python
# Bad: Everything mixed together
def train():
    df = pd.read_csv('data.csv')
    df = df.dropna()
    X = df.drop('target', axis=1)
    y = df['target']
    model = RandomForestClassifier()
    model.fit(X, y)
    # ... more evaluation, saving, plotting
```

**Pattern**:
```python
# Good: Each responsibility isolated
from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import build_preprocessing_pipeline
from src.train import train_model
from src.evaluate import evaluate_model
from src.persistence import save_artifacts

df = load_data('data.csv')               # Load stage
df = clean_data(df)                       # Clean stage
X_train, X_test, y_train, y_test = split_data(df, 'target')  # Split stage
pipeline = build_preprocessing_pipeline(cat_cols, num_cols)   # Feature stage
X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)
model = train_model(X_train, y_train)     # Train stage
metrics = evaluate_model(model, X_test, y_test)  # Evaluate stage
save_artifacts(model, pipeline, 'model.pkl', 'pipeline.pkl')  # Persist stage
```

### 2. Preventing Data Leakage
**Pattern in `train.py`**:
```python
pipeline.fit_transform(X_train)  # FIT on training data only
```

**Pattern in `predict.py`**:
```python
pipeline.transform(new_data)  # TRANSFORM only (never fit)
```

This architectural separation makes data leakage impossible.

### 3. Configuration Management
**Anti-pattern**:
```python
# Bad: Hardcoded everywhere
model = RandomForestClassifier(random_state=42, n_estimators=100)
path_to_model = "/Users/yourname/ML_project/models/model.pkl"
```

**Pattern**:
```python
# Good: Centralized in config.py
from src.config import RANDOM_STATE, N_ESTIMATORS, MODEL_PATH

model = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS)
save_artifacts(model, pipeline, MODEL_PATH, PIPELINE_PATH)
```

### 4. Input Validation
Every function validates inputs:
```python
def train_model(X_train, y_train, ...):
    if X_train is None or len(X_train) == 0:
        raise ValueError("X_train cannot be None or empty")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train length mismatch")
    # ... proceed with training
```

### 5. Comprehensive Logging
All operations are logged:
```python
logger.info(f"Loading data from {filepath}")
logger.warning(f"Found {missing_before} missing values")
logger.error(f"Error during training: {e}")
```

Check `logs/pipeline.log` for detailed execution trace.

## Modifying the Project

### Change Random Seed (For Reproducibility)
Edit `src/config.py`:
```python
RANDOM_STATE = 123  # Change from 42 to 123
```
All modules automatically use the new seed.

### Swap Model Algorithm
Edit `src/train.py`:
```python
from sklearn.ensemble import GradientBoostingClassifier

def train_model(...):
    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model
```
No other modules are affected.

### Add New Evaluation Metric
Edit `src/evaluate.py`:
```python
from sklearn.metrics import precision_recall_curve

def evaluate_model(...):
    metrics = { ... existing metrics ... }
    # Add new metric
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    metrics['pr_auc'] = auc(recall, precision)
    return metrics
```

### Add Cross-Validation
Create `src/validate.py` for cross-validation logic without modifying training.

## Testing Best Practices

All tests are isolated and use mock data:
```python
@pytest.fixture
def sample_data():
    """Create sample data without loading from disk."""
    return pd.DataFrame({...})

def test_clean_data_removes_nulls(sample_data):
    cleaned = clean_data(sample_data)
    assert cleaned.isnull().sum().sum() == 0
```

## Deployment Checklist

- [ ] Update `RANDOM_STATE` in `config.py` to a fixed value
- [ ] Pin all versions in `requirements.txt`
- [ ] Run full test suite: `pytest src/test_pipeline.py -v`
- [ ] Verify data paths in `config.py` match production environment
- [ ] Review logging output in `logs/pipeline.log`
- [ ] Save model and pipeline artifacts
- [ ] Document any environment-specific settings
- [ ] Set up automated retraining schedule if needed

## Troubleshooting

### FileNotFoundError on data
```
Error: Data file not found: data/raw/ride_data.csv
```
**Solution**: Place your CSV file at the correct path or update `DATA_PATH` in `config.py`.

### ValueError during preprocessing
```
Error: Target column 'ride_completed' not found in DataFrame
```
**Solution**: Verify your CSV has the target column or update `TARGET_COLUMN` in `config.py`.

### ImportError when running tests
```
ModuleNotFoundError: No module named 'pytest'
```
**Solution**: Install test dependencies:
```bash
pip install -r requirements.txt
```

## Contributing

1. Create feature branch: `git checkout -b feature/name`
2. Make changes in isolated modules
3. Run tests: `pytest src/test_pipeline.py -v`
4. Ensure all tests pass before committing
5. Commit with clear messages about what changed and why

## License

[Add your license here]

## Contact

[Add contact information]

---

**Remember**: A well-structured model is one that survives iteration, review, extension, and the inevitable moment when someone else — or future you — needs to understand it and modify it safely.


For more details, see code comments and docstrings in each module.
# S86-0326-Bcube-Vision-ML-Python-Kapido