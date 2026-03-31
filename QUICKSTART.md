# Quick Start Guide

Get this ML pipeline running in 5 minutes.

## Installation

### 1. Clone & Navigate
```bash
cd S86-0326-Bcube-Vision-ML-Python-Kapido
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Project
```bash
python setup_project.py
```
This creates required directories and generates sample data.

## Run the Pipeline

### Train Model
```bash
python -m src.main
```

Output:
```
================================================================================
STARTING ML PIPELINE
================================================================================
...
EVALUATION METRICS
...
F1             : 0.8234
...
ML PIPELINE COMPLETED SUCCESSFULLY
```

Model and preprocessing pipeline saved to `models/`

### Generate Predictions
```bash
python -m src.predict
```

Or in Python:
```python
from src.persistence import load_artifacts
from src.predict import predict
import pandas as pd

model, pipeline = load_artifacts('models/model.pkl', 'models/preprocessing_pipeline.pkl')

new_data = pd.DataFrame({
    'pickup_location': ['Downtown', 'Airport'],
    'dropoff_location': ['Airport', 'Downtown'],
    'hour_of_day': [8, 18],
    'day_of_week': ['Mon', 'Fri'],
    'trip_distance': [15.2, 22.5],
    'estimated_time': [25, 35]
})

predictions = predict(new_data, model, pipeline)
print(predictions)
```

### Run Tests
```bash
# All tests
pytest src/test_pipeline.py -v

# With coverage
pytest src/test_pipeline.py -v --cov=src
```

## Project Structure
```
src/
├── config.py              ← All constants & paths
├── data_preprocessing.py  ← Data loading & cleaning
├── feature_engineering.py ← Encoding & scaling
├── train.py               ← Model training
├── evaluate.py            ← Model evaluation
├── predict.py             ← Prediction on new data
├── persistence.py         ← Save/load artifacts
├── main.py                ← Pipeline orchestration
└── test_pipeline.py       ← Unit tests
```

## Key Concepts

### Separation of Concerns
Each module does ONE thing:
- Preprocessing ≠ Training ≠ Evaluation ≠ Prediction

### Data Leakage Prevention
- Training: `pipeline.fit_transform(X_train)` ✓ Fits on training data
- Prediction: `pipeline.transform(new_data)` ✓ Only applies, never fits

### Configuration
All constants in `src/config.py`:
- Paths
- Data columns
- Hyperparameters
- Random seed

Change once, affects everywhere.

## Modify the Project

### Change Random Seed
Edit `src/config.py`:
```python
RANDOM_STATE = 123
```

### Change Model Algorithm
Edit `src/train.py`:
```python
# Replace RandomForestClassifier with another
from sklearn.ensemble import GradientBoostingClassifier
```

### Add New Evaluation Metric
Edit `src/evaluate.py`:
```python
metrics['new_metric'] = some_metric(y_test, predictions)
```

### Add Preprocessing Step
Edit `src/feature_engineering.py`:
```python
# Add step to pipeline
transformers.append(('new_step', new_transformer, columns))
```

## Troubleshooting

### Missing Data File
```
FileNotFoundError: Data file not found: data/raw/ride_data.csv
```
Run: `python setup_project.py` to create sample data

### Import Errors
```
ModuleNotFoundError: No module named 'src'
```
Run from project root and ensure virtual environment is activated

### Test Failures
```bash
pytest src/test_pipeline.py -v
```
Check the output for which tests fail and why.

## Next Steps

1. **Replace Sample Data**
   - Place your actual CSV in `data/raw/ride_data.csv`
   - Or update `DATA_PATH` in `src/config.py`

2. **Adjust Columns**
   - Update `CATEGORICAL_COLS` and `NUMERICAL_COLS` in `src/config.py`
   - Update `TARGET_COLUMN` if different

3. **Tune Hyperparameters**
   - Edit hyperparameters in `src/config.py`
   - Rerun `python -m src.main`

4. **Add Experiments**
   - Create logged experiment tracking
   - Save multiple models with different configs
   - Compare metrics

5. **Deploy**
   - Use `src/predict.py` as your prediction service
   - Load artifacts and generate predictions in batch or real-time

## Help

- Read `README.md` for comprehensive documentation
- Check `CONTRIBUTING.md` for development guidelines
- Review existing functions as examples
- Run tests: `pytest src/test_pipeline.py -v`

---

**You're ready to go!** 🚀
