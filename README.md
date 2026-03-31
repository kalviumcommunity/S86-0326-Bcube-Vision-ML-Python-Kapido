# Project Structure

```
project_root/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── persistence.py
│   ├── predict.py
│   └── main.py
├── requirements.txt
└── README.md
```

## How to Run

1. Place your raw ride data CSV in `data/raw/ride_data.csv` (or update the path in `src/config.py`).
2. Install dependencies:
	```
	pip install -r requirements.txt
	```
3. Run the main pipeline:
	```
	python src/main.py
	```

## Key Modules
- **config.py**: Centralized configuration (paths, columns, params)
- **data_preprocessing.py**: Data loading, cleaning, splitting
- **feature_engineering.py**: Encoding, scaling, feature pipeline
- **train.py**: Model training
- **evaluate.py**: Model evaluation
- **persistence.py**: Save/load model and pipeline
- **predict.py**: Generate predictions on new data

## Contribution
- Refactored for modularity, reproducibility, and reusability.
- Each function is documented and type-annotated.

---

For more details, see code comments and docstrings in each module.
# S86-0326-Bcube-Vision-ML-Python-Kapido