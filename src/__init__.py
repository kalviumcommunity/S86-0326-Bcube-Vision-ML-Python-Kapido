"""
ML Pipeline Package

A production-ready machine learning pipeline for ride-sharing demand prediction.

This package demonstrates professional code organization with separation of
concerns, reproducibility, and maintainability at its foundation.

Key modules:
- config: Centralized configuration
- data_preprocessing: Data loading and cleaning
- preprocessing: Feature transformation pipelines
- train: Model training
- evaluate: Model evaluation
- predict: Prediction on new data
- persistence: Save/load artifacts
"""

__version__ = "1.0.0"
__author__ = "ML Team"

# Make key functions easily importable
from src.config import (
    DATA_PATH, TARGET_COLUMN, CATEGORICAL_COLS, NUMERICAL_COLS,
    RANDOM_STATE, TEST_SIZE, N_ESTIMATORS
)

__all__ = [
    'DATA_PATH',
    'TARGET_COLUMN', 
    'CATEGORICAL_COLS',
    'NUMERICAL_COLS',
    'RANDOM_STATE',
    'TEST_SIZE',
    'N_ESTIMATORS'
]
