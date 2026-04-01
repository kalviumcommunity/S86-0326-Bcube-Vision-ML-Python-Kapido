"""
Centralized configuration for ML pipeline.

All constants, file paths, hyperparameters, and column definitions are
defined here. Changes to configuration propagate automatically to all
modules that import from this file.

This eliminates magic numbers and hardcoded strings scattered across
the codebase, making the system more maintainable and reproducible.
"""
from typing import List

# ============================================================================
# PROBLEM TYPE CONFIGURATION
# ============================================================================
# This project solves a binary classification problem
# Problem type: supervised learning - classification - binary
# Target: Predict whether a ride-sharing request will be completed (1) or not (0)
PROBLEM_TYPE: str = 'binary_classification'
POSITIVE_CLASS: int = 1  # Completed ride
NEGATIVE_CLASS: int = 0  # Not completed ride

# ============================================================================
# FILE PATHS
# ============================================================================
DATA_PATH: str = 'data/raw/ride_data.csv'
MODEL_PATH: str = 'models/model.pkl'
PIPELINE_PATH: str = 'models/preprocessing_pipeline.pkl'
REPORT_PATH: str = 'reports/metrics.json'
LOG_PATH: str = 'logs/pipeline.log'

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
TARGET_COLUMN: str = 'ride_completed'

CATEGORICAL_COLS: List[str] = [
    'pickup_location',
    'dropoff_location',
    'hour_of_day',
    'day_of_week'
]

NUMERICAL_COLS: List[str] = [
    'trip_distance',
    'estimated_time'
]

# ============================================================================
# MACHINE LEARNING HYPERPARAMETERS
# ============================================================================
# Reproducibility seed (used across all modules)
RANDOM_STATE: int = 42

# Train/test split
TEST_SIZE: float = 0.2

# Random Forest hyperparameters
N_ESTIMATORS: int = 100
MAX_DEPTH: int = 10
MIN_SAMPLES_SPLIT: int = 5
MIN_SAMPLES_LEAF: int = 2

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL: str = 'INFO'
LOG_FORMAT: str = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
