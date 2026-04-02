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
# Target definition
TARGET_COLUMN: str = 'ride_completed'

# Numerical features
NUMERICAL_FEATURES: List[str] = [
    'trip_distance',     # Distance of the ride in kilometers
    'estimated_time'     # Estimated duration in minutes
]

# Categorical features
CATEGORICAL_FEATURES: List[str] = [
    'pickup_location',   # Starting location (Downtown, Airport, Suburb, etc.)
    'dropoff_location',  # Destination location
    'hour_of_day',       # Hour when ride was requested (0-23)
    'day_of_week'        # Day of week (Mon, Tue, Wed, etc.)
]

# Excluded columns with reasons
EXCLUDED_COLUMNS: List[str] = [
    # No ID columns in current dataset, but this would be common
    # Example: 'ride_id' - identifier, not predictive feature
]

# Derived feature list (all available features)
ALL_FEATURES: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Legacy aliases for backward compatibility
CATEGORICAL_COLS: List[str] = CATEGORICAL_FEATURES
NUMERICAL_COLS: List[str] = NUMERICAL_FEATURES

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
