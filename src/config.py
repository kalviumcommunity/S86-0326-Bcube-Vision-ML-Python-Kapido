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
TARGET_COLUMN: str = 'ride_completed'  # 0 = not completed, 1 = completed   

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
# FEATURE SCALING CONFIGURATION
# ============================================================================
# SCALING STRATEGY FOR NUMERICAL FEATURES:
#
# This project uses StandardScaler for numerical features:
# 1. All numerical features are on a comparable scale (mean ≈ 0, std ≈ 1)
# 2. Gradient-based optimization remains stable during model training
# 3. Distance-based metrics are not biased toward larger-magnitude features
#
# WHY NOT RANDOM FOREST?
# Random Forest is tree-based and scale-invariant—it splits on feature
# thresholds regardless of numeric magnitude. Scaling is optional but does
# not harm. It is applied for pipeline consistency and numerical stability.
#
# For reference: MinMaxScaler (bounds features to [0,1]) is useful for:
# - Neural networks (bounded activations)
# - Distance-based models (kNN, k-Means, SVM with RBF kernel)
# - Algorithms sensitive to feature magnitude (PCA, anomaly detection)
# MinMaxScaler is avoided here due to outlier sensitivity.
#
# DATA LEAKAGE PREVENTION:
# Critical workflow: Scale AFTER splitting, not before.
# 1. Split data into train/test FIRST (using full dataset statistics is leakage)
# 2. Fit StandardScaler on training data ONLY
# 3. Transform both train and test using the fitted scaler
# 4. Save the fitted scaler for inference on new data
#
# This is enforced in preprocessing.py's build_preprocessing_pipeline() and
# train.py's train_model() which both follow the correct sequence.

# Scaling verification: Enable/disable verification output
VERIFY_SCALING: bool = True

# Tolerance thresholds for StandardScaler verification
SCALING_MEAN_TOLERANCE: float = 0.1    # Accept mean within ±0.1 (ideally ≈ 0)
SCALING_STD_TOLERANCE: float = 0.2     # Accept std within ±0.2 from 1.0 (ideally ≈ 1)

# Normalization verification: Enable/disable verification output for MinMaxScaler
VERIFY_NORMALIZATION: bool = False

# Tolerance thresholds for MinMaxScaler verification
NORMALIZATION_MIN_TOLERANCE: float = 0.01  # Accept min values near 0
NORMALIZATION_MAX_TOLERANCE: float = 0.01  # Accept max values near 1

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL: str = 'INFO'
LOG_FORMAT: str = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
