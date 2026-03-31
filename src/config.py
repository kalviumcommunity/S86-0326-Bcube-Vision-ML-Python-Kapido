"""
Centralized configuration for ML pipeline
"""
from typing import List

class Config:
    # File paths
    DATA_PATH: str = 'data/raw/ride_data.csv'
    MODEL_PATH: str = 'models/model.pkl'
    PIPELINE_PATH: str = 'models/pipeline.pkl'
    REPORT_PATH: str = 'reports/metrics.json'

    # Data columns
    TARGET_COLUMN: str = 'ride_completed'
    CATEGORICAL_COLS: List[str] = ['pickup_location', 'dropoff_location', 'hour_of_day', 'day_of_week']
    NUMERICAL_COLS: List[str] = ['trip_distance', 'estimated_time']

    # ML params
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    N_ESTIMATORS: int = 100
