"""
Basic test script to verify modular ML pipeline functions.
"""
from src.config import Config
from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import build_preprocessing_pipeline
from src.train import train_model
from src.evaluate import evaluate_model
from src.persistence import save_artifacts, load_artifacts
import os

# Use a small sample or mock data for testing
import pandas as pd

def test_pipeline():
    # Create mock data
    df = pd.DataFrame({
        'pickup_location': ['A', 'B', 'A', 'C'],
        'dropoff_location': ['X', 'Y', 'X', 'Z'],
        'hour_of_day': [8, 9, 10, 11],
        'day_of_week': ['Mon', 'Tue', 'Wed', 'Thu'],
        'trip_distance': [2.5, 3.0, 1.2, 4.1],
        'estimated_time': [10, 12, 8, 15],
        'ride_completed': [1, 0, 1, 1]
    })
    df_clean = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df_clean, Config.TARGET_COLUMN, 0.5, 0)
    pipeline = build_preprocessing_pipeline(Config.CATEGORICAL_COLS, Config.NUMERICAL_COLS)
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    model = train_model(X_train_processed, y_train, 0, 10)
    metrics = evaluate_model(model, X_test_processed, y_test)
    print('Test metrics:', metrics)
    # Save and load artifacts
    save_artifacts(model, pipeline, 'models/test_model.pkl', 'models/test_pipeline.pkl')
    loaded_model, loaded_pipeline = load_artifacts('models/test_model.pkl', 'models/test_pipeline.pkl')
    assert loaded_model is not None and loaded_pipeline is not None
    print('Artifacts saved and loaded successfully.')

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    test_pipeline()
