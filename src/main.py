"""
Orchestration script for ride-sharing demand/supply ML pipeline.
"""
from src.config import Config
from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import build_preprocessing_pipeline
from src.train import train_model
from src.evaluate import evaluate_model
from src.persistence import save_artifacts

import pandas as pd

if __name__ == "__main__":
    # Load and clean data
    df = load_data(Config.DATA_PATH)
    df_clean = clean_data(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        df_clean, Config.TARGET_COLUMN, Config.TEST_SIZE, Config.RANDOM_STATE
    )

    # Build and fit preprocessing pipeline
    pipeline = build_preprocessing_pipeline(Config.CATEGORICAL_COLS, Config.NUMERICAL_COLS)
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    # Train model
    model = train_model(X_train_processed, y_train, Config.RANDOM_STATE, Config.N_ESTIMATORS)

    # Evaluate
    metrics = evaluate_model(model, X_test_processed, y_test)

    # Save artifacts
    save_artifacts(model, pipeline, Config.MODEL_PATH, Config.PIPELINE_PATH)

    # Print metrics
    print(f"Model trained. F1 Score: {metrics['f1']:.3f}")
    print("All metrics:", metrics)
