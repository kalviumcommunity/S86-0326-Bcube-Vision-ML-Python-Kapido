"""
Prediction module for generating predictions on new ride-sharing data.

This module handles the prediction stage of the ML pipeline. It is
completely isolated from training code to prevent accidental data leakage.

Key principle: Prediction NEVER refits models or retrains pipelines.
It only loads already-fitted artifacts and applies them to new data.

Isolation: This module imports from persistence (to load artifacts),
config (for paths), but NEVER from train or evaluate modules.
"""
import logging
import sys
from typing import Any
import pandas as pd
import numpy as np

# Create logger for this module
logger = logging.getLogger(__name__)


def validate_input(data: pd.DataFrame) -> None:
    """
    Validate that input data is in expected format.
    
    Args:
        data: DataFrame to validate.
        
    Raises:
        ValueError: If data is invalid.
    """
    if data is None:
        raise ValueError("Input data cannot be None")
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"Input must be DataFrame, got {type(data)}")
    if data.empty:
        raise ValueError("Input DataFrame is empty")
    
    logger.info(f"Input validation passed: {len(data)} rows, {len(data.columns)} columns")


def predict(
    new_data: pd.DataFrame,
    model: Any,
    pipeline: Any
) -> pd.DataFrame:
    """
    Generate predictions on new ride-sharing data.
    
    This function:
    1. Validates input data
    2. Applies the fitted preprocessing pipeline (transform only, no fit)
    3. Generates predictions using the fitted model
    4. Returns predictions with confidence scores
    
    CRITICAL: Does not refit the pipeline or model. Uses only transform()
    to prevent data leakage.
    
    Args:
        new_data: DataFrame of new ride data with feature columns.
        model: Fitted model loaded from persistence.
        pipeline: Fitted preprocessing pipeline loaded from persistence.
        
    Returns:
        DataFrame with 'prediction' and 'probability' columns.
        
    Raises:
        ValueError: If inputs are invalid.
    """
    try:
        validate_input(new_data)
        
        if model is None:
            raise ValueError("Model cannot be None")
        if pipeline is None:
            raise ValueError("Pipeline cannot be None")
        
        logger.info(f"Generating predictions for {len(new_data)} samples")
        
        # Apply preprocessing (transform only, never fit)
        processed = pipeline.transform(new_data)
        logger.debug(f"Preprocessing complete: {processed.shape} output shape")
        
        # Generate predictions
        predictions = model.predict(processed)
        
        # Get prediction probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed)[:, 1]
        
        # Build output DataFrame
        output = pd.DataFrame({
            'prediction': predictions
        })
        
        if probabilities is not None:
            output['probability'] = probabilities
        
        logger.info(f"Prediction complete. Generated {len(output)} predictions")
        return output
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise


def main():
    """
    Example entry point for prediction on new data.
    
    In production, this might:
    - Read data from a database or API
    - Load models from a model registry
    - Write predictions to a storage system
    - Handle batch or streaming prediction
    """
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
        
        from src.persistence import load_artifacts
        from src.config import MODEL_PATH, PIPELINE_PATH
        
        logger.info("Starting prediction pipeline")
        
        # Load artifacts
        model, pipeline = load_artifacts(MODEL_PATH, PIPELINE_PATH)
        logger.info("Model artifacts loaded")
        
        # Create sample data (replace with actual new data in production)
        sample_data = pd.DataFrame({
            'pickup_location': ['A', 'B', 'C'],
            'dropoff_location': ['X', 'Y', 'Z'],
            'hour_of_day': [8, 14, 20],
            'day_of_week': ['Mon', 'Wed', 'Fri'],
            'trip_distance': [2.5, 3.2, 1.8],
            'estimated_time': [10, 15, 8]
        })
        
        # Generate predictions
        predictions = predict(sample_data, model, pipeline)
        logger.info(f"Predictions:\n{predictions}")
        
        return predictions
    
    except Exception as e:
        logger.error(f"Prediction pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

