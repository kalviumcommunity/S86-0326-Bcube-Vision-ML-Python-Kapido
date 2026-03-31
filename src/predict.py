"""
Prediction function for ride-sharing demand/supply using saved artifacts.
"""
from typing import Any
import pandas as pd

def predict(new_data: pd.DataFrame, model: Any, pipeline: Any) -> pd.DataFrame:
    """
    Generate predictions on new data using saved model and pipeline.
    Args:
        new_data: DataFrame of new ride data.
        model: Trained model artifact.
        pipeline: Fitted preprocessing pipeline.
    Returns:
        DataFrame with predictions.
    """
    processed = pipeline.transform(new_data)
    preds = model.predict(processed)
    return pd.DataFrame({'prediction': preds})
