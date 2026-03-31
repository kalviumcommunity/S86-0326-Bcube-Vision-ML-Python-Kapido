"""
Functions for saving and loading model and pipeline artifacts.
"""
from typing import Any, Tuple
import joblib

def save_artifacts(model: Any, pipeline: Any, model_path: str, pipeline_path: str):
    """
    Serialize and save model and preprocessing pipeline.
    Args:
        model: Trained model.
        pipeline: Fitted pipeline.
        model_path: Path to save model.
        pipeline_path: Path to save pipeline.
    """
    joblib.dump(model, model_path)
    joblib.dump(pipeline, pipeline_path)

def load_artifacts(model_path: str, pipeline_path: str) -> Tuple[Any, Any]:
    """
    Load saved model and preprocessing pipeline.
    Args:
        model_path: Path to saved model.
        pipeline_path: Path to saved pipeline.
    Returns:
        Tuple of (model, pipeline)
    """
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    return model, pipeline
