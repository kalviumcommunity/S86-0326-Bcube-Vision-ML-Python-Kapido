"""
Artifact persistence module for saving and loading model artifacts.

This module handles serialization of trained models and preprocessing
pipelines. Artifacts saved here can be loaded later for prediction
without retraining.

Key principle: Trained models and pipelines are serialized to disk using
joblib. Loading artifacts enables prediction on new data without access
to training code or training data.

Separation: Training code and prediction code have different artifact
requirements:
- Training: saves fitted model and pipeline
- Prediction: loads saved model and pipeline, never saves anything
"""
import logging
import os
from typing import Any, Tuple
import joblib

# Create logger for this module
logger = logging.getLogger(__name__)


def save_artifacts(
    model: Any,
    pipeline: Any,
    model_path: str,
    pipeline_path: str
) -> None:
    """
    Serialize and save trained model and preprocessing pipeline to disk.
    
    Args:
        model: Fitted model object.
        pipeline: Fitted preprocessing pipeline (ColumnTransformer).
        model_path: Path where model will be saved.
        pipeline_path: Path where pipeline will be saved.
        
    Raises:
        ValueError: If model or pipeline is None.
        IOError: If files cannot be written.
    """
    if model is None:
        raise ValueError("Model cannot be None")
    if pipeline is None:
        raise ValueError("Pipeline cannot be None")
    
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
        
        # Save model
        logger.info(f"Saving model to {model_path}")
        joblib.dump(model, model_path)
        
        # Save pipeline
        logger.info(f"Saving preprocessing pipeline to {pipeline_path}")
        joblib.dump(pipeline, pipeline_path)
        
        logger.info("Artifacts saved successfully")
    
    except IOError as e:
        logger.error(f"Error saving artifacts: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while saving artifacts: {e}")
        raise


def load_artifacts(
    model_path: str,
    pipeline_path: str
) -> Tuple[Any, Any]:
    """
    Load saved model and preprocessing pipeline from disk.
    
    Args:
        model_path: Path to saved model file.
        pipeline_path: Path to saved pipeline file.
        
    Returns:
        Tuple of (model, pipeline) loaded from disk.
        
    Raises:
        FileNotFoundError: If artifact files do not exist.
        EOFError: If artifact files are corrupt.
    """
    try:
        # Verify files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        # Load pipeline
        logger.info(f"Loading preprocessing pipeline from {pipeline_path}")
        pipeline = joblib.load(pipeline_path)
        
        logger.info("Artifacts loaded successfully")
        return model, pipeline
    
    except FileNotFoundError as e:
        logger.error(f"Artifact file not found: {e}")
        raise
    except EOFError as e:
        logger.error(f"Artifact file is corrupt: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        raise

