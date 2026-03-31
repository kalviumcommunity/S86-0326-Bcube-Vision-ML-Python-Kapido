"""
Main orchestration script for the complete ML pipeline.

This is the entry point that coordinates all pipeline stages:
1. Data loading and cleaning
2. Feature engineering
3. Model training
4. Model evaluation
5. Artifact persistence

Execution flow:
    python -m src.main  (from project root)
or
    python src/main.py  (from project root)

Each stage is encapsulated in separate modules. If any stage fails,
the entire pipeline stops and an error is logged.
"""
import logging
import sys
import os
from typing import Dict, Any

# Configure logging at module level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/pipeline.log')
    ]
)

logger = logging.getLogger(__name__)

# Now import all modules (safe after logging is configured)
from src.config import (
    DATA_PATH, TARGET_COLUMN, CATEGORICAL_COLS, NUMERICAL_COLS,
    TEST_SIZE, RANDOM_STATE, N_ESTIMATORS, MAX_DEPTH,
    MODEL_PATH, PIPELINE_PATH, REPORT_PATH
)
from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import build_preprocessing_pipeline
from src.train import train_model
from src.evaluate import evaluate_model
from src.persistence import save_artifacts


def create_directories() -> None:
    """Create required directories if they don't exist."""
    dirs = ['data/raw', 'data/processed', 'models', 'reports', 'logs']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Directory ready: {dir_path}")


def main() -> Dict[str, Any]:
    """
    Execute the complete ML pipeline.
    
    Returns:
        Dictionary with final metrics and artifact paths.
        
    Raises:
        Exception: If any pipeline stage fails.
    """
    try:
        logger.info("=" * 80)
        logger.info("STARTING ML PIPELINE")
        logger.info("=" * 80)
        
        # Create required directories
        create_directories()
        
        # ====================================================================
        # STAGE 1: LOAD AND CLEAN DATA
        # ====================================================================
        logger.info("STAGE 1: Loading and cleaning data")
        df = load_data(DATA_PATH)
        df_clean = clean_data(df)
        logger.info(f"Data shape after cleaning: {df_clean.shape}")
        
        # ====================================================================
        # STAGE 2: SPLIT DATA
        # ====================================================================
        logger.info("STAGE 2: Splitting data into train/test")
        X_train, X_test, y_train, y_test = split_data(
            df_clean,
            target_column=TARGET_COLUMN,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        
        # ====================================================================
        # STAGE 3: BUILD PREPROCESSING PIPELINE (UNFITTED)
        # ====================================================================
        logger.info("STAGE 3: Building preprocessing pipeline")
        pipeline = build_preprocessing_pipeline(CATEGORICAL_COLS, NUMERICAL_COLS)
        
        # ====================================================================
        # STAGE 4: FIT PIPELINE ON TRAINING DATA
        # ====================================================================
        logger.info("STAGE 4: Fitting preprocessing pipeline on training data")
        X_train_processed = pipeline.fit_transform(X_train)
        logger.info(f"Training data shape after preprocessing: {X_train_processed.shape}")
        
        # ====================================================================
        # STAGE 5: APPLY PIPELINE TO TEST DATA (NO REFITTING)
        # ====================================================================
        logger.info("STAGE 5: Applying pipeline to test data")
        X_test_processed = pipeline.transform(X_test)
        logger.info(f"Test data shape after preprocessing: {X_test_processed.shape}")
        
        # ====================================================================
        # STAGE 6: TRAIN MODEL
        # ====================================================================
        logger.info("STAGE 6: Training model")
        model = train_model(
            X_train_processed,
            y_train,
            random_state=RANDOM_STATE,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH
        )
        
        # ====================================================================
        # STAGE 7: EVALUATE MODEL
        # ====================================================================
        logger.info("STAGE 7: Evaluating model on test data")
        metrics = evaluate_model(model, X_test_processed, y_test)
        
        # Log all metrics
        logger.info("-" * 80)
        logger.info("EVALUATION METRICS")
        logger.info("-" * 80)
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name.upper():15s}: {metric_value:.4f}")
        logger.info("-" * 80)
        
        # ====================================================================
        # STAGE 8: SAVE ARTIFACTS
        # ====================================================================
        logger.info("STAGE 8: Saving trained artifacts")
        save_artifacts(model, pipeline, MODEL_PATH, PIPELINE_PATH)
        logger.info(f"Model saved to: {MODEL_PATH}")
        logger.info(f"Pipeline saved to: {PIPELINE_PATH}")
        
        # ====================================================================
        # PIPELINE COMPLETE
        # ====================================================================
        logger.info("=" * 80)
        logger.info("ML PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return {
            'status': 'success',
            'metrics': metrics,
            'model_path': MODEL_PATH,
            'pipeline_path': PIPELINE_PATH,
            'samples_trained': len(X_train),
            'samples_tested': len(X_test)
        }
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please ensure data file exists at: " + DATA_PATH)
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid data or configuration: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    result = main()
    logger.info(f"Pipeline result: {result}")

