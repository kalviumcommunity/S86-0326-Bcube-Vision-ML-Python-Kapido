"""
Feature engineering module for ride-sharing dataset.

This module handles the second stage of the ML pipeline: transforming
cleaned data into model-ready features through encoding and scaling.

Functions here:
- Define preprocessing strategies (encoding categorical, scaling numerical)
- Build and return a transformer pipeline

Key principle: Preprocessing pipelines are built but NOT fit here. Fitting
happens during training. This ensures the pipeline can be fit on training
data and applied to test/prediction data without data leakage.

Note: Feature engineering is completely separate from model training
(train.py) and prediction (predict.py).
"""
import logging
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Create logger for this module
logger = logging.getLogger(__name__)


def build_preprocessing_pipeline(
    categorical_cols: List[str],
    numerical_cols: List[str]
) -> ColumnTransformer:
    """
    Construct a preprocessing pipeline for categorical and numerical features.
    
    The pipeline includes:
    - Categorical: imputation (mode) + one-hot encoding
    - Numerical: imputation (median) + standardization
    
    This pipeline is unfitted and ready to be fit on training data.
    
    Args:
        categorical_cols: List of categorical column names.
        numerical_cols: List of numerical column names.
        
    Returns:
        ColumnTransformer object (unfitted, ready for fit_transform).
        
    Raises:
        ValueError: If column lists are empty.
    """
    if not categorical_cols and not numerical_cols:
        raise ValueError("Must provide at least one categorical or numerical column")
    
    logger.info(
        f"Building preprocessing pipeline with "
        f"{len(categorical_cols)} categorical and {len(numerical_cols)} numerical columns"
    )
    
    transformers = []
    
    # Categorical transformation: impute mode + one-hot encode
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_cols))
        logger.debug(f"Added categorical transformer for columns: {categorical_cols}")
    
    # Numerical transformation: impute median + standardize
    if numerical_cols:
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numerical_transformer, numerical_cols))
        logger.debug(f"Added numerical transformer for columns: {numerical_cols}")
    
    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(transformers=transformers)
    
    logger.info("Preprocessing pipeline created successfully")
    return preprocessor

