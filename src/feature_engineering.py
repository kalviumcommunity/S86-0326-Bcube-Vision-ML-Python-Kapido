"""
Functions for encoding, scaling, and feature engineering for ride-sharing data.
"""
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessing_pipeline(categorical_cols: List[str], numerical_cols: List[str]) -> Pipeline:
    """
    Construct a sklearn Pipeline for encoding categorical and scaling numerical features.
    Args:
        categorical_cols: List of categorical column names.
        numerical_cols: List of numerical column names.
    Returns:
        A fitted sklearn Pipeline object.
    """
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()
    preprocessor = ColumnTransformer([
        ('cat', categorical_transformer, categorical_cols),
        ('num', numerical_transformer, numerical_cols)
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])
    return pipeline
