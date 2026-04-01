"""
Data preprocessing module for ride-sharing dataset.

This module handles the first stage of the ML pipeline: loading, cleaning,
and splitting data. Functions here:
- Load raw data from CSV files
- Handle missing values and inconsistencies
- Split data into train and test sets with reproducible randomization

Note: This module ONLY loads and prepares data. Feature engineering
(encoding, scaling) is handled by preprocessing.py.

Key principle: Preprocessing logic is separate and reusable, so changes
to how we clean data propagate consistently to all models.
"""
import logging
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

# Create logger for this module
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw ride-sharing data from a CSV file.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        DataFrame containing ride data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or cannot be parsed.
    """
    try:
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        if df.empty:
            raise ValueError(f"Data file {filepath} is empty.")
        
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    except FileNotFoundError:
        logger.error(f"Data file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and format inconsistencies in ride data.
    
    Strategy:
    - Numerical columns: fill with median (robust to outliers)
    - Categorical columns: fill with mode (most frequent value)
    - Remove rows where target variable is missing
    
    Args:
        df: Raw ride data DataFrame.
        
    Returns:
        Cleaned DataFrame with no missing values.
        
    Raises:
        ValueError: If DataFrame is None or empty.
    """
    if df is None or df.empty:
        raise ValueError("Cannot clean empty or None DataFrame")
    
    logger.info("Starting data cleaning")
    df = df.copy()
    
    # Track missing values before cleaning
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        logger.warning(f"Found {missing_before} missing values in raw data")
    
    # Fill numerical columns with median
    for col in df.select_dtypes(include='number').columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.debug(f"Filled numerical column '{col}' with median: {median_val}")
    
    # Fill categorical columns with mode
    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_val, inplace=True)
            logger.debug(f"Filled categorical column '{col}' with mode: {mode_val}")
    
    missing_after = df.isnull().sum().sum()
    logger.info(f"Data cleaning complete. Remaining missing values: {missing_after}")
    
    return df


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    This ensures reproducibility by using a fixed random_state. Test size
    controls the proportion of data reserved for testing (default 20%).
    
    Args:
        df: Cleaned DataFrame.
        target_column: Name of the target column.
        test_size: Proportion of data for test set (default 0.2).
        random_state: Seed for reproducibility (default 42).
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        
    Raises:
        ValueError: If target_column not in DataFrame or invalid test_size.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    if not (0 < test_size < 1):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    
    logger.info(f"Splitting data: test_size={test_size}, random_state={random_state}")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    logger.info(f"Split complete: {len(X_train)} train samples, {len(X_test)} test samples")
    
    return X_train, X_test, y_train, y_test

