"""
Functions for loading, cleaning, and splitting ride-sharing data.
"""
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw ride-sharing data from a CSV file.
    Args:
        filepath: Path to the CSV file.
    Returns:
        DataFrame containing ride data.
    """
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and format inconsistencies in ride data.
    Args:
        df: Raw ride data DataFrame.
    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()
    # Example: Fill missing numerical values with median
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())
    # Example: Fill missing categorical values with mode
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

def split_data(df: pd.DataFrame, target_column: str, test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    Args:
        df: Cleaned DataFrame.
        target_column: Name of the target column.
        test_size: Proportion for test set.
        random_state: Seed for reproducibility.
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
