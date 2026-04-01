"""
Data loading module for ride-sharing dataset.

This module handles only the data loading responsibility: reading raw data
from files into memory in a predictable format.

Key principle: This module ONLY loads data. It does not clean, split,
transform, or modify the data in any way. Modifications happen in separate
modules to maintain clear separation of concerns.

Functions here:
- Load raw data from CSV files
- Validate basic file integrity
- Return DataFrame as-is from source
"""
import logging
import pandas as pd

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