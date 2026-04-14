"""
Backward-compatible feature engineering module.

The project now keeps the preprocessing implementation in src/preprocessing.py,
but several docs and older imports still refer to src/feature_engineering.py.
This module re-exports the public preprocessing API so both names continue to
work without duplicating logic.
"""

from .preprocessing import build_preprocessing_pipeline

__all__ = ["build_preprocessing_pipeline"]
