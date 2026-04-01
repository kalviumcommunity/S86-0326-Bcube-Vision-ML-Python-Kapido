"""
Unit tests for ML pipeline modules.

This test suite verifies that each module works correctly in isolation.
Tests use mock data created in-memory rather than loading actual files.

Execution:
    pytest src/test_pipeline.py -v
or
    pytest src/test_pipeline.py -v --cov=src  (with coverage report)
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.data_preprocessing import load_data, clean_data, split_data
from src.preprocessing import build_preprocessing_pipeline
from src.train import train_model
from src.evaluate import evaluate_model


# ============================================================================
# FIXTURES: Reusable test data
# ============================================================================

@pytest.fixture
def sample_data():
    """Create sample ride-sharing data for testing."""
    return pd.DataFrame({
        'pickup_location': ['A', 'B', 'A', 'C', 'A', 'B'],
        'dropoff_location': ['X', 'Y', 'X', 'Z', 'X', 'Y'],
        'hour_of_day': [8, 9, 10, 11, 12, 13],
        'day_of_week': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
        'trip_distance': [2.5, 3.0, 1.2, 4.1, 2.8, 3.5],
        'estimated_time': [10, 12, 8, 15, 11, 13],
        'ride_completed': [1, 0, 1, 1, 0, 1]
    })


@pytest.fixture
def sample_data_with_nulls():
    """Create sample data with missing values."""
    df = pd.DataFrame({
        'pickup_location': ['A', 'B', None, 'C'],
        'dropoff_location': ['X', None, 'X', 'Z'],
        'hour_of_day': [8, 9, 10, None],
        'day_of_week': ['Mon', 'Tue', 'Wed', 'Thu'],
        'trip_distance': [2.5, None, 1.2, 4.1],
        'estimated_time': [10, 12, 8, None],
        'ride_completed': [1, 0, 1, 1]
    })
    return df


# ============================================================================
# TESTS FOR DATA PREPROCESSING
# ============================================================================

class TestDataPreprocessing:
    """Test data loading, cleaning, and splitting."""
    
    def test_clean_data_removes_nulls(self, sample_data_with_nulls):
        """Test that clean_data removes all missing values."""
        cleaned = clean_data(sample_data_with_nulls)
        assert cleaned.isnull().sum().sum() == 0, "clean_data should remove all nulls"
    
    def test_clean_data_preserves_shape(self, sample_data):
        """Test that clean_data preserves data shape."""
        original_shape = sample_data.shape
        cleaned = clean_data(sample_data)
        assert cleaned.shape[0] == original_shape[0], "clean_data should not drop rows"
    
    def test_split_data_correct_sizes(self, sample_data):
        """Test that split_data produces correct train/test sizes."""
        X_train, X_test, y_train, y_test = split_data(
            sample_data, 'ride_completed', test_size=0.33, random_state=42
        )
        total_samples = len(X_train) + len(X_test)
        assert total_samples == len(sample_data), "All samples should be used"
        assert len(X_train) == len(y_train), "X_train and y_train mismatch"
        assert len(X_test) == len(y_test), "X_test and y_test mismatch"
    
    def test_split_data_invalid_target(self, sample_data):
        """Test that split_data raises error for missing target column."""
        with pytest.raises(ValueError):
            split_data(sample_data, 'nonexistent_column')
    
    def test_split_data_invalid_test_size(self, sample_data):
        """Test that split_data raises error for invalid test_size."""
        with pytest.raises(ValueError):
            split_data(sample_data, 'ride_completed', test_size=1.5)


# ============================================================================
# TESTS FOR FEATURE ENGINEERING
# ============================================================================

class TestFeatureEngineering:
    """Test preprocessing pipeline construction."""
    
    def test_build_pipeline_creates_transformer(self):
        """Test that build_preprocessing_pipeline returns ColumnTransformer."""
        pipeline = build_preprocessing_pipeline(
            ['pickup_location', 'day_of_week'],
            ['trip_distance', 'estimated_time']
        )
        assert pipeline is not None, "Pipeline should be created"
        assert hasattr(pipeline, 'fit_transform'), "Pipeline should have fit_transform"
    
    def test_build_pipeline_transforms_data(self, sample_data):
        """Test that pipeline transforms data correctly."""
        X = sample_data.drop('ride_completed', axis=1)
        pipeline = build_preprocessing_pipeline(
            ['pickup_location', 'dropoff_location', 'hour_of_day', 'day_of_week'],
            ['trip_distance', 'estimated_time']
        )
        X_transformed = pipeline.fit_transform(X)
        assert X_transformed.shape[0] == X.shape[0], "Row count should match"
        assert X_transformed.shape[1] > 0, "Should produce features"
    
    def test_build_pipeline_empty_columns_raises(self):
        """Test that empty column lists raise error."""
        with pytest.raises(ValueError):
            build_preprocessing_pipeline([], [])


# ============================================================================
# TESTS FOR MODEL TRAINING
# ============================================================================

class TestModelTraining:
    """Test model training."""
    
    def test_train_model_returns_classifier(self, sample_data):
        """Test that train_model returns a fitted classifier."""
        X = sample_data.drop('ride_completed', axis=1)
        y = sample_data['ride_completed']
        
        pipeline = build_preprocessing_pipeline(
            ['pickup_location', 'dropoff_location', 'hour_of_day', 'day_of_week'],
            ['trip_distance', 'estimated_time']
        )
        X_processed = pipeline.fit_transform(X)
        
        model = train_model(X_processed, y, random_state=42)
        assert isinstance(model, RandomForestClassifier), "Should return RandomForestClassifier"
    
    def test_train_model_invalid_inputs(self):
        """Test that train_model raises error for invalid inputs."""
        with pytest.raises(ValueError):
            train_model(None, np.array([1, 0, 1]))
        
        with pytest.raises(ValueError):
            train_model(np.array([[1, 2], [3, 4]]), None)
    
    def test_train_model_length_mismatch(self):
        """Test that train_model raises error for length mismatch."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 0])  # Wrong length
        
        with pytest.raises(ValueError):
            train_model(X, y)


# ============================================================================
# TESTS FOR MODEL EVALUATION
# ============================================================================

class TestModelEvaluation:
    """Test model evaluation."""
    
    def test_evaluate_model_returns_metrics(self, sample_data):
        """Test that evaluate_model returns expected metrics."""
        X = sample_data.drop('ride_completed', axis=1)
        y = sample_data['ride_completed']
        
        pipeline = build_preprocessing_pipeline(
            ['pickup_location', 'dropoff_location', 'hour_of_day', 'day_of_week'],
            ['trip_distance', 'estimated_time']
        )
        X_processed = pipeline.fit_transform(X)
        
        model = train_model(X_processed, y, random_state=42)
        metrics = evaluate_model(model, X_processed, y)
        
        assert isinstance(metrics, dict), "Should return dictionary"
        assert 'accuracy' in metrics, "Should have accuracy metric"
        assert 'precision' in metrics, "Should have precision metric"
        assert 'recall' in metrics, "Should have recall metric"
        assert 'f1' in metrics, "Should have f1 metric"
    
    def test_evaluate_model_metrics_in_range(self, sample_data):
        """Test that metrics are in valid ranges."""
        X = sample_data.drop('ride_completed', axis=1)
        y = sample_data['ride_completed']
        
        pipeline = build_preprocessing_pipeline(
            ['pickup_location', 'dropoff_location', 'hour_of_day', 'day_of_week'],
            ['trip_distance', 'estimated_time']
        )
        X_processed = pipeline.fit_transform(X)
        
        model = train_model(X_processed, y, random_state=42)
        metrics = evaluate_model(model, X_processed, y)
        
        # All metrics should be between 0 and 1
        for metric_name, metric_value in metrics.items():
            if not np.isnan(metric_value):
                assert 0 <= metric_value <= 1, f"{metric_name} should be between 0 and 1"
    
    def test_evaluate_model_invalid_inputs(self):
        """Test that evaluate_model raises error for invalid inputs."""
        model = RandomForestClassifier()
        
        with pytest.raises(ValueError):
            evaluate_model(None, np.array([[1, 2]]), np.array([1]))
        
        with pytest.raises(ValueError):
            evaluate_model(model, None, np.array([1]))


# ============================================================================
# INTEGRATION TEST
# ============================================================================

class TestPipelineIntegration:
    """Test the complete pipeline end-to-end."""
    
    def test_complete_pipeline(self, sample_data):
        """Test complete pipeline from data to evaluation."""
        # Prepare data
        df_clean = clean_data(sample_data)
        X_train, X_test, y_train, y_test = split_data(
            df_clean, 'ride_completed', test_size=0.33, random_state=42
        )
        
        # Build and fit pipeline
        pipeline = build_preprocessing_pipeline(
            ['pickup_location', 'dropoff_location', 'hour_of_day', 'day_of_week'],
            ['trip_distance', 'estimated_time']
        )
        X_train_processed = pipeline.fit_transform(X_train)
        X_test_processed = pipeline.transform(X_test)
        
        # Train model
        model = train_model(X_train_processed, y_train, random_state=42)
        
        # Evaluate
        metrics = evaluate_model(model, X_test_processed, y_test)
        
        # Verify results
        assert isinstance(metrics, dict), "Should return metrics dictionary"
        assert 'f1' in metrics, "Should have f1 score"
        assert 0 <= metrics['f1'] <= 1, "F1 score should be valid"


if __name__ == "__main__":
    # Run tests with: pytest src/test_pipeline.py -v
    pytest.main([__file__, '-v'])

