"""
Model training function for ride-sharing demand/supply prediction.
"""
from typing import Any
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, random_state: int, n_estimators: int = 100) -> Any:
    """
    Fit a Random Forest model on training data.
    Args:
        X_train: Training features.
        y_train: Training target.
        random_state: Seed for reproducibility.
        n_estimators: Number of trees in the forest.
    Returns:
        Fitted RandomForestClassifier model.
    """
    model = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators)
    model.fit(X_train, y_train)
    return model
