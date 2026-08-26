"""Logistic Regression implementation of the UrgencyClassifier contract."""

from typing import Any

import joblib
from app.core.interfaces import UrgencyClassifier
from sklearn.linear_model import LogisticRegression


class LogisticClassifier(UrgencyClassifier):
    """Logistic Regression implementation of the UrgencyClassifier contract."""

    def __init__(self, max_iter: int = 1000, class_weight: str = "balanced") -> None:
        """Creates a LogisticRegression with the project's EDA-validated defaults.

        Args:
            max_iter: Maximum number of solver iterations before giving up.
            class_weight: Class weighting strategy, passed to LogisticRegression.
        """
        self.classifier = LogisticRegression(
            class_weight=class_weight, random_state=42, max_iter=max_iter
        )

    def fit(self, features: Any, labels: list[str]) -> None:
        """Trains the classifier on already-preprocessed features.

        Args:
            features: Feature representation produced by a TextPreprocessor.
            labels: Ground-truth urgency labels.
        """
        self.classifier.fit(features, labels)

    def predict(self, features: Any) -> list[str]:
        """Predicts urgency labels for already-preprocessed features.

        Args:
            features: Feature representation produced by a TextPreprocessor.

        Returns:
            Predicted urgency labels.
        """
        return self.classifier.predict(features).tolist()

    def save(self, path: str) -> None:
        """Saves the trained model to disk."""
        joblib.dump(self.classifier, path)

    def load(self, path: str) -> None:
        """Loads a previously trained model from disk."""
        self.classifier = joblib.load(path)
