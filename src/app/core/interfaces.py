"""Abstract contracts for the urgency triage pipeline.

Defines the Strategy-pattern interfaces that concrete implementations
(TF-IDF preprocessing, Logistic Regression classification, etc.) must follow.
Keeping these as abstractions allows swapping implementations (e.g. TF-IDF
for embeddings, Logistic Regression for another model) without touching the
API or training script.
"""

from abc import ABC, abstractmethod
from typing import Any


class TextPreprocessor(ABC):
    """Contract for turning raw text into model-ready features."""

    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        """Learns preprocessing parameters (e.g. vocabulary) from training texts.

        Args:
            texts: Raw training texts (e.g. train["medical_abstract"]).
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, texts: list[str]) -> Any:
        """Transforms texts into model-ready features using what was learned in fit.

        Args:
            texts: Raw texts to transform. A single text must be wrapped in a
                list of length 1 (e.g. [text]).

        Returns:
            Feature representation ready to be consumed by an UrgencyClassifier.
        """
        raise NotImplementedError


class UrgencyClassifier(ABC):
    """Contract for training and using an urgency classification model."""

    @abstractmethod
    def fit(self, features: Any, labels: list[str]) -> None:
        """Trains the classifier on already-preprocessed features.

        Args:
            features: Feature representation produced by a TextPreprocessor.
            labels: Ground-truth urgency labels ("urgent", "attention", "normal").
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, features: Any) -> list[str]:
        """Predicts urgency labels for already-preprocessed features.

        Args:
            features: Feature representation produced by a TextPreprocessor.

        Returns:
            Predicted urgency labels, in the same order as the input features.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, features: Any) -> list[list[float]]:
        """Predicts probabilities for each urgency label.

        Args:
            features: Feature representation produced by a TextPreprocessor.

        Returns:
            Predicted probabilities for each urgency label, in the same order
            as the input features.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persists the trained model to disk.

        Args:
            path: Filesystem path where the model artifact should be written.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Loads a previously trained model from disk.

        Args:
            path: Filesystem path to a model artifact created by save().
        """
        raise NotImplementedError

    @abstractmethod
    def classes(self) -> list[str]:
        """Returns the list of possible urgency classes.

        Returns:
            List of possible urgency classes.
        """
        raise NotImplementedError
