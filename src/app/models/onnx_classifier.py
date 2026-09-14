"""ONNX Runtime implementation of the UrgencyClassifier contract."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from app.core.interfaces import UrgencyClassifier

CLASS_LABELS_METADATA_KEY = "class_labels"


class OnnxClassifier(UrgencyClassifier):
    """Runs a LogisticRegression exported to ONNX via ONNX Runtime (CPU)."""

    def __init__(self) -> None:
        """Creates an empty wrapper; call ``load`` before inference."""
        self._session: ort.InferenceSession | None = None
        self._input_name: str = ""
        self._classes: list[str] = []
        self._model_path: str | None = None

    def fit(self, features: Any, labels: list[str]) -> None:
        """Not supported: training stays on ``LogisticClassifier``.

        Args:
            features: Unused.
            labels: Unused.
        """
        raise RuntimeError(
            "OnnxClassifier is inference-only. Train LogisticClassifier and export ONNX."
        )

    def predict(self, features: Any) -> list[str]:
        """Predicts urgency labels for already-preprocessed features.

        Args:
            features: TF-IDF matrix (sparse or dense) with shape (n_samples, n_features).

        Returns:
            Predicted urgency labels.
        """
        labels, _ = self.predict_with_proba(features)
        return labels

    def predict_proba(self, features: Any) -> list[list[float]]:
        """Predicts probabilities for each urgency label.

        Args:
            features: TF-IDF matrix (sparse or dense) with shape (n_samples, n_features).

        Returns:
            Predicted probabilities, one list per sample, aligned with ``classes()``.
        """
        _, probabilities = self.predict_with_proba(features)
        return probabilities

    def predict_with_proba(self, features: Any) -> tuple[list[str], list[list[float]]]:
        """Runs a single ONNX session to obtain labels and probabilities.

        Args:
            features: TF-IDF matrix (sparse or dense) with shape (n_samples, n_features).

        Returns:
            A tuple of (labels, probabilities).
        """
        if self._session is None:
            raise RuntimeError("ONNX model is not loaded. Call load() first.")

        outputs = self._session.run(None, {self._input_name: self._to_float32(features)})
        raw_labels, raw_probabilities = outputs[0], outputs[1]
        labels = [str(label) for label in np.asarray(raw_labels).reshape(-1)]
        probabilities = np.asarray(raw_probabilities, dtype=np.float64).tolist()
        return labels, probabilities

    def save(self, path: str) -> None:
        """Copies the loaded ONNX artifact to ``path``.

        Args:
            path: Destination filesystem path.
        """
        if self._model_path is None:
            raise RuntimeError("ONNX model is not loaded. Call load() before save().")
        shutil.copyfile(self._model_path, path)

    def load(self, path: str) -> None:
        """Loads an ONNX classifier produced by ``export_sklearn_logistic``.

        Args:
            path: Filesystem path to a ``.onnx`` artifact.
        """
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        metadata = session.get_modelmeta().custom_metadata_map
        if CLASS_LABELS_METADATA_KEY not in metadata:
            raise RuntimeError(
                f"ONNX model at {path} is missing metadata '{CLASS_LABELS_METADATA_KEY}'."
            )
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._classes = json.loads(metadata[CLASS_LABELS_METADATA_KEY])
        self._model_path = str(Path(path))

    def classes(self) -> list[str]:
        """Returns the list of possible urgency classes.

        Returns:
            Class names in the same order used by ``predict_proba``.
        """
        return list(self._classes)

    @staticmethod
    def _to_float32(features: Any) -> np.ndarray:
        """Converts sklearn sparse/dense features to a dense float32 array."""
        if hasattr(features, "toarray"):
            return features.toarray().astype(np.float32, copy=False)
        return np.asarray(features, dtype=np.float32)
