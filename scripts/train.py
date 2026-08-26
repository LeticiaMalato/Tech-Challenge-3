"""Training pipeline: fits the TF-IDF preprocessor and Logistic Regression classifier.

Evaluates on validation data and saves both artifacts to disk. Runs once (or
whenever retraining is needed). Reads data/processed/{train,val}.csv, produced
by scripts/prepare_data.py.
"""

import os

import joblib
import pandas as pd
from app.models.logistic_classifier import LogisticClassifier
from app.preprocessing.tfidf_preprocessor import TfidfPreprocessor
from sklearn.metrics import classification_report, f1_score

PROCESSED_DATA_DIR = "data/processed"
MODEL_ARTIFACTS_DIR = "model_artifacts"


def load_split(split_name: str) -> tuple[list[str], list[str]]:
    """Loads texts and labels for a given data split (train/val/test).

    Args:
        split_name: One of "train", "val", "test".

    Returns:
        A tuple of (texts, labels).
    """
    df = pd.read_csv(f"{PROCESSED_DATA_DIR}/{split_name}.csv")
    return df["medical_abstract"].tolist(), df["urgency_label"].tolist()


def main() -> None:
    """Orchestrates the training pipeline: fit, evaluate, save."""
    train_texts, train_labels = load_split("train")
    val_texts, val_labels = load_split("val")

    # Initialize the preprocessor and classifier
    preprocessor = TfidfPreprocessor()
    classifier = LogisticClassifier()

    # Fit the preprocessor and classifier
    preprocessor.fit(train_texts)
    train_features = preprocessor.transform(train_texts)
    classifier.fit(train_features, train_labels)

    # Evaluate on validation data
    val_features = preprocessor.transform(val_texts)
    val_predictions = classifier.predict(val_features)

    # Print evaluation metrics
    print("Validation Results:")
    print(classification_report(val_labels, val_predictions))
    print("Macro F1:", f1_score(val_labels, val_predictions, average="macro"))

    os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)
    classifier.save(f"{MODEL_ARTIFACTS_DIR}/classifier.joblib")
    joblib.dump(preprocessor, f"{MODEL_ARTIFACTS_DIR}/preprocessor.joblib")
    print(f"Artifacts saved to {MODEL_ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
