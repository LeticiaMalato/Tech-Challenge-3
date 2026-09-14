"""Tests for ONNX export and OnnxClassifier inference."""

from pathlib import Path

from app.models.logistic_classifier import LogisticClassifier
from app.models.onnx_classifier import OnnxClassifier
from app.models.onnx_export import export_sklearn_logistic
from app.preprocessing.tfidf_preprocessor import TfidfPreprocessor


def test_onnx_labels_and_probas_match_sklearn(tmp_path: Path):
    """Exported ONNX model should reproduce sklearn labels and probabilities."""
    texts = [
        "chest pain myocardial infarction cardiac arrest",
        "routine follow up laboratory results within normal limits",
        "neoplasm biopsy chemotherapy oncology treatment plan",
        "acute stroke emergency neurological deficit sudden onset",
        "abdominal pain gastritis digestive symptoms outpatient",
        "headache migraine chronic without red flags",
    ]
    labels = ["urgent", "normal", "attention", "urgent", "attention", "normal"]

    preprocessor = TfidfPreprocessor(min_df=1)
    preprocessor.fit(texts)
    features = preprocessor.transform(texts)

    sklearn_clf = LogisticClassifier()
    sklearn_clf.fit(features, labels)

    onnx_path = tmp_path / "classifier.onnx"
    export_sklearn_logistic(sklearn_clf.classifier, str(onnx_path))

    onnx_clf = OnnxClassifier()
    onnx_clf.load(str(onnx_path))

    assert onnx_clf.classes() == sklearn_clf.classes()
    assert onnx_clf.predict(features) == sklearn_clf.predict(features)

    sklearn_probas = sklearn_clf.predict_proba(features)
    _, onnx_probas = onnx_clf.predict_with_proba(features)
    for onnx_row, sklearn_row in zip(onnx_probas, sklearn_probas, strict=True):
        for onnx_p, sklearn_p in zip(onnx_row, sklearn_row, strict=True):
            assert abs(onnx_p - sklearn_p) < 1e-5
