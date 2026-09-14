"""Converts a fitted scikit-learn LogisticRegression to ONNX."""

import json

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression

CLASS_LABELS_METADATA_KEY = "class_labels"


def export_sklearn_logistic(classifier: LogisticRegression, output_path: str) -> None:
    """Serializes a fitted LogisticRegression as an ONNX graph.

    Class names are stored in ONNX metadata so inference does not need a sidecar
    file or a live sklearn object.

    Args:
        classifier: Fitted LogisticRegression (same object used for joblib save).
        output_path: Destination path for the ``.onnx`` file.
    """
    n_features = int(classifier.n_features_in_)
    onnx_model = convert_sklearn(
        classifier,
        initial_types=[("features", FloatTensorType([None, n_features]))],
        options={LogisticRegression: {"zipmap": False}},
        target_opset=15,
    )
    metadata = onnx_model.metadata_props.add()
    metadata.key = CLASS_LABELS_METADATA_KEY
    metadata.value = json.dumps(classifier.classes_.tolist())

    with open(output_path, "wb") as file:
        file.write(onnx_model.SerializeToString())
