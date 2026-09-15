"""Compares sklearn vs ONNX Runtime inference latency on the same texts.

Does not require the API to be running. Uses artifacts in model_artifacts/.
When data/processed/val.csv exists, uses real validation abstracts; otherwise
falls back to a fixed sample text.
"""

from __future__ import annotations

import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
from app.models.logistic_classifier import LogisticClassifier
from app.models.onnx_classifier import OnnxClassifier

MODEL_ARTIFACTS_DIR = os.getenv("MODEL_ARTIFACTS_DIR", "model_artifacts")
VAL_CSV = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed")) / "val.csv"
SAMPLE_TEXT = "Patient presents with severe chest pain and shortness of breath."
NUM_WARMUP = 20
NUM_RUNS = 200


def _load_texts() -> list[str]:
    """Loads validation abstracts when available, else a repeated sample."""
    if VAL_CSV.exists():
        import pandas as pd

        texts = pd.read_csv(VAL_CSV)["medical_abstract"].astype(str).tolist()
        return texts[:NUM_RUNS]
    return [SAMPLE_TEXT] * NUM_RUNS


def _percentile_ms(durations: list[float], q: float) -> float:
    """Converts a duration list in seconds to a percentile in milliseconds."""
    return float(np.percentile(durations, q) * 1000)


def _summarize(name: str, durations: list[float]) -> dict[str, float]:
    """Prints and returns mean/median/p95 in milliseconds."""
    stats = {
        "mean": statistics.mean(durations) * 1000,
        "median": statistics.median(durations) * 1000,
        "p95": _percentile_ms(durations, 95),
    }
    print(
        f"{name:28}  mean={stats['mean']:7.3f} ms  "
        f"median={stats['median']:7.3f} ms  p95={stats['p95']:7.3f} ms"
    )
    return stats


def _time_calls(fn: Callable[[], None], n_warmup: int, n_runs: int) -> list[float]:
    """Warms up, then times ``n_runs`` calls of ``fn``."""
    for _ in range(n_warmup):
        fn()
    durations: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    return durations


def main() -> None:
    """Times sklearn and ONNX on identical preprocessed inputs and end-to-end."""
    preprocessor = joblib.load(f"{MODEL_ARTIFACTS_DIR}/preprocessor.joblib")
    sklearn_clf = LogisticClassifier()
    sklearn_clf.load(f"{MODEL_ARTIFACTS_DIR}/classifier.joblib")
    onnx_clf = OnnxClassifier()
    onnx_clf.load(f"{MODEL_ARTIFACTS_DIR}/classifier.onnx")

    texts = _load_texts()
    n = min(len(texts), NUM_RUNS)
    texts = texts[:n]
    print(f"Comparing on {n} texts (warmup={NUM_WARMUP})\n")

    features_by_row = [preprocessor.transform([text]) for text in texts]
    cycle = {"i": 0}

    def reset_cycle() -> None:
        cycle["i"] = 0

    def next_features() -> object:
        feats = features_by_row[cycle["i"] % n]
        cycle["i"] += 1
        return feats

    def next_text() -> str:
        text = texts[cycle["i"] % n]
        cycle["i"] += 1
        return text

    def sklearn_classifier_only() -> None:
        feats = next_features()
        sklearn_clf.predict(feats)
        sklearn_clf.predict_proba(feats)

    def onnx_classifier_only() -> None:
        onnx_clf.predict_with_proba(next_features())

    def sklearn_end_to_end() -> None:
        feats = preprocessor.transform([next_text()])
        sklearn_clf.predict(feats)
        sklearn_clf.predict_proba(feats)

    def onnx_end_to_end() -> None:
        feats = preprocessor.transform([next_text()])
        onnx_clf.predict_with_proba(feats)

    reset_cycle()
    sklearn_clf_stats = _summarize(
        "sklearn classifier",
        _time_calls(sklearn_classifier_only, NUM_WARMUP, n),
    )
    reset_cycle()
    onnx_clf_stats = _summarize(
        "ONNX classifier",
        _time_calls(onnx_classifier_only, NUM_WARMUP, n),
    )
    reset_cycle()
    sklearn_e2e_stats = _summarize(
        "sklearn TF-IDF + clf",
        _time_calls(sklearn_end_to_end, NUM_WARMUP, n),
    )
    reset_cycle()
    onnx_e2e_stats = _summarize(
        "TF-IDF + ONNX clf",
        _time_calls(onnx_end_to_end, NUM_WARMUP, n),
    )

    labels_sk = sklearn_clf.predict(preprocessor.transform(texts))
    labels_onnx, _ = onnx_clf.predict_with_proba(preprocessor.transform(texts))
    agreement = sum(a == b for a, b in zip(labels_sk, labels_onnx, strict=True)) / n
    print(f"\nLabel agreement (sklearn vs ONNX): {agreement:.1%} over {n} texts")

    clf_speedup = sklearn_clf_stats["median"] / onnx_clf_stats["median"]
    e2e_speedup = sklearn_e2e_stats["median"] / onnx_e2e_stats["median"]
    print(f"Classifier-only median speedup: {clf_speedup:.2f}x")
    print(f"End-to-end median speedup:      {e2e_speedup:.2f}x")


if __name__ == "__main__":
    main()
