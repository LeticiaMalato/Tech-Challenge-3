"""Tests for TfidfPreprocessor."""

from app.preprocessing.tfidf_preprocessor import TfidfPreprocessor
from scipy.sparse import csr_matrix


def test_transform_returns_correct_shape_and_type():
    """transform() should return a csr_matrix with one row per input text."""
    texts = ["texto numero um", "outro texto aqui", "terceiro texto de exemplo"]

    preprocessor = TfidfPreprocessor(min_df=1)
    preprocessor.fit(texts)
    matrix = preprocessor.transform(texts)

    assert isinstance(matrix, csr_matrix)
    assert matrix.shape[0] == len(texts)
