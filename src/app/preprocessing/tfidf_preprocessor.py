"""TF-IDF-based implementation of the TextPreprocessor contract."""

from app.core.interfaces import TextPreprocessor
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfPreprocessor(TextPreprocessor):
    """TF-IDF implementation of the TextPreprocessor contract."""

    def __init__(self, min_df: int = 2) -> None:
        """Creates a TfidfVectorizer configured with the project's EDA-validated defaults.

        Args:
            min_df: Minimum number of documents a term must appear in to be
                kept in the vocabulary.
        """
        self.vectorizer = TfidfVectorizer(min_df=min_df, stop_words="english")

    def fit(self, texts: list[str]) -> None:
        """Learns the vocabulary from training texts.

        Args:
            texts: Raw training texts.
        """
        self.vectorizer.fit(texts)

    def transform(self, texts: list[str]) -> csr_matrix:
        """Transforms texts into TF-IDF feature vectors.

        Args:
            texts: Raw texts to transform.

        Returns:
            Sparse TF-IDF matrix.
        """
        return self.vectorizer.transform(texts)
