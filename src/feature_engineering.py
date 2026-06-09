"""
Feature Engineering Module
===========================

Wraps scikit-learn's :class:`TfidfVectorizer` into a reusable
:class:`FeatureEngineer` class with persistence, vocabulary diagnostics,
and convenience helpers for the Fake News Detection pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """TF-IDF feature extractor with persistence and analysis helpers.

    Parameters
    ----------
    max_features : int, default 50_000
        Maximum vocabulary size.
    ngram_range : tuple, default (1, 2)
        Unigram + bigram features.
    sublinear_tf : bool, default True
        Apply sub-linear TF scaling (``1 + log(tf)``).
    min_df : int, default 2
        Ignore terms that appear in fewer than *min_df* documents.
    max_df : float, default 0.95
        Ignore terms that appear in more than 95 % of documents.

    Examples
    --------
    >>> fe = FeatureEngineer()
    >>> X_train = fe.fit_transform(train_texts)
    >>> X_test  = fe.transform(test_texts)
    >>> fe.save("models/tfidf_vectorizer.joblib")
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        min_df: int = 2,
        max_df: float = 0.95,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.max_df = max_df

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            min_df=self.min_df,
            max_df=self.max_df,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",
        )
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit_transform(self, texts: pd.Series) -> spmatrix:
        """Fit the vectorizer on *texts* and return the TF-IDF matrix.

        Parameters
        ----------
        texts : pd.Series
            Cleaned text strings.

        Returns
        -------
        scipy.sparse.spmatrix
            Sparse TF-IDF feature matrix of shape ``(n_docs, n_features)``.
        """
        logger.info(
            "Fitting TF-IDF (max_features=%d, ngrams=%s) on %d documents …",
            self.max_features,
            self.ngram_range,
            len(texts),
        )
        X = self._vectorizer.fit_transform(texts.fillna(""))
        self._is_fitted = True
        logger.info(
            "TF-IDF matrix: %d docs × %d features (%.2f MB nnz data).",
            X.shape[0],
            X.shape[1],
            X.data.nbytes / 1e6,
        )
        return X

    def transform(self, texts: pd.Series) -> spmatrix:
        """Transform new *texts* using the already-fitted vectorizer.

        Parameters
        ----------
        texts : pd.Series
            Cleaned text strings.

        Returns
        -------
        scipy.sparse.spmatrix
            Sparse TF-IDF feature matrix.

        Raises
        ------
        RuntimeError
            If the vectorizer has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureEngineer has not been fitted. "
                "Call fit_transform() first or load a saved vectorizer."
            )
        logger.info("Transforming %d documents …", len(texts))
        return self._vectorizer.transform(texts.fillna(""))

    # ------------------------------------------------------------------
    # Feature inspection
    # ------------------------------------------------------------------

    def get_feature_names(self) -> List[str]:
        """Return the list of feature (token) names in vocabulary order."""
        if not self._is_fitted:
            raise RuntimeError("Vectorizer is not fitted yet.")
        return list(self._vectorizer.get_feature_names_out())

    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """Return the *n* features with the highest mean IDF score.

        Parameters
        ----------
        n : int, default 20
            Number of top features to return.

        Returns
        -------
        list of (feature_name, idf_score) tuples, sorted descending.
        """
        if not self._is_fitted:
            raise RuntimeError("Vectorizer is not fitted yet.")

        feature_names = self.get_feature_names()
        idf_scores: np.ndarray = self._vectorizer.idf_

        indices = np.argsort(idf_scores)[::-1][:n]
        top = [(feature_names[i], float(idf_scores[i])) for i in indices]
        logger.info("Top %d features by IDF: %s", n, [f[0] for f in top])
        return top

    # ------------------------------------------------------------------
    # Vocabulary analysis
    # ------------------------------------------------------------------

    def vocabulary_analysis(self, texts: pd.Series) -> Dict[str, Any]:
        """Return summary statistics about the corpus vocabulary.

        Parameters
        ----------
        texts : pd.Series
            The (cleaned) text corpus.

        Returns
        -------
        dict
            Keys: ``vocab_size``, ``avg_doc_length``, ``median_doc_length``,
            ``max_doc_length``, ``min_doc_length``, ``total_tokens``,
            ``num_documents``.
        """
        doc_lengths = texts.fillna("").str.split().apply(len)
        stats: Dict[str, Any] = {
            "vocab_size": len(self._vectorizer.vocabulary_) if self._is_fitted else None,
            "num_documents": len(texts),
            "total_tokens": int(doc_lengths.sum()),
            "avg_doc_length": float(doc_lengths.mean()),
            "median_doc_length": float(doc_lengths.median()),
            "max_doc_length": int(doc_lengths.max()),
            "min_doc_length": int(doc_lengths.min()),
        }
        logger.info("Vocabulary analysis: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the fitted vectorizer to disk using joblib.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``models/tfidf.joblib``).
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted vectorizer.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._vectorizer, path)
        logger.info("Vectorizer saved to %s", path)

    def load(self, path: str | Path) -> "FeatureEngineer":
        """Load a previously saved vectorizer.

        Parameters
        ----------
        path : str or Path
            File path to the joblib-serialized vectorizer.

        Returns
        -------
        FeatureEngineer
            ``self``, with the loaded vectorizer ready for ``transform()``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {path}")
        self._vectorizer = joblib.load(path)
        self._is_fitted = True
        logger.info("Vectorizer loaded from %s", path)
        return self
