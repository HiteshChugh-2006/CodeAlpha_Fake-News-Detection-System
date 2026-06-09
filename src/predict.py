"""
Prediction / Inference Module
==============================

Provides :class:`NewsPredictor` for classifying unseen news articles as
**FAKE** or **REAL** using a saved model and TF-IDF vectorizer.

Handles both ``predict_proba``-capable models and ``decision_function``-
only models (e.g. raw ``LinearSVC``) by applying a simple sigmoid
normalisation when needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from scipy.sparse import spmatrix

from src.preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)),
    )


class NewsPredictor:
    """End-to-end predictor: raw text → {prediction, confidence, probabilities}.

    Parameters
    ----------
    model_path : str or Path
        Path to a joblib-serialised trained classifier.
    vectorizer_path : str or Path
        Path to a joblib-serialised :class:`TfidfVectorizer`.
    preprocessor : TextPreprocessor or None
        An already-configured preprocessor.  If *None* a default instance
        is created and fitted automatically.
    """

    LABEL_MAP = {0: "REAL", 1: "FAKE"}

    def __init__(
        self,
        model_path: str | Path,
        vectorizer_path: str | Path,
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> None:
        model_path = Path(model_path)
        vectorizer_path = Path(vectorizer_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

        self.model: Any = joblib.load(model_path)
        self.vectorizer: Any = joblib.load(vectorizer_path)
        self.preprocessor: TextPreprocessor = preprocessor or TextPreprocessor()
        self.preprocessor.fit()

        self._feature_names: List[str] = list(
            self.vectorizer.get_feature_names_out()
        )
        logger.info(
            "NewsPredictor ready  (model=%s, vocab=%d features).",
            model_path.name,
            len(self._feature_names),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_probabilities(self, X: spmatrix) -> np.ndarray:
        """Return an (n_samples, 2) probability array.

        Falls back to a sigmoid-mapped ``decision_function`` when
        ``predict_proba`` is unavailable.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)

        if hasattr(self.model, "decision_function"):
            decisions = self.model.decision_function(X)
            prob_pos = _sigmoid(decisions)
            return np.column_stack([1 - prob_pos, prob_pos])

        # Last resort: hard predictions → 0 / 1 probs
        preds = self.model.predict(X)
        return np.column_stack([1 - preds, preds]).astype(float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Dict[str, Any]:
        """Classify a single article.

        Parameters
        ----------
        text : str
            Raw news-article text.

        Returns
        -------
        dict
            ``prediction`` – ``'FAKE'`` or ``'REAL'``
            ``confidence`` – probability of the predicted class
            ``probabilities`` – ``{'REAL': float, 'FAKE': float}``
        """
        if not text or not text.strip():
            return {
                "prediction": "UNKNOWN",
                "confidence": 0.0,
                "probabilities": {"REAL": 0.0, "FAKE": 0.0},
            }

        cleaned = self.preprocessor.clean_text(text)
        X = self.vectorizer.transform([cleaned])

        pred_label = int(self.model.predict(X)[0])
        probs = self._get_probabilities(X)[0]

        result = {
            "prediction": self.LABEL_MAP.get(pred_label, "UNKNOWN"),
            "confidence": float(probs[pred_label]),
            "probabilities": {
                "REAL": float(probs[0]),
                "FAKE": float(probs[1]),
            },
        }
        logger.debug("Prediction: %s", result)
        return result

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify a list of articles.

        Parameters
        ----------
        texts : list of str
            Raw article texts.

        Returns
        -------
        list of dict
            One prediction dict per input text (same schema as
            :meth:`predict`).
        """
        if not texts:
            return []

        cleaned = [self.preprocessor.clean_text(t) for t in texts]
        X = self.vectorizer.transform(cleaned)

        preds = self.model.predict(X)
        probs = self._get_probabilities(X)

        results: List[Dict[str, Any]] = []
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            pred_int = int(pred)
            results.append(
                {
                    "prediction": self.LABEL_MAP.get(pred_int, "UNKNOWN"),
                    "confidence": float(prob[pred_int]),
                    "probabilities": {
                        "REAL": float(prob[0]),
                        "FAKE": float(prob[1]),
                    },
                }
            )
        logger.info("Batch prediction complete: %d articles.", len(results))
        return results

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def get_important_words(
        self, text: str, top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Identify the words most influential for the model's prediction.

        For linear models the contribution of each word is its TF-IDF weight
        multiplied by the model coefficient.  For non-linear models (e.g.
        XGBoost) this falls back to just the TF-IDF weights of non-zero
        features.

        Parameters
        ----------
        text : str
            Raw input text.
        top_n : int, default 10
            Number of top words to return.

        Returns
        -------
        list of dict
            Each dict has ``word``, ``score``, ``direction``
            (``'fake'`` or ``'real'``).
        """
        cleaned = self.preprocessor.clean_text(text)
        X = self.vectorizer.transform([cleaned])

        tfidf_weights = np.asarray(X.todense()).ravel()

        # Try to get model coefficients
        coefs: Optional[np.ndarray] = None
        _model = self.model
        if hasattr(_model, "calibrated_classifiers_"):
            _model = _model.calibrated_classifiers_[0].estimator
        if hasattr(_model, "coef_"):
            coefs = np.asarray(_model.coef_).ravel()

        if coefs is not None and len(coefs) == len(tfidf_weights):
            contributions = tfidf_weights * coefs
        else:
            contributions = tfidf_weights  # fallback

        nonzero_idx = np.nonzero(tfidf_weights)[0]
        if len(nonzero_idx) == 0:
            return []

        abs_scores = np.abs(contributions[nonzero_idx])
        top_idx = nonzero_idx[np.argsort(abs_scores)[::-1][:top_n]]

        words: List[Dict[str, Any]] = []
        for idx in top_idx:
            score = float(contributions[idx])
            words.append(
                {
                    "word": self._feature_names[idx],
                    "score": score,
                    "direction": "fake" if score > 0 else "real",
                }
            )
        return words
