"""
Text Preprocessing Module
==========================

Provides the :class:`TextPreprocessor` class which cleans and normalises raw
news-article text so it is ready for downstream feature engineering.

Pipeline applied by :meth:`clean_text`:
    1. Lowercase
    2. URL removal
    3. HTML tag removal
    4. Punctuation removal
    5. Digit removal
    6. Tokenization  (``nltk.word_tokenize``)
    7. Stopword removal (NLTK English stopwords)
    8. Lemmatization  (``WordNetLemmatizer``)
"""

from __future__ import annotations

import logging
import re
import string
from typing import List, Optional

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure required NLTK data packages are available
# ---------------------------------------------------------------------------
_NLTK_PACKAGES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]


def _download_nltk_data() -> None:
    """Download NLTK resources if they are not already present."""
    for pkg in _NLTK_PACKAGES:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else pkg)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", pkg)
            nltk.download(pkg, quiet=True)


_download_nltk_data()


# ---------------------------------------------------------------------------
# Pre-compiled regex patterns (compiled once for performance)
# ---------------------------------------------------------------------------
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_HTML = re.compile(r"<[^>]+>")
_RE_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")
_RE_DIGITS = re.compile(r"\d+")
_RE_WHITESPACE = re.compile(r"\s+")


class TextPreprocessor:
    """Reusable text preprocessor with a scikit-learn-style fit/transform API.

    Parameters
    ----------
    remove_stopwords : bool, default True
        Whether to strip English stopwords.
    lemmatize : bool, default True
        Whether to apply WordNet lemmatization.
    min_token_length : int, default 2
        Discard tokens shorter than this after cleaning.

    Examples
    --------
    >>> tp = TextPreprocessor()
    >>> tp.fit()                       # optional – loads resources
    >>> tp.clean_text("Check http://x.com <b>BREAKING</b> news!!1")
    'check breaking news'
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_token_length: int = 2,
    ) -> None:
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_token_length = min_token_length

        self._stop_words: Optional[set] = None
        self._lemmatizer: Optional[WordNetLemmatizer] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fit / Transform API
    # ------------------------------------------------------------------

    def fit(self, X=None, y=None) -> "TextPreprocessor":
        """Initialise internal NLTK resources.

        Parameters are accepted (but ignored) so the object is compatible with
        scikit-learn ``Pipeline``.
        """
        if self.remove_stopwords:
            self._stop_words = set(stopwords.words("english"))
            logger.debug("Loaded %d English stopwords.", len(self._stop_words))
        if self.lemmatize:
            self._lemmatizer = WordNetLemmatizer()
        self._is_fitted = True
        logger.info("TextPreprocessor fitted.")
        return self

    def transform(self, texts: pd.Series) -> pd.Series:
        """Apply :meth:`clean_text` to every element of *texts*.

        Parameters
        ----------
        texts : pd.Series
            Raw text strings.

        Returns
        -------
        pd.Series
            Cleaned text strings.
        """
        if not self._is_fitted:
            self.fit()
        total = len(texts)
        logger.info("Preprocessing %d documents …", total)

        cleaned: List[str] = []
        log_every = max(1, total // 10)
        for idx, text in enumerate(texts):
            cleaned.append(self.clean_text(text))
            if (idx + 1) % log_every == 0:
                logger.info(
                    "  Progress: %d / %d (%.0f%%)",
                    idx + 1,
                    total,
                    (idx + 1) / total * 100,
                )
        logger.info("Preprocessing complete.")
        return pd.Series(cleaned, index=texts.index)

    def fit_transform(self, texts: pd.Series, y=None) -> pd.Series:
        """Convenience wrapper: ``fit`` then ``transform``."""
        return self.fit().transform(texts)

    # ------------------------------------------------------------------
    # Core cleaning logic
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """Clean a single text string.

        Steps applied (in order):
            1. Lowercase
            2. URL removal
            3. HTML tag removal
            4. Punctuation removal
            5. Digit removal
            6. Tokenization
            7. Stopword removal (optional)
            8. Lemmatization (optional)
            9. Short-token removal
            10. Re-join tokens

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        str
            Cleaned, space-joined tokens.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # Lazy-fit if the user skipped calling fit()
        if not self._is_fitted:
            self.fit()

        # 1. Lowercase
        text = text.lower()

        # 2. Remove URLs
        text = _RE_URL.sub("", text)

        # 3. Remove HTML tags
        text = _RE_HTML.sub("", text)

        # 4. Remove punctuation
        text = _RE_PUNCT.sub(" ", text)

        # 5. Remove digits
        text = _RE_DIGITS.sub("", text)

        # 6. Collapse whitespace then tokenize
        text = _RE_WHITESPACE.sub(" ", text).strip()
        tokens: List[str] = word_tokenize(text)

        # 7. Stopword removal
        if self.remove_stopwords and self._stop_words:
            tokens = [t for t in tokens if t not in self._stop_words]

        # 8. Lemmatization
        if self.lemmatize and self._lemmatizer:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]

        # 9. Drop short tokens
        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        return " ".join(tokens)

    # ------------------------------------------------------------------
    # DataFrame helper
    # ------------------------------------------------------------------

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        output_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Apply preprocessing to a DataFrame column *in-place*.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        text_column : str
            Name of the column containing raw text.
        output_column : str or None
            Name for the cleaned-text column.  Defaults to
            ``{text_column}_clean``.

        Returns
        -------
        pd.DataFrame
            The same DataFrame with the new cleaned column added.

        Raises
        ------
        ValueError
            If *text_column* is not found in *df*.
        """
        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        out_col = output_column or f"{text_column}_clean"
        logger.info(
            "Preprocessing column '%s' → '%s' (%d rows)",
            text_column,
            out_col,
            len(df),
        )

        df[out_col] = self.transform(df[text_column].fillna(""))
        return df
