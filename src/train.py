"""
Model Training Module
======================

Trains and tunes four classifiers for fake-news detection:

* Logistic Regression
* Multinomial Naive Bayes
* Linear SVC  (wrapped with :class:`CalibratedClassifierCV` so
  ``predict_proba`` is available)
* XGBoost (optional – skipped gracefully when the package is missing)

Each model is tuned via :class:`GridSearchCV` with stratified 5-fold
cross-validation, optimising the **F1 score**.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from scipy.sparse import spmatrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional XGBoost import
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
    logger.debug("XGBoost is available.")
except ImportError:
    _HAS_XGBOOST = False
    logger.warning(
        "xgboost is not installed – XGBClassifier will be skipped. "
        "Install with: pip install xgboost"
    )

# Type alias for the per-model result dictionary
ModelResult = Dict[str, Any]


class ModelTrainer:
    """Train and select the best fake-news classifier.

    Parameters
    ----------
    cv_folds : int, default 5
        Number of stratified cross-validation folds.
    scoring : str, default "f1"
        Metric used by :class:`GridSearchCV`.
    n_jobs : int, default -1
        Number of parallel jobs (``-1`` = use all cores).
    random_state : int, default 42
        Seed for reproducibility.
    """

    def __init__(
        self,
        cv_folds: int = 5,
        scoring: str = "f1",
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state

        self._cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        self._results: Dict[str, ModelResult] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Return a dict mapping model names to (estimator, param_grid)."""
        configs: Dict[str, Dict[str, Any]] = {
            "LogisticRegression": {
                "estimator": LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=self.random_state,
                ),
                "param_grid": {
                    "C": [0.1, 1, 10],
                    "penalty": ["l2"],
                },
            },
            "MultinomialNB": {
                "estimator": MultinomialNB(),
                "param_grid": {
                    "alpha": [0.1, 0.5, 1.0],
                },
            },
            "LinearSVC": {
                "estimator": CalibratedClassifierCV(
                    LinearSVC(
                        max_iter=2000,
                        random_state=self.random_state,
                    ),
                    cv=3,
                ),
                "param_grid": {
                    "estimator__C": [0.1, 1, 10],
                },
            },
        }

        if _HAS_XGBOOST:
            configs["XGBClassifier"] = {
                "estimator": XGBClassifier(
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=self.random_state,
                    verbosity=0,
                ),
                "param_grid": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5],
                    "learning_rate": [0.1, 0.01],
                },
            }
        else:
            logger.info("Skipping XGBClassifier (xgboost not installed).")

        return configs

    def _train_single(
        self,
        name: str,
        estimator: Any,
        param_grid: Dict[str, Any],
        X: spmatrix,
        y: np.ndarray,
    ) -> ModelResult:
        """Run GridSearchCV for a single model."""
        logger.info("Training %s ...", name)
        logger.info("  Param grid: %s", param_grid)

        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=self._cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=True,
            verbose=0,
            return_train_score=False,
        )
        grid.fit(X, y)

        result: ModelResult = {
            "model": grid.best_estimator_,
            "best_params": grid.best_params_,
            "cv_scores": grid.cv_results_["mean_test_score"].tolist(),
            "best_score": float(grid.best_score_),
        }

        logger.info("  Best params : %s", result["best_params"])
        logger.info("  Best CV F1  : %.4f", result["best_score"])
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_all(
        self,
        X_train: spmatrix,
        y_train: np.ndarray,
    ) -> Dict[str, ModelResult]:
        """Train all configured models and return their results.

        Parameters
        ----------
        X_train : sparse matrix
            TF-IDF feature matrix for the training set.
        y_train : array-like of int
            Binary labels (0 = real, 1 = fake).

        Returns
        -------
        dict
            ``{model_name: {model, best_params, cv_scores, best_score}}``
        """
        logger.info(
            "Starting training pipeline (%d models, %d-fold CV, scoring=%s).",
            len(self._model_configs()),
            self.cv_folds,
            self.scoring,
        )

        for name, cfg in self._model_configs().items():
            try:
                self._results[name] = self._train_single(
                    name, cfg["estimator"], cfg["param_grid"], X_train, y_train
                )
            except Exception:
                logger.exception("Failed to train %s – skipping.", name)

        logger.info(
            "Training complete. Models trained: %s",
            list(self._results.keys()),
        )
        return self._results

    def select_best_model(
        self,
        results: Optional[Dict[str, ModelResult]] = None,
    ) -> tuple[str, ModelResult]:
        """Pick the model with the highest CV F1 score.

        Parameters
        ----------
        results : dict or None
            Model results dict; defaults to the internally stored results
            from the last :meth:`train_all` call.

        Returns
        -------
        (best_name, best_result)
        """
        results = results or self._results
        if not results:
            raise RuntimeError("No training results available. Call train_all() first.")

        best_name = max(results, key=lambda k: results[k]["best_score"])
        logger.info(
            "Best model: %s (CV F1 = %.4f)",
            best_name,
            results[best_name]["best_score"],
        )
        return best_name, results[best_name]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_model(model: Any, path: str | Path) -> None:
        """Save a single model to disk.

        Parameters
        ----------
        model : estimator
            A fitted scikit-learn / xgboost estimator.
        path : str or Path
            Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        logger.info("Model saved to %s", path)

    @staticmethod
    def save_all_models(
        results: Dict[str, ModelResult],
        directory: str | Path,
    ) -> Dict[str, Path]:
        """Save every trained model into *directory*.

        Parameters
        ----------
        results : dict
            Output of :meth:`train_all`.
        directory : str or Path
            Target directory.

        Returns
        -------
        dict
            ``{model_name: saved_path}``
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {}
        for name, res in results.items():
            p = directory / f"{name}.joblib"
            joblib.dump(res["model"], p)
            paths[name] = p
            logger.info("Saved %s -> %s", name, p)
        return paths
