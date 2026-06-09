"""
Model Evaluation Module
========================

Provides the :class:`ModelEvaluator` class for computing metrics, comparing
models, and generating publication-ready plots and CSV reports for the
Fake News Detection pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – safe for servers / CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# Consistent visual style
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


class ModelEvaluator:
    """Evaluate, compare, and visualise fake-news classifiers.

    All plotting methods accept an optional *save_path*; when provided the
    figure is written to disk as a PNG file.
    """

    # ------------------------------------------------------------------
    # Single-model evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate(
        model: Any,
        X_test: Any,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Compute standard binary-classification metrics.

        Parameters
        ----------
        model : estimator
            A fitted classifier exposing ``predict`` (and optionally
            ``predict_proba`` or ``decision_function``).
        X_test : array-like / sparse matrix
            Test features.
        y_test : array-like
            True labels.

        Returns
        -------
        dict
            Keys: ``accuracy``, ``precision``, ``recall``, ``f1``,
            ``roc_auc``.
        """
        y_pred = model.predict(X_test)

        # Obtain probability estimates for ROC-AUC
        roc_auc: float
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
            else:
                y_prob = y_pred.astype(float)
            roc_auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            logger.warning("Could not compute ROC-AUC; defaulting to 0.0.")
            roc_auc = 0.0

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": roc_auc,
        }
        logger.info("Evaluation metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Multi-model comparison
    # ------------------------------------------------------------------

    def compare_models(
        self,
        models_dict: Dict[str, Dict[str, Any]],
        X_test: Any,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        """Evaluate every model and return a comparison DataFrame.

        Parameters
        ----------
        models_dict : dict
            ``{name: {model: …, …}}``  as returned by
            :meth:`ModelTrainer.train_all`.
        X_test, y_test
            Test data.

        Returns
        -------
        pd.DataFrame
            Rows = models, columns = metrics.
        """
        rows: List[Dict[str, Any]] = []
        for name, res in models_dict.items():
            logger.info("Evaluating %s on the test set …", name)
            m = self.evaluate(res["model"], X_test, y_test)
            m["model_name"] = name
            m["cv_f1"] = res.get("best_score", np.nan)
            rows.append(m)

        df = pd.DataFrame(rows).set_index("model_name")
        df.sort_values("f1", ascending=False, inplace=True)
        logger.info("\nModel comparison:\n%s", df.to_string())
        return df

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str | Path] = None,
    ) -> None:
        """Plot and (optionally) save a confusion-matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix – {model_name}")

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            logger.info("Confusion matrix saved to %s", save_path)
        plt.close(fig)

    @staticmethod
    def plot_roc_curve(
        model: Any,
        X_test: Any,
        y_test: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str | Path] = None,
    ) -> None:
        """Plot and (optionally) save an ROC curve."""
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
            else:
                logger.warning("Model has neither predict_proba nor decision_function.")
                return

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_val = roc_auc_score(y_test, y_prob)

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.4f})", lw=2)
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve – {model_name}")
            ax.legend(loc="lower right")

            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path)
                logger.info("ROC curve saved to %s", save_path)
            plt.close(fig)
        except Exception:
            logger.exception("Failed to plot ROC curve for %s.", model_name)

    @staticmethod
    def plot_model_comparison(
        comparison_df: pd.DataFrame,
        save_path: Optional[str | Path] = None,
    ) -> None:
        """Bar chart comparing all models across key metrics."""
        metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        plot_df = comparison_df[[c for c in metrics_to_plot if c in comparison_df.columns]]

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_df.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="black")
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        plt.xticks(rotation=30, ha="right")

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            logger.info("Model comparison chart saved to %s", save_path)
        plt.close(fig)

    @staticmethod
    def plot_feature_importance(
        model: Any,
        feature_names: List[str],
        top_n: int = 20,
        save_path: Optional[str | Path] = None,
    ) -> None:
        """Horizontal bar chart of the most influential features.

        Works with models that expose ``coef_`` (LR, SVM) or
        ``feature_importances_`` (tree-based models such as XGBoost).
        For :class:`CalibratedClassifierCV` wrappers, the inner
        estimator's coefficients are extracted.
        """
        coefs: Optional[np.ndarray] = None

        # Unwrap CalibratedClassifierCV
        _model = model
        if hasattr(_model, "calibrated_classifiers_"):
            _model = _model.calibrated_classifiers_[0].estimator

        if hasattr(_model, "coef_"):
            coefs = np.asarray(_model.coef_).ravel()
        elif hasattr(_model, "feature_importances_"):
            coefs = np.asarray(_model.feature_importances_).ravel()
        else:
            logger.warning("Model does not expose coef_ or feature_importances_.")
            return

        if len(coefs) != len(feature_names):
            logger.warning(
                "Feature-name length (%d) ≠ coefficient length (%d); skipping plot.",
                len(feature_names),
                len(coefs),
            )
            return

        top_idx = np.argsort(np.abs(coefs))[::-1][:top_n]
        top_features = [feature_names[i] for i in top_idx]
        top_values = coefs[top_idx]

        fig, ax = plt.subplots(figsize=(8, 7))
        colours = ["#d62728" if v < 0 else "#2ca02c" for v in top_values]
        ax.barh(range(len(top_features)), top_values, color=colours, edgecolor="black")
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel("Coefficient / Importance")
        ax.set_title(f"Top {top_n} Feature Importances")

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            logger.info("Feature importance plot saved to %s", save_path)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Full report generation
    # ------------------------------------------------------------------

    def generate_all_reports(
        self,
        models_dict: Dict[str, Dict[str, Any]],
        best_model_name: str,
        X_test: Any,
        y_test: np.ndarray,
        feature_names: List[str],
        reports_dir: str | Path = "reports",
    ) -> pd.DataFrame:
        """Generate every plot and CSV report in one call.

        Parameters
        ----------
        models_dict : dict
            Training results.
        best_model_name : str
            Name of the selected best model.
        X_test, y_test
            Test data.
        feature_names : list of str
            TF-IDF feature names.
        reports_dir : str or Path
            Directory for all output artefacts.

        Returns
        -------
        pd.DataFrame
            The comparison DataFrame.
        """
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Generating reports in %s …", reports_dir)

        # 1. Comparison table + CSV
        comparison_df = self.compare_models(models_dict, X_test, y_test)
        csv_path = reports_dir / "model_comparison.csv"
        comparison_df.to_csv(csv_path)
        logger.info("Comparison CSV saved to %s", csv_path)

        # 2. Comparison bar chart
        self.plot_model_comparison(
            comparison_df, save_path=reports_dir / "model_comparison.png"
        )

        # 3. Per-model confusion matrix + ROC curve
        for name, res in models_dict.items():
            model = res["model"]
            y_pred = model.predict(X_test)

            self.plot_confusion_matrix(
                y_test,
                y_pred,
                model_name=name,
                save_path=reports_dir / f"confusion_matrix_{name}.png",
            )
            self.plot_roc_curve(
                model,
                X_test,
                y_test,
                model_name=name,
                save_path=reports_dir / f"roc_curve_{name}.png",
            )

        # 4. Feature importance for the best model
        best_model = models_dict[best_model_name]["model"]
        self.plot_feature_importance(
            best_model,
            feature_names,
            top_n=20,
            save_path=reports_dir / f"feature_importance_{best_model_name}.png",
        )

        # 5. Classification report (text)
        y_pred_best = best_model.predict(X_test)
        report_text = classification_report(
            y_test, y_pred_best, target_names=["Real", "Fake"]
        )
        report_path = reports_dir / f"classification_report_{best_model_name}.txt"
        report_path.write_text(report_text, encoding="utf-8")
        logger.info("Classification report saved to %s", report_path)
        logger.info("\n%s", report_text)

        logger.info("All reports generated successfully.")
        return comparison_df
