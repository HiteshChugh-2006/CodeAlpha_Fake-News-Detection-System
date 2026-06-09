"""
Fake News Detection System
===========================

A complete ML pipeline for detecting fake news articles using NLP and
classical machine learning models (Logistic Regression, Naive Bayes,
Linear SVC, XGBoost).

Modules:
    - preprocessing: Text cleaning and normalization
    - feature_engineering: TF-IDF vectorization and vocabulary analysis
    - train: Model training with cross-validation and hyperparameter tuning
    - evaluate: Model evaluation, comparison, and report generation
    - predict: Inference on new articles
"""

from src.preprocessing import TextPreprocessor
from src.feature_engineering import FeatureEngineer
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.predict import NewsPredictor

__all__ = [
    "TextPreprocessor",
    "FeatureEngineer",
    "ModelTrainer",
    "ModelEvaluator",
    "NewsPredictor",
]

__version__ = "1.0.0"
