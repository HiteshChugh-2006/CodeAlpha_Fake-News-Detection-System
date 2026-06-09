#!/usr/bin/env python
"""
Fake News Detection – Main Training Pipeline
==============================================

End-to-end script that:

1. Loads ``Fake.csv`` / ``True.csv`` from ``dataset/`` (or generates
   synthetic sample data for demo purposes).
2. Combines, shuffles, and preprocesses the text.
3. Builds TF-IDF features.
4. Trains and tunes four classifiers.
5. Evaluates, compares, and selects the best model.
6. Saves the best model + vectorizer to ``models/``.
7. Generates all reports to ``reports/``.

Usage
-----
    python main.py
"""

from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── project modules ──────────────────────────────────────────────────
from src.preprocessing import TextPreprocessor
from src.feature_engineering import FeatureEngineer
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.predict import NewsPredictor

# ── paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# ── logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "pipeline.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


# =====================================================================
# Synthetic dataset generator
# =====================================================================

def _create_synthetic_dataset() -> pd.DataFrame:
    """Generate a small synthetic dataset (50 fake + 50 real) for demo.

    The articles are intentionally simplistic but varied enough to let every
    stage of the pipeline execute without errors.
    """
    logger.warning(
        "Real dataset files (Fake.csv / True.csv) not found in %s.\n"
        "  -> Generating synthetic sample data (50 fake + 50 real) for demo.\n"
        "  -> For real results, download the Kaggle Fake-and-Real-News dataset\n"
        "    and place Fake.csv and True.csv into the dataset/ folder.",
        DATASET_DIR,
    )

    random.seed(42)

    fake_titles = [
        "BREAKING: Secret Government Plot Exposed",
        "Scientists Discover Earth Is Actually Flat",
        "Aliens Land in Central Park, Government Covers Up",
        "Celebrity Exposed in Massive Scandal",
        "Miracle Cure Found in Common Household Item",
        "Government Implanting Microchips Through Vaccines",
        "Secret Society Controls World Economy",
        "Moon Landing Was Filmed in Hollywood Studio",
        "Election Rigged by Foreign Hackers",
        "Famous Politician Secretly a Lizard Person",
    ]

    fake_bodies = [
        "According to anonymous sources who spoke exclusively to us, the government "
        "has been secretly conducting experiments on citizens without consent. Multiple "
        "whistleblowers have come forward with shocking revelations that will change "
        "everything you thought you knew about the world.",
        "A group of rogue scientists have allegedly discovered proof that contradicts "
        "everything mainstream science has told us. The establishment is desperately "
        "trying to suppress this information, but brave truth-seekers are sharing it "
        "on social media before it gets censored.",
        "Shocking footage that THEY don't want you to see has surfaced on underground "
        "websites. This bombshell revelation proves that the mainstream media has been "
        "lying to the public for decades. Share this before it gets deleted!",
        "Insiders reveal a massive cover-up involving top officials. The deep state "
        "has been manipulating events behind the scenes. Wake up sheeple! The truth "
        "is out there and we are the only ones brave enough to report it.",
        "You won't believe what we've uncovered. A secret document leaked by an "
        "anonymous patriot reveals the shocking truth. The elites are terrified that "
        "this information is getting out. Forward this to everyone you know.",
    ]

    real_titles = [
        "Federal Reserve Holds Interest Rates Steady",
        "New Climate Report Shows Rising Sea Levels",
        "Congress Debates Infrastructure Spending Bill",
        "Tech Companies Report Quarterly Earnings",
        "Supreme Court Hears Arguments on Healthcare Law",
        "International Summit Focuses on Trade Agreements",
        "Research Shows Decline in Unemployment Rate",
        "New Study Links Exercise to Improved Mental Health",
        "Local Government Approves Budget for School Funding",
        "Scientists Publish Findings on Renewable Energy",
    ]

    real_bodies = [
        "In a widely anticipated decision, policymakers announced that interest rates "
        "would remain unchanged for the current quarter. Economists noted that the move "
        "reflects cautious optimism about economic growth, while inflation remains within "
        "the target range set earlier this year.",
        "A comprehensive study published in a peer-reviewed journal has documented changes "
        "in global temperature patterns over the past decade. Researchers from multiple "
        "universities collaborated on the project, analyzing data from weather stations "
        "and satellite measurements across six continents.",
        "Lawmakers engaged in extensive debate over proposed legislation aimed at "
        "modernizing the nation's transportation and communications infrastructure. "
        "The bill, which has bipartisan support, would allocate funding over a ten-year "
        "period for roads, bridges, and broadband internet expansion.",
        "Several major technology firms released their quarterly financial results this "
        "week, showing mixed performance across the sector. Revenue growth exceeded "
        "analyst expectations for some companies, while others reported challenges "
        "related to supply chain disruptions and regulatory pressures.",
        "Researchers at a leading university published results from a longitudinal "
        "study examining the relationship between regular physical activity and mental "
        "health outcomes. The study, which followed participants over five years, found "
        "statistically significant improvements in reported well-being.",
    ]

    rows = []
    for i in range(50):
        title = random.choice(fake_titles)
        body = random.choice(fake_bodies)
        rows.append({"title": title, "text": body, "label": 1})  # Fake

    for i in range(50):
        title = random.choice(real_titles)
        body = random.choice(real_bodies)
        rows.append({"title": title, "text": body, "label": 0})  # Real

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# =====================================================================
# Data loading
# =====================================================================

def load_data() -> pd.DataFrame:
    """Load and combine Fake.csv + True.csv, or fall back to synthetic data."""
    fake_path = DATASET_DIR / "Fake.csv"
    true_path = DATASET_DIR / "True.csv"

    if fake_path.exists() and true_path.exists():
        logger.info("Loading real dataset from %s ...", DATASET_DIR)
        df_fake = pd.read_csv(fake_path)
        df_true = pd.read_csv(true_path)

        df_fake["label"] = 1  # Fake
        df_true["label"] = 0  # Real

        df = pd.concat([df_fake, df_true], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Combine title + text into a single column
        if "title" in df.columns and "text" in df.columns:
            df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
        elif "text" not in df.columns:
            raise ValueError("Dataset must contain a 'text' column.")

        logger.info(
            "Loaded %d articles (Fake=%d, Real=%d).",
            len(df),
            (df["label"] == 1).sum(),
            (df["label"] == 0).sum(),
        )
    else:
        df = _create_synthetic_dataset()
        # Also combine title + text
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
        logger.info("Synthetic dataset created: %d articles.", len(df))

    return df


# =====================================================================
# Main pipeline
# =====================================================================

def main() -> None:
    """Run the full Fake News Detection training pipeline."""
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("  FAKE NEWS DETECTION PIPELINE")
    logger.info("=" * 70)

    # -- 0. Create directories ----------------------------------------
    for d in (DATASET_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
        logger.info("Directory ready: %s", d)

    # -- 1. Load data -------------------------------------------------
    logger.info("\n>> Step 1/7: Loading data ...")
    df = load_data()

    # -- 2. Preprocess ------------------------------------------------
    logger.info("\n>> Step 2/7: Text preprocessing ...")
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df, text_column="text", output_column="text_clean")

    # Drop rows where cleaning produced empty strings
    before = len(df)
    df = df[df["text_clean"].str.strip().astype(bool)].reset_index(drop=True)
    if len(df) < before:
        logger.warning("Dropped %d empty-after-cleaning rows.", before - len(df))

    # -- 3. Feature engineering ---------------------------------------
    logger.info("\n>> Step 3/7: Feature engineering (TF-IDF) ...")
    feature_engineer = FeatureEngineer()
    X_all = feature_engineer.fit_transform(df["text_clean"])

    # Vocabulary diagnostics
    vocab_stats = feature_engineer.vocabulary_analysis(df["text_clean"])
    logger.info("Vocabulary stats: %s", vocab_stats)

    top_features = feature_engineer.get_top_features(n=20)
    logger.info("Top 20 features by IDF: %s", [f[0] for f in top_features])

    # -- 4. Train / test split ----------------------------------------
    logger.info("\n>> Step 4/7: Train / test split (80 / 20) ...")
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(
        "Train: %d samples | Test: %d samples", X_train.shape[0], X_test.shape[0]
    )

    # -- 5. Train models ---------------------------------------------
    logger.info("\n>> Step 5/7: Model training + hyperparameter tuning ...")
    trainer = ModelTrainer()
    results = trainer.train_all(X_train, y_train)

    # -- 6. Evaluate & compare ----------------------------------------
    logger.info("\n>> Step 6/7: Evaluation & comparison ...")
    evaluator = ModelEvaluator()
    comparison_df = evaluator.generate_all_reports(
        models_dict=results,
        best_model_name=trainer.select_best_model(results)[0],
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_engineer.get_feature_names(),
        reports_dir=REPORTS_DIR,
    )

    # -- 7. Save best model & vectorizer ------------------------------
    logger.info("\n>> Step 7/7: Saving artefacts ...")
    best_name, best_result = trainer.select_best_model(results)
    best_model_path = MODELS_DIR / "best_model.joblib"
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.joblib"

    ModelTrainer.save_model(best_result["model"], best_model_path)
    feature_engineer.save(vectorizer_path)

    # Also save all individual models
    trainer.save_all_models(results, MODELS_DIR)

    # -- Summary ------------------------------------------------------
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("  Best model      : %s", best_name)
    logger.info("  Best CV F1      : %.4f", best_result["best_score"])
    logger.info("  Best params     : %s", best_result["best_params"])
    logger.info("  Model saved to  : %s", best_model_path)
    logger.info("  Vectorizer saved: %s", vectorizer_path)
    logger.info("  Reports dir     : %s", REPORTS_DIR)
    logger.info("  Total time      : %.1f seconds", elapsed)
    logger.info("=" * 70)

    # -- Quick demo prediction ----------------------------------------
    logger.info("\n>> Quick demo prediction ...")
    predictor = NewsPredictor(best_model_path, vectorizer_path)

    demo_texts = [
        "BREAKING: Scientists confirm that the Earth is secretly controlled by lizard people!",
        "The Federal Reserve announced today that interest rates will remain unchanged.",
    ]
    for text in demo_texts:
        result = predictor.predict(text)
        logger.info("  Text : %s", text[:80] + "..." if len(text) > 80 else text)
        logger.info(
            "  Pred : %s (confidence=%.2f%%)\n",
            result["prediction"],
            result["confidence"] * 100,
        )

    # Print comparison table to stdout for convenience
    print("\n" + comparison_df.to_string() + "\n")


if __name__ == "__main__":
    main()
