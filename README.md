# 📰 Fake News Detection System

A production-ready **Fake News Detection System** built with Natural Language Processing and Machine Learning. This project predicts whether a news article is **REAL** or **FAKE** with high accuracy, featuring an interactive Streamlit dashboard for real-time analysis.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange?style=for-the-badge&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?style=for-the-badge&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green?style=for-the-badge&logo=xgboost)

---

## 🎯 Features

- **Multi-Model Comparison**: Logistic Regression, Naive Bayes, Linear SVM, XGBoost
- **Advanced NLP Pipeline**: TF-IDF with unigrams & bigrams, comprehensive text preprocessing
- **Interactive Dashboard**: Professional Streamlit app with dark theme
- **Model Explainability**: Feature importance, word contributions, prediction explanations
- **Real-time Prediction**: Input any news article for instant classification
- **Dataset Analytics**: Comprehensive visualizations with Plotly and WordClouds
- **Automated Pipeline**: End-to-end training, evaluation, and deployment

---

## 📁 Project Structure

```
Fake_News_Detection/
│
├── dataset/                    # Dataset files
│   ├── Fake.csv
│   └── True.csv
│
├── models/                     # Saved trained models
│   ├── best_model.joblib
│   └── tfidf_vectorizer.joblib
│
├── reports/                    # Generated evaluation reports
│   ├── confusion_matrix_*.png
│   ├── roc_curve_*.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── model_comparison.csv
│
├── pages/                      # Streamlit multi-page app
│   ├── 1_News_Detection.py
│   ├── 2_Analytics.py
│   ├── 3_Model_Performance.py
│   ├── 4_Explainability.py
│   └── 5_About_Project.py
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── preprocessing.py        # Text preprocessing pipeline
│   ├── feature_engineering.py  # TF-IDF feature extraction
│   ├── train.py                # Model training & tuning
│   ├── evaluate.py             # Evaluation & reporting
│   └── predict.py              # Prediction interface
│
├── app.py                      # Streamlit main entry point
├── main.py                     # Training pipeline entry point
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version for deployment
├── README.md                   # This file
└── .gitignore                  # Git ignore rules
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Fake_News_Detection.git
cd Fake_News_Detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download the **Fake and Real News Dataset** from Kaggle:

1. Visit: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Download `Fake.csv` and `True.csv`
3. Place both files in the `dataset/` folder

> **Note**: If dataset files are not present, the system will generate synthetic sample data for demonstration purposes.

### 4. Train the Models

```bash
python main.py
```

This will:
- Load and preprocess the dataset
- Extract TF-IDF features
- Train 4 models with hyperparameter tuning
- Evaluate and compare all models
- Save the best model to `models/`
- Generate reports in `reports/`

### 5. Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## 🧠 Models

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear model with L2 regularization |
| **Multinomial Naive Bayes** | Probabilistic classifier for text |
| **Linear SVM** | Support Vector Machine with linear kernel |
| **XGBoost** | Gradient boosted decision trees |

### Training Features
- **Stratified 5-Fold Cross Validation**
- **Grid Search Hyperparameter Tuning**
- **F1 Score** as primary optimization metric
- **Automatic Best Model Selection**

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

### Generated Reports
- Confusion Matrix for each model
- ROC Curves
- Model Comparison Charts
- Feature Importance Plots

---

## 🖥️ Dashboard Pages

1. **Home** - Project overview with KPI cards
2. **News Detection** - Real-time article classification with confidence gauge
3. **Analytics** - Dataset exploration with interactive charts and word clouds
4. **Model Performance** - Detailed model comparison and evaluation metrics
5. **Explainability** - Feature importance and word contribution analysis
6. **About** - Project documentation and technical details

---

## 🛠️ Text Preprocessing Pipeline

1. Lowercase conversion
2. URL removal
3. HTML tag removal
4. Punctuation removal
5. Digit removal
6. Tokenization (NLTK)
7. Stopword removal
8. Lemmatization (WordNet)

---

## 📦 Deployment

### Streamlit Cloud

1. Push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set `app.py` as the main file
5. Deploy!

### Docker (Optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/Fake_News_Detection/issues).

---

## ⭐ Acknowledgments

- [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) by Clément Bisaillon
- Built with [Streamlit](https://streamlit.io/), [Scikit-learn](https://scikit-learn.org/), [NLTK](https://www.nltk.org/), [Plotly](https://plotly.com/)
