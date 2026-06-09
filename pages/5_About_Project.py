"""
Fake News Detection - About Project Page
==========================================
Project overview, architecture, tech stack, model descriptions,
and future improvements.
"""

import streamlit as st

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="About | Fake News Detector",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Shared CSS ──────────────────────────────────────────────────────────────
SHARED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 50%, #0e1117 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid rgba(0, 212, 255, 0.1);
}
.section-header { font-size: 1.5rem; font-weight: 700; color: #e0e0e0;
    margin: 40px 0 20px; padding-bottom: 12px;
    border-bottom: 2px solid rgba(0, 212, 255, 0.2); }
.gradient-divider { height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, #00ff88, transparent);
    border: none; margin: 30px 0; border-radius: 2px; }

/* Tech badges */
.tech-badge {
    display: inline-block; padding: 8px 20px; margin: 6px;
    border-radius: 50px; font-size: 0.85rem; font-weight: 600;
    background: rgba(0, 212, 255, 0.1); color: #00d4ff;
    border: 1px solid rgba(0, 212, 255, 0.3);
    transition: all 0.2s ease;
}
.tech-badge:hover { background: rgba(0, 212, 255, 0.25); transform: scale(1.05); }
.tech-badge-green { background: rgba(0, 255, 136, 0.1); color: #00ff88;
    border-color: rgba(0, 255, 136, 0.3); }
.tech-badge-green:hover { background: rgba(0, 255, 136, 0.25); }
.tech-badge-orange { background: rgba(255, 165, 0, 0.1); color: #ffa500;
    border-color: rgba(255, 165, 0, 0.3); }
.tech-badge-orange:hover { background: rgba(255, 165, 0, 0.25); }
.tech-badge-red { background: rgba(255, 71, 87, 0.1); color: #ff4757;
    border-color: rgba(255, 71, 87, 0.3); }
.tech-badge-red:hover { background: rgba(255, 71, 87, 0.25); }
.tech-badge-purple { background: rgba(168, 85, 247, 0.1); color: #a855f7;
    border-color: rgba(168, 85, 247, 0.3); }
.tech-badge-purple:hover { background: rgba(168, 85, 247, 0.25); }

/* Info cards */
.info-card {
    background: linear-gradient(135deg, #1e2a3a 0%, #16213e 100%);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 16px; padding: 28px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    margin-bottom: 16px;
}

/* Pipeline steps */
.pipeline-step {
    background: linear-gradient(135deg, #1e2a3a, #16213e);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 16px; padding: 24px; text-align: center;
    position: relative; min-height: 180px;
    display: flex; flex-direction: column; justify-content: center;
    transition: all 0.3s ease;
}
.pipeline-step:hover { border-color: rgba(0, 212, 255, 0.4);
    transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0,212,255,0.1); }
.pipeline-num {
    position: absolute; top: -14px; left: 50%; transform: translateX(-50%);
    background: linear-gradient(135deg, #00d4ff, #00ff88);
    color: #0e1117; width: 28px; height: 28px; border-radius: 50%;
    font-size: 0.8rem; font-weight: 800; line-height: 28px;
    text-align: center;
}

/* Model cards */
.model-card {
    background: linear-gradient(135deg, #1e2a3a 0%, #16213e 100%);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 16px; padding: 28px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease; min-height: 300px;
}
.model-card:hover { border-color: rgba(0, 212, 255, 0.35);
    transform: translateY(-4px); box-shadow: 0 8px 28px rgba(0,212,255,0.1); }
</style>
"""

st.markdown(SHARED_CSS, unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0;">
        <div style="font-size: 3rem;">ℹ️</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #00d4ff;">About Project</div>
        <div style="font-size: 0.75rem; color: #556677; margin-top: 4px;">DOCUMENTATION</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 8px 0; color: #8899aa; font-size: 0.82rem; line-height: 1.8;">
        <strong style="color:#e0e0e0;">📑 Sections</strong><br>
        🏗️ Project Overview<br>
        🛠️ Tech Stack<br>
        🔄 Pipeline<br>
        🤖 Models<br>
        📊 Dataset<br>
        🚀 Future Work
    </div>
    """, unsafe_allow_html=True)


# ─── Page Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 30px 0 10px;">
    <div style="font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #00ff88);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
        ℹ️ About the Project
    </div>
    <div style="color: #8899aa; font-size: 1rem; margin-top: 8px;">
        Architecture, methodology, and technical details of the Fake News Detection System
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ─── Project Overview ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🏗️ Project Overview</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    <div style="color: #e0e0e0; font-size: 1.05rem; line-height: 1.8;">
        The <strong style="color: #00d4ff;">Fake News Detection System</strong> is an end-to-end 
        machine learning project that classifies news articles as <strong style="color:#00ff88;">Real</strong> 
        or <strong style="color:#ff4757;">Fake</strong> using Natural Language Processing (NLP) techniques 
        and classical ML algorithms.
        <br><br>
        The system features a complete ML pipeline — from raw text preprocessing and TF-IDF feature 
        engineering to model training with hyperparameter tuning, evaluation with cross-validation, 
        and deployment through an interactive Streamlit dashboard.
        <br><br>
        Key capabilities include:
    </div>
    <div style="margin-top: 16px; display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
        <div style="color: #8899aa; font-size: 0.9rem;">
            ✅ Real-time article classification<br>
            ✅ Confidence scoring<br>
            ✅ Word-level explainability
        </div>
        <div style="color: #8899aa; font-size: 0.9rem;">
            ✅ Multi-model comparison<br>
            ✅ Interactive analytics dashboard<br>
            ✅ Modular, extensible architecture
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Architecture Diagram ───────────────────────────────────────────────────
st.markdown('<div class="section-header">📐 System Architecture</div>', unsafe_allow_html=True)

st.markdown("""
```mermaid
graph LR
    A["📰 Raw Text"] --> B["🧹 Preprocessing"]
    B --> C["🔢 TF-IDF Vectorization"]
    C --> D["🤖 Model Training"]
    D --> E["📊 Evaluation"]
    D --> F["💾 Saved Model"]
    F --> G["🔍 Prediction API"]
    G --> H["📱 Streamlit Dashboard"]
    E --> I["📈 Reports"]
    I --> H

    style A fill:#1e2a3a,stroke:#00d4ff,color:#e0e0e0
    style B fill:#1e2a3a,stroke:#00ff88,color:#e0e0e0
    style C fill:#1e2a3a,stroke:#00d4ff,color:#e0e0e0
    style D fill:#1e2a3a,stroke:#ff4757,color:#e0e0e0
    style E fill:#1e2a3a,stroke:#ffa500,color:#e0e0e0
    style F fill:#1e2a3a,stroke:#a855f7,color:#e0e0e0
    style G fill:#1e2a3a,stroke:#00ff88,color:#e0e0e0
    style H fill:#1e2a3a,stroke:#00d4ff,color:#e0e0e0
    style I fill:#1e2a3a,stroke:#ffa500,color:#e0e0e0
```
""")

st.markdown("""
<div style="background: linear-gradient(135deg, #1e2a3a, #16213e);
    border: 1px solid rgba(0, 212, 255, 0.15); border-radius: 16px;
    padding: 20px; margin-top: 12px; font-family: 'Courier New', monospace;
    font-size: 0.82rem; color: #8899aa; line-height: 1.6;">
<pre style="margin: 0; color: #8899aa;">
Fake_News_Detection/
├── app.py                     # 🏠 Main dashboard (Home page)
├── pages/
│   ├── 1_News_Detection.py    # 🔍 Real-time article analysis
│   ├── 2_Analytics.py         # 📊 Dataset exploration
│   ├── 3_Model_Performance.py # 📈 Model evaluation & comparison
│   ├── 4_Explainability.py    # 🧠 Feature importance & reasoning
│   └── 5_About_Project.py     # ℹ️ Project documentation
├── src/
│   ├── preprocessing.py       # Text cleaning & normalization
│   ├── feature_engineering.py # TF-IDF vectorization
│   ├── train.py               # Model training & tuning
│   ├── evaluate.py            # Evaluation & report generation
│   └── predict.py             # Inference API
├── models/                    # Saved models (.joblib)
├── dataset/                   # Training data (Fake.csv, True.csv)
├── reports/                   # Generated charts & metrics
└── .streamlit/config.toml     # Theme configuration
</pre>
</div>
""", unsafe_allow_html=True)


# ─── Tech Stack ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🛠️ Technology Stack</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-card" style="text-align: center;">
    <div style="margin-bottom: 16px; color: #8899aa; font-size: 0.85rem; text-transform: uppercase;
        letter-spacing: 1.5px; font-weight: 600;">Core Technologies</div>
    <span class="tech-badge">🐍 Python 3.10+</span>
    <span class="tech-badge-green">📊 Scikit-learn</span>
    <span class="tech-badge-orange">🚀 XGBoost</span>
    <span class="tech-badge-red">📱 Streamlit</span>
    <span class="tech-badge-purple">📈 Plotly</span>
    <span class="tech-badge">📝 NLTK</span>
    <span class="tech-badge-green">🔢 NumPy</span>
    <span class="tech-badge-orange">🐼 Pandas</span>
    <span class="tech-badge-red">☁️ WordCloud</span>
    <span class="tech-badge-purple">💾 Joblib</span>
</div>
""", unsafe_allow_html=True)

# Detailed tech descriptions
tc1, tc2 = st.columns(2)

with tc1:
    st.markdown("""
    <div class="info-card">
        <div style="color: #00d4ff; font-size: 1.1rem; font-weight: 700; margin-bottom: 12px;">
            🔬 ML & NLP Stack
        </div>
        <div style="color: #8899aa; font-size: 0.88rem; line-height: 1.8;">
            <strong style="color:#e0e0e0;">Scikit-learn</strong> — Model training, evaluation, TF-IDF vectorization, 
            cross-validation, and hyperparameter tuning via GridSearchCV.<br><br>
            <strong style="color:#e0e0e0;">XGBoost</strong> — Gradient boosted trees for ensemble learning, 
            providing robust classification with built-in regularization.<br><br>
            <strong style="color:#e0e0e0;">NLTK</strong> — Text preprocessing: tokenization, stopword removal, 
            lemmatization, and part-of-speech tagging.<br><br>
            <strong style="color:#e0e0e0;">Joblib</strong> — Efficient model serialization and deserialization 
            for production deployment.
        </div>
    </div>
    """, unsafe_allow_html=True)

with tc2:
    st.markdown("""
    <div class="info-card">
        <div style="color: #00ff88; font-size: 1.1rem; font-weight: 700; margin-bottom: 12px;">
            📱 Visualization & Dashboard
        </div>
        <div style="color: #8899aa; font-size: 0.88rem; line-height: 1.8;">
            <strong style="color:#e0e0e0;">Streamlit</strong> — Interactive web application framework with 
            multi-page support, caching, and real-time updates.<br><br>
            <strong style="color:#e0e0e0;">Plotly</strong> — Interactive, publication-quality charts including 
            bar charts, pie charts, gauges, radar plots, and heatmaps.<br><br>
            <strong style="color:#e0e0e0;">Matplotlib</strong> — Backend visualization for confusion matrices, 
            ROC curves, and report image generation.<br><br>
            <strong style="color:#e0e0e0;">WordCloud</strong> — Visual representation of word frequencies 
            in fake vs real news articles.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── ML Pipeline ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔄 Machine Learning Pipeline</div>', unsafe_allow_html=True)

steps = [
    ("1", "📥 Data Loading", "Load Fake.csv and True.csv datasets. Combine and label articles with binary classes (0=Real, 1=Fake)."),
    ("2", "🧹 Text Preprocessing", "Clean raw text: lowercase, remove URLs/HTML/special chars, tokenize, remove stopwords, and lemmatize with NLTK WordNetLemmatizer."),
    ("3", "🔢 Feature Engineering", "Transform cleaned text to numerical features using TF-IDF Vectorizer with configurable max_features, n-gram range, and sublinear TF scaling."),
    ("4", "✂️ Train-Test Split", "Split data into 80% training and 20% testing sets with stratification to maintain class balance."),
    ("5", "🤖 Model Training", "Train multiple classifiers (LR, NB, SVM, XGBoost) with hyperparameter tuning via GridSearchCV and 5-fold cross-validation."),
    ("6", "📊 Evaluation & Export", "Evaluate all models on test set. Generate comparison metrics, confusion matrices, ROC curves. Save best model and vectorizer."),
]

for i in range(0, len(steps), 3):
    cols = st.columns(3)
    for j in range(3):
        if i + j < len(steps):
            num, title, desc = steps[i + j]
            with cols[j]:
                st.markdown(f"""
                <div class="pipeline-step">
                    <div class="pipeline-num">{num}</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #00d4ff; margin-bottom: 10px;">
                        {title}
                    </div>
                    <div style="color: #8899aa; font-size: 0.83rem; line-height: 1.6;">
                        {desc}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


# ─── Model Descriptions ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">🤖 Model Descriptions</div>', unsafe_allow_html=True)

models = [
    {
        "name": "Logistic Regression",
        "icon": "📐",
        "color": "#00d4ff",
        "desc": "A linear classification model that applies a logistic (sigmoid) function to a linear combination of features. Despite its simplicity, it is one of the most effective models for text classification due to the high-dimensional, sparse nature of TF-IDF features.",
        "pros": ["Highly interpretable (coefficients = word importance)", "Fast training and inference", "Robust to noise in high-dimensional data", "Well-suited for binary classification"],
        "cons": ["Assumes linear decision boundary", "May underperform on highly non-linear patterns"],
    },
    {
        "name": "Multinomial Naive Bayes",
        "icon": "📊",
        "color": "#00ff88",
        "desc": "A probabilistic classifier based on Bayes' theorem with the 'naive' assumption that features are conditionally independent given the class label. The multinomial variant works directly with word counts/frequencies, making it a natural fit for text.",
        "pros": ["Very fast training (O(n) complexity)", "Works well with small datasets", "Natural probabilistic interpretation", "Strong baseline for NLP tasks"],
        "cons": ["'Naive' independence assumption may not hold", "Sensitive to feature scaling"],
    },
    {
        "name": "Linear SVC",
        "icon": "🎯",
        "color": "#ffa500",
        "desc": "A Support Vector Machine with a linear kernel that finds the maximum-margin hyperplane separating classes. The linear variant is efficient for high-dimensional text data and provides regularization through the C parameter.",
        "pros": ["Effective in high-dimensional spaces", "Memory-efficient (uses support vectors only)", "Robust margin-based optimization", "Handles sparse TF-IDF well"],
        "cons": ["No native probability estimates", "Sensitive to feature scaling", "Slower with very large datasets"],
    },
    {
        "name": "XGBoost",
        "icon": "🚀",
        "color": "#a855f7",
        "desc": "An optimized gradient boosting framework that builds an ensemble of decision trees sequentially, where each tree corrects errors from previous ones. Known for winning ML competitions, though its advantage is less pronounced on purely linear text classification tasks.",
        "pros": ["Handles non-linear patterns", "Built-in regularization (L1/L2)", "Feature importance via gain/cover", "Robust to overfitting"],
        "cons": ["Slower training than linear models", "More hyperparameters to tune", "May not outperform linear models on TF-IDF"],
    },
]

model_cols = st.columns(2)
for i, m in enumerate(models):
    with model_cols[i % 2]:
        pros_html = "".join(f'<div style="color:#00ff88; font-size:0.82rem; margin:3px 0;">✅ {p}</div>' for p in m["pros"])
        cons_html = "".join(f'<div style="color:#ff4757; font-size:0.82rem; margin:3px 0;">⚠️ {c}</div>' for c in m["cons"])

        st.markdown(f"""
        <div class="model-card">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 14px;">
                <span style="font-size: 2rem;">{m['icon']}</span>
                <span style="font-size: 1.2rem; font-weight: 700; color: {m['color']};">{m['name']}</span>
            </div>
            <div style="color: #8899aa; font-size: 0.88rem; line-height: 1.7; margin-bottom: 16px;">
                {m['desc']}
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                <div>
                    <div style="color: #556677; font-size: 0.75rem; text-transform: uppercase;
                        letter-spacing: 1px; margin-bottom: 6px; font-weight: 600;">Strengths</div>
                    {pros_html}
                </div>
                <div>
                    <div style="color: #556677; font-size: 0.75rem; text-transform: uppercase;
                        letter-spacing: 1px; margin-bottom: 6px; font-weight: 600;">Limitations</div>
                    {cons_html}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Dataset Information ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Dataset Information</div>', unsafe_allow_html=True)

ds1, ds2 = st.columns(2)

with ds1:
    st.markdown("""
    <div class="info-card">
        <div style="color: #00d4ff; font-size: 1.1rem; font-weight: 700; margin-bottom: 14px;">
            📂 Dataset Source
        </div>
        <div style="color: #8899aa; font-size: 0.88rem; line-height: 1.8;">
            The project uses the <strong style="color:#e0e0e0;">ISOT Fake News Dataset</strong>, 
            containing articles collected from legitimate news sources (Reuters.com) and fake news 
            sources flagged by Politifact and Wikipedia.<br><br>
            <strong style="color:#e0e0e0;">Files:</strong><br>
            📄 <code style="color:#00d4ff;">Fake.csv</code> — Fake news articles<br>
            📄 <code style="color:#00ff88;">True.csv</code> — Real news articles<br><br>
            <strong style="color:#e0e0e0;">Columns:</strong> title, text, subject, date
        </div>
    </div>
    """, unsafe_allow_html=True)

with ds2:
    st.markdown("""
    <div class="info-card">
        <div style="color: #00ff88; font-size: 1.1rem; font-weight: 700; margin-bottom: 14px;">
            📋 Data Characteristics
        </div>
        <div style="color: #8899aa; font-size: 0.88rem; line-height: 1.8;">
            <strong style="color:#e0e0e0;">Size:</strong> ~44,000 total articles<br>
            <strong style="color:#e0e0e0;">Balance:</strong> Roughly equal split between classes<br>
            <strong style="color:#e0e0e0;">Coverage:</strong> Political, world news, and general topics<br>
            <strong style="color:#e0e0e0;">Time Period:</strong> 2015–2018<br>
            <strong style="color:#e0e0e0;">Language:</strong> English<br><br>
            <strong style="color:#e0e0e0;">Preprocessing Applied:</strong><br>
            Lowercasing, URL/HTML removal, stopword filtering, lemmatization, 
            and TF-IDF vectorization with up to 50,000 features.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Future Improvements ────────────────────────────────────────────────────
st.markdown('<div class="section-header">🚀 Future Improvements</div>', unsafe_allow_html=True)

improvements = [
    ("🧠", "Deep Learning Models", "Integrate BERT, RoBERTa, or DistilBERT for contextual embeddings and improved classification accuracy on nuanced fake news patterns."),
    ("🌐", "Multi-Language Support", "Extend the system to detect fake news in multiple languages using multilingual transformers (mBERT, XLM-RoBERTa)."),
    ("📡", "Real-Time News Feed", "Connect to live news APIs (NewsAPI, RSS feeds) for continuous monitoring and automated fake news detection at scale."),
    ("🔗", "Source Credibility", "Incorporate source reputation scores and cross-referencing with fact-checking databases (Snopes, PolitiFact API)."),
    ("📊", "Advanced Explainability", "Add SHAP and LIME explanations for deeper interpretability, especially for non-linear models like XGBoost."),
    ("🐳", "Containerization", "Dockerize the application for easy deployment on cloud platforms (AWS, GCP, Azure) with CI/CD pipelines."),
]

imp_cols = st.columns(3)
for i, (icon, title, desc) in enumerate(improvements):
    with imp_cols[i % 3]:
        st.markdown(f"""
        <div class="info-card" style="min-height: 180px;">
            <div style="font-size: 1.8rem; margin-bottom: 10px;">{icon}</div>
            <div style="font-size: 1rem; font-weight: 700; color: #00d4ff; margin-bottom: 8px;">{title}</div>
            <div style="color: #8899aa; font-size: 0.83rem; line-height: 1.6;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Author / Contact ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">👤 Author & Contact</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-card" style="text-align: center;">
    <div style="font-size: 3rem; margin-bottom: 12px;">👨‍💻</div>
    <div style="font-size: 1.3rem; font-weight: 700; color: #e0e0e0; margin-bottom: 4px;">
        Your Name Here
    </div>
    <div style="color: #8899aa; font-size: 0.9rem; margin-bottom: 20px;">
        Machine Learning Engineer · NLP Enthusiast
    </div>
    <div style="display: flex; justify-content: center; gap: 16px; flex-wrap: wrap;">
        <a href="https://github.com" target="_blank" style="text-decoration: none;">
            <span class="tech-badge">🐙 GitHub</span>
        </a>
        <a href="https://linkedin.com" target="_blank" style="text-decoration: none;">
            <span class="tech-badge tech-badge-green">💼 LinkedIn</span>
        </a>
        <a href="mailto:your.email@example.com" style="text-decoration: none;">
            <span class="tech-badge tech-badge-orange">📧 Email</span>
        </a>
        <a href="#" style="text-decoration: none;">
            <span class="tech-badge tech-badge-purple">🌐 Portfolio</span>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Quick Start Guide ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚡ Quick Start Guide</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    <div style="color: #00d4ff; font-size: 1.1rem; font-weight: 700; margin-bottom: 16px;">
        🚀 Getting Started
    </div>
    <div style="color: #8899aa; font-size: 0.9rem; line-height: 2;">
        <code style="background: rgba(0,212,255,0.1); padding: 4px 10px; border-radius: 6px;
            color: #00d4ff; font-size: 0.85rem;">1.</code>
        &nbsp; Install dependencies: 
        <code style="color: #00ff88;">pip install -r requirements.txt</code><br>
        
        <code style="background: rgba(0,212,255,0.1); padding: 4px 10px; border-radius: 6px;
            color: #00d4ff; font-size: 0.85rem;">2.</code>
        &nbsp; Place dataset files in <code style="color: #00ff88;">dataset/</code> directory<br>
        
        <code style="background: rgba(0,212,255,0.1); padding: 4px 10px; border-radius: 6px;
            color: #00d4ff; font-size: 0.85rem;">3.</code>
        &nbsp; Train models: 
        <code style="color: #00ff88;">python main.py</code><br>
        
        <code style="background: rgba(0,212,255,0.1); padding: 4px 10px; border-radius: 6px;
            color: #00d4ff; font-size: 0.85rem;">4.</code>
        &nbsp; Launch dashboard: 
        <code style="color: #00ff88;">streamlit run app.py</code><br>
        
        <code style="background: rgba(0,212,255,0.1); padding: 4px 10px; border-radius: 6px;
            color: #00d4ff; font-size: 0.85rem;">5.</code>
        &nbsp; Navigate to <code style="color: #00ff88;">http://localhost:8501</code> in your browser
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 40px 20px 20px; margin-top: 60px;
    border-top: 1px solid rgba(0, 212, 255, 0.1);">
    <div style="color: #556677; font-size: 0.85rem; margin-bottom: 8px;">
        <span style="color: #00d4ff;">📰 Fake News Detection System</span> · v1.0.0
    </div>
    <div style="color: #445566; font-size: 0.75rem;">
        Built with ❤️ using Python, Streamlit, Scikit-learn & Plotly
    </div>
    <div style="color: #334455; font-size: 0.7rem; margin-top: 8px;">
        © 2025 All rights reserved.
    </div>
</div>
""", unsafe_allow_html=True)
