"""
Fake News Detection System - Main Dashboard
=============================================
Professional Streamlit dashboard with dark theme, KPI cards,
and comprehensive navigation.
"""

import streamlit as st
import pandas as pd
import os
import json

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Shared CSS ──────────────────────────────────────────────────────────────
SHARED_CSS = """
<style>
/* ===== Global Dark Theme ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 50%, #0e1117 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid rgba(0, 212, 255, 0.1);
}

/* ===== KPI Cards ===== */
.kpi-card {
    background: linear-gradient(135deg, #1e2a3a 0%, #16213e 100%);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00d4ff, #00ff88);
    border-radius: 16px 16px 0 0;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 212, 255, 0.15);
    border-color: rgba(0, 212, 255, 0.4);
}
.kpi-icon {
    font-size: 2.2rem;
    margin-bottom: 8px;
    display: block;
}
.kpi-value {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff, #00ff88);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin: 4px 0;
}
.kpi-label {
    font-size: 0.85rem;
    color: #8899aa;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-top: 4px;
}

/* ===== Hero Section ===== */
.hero-container {
    text-align: center;
    padding: 60px 20px 40px;
    position: relative;
}
.hero-title {
    font-size: 3.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00d4ff 0%, #00ff88 50%, #00d4ff 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradient-shift 4s ease-in-out infinite;
    margin-bottom: 12px;
    line-height: 1.1;
}
@keyframes gradient-shift {
    0%, 100% { background-position: 0% center; }
    50% { background-position: 100% center; }
}
.hero-subtitle {
    font-size: 1.2rem;
    color: #8899aa;
    max-width: 700px;
    margin: 0 auto;
    line-height: 1.7;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 50px;
    padding: 6px 20px;
    font-size: 0.8rem;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600;
    margin-bottom: 20px;
}

/* ===== Section Headers ===== */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e0e0e0;
    margin: 40px 0 20px;
    padding-bottom: 12px;
    border-bottom: 2px solid rgba(0, 212, 255, 0.2);
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ===== Feature Cards ===== */
.feature-card {
    background: linear-gradient(135deg, #1e2a3a 0%, #16213e 100%);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
    min-height: 220px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.feature-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 40px rgba(0, 212, 255, 0.12);
    border-color: rgba(0, 212, 255, 0.35);
}
.feature-icon {
    font-size: 2.5rem;
    margin-bottom: 16px;
}
.feature-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 10px;
}
.feature-desc {
    font-size: 0.88rem;
    color: #8899aa;
    line-height: 1.6;
}

/* ===== Footer ===== */
.footer {
    text-align: center;
    padding: 40px 20px 20px;
    margin-top: 60px;
    border-top: 1px solid rgba(0, 212, 255, 0.1);
    color: #556677;
    font-size: 0.85rem;
}
.footer a {
    color: #00d4ff;
    text-decoration: none;
}

/* ===== Scrollbar ===== */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #0e1117; }
::-webkit-scrollbar-thumb { background: #1e2a3a; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #2a3a4a; }

/* ===== Divider ===== */
.gradient-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, #00ff88, transparent);
    border: none;
    margin: 30px 0;
    border-radius: 2px;
}

/* ===== Streamlit Overrides ===== */
.stMetric { background: transparent; }
div[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    font-weight: 700;
    color: #00d4ff;
}
</style>
"""

st.markdown(SHARED_CSS, unsafe_allow_html=True)

# ─── Helper: Load Dataset Stats ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_dataset_stats():
    """Load dataset and compute statistics."""
    base = os.path.dirname(os.path.abspath(__file__))
    fake_path = os.path.join(base, "dataset", "Fake.csv")
    true_path = os.path.join(base, "dataset", "True.csv")
    stats = {"total": 0, "fake": 0, "real": 0, "loaded": False}
    try:
        if os.path.exists(fake_path) and os.path.exists(true_path):
            fake_df = pd.read_csv(fake_path)
            true_df = pd.read_csv(true_path)
            stats["fake"] = len(fake_df)
            stats["real"] = len(true_df)
            stats["total"] = stats["fake"] + stats["real"]
            stats["loaded"] = True
    except Exception:
        pass
    return stats


@st.cache_data(ttl=3600)
def load_model_metrics():
    """Load model comparison metrics."""
    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, "reports", "model_comparison.csv")
    metrics = {"accuracy": "N/A", "f1": "N/A", "models_trained": 0, "loaded": False}
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            metrics["models_trained"] = len(df)
            # Try common column names for accuracy & f1
            acc_col = None
            f1_col = None
            for c in df.columns:
                cl = c.lower().strip()
                if "accuracy" in cl or "acc" == cl:
                    acc_col = c
                if "f1" in cl:
                    f1_col = c
            if acc_col:
                metrics["accuracy"] = f"{df[acc_col].max():.2%}"
            if f1_col:
                metrics["f1"] = f"{df[f1_col].max():.2%}"
            metrics["loaded"] = True
    except Exception:
        pass
    return metrics


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0;">
        <div style="font-size: 3rem;">📰</div>
        <div style="font-size: 1.2rem; font-weight: 700; color: #00d4ff; margin-top: 8px;">
            Fake News Detector
        </div>
        <div style="font-size: 0.75rem; color: #556677; margin-top: 4px; letter-spacing: 1px;">
            ML-POWERED ANALYSIS
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 12px 0; color: #8899aa; font-size: 0.85rem; line-height: 1.8;">
        <strong style="color:#e0e0e0;">📌 Navigation</strong><br>
        🏠 <strong>Home</strong> — Dashboard overview<br>
        🔍 <strong>Detection</strong> — Analyze articles<br>
        📊 <strong>Analytics</strong> — Dataset insights<br>
        📈 <strong>Performance</strong> — Model metrics<br>
        🧠 <strong>Explainability</strong> — Model reasoning<br>
        ℹ️ <strong>About</strong> — Project details
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding: 10px 0; color: #445566; font-size: 0.75rem;">
        Built with ❤️ using<br>
        <span style="color:#00d4ff;">Streamlit</span> · 
        <span style="color:#00ff88;">Scikit-learn</span> · 
        <span style="color:#ff4757;">Plotly</span>
    </div>
    """, unsafe_allow_html=True)


# ─── Hero Section ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">⚡ Machine Learning Powered</div>
    <div class="hero-title">Fake News Detection<br>System</div>
    <div class="hero-subtitle">
        An intelligent NLP pipeline that leverages advanced machine learning models
        to classify news articles as <strong style="color:#00ff88;">Real</strong> or 
        <strong style="color:#ff4757;">Fake</strong> with high accuracy and full explainability.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ─── KPI Cards ───────────────────────────────────────────────────────────────
ds_stats = load_dataset_stats()
model_stats = load_model_metrics()

st.markdown('<div class="section-header">📊 System Overview</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

with k1:
    total = f"{ds_stats['total']:,}" if ds_stats["loaded"] else "—"
    st.markdown(f"""
    <div class="kpi-card">
        <span class="kpi-icon">📰</span>
        <div class="kpi-value">{total}</div>
        <div class="kpi-label">Total Articles</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    acc = model_stats["accuracy"]
    st.markdown(f"""
    <div class="kpi-card">
        <span class="kpi-icon">🎯</span>
        <div class="kpi-value">{acc}</div>
        <div class="kpi-label">Best Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    f1 = model_stats["f1"]
    st.markdown(f"""
    <div class="kpi-card">
        <span class="kpi-icon">⚡</span>
        <div class="kpi-value">{f1}</div>
        <div class="kpi-label">Best F1 Score</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    mt = model_stats["models_trained"] if model_stats["loaded"] else "—"
    st.markdown(f"""
    <div class="kpi-card">
        <span class="kpi-icon">🤖</span>
        <div class="kpi-value">{mt}</div>
        <div class="kpi-label">Models Trained</div>
    </div>
    """, unsafe_allow_html=True)


# ─── Quick Navigation Cards ─────────────────────────────────────────────────
st.markdown('<div class="section-header">🚀 Explore the Dashboard</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <div class="feature-title">Real-Time Detection</div>
        <div class="feature-desc">
            Paste any news article and get an instant AI-powered verdict 
            with confidence scores and word-level explanations.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Dataset Analytics</div>
        <div class="feature-desc">
            Explore the training data with interactive charts, word clouds,
            class distributions, and text length analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📈</div>
        <div class="feature-title">Model Performance</div>
        <div class="feature-desc">
            Compare accuracy, F1 scores, and ROC curves across all trained 
            models with detailed evaluation reports.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

c4, c5, c6 = st.columns(3)

with c4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">Explainability</div>
        <div class="feature-desc">
            Understand <em>why</em> the model made its decision. See which 
            words pushed the prediction towards FAKE or REAL.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚙️</div>
        <div class="feature-title">ML Pipeline</div>
        <div class="feature-desc">
            Full NLP pipeline: text preprocessing, TF-IDF features,
            hyperparameter tuning, and cross-validated training.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c6:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🏗️</div>
        <div class="feature-title">Architecture</div>
        <div class="feature-desc">
            Built with Scikit-learn, XGBoost, NLTK, and Streamlit — 
            a modern, modular, and extensible design.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── How It Works ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔄 How It Works</div>', unsafe_allow_html=True)

steps = [
    ("1️⃣", "Input", "Paste a news article into the detection page"),
    ("2️⃣", "Preprocess", "Text is cleaned, tokenized, and lemmatized using NLTK"),
    ("3️⃣", "Vectorize", "TF-IDF transforms text into numerical features"),
    ("4️⃣", "Predict", "Trained ML model classifies the article"),
    ("5️⃣", "Explain", "Top contributing words are highlighted"),
]

cols = st.columns(5)
for col, (icon, title, desc) in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div style="text-align:center; padding: 16px 8px;">
            <div style="font-size: 2rem; margin-bottom: 8px;">{icon}</div>
            <div style="font-size: 1rem; font-weight: 700; color: #00d4ff; margin-bottom: 6px;">{title}</div>
            <div style="font-size: 0.8rem; color: #8899aa; line-height: 1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div style="margin-bottom: 8px;">
        <span style="color: #00d4ff;">📰 Fake News Detection System</span> · v1.0.0
    </div>
    <div>
        Built with 
        <a href="https://streamlit.io" target="_blank">Streamlit</a> · 
        <a href="https://scikit-learn.org" target="_blank">Scikit-learn</a> · 
        <a href="https://plotly.com" target="_blank">Plotly</a>
    </div>
    <div style="margin-top: 8px; font-size: 0.75rem;">
        © 2025 Fake News Detection Project. All rights reserved.
    </div>
</div>
""", unsafe_allow_html=True)
