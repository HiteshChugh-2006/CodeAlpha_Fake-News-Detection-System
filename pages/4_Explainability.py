"""
Fake News Detection - Model Explainability Page
=================================================
Understand model decisions through feature importance,
word contributions, and interactive text analysis.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import sys
import re

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explainability | Fake News Detector",
    page_icon="🧠",
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
.kpi-card {
    background: linear-gradient(135deg, #1e2a3a 0%, #16213e 100%);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 16px; padding: 28px 24px; text-align: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255,255,255,0.05);
    position: relative; overflow: hidden; transition: all 0.3s ease;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #00d4ff, #00ff88); border-radius: 16px 16px 0 0;
}
.kpi-card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(0,212,255,0.15); }
.kpi-icon { font-size: 2.2rem; margin-bottom: 8px; display: block; }
.kpi-value { font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #00d4ff, #00ff88);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    line-height: 1.2; margin: 4px 0; }
.kpi-label { font-size: 0.85rem; color: #8899aa; text-transform: uppercase;
    letter-spacing: 1.5px; font-weight: 600; margin-top: 4px; }
.section-header { font-size: 1.5rem; font-weight: 700; color: #e0e0e0;
    margin: 40px 0 20px; padding-bottom: 12px;
    border-bottom: 2px solid rgba(0, 212, 255, 0.2); }
.gradient-divider { height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, #00ff88, transparent);
    border: none; margin: 30px 0; border-radius: 2px; }

/* Word contribution highlighting */
.word-positive { background: rgba(255, 71, 87, 0.2); color: #ff4757;
    padding: 2px 6px; border-radius: 4px; font-weight: 600;
    border-bottom: 2px solid #ff4757; }
.word-negative { background: rgba(0, 255, 136, 0.2); color: #00ff88;
    padding: 2px 6px; border-radius: 4px; font-weight: 600;
    border-bottom: 2px solid #00ff88; }
.word-neutral { color: #8899aa; }
</style>
"""

st.markdown(SHARED_CSS, unsafe_allow_html=True)

# ─── Add project root to path ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ─── Load Model & Vectorizer ────────────────────────────────────────────────
@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained model and TF-IDF vectorizer."""
    try:
        import joblib
        model_dir = os.path.join(BASE_DIR, "models")
        model_path = os.path.join(model_dir, "best_model.joblib")
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")

        if not os.path.exists(model_path):
            return None, None, "Model file not found. Run python main.py first."
        if not os.path.exists(vectorizer_path):
            return None, None, "Vectorizer file not found. Run python main.py first."

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer, None
    except Exception as e:
        return None, None, str(e)


@st.cache_data(ttl=3600)
def get_feature_importance(_model, _vectorizer, top_n=20):
    """Extract feature importances from the model."""
    feature_names = _vectorizer.get_feature_names_out()

    # Try coef_ (linear models: LR, SVM, NB)
    if hasattr(_model, "coef_"):
        coefficients = _model.coef_[0] if len(_model.coef_.shape) > 1 else _model.coef_
        coefficients = np.array(coefficients).flatten()

        # Top positive = FAKE indicators
        top_fake_idx = coefficients.argsort()[-top_n:][::-1]
        fake_words = [(feature_names[i], coefficients[i]) for i in top_fake_idx]

        # Top negative = REAL indicators
        top_real_idx = coefficients.argsort()[:top_n]
        real_words = [(feature_names[i], coefficients[i]) for i in top_real_idx]

        return fake_words, real_words, "coefficient"

    # Try feature_importances_ (tree-based: XGBoost, RF)
    elif hasattr(_model, "feature_importances_"):
        importances = np.array(_model.feature_importances_).flatten()
        top_idx = importances.argsort()[-top_n * 2:][::-1]

        # For tree models, we can't directly split into FAKE/REAL
        # Show top features overall
        top_words = [(feature_names[i], importances[i]) for i in top_idx[:top_n]]
        return top_words, [], "importance"

    else:
        return [], [], "unknown"


def analyze_text_contributions(text, model, vectorizer):
    """Analyze which words in the input text contribute to the prediction."""
    feature_names = vectorizer.get_feature_names_out()
    text_vector = vectorizer.transform([text])

    # Get non-zero features in the text
    non_zero_indices = text_vector.nonzero()[1]

    contributions = []

    if hasattr(model, "coef_"):
        coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        coefficients = np.array(coefficients).flatten()

        for idx in non_zero_indices:
            word = feature_names[idx]
            tfidf_val = text_vector[0, idx]
            coef_val = coefficients[idx]
            contribution = tfidf_val * coef_val
            contributions.append((word, float(contribution), float(coef_val), float(tfidf_val)))

    elif hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_).flatten()
        for idx in non_zero_indices:
            word = feature_names[idx]
            tfidf_val = text_vector[0, idx]
            imp_val = importances[idx]
            contributions.append((word, float(tfidf_val * imp_val), float(imp_val), float(tfidf_val)))

    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0;">
        <div style="font-size: 3rem;">🧠</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #00d4ff;">Explainability</div>
        <div style="font-size: 0.75rem; color: #556677; margin-top: 4px;">MODEL REASONING</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 8px 0; color: #8899aa; font-size: 0.82rem; line-height: 1.8;">
        <strong style="color:#e0e0e0;">📖 How to read this</strong><br>
        • <span style="color:#ff4757;">Red bars</span> → Push toward FAKE<br>
        • <span style="color:#00ff88;">Green bars</span> → Push toward REAL<br>
        • Longer bars = stronger influence<br>
        • TF-IDF × coefficient = contribution
    </div>
    """, unsafe_allow_html=True)


# ─── Page Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 30px 0 10px;">
    <div style="font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #00ff88);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
        🧠 Model Explainability
    </div>
    <div style="color: #8899aa; font-size: 1rem; margin-top: 8px;">
        Understand why the model makes its predictions — see the words that matter most
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ─── Load Model ──────────────────────────────────────────────────────────────
model, vectorizer, load_error = load_model_and_vectorizer()

if model is None or vectorizer is None:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #2d1b1b, #1a1a2e);
        border: 1px solid rgba(255, 71, 87, 0.3); border-radius: 16px;
        padding: 40px; text-align: center; margin: 20px 0;">
        <div style="font-size: 3rem; margin-bottom: 16px;">🔧</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #ff4757; margin-bottom: 12px;">
            Model Not Available
        </div>
        <div style="color: #8899aa; font-size: 0.95rem; line-height: 1.7; max-width: 500px; margin: 0 auto;">
            Please run <code style="color:#00d4ff;">python main.py</code> first to train the model.
            The system needs <code style="color:#00d4ff;">best_model.joblib</code> and 
            <code style="color:#00d4ff;">tfidf_vectorizer.joblib</code> in the models/ directory.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if load_error:
        with st.expander("🔧 Error Details"):
            st.code(load_error)
    st.stop()


# ─── Global Feature Importance ───────────────────────────────────────────────
st.markdown('<div class="section-header">🔬 Global Feature Importance</div>', unsafe_allow_html=True)

top_n_global = st.slider("Number of top features to display", 10, 50, 20, 5)

try:
    result = get_feature_importance(model, vectorizer, top_n=top_n_global)

    if result[2] == "coefficient":
        fake_words, real_words, _ = result

        # KPI cards
        k1, k2, k3 = st.columns(3)
        with k1:
            feature_count = len(vectorizer.get_feature_names_out())
            st.markdown(f"""
            <div class="kpi-card">
                <span class="kpi-icon">🔤</span>
                <div class="kpi-value">{feature_count:,}</div>
                <div class="kpi-label">Total Features</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <span class="kpi-icon">🔴</span>
                <div class="kpi-value">{fake_words[0][0]}</div>
                <div class="kpi-label">Strongest Fake Indicator</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <span class="kpi-icon">🟢</span>
                <div class="kpi-value">{real_words[0][0]}</div>
                <div class="kpi-label">Strongest Real Indicator</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Two-column chart layout
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("""
            <div style="text-align: center; color: #ff4757; font-size: 1.1rem;
                font-weight: 700; margin-bottom: 8px;">
                🔴 Top Words → FAKE Prediction
            </div>
            """, unsafe_allow_html=True)

            fw_words, fw_scores = zip(*fake_words)
            fig_fake = go.Figure(go.Bar(
                x=list(fw_scores)[::-1],
                y=list(fw_words)[::-1],
                orientation="h",
                marker=dict(
                    color=list(fw_scores)[::-1],
                    colorscale=[[0, "#ff6b81"], [1, "#ff4757"]],
                ),
                text=[f" {s:.4f}" for s in list(fw_scores)[::-1]],
                textposition="outside",
                textfont=dict(color="#e0e0e0", size=10),
            ))
            fig_fake.update_layout(
                template="plotly_dark", height=max(500, top_n_global * 25),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Coefficient Value", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(tickfont=dict(size=11)),
                margin=dict(t=20, b=40, l=110, r=60),
            )
            st.plotly_chart(fig_fake, use_container_width=True)

        with chart_col2:
            st.markdown("""
            <div style="text-align: center; color: #00ff88; font-size: 1.1rem;
                font-weight: 700; margin-bottom: 8px;">
                🟢 Top Words → REAL Prediction
            </div>
            """, unsafe_allow_html=True)

            rw_words, rw_scores = zip(*real_words)
            fig_real = go.Figure(go.Bar(
                x=[abs(s) for s in list(rw_scores)[::-1]],
                y=list(rw_words)[::-1],
                orientation="h",
                marker=dict(
                    color=[abs(s) for s in list(rw_scores)[::-1]],
                    colorscale=[[0, "#00ffaa"], [1, "#00ff88"]],
                ),
                text=[f" {abs(s):.4f}" for s in list(rw_scores)[::-1]],
                textposition="outside",
                textfont=dict(color="#e0e0e0", size=10),
            ))
            fig_real.update_layout(
                template="plotly_dark", height=max(500, top_n_global * 25),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Coefficient Magnitude", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(tickfont=dict(size=11)),
                margin=dict(t=20, b=40, l=110, r=60),
            )
            st.plotly_chart(fig_real, use_container_width=True)

        # Combined butterfly chart
        st.markdown('<div class="section-header">🦋 Feature Importance Butterfly Chart</div>', unsafe_allow_html=True)

        # Combine top fake and real words
        combined = list(fake_words[:10]) + list(real_words[:10])
        combined.sort(key=lambda x: x[1])
        c_words, c_scores = zip(*combined)

        fig_butterfly = go.Figure(go.Bar(
            x=list(c_scores),
            y=list(c_words),
            orientation="h",
            marker_color=[
                "#ff4757" if s > 0 else "#00ff88" for s in c_scores
            ],
            text=[f" {s:.4f}" for s in c_scores],
            textposition="outside",
            textfont=dict(color="#e0e0e0", size=11),
        ))
        fig_butterfly.add_vline(x=0, line_width=2, line_color="#556677")
        fig_butterfly.update_layout(
            template="plotly_dark", height=600,
            title=dict(text="← REAL indicators   |   FAKE indicators →",
                      font=dict(size=14, color="#8899aa"), x=0.5),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Coefficient Value", gridcolor="rgba(255,255,255,0.05)",
                      zeroline=True, zerolinecolor="#556677"),
            yaxis=dict(tickfont=dict(size=12)),
            margin=dict(t=60, b=40, l=120, r=60),
        )
        st.plotly_chart(fig_butterfly, use_container_width=True)

    elif result[2] == "importance":
        top_words, _, _ = result

        st.markdown("""
        <div style="background: rgba(0, 212, 255, 0.08); border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 12px; padding: 16px; margin-bottom: 20px; color: #8899aa; font-size: 0.9rem;">
            ℹ️ Tree-based models (XGBoost) show overall feature importance rather than 
            directional (fake/real) indicators.
        </div>
        """, unsafe_allow_html=True)

        tw_words, tw_scores = zip(*top_words)
        fig_imp = go.Figure(go.Bar(
            x=list(tw_scores)[::-1],
            y=list(tw_words)[::-1],
            orientation="h",
            marker=dict(
                color=list(tw_scores)[::-1],
                colorscale=[[0, "#16213e"], [0.5, "#00d4ff"], [1, "#00ff88"]],
            ),
            text=[f" {s:.4f}" for s in list(tw_scores)[::-1]],
            textposition="outside",
            textfont=dict(color="#e0e0e0", size=11),
        ))
        fig_imp.update_layout(
            template="plotly_dark", height=max(500, len(top_words) * 25),
            title=dict(text="Feature Importance (Gini / Gain)",
                      font=dict(size=16, color="#e0e0e0")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Importance Score", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(tickfont=dict(size=12)),
            margin=dict(t=60, b=40, l=120, r=60),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    else:
        st.info("Feature importance extraction is not supported for this model type.")

except Exception as e:
    st.warning(f"Could not extract feature importance: {str(e)}")
    with st.expander("🔧 Error Details"):
        import traceback
        st.code(traceback.format_exc())


# ─── Interactive Text Analysis ───────────────────────────────────────────────
st.markdown('<div class="section-header">🔍 Interactive Text Analysis</div>', unsafe_allow_html=True)

st.markdown("""
<div style="background: rgba(0, 212, 255, 0.08); border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 12px; padding: 16px; margin-bottom: 20px; color: #8899aa; font-size: 0.9rem;">
    ✍️ Enter a news article below to see which words influence the prediction and by how much.
</div>
""", unsafe_allow_html=True)

explain_text = st.text_area(
    "Enter text to explain:",
    height=150,
    placeholder="Paste a news article to see word-level contribution analysis...",
    key="explain_input",
    label_visibility="collapsed",
)

if st.button("🔬 Analyze Contributions", type="primary", use_container_width=True):
    if explain_text.strip():
        with st.spinner("Analyzing word contributions..."):
            try:
                # Get prediction
                from src.predict import NewsPredictor
                predictor = NewsPredictor(model_dir=os.path.join(BASE_DIR, "models"))
                pred_result = predictor.predict(explain_text)

                prediction = pred_result.get("prediction", "Unknown")
                confidence = pred_result.get("confidence", 0.0)
                conf_pct = confidence * 100 if confidence <= 1 else confidence
                is_fake = prediction.upper() in ["FAKE", "1", 1]

                # Show prediction
                badge_class = "fake-badge" if is_fake else "real-badge"
                label_text = "FAKE NEWS" if is_fake else "REAL NEWS"
                icon = "🚫" if is_fake else "✅"

                st.markdown(f"""
                <div style="text-align: center; margin: 20px 0;">
                    <span style="display: inline-block; padding: 12px 30px; border-radius: 12px;
                        font-size: 1.3rem; font-weight: 800; letter-spacing: 1px;
                        {'background: linear-gradient(135deg, #ff4757, #ff6b81); color: #fff;' if is_fake else 'background: linear-gradient(135deg, #00ff88, #00d4ff); color: #0e1117;'}">
                        {icon} {label_text} — {conf_pct:.1f}% confidence
                    </span>
                </div>
                """, unsafe_allow_html=True)

                # Analyze contributions
                contributions = analyze_text_contributions(explain_text, model, vectorizer)

                if contributions:
                    top_contribs = contributions[:25]

                    # Contribution chart
                    st.markdown("#### 📊 Word Contribution Breakdown")

                    words = [c[0] for c in top_contribs]
                    contrib_vals = [c[1] for c in top_contribs]

                    fig_contrib = go.Figure(go.Bar(
                        x=contrib_vals,
                        y=words,
                        orientation="h",
                        marker_color=[
                            "#ff4757" if v > 0 else "#00ff88" for v in contrib_vals
                        ],
                        text=[f" {v:.4f}" for v in contrib_vals],
                        textposition="outside",
                        textfont=dict(color="#e0e0e0", size=10),
                    ))
                    fig_contrib.add_vline(x=0, line_width=2, line_color="#556677")
                    fig_contrib.update_layout(
                        template="plotly_dark", height=max(400, len(top_contribs) * 28),
                        title=dict(
                            text="← Pushes toward REAL  |  Pushes toward FAKE →",
                            font=dict(size=13, color="#8899aa"), x=0.5,
                        ),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(title="Contribution (TF-IDF × Coefficient)",
                                  gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
                        margin=dict(t=50, b=40, l=120, r=60),
                    )
                    st.plotly_chart(fig_contrib, use_container_width=True)

                    # Highlighted text
                    st.markdown("#### 🎨 Highlighted Text")

                    # Build word-color mapping
                    word_colors = {}
                    for word, contrib, _, _ in contributions:
                        if contrib > 0:
                            word_colors[word] = "word-positive"
                        elif contrib < 0:
                            word_colors[word] = "word-negative"

                    # Process text for highlighting
                    text_words = explain_text.split()
                    highlighted_parts = []
                    for w in text_words:
                        clean_w = re.sub(r"[^a-zA-Z]", "", w).lower()
                        if clean_w in word_colors:
                            css_class = word_colors[clean_w]
                            highlighted_parts.append(f'<span class="{css_class}">{w}</span>')
                        else:
                            highlighted_parts.append(f'<span class="word-neutral">{w}</span>')

                    highlighted_html = " ".join(highlighted_parts)

                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1e2a3a, #16213e);
                        border: 1px solid rgba(0, 212, 255, 0.15); border-radius: 16px;
                        padding: 24px; line-height: 2; font-size: 0.95rem;">
                        <div style="margin-bottom: 12px; font-size: 0.8rem; color: #556677;">
                            <span class="word-positive">Red</span> = pushes FAKE &nbsp;|&nbsp;
                            <span class="word-negative">Green</span> = pushes REAL &nbsp;|&nbsp;
                            <span class="word-neutral">Gray</span> = not in vocabulary
                        </div>
                        {highlighted_html}
                    </div>
                    """, unsafe_allow_html=True)

                    # Detailed table
                    with st.expander("📋 Detailed Contribution Table"):
                        import pandas as pd
                        contrib_df = pd.DataFrame(
                            [(w, f"{c:.6f}", f"{coef:.6f}", f"{tfidf:.6f}")
                             for w, c, coef, tfidf in top_contribs],
                            columns=["Word", "Contribution", "Coefficient", "TF-IDF"],
                        )
                        st.dataframe(contrib_df, use_container_width=True, height=400)
                else:
                    st.info("No matching features found in the input text.")

            except ImportError:
                st.warning("Could not import NewsPredictor. Showing contribution analysis only.")
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                with st.expander("🔧 Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Please enter some text to analyze.")


# ─── Methodology ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📚 Methodology</div>', unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #1e2a3a, #16213e);
    border: 1px solid rgba(0, 212, 255, 0.15); border-radius: 16px;
    padding: 28px; color: #8899aa; font-size: 0.9rem; line-height: 1.8;">
    
<strong style="color: #e0e0e0; font-size: 1rem;">How Explainability Works</strong><br><br>

<strong style="color: #00d4ff;">For Linear Models (Logistic Regression, Linear SVC):</strong><br>
Each word (feature) has a learned coefficient. Positive coefficients indicate FAKE news, 
while negative coefficients indicate REAL news. The magnitude reflects the strength of the signal.
When analyzing a specific text, the contribution of each word is: <code style="color:#00ff88;">TF-IDF score × coefficient</code>.<br><br>

<strong style="color: #00d4ff;">For Naive Bayes:</strong><br>
Log-probability ratios are used. Higher log-probability for a class means the word is 
more indicative of that class. The coefficients represent log-odds.<br><br>

<strong style="color: #00d4ff;">For Tree-Based Models (XGBoost):</strong><br>
Feature importance is measured by information gain — how much each feature reduces 
uncertainty when used for splitting. This shows overall importance but not directionality.

</div>
""", unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 40px 20px 20px; margin-top: 60px;
    border-top: 1px solid rgba(0, 212, 255, 0.1); color: #556677; font-size: 0.85rem;">
    📰 Fake News Detection System · Model Explainability Module
</div>
""", unsafe_allow_html=True)
