"""
Fake News Detection - Real-Time News Analysis Page
====================================================
Analyze any news article with AI-powered classification,
confidence gauges, and word-level explainability.
"""

import streamlit as st
import plotly.graph_objects as go
import os
import sys

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="News Detection | Fake News Detector",
    page_icon="🔍",
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
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #00d4ff, #00ff88); border-radius: 16px 16px 0 0;
}
.kpi-value { font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #00d4ff, #00ff88);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.kpi-label { font-size: 0.85rem; color: #8899aa; text-transform: uppercase;
    letter-spacing: 1.5px; font-weight: 600; margin-top: 4px; }
.section-header { font-size: 1.5rem; font-weight: 700; color: #e0e0e0;
    margin: 40px 0 20px; padding-bottom: 12px;
    border-bottom: 2px solid rgba(0, 212, 255, 0.2); }
.gradient-divider { height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, #00ff88, transparent);
    border: none; margin: 30px 0; border-radius: 2px; }

/* Prediction badges */
.prediction-badge {
    display: inline-flex; align-items: center; gap: 12px;
    padding: 20px 40px; border-radius: 16px; font-size: 2rem; font-weight: 800;
    letter-spacing: 2px; text-transform: uppercase;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.fake-badge {
    background: linear-gradient(135deg, #ff4757, #ff6b81);
    color: #fff; border: 2px solid rgba(255,71,87,0.5);
    box-shadow: 0 8px 32px rgba(255,71,87,0.3);
}
.real-badge {
    background: linear-gradient(135deg, #00ff88, #00d4ff);
    color: #0e1117; border: 2px solid rgba(0,255,136,0.5);
    box-shadow: 0 8px 32px rgba(0,255,136,0.3);
}

/* Word tags */
.word-tag {
    display: inline-block; padding: 6px 16px; margin: 4px;
    border-radius: 50px; font-size: 0.85rem; font-weight: 600;
    background: rgba(0, 212, 255, 0.12); color: #00d4ff;
    border: 1px solid rgba(0, 212, 255, 0.3);
    transition: all 0.2s ease;
}
.word-tag:hover { background: rgba(0, 212, 255, 0.25); transform: scale(1.05); }
.word-tag-fake {
    background: rgba(255, 71, 87, 0.12); color: #ff4757;
    border-color: rgba(255, 71, 87, 0.3);
}
.word-tag-real {
    background: rgba(0, 255, 136, 0.12); color: #00ff88;
    border-color: rgba(0, 255, 136, 0.3);
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #1e2a3a 0%, #16213e 100%);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 20px; padding: 40px; text-align: center;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
}

/* Sample buttons */
.sample-card {
    background: linear-gradient(135deg, #1e2a3a 0%, #16213e 100%);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 12px; padding: 16px; margin-bottom: 10px;
    cursor: pointer; transition: all 0.3s ease;
}
.sample-card:hover { border-color: rgba(0, 212, 255, 0.4);
    box-shadow: 0 4px 16px rgba(0, 212, 255, 0.1); }
</style>
"""

st.markdown(SHARED_CSS, unsafe_allow_html=True)

# ─── Add project root to path ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ─── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    """Load the NewsPredictor with trained model."""
    try:
        from src.predict import NewsPredictor
        model_dir = os.path.join(BASE_DIR, "models")
        model_path = os.path.join(model_dir, "best_model.joblib")
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
        predictor = NewsPredictor(model_path=model_path, vectorizer_path=vectorizer_path)
        return predictor, None
    except Exception as e:
        return None, str(e)


import random

# ─── Sample Articles ────────────────────────────────────────────────────────
SAMPLE_ARTICLES = {
    "🟢 Real News Example": [
        "WASHINGTON (Reuters) - The Federal Reserve announced on Wednesday that it would raise interest rates "
        "by 0.25 percentage points, citing persistent inflation concerns. Fed Chair Jerome "
        "Powell stated during the press conference that the central bank remains committed "
        "to achieving its 2% inflation target. Markets reacted with modest gains as "
        "investors had largely anticipated the move.",
        
        "LONDON (Reuters) - Global markets rallied on Friday following a strong jobs report "
        "from the United States. European indices closed up by more than 1.5%, led by "
        "technology and healthcare sectors. Analysts say the positive sentiment could "
        "continue into next week depending on upcoming corporate earnings calls.",
        
        "NEW YORK (Reuters) - The city council voted unanimously to approve the new "
        "infrastructure spending package. The $4 billion initiative will focus on repairing "
        "aging bridges, upgrading the public transit system, and expanding broadband access "
        "in underserved communities. Construction is slated to begin early next year."
    ],
    "🔴 Fake News Example": [
        "BREAKING: Scientists at a secret underground lab have discovered that drinking "
        "bleach can cure all known diseases! The government has been hiding this miracle "
        "cure for decades to protect Big Pharma profits. Share this before they delete it! "
        "Thousands of doctors have been silenced for trying to reveal the truth. Wake up!",
        
        "SHOCKING: The President was caught on hot mic admitting that the earth is actually "
        "flat! Insiders say that NASA has been faking satellite images for years to keep "
        "the public completely blind. Read the leaked documents here before the deep state "
        "takes this page offline. You won't believe what else they're hiding!",
        
        "EXCLUSIVE: Top politicians are secretly using taxpayer money to build luxury "
        "bunkers on Mars! A whistleblower who recently escaped from a top-secret facility "
        "provided us with blurry photos of the spaceships. The mainstream media is completely "
        "ignoring this massive story because they are in on the conspiracy."
    ],
    "🟡 Ambiguous Example": [
        "A new study published online claims that a popular vitamin supplement may have "
        "unexpected health benefits that doctors don't want you to know about. The research, "
        "which has not been peer-reviewed, suggests that taking high doses could reverse "
        "aging. Several wellness influencers have already started promoting the findings.",
        
        "WASHINGTON (Reuters) - Unconfirmed reports circulated late Tuesday regarding a "
        "potential major shakeup in the cabinet. Anonymous sources suggested that two senior "
        "officials might resign by the end of the week due to internal disagreements over "
        "policy direction. The White House has declined to comment on the rumors.",
        
        "Local officials are investigating reports of strange lights in the sky over the city. "
        "Some residents claim they saw a massive triangular object hovering silently above the "
        "power plant for several minutes before it disappeared. The military base nearby stated "
        "they had no aircraft in the area at the time."
    ],
}


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0;">
        <div style="font-size: 3rem;">🔍</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #00d4ff;">News Detection</div>
        <div style="font-size: 0.75rem; color: #556677; margin-top: 4px;">REAL-TIME ANALYSIS</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 8px 0; color: #8899aa; font-size: 0.82rem; line-height: 1.8;">
        <strong style="color:#e0e0e0;">💡 Tips</strong><br>
        • Paste a complete article for best results<br>
        • Longer texts give more reliable predictions<br>
        • Check the confidence score<br>
        • Review important words for reasoning
    </div>
    """, unsafe_allow_html=True)


# ─── Page Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 30px 0 10px;">
    <div style="font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #00ff88);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
        🔍 Real-Time News Detection
    </div>
    <div style="color: #8899aa; font-size: 1rem; margin-top: 8px;">
        Paste any news article below and let AI determine its authenticity
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ─── Check Model Availability ───────────────────────────────────────────────
predictor, load_error = load_predictor()

if predictor is None:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2d1b1b, #1a1a2e);
        border: 1px solid rgba(255, 71, 87, 0.3); border-radius: 16px;
        padding: 40px; text-align: center; margin: 20px 0;">
        <div style="font-size: 3rem; margin-bottom: 16px;">⚠️</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #ff4757; margin-bottom: 12px;">
            Model Not Found
        </div>
        <div style="color: #8899aa; font-size: 0.95rem; line-height: 1.7; max-width: 500px; margin: 0 auto;">
            The trained model has not been loaded yet. Please run <code style="color:#00d4ff;">python main.py</code> 
            first to train the model and generate the necessary files.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if load_error:
        with st.expander("🔧 Error Details"):
            st.code(load_error)

# ─── Sample Articles ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Quick Test with Sample Articles</div>', unsafe_allow_html=True)

sample_cols = st.columns(3)

for i, (label, texts) in enumerate(SAMPLE_ARTICLES.items()):
    with sample_cols[i]:
        if st.button(label, key=f"sample_{i}", use_container_width=True):
            st.session_state.news_input = random.choice(texts)

# ─── Input Area ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">✍️ Enter News Article</div>', unsafe_allow_html=True)

news_text = st.text_area(
    "Paste your news article here:",
    height=200,
    placeholder="Enter or paste a news article to analyze...",
    key="news_input",
    label_visibility="collapsed",
)

col_btn, col_clear = st.columns([3, 1])
with col_btn:
    analyze_btn = st.button(
        "🚀 Analyze Article",
        type="primary",
        use_container_width=True,
        disabled=(predictor is None),
    )
with col_clear:
    if st.button("🗑️ Clear", use_container_width=True):
        st.rerun()


# ─── Analysis ────────────────────────────────────────────────────────────────
if analyze_btn and news_text.strip() and predictor is not None:
    with st.spinner("🔄 Analyzing article..."):
        try:
            result = predictor.predict(news_text)
            prediction = result.get("prediction", "Unknown")
            confidence = result.get("confidence", 0.0)
            probabilities = result.get("probabilities", {})

            # Determine label
            is_fake = prediction.upper() in ["FAKE", "1", 1]
            label_text = "FAKE NEWS" if is_fake else "REAL NEWS"
            badge_class = "fake-badge" if is_fake else "real-badge"
            icon = "🚫" if is_fake else "✅"
            conf_pct = confidence * 100 if confidence <= 1 else confidence

            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">📋 Analysis Results</div>', unsafe_allow_html=True)

            # ── Main Result Card ──
            st.markdown(f"""
            <div class="result-card">
                <div style="margin-bottom: 20px;">
                    <span class="prediction-badge {badge_class}">
                        {icon} {label_text}
                    </span>
                </div>
                <div style="color: #8899aa; font-size: 0.95rem;">
                    Confidence: <strong style="color: #00d4ff;">{conf_pct:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Detailed Metrics Row ──
            mc1, mc2, mc3 = st.columns(3)

            # Gauge chart
            with mc1:
                gauge_color = "#ff4757" if is_fake else "#00ff88"
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=conf_pct,
                    number={"suffix": "%", "font": {"size": 42, "color": "#e0e0e0"}},
                    title={"text": "Confidence Score", "font": {"size": 16, "color": "#8899aa"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#556677",
                                 "tickfont": {"color": "#556677"}},
                        "bar": {"color": gauge_color, "thickness": 0.3},
                        "bgcolor": "#1e2a3a",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 50], "color": "rgba(255,71,87,0.1)"},
                            {"range": [50, 75], "color": "rgba(255,165,0,0.1)"},
                            {"range": [75, 100], "color": "rgba(0,255,136,0.1)"},
                        ],
                        "threshold": {
                            "line": {"color": "#00d4ff", "width": 3},
                            "thickness": 0.8,
                            "value": conf_pct,
                        },
                    },
                ))
                fig_gauge.update_layout(
                    height=300, margin=dict(t=60, b=20, l=40, r=40),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#e0e0e0"},
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Reliability meter
            with mc2:
                st.markdown("""
                <div class="kpi-card" style="padding: 30px 20px; min-height: 260px;">
                    <div style="font-size: 1rem; color: #8899aa; margin-bottom: 16px;
                        text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">
                        Reliability Assessment
                    </div>
                """, unsafe_allow_html=True)

                if conf_pct >= 85:
                    reliability = "Very High"
                    rel_color = "#00ff88"
                    rel_icon = "🟢"
                elif conf_pct >= 70:
                    reliability = "High"
                    rel_color = "#00d4ff"
                    rel_icon = "🔵"
                elif conf_pct >= 55:
                    reliability = "Moderate"
                    rel_color = "#ffa500"
                    rel_icon = "🟡"
                else:
                    reliability = "Low"
                    rel_color = "#ff4757"
                    rel_icon = "🔴"

                st.markdown(f"""
                    <div style="font-size: 2.2rem; margin: 12px 0;">{rel_icon}</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: {rel_color};">
                        {reliability}
                    </div>
                    <div style="margin-top: 16px;">
                """, unsafe_allow_html=True)

                st.progress(min(conf_pct / 100, 1.0))

                st.markdown("</div></div>", unsafe_allow_html=True)

            # Probabilities breakdown
            with mc3:
                if probabilities:
                    prob_fake = probabilities.get("FAKE", probabilities.get("1", probabilities.get(1, 0)))
                    prob_real = probabilities.get("REAL", probabilities.get("0", probabilities.get(0, 0)))

                    if isinstance(prob_fake, (int, float)) and isinstance(prob_real, (int, float)):
                        pf = prob_fake * 100 if prob_fake <= 1 else prob_fake
                        pr = prob_real * 100 if prob_real <= 1 else prob_real
                    else:
                        pf, pr = (conf_pct, 100 - conf_pct) if is_fake else (100 - conf_pct, conf_pct)
                else:
                    pf, pr = (conf_pct, 100 - conf_pct) if is_fake else (100 - conf_pct, conf_pct)

                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=[pr], y=["Prediction"], orientation="h",
                    name="Real", marker_color="#00ff88",
                    text=[f"{pr:.1f}%"], textposition="inside",
                    textfont={"size": 14, "color": "#0e1117", "family": "Inter"},
                ))
                fig_bar.add_trace(go.Bar(
                    x=[pf], y=["Prediction"], orientation="h",
                    name="Fake", marker_color="#ff4757",
                    text=[f"{pf:.1f}%"], textposition="inside",
                    textfont={"size": 14, "color": "#ffffff", "family": "Inter"},
                ))
                fig_bar.update_layout(
                    barmode="stack", template="plotly_dark", height=300,
                    title={"text": "Class Probabilities", "font": {"size": 16, "color": "#8899aa"}},
                    margin=dict(t=60, b=20, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=True, legend=dict(orientation="h", y=-0.15,
                        font={"color": "#8899aa"}),
                    yaxis=dict(visible=False),
                    xaxis=dict(visible=False, range=[0, 100]),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # ── Important Words ──
            st.markdown('<div class="section-header">🔤 Key Influential Words</div>', unsafe_allow_html=True)
            try:
                important_words = predictor.get_important_words(news_text, top_n=15)
                if important_words:
                    tag_class = "word-tag-fake" if is_fake else "word-tag-real"
                    # Handle both list of strings and list of tuples
                    words_html = ""
                    if isinstance(important_words[0], (list, tuple)):
                        for word, score in important_words:
                            words_html += f'<span class="word-tag {tag_class}">{word} ({score:.3f})</span>'
                    else:
                        for word in important_words:
                            words_html += f'<span class="word-tag {tag_class}">{word}</span>'

                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1e2a3a, #16213e);
                        border: 1px solid rgba(0, 212, 255, 0.15); border-radius: 16px;
                        padding: 24px; text-align: center;">
                        <div style="color: #8899aa; font-size: 0.85rem; margin-bottom: 16px;
                            text-transform: uppercase; letter-spacing: 1px;">
                            Words that most influenced the prediction
                        </div>
                        <div>{words_html}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No important words could be extracted for this article.")
            except Exception as e:
                st.info("Word importance analysis is not available for this model type.")
                with st.expander("Details"):
                    st.code(str(e))

            # ── Prediction Details Expander ──
            with st.expander("📄 Full Prediction Details"):
                det1, det2 = st.columns(2)
                with det1:
                    st.markdown("**Prediction:**")
                    st.write(prediction)
                    st.markdown("**Confidence:**")
                    st.write(f"{conf_pct:.2f}%")
                with det2:
                    st.markdown("**Raw Probabilities:**")
                    st.json(probabilities if probabilities else {"info": "Not available"})
                st.markdown("**Input Text Length:**")
                st.write(f"{len(news_text):,} characters | ~{len(news_text.split()):,} words")

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            with st.expander("🔧 Error Details"):
                import traceback
                st.code(traceback.format_exc())

elif analyze_btn and not news_text.strip():
    st.warning("⚠️ Please enter some text to analyze.")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 40px 20px 20px; margin-top: 60px;
    border-top: 1px solid rgba(0, 212, 255, 0.1); color: #556677; font-size: 0.85rem;">
    📰 Fake News Detection System · Real-Time Analysis Module
</div>
""", unsafe_allow_html=True)
