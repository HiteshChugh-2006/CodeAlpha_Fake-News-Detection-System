"""
Fake News Detection - Dataset Analytics Page
=============================================
Interactive exploration of the training dataset with charts,
word clouds, and statistical analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import re
from collections import Counter

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analytics | Fake News Detector",
    page_icon="📊",
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
    transition: all 0.3s ease;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #00d4ff, #00ff88); border-radius: 16px 16px 0 0;
}
.kpi-card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(0,212,255,0.15); }
.kpi-icon { font-size: 2.2rem; margin-bottom: 8px; display: block; }
.kpi-value { font-size: 2.4rem; font-weight: 800;
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
.chart-container {
    background: linear-gradient(135deg, #1e2a3a 0%, #16213e 100%);
    border: 1px solid rgba(0, 212, 255, 0.15); border-radius: 16px;
    padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
</style>
"""

st.markdown(SHARED_CSS, unsafe_allow_html=True)

# ─── Helper Functions ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "because", "but", "and", "or",
    "if", "while", "this", "that", "these", "those", "it", "its", "he",
    "she", "they", "them", "their", "we", "us", "our", "you", "your",
    "i", "me", "my", "said", "also", "one", "new", "s", "t", "would",
    "people", "like", "get", "make", "know", "say", "go", "see", "come",
    "take", "want", "look", "use", "find", "give", "tell", "think",
    "well", "even", "back", "still", "way", "time", "year", "two",
    "first", "last", "many", "much", "about", "up", "down", "now",
    "long", "made", "man", "day", "don", "re", "ll", "ve", "didn",
    "doesn", "won", "isn", "aren", "wasn", "weren", "haven", "hasn",
    "couldn", "wouldn", "shouldn", "mustn",
}


@st.cache_data(ttl=3600)
def load_dataset():
    """Load fake and true news datasets."""
    fake_path = os.path.join(BASE_DIR, "dataset", "Fake.csv")
    true_path = os.path.join(BASE_DIR, "dataset", "True.csv")

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        return None, "Dataset files not found."

    try:
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)

        fake_df["label"] = "Fake"
        true_df["label"] = "Real"

        # Add text length column - handle different column names
        text_col = None
        for col in ["text", "Text", "content", "Content", "article", "Article"]:
            if col in fake_df.columns:
                text_col = col
                break

        if text_col is None:
            # Use the last column as text column
            text_col = fake_df.columns[-1] if len(fake_df.columns) > 1 else fake_df.columns[0]

        df = pd.concat([fake_df, true_df], ignore_index=True)
        df["text_length"] = df[text_col].astype(str).apply(len)
        df["word_count"] = df[text_col].astype(str).apply(lambda x: len(x.split()))

        return df, text_col
    except Exception as e:
        return None, str(e)


def get_word_frequencies(texts, top_n=20):
    """Get most common words after basic cleaning."""
    all_words = []
    for text in texts:
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
        all_words.extend(words)
    return Counter(all_words).most_common(top_n)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0;">
        <div style="font-size: 3rem;">📊</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #00d4ff;">Dataset Analytics</div>
        <div style="font-size: 0.75rem; color: #556677; margin-top: 4px;">DATA EXPLORATION</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ─── Page Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 30px 0 10px;">
    <div style="font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #00ff88);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
        📊 Dataset Analytics
    </div>
    <div style="color: #8899aa; font-size: 1rem; margin-top: 8px;">
        Explore and understand the training data through interactive visualizations
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ─── Load Data ───────────────────────────────────────────────────────────────
result = load_dataset()

if result[0] is None:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #2d1b1b, #1a1a2e);
        border: 1px solid rgba(255, 71, 87, 0.3); border-radius: 16px;
        padding: 40px; text-align: center; margin: 20px 0;">
        <div style="font-size: 3rem; margin-bottom: 16px;">📂</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #ff4757; margin-bottom: 12px;">
            Dataset Not Found
        </div>
        <div style="color: #8899aa; font-size: 0.95rem; line-height: 1.7; max-width: 500px; margin: 0 auto;">
            Please ensure <code style="color:#00d4ff;">Fake.csv</code> and 
            <code style="color:#00d4ff;">True.csv</code> are in the 
            <code style="color:#00d4ff;">dataset/</code> directory.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if isinstance(result[1], str) and result[1] != "Dataset files not found.":
        with st.expander("🔧 Error Details"):
            st.code(result[1])
    st.stop()

df, text_col = result

# ─── KPI Row ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Dataset Overview</div>', unsafe_allow_html=True)

total = len(df)
fake_count = len(df[df["label"] == "Fake"])
real_count = len(df[df["label"] == "Real"])
avg_length = int(df["word_count"].mean())

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <span class="kpi-icon">📰</span>
        <div class="kpi-value">{total:,}</div>
        <div class="kpi-label">Total Articles</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <span class="kpi-icon">🔴</span>
        <div class="kpi-value">{fake_count:,}</div>
        <div class="kpi-label">Fake Articles</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <span class="kpi-icon">🟢</span>
        <div class="kpi-value">{real_count:,}</div>
        <div class="kpi-label">Real Articles</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <span class="kpi-icon">📝</span>
        <div class="kpi-value">{avg_length:,}</div>
        <div class="kpi-label">Avg Words/Article</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Class Distribution ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">🎯 Class Distribution</div>', unsafe_allow_html=True)

dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    fig_pie = px.pie(
        df, names="label", title="Real vs Fake News Distribution",
        color="label",
        color_discrete_map={"Fake": "#ff4757", "Real": "#00ff88"},
        hole=0.45,
    )
    fig_pie.update_traces(
        textposition="inside", textinfo="percent+label+value",
        textfont=dict(size=14, color="#ffffff"),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>",
    )
    fig_pie.update_layout(
        template="plotly_dark", height=500,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=16, color="#8899aa")),
        legend=dict(font=dict(size=13, color="#8899aa")),
        margin=dict(t=60, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with dist_col2:
    dist_data = df["label"].value_counts().reset_index()
    dist_data.columns = ["Label", "Count"]
    fig_bar = px.bar(
        dist_data, x="Label", y="Count", color="Label",
        color_discrete_map={"Fake": "#ff4757", "Real": "#00ff88"},
        title="Article Count by Class",
        text="Count",
    )
    fig_bar.update_traces(textposition="outside", textfont=dict(size=14, color="#e0e0e0"))
    fig_bar.update_layout(
        template="plotly_dark", height=500, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=16, color="#8899aa")),
        xaxis=dict(title="", tickfont=dict(size=14)),
        yaxis=dict(title="Number of Articles", gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(t=60, b=40, l=60, r=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ─── Article Length Distribution ─────────────────────────────────────────────
st.markdown('<div class="section-header">📏 Article Length Distribution</div>', unsafe_allow_html=True)

len_col1, len_col2 = st.columns(2)

with len_col1:
    fig_hist = px.histogram(
        df, x="word_count", color="label", barmode="overlay",
        color_discrete_map={"Fake": "#ff4757", "Real": "#00ff88"},
        title="Word Count Distribution by Class",
        labels={"word_count": "Word Count", "label": "Class"},
        nbins=80, opacity=0.7,
    )
    fig_hist.update_layout(
        template="plotly_dark", height=500,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=16, color="#8899aa")),
        xaxis=dict(title="Word Count", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(font=dict(size=13, color="#8899aa")),
        margin=dict(t=60, b=40, l=60, r=20),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with len_col2:
    fig_box = px.box(
        df, x="label", y="word_count", color="label",
        color_discrete_map={"Fake": "#ff4757", "Real": "#00ff88"},
        title="Word Count Statistics by Class",
        labels={"word_count": "Word Count", "label": "Class"},
    )
    fig_box.update_layout(
        template="plotly_dark", height=500, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=16, color="#8899aa")),
        xaxis=dict(title="", tickfont=dict(size=14)),
        yaxis=dict(title="Word Count", gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(t=60, b=40, l=60, r=20),
    )
    st.plotly_chart(fig_box, use_container_width=True)


# ─── Top Words ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔤 Most Common Words</div>', unsafe_allow_html=True)

tw_col1, tw_col2 = st.columns(2)

with tw_col1:
    fake_texts = df[df["label"] == "Fake"][text_col].dropna().tolist()
    fake_words = get_word_frequencies(fake_texts, top_n=20)
    if fake_words:
        words, counts = zip(*fake_words)
        fig_fw = go.Figure(go.Bar(
            x=list(counts)[::-1], y=list(words)[::-1], orientation="h",
            marker=dict(
                color=list(counts)[::-1],
                colorscale=[[0, "#ff6b81"], [1, "#ff4757"]],
            ),
            text=[f" {c:,}" for c in list(counts)[::-1]],
            textposition="outside",
            textfont=dict(color="#e0e0e0", size=11),
        ))
        fig_fw.update_layout(
            template="plotly_dark", height=600, title="Top 20 Words in Fake News",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title_font=dict(size=16, color="#ff4757"),
            xaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(tickfont=dict(size=12)),
            margin=dict(t=60, b=40, l=100, r=60),
        )
        st.plotly_chart(fig_fw, use_container_width=True)

with tw_col2:
    real_texts = df[df["label"] == "Real"][text_col].dropna().tolist()
    real_words = get_word_frequencies(real_texts, top_n=20)
    if real_words:
        words, counts = zip(*real_words)
        fig_rw = go.Figure(go.Bar(
            x=list(counts)[::-1], y=list(words)[::-1], orientation="h",
            marker=dict(
                color=list(counts)[::-1],
                colorscale=[[0, "#00ffaa"], [1, "#00ff88"]],
            ),
            text=[f" {c:,}" for c in list(counts)[::-1]],
            textposition="outside",
            textfont=dict(color="#e0e0e0", size=11),
        ))
        fig_rw.update_layout(
            template="plotly_dark", height=600, title="Top 20 Words in Real News",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title_font=dict(size=16, color="#00ff88"),
            xaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(tickfont=dict(size=12)),
            margin=dict(t=60, b=40, l=100, r=60),
        )
        st.plotly_chart(fig_rw, use_container_width=True)


# ─── Word Clouds ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">☁️ Word Clouds</div>', unsafe_allow_html=True)

try:
    from wordcloud import WordCloud
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wc_col1, wc_col2 = st.columns(2)

    with wc_col1:
        st.markdown("""
        <div style="text-align:center; color: #ff4757; font-size: 1.1rem;
            font-weight: 700; margin-bottom: 12px;">
            🔴 Fake News Word Cloud
        </div>
        """, unsafe_allow_html=True)
        fake_text = " ".join(fake_texts[:5000])
        fake_text_clean = re.sub(r"[^a-zA-Z\s]", "", fake_text.lower())
        wc_fake = WordCloud(
            width=800, height=400, background_color="#0e1117",
            colormap="Reds", max_words=100, stopwords=STOP_WORDS,
            contour_width=1, contour_color="#ff4757",
        ).generate(fake_text_clean)
        fig_wc_f, ax_f = plt.subplots(figsize=(10, 5))
        ax_f.imshow(wc_fake, interpolation="bilinear")
        ax_f.axis("off")
        fig_wc_f.patch.set_facecolor("#0e1117")
        st.pyplot(fig_wc_f)
        plt.close(fig_wc_f)

    with wc_col2:
        st.markdown("""
        <div style="text-align:center; color: #00ff88; font-size: 1.1rem;
            font-weight: 700; margin-bottom: 12px;">
            🟢 Real News Word Cloud
        </div>
        """, unsafe_allow_html=True)
        real_text = " ".join(real_texts[:5000])
        real_text_clean = re.sub(r"[^a-zA-Z\s]", "", real_text.lower())
        wc_real = WordCloud(
            width=800, height=400, background_color="#0e1117",
            colormap="Greens", max_words=100, stopwords=STOP_WORDS,
            contour_width=1, contour_color="#00ff88",
        ).generate(real_text_clean)
        fig_wc_r, ax_r = plt.subplots(figsize=(10, 5))
        ax_r.imshow(wc_real, interpolation="bilinear")
        ax_r.axis("off")
        fig_wc_r.patch.set_facecolor("#0e1117")
        st.pyplot(fig_wc_r)
        plt.close(fig_wc_r)

except ImportError:
    st.info("💡 Install `wordcloud` package to enable word cloud visualizations: `pip install wordcloud`")
except Exception as e:
    st.warning(f"Word cloud generation encountered an issue: {str(e)}")


# ─── Subject/Category Distribution ──────────────────────────────────────────
subject_col = None
for col in ["subject", "Subject", "category", "Category", "topic", "Topic"]:
    if col in df.columns:
        subject_col = col
        break

if subject_col:
    st.markdown('<div class="section-header">🏷️ Subject Distribution</div>', unsafe_allow_html=True)

    sub_dist = df.groupby([subject_col, "label"]).size().reset_index(name="count")
    fig_sub = px.bar(
        sub_dist, x=subject_col, y="count", color="label",
        color_discrete_map={"Fake": "#ff4757", "Real": "#00ff88"},
        title="Articles by Subject and Class",
        barmode="group", text="count",
    )
    fig_sub.update_traces(textposition="outside", textfont=dict(size=11, color="#e0e0e0"))
    fig_sub.update_layout(
        template="plotly_dark", height=500,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=16, color="#8899aa")),
        xaxis=dict(title="", tickangle=-45, tickfont=dict(size=12)),
        yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(font=dict(size=13, color="#8899aa")),
        margin=dict(t=60, b=100, l=60, r=20),
    )
    st.plotly_chart(fig_sub, use_container_width=True)


# ─── Data Sample ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔎 Data Sample</div>', unsafe_allow_html=True)

with st.expander("Preview Dataset (first 50 rows)", expanded=False):
    display_cols = [c for c in df.columns if c not in ["text_length", "word_count"]]
    st.dataframe(
        df[display_cols].head(50),
        use_container_width=True,
        height=400,
    )

# ─── Dataset Statistics ─────────────────────────────────────────────────────
with st.expander("📊 Detailed Statistics"):
    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.markdown("**Fake News Statistics**")
        st.dataframe(
            df[df["label"] == "Fake"][["text_length", "word_count"]].describe().round(1),
            use_container_width=True,
        )
    with stat_col2:
        st.markdown("**Real News Statistics**")
        st.dataframe(
            df[df["label"] == "Real"][["text_length", "word_count"]].describe().round(1),
            use_container_width=True,
        )


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 40px 20px 20px; margin-top: 60px;
    border-top: 1px solid rgba(0, 212, 255, 0.1); color: #556677; font-size: 0.85rem;">
    📰 Fake News Detection System · Dataset Analytics Module
</div>
""", unsafe_allow_html=True)
