"""
Fake News Detection - Model Performance Page
==============================================
Comprehensive model evaluation with comparison charts,
confusion matrices, ROC curves, and detailed metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Performance | Fake News Detector",
    page_icon="📈",
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

/* Best model highlight */
.best-model-badge {
    display: inline-block; padding: 4px 14px;
    border-radius: 50px; font-size: 0.75rem; font-weight: 700;
    background: linear-gradient(135deg, #00ff88, #00d4ff);
    color: #0e1117; letter-spacing: 0.5px;
}
</style>
"""

st.markdown(SHARED_CSS, unsafe_allow_html=True)

# ─── Helper Functions ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@st.cache_data(ttl=3600)
def load_model_comparison():
    """Load model comparison CSV."""
    csv_path = os.path.join(BASE_DIR, "reports", "model_comparison.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None


def find_report_images():
    """Find all PNG report images."""
    reports_dir = os.path.join(BASE_DIR, "reports")
    if not os.path.exists(reports_dir):
        return []
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    images = []
    for pattern in patterns:
        images.extend(glob.glob(os.path.join(reports_dir, pattern)))
    return sorted(images)


def identify_columns(df):
    """Identify model name and metric columns."""
    cols = df.columns.tolist()
    model_col = None
    metric_cols = []

    # Find model name column
    for c in cols:
        cl = c.lower().strip()
        if cl in ["model", "model_name", "name", "classifier", "algorithm"]:
            model_col = c
            break

    if model_col is None:
        # Try first column if it seems like model names
        if df[cols[0]].dtype == object:
            model_col = cols[0]

    # Find metric columns (numeric)
    for c in cols:
        if c != model_col:
            try:
                pd.to_numeric(df[c], errors="raise")
                metric_cols.append(c)
            except (ValueError, TypeError):
                pass

    return model_col, metric_cols


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0;">
        <div style="font-size: 3rem;">📈</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #00d4ff;">Model Performance</div>
        <div style="font-size: 0.75rem; color: #556677; margin-top: 4px;">EVALUATION METRICS</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ─── Page Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 30px 0 10px;">
    <div style="font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #00ff88);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
        📈 Model Performance
    </div>
    <div style="color: #8899aa; font-size: 1rem; margin-top: 8px;">
        Compare trained models and evaluate their performance across key metrics
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ─── Load Data ───────────────────────────────────────────────────────────────
df = load_model_comparison()

if df is None:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2d1b1b, #1a1a2e);
        border: 1px solid rgba(255, 71, 87, 0.3); border-radius: 16px;
        padding: 40px; text-align: center; margin: 20px 0;">
        <div style="font-size: 3rem; margin-bottom: 16px;">📊</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #ff4757; margin-bottom: 12px;">
            Model Reports Not Found
        </div>
        <div style="color: #8899aa; font-size: 0.95rem; line-height: 1.7; max-width: 500px; margin: 0 auto;">
            Please run <code style="color:#00d4ff;">python main.py</code> first to train models 
            and generate the comparison report in <code style="color:#00d4ff;">reports/model_comparison.csv</code>.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    model_col, metric_cols = identify_columns(df)

    # ─── KPI Cards ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 Best Results</div>', unsafe_allow_html=True)

    # Find best metrics
    acc_col = None
    f1_col = None
    for c in metric_cols:
        cl = c.lower()
        if "accuracy" in cl or cl == "acc":
            acc_col = c
        if "f1" in cl:
            f1_col = c

    best_acc = df[acc_col].max() if acc_col else None
    best_f1 = df[f1_col].max() if f1_col else None

    best_model_name = "N/A"
    if model_col:
        # Determine best model by f1 if available, else accuracy
        sort_col = f1_col if f1_col else (acc_col if acc_col else metric_cols[0] if metric_cols else None)
        if sort_col:
            best_idx = df[sort_col].idxmax()
            best_model_name = df.loc[best_idx, model_col]

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <span class="kpi-icon">🏆</span>
            <div class="kpi-value" style="font-size: 1.5rem;">{best_model_name}</div>
            <div class="kpi-label">Best Model</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        acc_str = f"{best_acc:.2%}" if best_acc is not None else "N/A"
        st.markdown(f"""
        <div class="kpi-card">
            <span class="kpi-icon">🎯</span>
            <div class="kpi-value">{acc_str}</div>
            <div class="kpi-label">Best Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        f1_str = f"{best_f1:.2%}" if best_f1 is not None else "N/A"
        st.markdown(f"""
        <div class="kpi-card">
            <span class="kpi-icon">⚡</span>
            <div class="kpi-value">{f1_str}</div>
            <div class="kpi-label">Best F1 Score</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <span class="kpi-icon">🤖</span>
            <div class="kpi-value">{len(df)}</div>
            <div class="kpi-label">Models Compared</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Model Comparison Chart ──────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)

    if model_col and metric_cols:
        # Grouped bar chart
        colors = ["#00d4ff", "#00ff88", "#ff4757", "#ffa500", "#a855f7",
                  "#ff6b81", "#36d7b7", "#f9ca24"]

        fig_comp = go.Figure()
        for i, metric in enumerate(metric_cols):
            fig_comp.add_trace(go.Bar(
                name=metric,
                x=df[model_col].tolist(),
                y=df[metric].tolist(),
                marker_color=colors[i % len(colors)],
                text=[f"{v:.4f}" for v in df[metric]],
                textposition="outside",
                textfont=dict(size=10, color="#e0e0e0"),
            ))

        fig_comp.update_layout(
            template="plotly_dark", height=500, barmode="group",
            title=dict(text="Model Performance Comparison",
                      font=dict(size=18, color="#e0e0e0")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="", tickfont=dict(size=13)),
            yaxis=dict(title="Score", gridcolor="rgba(255,255,255,0.05)",
                      range=[0, max(df[metric_cols].max().max() * 1.15, 1.05)]),
            legend=dict(orientation="h", y=-0.15, font=dict(size=12, color="#8899aa")),
            margin=dict(t=60, b=80, l=60, r=20),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Radar chart
        st.markdown('<div class="section-header">🕸️ Radar Comparison</div>', unsafe_allow_html=True)

        if len(metric_cols) >= 3:
            fig_radar = go.Figure()
            radar_colors = ["#00d4ff", "#00ff88", "#ff4757", "#ffa500", "#a855f7"]

            for i, idx in enumerate(df.index):
                model_name = df.loc[idx, model_col] if model_col else f"Model {i+1}"
                values = df.loc[idx, metric_cols].tolist()
                values.append(values[0])  # close the polygon
                cats = metric_cols + [metric_cols[0]]

                fig_radar.add_trace(go.Scatterpolar(
                    r=values, theta=cats, fill="toself",
                    name=model_name,
                    line=dict(color=radar_colors[i % len(radar_colors)], width=2),
                    fillcolor=f"rgba({int(radar_colors[i % len(radar_colors)][1:3], 16)}, "
                              f"{int(radar_colors[i % len(radar_colors)][3:5], 16)}, "
                              f"{int(radar_colors[i % len(radar_colors)][5:7], 16)}, 0.1)",
                ))

            fig_radar.update_layout(
                template="plotly_dark", height=500,
                title=dict(text="Model Performance Radar", font=dict(size=18, color="#e0e0e0")),
                paper_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                ),
                legend=dict(font=dict(size=12, color="#8899aa")),
                margin=dict(t=60, b=40, l=60, r=60),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # ─── Detailed Metrics Table ──────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Detailed Metrics Table</div>', unsafe_allow_html=True)

    # Highlight best values
    def highlight_best(s):
        """Highlight best value in each column."""
        styles = [""] * len(s)
        try:
            numeric_s = pd.to_numeric(s, errors="coerce")
            if numeric_s.notna().any():
                best_idx = numeric_s.idxmax()
                styles[s.index.get_loc(best_idx)] = (
                    "background-color: rgba(0, 255, 136, 0.15); "
                    "color: #00ff88; font-weight: 700;"
                )
        except Exception:
            pass
        return styles

    styled_df = df.style.apply(highlight_best, axis=0)
    st.dataframe(styled_df, use_container_width=True, height=300)

    # ─── Per-metric bar charts ───────────────────────────────────────────
    if model_col and len(metric_cols) > 1:
        st.markdown('<div class="section-header">📊 Per-Metric Breakdown</div>', unsafe_allow_html=True)

        # Show 2 per row
        for row_start in range(0, len(metric_cols), 2):
            cols = st.columns(2)
            for j in range(2):
                idx = row_start + j
                if idx < len(metric_cols):
                    metric = metric_cols[idx]
                    with cols[j]:
                        sorted_df = df.sort_values(metric, ascending=True)
                        colors_list = ["#1e2a3a"] * len(sorted_df)
                        colors_list[-1] = "#00ff88"  # Best model

                        fig_m = go.Figure(go.Bar(
                            x=sorted_df[metric].tolist(),
                            y=sorted_df[model_col].tolist(),
                            orientation="h",
                            marker_color=colors_list,
                            text=[f"{v:.4f}" for v in sorted_df[metric]],
                            textposition="outside",
                            textfont=dict(size=12, color="#e0e0e0"),
                        ))
                        fig_m.update_layout(
                            template="plotly_dark", height=300,
                            title=dict(text=metric, font=dict(size=15, color="#00d4ff")),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(gridcolor="rgba(255,255,255,0.05)",
                                      range=[0, sorted_df[metric].max() * 1.15]),
                            yaxis=dict(tickfont=dict(size=12)),
                            margin=dict(t=50, b=20, l=120, r=60),
                        )
                        st.plotly_chart(fig_m, use_container_width=True)


# ─── Report Images ───────────────────────────────────────────────────────────
report_images = find_report_images()

if report_images:
    st.markdown('<div class="section-header">🖼️ Generated Reports</div>', unsafe_allow_html=True)

    # Categorize images
    categories = {}
    for img_path in report_images:
        name = os.path.splitext(os.path.basename(img_path))[0].replace("_", " ").title()
        categories[name] = img_path

    # Display in grid
    img_cols = st.columns(2)
    for i, (name, path) in enumerate(categories.items()):
        with img_cols[i % 2]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e2a3a, #16213e);
                border: 1px solid rgba(0, 212, 255, 0.15); border-radius: 16px;
                padding: 20px; margin-bottom: 20px; text-align: center;">
                <div style="color: #00d4ff; font-size: 1rem; font-weight: 600; margin-bottom: 12px;">
                    {name}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(path, use_container_width=True)
else:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e2a3a, #1a1a2e);
        border: 1px solid rgba(0, 212, 255, 0.15); border-radius: 16px;
        padding: 30px; text-align: center; margin: 20px 0;">
        <div style="font-size: 2rem; margin-bottom: 12px;">🖼️</div>
        <div style="color: #8899aa; font-size: 0.95rem;">
            No report images found. Run <code style="color:#00d4ff;">python main.py</code> 
            to generate confusion matrices, ROC curves, and other visualizations.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Model Descriptions ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔧 Model Details</div>', unsafe_allow_html=True)

model_info = {
    "Logistic Regression": {
        "icon": "📐",
        "desc": "Linear model that uses a logistic function to model binary classification. Fast, interpretable, and works well with TF-IDF features.",
        "params": "C (regularization), penalty (L1/L2), solver (lbfgs/liblinear)",
    },
    "Naive Bayes": {
        "icon": "📊",
        "desc": "Probabilistic classifier based on Bayes' theorem with naive independence assumptions. Excellent baseline for text classification.",
        "params": "alpha (smoothing parameter)",
    },
    "Linear SVC": {
        "icon": "🎯",
        "desc": "Support Vector Classifier with linear kernel. Finds optimal hyperplane to separate classes. Robust and effective for high-dimensional text data.",
        "params": "C (regularization), loss (hinge/squared_hinge), max_iter",
    },
    "XGBoost": {
        "icon": "🚀",
        "desc": "Gradient boosted ensemble of decision trees. Powerful, handles non-linear patterns, but may be slower to train on large text datasets.",
        "params": "n_estimators, max_depth, learning_rate, subsample",
    },
}

model_cols = st.columns(2)
for i, (name, info) in enumerate(model_info.items()):
    with model_cols[i % 2]:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e2a3a, #16213e);
            border: 1px solid rgba(0, 212, 255, 0.15); border-radius: 16px;
            padding: 24px; margin-bottom: 16px;">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">{info['icon']}</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #00d4ff; margin-bottom: 8px;">
                {name}
            </div>
            <div style="color: #8899aa; font-size: 0.88rem; line-height: 1.6; margin-bottom: 12px;">
                {info['desc']}
            </div>
            <div style="color: #556677; font-size: 0.8rem;">
                <strong style="color: #6699aa;">Hyperparameters:</strong> {info['params']}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 40px 20px 20px; margin-top: 60px;
    border-top: 1px solid rgba(0, 212, 255, 0.1); color: #556677; font-size: 0.85rem;">
    📰 Fake News Detection System · Model Performance Module
</div>
""", unsafe_allow_html=True)
