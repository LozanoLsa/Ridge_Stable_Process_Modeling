"""
app.py — Laser Cutting Surface Roughness Dashboard
LozanoLsa · Project 10 · Ridge Regression · 2026

Model: Ridge + RidgeCV (collinearity stabilisation)
Domain: Precision Fabrication — Surface Roughness Ra Prediction
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ridge · Laser Cutting Ra Predictor",
    page_icon="🔆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── FULL CSS INJECTION ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&family=Instrument+Serif:ital@0;1&display=swap');

:root {
    --bg:       #080c12;
    --surface:  #0e1420;
    --card:     #121922;
    --card2:    #161f2e;
    --border:   #1e2d45;
    --accent:   #2dd4bf;
    --accent2:  #5eead4;
    --danger:   #f87171;
    --warn:     #fbbf24;
    --ok:       #4ade80;
    --text:     #c8d8f0;
    --muted:    #4e6a8a;
    --fh: 'Syne', sans-serif;
    --fm: 'JetBrains Mono', monospace;
    --fs: 'Instrument Serif', Georgia, serif;
}

.stApp { background: var(--bg) !important; color: var(--text); font-family: var(--fh); }
.block-container { padding: 1.8rem 2.4rem 3rem !important; max-width: 1400px !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] label { font-family: var(--fm) !important; font-size: 0.7rem !important; color: var(--text) !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }

[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; border: 2px solid var(--accent2) !important; box-shadow: 0 0 8px rgba(45,212,191,0.4) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { font-family: var(--fm) !important; font-size: 0.65rem !important; color: var(--accent2) !important; background: var(--card) !important; border: 1px solid var(--border) !important; padding: 1px 5px !important; border-radius: 3px !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }

[data-testid="stSelectbox"] > div > div { background: var(--card) !important; border: 1px solid var(--border) !important; color: var(--text) !important; font-family: var(--fm) !important; font-size: 0.78rem !important; border-radius: 3px !important; }

[data-testid="stMetric"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-top: 2px solid var(--accent) !important; padding: 1rem 1.1rem 0.9rem !important; border-radius: 3px !important; }
[data-testid="stMetricLabel"] > div { font-family: var(--fm) !important; font-size: 0.6rem !important; text-transform: uppercase !important; letter-spacing: 0.18em !important; color: var(--muted) !important; font-weight: 400 !important; }
[data-testid="stMetricValue"] > div { font-family: var(--fm) !important; font-size: 1.7rem !important; font-weight: 600 !important; color: var(--accent2) !important; line-height: 1.1 !important; }

[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--border) !important; gap: 0 !important; background: transparent !important; }
[data-testid="stTabs"] [role="tab"] { font-family: var(--fm) !important; font-size: 0.68rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; color: var(--muted) !important; padding: 0.5rem 1.2rem !important; border: none !important; border-radius: 0 !important; background: transparent !important; transition: all 0.2s !important; }
[data-testid="stTabs"] [role="tab"]:hover { color: var(--accent2) !important; background: rgba(45,212,191,0.06) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; background: transparent !important; }
[data-testid="stTabsContent"] { padding-top: 1.4rem !important; }

[data-testid="stAlert"] { border-radius: 2px !important; font-family: var(--fm) !important; font-size: 0.75rem !important; letter-spacing: 0.04em !important; border: none !important; }

[data-testid="stExpander"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 2px !important; margin-bottom: 6px !important; }
[data-testid="stExpander"] summary { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; letter-spacing: 0.06em !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 2px !important; }
[data-testid="stDataFrame"] th { font-family: var(--fm) !important; font-size: 0.62rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; background: var(--card2) !important; color: var(--muted) !important; border-bottom: 1px solid var(--border) !important; }
[data-testid="stDataFrame"] td { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; background: var(--card) !important; }

hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
[data-testid="stCaptionContainer"] p { font-family: var(--fm) !important; font-size: 0.62rem !important; color: var(--muted) !important; letter-spacing: 0.08em !important; }

h1, h2, h3 { font-family: var(--fh) !important; color: var(--text) !important; letter-spacing: -0.01em !important; }
p, li { font-family: var(--fh) !important; font-size: 0.88rem !important; }

.lsa-header { border-bottom: 1px solid var(--border); padding-bottom: 1.2rem; margin-bottom: 0.2rem; }
.lsa-project-tag { font-family: var(--fm); font-size: 0.6rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.22em; margin-bottom: 4px; }
.lsa-title { font-family: var(--fh); font-size: 1.85rem; font-weight: 800; color: #fff; line-height: 1.1; letter-spacing: -0.02em; }
.lsa-tagline { font-family: var(--fs); font-style: italic; font-size: 0.9rem; color: var(--muted); margin-top: 4px; }
.lsa-chip { display: inline-block; background: rgba(45,212,191,0.1); border: 1px solid rgba(45,212,191,0.3); color: var(--accent2); font-family: var(--fm); font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px; border-radius: 2px; margin-right: 5px; }
.lsa-section { font-family: var(--fm); font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid var(--border); }
.lsa-footer { margin-top: 2.5rem; padding-top: 0.8rem; border-top: 1px solid var(--border); font-family: var(--fm); font-size: 0.58rem; color: var(--muted); letter-spacing: 0.1em; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_PATH     = "laser_cutting_data.csv"
DATA_PATH_ALT = "10_Ridge_Stable_Process_Modeling/laser_cutting_data.csv"
RANDOM_STATE  = 42
FEATURES      = ["laser_power_w", "cutting_speed_mm_s", "assist_gas_flow_l_min",
                 "focal_offset_mm", "material_thickness_mm", "material_type",
                 "oxygen_pct", "shop_temp_c"]
TARGET        = "surface_roughness_ra_um"
RA_SPEC       = 3.2

FEAT_LABELS = {
    "laser_power_w":          "Laser Power (W)",
    "cutting_speed_mm_s":     "Cutting Speed (mm/s)",
    "assist_gas_flow_l_min":  "Gas Flow (L/min)",
    "focal_offset_mm":        "Focal Offset (mm)",
    "material_thickness_mm":  "Thickness (mm)",
    "material_type":          "Material Type",
    "oxygen_pct":             "O₂ in Gas (%)",
    "shop_temp_c":            "Shop Temp (°C)",
}

# ─── MATPLOTLIB PALETTE ───────────────────────────────────────────────────────
C_BG    = "#080c12"
C_CARD  = "#121922"
C_TEAL  = "#2dd4bf"
C_TEAL2 = "#5eead4"
C_DANGER= "#f87171"
C_WARN  = "#fbbf24"
C_OK    = "#4ade80"
C_PURP  = "#a78bfa"
C_TEXT  = "#c8d8f0"
C_MUTED = "#4e6a8a"

def dark_fig(w=9, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_CARD)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.xaxis.label.set_color(C_MUTED)
    ax.yaxis.label.set_color(C_MUTED)
    ax.title.set_color(C_TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e2d45")
    return fig, ax

# ─── DATA & MODEL ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return pd.read_csv(DATA_PATH_ALT)

@st.cache_resource
def train_model(df):
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(Xtr); Xte_sc = sc.transform(Xte)
    rcv = RidgeCV(alphas=np.logspace(-3, 3, 80), cv=5, scoring="r2")
    rcv.fit(Xtr_sc, ytr)
    best_a = rcv.alpha_
    mdl = Ridge(alpha=best_a); mdl.fit(Xtr_sc, ytr)
    yp  = mdl.predict(Xte_sc)
    ols = LinearRegression(); ols.fit(Xtr_sc, ytr)
    metrics = {
        "r2"   : round(r2_score(yte, yp), 4),
        "rmse" : round(np.sqrt(mean_squared_error(yte, yp)), 4),
        "mae"  : round(mean_absolute_error(yte, yp), 4),
        "alpha": round(best_a, 5),
    }
    cdf = (pd.DataFrame({
        "Feature":    FEATURES,
        "Ridge_Coef": mdl.coef_,
        "OLS_Coef":   ols.coef_,
    }).sort_values("Ridge_Coef", key=abs, ascending=False).reset_index(drop=True))
    X_vif = sm.add_constant(Xtr.reset_index(drop=True))
    vif = pd.DataFrame({
        "Feature": FEATURES,
        "VIF":     [variance_inflation_factor(X_vif.values, i+1) for i in range(len(FEATURES))],
    })
    return mdl, sc, Xte, yte, yp, metrics, cdf, vif, ols

df = load_data()
model, scaler, X_test, y_test, y_pred, metrics, coef_df, vif_df, ols_m = train_model(df)
residuals = y_test.values - y_pred

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #10 · Ridge Regression · Laser Cutting</div>
    <div class="lsa-title">When Variables Move Together, Coefficients Lie</div>
    <div class="lsa-tagline">Power, speed, and gas flow are correlated by design. Ridge untangles them — OLS can't.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">RIDGE · RIDGECV</span>
        <span class="lsa-chip">8 FEATURES</span>
        <span class="lsa-chip">R² {metrics['r2']:.4f}</span>
        <span class="lsa-chip">α = {metrics['alpha']:.4f}</span>
        <span class="lsa-chip">VIF UP TO 7.6</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TOP KPI ROW ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("R² (Test Set)",      f"{metrics['r2']:.4f}",         "64.5% Ra variance explained")
k2.metric("RMSE",               f"±{metrics['rmse']:.3f} µm",  "13% of spec limit (3.2 µm)")
k3.metric("Optimal Alpha",      f"α = {metrics['alpha']:.4f}", "5-fold RidgeCV · 80 candidates")
k4.metric("Multicollinearity",  "VIF up to 7.6",               "Power–Speed–Gas triplet")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "CONTEXT", "DATA EXPLORER", "PERFORMANCE", "SIMULATOR", "RIDGE ANALYSIS", "ACTION PLAN"
])

# ══ TAB 0 · CONTEXT ════════════════════════════════════════════════════════════
with tab0:
    st.markdown('<div class="lsa-section">// Why this project exists</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:var(--card);border:1px solid var(--border);
                border-left:4px solid var(--accent);border-radius:3px;
                padding:1.4rem 1.6rem;margin-bottom:1.2rem;">
        <div style="font-family:'Instrument Serif',Georgia,serif;font-size:1.15rem;
                    font-style:italic;color:var(--text);line-height:1.6;">
            "When your process variables move together, your model needs a steadying hand — that's what Ridge does."
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1.05, 1])

    with col_a:
        st.markdown("""
        <div style="background:var(--card);border:1px solid var(--border);border-radius:3px;padding:1.3rem 1.5rem;margin-bottom:10px;">
            <div style="font-family:var(--fm);font-size:0.6rem;color:var(--accent);text-transform:uppercase;letter-spacing:.2em;margin-bottom:10px;">// The Business Problem</div>
            <div style="font-family:var(--fh);font-size:0.88rem;color:var(--text);line-height:1.75;">
                In <strong style="color:#fff;">precision laser cutting</strong>, surface roughness (Ra) is the primary quality gate.
                A part that exceeds Ra ≤ 3.2 µm must be deburred, polished, or scrapped —
                none of which is recoverable in high-mix, low-batch environments.<br><br>
                The problem is not a lack of data. Modern CNC laser systems log dozens of parameters
                at every cut. The problem is that the most important variables —
                <strong style="color:var(--accent2);">laser power, cutting speed, and assist gas pressure</strong> —
                are <em>structurally correlated</em>: an operator who increases power will
                also increase speed and gas flow, because that is correct engineering practice.<br><br>
                This creates <strong style="color:var(--warn);">multicollinearity</strong> that inflates OLS coefficient
                estimates and makes them unreliable for process guidance. A model that points
                the engineer in the wrong direction on a single lever is worse than no model.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:var(--card);border:1px solid var(--border);border-radius:3px;padding:1.3rem 1.5rem;">
            <div style="font-family:var(--fm);font-size:0.6rem;color:var(--accent);text-transform:uppercase;letter-spacing:.2em;margin-bottom:10px;">// Why Ridge, Not OLS or Lasso</div>
            <div style="font-family:var(--fh);font-size:0.88rem;color:var(--text);line-height:1.75;">
                <strong style="color:var(--danger);">OLS</strong> inflates coefficients under collinearity —
                a coefficient's sign can even flip depending on the training sample.<br><br>
                <strong style="color:var(--warn);">Lasso (Project 09)</strong> solves a different problem: it zeros out
                irrelevant variables. But here all 8 laser parameters have clear physical
                justification — none should be discarded.<br><br>
                <strong style="color:var(--accent2);">Ridge</strong> applies an L2 penalty that shrinks all coefficients
                proportionally, distributing the shared explanatory power across correlated
                variables rather than arbitrarily amplifying one. The result is a
                <em>stable, physically interpretable model</em> that can actually guide
                recipe decisions — before the cut is made.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style="background:var(--card);border:1px solid var(--border);border-radius:3px;padding:1.3rem 1.5rem;margin-bottom:10px;">
            <div style="font-family:var(--fm);font-size:0.6rem;color:var(--accent);text-transform:uppercase;letter-spacing:.2em;margin-bottom:10px;">// Dataset at a Glance</div>
            <div style="font-family:var(--fm);font-size:0.75rem;color:var(--text);line-height:2.1;">
                <span style="color:var(--muted);">Records</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1,847 cut records from a CNC laser system<br>
                <span style="color:var(--muted);">Target</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; surface_roughness_ra_um (Ra, µm)<br>
                <span style="color:var(--muted);">Ra Range</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.81 – 5.74 µm &nbsp;·&nbsp; Mean 3.19 µm<br>
                <span style="color:var(--muted);">Spec Limit</span> &nbsp;&nbsp;&nbsp; Ra ≤ 3.2 µm<br>
                <span style="color:var(--muted);">In-Spec</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 51.8% of cuts pass as-is<br>
                <span style="color:var(--muted);">Materials</span> &nbsp;&nbsp;&nbsp;&nbsp; Carbon Steel · Aluminium<br>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:var(--card);border:1px solid var(--border);border-radius:3px;padding:1.3rem 1.5rem;margin-bottom:10px;">
            <div style="font-family:var(--fm);font-size:0.6rem;color:var(--accent);text-transform:uppercase;letter-spacing:.2em;margin-bottom:10px;">// Key Results</div>
            <div style="font-family:var(--fm);font-size:0.75rem;color:var(--text);line-height:2.1;">
                <span style="color:var(--muted);">Algorithm</span> &nbsp;&nbsp;&nbsp; Ridge Regression · L2 + RidgeCV<br>
                <span style="color:var(--muted);">R²</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.6454 &nbsp;·&nbsp; 64.5% of Ra variance explained<br>
                <span style="color:var(--muted);">RMSE</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ±0.430 µm &nbsp;·&nbsp; 13% of spec limit<br>
                <span style="color:var(--muted);">Alpha (CV)</span> &nbsp; 4.422 &nbsp;·&nbsp; selected via 5-fold, 80 candidates<br>
                <span style="color:var(--muted);">Max VIF</span> &nbsp;&nbsp;&nbsp;&nbsp; 7.6 on laser power &nbsp;·&nbsp; Ridge flag<br>
                <span style="color:var(--muted);">vs OLS</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Same R² — Ridge wins on stability<br>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-left:3px solid var(--ok);border-radius:3px;padding:1.1rem 1.4rem;">
            <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.18em;margin-bottom:8px;">// What the simulator shows</div>
            <div style="font-family:var(--fh);font-size:0.84rem;color:var(--text);line-height:1.7;">
                Correcting a bad recipe on <strong style="color:#fff;">12 mm steel</strong> —
                raising power by 1300W, reducing speed by 17 mm/s, increasing gas flow by 5 L/min —
                recovers <strong style="color:var(--ok);">0.984 µm Ra</strong>.<br><br>
                From <strong style="color:var(--danger);">3.976 µm (fail)</strong> to
                <strong style="color:var(--ok);">2.992 µm (pass)</strong>.
                The correction is <em>quantified</em>, not guessed.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="lsa-section">// Top coefficient drivers — what moves the needle on Ra</div>', unsafe_allow_html=True)

    coeff_data = [
        ("+0.376", "Material Thickness (mm)", "↑ Increases Ra", C_DANGER,
         "Dominant driver. Thicker material demands more energy; insufficient energy → rough cut edge."),
        ("−0.297", "Material Type (Aluminium)", "↓ Reduces Ra", C_TEAL,
         "Aluminium's superior thermal conductivity consistently produces smoother cuts than steel."),
        ("−0.234", "Laser Power (W)", "↓ Reduces Ra", C_TEAL,
         "More power → cleaner melt ejection → smoother edge. Ridge keeps this stable vs OLS."),
        ("−0.184", "Gas Flow (L/min)", "↓ Reduces Ra", C_TEAL,
         "Assist gas removes molten material from the cut zone. Stable estimate — not inflated by collinearity."),
        ("+0.046", "O₂ in Gas (%)", "↑ Increases Ra", C_WARN,
         "Higher oxygen increases oxidation on the cut edge, mildly roughening it."),
    ]

    cols_c = st.columns(5)
    for col, (coef, feat, direction, color, note) in zip(cols_c, coeff_data):
        with col:
            st.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--border);
                        border-top:3px solid {color};border-radius:3px;padding:1rem 1rem;height:100%;">
                <div style="font-family:var(--fm);font-size:1.6rem;font-weight:700;
                            color:{color};line-height:1;">{coef}</div>
                <div style="font-family:var(--fm);font-size:0.58rem;color:#fff;
                            font-weight:600;margin-top:6px;margin-bottom:2px;">{feat}</div>
                <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                            margin-bottom:8px;">{direction}</div>
                <div style="font-family:var(--fh);font-size:0.72rem;color:var(--muted);line-height:1.5;">{note}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:10px;font-family:var(--fm);font-size:0.62rem;color:var(--muted);
                letter-spacing:.06em;text-align:center;">
        All coefficients are standardised (per-σ units) — comparable across features measured in different physical units.
        &nbsp;·&nbsp; Cutting speed, focal offset, and shop temp also active but smaller in magnitude.
    </div>
    """, unsafe_allow_html=True)

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="lsa-section">// Dataset — 1,847 laser cut records</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", "1,847")
    c2.metric("Steel / Aluminium",
              f"{(df['material_type']==0).sum()} / {(df['material_type']==1).sum()}")
    c3.metric("Spec Limit", f"Ra ≤ {RA_SPEC} µm")
    with st.expander("Preview first 20 rows"):
        st.dataframe(df.head(20), use_container_width=True)

    st.divider()
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="lsa-section">// Roughness distribution by material</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(7, 4)
        ax.hist(df[df["material_type"]==0][TARGET], bins=30, alpha=0.70,
                color=C_TEAL, edgecolor=C_BG, lw=0.3, label="Steel (0)")
        ax.hist(df[df["material_type"]==1][TARGET], bins=30, alpha=0.70,
                color=C_WARN, edgecolor=C_BG, lw=0.3, label="Aluminium (1)")
        ax.axvline(RA_SPEC, color=C_DANGER, ls="--", lw=1.8, label=f"Spec {RA_SPEC} µm")
        ax.set_xlabel("Ra (µm)"); ax.set_ylabel("Count")
        ax.legend(fontsize=8, facecolor=C_CARD, labelcolor=C_TEXT)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        pct_ok = (df[TARGET] <= RA_SPEC).mean() * 100
        st.caption(f"{pct_ok:.1f}% of cuts meet spec.  "
                   f"Steel mean = {df[df['material_type']==0][TARGET].mean():.3f} µm  ·  "
                   f"Aluminium mean = {df[df['material_type']==1][TARGET].mean():.3f} µm.")

    with cb:
        st.markdown('<div class="lsa-section">// Multicollinearity — power vs speed</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(7, 4)
        sc_plot = ax.scatter(df["laser_power_w"], df["cutting_speed_mm_s"],
                             c=df[TARGET], cmap="RdYlGn_r", alpha=0.35, s=8)
        plt.colorbar(sc_plot, ax=ax, label="Ra (µm)")
        m, b, r, *_ = linregress(df["laser_power_w"], df["cutting_speed_mm_s"])
        xr = np.linspace(1500, 4000, 100)
        ax.plot(xr, m * xr + b, color="white", lw=1.5, ls="--", label=f"r = {r:.3f}")
        ax.set_xlabel("Laser Power (W)"); ax.set_ylabel("Cutting Speed (mm/s)")
        ax.set_title(f"Power vs Speed — r = {r:.3f} — Why Ridge, not OLS", color=C_TEXT)
        ax.legend(fontsize=8, facecolor=C_CARD, labelcolor=C_TEXT)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Power and speed are structurally correlated (r ≈ 0.91) — operators increase them "
                   "in concert. OLS inflates both coefficients; Ridge stabilises them.")

    st.divider()
    st.markdown('<div class="lsa-section">// Feature scatter vs Ra</div>',
                unsafe_allow_html=True)
    sel = st.selectbox("Feature:", FEATURES, format_func=lambda x: FEAT_LABELS.get(x, x))
    m, b, r, p, _ = linregress(df[sel], df[TARGET])
    fig, ax = dark_fig(10, 4)
    ax.scatter(df[sel], df[TARGET], alpha=0.25, s=8, color=C_TEAL)
    xr = np.linspace(df[sel].min(), df[sel].max(), 100)
    ax.plot(xr, m * xr + b, color=C_DANGER, lw=1.5, ls="--", label=f"r = {r:.3f}")
    ax.axhline(RA_SPEC, color=C_WARN, lw=1.2, ls=":", label=f"Spec {RA_SPEC} µm")
    ax.set_xlabel(FEAT_LABELS.get(sel, sel)); ax.set_ylabel("Ra (µm)")
    ax.legend(fontsize=9, facecolor=C_CARD, labelcolor=C_TEXT)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption(f"r = {r:.3f}  ·  slope = {m:+.5f} µm/unit  ·  p {'< 0.001' if p < 0.001 else f'= {p:.4f}'}")

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="lsa-section">// Ridge regression — test performance (n=370)</div>',
                unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="lsa-section">// Predicted vs actual</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(6, 5)
        in_spec = y_test <= RA_SPEC
        ax.scatter(y_test[in_spec],  y_pred[in_spec],  alpha=0.50, s=12, color=C_TEAL,   label="Within spec")
        ax.scatter(y_test[~in_spec], y_pred[~in_spec], alpha=0.50, s=12, color=C_DANGER, label="Out of spec")
        lims = [y_test.min() - 0.1, y_test.max() + 0.1]
        ax.plot(lims, lims, color="white", ls="--", lw=1.5, label="Perfect")
        ax.axvline(RA_SPEC, color=C_WARN, lw=1.0, ls=":", alpha=0.7)
        ax.axhline(RA_SPEC, color=C_WARN, lw=1.0, ls=":", alpha=0.7, label=f"Spec {RA_SPEC} µm")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual Ra (µm)"); ax.set_ylabel("Predicted Ra (µm)")
        ax.set_title(f"R² = {metrics['r2']}", color=C_TEXT)
        ax.legend(fontsize=8, facecolor=C_CARD, labelcolor=C_TEXT)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Teal = within spec · Red = exceeded Ra limit · Amber crosshairs = spec boundary.")

    with cb:
        st.markdown('<div class="lsa-section">// Residuals vs fitted — homoscedasticity check</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(6, 5)
        ax.scatter(y_pred, residuals, alpha=0.35, s=10,
                   c=[C_DANGER if abs(r) > 1.0 else C_TEAL for r in residuals])
        ax.axhline(0, color="white", lw=1.5, ls="--")
        ax.axhline(+2 * metrics['rmse'], color=C_WARN, lw=1, ls=":", label="±2·RMSE")
        ax.axhline(-2 * metrics['rmse'], color=C_WARN, lw=1, ls=":")
        ax.set_xlabel("Fitted Ra (µm)"); ax.set_ylabel("Residual (µm)")
        ax.legend(fontsize=8, facecolor=C_CARD, labelcolor=C_TEXT)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Random scatter around zero — no systematic pattern missed by the model.")

    st.divider()
    st.markdown('<div class="lsa-section">// Ridge vs OLS — why regularisation matters</div>',
                unsafe_allow_html=True)
    Xtr2, Xte2, ytr2, yte2 = train_test_split(df[FEATURES], df[TARGET], test_size=0.2, random_state=RANDOM_STATE)
    sc2 = StandardScaler(); Xtr2s = sc2.fit_transform(Xtr2); Xte2s = sc2.transform(Xte2)
    ols2 = LinearRegression(); ols2.fit(Xtr2s, ytr2)
    r2_ols   = r2_score(yte2, ols2.predict(Xte2s))
    rmse_ols = np.sqrt(mean_squared_error(yte2, ols2.predict(Xte2s)))
    st.markdown(f"""| Model | R² Test | RMSE Test | Notes |
|---|---|---|---|
| **Ridge (α = {metrics['alpha']:.4f})** | **{metrics['r2']:.4f}** | **{metrics['rmse']:.4f} µm** | Stable coefficients — multicollinearity controlled |
| OLS (no regularisation) | {r2_ols:.4f} | {rmse_ols:.4f} µm | Same accuracy, but inflated collinear coefficients |""")
    st.caption("Ridge and OLS deliver nearly identical test R² — Ridge's advantage is coefficient stability, "
               "not higher accuracy. Under collinearity, OLS coefficients are unreliable for process guidance; "
               "Ridge coefficients are not.")

    st.divider()
    st.markdown('<div class="lsa-section">// Metric explanations</div>',
                unsafe_allow_html=True)
    for name, expl in {
        "R²":   "64.5% of Ra variance is explained. Lower than projects 07–09 — laser Ra has genuine nonlinear physics that linear models can only partially capture.",
        "RMSE": "Root Mean Squared Error in µm. At 13% of the 3.2 µm spec limit, the model is precise enough to screen recipes before production.",
        "MAE":  "Mean Absolute Error — average miss per cut. Use this as the practical precision estimate when setting process tolerances.",
    }.items():
        with st.expander(f"{name}  —  {metrics.get(name.lower().replace('²','2'), '—')}"):
            st.write(expl)

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    medians = df[FEATURES].median().to_dict()
    ci, co  = st.columns([1.1, 1])

    with ci:
        st.markdown('<div class="lsa-section">// Laser & motion</div>', unsafe_allow_html=True)
        power = st.slider("Laser Power (W)",       1500, 4000, int(medians["laser_power_w"]),         50)
        speed = st.slider("Cutting Speed (mm/s)",     5,   60, int(medians["cutting_speed_mm_s"]),      1)
        gas   = st.slider("Gas Flow (L/min)",          3.0, 16.0, float(medians["assist_gas_flow_l_min"]),0.5)
        focal = st.slider("Focal Offset (mm)",      -2.0,  2.0, 0.0,                                  0.05)
        o2    = st.slider("O₂ in Gas (%)",             0,   40, int(medians["oxygen_pct"]),             1)
        st.markdown('<div class="lsa-section">// Material</div>', unsafe_allow_html=True)
        thick = st.slider("Thickness (mm)",          1.0, 15.0, float(medians["material_thickness_mm"]),0.5)
        mat   = st.selectbox("Material Type", ["0 — Carbon Steel", "1 — Aluminium"])
        mat_v = int(mat[0])
        temp  = st.slider("Shop Temp (°C)",           15,   35, 23, 1)

    xsim    = pd.DataFrame([[power, speed, gas, focal, thick, mat_v, o2, temp]], columns=FEATURES)
    ra_pred = model.predict(scaler.transform(xsim))[0]
    is_pass = ra_pred <= RA_SPEC
    margin  = RA_SPEC - ra_pred
    qual_c  = C_OK if is_pass else C_DANGER
    qual_l  = "PASS — Within Ra Spec" if is_pass else "FAIL — Exceeds Ra Spec"
    badge_bg = "#0f2e1a" if is_pass else "#2e0f0f"

    with co:
        st.markdown(
            f'''<div style="background:var(--card);border:1px solid var(--border);
                        border-radius:4px;padding:1.6rem 1.8rem;">
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                            color:#fff;margin-bottom:1rem;">Prediction Result</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:3.4rem;
                            font-weight:700;color:{qual_c};line-height:1;
                            letter-spacing:-0.02em;">{ra_pred:.3f}
                    <span style="font-size:1.4rem;font-weight:400;color:{C_MUTED};">µm Ra</span>
                </div>
                <div style="margin-top:14px;">
                    <span style="background:{badge_bg};color:{qual_c};
                                 font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                 font-weight:600;letter-spacing:.08em;
                                 padding:5px 16px;border-radius:20px;">{qual_l}</span>
                </div>
                <div style="margin-top:18px;font-family:'JetBrains Mono',monospace;
                            font-size:0.68rem;color:var(--muted);line-height:2.1;">
                    Spec limit : Ra &#8804; {RA_SPEC} µm<br>
                    Margin &nbsp;&nbsp;&nbsp;:
                    <strong style="color:{qual_c};">{margin:+.3f} µm</strong>
                    {'(within spec)' if is_pass else '(exceeds spec)'}
                </div>
            </div>''',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown('<div class="lsa-section">// Position within Ra scale</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 1.4))
    fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)
    ax.barh(0, 6.0 - 0.3, left=0.3, height=0.55, color="#1e2d45")
    ax.barh(0, RA_SPEC - 0.3, left=0.3, height=0.55, color=(0.29, 0.87, 0.50, 0.18))
    ax.axvline(RA_SPEC, color=C_OK, lw=1.2, ls=":")
    mc = C_OK if is_pass else C_DANGER
    ax.plot([ra_pred, ra_pred], [-0.38, 0.38], color=mc, lw=2.5)
    ax.scatter([ra_pred], [0], s=130, color=mc, zorder=5)
    ax.set_xlim(0.3, 6.0); ax.set_ylim(-0.65, 0.65); ax.set_yticks([])
    ax.tick_params(colors=C_MUTED, labelsize=8)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xlabel("Ra (µm)", color=C_MUTED, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption("Green zone = in-spec (Ra ≤ 3.2 µm) · Marker = predicted Ra.")

    st.divider()
    st.markdown('<div class="lsa-section">// Three reference scenarios</div>', unsafe_allow_html=True)
    scen = {
        "A · 3mm Al · Optimised":      {"laser_power_w":3200,"cutting_speed_mm_s":20,"assist_gas_flow_l_min":10,
                                         "focal_offset_mm":0.0,"material_thickness_mm":3.0,"material_type":1,"oxygen_pct":8,"shop_temp_c":23},
        "B · 12mm Steel · Under-Recipe":{"laser_power_w":2500,"cutting_speed_mm_s":35,"assist_gas_flow_l_min":5,
                                         "focal_offset_mm":0.0,"material_thickness_mm":12.0,"material_type":0,"oxygen_pct":25,"shop_temp_c":23},
        "C · 12mm Steel · Corrected":  {"laser_power_w":3800,"cutting_speed_mm_s":18,"assist_gas_flow_l_min":10,
                                         "focal_offset_mm":0.0,"material_thickness_mm":12.0,"material_type":0,"oxygen_pct":25,"shop_temp_c":23},
    }
    sc_preds = {}
    cols_s   = st.columns(3)
    for col, (name, params) in zip(cols_s, scen.items()):
        p  = model.predict(scaler.transform(pd.DataFrame([params])))[0]
        sc_preds[name] = p
        ok = p <= RA_SPEC
        c  = C_OK if ok else C_DANGER
        with col:
            st.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--border);
                        border-left:3px solid {c};border-radius:2px;padding:1.1rem 1.2rem;">
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);
                            letter-spacing:.15em;text-transform:uppercase;margin-bottom:8px;">{name}</div>
                <div style="font-family:var(--fm);font-size:2.4rem;font-weight:700;
                            color:{c};line-height:1;">{p:.3f}</div>
                <div style="font-family:var(--fm);font-size:0.72rem;color:var(--muted);
                            margin-top:4px;">µm Ra · {'Pass' if ok else 'Fail'}</div>
            </div>""", unsafe_allow_html=True)

    pv = list(sc_preds.values())
    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border);
                border-left:3px solid {C_WARN};border-radius:2px;
                padding:0.9rem 1.2rem;margin-top:10px;">
        <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Recipe recovery</div>
        <div style="font-family:var(--fm);font-size:0.72rem;color:var(--text);line-height:1.7;">
            Correcting the under-recipe on 12 mm steel (higher power, lower speed, more gas)
            recovers <strong style="color:{C_WARN};">{pv[1]-pv[2]:.3f} µm Ra</strong> —
            from {pv[1]:.3f} µm (fail) to {pv[2]:.3f} µm (pass).
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    ca, cb = st.columns([1.2, 1])
    with ca:
        st.markdown('<div class="lsa-section">// Ridge coefficients — stable under collinearity</div>',
                    unsafe_allow_html=True)
        cs = coef_df.sort_values("Ridge_Coef", ascending=True)
        fig, ax = dark_fig(7, 5.5)
        bar_c = [C_TEAL if c < 0 else C_DANGER for c in cs["Ridge_Coef"]]
        bars  = ax.barh(
            [FEAT_LABELS.get(f, f) for f in cs["Feature"]],
            cs["Ridge_Coef"], color=bar_c, alpha=0.82, edgecolor="none", height=0.60
        )
        ax.axvline(0, color="white", lw=0.8)
        for bar, val in zip(bars, cs["Ridge_Coef"]):
            off = 0.003 if val >= 0 else -0.003
            ax.text(val + off, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.4f}", va="center",
                    ha="left" if val >= 0 else "right", fontsize=9, color=C_TEXT)
        ax.set_xlabel("Standardised Ridge Coefficient (µm per σ input change)")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Red = increases Ra (worse quality) · Teal = decreases Ra (better quality). "
                   "Coefficients are stable because Ridge L2 dampens collinear inflation.")

    with cb:
        st.markdown('<div class="lsa-section">// VIF — multicollinearity severity</div>',
                    unsafe_allow_html=True)
        vif_s = vif_df.sort_values("VIF", ascending=False).copy()
        vif_s["Feature"] = vif_s["Feature"].map(lambda x: FEAT_LABELS.get(x, x))
        vif_s["Flag"]    = vif_s["VIF"].apply(
            lambda v: "Ridge needed" if v > 5 else ("Moderate" if v > 2 else "OK")
        )
        st.dataframe(vif_s.style.format({"VIF": "{:.2f}"}),
                     use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-left:3px solid {C_WARN};border-radius:2px;
                    padding:1rem 1.2rem;margin-top:12px;">
            <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Collinear triplet</div>
            <div style="font-family:var(--fm);font-size:0.72rem;color:var(--text);line-height:1.7;">
                <code>laser_power_w</code> (VIF 7.6) · <code>cutting_speed_mm_s</code> (VIF 5.6) ·
                <code>assist_gas_flow_l_min</code> (VIF 2.8)<br>
                These three variables move together in practice. OLS over-amplifies their individual
                coefficients. Ridge distributes the shared effect smoothly across all three.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="lsa-section">// 2D process window — power × speed for 12mm steel</div>',
                unsafe_allow_html=True)
    power_r = np.linspace(1500, 4000, 60); speed_r = np.linspace(5, 50, 60)
    P, S    = np.meshgrid(power_r, speed_r)
    grid    = pd.DataFrame({
        "laser_power_w": P.ravel(), "cutting_speed_mm_s": S.ravel(),
        "assist_gas_flow_l_min": 9.0, "focal_offset_mm": 0.0,
        "material_thickness_mm": 12.0, "material_type": 0,
        "oxygen_pct": 20.0, "shop_temp_c": 23.0,
    })
    Z    = model.predict(scaler.transform(grid[FEATURES])).reshape(P.shape)
    fig, ax = dark_fig(10, 5.5)
    cf   = ax.contourf(P, S, Z, levels=25, cmap="RdYlGn_r", alpha=0.88)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label("Predicted Ra (µm)", color=C_MUTED)
    cbar.ax.yaxis.set_tick_params(color=C_MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=C_MUTED)
    cs2  = ax.contour(P, S, Z, levels=[RA_SPEC], colors=["white"], linewidths=2.0)
    ax.clabel(cs2, fmt=f"Spec {RA_SPEC} µm", fontsize=9, colors="white")
    ax.contourf(P, S, Z, levels=[0, RA_SPEC], colors=["lime"], alpha=0.12, hatches=["////"])
    ax.set_xlabel("Laser Power (W)"); ax.set_ylabel("Cutting Speed (mm/s)")
    ax.set_title("Ra Response Surface — 12mm Carbon Steel (gas=9 L/min, focal=0)", color=C_TEXT)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption("Green-hatched zone = Power/Speed combinations predicted to meet Ra ≤ 3.2 µm. "
               "Use as process window reference for thick steel jobs.")

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="lsa-section">// Operational recommendations</div>',
                unsafe_allow_html=True)
    actions = [
        (C_DANGER, "Thickness is the primary Ra driver — schedule by material class",
         "material_thickness_mm coefficient is +0.376 (strongest driver). "
         "Create separate recipe libraries for thin (≤ 6mm) and thick (> 6mm) sheets. "
         "Never apply a thin-sheet recipe to thick material without model verification."),
        (C_TEAL,   "Use the power–speed–gas response surface as your process window",
         "The 2D contour map shows exactly which Power/Speed combinations achieve Ra ≤ 3.2 µm "
         "for 12mm steel. Laminate it at the operator console as a first-response reference "
         "without needing the simulator for every job."),
        (C_WARN,   "Material switch protocol: re-query the simulator",
         "The aluminium/steel switch carries a −0.297 coefficient (0.3 µm improvement per "
         "material change). When batch switching material type, run the simulator before the "
         "first cut — the recipe may need power/speed adjustment."),
        (C_OK,     "Focal offset is less critical than expected",
         "focal_offset_mm coefficient is only −0.010 — the smallest active driver. "
         "In practice, ±0.5 mm focal deviation contributes < 0.01 µm Ra change. "
         "Re-invest calibration time in power and gas flow settings instead."),
        (C_PURP,   "Retrain when consumables change (nozzle, lens, assist gas supplier)",
         "The model is calibrated on a specific machine configuration. Nozzle wear, lens "
         "contamination, or gas supplier changes shift the baseline Ra. Collect 200+ records "
         "after any major maintenance and refit with RidgeCV — alpha selection is automatic; "
         "recalibration takes minutes."),
    ]
    for color, title, body in actions:
        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-left:3px solid {color};border-radius:2px;
                    padding:1.1rem 1.3rem;margin-bottom:10px;">
            <div style="font-family:var(--fm);font-size:0.72rem;font-weight:600;
                        color:{color};margin-bottom:6px;">{title}</div>
            <div style="font-family:var(--fm);font-size:0.7rem;color:var(--muted);line-height:1.7;">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="background:var(--card);border:1px solid var(--border);border-radius:2px;
                padding:1rem 1.3rem;text-align:center;">
        <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Full project pack</div>
        <div style="font-family:var(--fm);font-size:0.68rem;color:var(--muted);line-height:1.7;">
            Complete dataset · notebook with outputs · presentation deck (PPTX + PDF) · simulator
            available on <span style="color:#2dd4bf;">lozanolsa.gumroad.com</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lsa-footer">
    LozanoLsa · Turning Operations into Predictive Systems · Laser Cutting Ra Predictor · Project 10 · v2.0
</div>
""", unsafe_allow_html=True)
