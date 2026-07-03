"""
Carbon CO2 Emission Prediction – Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────── PAGE CONFIG ────────────────────────────────────
st.set_page_config(
    page_title="Carbon CO₂ Prediction",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── THEME / CSS ────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.main { background-color: #0e1117; }

/* Card-style containers */
.metric-card {
    background: linear-gradient(135deg, #1c3a2a, #0f2d1e);
    border: 1px solid #2d6a4f;
    border-radius: 12px;
    padding: 18px 24px;
    text-align: center;
    margin-bottom: 12px;
}
.metric-card .label {
    font-size: 13px;
    color: #81b29a;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-card .value {
    font-size: 32px;
    font-weight: 800;
    color: #52b788;
    margin-top: 4px;
}
.metric-card .sub {
    font-size: 11px;
    color: #555f61;
    margin-top: 2px;
}

/* Prediction result box */
.prediction-result {
    background: linear-gradient(135deg, #1b4332, #40916c);
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    margin: 20px 0;
    box-shadow: 0 8px 32px rgba(82,183,136,0.25);
}
.prediction-result .title {
    font-size: 16px;
    color: #b7e4c7;
    font-weight: 600;
    margin-bottom: 8px;
}
.prediction-result .val {
    font-size: 52px;
    font-weight: 900;
    color: #ffffff;
}
.prediction-result .unit {
    font-size: 18px;
    color: #95d5b2;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    font-size: 18px;
    font-weight: 700;
    color: #52b788;
    border-bottom: 2px solid #2d6a4f;
    padding-bottom: 6px;
    margin: 20px 0 16px;
}

/* Progress bar override */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #52b788, #40916c);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0b1e16;
}
[data-testid="stSidebar"] label {
    color: #95d5b2 !important;
    font-size: 13px;
}

/* Alert boxes */
.stAlert { border-radius: 8px; }

/* Model badge */
.badge {
    display:inline-block;
    background:#1b4332;
    color:#52b788;
    padding:3px 10px;
    border-radius:20px;
    font-size:12px;
    font-weight:600;
    border:1px solid #40916c;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── LOAD MODEL ──────────────────────────────────────
@st.cache_resource
def load_bundle(path="carbon_capture_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

MODEL_FILE = Path("carbon_capture_model.pkl")
if not MODEL_FILE.exists():
    st.error(
        "❌ **Model file not found!**\n\n"
        "Please run `python train_model.py` first to generate `carbon_capture_model.pkl`."
    )
    st.stop()

bundle = load_bundle()

model          = bundle["model"]
all_models     = bundle["all_models"]
le_country     = bundle["le_country"]
le_energy      = bundle["le_energy"]
FEATURES       = bundle["features"]
results        = bundle["results"]
best_name      = bundle["best_model_name"]
countries      = bundle["countries"]
energy_types   = bundle["energy_types"]
year_min, year_max = bundle["year_range"]
col_stats      = bundle["col_stats"]


# ─────────────────────────── HELPER FUNCTIONS ────────────────────────────────
def interpret_co2(value: float) -> tuple[str, str]:
    """Return emoji + interpretation string for a CO2 value."""
    if value <= 0:
        return "✅", "Negligible / No emission"
    elif value < 10:
        return "🟢", "Very Low emission"
    elif value < 100:
        return "🟡", "Low – Moderate emission"
    elif value < 1000:
        return "🟠", "High emission"
    else:
        return "🔴", "Very High / Critical emission"


def build_input_row(country, energy_type, year, consumption,
                    production, gdp, population, intensity_pc, intensity_gdp) -> pd.DataFrame:
    """Encode inputs and return a 1-row DataFrame matching model's feature order."""
    try:
        country_enc = le_country.transform([country])[0]
    except ValueError:
        country_enc = 0

    try:
        energy_enc = le_energy.transform([energy_type])[0]
    except ValueError:
        energy_enc = 0

    row = {
        "Year":                       year,
        "Country_enc":                country_enc,
        "EnergyType_enc":             energy_enc,
        "Energy_consumption":         consumption,
        "Energy_production":          production,
        "GDP":                        gdp,
        "Population":                 population,
        "Energy_intensity_per_capita": intensity_pc,
        "Energy_intensity_by_GDP":    intensity_gdp,
    }
    return pd.DataFrame([row])[FEATURES]


# ─────────────────────────── SIDEBAR ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Input Parameters")
    st.markdown("---")

    # Country & Energy Type
    country     = st.selectbox("🌍 Country", countries,
                               index=countries.index("United States") if "United States" in countries else 0)
    energy_type = st.selectbox("⚡ Energy Type", energy_types)
    year        = st.slider("📅 Year", int(year_min), int(year_max), int(year_max))

    st.markdown("---")
    st.markdown("**📊 Energy & Economy**")

    avg_cons = col_stats["Energy_consumption"]["mean"]
    avg_prod = col_stats["Energy_production"]["mean"]
    avg_gdp  = col_stats["GDP"]["mean"]
    avg_pop  = col_stats["Population"]["mean"]
    avg_ipc  = col_stats["Energy_intensity_per_capita"]["mean"]
    avg_igdp = col_stats["Energy_intensity_by_GDP"]["mean"]

    consumption   = st.number_input("Energy Consumption (TWh)", min_value=0.0,
                                    value=round(avg_cons, 2), step=10.0, format="%.2f")
    production    = st.number_input("Energy Production (TWh)",  min_value=0.0,
                                    value=round(avg_prod, 2), step=10.0, format="%.2f")
    gdp           = st.number_input("GDP (Billion USD)",         min_value=0.0,
                                    value=round(avg_gdp, 2),  step=10.0, format="%.2f")
    population    = st.number_input("Population (Millions)",     min_value=0.0,
                                    value=round(avg_pop, 2),  step=1.0,  format="%.2f")
    intensity_pc  = st.number_input("Energy Intensity / Capita", min_value=0.0,
                                    value=round(avg_ipc, 4),  step=0.01, format="%.4f")
    intensity_gdp = st.number_input("Energy Intensity / GDP",    min_value=0.0,
                                    value=round(avg_igdp, 4), step=0.01, format="%.4f")

    st.markdown("---")
    st.markdown("**🤖 Model Selection**")
    chosen_model_name = st.selectbox("Choose model for prediction",
                                     list(all_models.keys()),
                                     index=list(all_models.keys()).index(best_name))

    st.markdown(f'<span class="badge">🏆 Best: {best_name}</span>', unsafe_allow_html=True)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict CO₂ Emission", use_container_width=True, type="primary")


# ─────────────────────────── HEADER ─────────────────────────────────────────
st.markdown("# 🌿 Carbon CO₂ Emission Predictor")
st.markdown(
    "Predict **CO₂ emissions** (million tonnes) based on country, energy type, "
    "and economic indicators using Machine Learning."
)

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Evaluation", "📁 Dataset Info"])


# ═══════════════════════════ TAB 1: PREDICTION ════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown('<div class="section-header">Prediction Input Summary</div>', unsafe_allow_html=True)
        input_df = pd.DataFrame({
            "Parameter": ["Country", "Energy Type", "Year", "Consumption (TWh)", "Production (TWh)",
                          "GDP (B USD)", "Population (M)", "Intensity/Capita", "Intensity/GDP"],
            "Value":     [country, energy_type, str(year), f"{consumption:,.2f}", f"{production:,.2f}",
                          f"{gdp:,.2f}", f"{population:,.2f}", f"{intensity_pc:.4f}", f"{intensity_gdp:.4f}"],
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            chosen_model = all_models[chosen_model_name]
            X_input = build_input_row(country, energy_type, year, consumption,
                                      production, gdp, population, intensity_pc, intensity_gdp)
            prediction = float(chosen_model.predict(X_input)[0])
            prediction = max(prediction, 0.0)

            emoji, interpretation = interpret_co2(prediction)

            st.markdown(f"""
            <div class="prediction-result">
                <div class="title">Estimated CO₂ Emission</div>
                <div class="val">{prediction:,.2f}</div>
                <div class="unit">Million Tonnes CO₂</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**Interpretation:** {emoji} {interpretation}")
            st.caption(f"Model used: **{chosen_model_name}**")

            # Prediction bar relative to known max
            known_max = 35584.0
            frac = min(prediction / known_max, 1.0)
            st.markdown("**Relative to global max (35,584 Mt):**")
            st.progress(frac)

            # Compare across all models
            st.markdown("**All Models Comparison:**")
            compare_rows = []
            for mname, mobj in all_models.items():
                pred_val = max(float(mobj.predict(X_input)[0]), 0.0)
                compare_rows.append({"Model": mname, "Prediction (Mt CO₂)": f"{pred_val:,.2f}"})
            st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

        else:
            st.info("👈 Fill parameters in the sidebar and click **Predict CO₂ Emission**")
            st.markdown("""
            **How it works:**
            - Select your country, energy type & year
            - Adjust energy & economic parameters
            - Click Predict to see CO₂ emission estimate
            - Switch models to compare results
            """)


# ═══════════════════════════ TAB 2: EVALUATION ════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📊 Model Evaluation Metrics</div>', unsafe_allow_html=True)
    st.markdown(
        "All models trained on **80% data**, evaluated on **20% held-out test set**. "
        "Target: `CO₂_emission` (million tonnes)."
    )

    # Metrics table
    eval_rows = []
    for mname, metrics in results.items():
        badge = " 🏆" if mname == best_name else ""
        eval_rows.append({
            "Model":           mname + badge,
            "R² Score":        metrics["R2"],
            "RMSE":            f"{metrics['RMSE']:,}",
            "MAE":             f"{metrics['MAE']:,}",
        })
    eval_df = pd.DataFrame(eval_rows)
    st.dataframe(eval_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Metric explanations in cards
    st.markdown("**Understanding the Metrics:**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="label">R² Score</div>
            <div class="value" style="font-size:20px">Coefficient of Determination</div>
            <div class="sub">Closer to 1.0 = better fit. Measures what % of variance the model explains.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="label">RMSE</div>
            <div class="value" style="font-size:20px">Root Mean Squared Error</div>
            <div class="sub">Average prediction error in same units as CO₂. Lower is better. Penalizes large errors.</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="label">MAE</div>
            <div class="value" style="font-size:20px">Mean Absolute Error</div>
            <div class="sub">Average absolute difference between predicted and actual values. Robust to outliers.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Best model metrics highlight
    bm = results[best_name]
    st.markdown(f"### 🏆 Best Model: {best_name}")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score",  f"{bm['R2']:.4f}", help="1.0 = perfect prediction")
    m2.metric("RMSE",      f"{bm['RMSE']:,} Mt",  help="Root Mean Squared Error")
    m3.metric("MAE",       f"{bm['MAE']:,} Mt",   help="Mean Absolute Error")

    # R² visual bars
    st.markdown("**R² Score Comparison:**")
    for mname, metrics in results.items():
        col_a, col_b = st.columns([2, 5])
        col_a.markdown(f"`{mname}`")
        col_b.progress(max(0.0, min(float(metrics["R2"]), 1.0)))


# ═══════════════════════════ TAB 3: DATASET ═══════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📁 Dataset Overview</div>', unsafe_allow_html=True)

    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown("""
        **Dataset:** Global Energy & CO₂ Statistics

        | Detail | Value |
        |--------|-------|
        | **Rows (after cleaning)** | ~51,614 |
        | **Features used** | 9 |
        | **Target variable** | CO₂ Emission (Mt) |
        | **Year range** | 1980 – 2019 |
        | **Countries** | 200+ |
        | **Energy types** | 6 |
        """)
    with info_col2:
        st.markdown("""
        **Energy Types in Dataset:**
        - `all_energy_types` – Combined all types
        - `coal` – Coal-based energy
        - `natural_gas` – Natural gas
        - `petroleum_n_other_liquids` – Oil & liquids
        - `nuclear` – Nuclear power
        - `renewables_n_other` – Renewables & others

        **Feature Engineering:**
        - Country & Energy_type → Label Encoded
        - Missing values → Median imputation
        """)

    st.markdown("---")
    st.markdown("**Column Statistics (Numeric Features):**")
    stats_data = []
    for col, stat in col_stats.items():
        stats_data.append({
            "Feature": col,
            "Min":  f"{stat['min']:,.3f}",
            "Max":  f"{stat['max']:,.3f}",
            "Mean": f"{stat['mean']:,.3f}",
        })
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**ML Pipeline Summary:**")
    st.code("""
1. Load CSV  →  Drop rows with missing CO₂ target
2. Fill numeric NaNs  →  Column median
3. Label Encode  →  Country, Energy_type
4. Train/Test Split  →  80% train / 20% test  (random_state=42)
5. Train 4 models:
     • Gradient Boosting  (n=200, depth=6, lr=0.1)
     • Random Forest      (n=200, depth=15)
     • Ridge Regression   (alpha=10)
     • Linear Regression
6. Evaluate  →  R², RMSE, MAE on test set
7. Save best model bundle as carbon_capture_model.pkl
    """, language="text")

# ─────────────────────────── FOOTER ──────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#4a7c59; font-size:13px;'>"
    "🌿 Carbon CO₂ Emission Predictor &nbsp;|&nbsp; "
    "Powered by Scikit-Learn &nbsp;|&nbsp; Built with Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
