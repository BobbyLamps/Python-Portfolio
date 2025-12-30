import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Yield Curve Deformation Simulator",
    layout="wide"
)

# =====================================================
# STYLING (DESK GRADE)
# =====================================================
st.markdown("""
<style>

/* Sidebar container */
[data-testid="stSidebar"] {
    background-color: #004225; /* British Racing Green */
}

/* Sidebar section headers */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white;
}

/* Input containers */
[data-testid="stSidebar"] .stNumberInput,
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stSlider,
[data-testid="stSidebar"] .stTextInput {
    background-color: white;
    border-radius: 6px;
    padding: 6px;
}

/* Input labels */
[data-testid="stSidebar"] label {
    color: black !important;
    font-weight: 600;
}

/* Slider values */
[data-testid="stSidebar"] .stSlider span {
    color: black !important;
}

/* Selectbox text */
[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: black !important;
}

/* Metric boxes */
.metric-box {
    background-color: #0a1f44;
    padding: 14px;
    border-radius: 8px;
    color: white;
    text-align: center;
    margin-bottom: 10px;
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# BASE CURVE DATA
# =====================================================
TENORS = np.array([0.25, 0.5, 1, 2, 5, 10, 30])
BASE_YIELDS = np.array([4.5, 4.7, 4.9, 5.0, 4.8, 4.6, 4.4]) / 100

# =====================================================
# SIDEBAR CONTROLS
# =====================================================
st.sidebar.title("Yield Curve Controls")

portfolio_mm = st.sidebar.number_input(
    "Portfolio Size (USD mm)",
    min_value=1.0,
    value=50.0,
    step=5.0
)

scenario = st.sidebar.selectbox(
    "Scenario Type",
    ["Parallel", "Steepener", "Flattener", "Butterfly", "Custom"]
)

shock_bp = st.sidebar.slider(
    "Shock Magnitude (bp)",
    min_value=-150,
    max_value=150,
    value=25
)

if scenario == "Custom":
    custom_input = st.sidebar.text_input(
        "Custom Curve Shocks (bp, comma-separated)",
        "0,0,0,0,0,0,0"
    )
    custom_shocks = np.array([float(x) for x in custom_input.split(",")])
else:
    custom_shocks = None

# =====================================================
# CURVE DEFORMATION ENGINE
# =====================================================
def build_shock_curve():
    n = len(TENORS)

    level = np.full(n, shock_bp)
    slope = np.linspace(-shock_bp, shock_bp, n)
    butterfly = shock_bp * np.array([0, 1, 2, 3, 2, 1, 0]) / 3

    if scenario == "Parallel":
        shock = level
    elif scenario == "Steepener":
        shock = slope
    elif scenario == "Flattener":
        shock = -slope
    elif scenario == "Butterfly":
        shock = butterfly
    else:
        shock = custom_shocks
        if len(shock) != n:
            shock = np.interp(
                TENORS,
                np.linspace(TENORS.min(), TENORS.max(), len(shock)),
                shock
            )

    return shock / 10000  # bp → decimal

shock_curve = build_shock_curve()
scenario_yields = BASE_YIELDS + shock_curve

# =====================================================
# PRICING & RISK METRICS
# =====================================================
def zc_price(y, t):
    return np.exp(-y * t)

base_prices = zc_price(BASE_YIELDS, TENORS)
scenario_prices = zc_price(scenario_yields, TENORS)

duration = TENORS
dv01 = -duration * base_prices * 0.0001
convexity = duration**2 * base_prices

delta_y = shock_curve

pnl_dv01 = np.sum(dv01 * delta_y) * portfolio_mm * 1e6
pnl_conv = 0.5 * np.sum(convexity * delta_y**2) * portfolio_mm * 1e6
total_pnl = np.sum(scenario_prices - base_prices) * portfolio_mm * 1e6
residual_pnl = total_pnl - (pnl_dv01 + pnl_conv)

# =====================================================
# KEY RATE DV01
# =====================================================
KEY_RATES = [2, 5, 10, 30]
krd = {}

for kr in KEY_RATES:
    bump = np.zeros(len(TENORS))
    idx = np.argmin(np.abs(TENORS - kr))
    bump[idx] = 0.0001
    bumped_prices = zc_price(BASE_YIELDS + bump, TENORS)
    krd[kr] = np.sum(bumped_prices - base_prices) * portfolio_mm * 1e6

krd_df = pd.DataFrame({
    "Key Rate": [f"{k}Y" for k in KEY_RATES],
    "DV01 ($)": list(krd.values())
})

# =====================================================
# MAIN DASHBOARD
# =====================================================
st.title("Yield Curve Deformation Simulator")

col_curve, col_metrics = st.columns([2.3, 1])

# ---- Curve Plot ----
with col_curve:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=TENORS,
        y=BASE_YIELDS * 100,
        name="Base Curve",
        line=dict(width=3, color="#1f77b4")  # BLUE
    ))

    fig.add_trace(go.Scatter(
        x=TENORS,
        y=scenario_yields * 100,
        name="Scenario Curve",
        line=dict(width=3)
    ))

    fig.update_layout(
        xaxis_title="Tenor (Years)",
        yaxis_title="Yield (%)",
        template="plotly_dark",
        legend=dict(x=0.02, y=0.98)
    )

    st.plotly_chart(fig, use_container_width=True)

# ---- Metrics ----
with col_metrics:
    st.markdown(f"<div class='metric-box'>DV01 PnL<br>${pnl_dv01:,.0f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-box'>Convexity PnL<br>${pnl_conv:,.0f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-box'>Residual PnL<br>${residual_pnl:,.0f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-box'><b>Total PnL</b><br>${total_pnl:,.0f}</div>", unsafe_allow_html=True)

# ---- Key Rate DV01 ----
st.subheader("Key Rate DV01")
st.dataframe(krd_df, use_container_width=True)

st.caption(
    "Institutional curve deformation, DV01, convexity & KRD attribution tool | Rates Sales & Macro PM"
)
