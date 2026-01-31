# app.py
# ------------------------------------------------------------
# Professional Portfolio Analysis Dashboard (Hedge-Fund Style)
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(
    page_title="Portfolio Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Portfolio Monitoring & Risk Dashboard")
st.caption("Institutional-style portfolio analytics")

# ----------------------
# Sidebar – Portfolio Input
# ----------------------
st.sidebar.header("Portfolio Settings")

portfolio_file = st.sidebar.file_uploader(
    "Upload portfolio CSV",
    type=["csv"]
)

benchmark = st.sidebar.selectbox(
    "Benchmark",
    ["^GSPC", "ACWI", "SPY"],
    index=0
)

start_date = st.sidebar.date_input(
    "Start Date",
    datetime.today() - timedelta(days=365)
)

# ----------------------
# Load Portfolio
# ----------------------
@st.cache_data
def load_portfolio(file):
    df = pd.read_csv(file, sep=None, engine="python")

    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\t", "", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    required = {"ticker", "quantity"}
    if not required.issubset(df.columns):
        st.error("CSV must contain ticker and quantity columns")
        st.stop()

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["quantity"] = df["quantity"].astype(float)

    return df

# ----------------------
# Metadata
# ----------------------
@st.cache_data
def load_metadata(tickers):
    records = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            records.append({
                "ticker": t,
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "country": info.get("country", "Unknown"),
                "market_cap": info.get("marketCap", np.nan)
            })
        except:
            records.append({
                "ticker": t,
                "sector": "Unknown",
                "industry": "Unknown",
                "country": "Unknown",
                "market_cap": np.nan
            })
    return pd.DataFrame(records)

# ----------------------
# ETF Look-Through Exposure Templates
# ----------------------
ETF_LOOKTHROUGH = {
    # Broad US equity
    "SWPPX": {
        "sector": {
            "Technology": 0.30,
            "Health Care": 0.13,
            "Financials": 0.12,
            "Consumer Discretionary": 0.11,
            "Industrials": 0.10,
            "Communication Services": 0.09,
            "Other": 0.15
        },
        "country": {
            "United States": 1.0
        }
    },

    # US total market / completion style
    "VTHR": {
        "sector": {
            "Technology": 0.27,
            "Health Care": 0.14,
            "Financials": 0.13,
            "Industrials": 0.12,
            "Consumer Discretionary": 0.11,
            "Other": 0.23
        },
        "country": {
            "United States": 1.0
        }
    },

    # Small-cap blend
    "IJR": {
        "sector": {
            "Industrials": 0.22,
            "Financials": 0.18,
            "Consumer Discretionary": 0.15,
            "Technology": 0.14,
            "Health Care": 0.11,
            "Other": 0.20
        },
        "country": {
            "United States": 1.0
        }
    },

    # Actively managed US equity fund (treated as US total market proxy)
    "DIISX": {
        "sector": {
            "Technology": 0.28,
            "Health Care": 0.14,
            "Financials": 0.13,
            "Industrials": 0.11,
            "Consumer Discretionary": 0.10,
            "Other": 0.24
        },
        "country": {
            "United States": 1.0
        }
    },

    # Sector ETF
    "PPA": {
        "sector": {
            "Industrials": 0.80,
            "Technology": 0.15,
            "Other": 0.05
        },
        "country": {
            "United States": 0.90,
            "Other": 0.10
        }
    },
    
    # --- Gold ETF ---
    "GLD": {
        "sector": {
            "Precious Metals": 1.0
        },
        "country": {
            "United States": 1.0
        }
    }
}

ETF_CLASSIFICATION = {
    "SPY": "US_EQUITY",
    "IVV": "US_EQUITY",
    "VTI": "US_EQUITY",
    "ACWI": "GLOBAL_EQUITY",
    "VEA": "INTL_EQUITY",
    "VWO": "EM_EQUITY"
}
ASSET_CLASS_PROXIES = {
    "US_EQUITY": {
        "sector": {
            "Technology": 0.30,
            "Health Care": 0.13,
            "Financials": 0.12,
            "Consumer Discretionary": 0.10,
            "Industrials": 0.08,
            "Other": 0.27
        },
        "country": {
            "United States": 1.0
        }
    },
    "GLOBAL_EQUITY": {
        "sector": {
            "Technology": 0.22,
            "Financials": 0.18,
            "Industrials": 0.15,
            "Other": 0.45
        },
        "country": {
            "United States": 0.60,
            "International": 0.40
        }
    }
}

# ----------------------
# Prices
# ----------------------
@st.cache_data
def load_prices(tickers, start_date):
    frames = []

    for t in tickers:
        data = yf.download(
            t,
            start=start_date.strftime("%Y-%m-%d"),
            progress=False
        )

        if not data.empty and "Close" in data.columns:
            s = data["Close"].copy()
            s.name = t
            frames.append(s)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=1).dropna()
# ----------------------
# Factor Data (Fama-French via CSV)
# ----------------------
@st.cache_data
def load_ff_factors(start_date):
    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/"
        "ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    )

    ff = pd.read_csv(
        url,
        skiprows=3
    )

    # Clean column names
    ff = ff.rename(columns=lambda x: x.strip())

    # Parse dates safely (THIS IS THE KEY FIX)
    ff.iloc[:, 0] = pd.to_datetime(
        ff.iloc[:, 0],
        format="%Y%m%d",
        errors="coerce"
    )

    # Drop footer / non-date rows
    ff = ff.dropna(subset=[ff.columns[0]])

    # Set index
    ff = ff.set_index(ff.columns[0])

    # Convert percent to decimal
    ff = ff.astype(float) / 100

    # Filter by start date
    ff = ff[ff.index >= pd.to_datetime(start_date)]

    return ff



# ----------------------
# Guard
# ----------------------
if portfolio_file is None:
    st.info("Upload a portfolio CSV to begin")
    st.stop()

portfolio = load_portfolio(portfolio_file)

# ----------------------
# Market Data
# ----------------------
tickers = portfolio["ticker"].tolist()
prices = load_prices(tickers + [benchmark], start_date)
returns = prices.pct_change().dropna()

# ----------------------
# Portfolio Construction
# ----------------------
latest_prices = prices.iloc[-1]
portfolio["market_value"] = portfolio["quantity"] * latest_prices[portfolio["ticker"]].values
portfolio["weight"] = portfolio["market_value"] / portfolio["market_value"].sum()

metadata = load_metadata(portfolio["ticker"].tolist())
portfolio = portfolio.merge(metadata, on="ticker", how="left")

def cap_bucket(m):
    if pd.isna(m): return "Unknown"
    if m >= 200e9: return "Mega Cap"
    if m >= 10e9: return "Large Cap"
    if m >= 2e9: return "Mid Cap"
    return "Small Cap"

portfolio["cap_bucket"] = portfolio["market_cap"].apply(cap_bucket)

# ----------------------
# Look-Through Exposure Engine
# ----------------------

def compute_lookthrough_exposure(portfolio, exposure_type="sector"):
    """
    exposure_type: 'sector' or 'country'
    """
    exposure = {}

    for _, row in portfolio.iterrows():
        ticker = row["ticker"]
        weight = row["weight"]

        # 1️⃣ Explicit ETF look-through
        if ticker in ETF_LOOKTHROUGH:
            breakdown = ETF_LOOKTHROUGH[ticker].get(exposure_type, {})
            for k, v in breakdown.items():
                exposure[k] = exposure.get(k, 0) + weight * v

        # 2️⃣ ETF classification proxy
        elif ticker in ETF_CLASSIFICATION:
            proxy = ETF_CLASSIFICATION[ticker]
            breakdown = ASSET_CLASS_PROXIES.get(proxy, {}).get(exposure_type, {})

            if breakdown:
                for k, v in breakdown.items():
                    exposure[k] = exposure.get(k, 0) + weight * v
            else:
                exposure["Unknown"] = exposure.get("Unknown", 0) + weight

        # 3️⃣ Single-name fallback
        else:
            key = row.get(exposure_type, "Unknown")
            exposure[key] = exposure.get(key, 0) + weight

    # ✅ RETURN MUST BE HERE (outside loop)
    return pd.Series(exposure).sort_values(ascending=False)

# ----------------------
# Returns
# ----------------------
portfolio_returns = (returns[portfolio["ticker"]] * portfolio["weight"].values).sum(axis=1)
benchmark_returns = returns[benchmark]

cum_portfolio = (1 + portfolio_returns).cumprod()
cum_benchmark = (1 + benchmark_returns).cumprod()
# ----------------------
# Factor Data
# ----------------------
ff_factors = load_ff_factors(start_date)

aligned = portfolio_returns.to_frame("portfolio").join(ff_factors, how="inner")

factors = ["Mkt-RF", "SMB", "HML"]
X = aligned[factors]
y = aligned["portfolio"] - aligned["RF"]

betas = np.linalg.lstsq(X.values, y.values, rcond=None)[0]

factor_exposure = pd.Series(
    betas,
    index=["Market", "Size", "Value"]
)

# ----------------------
# Rolling Risk Metrics
# ----------------------
rolling_window = 63  # ~3 months

rolling_vol = (
    portfolio_returns
    .rolling(rolling_window)
    .std()
    * np.sqrt(252)
)

rolling_beta = (
    portfolio_returns
    .rolling(rolling_window)
    .cov(benchmark_returns)
    / benchmark_returns.rolling(rolling_window).var()
)

# ----------------------
# Drawdowns
# ----------------------
portfolio_dd = cum_portfolio / cum_portfolio.cummax() - 1
benchmark_dd = cum_benchmark / cum_benchmark.cummax() - 1

# ----------------------
# KPIs
# ----------------------
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Portfolio Value", f"${portfolio['market_value'].sum():,.0f}")
c2.metric("YTD Return", f"{(cum_portfolio.iloc[-1]-1)*100:.2f}%")
c3.metric("Volatility", f"{portfolio_returns.std()*np.sqrt(252)*100:.2f}%")
c4.metric("Max Drawdown", f"{((cum_portfolio/cum_portfolio.cummax())-1).min()*100:.2f}%")
c5.metric("Active Return", f"{(cum_portfolio.iloc[-1]-cum_benchmark.iloc[-1])*100:.2f}%")

# ----------------------
# Tabs
# ----------------------
tab_overview, tab_perf, tab_risk, tab_expo = st.tabs(
    ["Overview", "Performance", "Risk", "Exposure"]
)

# ----------------------
# Overview
# ----------------------
with tab_overview:
    fig = px.line(
        pd.DataFrame({"Portfolio": cum_portfolio, "Benchmark": cum_benchmark}),
        title="Cumulative Performance"
    )
    st.plotly_chart(fig, use_container_width=True)


# ----------------------
# Performance
# ----------------------
with tab_perf:
    st.subheader("PnL Attribution")

    contrib = returns[portfolio["ticker"]] * portfolio["weight"].values
    pnl = contrib.sum().sort_values(ascending=False)

    st.plotly_chart(
        px.bar(pnl, title="Return Contribution by Asset"),
        use_container_width=True
    )

    st.subheader("Correlation Matrix")

    corr = returns[portfolio["ticker"]].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Asset Return Correlation Matrix"
    )

    fig.update_layout(
        height=700,
        xaxis_title="Assets",
        yaxis_title="Assets"
    )

    fig.update_traces(textfont_size=12)

    st.plotly_chart(fig, use_container_width=True)

# ----------------------
# Risk
# ----------------------
with tab_risk:
    vol = returns[portfolio["ticker"]].std() * np.sqrt(252)
    beta = [
        np.cov(returns[t], benchmark_returns)[0][1] / np.var(benchmark_returns)
        for t in portfolio["ticker"]
    ]

    risk_df = pd.DataFrame({"Volatility": vol, "Beta": beta}, index=portfolio["ticker"])
    st.dataframe(risk_df.style.format("{:.2f}"))
# ----------------------
# Risk (Phase 2 Enhancements)
# ----------------------
with tab_risk:
    st.subheader("Rolling Risk")

    st.plotly_chart(
        px.line(
            pd.DataFrame({
                "Portfolio Volatility": rolling_vol,
                "Benchmark Volatility": benchmark_returns.rolling(63).std() * np.sqrt(252)
            }),
            title="Rolling Volatility (63D)"
        ),
        use_container_width=True
    )

    st.plotly_chart(
        px.line(
            rolling_beta,
            title="Rolling Beta vs Benchmark (63D)"
        ),
        use_container_width=True
    )

    st.subheader("Factor Exposure")

    st.bar_chart(factor_exposure)

    st.subheader("Drawdowns")

    st.plotly_chart(
        px.line(
            pd.DataFrame({
                "Portfolio": portfolio_dd,
                "Benchmark": benchmark_dd
            }),
            title="Drawdown Comparison"
        ),
        use_container_width=True
    )

# ----------------------
# Exposure
# ----------------------
with tab_expo:
    st.subheader("Portfolio Composition")

    st.plotly_chart(
        px.pie(
            portfolio,
            names="ticker",
            values="weight",
            title="Holdings Weight"
        ),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    # Look-through sector exposure
    sector_exposure = compute_lookthrough_exposure(portfolio, "sector")

    with col1:
        st.plotly_chart(
            px.pie(
                sector_exposure,
                values=sector_exposure.values,
                names=sector_exposure.index,
                title="Sector Exposure (Look-Through)"
            ),
            use_container_width=True
        )

    # Look-through geographic exposure
    geo_exposure = compute_lookthrough_exposure(portfolio, "country")

    with col2:
        st.plotly_chart(
            px.pie(
                geo_exposure,
                values=geo_exposure.values,
                names=geo_exposure.index,
                title="Geographic Exposure (Look-Through)"
            ),
            use_container_width=True
        )

# ----------------------
# Footer
# ----------------------
st.caption("Built with Python | Streamlit | Plotly")
