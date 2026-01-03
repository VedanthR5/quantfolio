"""
VR Quantfolio Intro - Quantitative Finance Platform
====================================================
A multipage Streamlit application for stock analysis, ML-based predictions,
and portfolio optimization.

Author: Vedanth R
Repository: https://github.com/vedanthr5/vr-quantfolio-intro
"""

import streamlit as st

# Page configuration - ONLY set this in the main entry point
st.set_page_config(
    page_title="VR Quantfolio Intro",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/vedanthr5/vr-quantfolio-intro',
        'Report a bug': 'https://github.com/vedanthr5/vr-quantfolio-intro/issues',
        'About': "# VR Quantfolio Intro\nA quantitative finance learning platform."
    }
)

# Import shared styles
from utils.styles import apply_custom_css, add_footer
apply_custom_css()

# =============================================================================
# MAIN HOME PAGE CONTENT
# =============================================================================

st.title("VR Quantfolio Intro")
st.markdown("#### A hands-on platform for learning quantitative finance with Python")

st.markdown("---")

# Welcome section
st.markdown("""
This application walks you through the core building blocks of quantitative finance—from 
fetching and visualizing stock data, to training predictive models and constructing 
optimized portfolios.

**What you can do here:**
- Analyze historical stock prices with interactive charts
- Train machine learning models using automated model selection (PyCaret)
- Forecast prices with ARIMA and NeuralProphet
- Build portfolios using mean-variance optimization
""")

st.markdown("---")

# Feature cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("### AutoML Stock Analysis")
    st.markdown("""
    PyCaret's automated machine learning pipeline lets you:
    - Compare 15+ regression models in one click
    - Identify the best model for predicting stock prices
    - Inspect feature importance and prediction errors
    - Export trained models for later use
    """)
    st.page_link("pages/1_Data_Selection.py", label="Start with Data Selection")

with col2:
    st.markdown("### Time Series Forecasting")
    st.markdown("""
    Explore classical and deep learning approaches:
    - Forecast prices using ARIMA with walk-forward validation
    - Try NeuralProphet for trend and seasonality modeling
    - Test for stationarity with the ADF test
    - Evaluate accuracy with MSE and SMAPE
    """)
    st.page_link("pages/5_ARIMA_Prediction.py", label="Try Price Prediction")

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    st.markdown("### Portfolio Optimization")
    st.markdown("""
    Construct portfolios that balance risk and return:
    - Visualize the efficient frontier
    - Maximize Sharpe ratio or minimize volatility
    - Explore risk parity and advanced methods via Riskfolio-Lib
    - Analyze asset correlations
    """)
    st.page_link("pages/6_Portfolio_Optimization.py", label="Optimize Portfolio")

with col4:
    st.markdown("### Learning Resources")
    st.markdown("""
    Dive deeper into the methodology:
    - Jupyter notebooks with step-by-step explanations
    - ARIMA fundamentals tutorial
    - Links to documentation for all libraries used
    """)
    st.page_link("pages/7_Resources.py", label="View Resources")

st.markdown("---")

# Quick start guide
with st.expander("Quick Start Guide", expanded=False):
    st.markdown("""
    ### Getting Started
    
    1. **Select a Stock** — Head to Data Selection and pick a ticker symbol
    2. **Explore the Data** — Use the Visualization page to see price patterns
    3. **Train Models** — Let AutoML compare models and find the best one
    4. **Forecast Prices** — Use ARIMA or NeuralProphet for future predictions
    5. **Optimize a Portfolio** — Combine multiple assets and find the optimal allocation
    
    ### Navigation
    
    Use the sidebar on the left to move between pages. Each page is self-contained, 
    but they share data through session state so your selections carry forward.
    """)

# Footer
add_footer()
