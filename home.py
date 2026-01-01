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
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/vedanthr5/vr-quantfolio-intro',
        'Report a bug': 'https://github.com/vedanthr5/vr-quantfolio-intro/issues',
        'About': "# VR Quantfolio Intro\nA quantitative finance learning platform."
    }
)

# Import shared styles
from utils.styles import apply_custom_css
apply_custom_css()

# =============================================================================
# MAIN HOME PAGE CONTENT
# =============================================================================

st.title("📈 VR Quantfolio Intro")
st.markdown("### A Quantitative Finance Learning & Analysis Platform")

st.markdown("---")

# Welcome section
st.markdown("""
Welcome to **VR Quantfolio Intro** - an interactive platform for learning and applying 
quantitative finance techniques using Python and machine learning.

This application combines:
- 🤖 **AutoML Stock Analysis** - Automated machine learning for stock prediction
- 🔮 **Time Series Forecasting** - ARIMA and NeuralProphet models
- 💼 **Portfolio Optimization** - Mean-variance optimization with efficient frontier
- 📊 **Interactive Visualizations** - Explore stock data with Plotly charts
""")

st.markdown("---")

# Feature cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🤖 AutoML Stock Analysis")
    st.markdown("""
    Use PyCaret's automated machine learning to:
    - Compare 15+ regression models automatically
    - Find the best model for stock price prediction
    - Visualize feature importance and residuals
    - Export trained models for production use
    """)
    st.page_link("pages/1_📊_Data_Selection.py", label="Start with Data Selection →", icon="📊")

with col2:
    st.markdown("### 🔮 Price Prediction & Forecasting")
    st.markdown("""
    Leverage time series models to:
    - Forecast future stock prices with NeuralProphet
    - Understand ARIMA methodology step-by-step
    - Analyze autocorrelation and stationarity
    - Evaluate prediction accuracy with SMAPE
    """)
    st.page_link("pages/5_🔮_ARIMA_Prediction.py", label="Try Price Prediction →", icon="🔮")

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    st.markdown("### 💼 Portfolio Optimization")
    st.markdown("""
    Build optimal portfolios using:
    - Mean-variance optimization (Markowitz)
    - Efficient frontier visualization
    - Sharpe ratio maximization
    - Risk-return analysis with QuantStats
    """)
    st.page_link("pages/6_💼_Portfolio_Optimization.py", label="Optimize Portfolio →", icon="💼")

with col4:
    st.markdown("### 📚 Learning Resources")
    st.markdown("""
    Explore the methodology:
    - Interactive Jupyter notebooks
    - Step-by-step ARIMA walkthrough
    - Rendered tutorials on GitHub Pages
    - Code examples and documentation
    """)
    st.page_link("pages/7_📚_Resources.py", label="View Resources →", icon="📚")

st.markdown("---")

# Quick start guide
with st.expander("🚀 Quick Start Guide", expanded=False):
    st.markdown("""
    ### Getting Started
    
    1. **Select a Stock**: Navigate to Data Selection and choose a ticker symbol
    2. **Explore Data**: Use the Visualization page to understand price patterns
    3. **Train Models**: Let AutoML find the best prediction model
    4. **Forecast Prices**: Use ARIMA/NeuralProphet for future predictions
    5. **Optimize Portfolio**: Build a multi-asset optimized portfolio
    
    ### Navigation
    
    Use the **sidebar** on the left to navigate between pages. Each page is self-contained
    but shares data through session state for a seamless workflow.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>Built with ❤️ using Streamlit, PyCaret, and Riskfolio-Lib</p>
    <p>📖 <a href="https://vedanthr5.github.io/vr-quantfolio-intro/">View Documentation</a> | 
    🐙 <a href="https://github.com/vedanthr5/vr-quantfolio-intro">GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)
